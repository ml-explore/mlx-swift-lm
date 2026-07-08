import Foundation
import MLX
import MLXNN

// MARK: - Metal Kernel Source

/// Pairwise Givens rotation kernel for Metal (Apple Silicon).
/// Template parameters are substituted at compile time.
private func metalSource(
    rowsPerTile: Int, maxGroupSize: Int = 128, maxKrot: Int = 16
) -> String {
    """
    constexpr int ROWS_PER_TILE = \(rowsPerTile);
    constexpr int MAX_KROT      = \(maxKrot);

    const int batch_size  = params[0];
    const int hidden_size = params[1];
    const int krot        = params[2];
    const int group_size  = params[3];

    const int half_gs     = group_size / 2;
    const int half_hidden = hidden_size / 2;

    const int tile_idx  = threadgroup_position_in_grid.x;
    const int group_idx = threadgroup_position_in_grid.y;
    const int tid       = thread_index_in_threadgroup;

    if (tid >= half_gs) return;

    // Load rotation coefficients into registers
    float cos_vals[MAX_KROT], sin_vals[MAX_KROT];
    int   pair_vals[MAX_KROT];

    for (int k = 0; k < krot; k++) {
        int idx = k * half_hidden + group_idx * half_gs + tid;
        cos_vals[k]  = float(cos_theta[idx]);
        sin_vals[k]  = float(sin_theta[idx]);
        pair_vals[k] = int(packed_pairs[idx]);
    }

    // Load activation tile into shared memory (fuse channel scales)
    threadgroup float tile[\(maxGroupSize) * ROWS_PER_TILE];

    const int ch_lo = group_idx * group_size + tid;
    const int ch_hi = ch_lo + half_gs;
    float scale_lo = float(channel_scales[ch_lo]);
    float scale_hi = float(channel_scales[ch_hi]);

    for (int r = 0; r < ROWS_PER_TILE; r++) {
        int row = tile_idx * ROWS_PER_TILE + r;
        if (row < batch_size) {
            tile[tid * ROWS_PER_TILE + r]              = float(x[row * hidden_size + ch_lo]) * scale_lo;
            tile[(tid + half_gs) * ROWS_PER_TILE + r]  = float(x[row * hidden_size + ch_hi]) * scale_hi;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Apply pairwise Givens rotations in-place
    for (int k = 0; k < krot; k++) {
        int i_local = pair_vals[k] & 0xFFFF;
        int j_local = pair_vals[k] >> 16;
        float c = cos_vals[k], s = sin_vals[k];

        for (int m = 0; m < ROWS_PER_TILE; m++) {
            float a = tile[i_local * ROWS_PER_TILE + m];
            float b = tile[j_local * ROWS_PER_TILE + m];
            tile[i_local * ROWS_PER_TILE + m] = a * c + b * s;
            tile[j_local * ROWS_PER_TILE + m] = b * c - a * s;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write results back
    for (int r = 0; r < ROWS_PER_TILE; r++) {
        int row = tile_idx * ROWS_PER_TILE + r;
        if (row < batch_size) {
            out[row * hidden_size + ch_lo] = tile[tid * ROWS_PER_TILE + r];
            out[row * hidden_size + ch_hi] = tile[(tid + half_gs) * ROWS_PER_TILE + r];
        }
    }
    """
}

// MARK: - Kernel Cache

/// Cached compiled Metal kernels keyed by tile size, guarded by `kernelCacheLock`.
/// Callers are multi-threaded (each `ModelContainer.perform` closure can run on its
/// own task), so the dictionary read-modify-write is serialised. Contention is
/// practically nil — only two tile sizes (1 and 4) are ever requested, so the lock
/// is contended exactly twice per process before steady-state hits.
nonisolated(unsafe) private var kernelCache: [Int: MLXFast.MLXFastKernel] = [:]
private let kernelCacheLock = NSLock()

/// Internal (not file-private): shared by every module that dispatches the
/// pairwise rotation kernel — `PairwiseRotation` and `RotateQuantizedLinear`.
nonisolated func getRotationKernel(tile: Int) -> MLXFast.MLXFastKernel {
    kernelCacheLock.withLock {
        if let cached = kernelCache[tile] {
            return cached
        }
        let kernel = MLXFast.metalKernel(
            name: "paro_rotate_r\(tile)",
            inputNames: [
                "x", "packed_pairs", "cos_theta", "sin_theta", "channel_scales", "params",
            ],
            outputNames: ["out"],
            source: metalSource(rowsPerTile: tile)
        )
        kernelCache[tile] = kernel
        return kernel
    }
}

// MARK: - Pair Packing

/// Pack int16 pair indices into int32 for the Metal kernel.
///
/// Each pair `(i, j)` is packed as `i | (j << 16)` within each group.
///
/// Internal (not file-private): shared by `PairwiseRotation` and
/// `RotateQuantizedLinear`, and unit-tested directly.
nonisolated func packPairs(_ pairs: MLXArray, groupSize: Int) -> MLXArray {
    let krot = pairs.dim(0)
    let numGroups = pairs.dim(1) / groupSize

    // Reshape to [krot, numGroups, groupSize]
    let p = pairs.reshaped(krot, numGroups, groupSize).asType(.int32)

    // Even indices (lo) and odd indices (hi) within each group
    let lo = p[0..., 0..., .stride(by: 2)]
    let hi = p[0..., 0..., .stride(from: 1, by: 2)]
    return (lo | (hi << 16)).reshaped(krot, -1)
}

// MARK: - PairwiseRotation

/// Standalone pairwise Givens rotation over the last axis of an activation
/// tensor, fused with per-channel scaling in a single Metal kernel.
///
/// This is the rotation half of `RotateQuantizedLinear`, extracted as a
/// composable `Module` for layers whose rotation is *shared* across several
/// quantized projections instead of owned by one — the MoE `RotateSwitchGLU`
/// composes two of these (`gate_up_rot`, `down_rot`) around stock
/// `QuantizedSwitchLinear` experts.
///
/// Checkpoint contract: `theta` / `pairs` / `channel_scales` load via Module
/// reflection under this module's key prefix (e.g.
/// `switch_mlp.gate_up_rot.theta`). After loading, the owner must call
/// `prepareDerivedRotationState()` once, before any forward pass.
public class PairwiseRotation: Module {

    // Rotation parameters — discovered by Module reflection for update(parameters:).
    // `channelScales` uses @ParameterInfo so it can keep the snake_case checkpoint
    // key while having a Swift-idiomatic property name.
    let theta: MLXArray
    let pairs: MLXArray
    @ParameterInfo(key: "channel_scales") var channelScales: MLXArray

    let groupSize: Int

    // Rotation-derived state. Populated once by `prepareDerivedRotationState()`
    // after the checkpoint parameters are loaded (see ParoQuantLoader), and
    // never mutated afterwards. Underscore-prefixed private properties are
    // ignored by Module reflection — see Documentation.docc/porting.md
    // "Computed vs Loaded Parameters" — so they don't participate in weight
    // loading, which keeps the loader's strict `verify: [.allModelKeysSet]`
    // contract intact.
    private var _cosTheta: MLXArray
    private var _sinTheta: MLXArray
    private var _packedPairs: MLXArray
    private var _scalesFlat: MLXArray

    public init(dims: Int, groupSize: Int, krot: Int) {
        self.theta = MLXArray.zeros([krot, dims / 2])
        self.pairs = MLXArray.zeros([krot, dims], type: Int16.self)
        // Assign through `.wrappedValue` so the `@ParameterInfo(key:)` metadata
        // survives init — see the matching note in RotateQuantizedLinear.
        self._channelScales.wrappedValue = MLXArray.ones([1, dims])
        self.groupSize = groupSize

        // Placeholder values — `prepareDerivedRotationState()` overwrites
        // these with real derived tensors after checkpoint load. Shapes are
        // correct so a forward pass before finalize would be degenerate
        // (identity-ish rotation) rather than crash.
        self._cosTheta = MLXArray.ones([krot, dims / 2])
        self._sinTheta = MLXArray.zeros([krot, dims / 2])
        self._packedPairs = MLXArray.zeros([krot, dims / 2], type: Int32.self)
        self._scalesFlat = MLXArray.ones([dims])

        super.init()
    }

    /// Compute rotation-derived tensors from the loaded checkpoint parameters.
    ///
    /// Must be called once, after `update(parameters:)` populates
    /// `theta` / `pairs` / `channelScales`, and before any forward pass.
    /// Must not be called concurrently with forward passes — the loader
    /// owns this call, nothing else should.
    ///
    /// Deriving lazily on first forward and caching in a mutable field is
    /// unsafe under multi-threaded inference (issue #157 — a shared model
    /// container is driven by multiple tasks simultaneously), so derivation
    /// is done explicitly at load time.
    ///
    /// The four derived arrays are `eval(...)`ed here because underscore-
    /// prefixed private fields are invisible to Module reflection — the
    /// loader's later `eval(model)` walks `@ParameterInfo` tensors only, so
    /// these would otherwise stay unmaterialised promises until the first
    /// forward pass, and materialisation would then become part of that
    /// pass's graph (exactly the eval-time state we're eliminating).
    public func prepareDerivedRotationState() {
        _cosTheta = MLX.cos(theta)
        _sinTheta = MLX.sin(theta)
        _packedPairs = packPairs(pairs, groupSize: groupSize)
        _scalesFlat = channelScales.reshaped(-1)
        eval(_cosTheta, _sinTheta, _packedPairs, _scalesFlat)
    }

    /// Apply channel scaling + pairwise Givens rotations to the last axis.
    ///
    /// Accepts any leading shape (the MoE path passes gathered 4-D
    /// activations); the input is flattened to 2-D for the kernel and the
    /// original shape is restored on return. No mutable state is read or
    /// written by this method.
    public func rotate(_ x: MLXArray) -> MLXArray {
        let dim = _scalesFlat.dim(0)
        let shape = x.shape
        let flat = x.reshaped(-1, dim)

        let batch = flat.dim(0)
        // Gathered MoE activations can be legitimately empty; a zero-sized
        // grid dispatch is undefined, so pass the input through.
        if batch == 0 { return x }

        let halfGroup = groupSize / 2
        let numGroups = dim / groupSize
        let krot = theta.dim(0)

        let tile = batch <= 1 ? 1 : 4
        let gridX = ((batch + tile - 1) / tile) * halfGroup
        let params = MLXArray([Int32(batch), Int32(dim), Int32(krot), Int32(groupSize)])

        let rotated = getRotationKernel(tile: tile)(
            [flat, _packedPairs, _cosTheta, _sinTheta, _scalesFlat, params],
            grid: (gridX, numGroups, 1),
            threadGroup: (halfGroup, 1, 1),
            outputShapes: [flat.shape],
            outputDTypes: [flat.dtype]
        )[0]
        return rotated.reshaped(shape)
    }
}
