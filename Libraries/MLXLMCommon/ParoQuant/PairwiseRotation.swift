import Foundation
import MLX
import MLXNN

// MARK: - Metal Kernel Source

/// Pairwise Givens rotation kernel for Metal (Apple Silicon).
/// Template parameters are substituted at compile time.
///
/// One CTA is a single simdgroup (32 lanes) per (row-tile, channel-group):
///
/// - Each lane caches the cos/sin/pair coefficients of its two pair slots
///   (lane, lane+32) for every round in registers. `krot` is a compile-time
///   constant, so these are constant-index register arrays — the old kernel
///   indexed its coefficient arrays by a runtime loop bound, which pushed
///   them into local (DRAM-backed) memory.
/// - Per-round sync is `simdgroup_barrier(mem_threadgroup)` instead of
///   `threadgroup_barrier` — with a one-simdgroup CTA there is no
///   cross-simdgroup rendezvous to pay for. The tile layout is row-major
///   (`tile[row * 128 + ch]`), so pair accesses are bank-conflict-free
///   for any ROWS_PER_TILE (the old channel-major `tile[ch * R + row]`
///   layout collapsed onto 8 banks for R = 4).
/// - x/out IO is vectorized (`\(t4)` per lane covers the 128-channel
///   group exactly); the f32 threadgroup tile is written/read as float4.
///   `channel_scales` is loaded scalar + converted so its dtype may
///   legitimately differ from the activation dtype.
/// - The write-back casts explicitly (`\(t)(...)`): the old kernel's
///   implicit float->element conversion did not compile for bfloat16.
///
/// Correctness notes:
/// - All lanes execute every barrier (no early returns; `row < batch_size`
///   guards wrap memory accesses only and are CTA-uniform).
/// - The math is bit-identical to the old kernel per element: f32 loads of
///   `float(x) * scale`, the same krot Givens rounds in order with the same
///   pairs/cos/sin (`a * c + b * s`, `b * c - a * s` in f32), then one
///   rounding to the element type on write-back.
/// - Requires groupSize == 128 (64 pair slots per group = 2 per lane) and
///   krot >= 1; both are enforced by the caller.
private func simdgroupMetalSource(
    rowsPerTile: Int, krot: Int, elementType t: String, elementType4 t4: String
) -> String {
    """
    constexpr int ROWS_PER_TILE = \(rowsPerTile);
    constexpr int KROT          = \(krot);

    const int batch_size  = params[0];
    const int hidden_size = params[1];
    const int group_size  = params[3];

    const int half_gs     = group_size / 2;
    const int half_hidden = hidden_size / 2;

    const int tile_idx  = threadgroup_position_in_grid.x;
    const int group_idx = threadgroup_position_in_grid.y;
    const int lane      = thread_index_in_threadgroup;
    const int gbase     = group_idx * group_size;

    // Rotation coefficients for this lane's two pair slots of every round
    float cos_vals[KROT][2], sin_vals[KROT][2];
    int   pair_vals[KROT][2];

    for (int k = 0; k < KROT; k++) {
        for (int u = 0; u < 2; u++) {
            int idx = k * half_hidden + group_idx * half_gs + lane + u * 32;
            cos_vals[k][u]  = float(cos_theta[idx]);
            sin_vals[k][u]  = float(sin_theta[idx]);
            pair_vals[k][u] = int(packed_pairs[idx]);
        }
    }

    threadgroup float tile[ROWS_PER_TILE * 128];

    // Load activation tile into shared memory (fuse channel scales).
    // Lane owns channels lane*4 .. lane*4+3 of the group.
    float sc0 = float(channel_scales[gbase + lane * 4 + 0]);
    float sc1 = float(channel_scales[gbase + lane * 4 + 1]);
    float sc2 = float(channel_scales[gbase + lane * 4 + 2]);
    float sc3 = float(channel_scales[gbase + lane * 4 + 3]);
    for (int r = 0; r < ROWS_PER_TILE; r++) {
        int row = tile_idx * ROWS_PER_TILE + r;
        if (row < batch_size) {
            \(t4) xh = ((const device \(t4)*)(x + row * hidden_size + gbase))[lane];
            float4 tv;
            tv[0] = float(xh[0]) * sc0;
            tv[1] = float(xh[1]) * sc1;
            tv[2] = float(xh[2]) * sc2;
            tv[3] = float(xh[3]) * sc3;
            *(threadgroup float4*)(tile + r * 128 + lane * 4) = tv;
        }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    // Apply pairwise Givens rotations in-place
    for (int k = 0; k < KROT; k++) {
        for (int u = 0; u < 2; u++) {
            int i_local = pair_vals[k][u] & 0xFFFF;
            int j_local = pair_vals[k][u] >> 16;
            float c = cos_vals[k][u], s = sin_vals[k][u];
            for (int m = 0; m < ROWS_PER_TILE; m++) {
                float a = tile[m * 128 + i_local];
                float b = tile[m * 128 + j_local];
                tile[m * 128 + i_local] = a * c + b * s;
                tile[m * 128 + j_local] = b * c - a * s;
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write results back
    for (int r = 0; r < ROWS_PER_TILE; r++) {
        int row = tile_idx * ROWS_PER_TILE + r;
        if (row < batch_size) {
            float4 tv = *(threadgroup float4*)(tile + r * 128 + lane * 4);
            \(t4) ov;
            ov[0] = \(t)(tv[0]);
            ov[1] = \(t)(tv[1]);
            ov[2] = \(t)(tv[2]);
            ov[3] = \(t)(tv[3]);
            *(device \(t4)*)(out + row * hidden_size + gbase + lane * 4) = ov;
        }
    }
    """
}

// MARK: - Kernel Cache

/// Metal scalar + 4-wide vector type names used for the rotation kernel's
/// vectorized IO, or nil for dtypes the kernel is not instantiated for.
private func rotationKernelTypeNames(_ dtype: DType) -> (String, String)? {
    switch dtype {
    case .float16: return ("half", "half4")
    case .bfloat16: return ("bfloat16_t", "bfloat4")
    case .float32: return ("float", "float4")
    default: return nil
    }
}

/// Cached compiled Metal kernels keyed by tile size, krot and IO dtype,
/// guarded by `kernelCacheLock`. Callers are multi-threaded (each
/// `ModelContainer.perform` closure can run on its own task), so the
/// dictionary read-modify-write is serialised. Contention is practically
/// nil — only two tile sizes (1 and 4), one krot and one dtype are ever
/// requested per model, so the lock is contended a handful of times per
/// process before steady-state hits.
nonisolated(unsafe) private var kernelCache: [String: MLXFast.MLXFastKernel] = [:]
private let kernelCacheLock = NSLock()

/// Internal (not file-private): shared by every module that dispatches the
/// pairwise rotation kernel — `PairwiseRotation` and `RotateQuantizedLinear`.
nonisolated func getRotationKernel(tile: Int, krot: Int, dtype: DType) -> MLXFast.MLXFastKernel {
    kernelCacheLock.withLock {
        guard let (t, t4) = rotationKernelTypeNames(dtype) else {
            preconditionFailure(
                "PairwiseRotation: unsupported activation dtype \(dtype) (expected float16/bfloat16/float32)"
            )
        }
        let key = "\(tile)_\(krot)_\(t)"
        if let cached = kernelCache[key] {
            return cached
        }
        let kernel = MLXFast.metalKernel(
            name: "paro_rotate_r\(tile)_k\(krot)_\(t)",
            inputNames: [
                "x", "packed_pairs", "cos_theta", "sin_theta", "channel_scales", "params",
            ],
            outputNames: ["out"],
            source: simdgroupMetalSource(
                rowsPerTile: tile, krot: krot, elementType: t, elementType4: t4)
        )
        kernelCache[key] = kernel
        return kernel
    }
}

// MARK: - Generic Fallback Kernel (groupSize != 128)

/// Pre-simdgroup rotation kernel, kept as the fallback for groupSize != 128:
/// one thread per pair slot (`tid < group_size / 2`), a runtime `krot` loop
/// bound, and a channel-major threadgroup tile. Handles any groupSize <= 128
/// for float16/float32 IO — its implicit float->element write-back does not
/// compile for bfloat16, the same restriction this path had before the
/// simdgroup kernel existed (the simdgroup kernel covers bfloat16 at 128).
private func genericMetalSource(
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

/// Cached compiled generic kernels keyed by tile size, guarded by the same
/// `kernelCacheLock` as the simdgroup cache (see `getRotationKernel`).
nonisolated(unsafe) private var genericKernelCache: [Int: MLXFast.MLXFastKernel] = [:]

nonisolated private func getGenericRotationKernel(tile: Int) -> MLXFast.MLXFastKernel {
    kernelCacheLock.withLock {
        if let cached = genericKernelCache[tile] {
            return cached
        }
        let kernel = MLXFast.metalKernel(
            name: "paro_rotate_generic_r\(tile)",
            inputNames: [
                "x", "packed_pairs", "cos_theta", "sin_theta", "channel_scales", "params",
            ],
            outputNames: ["out"],
            source: genericMetalSource(rowsPerTile: tile)
        )
        genericKernelCache[tile] = kernel
        return kernel
    }
}

// MARK: - Dispatch

/// Dispatch the pairwise rotation on a 2-D `[batch, dim]` activation.
///
/// groupSize == 128 takes the simdgroup-resident kernel (2 pair slots per
/// lane, no CTA rendezvous); any other groupSize <= 128 falls back to the
/// generic kernel. Shared by `PairwiseRotation` and `RotateQuantizedLinear`.
nonisolated func dispatchPairwiseRotation(
    _ flat: MLXArray,
    packedPairs: MLXArray, cosTheta: MLXArray, sinTheta: MLXArray, scalesFlat: MLXArray,
    groupSize: Int, krot: Int
) -> MLXArray {
    let batch = flat.dim(0)
    let dim = scalesFlat.dim(0)
    let numGroups = dim / groupSize
    let tile = batch <= 1 ? 1 : 4
    let params = MLXArray([Int32(batch), Int32(dim), Int32(krot), Int32(groupSize)])

    if groupSize == 128 {
        let gridX = ((batch + tile - 1) / tile) * 32
        return getRotationKernel(tile: tile, krot: krot, dtype: flat.dtype)(
            [flat, packedPairs, cosTheta, sinTheta, scalesFlat, params],
            grid: (gridX, numGroups, 1),
            threadGroup: (32, 1, 1),
            outputShapes: [flat.shape],
            outputDTypes: [flat.dtype]
        )[0]
    }

    let halfGroup = groupSize / 2
    let gridX = ((batch + tile - 1) / tile) * halfGroup
    return getGenericRotationKernel(tile: tile)(
        [flat, packedPairs, cosTheta, sinTheta, scalesFlat, params],
        grid: (gridX, numGroups, 1),
        threadGroup: (halfGroup, 1, 1),
        outputShapes: [flat.shape],
        outputDTypes: [flat.dtype]
    )[0]
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
    ///
    /// Kernel selection lives in `dispatchPairwiseRotation`: groupSize == 128
    /// takes the simdgroup-resident kernel, any other group size the generic
    /// fallback.
    public func rotate(_ x: MLXArray) -> MLXArray {
        let dim = _scalesFlat.dim(0)
        let shape = x.shape
        let flat = x.reshaped(-1, dim)

        // Gathered MoE activations can be legitimately empty; a zero-sized
        // grid dispatch is undefined, so pass the input through.
        if flat.dim(0) == 0 { return x }

        return dispatchPairwiseRotation(
            flat,
            packedPairs: _packedPairs, cosTheta: _cosTheta, sinTheta: _sinTheta,
            scalesFlat: _scalesFlat, groupSize: groupSize, krot: theta.dim(0)
        ).reshaped(shape)
    }
}
