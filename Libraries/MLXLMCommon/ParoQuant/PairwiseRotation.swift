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
/// - The write-back casts explicitly (`\(t)(...)`) so the same source
///   instantiates for float16, bfloat16 and float32.
///
/// Correctness notes:
/// - All lanes execute every barrier (no early returns; `row < batch_size`
///   guards wrap memory accesses only and are CTA-uniform).
/// - The math is bit-identical to the old kernel per element: f32 loads of
///   `float(x) * scale`, the same krot Givens rounds in order with the same
///   pairs/cos/sin (`a * c + b * s`, `b * c - a * s` in f32), then one
///   rounding to the element type on write-back.
/// - Requires groupSize == 128 (64 pair slots per group = 2 per lane) and
///   krot >= 1; both are enforced by `dispatchPairwiseRotation`.
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

/// Metal scalar + 4-wide vector type names used for the rotation kernels'
/// IO, or nil for dtypes the kernels are not instantiated for.
private func rotationKernelTypeNames(_ dtype: DType) -> (String, String)? {
    switch dtype {
    case .float16: return ("half", "half4")
    case .bfloat16: return ("bfloat16_t", "bfloat4")
    case .float32: return ("float", "float4")
    default: return nil
    }
}

/// Cache key for the compiled rotation kernels. A value type rather than a
/// formatted string: the lookup runs on every rotation dispatch (hundreds of
/// times per token on a MoE model), so the key must hash without allocating.
private struct RotationKernelKey: Hashable {
    let tile: Int
    let groupSize: Int
    let krot: Int
    let dtype: DType
}

/// Cached compiled Metal kernels keyed by tile size, group size, krot and IO
/// dtype, guarded by `kernelCacheLock`. Callers are multi-threaded (each
/// `ModelContainer.perform` closure can run on its own task), so the
/// dictionary read-modify-write is serialised. Contention is practically
/// nil — only two tile sizes (1 and 4), one group size, one krot and one
/// dtype are ever requested per model, so the lock is contended a handful of
/// times per process before steady-state hits.
nonisolated(unsafe) private var kernelCache: [RotationKernelKey: MLXFast.MLXFastKernel] = [:]
private let kernelCacheLock = NSLock()

/// Sole entry point is `dispatchPairwiseRotation`, which owns the geometry
/// checks. groupSize == 128 compiles the simdgroup-resident kernel, any
/// other group size the generic one; both are specialised on the full key.
nonisolated private func getRotationKernel(
    tile: Int, groupSize: Int, krot: Int, dtype: DType
) -> MLXFast.MLXFastKernel {
    kernelCacheLock.withLock {
        let key = RotationKernelKey(tile: tile, groupSize: groupSize, krot: krot, dtype: dtype)
        if let cached = kernelCache[key] {
            return cached
        }
        guard let (t, t4) = rotationKernelTypeNames(dtype) else {
            preconditionFailure(
                "PairwiseRotation: unsupported activation dtype \(dtype) (expected float16/bfloat16/float32)"
            )
        }
        let source: String
        let name: String
        if groupSize == 128 {
            name = "paro_rotate_r\(tile)_k\(krot)_\(t)"
            source = simdgroupMetalSource(
                rowsPerTile: tile, krot: krot, elementType: t, elementType4: t4)
        } else {
            name = "paro_rotate_generic_r\(tile)_g\(groupSize)_k\(krot)_\(t)"
            source = genericMetalSource(
                rowsPerTile: tile, groupSize: groupSize, krot: krot, elementType: t)
        }
        let kernel = MLXFast.metalKernel(
            name: name,
            inputNames: [
                "x", "packed_pairs", "cos_theta", "sin_theta", "channel_scales", "params",
            ],
            outputNames: ["out"],
            source: source
        )
        kernelCache[key] = kernel
        return kernel
    }
}

// MARK: - Generic Fallback Kernel (groupSize != 128)

/// Pre-simdgroup rotation kernel, kept as the fallback for groupSize != 128:
/// one thread per pair slot (`tid < GROUP_SIZE / 2`), a full CTA barrier per
/// round, and a channel-major threadgroup tile. Group size, krot and the
/// element type are compile-time constants, so the tile and the per-lane
/// coefficient arrays are sized exactly for the model instead of against a
/// fixed ceiling, and the write-back casts explicitly like the simdgroup
/// kernel. Any even groupSize whose half fits one threadgroup (<= 2048)
/// works; the loader and `dispatchPairwiseRotation` enforce the geometry.
private func genericMetalSource(
    rowsPerTile: Int, groupSize: Int, krot: Int, elementType t: String
) -> String {
    """
    constexpr int ROWS_PER_TILE = \(rowsPerTile);
    constexpr int GROUP_SIZE    = \(groupSize);
    constexpr int KROT          = \(krot);

    const int batch_size  = params[0];
    const int hidden_size = params[1];

    const int half_gs     = GROUP_SIZE / 2;
    const int half_hidden = hidden_size / 2;

    const int tile_idx  = threadgroup_position_in_grid.x;
    const int group_idx = threadgroup_position_in_grid.y;
    const int tid       = thread_index_in_threadgroup;

    // Load rotation coefficients into registers
    float cos_vals[KROT], sin_vals[KROT];
    int   pair_vals[KROT];

    for (int k = 0; k < KROT; k++) {
        int idx = k * half_hidden + group_idx * half_gs + tid;
        cos_vals[k]  = float(cos_theta[idx]);
        sin_vals[k]  = float(sin_theta[idx]);
        pair_vals[k] = int(packed_pairs[idx]);
    }

    // Load activation tile into shared memory (fuse channel scales)
    threadgroup float tile[GROUP_SIZE * ROWS_PER_TILE];

    const int ch_lo = group_idx * GROUP_SIZE + tid;
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
    for (int k = 0; k < KROT; k++) {
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
            out[row * hidden_size + ch_lo] = \(t)(tile[tid * ROWS_PER_TILE + r]);
            out[row * hidden_size + ch_hi] = \(t)(tile[(tid + half_gs) * ROWS_PER_TILE + r]);
        }
    }
    """
}

// MARK: - Geometry

/// The largest half-group the generic kernel can run: one thread per pair
/// slot in a single threadgroup (Metal's 1024-thread CTA limit).
private let maxGenericGroupSize = 2048

/// Validate a rotation's (dims, groupSize, krot) triple. The kernels assume
/// every one of these: `dims` splits into whole groups (a trailing partial
/// group would silently go un-rotated — `numGroups` floors), a group holds
/// whole pairs, its half fits one threadgroup, and there is at least one
/// round. Returns a reason on failure so the loader can surface it as a
/// typed error before any module is built; the module initializers assert
/// the same contract for direct callers.
nonisolated func rotationGeometryProblem(dims: Int, groupSize: Int, krot: Int) -> String? {
    if krot < 1 {
        return "krot must be >= 1 (got \(krot))"
    }
    if groupSize < 2 || !groupSize.isMultiple(of: 2) {
        return "groupSize must be even and >= 2 (got \(groupSize))"
    }
    if groupSize > maxGenericGroupSize {
        return "groupSize must be <= \(maxGenericGroupSize) (got \(groupSize))"
    }
    if dims < groupSize || !dims.isMultiple(of: groupSize) {
        return
            "dims must be a positive multiple of groupSize (got dims \(dims), groupSize \(groupSize))"
    }
    return nil
}

/// `precondition` form of `rotationGeometryProblem` for the module initializers.
nonisolated func assertRotationGeometry(dims: Int, groupSize: Int, krot: Int) {
    if let problem = rotationGeometryProblem(dims: dims, groupSize: groupSize, krot: krot) {
        preconditionFailure("PairwiseRotation: \(problem)")
    }
}

// MARK: - Dispatch

/// Dispatch the pairwise rotation on a 2-D `[batch, dim]` activation.
///
/// groupSize == 128 takes the simdgroup-resident kernel (2 pair slots per
/// lane, no CTA rendezvous); any other groupSize the generic kernel, which is
/// specialised on the group size. Both kernels are instantiated for
/// float16, bfloat16 and float32 activations. Shared by `PairwiseRotation`
/// and `RotateQuantizedLinear`.
///
/// Zero-row inputs pass straight through: gathered MoE activations can be
/// legitimately empty, and a zero-sized grid dispatch is undefined. The
/// guard lives here, with the grid math, so no caller needs one.
nonisolated func dispatchPairwiseRotation(
    _ flat: MLXArray, state: RotationDerivedState, groupSize: Int, krot: Int
) -> MLXArray {
    let batch = flat.dim(0)
    if batch == 0 { return flat }

    let dim = state.scalesFlat.dim(0)
    assertRotationGeometry(dims: dim, groupSize: groupSize, krot: krot)
    let numGroups = dim / groupSize
    let tile = batch <= 1 ? 1 : 4
    let params = MLXArray([Int32(batch), Int32(dim), Int32(krot), Int32(groupSize)])

    let kernel = getRotationKernel(tile: tile, groupSize: groupSize, krot: krot, dtype: flat.dtype)
    let threads = groupSize == 128 ? 32 : groupSize / 2

    let gridX = ((batch + tile - 1) / tile) * threads
    return kernel(
        [flat, state.packedPairs, state.cosTheta, state.sinTheta, state.scalesFlat, params],
        grid: (gridX, numGroups, 1),
        threadGroup: (threads, 1, 1),
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

// MARK: - Derived Rotation State

/// The four kernel-ready tensors derived from a rotation's checkpoint
/// parameters (`theta` / `pairs` / `channel_scales`), shared by
/// `RotateQuantizedLinear` (dense) and `PairwiseRotation` (MoE shared) so
/// the derivation cannot diverge between the two paths.
///
/// Owners store this in an underscore-prefixed property: Module reflection
/// drops `_`-prefixed keys (`Module.parameterIsValid`), so the derived
/// tensors don't participate in weight loading — which keeps the loader's
/// strict `verify: [.allModelKeysSet]` contract intact — and are skipped by
/// `eval(model)`, which walks reflected parameters only.
struct RotationDerivedState {
    var cosTheta: MLXArray
    var sinTheta: MLXArray
    var packedPairs: MLXArray
    var scalesFlat: MLXArray

    /// Placeholder state — `prepare(...)` overwrites it after checkpoint
    /// load. Shapes are correct so a forward pass before finalize would be
    /// degenerate (identity-ish rotation) rather than crash.
    init(dims: Int, krot: Int) {
        cosTheta = MLXArray.ones([krot, dims / 2])
        sinTheta = MLXArray.zeros([krot, dims / 2])
        packedPairs = MLXArray.zeros([krot, dims / 2], type: Int32.self)
        scalesFlat = MLXArray.ones([dims])
    }

    /// Recompute from the loaded checkpoint parameters. The results are
    /// lazy graph nodes — materializing them is the caller's job (see
    /// `RotationStatePreparing` for why).
    mutating func prepare(
        theta: MLXArray, pairs: MLXArray, channelScales: MLXArray, groupSize: Int
    ) {
        cosTheta = MLX.cos(theta)
        sinTheta = MLX.sin(theta)
        packedPairs = packPairs(pairs, groupSize: groupSize)
        scalesFlat = channelScales.reshaped(-1)
    }

    /// The derived tensors, for batching one `eval` across many modules.
    var all: [MLXArray] { [cosTheta, sinTheta, packedPairs, scalesFlat] }
}

/// Load-time finalization hook shared by every rotation-carrying module.
///
/// The loader walks leaf modules and finalizes each conformer after the
/// checkpoint update — a protocol rather than a class enumeration so a new
/// rotation carrier cannot be silently skipped (a missed module would keep
/// its degenerate placeholder state and produce wrong numbers without ever
/// crashing).
///
/// `prepareDerivedRotationState()` returns the freshly derived tensors
/// *unmaterialized*: the loader batches a single `eval` over every module's
/// tensors instead of paying one GPU round-trip per module (~480 modules on
/// a 48-layer MoE). Discarding the result leaves the tensors lazy until
/// first use — harmless for tests, but the loader must eval them so
/// materialization stays out of the first forward pass's graph. Deriving
/// lazily *during* a forward pass was issue #157.
protocol RotationStatePreparing: AnyObject {
    /// Recompute rotation-derived tensors from the loaded checkpoint
    /// parameters. Must run after `update(parameters:)` and never
    /// concurrently with forward passes — the loader owns this call.
    @discardableResult
    func prepareDerivedRotationState() -> [MLXArray]
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
public class PairwiseRotation: Module, RotationStatePreparing {

    // Rotation parameters — discovered by Module reflection for update(parameters:).
    // `channelScales` uses @ParameterInfo so it can keep the snake_case checkpoint
    // key while having a Swift-idiomatic property name.
    let theta: MLXArray
    let pairs: MLXArray
    @ParameterInfo(key: "channel_scales") var channelScales: MLXArray

    let groupSize: Int

    // Populated once by `prepareDerivedRotationState()` after the checkpoint
    // parameters are loaded (see ParoQuantLoader), and never mutated
    // afterwards. See `RotationDerivedState` for why the underscore prefix
    // keeps it out of weight loading.
    private var _rotation: RotationDerivedState

    /// - Precondition: `dims` is a positive multiple of an even `groupSize`,
    ///   and `krot >= 1` — see `rotationGeometryProblem`.
    public init(dims: Int, groupSize: Int, krot: Int) {
        assertRotationGeometry(dims: dims, groupSize: groupSize, krot: krot)
        self.theta = MLXArray.zeros([krot, dims / 2])
        self.pairs = MLXArray.zeros([krot, dims], type: Int16.self)
        // Assign through `.wrappedValue` so the `@ParameterInfo(key:)` metadata
        // survives init — see the matching note in RotateQuantizedLinear.
        self._channelScales.wrappedValue = MLXArray.ones([1, dims])
        self.groupSize = groupSize
        self._rotation = RotationDerivedState(dims: dims, krot: krot)

        super.init()

        // Rotation parameters are inference-only checkpoint constants, never
        // trained — the same contract `QuantizedLinear` applies to its scales
        // and biases. Freezing keeps `trainableParameters()` empty on any
        // module that composes this one, which `SwitchGLU`'s direct weighted
        // reduction requires.
        self.freeze()
    }

    /// See `RotationStatePreparing` — loader-owned, results batched into a
    /// single load-time `eval` with every other rotation module's.
    @discardableResult
    public func prepareDerivedRotationState() -> [MLXArray] {
        _rotation.prepare(
            theta: theta, pairs: pairs, channelScales: channelScales, groupSize: groupSize)
        return _rotation.all
    }

    /// Apply channel scaling + pairwise Givens rotations to the last axis.
    ///
    /// Accepts any leading shape (the MoE path passes gathered 4-D
    /// activations); the input is flattened to 2-D for the kernel and the
    /// original shape is restored on return. Empty batches pass through
    /// unchanged (guarded in `dispatchPairwiseRotation`). No mutable state
    /// is read or written by this method.
    ///
    /// Kernel selection lives in `dispatchPairwiseRotation`: groupSize == 128
    /// takes the simdgroup-resident kernel, any other group size the generic
    /// fallback specialised on that size.
    public func rotate(_ x: MLXArray) -> MLXArray {
        let shape = x.shape
        return dispatchPairwiseRotation(
            x.reshaped(-1, _rotation.scalesFlat.dim(0)),
            state: _rotation, groupSize: groupSize, krot: theta.dim(0)
        ).reshaped(shape)
    }
}
