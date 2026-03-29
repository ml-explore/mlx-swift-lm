import Foundation
import MLX
import MLXNN

// MARK: - Metal Kernel Source

/// Pairwise Givens rotation kernel for Metal (Apple Silicon).
/// Template parameters are substituted at compile time.
nonisolated private func metalSource(
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

/// Cached compiled Metal kernels keyed by tile size.
/// `nonisolated(unsafe)` — kernel creation is idempotent (same tile size always produces
/// the same compiled kernel), so concurrent stores are benign. Same pattern as
/// `GatedDeltaKernelManager` in the existing codebase.
nonisolated(unsafe) private var kernelCache: [Int: MLXFast.MLXFastKernel] = [:]

nonisolated private func getRotationKernel(tile: Int) -> MLXFast.MLXFastKernel {
    if let cached = kernelCache[tile] {
        return cached
    }
    let kernel = MLXFast.metalKernel(
        name: "paro_rotate_r\(tile)",
        inputNames: ["x", "packed_pairs", "cos_theta", "sin_theta", "channel_scales", "params"],
        outputNames: ["out"],
        source: metalSource(rowsPerTile: tile)
    )
    kernelCache[tile] = kernel
    return kernel
}

// MARK: - Pair Packing

/// Pack int16 pair indices into int32 for the Metal kernel.
///
/// Each pair `(i, j)` is packed as `i | (j << 16)` within each group.
nonisolated private func packPairs(_ pairs: MLXArray, groupSize: Int) -> MLXArray {
    let krot = pairs.dim(0)
    let numGroups = pairs.dim(1) / groupSize

    // Reshape to [krot, numGroups, groupSize]
    let p = pairs.reshaped(krot, numGroups, groupSize).asType(.int32)

    // Even indices (lo) and odd indices (hi) within each group
    let lo = p[0..., 0..., .stride(by: 2)]
    let hi = p[0..., 0..., .stride(from: 1, by: 2)]
    return (lo | (hi << 16)).reshaped(krot, -1)
}

// MARK: - RotateQuantizedLinear

/// Pairwise Givens rotation + quantized matmul.
///
/// Subclasses `QuantizedLinear` so it can replace `Linear` in `@ModuleInfo` slots
/// via `update(modules:)`. Only overrides `callAsFunction` to insert the rotation
/// step before the standard quantized matmul.
///
/// Rotation is applied to activations at runtime via a Metal kernel, preserving
/// the quantization-friendly properties of the original weights.
nonisolated open class RotateQuantizedLinear: QuantizedLinear {

    // Rotation parameters — discovered by Module reflection for update(parameters:)
    let theta: MLXArray
    let pairs: MLXArray
    let channel_scales: MLXArray  // swiftlint:disable:this identifier_name

    /// Pre-computed rotation data, lazily initialized on first forward pass.
    private struct CachedRotation {
        let cos: MLXArray
        let sin: MLXArray
        let packedPairs: MLXArray
        let scalesFlat: MLXArray
        let dim: Int
        let halfGroup: Int
        let numGroups: Int
        let krot: Int
    }

    private var cached: CachedRotation?
    private var cachedParams: MLXArray?
    private var cachedBatch: Int = -1

    public init(
        inputDims: Int, outputDims: Int, hasBias: Bool,
        groupSize: Int, bits: Int, krot: Int
    ) {
        self.theta = MLXArray.zeros([krot, inputDims / 2])
        self.pairs = MLXArray.zeros([krot, inputDims], type: Int16.self)
        self.channel_scales = MLXArray.ones([1, inputDims])

        super.init(
            weight: MLXArray.zeros([outputDims, inputDims * bits / 32], type: UInt32.self),
            bias: hasBias ? MLXArray.zeros([outputDims]) : nil,
            scales: MLXArray.zeros([outputDims, inputDims / groupSize]),
            biases: MLXArray.zeros([outputDims, inputDims / groupSize]),
            groupSize: groupSize,
            bits: bits
        )
    }

    private func ensureCached() -> CachedRotation {
        if let c = cached { return c }
        let dim = theta.dim(1) * 2
        let cosTheta = MLX.cos(theta)
        let sinTheta = MLX.sin(theta)
        let packed = packPairs(pairs, groupSize: groupSize)
        let flat = channel_scales.reshaped(-1)

        // Force GPU materialization so rotation constants are resident
        // and not recomputed as part of each forward pass graph.
        eval(cosTheta, sinTheta, packed, flat)

        let c = CachedRotation(
            cos: cosTheta,
            sin: sinTheta,
            packedPairs: packed,
            scalesFlat: flat,
            dim: dim,
            halfGroup: groupSize / 2,
            numGroups: dim / groupSize,
            krot: theta.dim(0)
        )
        cached = c
        return c
    }

    private func rotate(_ x: MLXArray, cache c: CachedRotation) -> MLXArray {
        let batch = x.dim(0)
        let tile = batch <= 1 ? 1 : 4

        // Cache params array — batch rarely changes during autoregressive generation
        if batch != cachedBatch {
            cachedParams = MLXArray([Int32(batch), Int32(c.dim), Int32(c.krot), Int32(groupSize)])
            cachedBatch = batch
        }

        let gridX = ((batch + tile - 1) / tile) * c.halfGroup
        return getRotationKernel(tile: tile)(
            [x, c.packedPairs, c.cos, c.sin, c.scalesFlat, cachedParams!],
            grid: (gridX, c.numGroups, 1),
            threadGroup: (c.halfGroup, 1, 1),
            outputShapes: [x.shape],
            outputDTypes: [x.dtype]
        )[0]
    }

    /// Forward pass: applies pairwise Givens rotation then quantized matmul.
    ///
    /// Computes `y = quantizedMM(rotate(x), W)` where `rotate(x)` fuses channel
    /// scaling and Givens rotations in a single Metal kernel.
    open override func callAsFunction(_ x: MLXArray) -> MLXArray {
        let c = ensureCached()
        let shape = x.shape
        let rotated = rotate(x.reshaped(-1, c.dim), cache: c)

        var y = quantizedMM(
            rotated.reshaped(shape), weight,
            scales: scales, biases: biases,
            transpose: true, groupSize: groupSize, bits: bits
        )
        if let bias { y = y + bias }
        return y
    }
}
