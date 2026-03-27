// Copyright © 2025 INQTR. All rights reserved.

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
/// After `preRotateWeights()` is called, the rotation is baked into the weights
/// and runtime is identical to a plain QuantizedLinear (no rotation kernel dispatch).
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

    /// When true, weights have been pre-rotated and callAsFunction skips the rotation kernel.
    private var preRotated = false

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

    /// Pre-rotate weights at load time to eliminate per-token rotation kernel dispatches.
    ///
    /// Transforms: `y = quantizedMM(rotate(x), W)` → `y = quantizedMM(x, W_pre)`
    /// where `W_pre` incorporates both channel_scales and the Givens rotation.
    ///
    /// The rotation matrix is block-diagonal (independent per group), so we build it
    /// group-by-group to keep memory usage at O(group_size²) per group.
    public func preRotateWeights() throws {
        let c = ensureCached()
        let dim = c.dim
        let gs = groupSize
        let numGroups = c.numGroups

        // 1. Dequantize current weights to float16
        let wFloat = dequantized(
            weight, scales: scales, biases: biases,
            groupSize: gs, bits: bits
        )
        // wFloat is [outputDims, inputDims] in float16

        // 2. Build rotation matrix group-by-group and pre-rotate columns of W.
        //    rotate(I_g) computes the rotation+scale block for group g.
        //    W_pre[:, g] = wFloat[:, g] @ block_g^T
        var preRotatedGroups = [MLXArray]()
        preRotatedGroups.reserveCapacity(numGroups)

        for g in 0 ..< numGroups {
            let colStart = g * gs
            let colEnd = colStart + gs

            // Build [gs, dim] input: identity block at columns [colStart, colEnd), zeros elsewhere
            let eye = MLXArray.identity(gs).asType(.float16)
            if colStart == 0 && colEnd == dim {
                // Single group spans entire dim
                let rotated = rotate(eye, cache: c)
                let wCols = wFloat[0..., colStart ..< colEnd]
                preRotatedGroups.append(matmul(wCols, rotated.transposed()))
            } else {
                let leftPad =
                    colStart > 0
                    ? MLXArray.zeros([gs, colStart], dtype: .float16) : nil
                let rightPad =
                    colEnd < dim
                    ? MLXArray.zeros([gs, dim - colEnd], dtype: .float16) : nil
                var parts = [MLXArray]()
                if let leftPad { parts.append(leftPad) }
                parts.append(eye)
                if let rightPad { parts.append(rightPad) }
                let groupInput = concatenated(parts, axis: 1)

                let rotated = rotate(groupInput, cache: c)
                let block = rotated[0..., colStart ..< colEnd]

                let wCols = wFloat[0..., colStart ..< colEnd]
                preRotatedGroups.append(matmul(wCols, block.transposed()))
            }

            // Eval every few groups to bound computation graph size
            if (g + 1) % 8 == 0 || g == numGroups - 1 {
                eval(preRotatedGroups)
            }
        }

        // 3. Reassemble the full pre-rotated weight matrix
        let wPre = concatenated(preRotatedGroups, axis: 1)
        eval(wPre)

        // 4. Re-quantize
        let (wq, sc, bi) = quantized(wPre, groupSize: gs, bits: bits)

        // 5. Update the QuantizedLinear properties
        let params: [String: MLXArray] = [
            "weight": wq,
            "scales": sc,
            "biases": bi ?? MLXArray.zeros(scales.shape),
        ]
        try self.update(parameters: ModuleParameters.unflattened(params), verify: [])

        preRotated = true
    }

    /// Forward pass: applies pairwise Givens rotation then quantized matmul.
    ///
    /// Standard path computes `y = quantizedMM(rotate(x), W)` where
    /// `rotate(x) = R @ diag(channel_scales) @ x` fuses channel scaling and
    /// Givens rotations in a single Metal kernel.
    ///
    /// - Note: The stored weights are quantized in the rotated+scaled space:
    ///   `W_q = Q(R @ diag(s) @ W_train)`. This means the effective computation
    ///   is `x @ diag(s²) @ W_train^T` rather than `x @ W_train^T`. The `diag(s²)`
    ///   factor is compensated during the original ParoQuant optimization, which
    ///   jointly trains W, θ, and s to match the pretrained model's layer outputs.
    ///   This is an inherent property of the algorithm (present in the reference
    ///   Python implementation as well), not a bug in this port.
    open override func callAsFunction(_ x: MLXArray) -> MLXArray {
        if preRotated {
            // Fast path: weights already include rotation + channel_scales
            var y = quantizedMM(
                x, weight,
                scales: scales, biases: biases,
                transpose: true, groupSize: groupSize, bits: bits
            )
            if let bias { y = y + bias }
            return y
        }

        // Standard path: apply rotation kernel then quantized matmul
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
