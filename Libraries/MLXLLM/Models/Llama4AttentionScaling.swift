// Copyright © 2026 Apple Inc.

import MLX

/// Per-row query scales for attention shaped `[B, H, L, D]`.
func batchedLlama4AttentionScale(
    offsets: MLXArray, length: Int, beta: Float, maxPositionEmbeddings: Int
) -> MLXArray {
    let positions = offsets[0..., .newAxis] + MLXArray.arange(length)[.newAxis]
    // Negative positions belong to left padding and must not produce log(0).
    let nonnegative = MLX.maximum(positions, MLXArray(Int32(0))).asType(.float32)
    let scaling =
        1 + beta
        * MLX.log(1 + MLX.floor(nonnegative / Float(maxPositionEmbeddings)))
    return scaling[0..., .newAxis, 0..., .newAxis]
}
