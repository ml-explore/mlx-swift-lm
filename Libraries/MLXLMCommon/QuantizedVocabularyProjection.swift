// Copyright © 2026 Apple Inc.

import MLX
import MLXNN

/// Retain the final quantized matrix tiles when only the last logit row is needed.
package func quantizedVocabularyProjectionInput(_ hidden: MLXArray, projection: Module) -> MLXArray
{
    guard hidden.dim(0) == 1,
        type(of: projection) == QuantizedEmbedding.self
            || type(of: projection) == QuantizedLinear.self,
        let quantized = projection as? any Quantized,
        quantized.mode == .affine, quantized.bits == 4 || quantized.bits == 8
    else {
        return hidden
    }

    // Preserve the quantized matrix kernel and its 32/64-row tile alignment.
    // A single-row projection changes the reduction order.
    let start = max(0, ((hidden.dim(1) - 32) / 64) * 64)
    return start > 0 ? hidden[0..., start..., 0...] : hidden
}
