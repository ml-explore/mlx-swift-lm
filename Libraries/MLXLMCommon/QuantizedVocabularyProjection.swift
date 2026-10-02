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

    // Keep matrix kernels and 64-row tile alignment. Some GPUs use vector
    // kernels for up to 32 rows, which changes the reduction order.
    var start = max(0, ((hidden.dim(1) - 33) / 64) * 64)
    guard start > 0 else { return hidden }
    let outputSize: Int
    if let embedding = projection as? QuantizedEmbedding {
        outputSize = embedding.weight.dim(0)
    } else if let linear = projection as? QuantizedLinear {
        outputSize = linear.weight.dim(0)
    } else {
        return hidden
    }

    // MLX 0.32.3 chooses split-K partitions from the matrix dimensions.
    // Keep the same partition count so trimming does not change the reduction.
    let inputSize = hidden.dim(-1)
    let alignment = max(32, quantized.groupSize)
    let outputTiles = (outputSize + 31) / 32
    func partitions(rows: Int) -> Int {
        let tiles = ((rows + 31) / 32) * outputTiles
        var count = min(max(1, 512 / tiles), inputSize / alignment)
        while count > 1 && inputSize % (count * alignment) != 0 {
            count -= 1
        }
        return max(1, count)
    }
    let fullPartitions = partitions(rows: hidden.dim(1))
    while start > 0 && partitions(rows: hidden.dim(1) - start) != fullPartitions {
        start -= 64
    }
    return start > 0 ? hidden[0..., start..., 0...] : hidden
}
