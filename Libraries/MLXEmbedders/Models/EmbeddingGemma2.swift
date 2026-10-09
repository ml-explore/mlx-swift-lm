// Copyright © 2026 Apple Inc.

import MLX
import MLXLMCommon

extension EmbeddingGemma2: EmbeddingModel {
    public var vocabularySize: Int { config.vocabularySize }
    public var poolingStrategy: Pooling.Strategy? { Pooling.Strategy.none }
    public var maxPositionEmbeddings: Int? { EmbeddingGemma2Configuration.contextLength }

    public func callAsFunction(
        _ inputs: MLXArray, positionIds: MLXArray?, tokenTypeIds: MLXArray?,
        attentionMask: MLXArray?
    ) -> EmbeddingModelOutput {
        let tokens = inputs.ndim == 1 ? inputs.expandedDimensions(axis: 0) : inputs
        precondition(tokens.ndim == 2 && tokens.dim(0) > 0 && tokens.dim(1) > 0)
        let mask = attentionMask.map { $0.ndim == 1 ? $0.expandedDimensions(axis: 0) : $0 }
        if let mask { precondition(mask.shape == tokens.shape) }
        let length = min(tokens.dim(1), EmbeddingGemma2Configuration.contextLength)
        let output = embed(
            inputIds: tokens[0..., ..<length], attentionMask: mask.map { $0[0..., ..<length] })
        return EmbeddingModelOutput(hiddenStates: nil, pooledOutput: output)
    }
}
