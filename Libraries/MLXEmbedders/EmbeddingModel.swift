// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

public struct EmbeddingModelOutput {
    public let hiddenStates: MLXArray?
    public let pooledOutput: MLXArray?
}

public protocol EmbeddingModel: BaseLanguageModel {
    var vocabularySize: Int { get }
    var poolingStrategy: Pooling.Strategy? { get }

    func callAsFunction(
        _ inputs: MLXArray,
        positionIds: MLXArray?,
        tokenTypeIds: MLXArray?,
        attentionMask: MLXArray?
    ) -> EmbeddingModelOutput
}

extension EmbeddingModel {
    public var poolingStrategy: Pooling.Strategy? {
        nil
    }

    func callAsFunction(
        _ inputs: MLXArray,
        positionIds: MLXArray? = nil,
        tokenTypeIds: MLXArray? = nil,
        attentionMask: MLXArray? = nil
    ) -> EmbeddingModelOutput {
        return callAsFunction(
            inputs, positionIds: positionIds, tokenTypeIds: tokenTypeIds,
            attentionMask: attentionMask)
    }
}
