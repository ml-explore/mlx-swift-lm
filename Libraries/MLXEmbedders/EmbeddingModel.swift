// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
@_spi(MaterializedModule) import MLXNN

public struct EmbeddingModelOutput {
    public let hiddenStates: MLXArray?
    public let pooledOutput: MLXArray?
}

public protocol EmbeddingModel: BaseLanguageModel {
    var vocabularySize: Int { get }
    var poolingStrategy: Pooling.Strategy? { get }

    /// The maximum number of position embeddings supported by this model, or `nil`
    /// if the model uses a position encoding (e.g. RoPE) that handles arbitrary lengths.
    ///
    /// Inputs exceeding this length are truncated internally with a warning.
    /// Callers may pre-truncate for efficiency or to implement custom strategies
    /// (e.g. chunking with pooling).
    var maxPositionEmbeddings: Int? { get }

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

    public var maxPositionEmbeddings: Int? { nil }

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

public typealias TrainableEmbeddingModel = EmbeddingModel & Module

extension MaterializedModule: EmbeddingModel where LayerType: EmbeddingModel {

    public var vocabularySize: Int { _base.vocabularySize }
    public var poolingStrategy: Pooling.Strategy? { _base.poolingStrategy }
    public var maxPositionEmbeddings: Int? { _base.maxPositionEmbeddings }

    public func callAsFunction(
        _ inputs: MLXArray,
        positionIds: MLXArray? = nil,
        tokenTypeIds: MLXArray? = nil,
        attentionMask: MLXArray? = nil
    ) -> EmbeddingModelOutput {
        return _base(
            inputs, positionIds: positionIds, tokenTypeIds: tokenTypeIds,
            attentionMask: attentionMask)
    }
}
