// Copyright © 2026 Apple Inc.

import Foundation
import HuggingFace
import MLX
import MLXEmbedders
import MLXHuggingFace
import MLXLMCommon
import Tokenizers

public protocol MLXEmbeddingServerEngine: Sendable {
    func availableModels() async throws -> [MLXServerModel]
    func createEmbedding(request: OpenAIEmbeddingRequest) async throws -> OpenAIEmbeddingResponse
}

public struct MLXEmbedderContainerEngine: MLXEmbeddingServerEngine {
    private let modelID: String
    private let model: EmbedderModelContainer

    public init(modelID: String, model: EmbedderModelContainer) {
        self.modelID = modelID
        self.model = model
    }

    public func availableModels() async throws -> [MLXServerModel] {
        [.init(id: modelID)]
    }

    public func createEmbedding(request: OpenAIEmbeddingRequest) async throws -> OpenAIEmbeddingResponse {
        let texts = request.input.texts
        let normalize = request.normalize ?? true
        let embeddings = await model.perform { context in
            let encoded = texts.map {
                context.tokenizer.encode(text: $0, addSpecialTokens: true)
            }
            let padToken = context.tokenizer.eosTokenId ?? 0
            let maxLength = max(1, encoded.map(\.count).max() ?? 1)
            let padded = stacked(
                encoded.map { tokens in
                    MLXArray(tokens + Array(repeating: padToken, count: maxLength - tokens.count))
                }
            )
            let mask = (padded .!= padToken)
            let tokenTypes = MLXArray.zeros(like: padded)
            let result = context.pooling(
                context.model(
                    padded,
                    positionIds: nil,
                    tokenTypeIds: tokenTypes,
                    attentionMask: mask
                ),
                normalize: normalize,
                applyLayerNorm: true
            )
            result.eval()
            return result.map { $0.asArray(Float.self) }
        }
        let promptTokens = await promptTokenCount(texts)
        return .init(
            data: embeddings.enumerated().map { .init(embedding: $0.element, index: $0.offset) },
            model: request.model,
            usage: .init(promptTokens: promptTokens, completionTokens: 0)
        )
    }

    private func promptTokenCount(_ texts: [String]) async -> Int {
        let tokenizer = await model.tokenizer
        return texts.reduce(0) { total, text in
            total + tokenizer.encode(text: text, addSpecialTokens: true).count
        }
    }
}

public enum MLXServerEmbedderLoader {
    public static func load(configuration: ModelConfiguration) async throws -> EmbedderModelContainer {
        try await EmbedderModelFactory.shared.loadContainer(
            from: #hubDownloader(),
            using: #huggingFaceTokenizerLoader(),
            configuration: configuration
        )
    }
}
