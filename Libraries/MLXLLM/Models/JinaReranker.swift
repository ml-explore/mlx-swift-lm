// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

private final class JinaRerankerProjector: Module {
    @ModuleInfo(key: "linear1") var linear1: Linear
    @ModuleInfo(key: "linear2") var linear2: Linear

    init(hiddenSize: Int, projectionSize: Int = 512) {
        _linear1.wrappedValue = Linear(hiddenSize, projectionSize, bias: false)
        _linear2.wrappedValue = Linear(projectionSize, projectionSize, bias: false)
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        linear2(relu(linear1(input)))
    }
}

/// Jina reranker v3 listwise model.
///
/// The checkpoint declares `model_type: qwen3` but `architectures: ["JinaForRanking"]`.
/// It uses Qwen3 hidden states at `<|embed_token|>` and `<|rerank_token|>` positions,
/// projects them with `projector.safetensors`, then scores documents by cosine similarity.
public final class JinaRerankerModel: Module, LanguageModel, KVCacheDimensionProvider,
    ListwiseRerankerModel
{
    public let vocabularySize: Int
    public let kvHeads: [Int]

    @ModuleInfo(key: "model") var model: Qwen3ModelInner
    @ModuleInfo(key: "projector") private var projector: JinaRerankerProjector

    public init(_ configuration: Qwen3Configuration) {
        self.vocabularySize = configuration.vocabularySize
        self.kvHeads = (0 ..< configuration.hiddenLayers).map { _ in configuration.kvHeads }
        _model.wrappedValue = Qwen3ModelInner(configuration)
        _projector.wrappedValue = JinaRerankerProjector(hiddenSize: configuration.hiddenSize)
    }

    public func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws
        -> PrepareResult
    {
        .tokens(input.text)
    }

    public func callAsFunction(_ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?)
        -> LMOutput
    {
        .init(logits: callAsFunction(input.tokens, cache: cache))
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        model(inputs, cache: cache)
    }

    public func score(input: RerankerInput, documentCount: Int) throws -> [Double] {
        guard !input.tokenIds.isEmpty else {
            throw RerankerError.emptyPrompt
        }
        guard let markerTokenIds = input.markerTokenIds else {
            throw RerankerError.unsupportedModel(
                "Jina reranker input is missing marker token IDs.")
        }

        let queryPositions = input.tokenIds.indices.filter {
            input.tokenIds[$0] == markerTokenIds.query
        }
        let documentPositions = input.tokenIds.indices.filter {
            input.tokenIds[$0] == markerTokenIds.document
        }

        guard let queryPosition = queryPositions.first else {
            throw RerankerError.missingSpecialToken("<|rerank_token|>")
        }
        guard queryPositions.count == 1 else {
            throw RerankerError.unsupportedModel(
                "Expected exactly one <|rerank_token|>, found \(queryPositions.count).")
        }
        guard documentPositions.count == documentCount else {
            throw RerankerError.missingSpecialToken("<|embed_token|>")
        }

        let inputIds = MLXArray(input.tokenIds).reshaped(1, -1)
        let hiddenStates = model(inputIds, cache: nil)[0]

        let queryHidden = hiddenStates[queryPosition][.newAxis, 0...]
        let documentHidden = stacked(documentPositions.map { hiddenStates[$0] })

        let queryEmbedding = projector(queryHidden)
        let documentEmbeddings = projector(documentHidden)

        let numerator = MLX.sum(documentEmbeddings * queryEmbedding, axis: -1)
        let documentNorm = MLX.sqrt(MLX.sum(documentEmbeddings * documentEmbeddings, axis: -1))
        let queryNorm = MLX.sqrt(MLX.sum(queryEmbedding * queryEmbedding, axis: -1))
        let scores = numerator / (documentNorm * queryNorm)

        scores.eval()
        return scores.asArray(Float.self).map(Double.init)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        weights.reduce(into: [:]) { result, item in
            switch item.key {
            case "linear1.weight":
                result["projector.linear1.weight"] = item.value
            case "linear2.weight":
                result["projector.linear2.weight"] = item.value
            default:
                result[item.key] = item.value
            }
        }
    }
}
