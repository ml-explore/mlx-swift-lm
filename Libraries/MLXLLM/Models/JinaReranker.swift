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

/// Jina reranker v3 and v3.5 listwise model.
///
/// The checkpoint declares `model_type: qwen3` but `architectures: ["JinaForRanking"]`.
/// It uses Qwen3 hidden states at `<|embed_token|>` and `<|rerank_token|>` positions,
/// projects them with `projector.safetensors`, then scores documents by cosine similarity.
public final class JinaRerankerModel: Module, LanguageModel, KVCacheDimensionProvider,
    JinaRerankerEmbeddingModel, AdditionalWeightFilesProviding
{
    public let vocabularySize: Int
    public let kvHeads: [Int]
    private let configuration: Qwen3Configuration

    @ModuleInfo(key: "model") var model: Qwen3ModelInner
    @ModuleInfo(key: "projector") private var projector: JinaRerankerProjector

    public init(_ configuration: Qwen3Configuration) {
        self.configuration = configuration
        self.vocabularySize = configuration.vocabularySize
        self.kvHeads = (0 ..< configuration.hiddenLayers).map { _ in configuration.kvHeads }
        _model.wrappedValue = Qwen3ModelInner(configuration)
        _projector.wrappedValue = JinaRerankerProjector(hiddenSize: configuration.hiddenSize)
    }

    public func prepare(
        _ input: LMInput,
        cache: [KVCache],
        state: LMOutput.State?,
        prefill: PrefillParameters
    ) throws -> PrepareResult {
        .tokens(input.text)
    }

    public func callAsFunction(_ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?)
        -> LMOutput
    {
        .init(logits: callAsFunction(input.tokens, cache: cache))
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        model.embedTokens.asLinear(model(inputs, cache: cache))
    }

    package func embeddings(input: RerankerInput, documentCount: Int, stepSize: Int) throws
        -> JinaRerankerEmbeddings
    {
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

        guard let queryPosition = queryPositions.last else {
            throw RerankerError.missingSpecialToken("<|rerank_token|>")
        }
        guard queryPositions.count == markerTokenIds.queryCount else {
            throw RerankerError.unsupportedModel(
                "Expected \(markerTokenIds.queryCount) <|rerank_token|> markers, found \(queryPositions.count)."
            )
        }
        guard documentPositions.count == documentCount,
            documentPositions.allSatisfy({ $0 < queryPosition })
        else {
            throw RerankerError.missingSpecialToken("<|embed_token|>")
        }

        // Attention is causal, so tokens after the final query marker cannot change its state.
        let length = queryPosition + 1
        let tokens = MLXArray(input.tokenIds[..<length]).reshaped(1, -1)
        let cache = try newCache(parameters: nil)
        var pending = (documentPositions + [queryPosition])[...]
        var states = [MLXArray]()
        for start in stride(from: 0, to: length, by: stepSize) {
            let end = min(start + stepSize, length)
            let hidden = model(tokens[0..., start ..< end], cache: cache)[0]
            let positions = pending.prefix { $0 < end }
            pending.removeFirst(positions.count)
            if !positions.isEmpty {
                states.append(hidden.take(MLXArray(positions.map { Int32($0 - start) }), axis: 0))
            }
            eval(cache)
            eval(states)
        }

        let projected = projector(concatenated(states, axis: 0)).asType(.float32)
        let embeddings = JinaRerankerEmbeddings(
            documents: projected[..<documentCount], query: projected[documentCount...])
        eval(embeddings.documents, embeddings.query)
        return embeddings
    }

    public func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
        try configuration.makeCache(parameters: parameters)
    }

    /// `jinaai/jina-reranker-v3-mlx` keeps the projector in `projector.safetensors`, which
    /// neither the conventional `model*.safetensors` names nor `model.safetensors.index.json`
    /// select, so it has to be requested by name -- as the checkpoint's own `rerank.py` does.
    ///
    /// `jinaai/jina-reranker-v3` packages the same model with the projector in its single
    /// `model.safetensors` and ships no such file. A name that is not present is ignored, so the
    /// declaration costs that layout nothing.
    public var additionalWeightFiles: [String] { ["projector.safetensors"] }

    /// Where the two projector layers arrive under each packaging of this model.
    ///
    /// - `jinaai/jina-reranker-v3` stores the projector as an `nn.Sequential` of
    ///   `Linear, ReLU, Linear`, so its layers are numbered by position in the main weight file.
    /// - `jinaai/jina-reranker-v3-mlx` renames them and moves them into `projector.safetensors`,
    ///   where they lose the `projector` prefix entirely.
    private static let projectorKeys = [
        "projector.0.weight": "projector.linear1.weight",
        "projector.2.weight": "projector.linear2.weight",
        "linear1.weight": "projector.linear1.weight",
        "linear2.weight": "projector.linear2.weight",
    ]

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        weights.reduce(into: [:]) { result, item in
            result[Self.projectorKeys[item.key] ?? item.key] = item.value
        }
    }
}
