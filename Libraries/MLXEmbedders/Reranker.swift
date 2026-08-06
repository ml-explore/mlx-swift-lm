// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon

/// Encoder models with a sequence-classification head can provide reranker logits.
///
/// BERT, RoBERTa, XLM-RoBERTa, and similar encoder checkpoints often expose a small
/// classifier head for cross-encoder reranking. Conforming models should return logits
/// with shape `[batch, labels]` or another shape that can be flattened per batch row.
public protocol RerankerModel: EmbeddingModel {
    func score(
        _ inputs: MLXArray,
        positionIds: MLXArray?,
        tokenTypeIds: MLXArray?,
        attentionMask: MLXArray?
    ) throws -> MLXArray
}

/// Selects a scalar relevance score from sequence-classification logits.
///
/// Reranker checkpoints are not all trained with the same classifier convention:
/// some return a single relevance logit, while others return two class logits. Select
/// the strategy that matches the checkpoint's model card.
public enum RerankerLogitSelection: Sendable {
    /// Use the only logit, or the last logit if the head returns more than one value.
    ///
    /// This is the default for single-logit rerankers such as BGE reranker checkpoints.
    case singleLogit

    /// Use the raw logit at `index`.
    case classLogit(index: Int)

    /// Use `positive - negative`.
    ///
    /// This is usually the right choice for two-label heads where label 0 is not relevant
    /// and label 1 is relevant.
    case logitDifference(positive: Int, negative: Int)

    /// Use the softmax probability at `index`.
    ///
    /// Use this when downstream code expects a probability rather than an unbounded margin.
    case softmaxProbability(index: Int)
}

/// Configuration for encoder rerankers such as BGE reranker models.
///
/// The configuration controls tokenization, padding, score extraction, and score
/// calibration for encoder cross-encoders.
///
/// Example:
///
/// ```swift
/// let results = try await embedder.process(
///     RerankRequest(query: "swift arrays", documents: candidates),
///     configuration: .bgeRerankerV2M3)
/// ```
///
/// If a checkpoint has a two-label classifier head, choose an explicit logit selection:
///
/// ```swift
/// let configuration = RerankerConfiguration(
///     logitSelection: .logitDifference(positive: 1, negative: 0),
///     scoreTransform: .sigmoid)
/// ```
public struct RerankerConfiguration: Sendable {
    /// Converts query-document strings into model inputs.
    public var inputProcessor: any RerankerInputProcessor

    /// Selects the scalar relevance score from classifier logits.
    public var logitSelection: RerankerLogitSelection

    /// Converts the selected raw score into the final public score.
    public var scoreTransform: RerankerScoreTransform

    /// Maximum token count per query-document pair, including special tokens.
    public var maxInputTokens: Int?

    /// Token ID used to pad batched encoder inputs.
    public var padTokenId: Int

    public init(
        inputProcessor: any RerankerInputProcessor = XLMRobertaRerankerInputProcessor(),
        logitSelection: RerankerLogitSelection = .singleLogit,
        scoreTransform: RerankerScoreTransform = .sigmoid,
        maxInputTokens: Int? = 8_192,
        padTokenId: Int = 0
    ) {
        self.inputProcessor = inputProcessor
        self.logitSelection = logitSelection
        self.scoreTransform = scoreTransform
        self.maxInputTokens = maxInputTokens
        self.padTokenId = padTokenId
    }

    /// Default configuration for `BAAI/bge-reranker-v2-m3`.
    public static let bgeRerankerV2M3 = RerankerConfiguration()
}

extension EmbedderModelContainer {
    /// Process a reranking request with an encoder reranker.
    ///
    /// The default ordering preserves the caller's document order.
    ///
    /// Use this method when you want scores aligned to the input candidates. Use
    /// ``rerank(query:documents:topK:configuration:)`` when you want documents sorted by
    /// descending relevance.
    public func process(
        _ request: RerankRequest,
        configuration: RerankerConfiguration = .bgeRerankerV2M3
    ) async throws -> [RerankResult] {
        guard !request.documents.isEmpty else { return [] }

        return try await perform(nonSendable: (request, configuration)) { context, values in
            let (request, configuration) = values
            guard let model = context.model as? any RerankerModel else {
                throw RerankerError.unsupportedModel(
                    "\(type(of: context.model)) does not expose reranker logits.")
            }

            let results = try scoreBatch(
                query: request.query,
                documents: request.documents,
                model: model,
                tokenizer: context.tokenizer,
                configuration: configuration)

            return orderedResults(results, by: request.resultOrdering)
        }
    }

    /// Score a single query-document pair with an encoder reranker.
    ///
    /// This is a convenience wrapper around ``process(_:configuration:)``.
    public func score(
        query: String,
        document: String,
        configuration: RerankerConfiguration = .bgeRerankerV2M3
    ) async throws -> RerankResult {
        let results = try await rerank(
            query: query, documents: [document], configuration: configuration)
        guard let result = results.first else {
            throw RerankerError.emptyPrompt
        }
        return result
    }

    /// Score and sort documents by descending relevance with an encoder reranker.
    ///
    /// This is the top-k retrieval convenience API. Use ``process(_:configuration:)`` if
    /// you need scores in the same order as the input documents. Pass `topK` to return
    /// only the highest-scoring results. A `nil` value returns all scored documents; a
    /// non-positive value returns an empty array.
    ///
    /// - Parameters:
    ///   - query: Search query or user intent to compare against each document.
    ///   - documents: Candidate documents to score.
    ///   - topK: Optional maximum number of sorted results to return.
    ///   - configuration: Encoder reranker tokenization, batching, and score configuration.
    /// - Returns: Scored results sorted by descending relevance, limited by `topK`.
    public func rerank(
        query: String,
        documents: [String],
        topK: Int? = nil,
        configuration: RerankerConfiguration = .bgeRerankerV2M3
    ) async throws -> [RerankResult] {
        let results = try await process(
            .init(query: query, documents: documents, resultOrdering: .sortedByScore),
            configuration: configuration)
        return topResults(results, topK: topK)
    }

    private func scoreBatch(
        query: String,
        documents: [String],
        model: any RerankerModel,
        tokenizer: any Tokenizer,
        configuration: RerankerConfiguration
    ) throws -> [RerankResult] {
        let maxInputTokens = effectiveMaxInputTokens(
            configuration.maxInputTokens,
            modelMaxPositionEmbeddings: model.maxPositionEmbeddings)
        let encoded = try documents.enumerated().map { index, document in
            let input = try configuration.inputProcessor.encode(
                query: query,
                document: document,
                tokenizer: tokenizer,
                maxInputTokens: maxInputTokens)
            guard !input.tokenIds.isEmpty else {
                throw RerankerError.emptyPrompt
            }
            return EncodedRerankerDocument(index: index, document: document, input: input)
        }

        let batch = try makeBatch(encoded, padTokenId: configuration.padTokenId)
        let logits = try model.score(
            batch.inputIds,
            positionIds: nil,
            tokenTypeIds: batch.tokenTypeIds,
            attentionMask: batch.attentionMask)

        logits.eval()
        let rawScores = try selectScores(logits, using: configuration.logitSelection)
        guard rawScores.count == encoded.count else {
            throw RerankerError.unsupportedModel(
                "\(type(of: model)) returned \(rawScores.count) scores for \(encoded.count) documents."
            )
        }

        return zip(encoded, rawScores).map { item, rawScore in
            RerankResult(
                index: item.index,
                document: item.document,
                score: configuration.scoreTransform(rawScore),
                rawScore: rawScore
            )
        }
    }
}

private struct EncodedRerankerDocument {
    var index: Int
    var document: String
    var input: RerankerInput
}

private struct RerankerBatch {
    var inputIds: MLXArray
    var attentionMask: MLXArray
    var tokenTypeIds: MLXArray?
}

private func makeBatch(
    _ encoded: [EncodedRerankerDocument],
    padTokenId: Int
) throws -> RerankerBatch {
    let maxLength = encoded.map(\.input.tokenIds.count).max() ?? 0
    guard maxLength > 0 else { throw RerankerError.emptyPrompt }

    let needsTokenTypes = encoded.contains { $0.input.tokenTypeIds != nil }
    var inputIds = [Int]()
    var attentionMask = [Int32]()
    var tokenTypeIds = [Int]()
    inputIds.reserveCapacity(encoded.count * maxLength)
    attentionMask.reserveCapacity(encoded.count * maxLength)
    tokenTypeIds.reserveCapacity(encoded.count * maxLength)

    for item in encoded {
        let paddingCount = maxLength - item.input.tokenIds.count
        inputIds += item.input.tokenIds
        inputIds += Array(repeating: padTokenId, count: paddingCount)
        attentionMask += Array(repeating: Int32(1), count: item.input.tokenIds.count)
        attentionMask += Array(repeating: Int32(0), count: paddingCount)

        if needsTokenTypes {
            let types =
                item.input.tokenTypeIds ?? Array(repeating: 0, count: item.input.tokenIds.count)
            guard types.count == item.input.tokenIds.count else {
                throw RerankerError.unsupportedModel(
                    "tokenTypeIds count (\(types.count)) does not match tokenIds count (\(item.input.tokenIds.count))."
                )
            }
            tokenTypeIds += types
            tokenTypeIds += Array(repeating: 0, count: paddingCount)
        }
    }

    let shape = [encoded.count, maxLength]
    return RerankerBatch(
        inputIds: MLXArray(inputIds).reshaped(shape),
        attentionMask: MLXArray(attentionMask).reshaped(shape),
        tokenTypeIds: needsTokenTypes ? MLXArray(tokenTypeIds).reshaped(shape) : nil)
}

private func selectScores(
    _ logits: MLXArray,
    using selection: RerankerLogitSelection
) throws -> [Double] {
    let matrix: MLXArray
    if logits.ndim == 1 {
        matrix = logits.reshaped(1, -1)
    } else {
        matrix = logits.reshaped(logits.dim(0), -1)
    }

    let batchSize = matrix.dim(0)
    let columnCount = matrix.dim(1)
    let values = matrix.asArray(Float.self).map(Double.init)

    func value(row: Int, column: Int) throws -> Double {
        guard column >= 0, column < columnCount else {
            throw RerankerError.unsupportedModel(
                "Reranker logit index \(column) is out of bounds for \(columnCount) logits.")
        }
        return values[row * columnCount + column]
    }

    return try (0 ..< batchSize).map { row in
        switch selection {
        case .singleLogit:
            return try value(row: row, column: columnCount - 1)
        case .classLogit(let index):
            return try value(row: row, column: index)
        case .logitDifference(let positive, let negative):
            return try value(row: row, column: positive) - value(row: row, column: negative)
        case .softmaxProbability(let index):
            let selectedValue = try value(row: row, column: index)
            let rowValues = (0 ..< columnCount).map { values[row * columnCount + $0] }
            let maxValue = rowValues.max() ?? 0
            let exps = rowValues.map { Foundation.exp($0 - maxValue) }
            let denominator = exps.reduce(0, +)
            guard denominator.isFinite, denominator > 0 else {
                throw RerankerError.unsupportedModel(
                    "Reranker softmax denominator is not finite for \(columnCount) logits.")
            }
            return Foundation.exp(selectedValue - maxValue) / denominator
        }
    }
}

private func effectiveMaxInputTokens(
    _ configuredMaxInputTokens: Int?,
    modelMaxPositionEmbeddings: Int?
) -> Int? {
    switch (configuredMaxInputTokens, modelMaxPositionEmbeddings) {
    case (.some(let configured), .some(let modelLimit)):
        min(configured, modelLimit)
    case (.some(let configured), .none):
        configured
    case (.none, .some(let modelLimit)):
        modelLimit
    case (.none, .none):
        nil
    }
}
