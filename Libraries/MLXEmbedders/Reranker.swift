// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon

package enum EncoderRerankerScorePolicy: Sendable, Equatable {
    case singleLogit(transform: ScalarTransform)
    case classLogit(index: Int, transform: ScalarTransform)
    case binaryMargin(positive: Int, negative: Int, transform: ScalarTransform)
    case softmaxProbability(classIndex: Int)

    package enum ScalarTransform: Sendable, Equatable {
        case identity
        case sigmoid

        func callAsFunction(_ value: Double) -> Double {
            switch self {
            case .identity:
                value
            case .sigmoid:
                if value >= 0 {
                    1 / (1 + Foundation.exp(-value))
                } else {
                    Foundation.exp(value) / (1 + Foundation.exp(value))
                }
            }
        }
    }
}

package struct EncoderRerankerModelConfiguration: Sendable {
    package enum InputKind: Sendable {
        case bert
        case xlmRoberta
    }

    package var inputKind: InputKind
    package var padTokenID: Int
    package var maxInputTokens: Int?
    package var scorePolicy: EncoderRerankerScorePolicy?
    package var scoreKind: RerankScoreKind
}

/// Encoder models with a sequence-classification head that produces reranker logits.
package protocol RerankerModel: EmbeddingModel {
    var rerankerConfiguration: EncoderRerankerModelConfiguration { get }

    func score(
        _ inputs: MLXArray,
        positionIds: MLXArray?,
        tokenTypeIds: MLXArray?,
        attentionMask: MLXArray?
    ) throws -> MLXArray
}

extension EmbedderModelContainer {
    /// The score semantics declared by the loaded encoder reranker.
    package var rerankerScoreKind: RerankScoreKind? {
        get async {
            await perform { context in
                guard
                    let configuration =
                        (context.model as? any RerankerModel)?.rerankerConfiguration,
                    configuration.scorePolicy != nil
                else {
                    return nil
                }
                return configuration.scoreKind
            }
        }
    }

    /// Score an encoder reranker in original document order.
    package func rerankerScores(
        query: String,
        documents: [String],
        options: RerankExecutionOptions
    ) async throws -> [Double] {
        guard !documents.isEmpty else { return [] }

        return try await perform(nonSendable: (query, documents, options)) { context, values in
            let (query, documents, options) = values
            guard let model = context.model as? any RerankerModel else {
                throw RerankerError.unsupportedModel(
                    "\(type(of: context.model)) does not expose reranker logits.")
            }

            let configuration = model.rerankerConfiguration
            guard let scorePolicy = configuration.scorePolicy else {
                throw RerankerError.unsupportedModel(
                    "Unable to identify the positive class for this encoder reranker. Provide id2label or label2id metadata in config.json."
                )
            }
            let processor: any RerankerInputProcessor =
                switch configuration.inputKind {
                case .bert: BERTRerankerInputProcessor()
                case .xlmRoberta: XLMRobertaRerankerInputProcessor()
                }
            let maxInputTokens = minimum(
                minimum(configuration.maxInputTokens, model.maxPositionEmbeddings),
                options.maxBatchTokens)

            var encoded = try documents.enumerated().map { index, document in
                try Task.checkCancellation()
                let input = try processor.encode(
                    query: query,
                    document: document,
                    tokenizer: context.tokenizer,
                    maxInputTokens: maxInputTokens,
                    truncation: options.truncation)
                guard !input.tokenIds.isEmpty else {
                    throw RerankerError.emptyPrompt
                }
                return EncodedDocument(index: index, input: input)
            }
            encoded.sort {
                if $0.input.tokenIds.count == $1.input.tokenIds.count {
                    $0.index < $1.index
                } else {
                    $0.input.tokenIds.count < $1.input.tokenIds.count
                }
            }

            var scores = [Double?](repeating: nil, count: documents.count)
            for batchDocuments in makeMicroBatches(encoded, options: options) {
                try Task.checkCancellation()
                let batch = try makeBatch(
                    batchDocuments, padTokenID: configuration.padTokenID)
                let logits = try model.score(
                    batch.inputIDs,
                    positionIds: nil,
                    tokenTypeIds: batch.tokenTypeIDs,
                    attentionMask: batch.attentionMask)
                logits.eval()
                let batchScores = try selectScores(
                    logits, batchSize: batchDocuments.count,
                    policy: scorePolicy)
                for (item, score) in zip(batchDocuments, batchScores) {
                    guard score.isFinite else {
                        throw RerankerError.nonFiniteScore(index: item.index, score: score)
                    }
                    scores[item.index] = score
                }
            }

            return try scores.enumerated().map { index, score in
                guard let score else {
                    throw RerankerError.invalidScoreCount(
                        expected: documents.count,
                        actual: scores.compactMap { $0 }.count)
                }
                guard score.isFinite else {
                    throw RerankerError.nonFiniteScore(index: index, score: score)
                }
                return score
            }
        }
    }
}

private struct EncodedDocument {
    var index: Int
    var input: RerankerInput
}

private struct RerankerBatch {
    var inputIDs: MLXArray
    var attentionMask: MLXArray
    var tokenTypeIDs: MLXArray?
}

private func makeMicroBatches(
    _ documents: [EncodedDocument],
    options: RerankExecutionOptions
) -> [[EncodedDocument]] {
    var batches = [[EncodedDocument]]()
    var batch = [EncodedDocument]()
    var longest = 0

    for document in documents {
        let nextLongest = max(longest, document.input.tokenIds.count)
        let nextCount = batch.count + 1
        let exceedsSize = nextCount > options.maxBatchSize
        let exceedsTokens = nextLongest * nextCount > options.maxBatchTokens
        if !batch.isEmpty, exceedsSize || exceedsTokens {
            batches.append(batch)
            batch = []
            longest = 0
        }
        batch.append(document)
        longest = max(longest, document.input.tokenIds.count)
    }
    if !batch.isEmpty {
        batches.append(batch)
    }
    return batches
}

private func makeBatch(
    _ documents: [EncodedDocument],
    padTokenID: Int
) throws -> RerankerBatch {
    let maxLength = documents.map(\.input.tokenIds.count).max() ?? 0
    guard maxLength > 0 else { throw RerankerError.emptyPrompt }

    let needsTokenTypes = documents.contains { $0.input.tokenTypeIds != nil }
    var inputIDs = [Int]()
    var attentionMask = [Int32]()
    var tokenTypeIDs = [Int]()
    inputIDs.reserveCapacity(documents.count * maxLength)
    attentionMask.reserveCapacity(documents.count * maxLength)
    tokenTypeIDs.reserveCapacity(documents.count * maxLength)

    for document in documents {
        let input = document.input
        let paddingCount = maxLength - input.tokenIds.count
        inputIDs += input.tokenIds
        inputIDs += Array(repeating: padTokenID, count: paddingCount)
        attentionMask += Array(repeating: 1, count: input.tokenIds.count)
        attentionMask += Array(repeating: 0, count: paddingCount)

        if needsTokenTypes {
            let values = input.tokenTypeIds ?? Array(repeating: 0, count: input.tokenIds.count)
            guard values.count == input.tokenIds.count else {
                throw RerankerError.unsupportedModel(
                    "tokenTypeIds count \(values.count) does not match tokenIds count \(input.tokenIds.count)."
                )
            }
            tokenTypeIDs += values
            tokenTypeIDs += Array(repeating: 0, count: paddingCount)
        }
    }

    let shape = [documents.count, maxLength]
    return RerankerBatch(
        inputIDs: MLXArray(inputIDs).reshaped(shape),
        attentionMask: MLXArray(attentionMask).reshaped(shape),
        tokenTypeIDs: needsTokenTypes ? MLXArray(tokenTypeIDs).reshaped(shape) : nil)
}

private func selectScores(
    _ logits: MLXArray,
    batchSize: Int,
    policy: EncoderRerankerScorePolicy
) throws -> [Double] {
    guard logits.ndim == 2, logits.dim(0) == batchSize else {
        throw RerankerError.invalidLogitShape(logits.shape)
    }
    let labelCount = logits.dim(1)
    let values = logits.asArray(Float.self).map(Double.init)

    func value(row: Int, column: Int) throws -> Double {
        guard column >= 0, column < labelCount else {
            throw RerankerError.unsupportedModel(
                "Reranker class index \(column) is out of bounds for \(labelCount) labels.")
        }
        return values[row * labelCount + column]
    }

    return try (0 ..< batchSize).map { row in
        switch policy {
        case .singleLogit(let transform):
            guard labelCount == 1 else {
                throw RerankerError.unsupportedModel(
                    "singleLogit requires exactly one classifier output, but received \(labelCount)."
                )
            }
            return transform(try value(row: row, column: 0))

        case .classLogit(let index, let transform):
            return transform(try value(row: row, column: index))

        case .binaryMargin(let positive, let negative, let transform):
            return transform(
                try value(row: row, column: positive) - value(row: row, column: negative))

        case .softmaxProbability(let classIndex):
            let selected = try value(row: row, column: classIndex)
            let rowValues = (0 ..< labelCount).map { values[row * labelCount + $0] }
            guard let maximum = rowValues.max(), maximum.isFinite else {
                throw RerankerError.nonFiniteScore(index: row, score: rowValues.max() ?? .nan)
            }
            let exponentials = rowValues.map { Foundation.exp($0 - maximum) }
            let denominator = exponentials.reduce(0, +)
            guard denominator.isFinite, denominator > 0 else {
                throw RerankerError.nonFiniteScore(index: row, score: denominator)
            }
            return Foundation.exp(selected - maximum) / denominator
        }
    }
}

private func minimum(_ lhs: Int?, _ rhs: Int?) -> Int? {
    switch (lhs, rhs) {
    case (.some(let lhs), .some(let rhs)): min(lhs, rhs)
    case (.some(let value), .none), (.none, .some(let value)): value
    case (.none, .none): nil
    }
}
