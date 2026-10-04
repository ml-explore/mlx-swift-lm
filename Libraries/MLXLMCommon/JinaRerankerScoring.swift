// Copyright © 2026 Apple Inc.

import MLX

package enum JinaRerankerVersion: Sendable, Equatable {
    case v3
    case v35
}

package protocol JinaRerankerEmbeddingModel: ListwiseRerankerModel {
    func embeddings(input: RerankerInput, documentCount: Int) throws -> JinaRerankerEmbeddings
}

package typealias JinaRerankerEmbeddings = (documents: MLXArray, query: MLXArray)

/// Jina v3.5 fuses query embeddings using each block's best cosine score.
package func jinaV35FusedScores(_ blocks: [JinaRerankerEmbeddings]) throws -> [Double] {
    guard !blocks.isEmpty else { return [] }
    let weights = stacked(
        blocks.map {
            ((jinaCosineSimilarity($0.documents, $0.query) + 1) / 2).max()
        })
    let weightSum = weights.sum().item(Float.self)
    guard weightSum.isFinite, weightSum > 0 else {
        throw RerankerError.nonFiniteScore(index: 0, score: .nan)
    }
    let queries = concatenated(blocks.map(\.query), axis: 0)
    let fusedQuery =
        MLX.sum(queries * weights.expandedDimensions(axis: -1), axis: 0, keepDims: true) / weightSum
    return jinaCosineSimilarity(concatenated(blocks.map(\.documents), axis: 0), fusedQuery)
        .asArray(Float.self).map(Double.init)
}

package func jinaCosineSimilarity(_ documents: MLXArray, _ query: MLXArray) -> MLXArray {
    let documents = documents.asType(.float32)
    let query = query.asType(.float32)
    let numerator = MLX.sum(documents * query, axis: -1)
    let denominator =
        MLX.sqrt(MLX.sum(documents * documents, axis: -1))
        * MLX.sqrt(MLX.sum(query * query, axis: -1))
    return MLX.clip(numerator / MLX.maximum(denominator, MLXArray(1e-12)), min: -1, max: 1)
}

package func jinaV35RerankerScores(
    context: ModelContext, query: String, documents: [String], instruction: String?,
    maxInputTokens: Int, options: RerankExecutionOptions
) throws -> [Double] {
    guard let model = context.model as? any JinaRerankerEmbeddingModel else {
        throw RerankerError.unsupportedModel(
            "Jina v3.5 requires projected query and document embeddings.")
    }
    func truncatedText(_ text: String, maximum: Int) throws -> (text: String, count: Int) {
        let tokens = context.tokenizer.encode(text: text, addSpecialTokens: false)
        guard tokens.count >= maximum else { return (text, tokens.count) }
        if tokens.count > maximum, options.truncation == .error {
            throw RerankerError.inputTooLong(actual: tokens.count, maximum: maximum)
        }
        let text = context.tokenizer.decode(tokenIds: Array(tokens.prefix(maximum)))
        return (text, context.tokenizer.encode(text: text, addSpecialTokens: false).count)
    }
    let limit = min(maxInputTokens, options.maxBatchTokens)
    let (query, queryLength) = try truncatedText(query, maximum: 2_048 - 64)
    var blocks = [[String]]()
    var block = [String]()
    var tokenCount = queryLength
    for document in documents {
        try Task.checkCancellation()
        let (document, length) = try truncatedText(document, maximum: 8_192 - 1)
        block.append(document)
        tokenCount += length
        if block.count >= 125 || tokenCount >= limit - 8_192 {
            blocks.append(block)
            block = []
            tokenCount = queryLength
        }
    }
    if !block.isEmpty { blocks.append(block) }
    let processor = JinaRerankerInputProcessor(instruction: instruction, version: .v35)
    let embeddings = try blocks.map { block in
        try Task.checkCancellation()
        let input = try processor.encode(
            query: query, documents: block,
            tokenizer: context.tokenizer, maxInputTokens: limit,
            truncation: options.truncation)
        return try model.embeddings(input: input, documentCount: block.count)
    }
    return try jinaV35FusedScores(embeddings)
}
