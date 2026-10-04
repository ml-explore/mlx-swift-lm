// Copyright © 2026 Apple Inc.

import MLX

/// The prompt and block protocol of a Jina listwise reranker.
package struct JinaRerankerFamily: Sendable, Equatable {
    /// Reference blocking: documents are split into blocks that share one fused query.
    package struct Blocking: Sendable, Equatable {
        package var documentsPerBlock = 125
        /// `max_query_length - 64` in the reference `rerank.py`.
        package var maxQueryTokens = 1_984
        /// `max_doc_length - 1` in the reference `rerank.py`.
        package var maxDocumentTokens = 8_191
        /// A block closes once fewer than `max_doc_length` tokens remain.
        package var headroomTokens = 8_192
    }

    package let name: String
    /// Repeats the query marker after the header query and reads the last marker.
    package let marksQueryInHeader: Bool
    package let rankingInstruction: String?
    /// Control tokens removed from user text besides the query and document markers.
    package let reservedTokens: [String]
    package let maximumDocuments: Int?
    /// When nil, all documents share one prompt bounded by `maxBatchTokens`.
    package let blocking: Blocking?

    /// `jinaai/jina-reranker-v3`: one prompt of at most 64 documents.
    package static let v3 = Self(
        name: "Jina reranker v3", marksQueryInHeader: false, rankingInstruction: nil,
        reservedTokens: [], maximumDocuments: 64, blocking: nil)

    /// `jinaai/jina-reranker-v3.5`: dual query markers and weighted fusion across blocks.
    package static let v35 = Self(
        name: "Jina reranker v3.5", marksQueryInHeader: true,
        rankingInstruction:
            "Please provide the ranking of all passages based on their relevance to the search query, in descending order of relevance, with each label enclosed in square brackets (e.g., [2] > [1] > [3] > [0]).",
        reservedTokens: ["<|score_token|>"], maximumDocuments: nil, blocking: Blocking())
}

/// Projected marker states of one listwise prompt, `[documents, dimension]` and `[1, dimension]`.
package struct JinaRerankerEmbeddings {
    package var documents: MLXArray
    package var query: MLXArray

    package init(documents: MLXArray, query: MLXArray) {
        self.documents = documents
        self.query = query
    }
}

package protocol JinaRerankerEmbeddingModel: BaseLanguageModel {
    /// Projects the document and final query marker states, prefilling `stepSize` tokens at a time.
    func embeddings(input: RerankerInput, documentCount: Int, stepSize: Int) throws
        -> JinaRerankerEmbeddings
}

extension ModelContainer {
    /// Score a Jina listwise reranker in document order.
    ///
    /// Blocked families follow the reference block geometry for `maxInputTokens`, so scores do
    /// not depend on execution options; `maxBatchTokens` bounds each prefill step instead.
    package func listwiseRerankerScores(
        query: String,
        documents: [String],
        instruction: String?,
        maxInputTokens: Int,
        family: JinaRerankerFamily,
        options: RerankExecutionOptions
    ) async throws -> [Double] {
        guard !documents.isEmpty else { return [] }
        if let maximum = family.maximumDocuments, documents.count > maximum {
            throw RerankerError.tooManyDocuments(actual: documents.count, maximum: maximum)
        }

        return try await perform(
            values: (query, documents, instruction, maxInputTokens, options)
        ) { context, values in
            try Task.checkCancellation()
            let (query, documents, instruction, maxInputTokens, options) = values
            guard let model = context.model as? any JinaRerankerEmbeddingModel else {
                throw RerankerError.unsupportedModel(
                    "\(type(of: context.model)) does not expose listwise reranker embeddings.")
            }
            let processor = JinaRerankerInputProcessor(instruction: instruction, family: family)
            let promptLimit: Int
            let blocks: (query: String, documents: [[String]])
            if let blocking = family.blocking {
                promptLimit = maxInputTokens
                blocks = try blocking.blocks(
                    query: query, documents: documents, tokenizer: context.tokenizer,
                    blockTokens: maxInputTokens, truncation: options.truncation)
            } else {
                promptLimit = min(maxInputTokens, options.maxBatchTokens)
                blocks = (query, [documents])
            }
            let embeddings = try blocks.documents.map { block in
                try Task.checkCancellation()
                let input = try processor.encode(
                    query: blocks.query, documents: block, tokenizer: context.tokenizer,
                    maxInputTokens: promptLimit, truncation: options.truncation)
                return try model.embeddings(
                    input: input, documentCount: block.count,
                    stepSize: min(options.prefillStepSize, options.maxBatchTokens))
            }
            return jinaFusedScores(embeddings)
        }
    }
}

extension JinaRerankerFamily.Blocking {
    /// Truncates the query and documents, then groups documents as the reference `rerank.py`
    /// does for a `blockTokens` context.
    package func blocks(
        query: String, documents: [String], tokenizer: any Tokenizer,
        blockTokens: Int, truncation: RerankTruncationPolicy
    ) throws -> (query: String, documents: [[String]]) {
        func truncated(_ text: String, to maximum: Int) throws -> (text: String, count: Int) {
            let tokens = tokenizer.encode(text: text, addSpecialTokens: false)
            guard tokens.count >= maximum else { return (text, tokens.count) }
            if tokens.count > maximum, truncation == .error {
                throw RerankerError.inputTooLong(actual: tokens.count, maximum: maximum)
            }
            return (tokenizer.decode(tokenIds: Array(tokens.prefix(maximum))), maximum)
        }
        let (query, queryCount) = try truncated(query, to: maxQueryTokens)
        var blocks = [[String]]()
        var block = [String]()
        var tokenCount = queryCount
        for document in documents {
            try Task.checkCancellation()
            let (document, count) = try truncated(document, to: maxDocumentTokens)
            block.append(document)
            tokenCount += count
            if block.count >= documentsPerBlock || tokenCount >= blockTokens - headroomTokens {
                blocks.append(block)
                block = []
                tokenCount = queryCount
            }
        }
        if !block.isEmpty { blocks.append(block) }
        return (query, blocks)
    }
}

/// Cosine scores against the query, fused across blocks by each block's best score.
///
/// One block reduces to plain cosine similarity, as in Jina v3.
package func jinaFusedScores(_ blocks: [JinaRerankerEmbeddings]) -> [Double] {
    let documents = concatenated(blocks.map(\.documents), axis: 0)
    guard blocks.count > 1 else {
        return jinaCosineSimilarity(documents, blocks[0].query).asArray(Float.self)
            .map(Double.init)
    }
    let weights = stacked(
        blocks.map { ((jinaCosineSimilarity($0.documents, $0.query) + 1) / 2).max() })
    let queries = concatenated(blocks.map(\.query), axis: 0)
    let query =
        MLX.sum(queries * weights.expandedDimensions(axis: -1), axis: 0, keepDims: true)
        / weights.sum()
    return jinaCosineSimilarity(documents, query).asArray(Float.self).map(Double.init)
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
