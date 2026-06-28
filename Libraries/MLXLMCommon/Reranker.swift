// Copyright © 2026 Apple Inc.

import Foundation
import MLX

/// Tokenized input for a reranker model.
///
/// Pairwise rerankers use one ``RerankerInput`` per query-document pair.
/// Listwise rerankers use one ``RerankerInput`` for the whole document list.
public struct RerankerInput: Sendable {
    /// Token IDs passed to the model.
    public var tokenIds: [Int]

    /// Optional segment IDs for BERT-style sentence-pair models.
    public var tokenTypeIds: [Int]?

    /// Optional model-specific marker IDs used to find pooled query and document states.
    public var markerTokenIds: RerankerMarkerTokenIds?

    public init(
        tokenIds: [Int], tokenTypeIds: [Int]? = nil,
        markerTokenIds: RerankerMarkerTokenIds? = nil
    ) {
        self.tokenIds = tokenIds
        self.tokenTypeIds = tokenTypeIds
        self.markerTokenIds = markerTokenIds
    }
}

/// Token IDs that mark query and document representations in listwise reranker prompts.
///
/// Some listwise rerankers, such as Jina reranker v3, score documents by reading hidden
/// states at special query/document marker tokens. The input processor resolves those
/// marker IDs through the tokenizer and stores them here so the model does not depend on
/// hard-coded tokenizer constants.
public struct RerankerMarkerTokenIds: Sendable {
    public var query: Int
    public var document: Int

    public init(query: Int, document: Int) {
        self.query = query
        self.document = document
    }
}

/// Encodes a query-document pair for a concrete reranker family.
///
/// Implement this protocol for pairwise rerankers that score each document independently,
/// including causal-LM yes/no rerankers and encoder sequence-classification rerankers.
public protocol RerankerInputProcessor: Sendable {
    func encode(
        query: String,
        document: String,
        tokenizer: any Tokenizer,
        maxInputTokens: Int?
    ) throws -> RerankerInput
}

/// Encodes one query with a batch of documents for listwise reranker models.
///
/// Implement this protocol for models whose prompt contains the entire candidate set and
/// whose forward pass returns one score per document.
public protocol ListwiseRerankerInputProcessor: Sendable {
    func encode(
        query: String,
        documents: [String],
        tokenizer: any Tokenizer,
        maxInputTokens: Int?
    ) throws -> RerankerInput
}

/// Language models that score a full document list from one reranker prompt.
public protocol ListwiseRerankerModel: LanguageModel {
    func score(input: RerankerInput, documentCount: Int) throws -> [Double]
}

/// Applies a final scalar transform to a reranker logit or logit margin.
///
/// Use ``identity`` when the model already returns calibrated scores, probabilities, or
/// cosine similarities. Use ``sigmoid`` when the raw value is a binary classifier logit or a
/// true-minus-false logit margin.
public enum RerankerScoreTransform: Sendable {
    /// Return the raw score.
    case identity

    /// Return `sigmoid(score)`.
    case sigmoid

    public func callAsFunction(_ score: Double) -> Double {
        switch self {
        case .identity:
            return score
        case .sigmoid:
            if score >= 0 {
                return 1 / (1 + Foundation.exp(-score))
            } else {
                let exponent = Foundation.exp(score)
                return exponent / (1 + exponent)
            }
        }
    }
}

/// Controls how reranker results are returned.
///
/// The `process` APIs preserve input order by default because many applications want a
/// score vector aligned with the original candidates. The `rerank` APIs use
/// ``sortedByScore`` because it is intended for top-k retrieval workflows.
public enum RerankResultOrdering: Sendable {
    /// Preserve the caller's document order.
    case original

    /// Sort by descending score, using original index as a stable tie-breaker.
    case sortedByScore
}

/// Raw classifier-token logits used by causal-LM rerankers.
///
/// These values are exposed for diagnostics and calibration. The final ``RerankResult/score``
/// is computed from ``difference`` after applying the configured ``RerankerScoreTransform``.
public struct RerankerLogits: Sendable {
    public let trueLogit: Double
    public let falseLogit: Double

    public init(trueLogit: Double, falseLogit: Double) {
        self.trueLogit = trueLogit
        self.falseLogit = falseLogit
    }

    public var difference: Double {
        trueLogit - falseLogit
    }
}

/// A scored query-document pair.
public struct RerankResult: Sendable {
    /// The document's position in the caller-provided `documents` array.
    public let index: Int

    /// The original document text.
    public let document: String

    /// The final score after applying the configured score transform.
    public let score: Double

    /// The model score before applying the configured score transform.
    public let rawScore: Double?

    /// Raw causal-LM classifier-token logits, when available.
    public let logits: RerankerLogits?

    public init(
        index: Int, document: String, score: Double, rawScore: Double? = nil,
        logits: RerankerLogits? = nil
    ) {
        self.index = index
        self.document = document
        self.score = score
        self.rawScore = rawScore
        self.logits = logits
    }
}

/// A high-level reranking request.
///
/// Use this with `process` when you want a single structured API that returns scores for
/// a query and a document list. By default, results preserve input order:
///
/// ```swift
/// let results = try await container.process(
///     RerankRequest(query: "swift arrays", documents: candidates),
///     configuration: CausalLMRerankerConfiguration.qwen3())
/// let scores = results.map(\.score)
/// ```
///
/// Use ``RerankResultOrdering/sortedByScore`` or the convenience `rerank` methods when you
/// want the highest-scoring documents first.
public struct RerankRequest: Sendable {
    /// The search query or user intent to compare against each document.
    public var query: String

    /// Candidate documents to score.
    public var documents: [String]

    /// Controls whether the returned results preserve input order or are sorted by score.
    public var resultOrdering: RerankResultOrdering

    public init(
        query: String,
        documents: [String],
        resultOrdering: RerankResultOrdering = .original
    ) {
        self.query = query
        self.documents = documents
        self.resultOrdering = resultOrdering
    }
}

/// Errors thrown while preparing or scoring reranker inputs.
public enum RerankerError: LocalizedError, Sendable {
    case emptyPrompt
    case missingClassifierToken(String)
    case classifierTokenIsNotSingleToken(String, [Int])
    case missingSpecialToken(String)
    case truncatedRequiredToken(String)
    case tokenLimitTooSmall(maxInputTokens: Int, requiredTemplateTokens: Int)
    case unsupportedModel(String)

    public var errorDescription: String? {
        switch self {
        case .emptyPrompt:
            "Reranker prompt is empty."
        case .missingClassifierToken(let token):
            "Unable to resolve classifier token '\(token)'."
        case .classifierTokenIsNotSingleToken(let token, let tokenIds):
            "Classifier token '\(token)' must encode to exactly one token, but encoded to \(tokenIds)."
        case .missingSpecialToken(let token):
            "Unable to resolve required special token '\(token)'."
        case .truncatedRequiredToken(let token):
            "Reranker prompt truncation removed required token '\(token)'."
        case .tokenLimitTooSmall(let maxInputTokens, let requiredTemplateTokens):
            "maxInputTokens (\(maxInputTokens)) is too small for the reranker template (\(requiredTemplateTokens) tokens)."
        case .unsupportedModel(let message):
            message
        }
    }
}

/// Qwen3 causal-LM reranker input processing.
///
/// This builds the instruction/query/document prompt used by yes/no causal-LM rerankers.
/// The scoring path reads the next-token logits for the configured positive and negative
/// classifier tokens and ranks by their margin.
public struct Qwen3RerankerInputProcessor: RerankerInputProcessor {
    public var instruction: String

    public init(
        instruction: String =
            "Given a web search query, retrieve relevant passages that answer the query"
    ) {
        self.instruction = instruction
    }

    public func encode(
        query: String,
        document: String,
        tokenizer: any Tokenizer,
        maxInputTokens: Int?
    ) throws -> RerankerInput {
        let prefixTokens = tokenizer.encode(text: Self.prefix, addSpecialTokens: false)
        let bodyTokens = tokenizer.encode(
            text:
                """
                <Instruct>: \(instruction)
                <Query>: \(query)
                <Document>: \(document)
                """,
            addSpecialTokens: false)
        let suffixTokens = tokenizer.encode(text: Self.suffix, addSpecialTokens: false)

        let templateTokenCount = prefixTokens.count + suffixTokens.count
        let body: ArraySlice<Int>
        if let maxInputTokens {
            let bodyBudget = maxInputTokens - templateTokenCount
            guard bodyBudget > 0 else {
                throw RerankerError.tokenLimitTooSmall(
                    maxInputTokens: maxInputTokens,
                    requiredTemplateTokens: templateTokenCount)
            }
            body = bodyTokens.prefix(bodyBudget)
        } else {
            body = bodyTokens[...]
        }

        return RerankerInput(tokenIds: prefixTokens + body + suffixTokens)
    }

    private static let prefix =
        """
        <|im_start|>system
        Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>
        <|im_start|>user

        """

    private static let suffix =
        """
        <|im_end|>
        <|im_start|>assistant
        <think>

        </think>

        """
}

/// XLM-RoBERTa sentence-pair processing used by BGE v2 rerankers.
///
/// The encoded sequence has the standard XLM-RoBERTa pair form:
/// `<s> query </s></s> document </s>`.
public struct XLMRobertaRerankerInputProcessor: RerankerInputProcessor {
    public var bosToken: String
    public var eosToken: String

    public init(bosToken: String = "<s>", eosToken: String = "</s>") {
        self.bosToken = bosToken
        self.eosToken = eosToken
    }

    public func encode(
        query: String,
        document: String,
        tokenizer: any Tokenizer,
        maxInputTokens: Int?
    ) throws -> RerankerInput {
        let bosTokenId = try resolveSpecialToken(bosToken, tokenizer: tokenizer)
        let eosTokenId = try resolveSpecialToken(eosToken, tokenizer: tokenizer)

        let queryTokens = tokenizer.encode(text: query, addSpecialTokens: false)
        let documentTokens = tokenizer.encode(text: document, addSpecialTokens: false)
        let truncated = try truncatePair(
            first: queryTokens,
            second: documentTokens,
            maxInputTokens: maxInputTokens,
            specialTokenCount: 4)

        return RerankerInput(
            tokenIds: [bosTokenId] + truncated.first + [eosTokenId, eosTokenId]
                + truncated.second + [eosTokenId])
    }
}

/// BERT sentence-pair processing for sequence-classification rerankers.
///
/// The encoded sequence has the standard BERT pair form:
/// `[CLS] query [SEP] document [SEP]`, with token type IDs set to 0 for the query
/// segment and 1 for the document segment.
public struct BERTRerankerInputProcessor: RerankerInputProcessor {
    public var clsToken: String
    public var sepToken: String

    public init(clsToken: String = "[CLS]", sepToken: String = "[SEP]") {
        self.clsToken = clsToken
        self.sepToken = sepToken
    }

    public func encode(
        query: String,
        document: String,
        tokenizer: any Tokenizer,
        maxInputTokens: Int?
    ) throws -> RerankerInput {
        let clsTokenId = try resolveSpecialToken(clsToken, tokenizer: tokenizer)
        let sepTokenId = try resolveSpecialToken(sepToken, tokenizer: tokenizer)

        let queryTokens = tokenizer.encode(text: query, addSpecialTokens: false)
        let documentTokens = tokenizer.encode(text: document, addSpecialTokens: false)
        let truncated = try truncatePair(
            first: queryTokens,
            second: documentTokens,
            maxInputTokens: maxInputTokens,
            specialTokenCount: 3)

        let tokenIds =
            [clsTokenId] + truncated.first + [sepTokenId] + truncated.second
            + [sepTokenId]
        let querySegmentLength = truncated.first.count + 2
        let tokenTypeIds =
            Array(repeating: 0, count: querySegmentLength)
            + Array(repeating: 1, count: truncated.second.count + 1)

        return RerankerInput(tokenIds: tokenIds, tokenTypeIds: tokenTypeIds)
    }
}

/// Jina reranker v3 listwise prompt processing.
///
/// This processor builds a single prompt containing all candidate passages. It appends
/// the configured document marker token to each passage and the query marker token to the
/// query block. The model then scores documents from the hidden states at those markers.
public struct JinaRerankerInputProcessor: ListwiseRerankerInputProcessor {
    public var instruction: String?
    public var queryEmbedToken: String
    public var documentEmbedToken: String

    public init(
        instruction: String? = nil,
        queryEmbedToken: String = "<|rerank_token|>",
        documentEmbedToken: String = "<|embed_token|>"
    ) {
        self.instruction = instruction
        self.queryEmbedToken = queryEmbedToken
        self.documentEmbedToken = documentEmbedToken
    }

    public func encode(
        query: String,
        documents: [String],
        tokenizer: any Tokenizer,
        maxInputTokens: Int?
    ) throws -> RerankerInput {
        let markerTokenIds = RerankerMarkerTokenIds(
            query: try resolveSpecialToken(queryEmbedToken, tokenizer: tokenizer),
            document: try resolveSpecialToken(documentEmbedToken, tokenizer: tokenizer))
        let specialTokens = [queryEmbedToken, documentEmbedToken]
        let sanitizedQuery = sanitize(query, removing: specialTokens)
        let sanitizedDocuments = documents.map { sanitize($0, removing: specialTokens) }

        let prompt = renderPrompt(query: sanitizedQuery, documents: sanitizedDocuments)
        var tokenIds = tokenizer.encode(text: prompt, addSpecialTokens: false)
        if let maxInputTokens, tokenIds.count > maxInputTokens {
            tokenIds = try renderTruncatedPromptTokens(
                query: sanitizedQuery,
                documents: sanitizedDocuments,
                tokenizer: tokenizer,
                maxInputTokens: maxInputTokens,
                markerTokenIds: markerTokenIds)
        }

        guard tokenIds.contains(markerTokenIds.query) else {
            throw RerankerError.truncatedRequiredToken(queryEmbedToken)
        }
        guard tokenIds.filter({ $0 == markerTokenIds.document }).count == documents.count else {
            throw RerankerError.truncatedRequiredToken(documentEmbedToken)
        }

        return RerankerInput(tokenIds: tokenIds, markerTokenIds: markerTokenIds)
    }

    private func renderPrompt(query: String, documents: [String]) -> String {
        var prompt = renderPromptPrefix(query: query, documentCount: documents.count)

        prompt += documents.enumerated().map { index, document in
            renderDocumentPrefix(index: index) + document + documentEmbedToken
                + renderDocumentSuffix()
        }.joined(separator: "\n")

        prompt += renderQueryPrefix(query: query) + queryEmbedToken + renderQuerySuffix()

        return prompt
    }

    private func renderTruncatedPromptTokens(
        query: String,
        documents: [String],
        tokenizer: any Tokenizer,
        maxInputTokens: Int,
        markerTokenIds: RerankerMarkerTokenIds
    ) throws -> [Int] {
        let promptPrefixTokens = tokenizer.encode(
            text: renderPromptPrefix(query: query, documentCount: documents.count),
            addSpecialTokens: false)
        let queryPrefixTokens = tokenizer.encode(
            text: renderQueryPrefix(query: query),
            addSpecialTokens: false)
        let querySuffixTokens = tokenizer.encode(text: renderQuerySuffix(), addSpecialTokens: false)
        let documentSeparatorTokens = tokenizer.encode(text: "\n", addSpecialTokens: false)

        let documentParts = documents.enumerated().map { index, document in
            JinaDocumentPromptTokens(
                prefix: tokenizer.encode(
                    text: renderDocumentPrefix(index: index), addSpecialTokens: false),
                body: tokenizer.encode(text: document, addSpecialTokens: false),
                suffix: [markerTokenIds.document]
                    + tokenizer.encode(text: renderDocumentSuffix(), addSpecialTokens: false))
        }

        let separatorCount = max(0, documents.count - 1) * documentSeparatorTokens.count
        let requiredTemplateTokens =
            promptPrefixTokens.count
            + queryPrefixTokens.count
            + 1
            + querySuffixTokens.count
            + separatorCount
            + documentParts.reduce(0) { $0 + $1.prefix.count + $1.suffix.count }

        guard maxInputTokens >= requiredTemplateTokens else {
            throw RerankerError.tokenLimitTooSmall(
                maxInputTokens: maxInputTokens,
                requiredTemplateTokens: requiredTemplateTokens)
        }

        let documentTokenBudget = maxInputTokens - requiredTemplateTokens
        let documentTokenCounts = truncateTokenCounts(
            documentParts.map(\.body.count), toTotal: documentTokenBudget)

        var tokenIds = promptPrefixTokens
        for index in documentParts.indices {
            if index > 0 {
                tokenIds += documentSeparatorTokens
            }
            tokenIds += documentParts[index].prefix
            tokenIds += documentParts[index].body.prefix(documentTokenCounts[index])
            tokenIds += documentParts[index].suffix
        }
        tokenIds += queryPrefixTokens
        tokenIds.append(markerTokenIds.query)
        tokenIds += querySuffixTokens
        return tokenIds
    }

    private func renderPromptPrefix(query: String, documentCount: Int) -> String {
        var prompt =
            """
            <|im_start|>system
            You are a search relevance expert who can determine a ranking of the passages based on how relevant they are to the query. If the query is a question, how relevant a passage is depends on how well it answers the question. If not, try to analyze the intent of the query and assess how well each passage satisfies the intent. If an instruction is provided, you should follow the instruction when determining the ranking.<|im_end|>
            <|im_start|>user
            I will provide you with \(documentCount) passages, each indicated by a numerical identifier. Rank the passages based on their relevance to query: \(query)

            """

        if let instruction {
            prompt +=
                """
                <instruct>
                \(instruction)
                </instruct>

                """
        }

        return prompt
    }

    private func renderDocumentPrefix(index: Int) -> String {
        "<passage id=\"\(index)\">\n"
    }

    private func renderDocumentSuffix() -> String {
        "\n</passage>"
    }

    private func renderQueryPrefix(query: String) -> String {
        "\n\n<query>\n\(query)"
    }

    private func renderQuerySuffix() -> String {
        "</query><|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    }

    private func sanitize(_ text: String, removing specialTokens: [String]) -> String {
        specialTokens.reduce(text) { result, token in
            result.replacingOccurrences(of: token, with: "")
        }
    }
}

private struct JinaDocumentPromptTokens {
    var prefix: [Int]
    var body: [Int]
    var suffix: [Int]
}

private func truncateTokenCounts(_ counts: [Int], toTotal totalLimit: Int) -> [Int] {
    guard totalLimit >= 0 else { return Array(repeating: 0, count: counts.count) }

    var result = counts
    var total = result.reduce(0, +)
    while total > totalLimit {
        guard let maxCount = result.max(), maxCount > 0 else {
            return result
        }

        let maxIndices = result.indices.filter { result[$0] == maxCount }
        let nextMax = result.filter { $0 < maxCount }.max() ?? 0
        let overflow = total - totalLimit
        let removableToNext = (maxCount - nextMax) * maxIndices.count

        if removableToNext <= overflow {
            for index in maxIndices {
                result[index] = nextMax
            }
            total -= removableToNext
        } else {
            let removeEach = overflow / maxIndices.count
            let remainder = overflow % maxIndices.count
            for (offset, index) in maxIndices.enumerated() {
                result[index] -= removeEach + (offset < remainder ? 1 : 0)
            }
            total = totalLimit
        }
    }

    return result
}

/// Configuration for causal-LM rerankers.
///
/// Causal-LM rerankers score a query-document pair by prompting the model to answer with
/// a positive or negative classifier token. For best reproducibility, pass explicit
/// `trueTokenId` and `falseTokenId` from the model card or tokenizer when they are known.
public struct CausalLMRerankerConfiguration: Sendable {
    public var inputProcessor: any RerankerInputProcessor
    public var trueToken: String
    public var falseToken: String
    public var trueTokenId: Int?
    public var falseTokenId: Int?
    public var scoreTransform: RerankerScoreTransform
    public var maxInputTokens: Int?
    public var prefillStepSize: Int

    public init(
        inputProcessor: any RerankerInputProcessor = Qwen3RerankerInputProcessor(),
        trueToken: String = "yes",
        falseToken: String = "no",
        trueTokenId: Int? = nil,
        falseTokenId: Int? = nil,
        scoreTransform: RerankerScoreTransform = .sigmoid,
        maxInputTokens: Int? = 8_192,
        prefillStepSize: Int = 512
    ) {
        self.inputProcessor = inputProcessor
        self.trueToken = trueToken
        self.falseToken = falseToken
        self.trueTokenId = trueTokenId
        self.falseTokenId = falseTokenId
        self.scoreTransform = scoreTransform
        self.maxInputTokens = maxInputTokens
        self.prefillStepSize = prefillStepSize
    }

    /// Default configuration for Qwen3 causal-LM rerankers.
    public static func qwen3(
        instruction: String =
            "Given a web search query, retrieve relevant passages that answer the query",
        trueTokenId: Int? = nil,
        falseTokenId: Int? = nil
    ) -> Self {
        .init(
            inputProcessor: Qwen3RerankerInputProcessor(instruction: instruction),
            trueTokenId: trueTokenId,
            falseTokenId: falseTokenId)
    }
}

/// Configuration for listwise rerankers.
///
/// Listwise rerankers score the whole candidate set in one forward pass. This is useful
/// for models that compare documents against each other inside a single prompt, or models
/// that expose one pooled representation per candidate.
public struct ListwiseRerankerConfiguration: Sendable {
    public var inputProcessor: any ListwiseRerankerInputProcessor
    public var scoreTransform: RerankerScoreTransform
    public var maxInputTokens: Int?

    public init(
        inputProcessor: any ListwiseRerankerInputProcessor = JinaRerankerInputProcessor(),
        scoreTransform: RerankerScoreTransform = .identity,
        maxInputTokens: Int? = 131_072
    ) {
        self.inputProcessor = inputProcessor
        self.scoreTransform = scoreTransform
        self.maxInputTokens = maxInputTokens
    }

    /// Default configuration for `jinaai/jina-reranker-v3-mlx`.
    public static func jinaRerankerV3(instruction: String? = nil) -> Self {
        .init(
            inputProcessor: JinaRerankerInputProcessor(instruction: instruction),
            maxInputTokens: 131_072)
    }
}

extension ModelContainer {
    /// Process a reranking request with a causal-LM reranker.
    ///
    /// Unlike the `rerank` convenience methods, this preserves document order by
    /// default, matching score-vector APIs such as Python `model.process(...)`.
    ///
    /// Use this method when your application needs one score per original candidate:
    ///
    /// ```swift
    /// let results = try await container.process(
    ///     RerankRequest(query: "best swift tokenizer", documents: candidates),
    ///     configuration: .qwen3(trueTokenId: yesTokenId, falseTokenId: noTokenId))
    /// let scores = results.map(\.score)
    /// ```
    ///
    /// - Parameters:
    ///   - request: Query, candidate documents, and result ordering.
    ///   - configuration: Causal-LM reranker prompt and classifier-token configuration.
    /// - Returns: One scored result per input document.
    public func process(
        _ request: RerankRequest,
        configuration: CausalLMRerankerConfiguration = .qwen3()
    ) async throws -> [RerankResult] {
        guard !request.documents.isEmpty else { return [] }

        return try await perform(values: (request, configuration)) { context, values in
            let (request, configuration) = values
            let scorer = CausalLMReranker(
                tokenizer: context.tokenizer, configuration: configuration)

            let results = try request.documents.enumerated().map { index, document in
                try scorer.score(
                    query: request.query, document: document, index: index, model: context.model)
            }

            return orderedResults(results, by: request.resultOrdering)
        }
    }

    /// Score a single query-document pair with a causal-LM reranker.
    ///
    /// This is a convenience wrapper for callers that only need one relevance score.
    public func score(
        query: String,
        document: String,
        configuration: CausalLMRerankerConfiguration = .qwen3()
    ) async throws -> RerankResult {
        let results = try await rerank(
            query: query, documents: [document], configuration: configuration)
        guard let result = results.first else {
            throw RerankerError.emptyPrompt
        }
        return result
    }

    /// Score and sort documents by descending relevance with a causal-LM reranker.
    ///
    /// This is the retrieval-oriented convenience API. It returns the same result type as
    /// `process` but requests ``RerankResultOrdering/sortedByScore``. Pass `topK` to return
    /// only the highest-scoring results. A `nil` value returns all scored documents; a
    /// non-positive value returns an empty array.
    ///
    /// - Parameters:
    ///   - query: Search query or user intent to compare against each document.
    ///   - documents: Candidate documents to score.
    ///   - topK: Optional maximum number of sorted results to return.
    ///   - configuration: Causal-LM reranker prompt and classifier-token configuration.
    /// - Returns: Scored results sorted by descending relevance, limited by `topK`.
    public func rerank(
        query: String,
        documents: [String],
        topK: Int? = nil,
        configuration: CausalLMRerankerConfiguration = .qwen3()
    ) async throws -> [RerankResult] {
        let results = try await process(
            .init(query: query, documents: documents, resultOrdering: .sortedByScore),
            configuration: configuration)
        return topResults(results, topK: topK)
    }

    /// Process a reranking request with a listwise reranker.
    ///
    /// The default ordering preserves the caller's document order.
    ///
    /// Use this method for models that score all documents from one prompt, such as Jina
    /// reranker v3:
    ///
    /// ```swift
    /// let results = try await container.process(
    ///     RerankRequest(query: query, documents: candidates),
    ///     configuration: .jinaRerankerV3())
    /// ```
    ///
    /// - Parameters:
    ///   - request: Query, candidate documents, and result ordering.
    ///   - configuration: Listwise prompt and score-transform configuration.
    /// - Returns: One scored result per input document.
    public func process(
        _ request: RerankRequest,
        configuration: ListwiseRerankerConfiguration
    ) async throws -> [RerankResult] {
        guard !request.documents.isEmpty else { return [] }

        return try await perform(values: (request, configuration)) { context, values in
            let (request, configuration) = values
            guard let model = context.model as? any ListwiseRerankerModel else {
                throw RerankerError.unsupportedModel(
                    "\(type(of: context.model)) does not expose listwise reranker scores.")
            }

            let input = try configuration.inputProcessor.encode(
                query: request.query,
                documents: request.documents,
                tokenizer: context.tokenizer,
                maxInputTokens: configuration.maxInputTokens)

            let scores = try model.score(input: input, documentCount: request.documents.count)
            guard scores.count == request.documents.count else {
                throw RerankerError.unsupportedModel(
                    "\(type(of: model)) returned \(scores.count) scores for \(request.documents.count) documents."
                )
            }

            let results = zip(request.documents.indices, request.documents).map { index, document in
                let rawScore = scores[index]
                return RerankResult(
                    index: index,
                    document: document,
                    score: configuration.scoreTransform(rawScore),
                    rawScore: rawScore
                )
            }

            return orderedResults(results, by: request.resultOrdering)
        }
    }

    /// Score and sort documents by descending relevance with a listwise reranker.
    ///
    /// This is the top-k retrieval convenience API. Use `process`
    /// if you need scores in the same order as the input documents. Pass `topK` to return
    /// only the highest-scoring results. A `nil` value returns all scored documents; a
    /// non-positive value returns an empty array.
    ///
    /// - Parameters:
    ///   - query: Search query or user intent to compare against each document.
    ///   - documents: Candidate documents to score.
    ///   - topK: Optional maximum number of sorted results to return.
    ///   - configuration: Listwise prompt and score-transform configuration.
    /// - Returns: Scored results sorted by descending relevance, limited by `topK`.
    public func rerank(
        query: String,
        documents: [String],
        topK: Int? = nil,
        configuration: ListwiseRerankerConfiguration
    ) async throws -> [RerankResult] {
        let results = try await process(
            .init(query: query, documents: documents, resultOrdering: .sortedByScore),
            configuration: configuration)
        return topResults(results, topK: topK)
    }
}

private struct CausalLMReranker {
    let tokenizer: any Tokenizer
    let configuration: CausalLMRerankerConfiguration

    func score(
        query: String, document: String, index: Int, model: any LanguageModel
    ) throws -> RerankResult {
        let classifierTokens = try resolveClassifierTokens()
        let input = try configuration.inputProcessor.encode(
            query: query,
            document: document,
            tokenizer: tokenizer,
            maxInputTokens: configuration.maxInputTokens)

        guard !input.tokenIds.isEmpty else {
            throw RerankerError.emptyPrompt
        }

        let lmInput = LMInput(tokens: MLXArray(input.tokenIds))
        let cache = model.newCache(parameters: nil)
        let output = try nextTokenLogits(input: lmInput, model: model, cache: cache)
        let logits = output[0..., -1, 0...]

        let trueLogit = scalarLogit(logits, tokenId: classifierTokens.trueTokenId)
        let falseLogit = scalarLogit(logits, tokenId: classifierTokens.falseTokenId)
        let rerankerLogits = RerankerLogits(trueLogit: trueLogit, falseLogit: falseLogit)

        return RerankResult(
            index: index,
            document: document,
            score: configuration.scoreTransform(rerankerLogits.difference),
            rawScore: rerankerLogits.difference,
            logits: rerankerLogits
        )
    }

    private func nextTokenLogits(
        input: LMInput, model: any LanguageModel, cache: [KVCache]
    ) throws -> MLXArray {
        switch try model.prepare(input, cache: cache, windowSize: configuration.prefillStepSize) {
        case .logits(let output):
            return output.logits
        case .tokens(let tokens):
            return withPreparedCache(cache, lengths: tokens.sequenceLengths) {
                model(tokens[text: .newAxis], cache: cache.isEmpty ? nil : cache, state: nil).logits
            }
        }
    }

    private func resolveClassifierTokens() throws -> (
        trueTokenId: Int, falseTokenId: Int
    ) {
        let trueTokenId = try resolveClassifierToken(
            configuration.trueToken, override: configuration.trueTokenId)
        let falseTokenId = try resolveClassifierToken(
            configuration.falseToken, override: configuration.falseTokenId)
        return (trueTokenId, falseTokenId)
    }

    private func resolveClassifierToken(_ token: String, override: Int?) throws -> Int {
        if let override {
            return override
        }
        if let tokenId = tokenizer.convertTokenToId(token) {
            return tokenId
        }

        let tokenIds = tokenizer.encode(text: token, addSpecialTokens: false)
        guard tokenIds.count == 1, let tokenId = tokenIds.first else {
            if tokenIds.isEmpty {
                throw RerankerError.missingClassifierToken(token)
            }
            throw RerankerError.classifierTokenIsNotSingleToken(token, tokenIds)
        }
        return tokenId
    }

    private func scalarLogit(_ logits: MLXArray, tokenId: Int) -> Double {
        if logits.ndim == 1 {
            return Double(logits[tokenId].item(Float.self))
        }
        return Double(logits[0, tokenId].item(Float.self))
    }
}

public func sortedByDescendingScore(_ results: [RerankResult]) -> [RerankResult] {
    results.sorted {
        if $0.score == $1.score {
            return $0.index < $1.index
        }
        return $0.score > $1.score
    }
}

public func orderedResults(
    _ results: [RerankResult],
    by ordering: RerankResultOrdering
) -> [RerankResult] {
    switch ordering {
    case .original:
        return results.sorted { $0.index < $1.index }
    case .sortedByScore:
        return sortedByDescendingScore(results)
    }
}

public func topResults(_ results: [RerankResult], topK: Int?) -> [RerankResult] {
    guard let topK else { return results }
    guard topK > 0 else { return [] }
    return Array(results.prefix(topK))
}

private func resolveSpecialToken(_ token: String, tokenizer: any Tokenizer) throws -> Int {
    if let tokenId = tokenizer.convertTokenToId(token) {
        return tokenId
    }
    let tokenIds = tokenizer.encode(text: token, addSpecialTokens: false)
    guard tokenIds.count == 1, let tokenId = tokenIds.first else {
        throw RerankerError.missingSpecialToken(token)
    }
    return tokenId
}

private func truncatePair(
    first: [Int],
    second: [Int],
    maxInputTokens: Int?,
    specialTokenCount: Int
) throws -> (first: [Int], second: [Int]) {
    guard let maxInputTokens else {
        return (first, second)
    }

    let tokenBudget = maxInputTokens - specialTokenCount
    guard tokenBudget > 0 else {
        throw RerankerError.tokenLimitTooSmall(
            maxInputTokens: maxInputTokens,
            requiredTemplateTokens: specialTokenCount)
    }

    var firstCount = first.count
    var secondCount = second.count
    var overflow = firstCount + secondCount - tokenBudget
    while overflow > 0 {
        if secondCount >= firstCount, secondCount > 0 {
            secondCount -= 1
        } else if firstCount > 0 {
            firstCount -= 1
        }
        overflow -= 1
    }

    return (Array(first.prefix(firstCount)), Array(second.prefix(secondCount)))
}
