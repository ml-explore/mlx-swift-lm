// Copyright © 2026 Apple Inc.

import Foundation
import MLX

/// The semantic meaning and numeric range of scores returned by a reranker.
public enum RerankScoreKind: Sendable, Equatable {
    /// A model-specific relevance score normalized to the closed range `0...1`.
    ///
    /// This value is not necessarily a calibrated probability. Its scale can differ across
    /// models, revisions, quantizations, prompts, and instructions.
    case normalizedRelevance

    /// Cosine similarity in the closed range `-1...1`.
    case cosineSimilarity

    /// An unbounded model logit.
    case logit

    package var validRange: ClosedRange<Double>? {
        switch self {
        case .normalizedRelevance:
            0 ... 1
        case .cosineSimilarity:
            -1 ... 1
        case .logit:
            nil
        }
    }
}

/// Controls how inputs that exceed a model's context limit are handled.
public enum RerankTruncationPolicy: Sendable {
    /// Truncate model input content while preserving required structural prompt tokens.
    case truncate

    /// Reject an overlong input instead of truncating it.
    case error
}

/// Resource limits and execution behavior for a reranking operation.
public struct RerankExecutionOptions: Sendable {
    /// Maximum number of query-document pairs in one pairwise model batch.
    public var maxBatchSize: Int

    /// Maximum number of token slots in one model forward pass.
    ///
    /// For pairwise models this bounds `batchSize * paddedSequenceLength`. For listwise
    /// models it bounds the complete query-and-document prompt.
    public var maxBatchTokens: Int

    /// Preferred number of tokens processed by each cached causal-model prefill step.
    public var prefillStepSize: Int

    /// Behavior when an input exceeds the model context limit.
    public var truncation: RerankTruncationPolicy

    /// Create execution options for pairwise and listwise rerankers.
    ///
    /// - Parameters:
    ///   - maxBatchSize: Maximum number of pairs evaluated in one batch.
    ///   - maxBatchTokens: Maximum token allocation for one model forward pass.
    ///   - prefillStepSize: Preferred cached prefill chunk size for causal models.
    ///   - truncation: Behavior for inputs that exceed the model context limit.
    public init(
        maxBatchSize: Int = 16,
        maxBatchTokens: Int = 8_192,
        prefillStepSize: Int = 512,
        truncation: RerankTruncationPolicy = .truncate
    ) {
        self.maxBatchSize = maxBatchSize
        self.maxBatchTokens = maxBatchTokens
        self.prefillStepSize = prefillStepSize
        self.truncation = truncation
    }
}

/// A request to score and order candidate documents by relevance to a query.
public struct RerankRequest: Sendable {
    /// The search query or user intent.
    public var query: String

    /// Candidate documents to score.
    public var documents: [String]

    /// Optional model instruction for instruction-aware rerankers.
    public var instruction: String?

    /// Maximum number of highest-scoring results to return.
    public var topK: Int?

    /// Optional inclusive lower score bound.
    ///
    /// Thresholds are supported only by rerankers with a declared bounded score range.
    public var minimumScore: Double?

    /// Create a request for sorted reranking.
    ///
    /// - Parameters:
    ///   - query: Search query or user intent.
    ///   - documents: Candidate documents to score.
    ///   - instruction: Optional task instruction for instruction-aware models.
    ///   - topK: Maximum number of sorted results to return, or `nil` for all results.
    ///   - minimumScore: Inclusive score threshold applied before `topK`.
    public init(
        query: String,
        documents: [String],
        instruction: String? = nil,
        topK: Int? = nil,
        minimumScore: Double? = nil
    ) {
        self.query = query
        self.documents = documents
        self.instruction = instruction
        self.topK = topK
        self.minimumScore = minimumScore
    }
}

/// A text document with an application-defined identifier and metadata.
///
/// The reranker evaluates only ``text``. The identifier and metadata are carried through
/// unchanged so callers can reconnect results to their retrieval records without copying
/// model-specific data into the scoring layer.
public struct RerankDocument: Sendable, Identifiable, Equatable {
    /// Application-defined stable identifier.
    public var id: String

    /// Text evaluated by the reranker.
    public var text: String

    /// Application metadata preserved in results but not evaluated by the model.
    public var metadata: [String: String]

    public init(id: String, text: String, metadata: [String: String] = [:]) {
        self.id = id
        self.text = text
        self.metadata = metadata
    }
}

/// The relevance score for one candidate document.
public struct RerankResult: Sendable {
    /// The document's position in the request's `documents` array.
    public let index: Int

    /// The model's relevance score.
    public let score: Double

    public init(index: Int, score: Double) {
        self.index = index
        self.score = score
    }
}

/// The result of a reranking operation.
public struct RerankResponse: Sendable {
    /// Identifier of the model that produced the scores.
    public let modelID: String

    /// Semantic meaning and numeric range of ``results`` scores.
    public let scoreKind: RerankScoreKind

    /// Scored documents in original or descending relevance order, depending on the API used.
    public let results: [RerankResult]

    public init(modelID: String, scoreKind: RerankScoreKind, results: [RerankResult]) {
        self.modelID = modelID
        self.scoreKind = scoreKind
        self.results = results
    }
}

/// A structured document and its relevance score.
public struct RerankedDocument: Sendable {
    /// The document's position in the request array.
    public let index: Int

    /// The original document, including its identifier and metadata.
    public let document: RerankDocument

    /// The model's relevance score.
    public let score: Double

    public init(index: Int, document: RerankDocument, score: Double) {
        self.index = index
        self.document = document
        self.score = score
    }
}

/// The result of scoring or reranking structured documents.
public struct RerankDocumentsResponse: Sendable {
    /// Identifier of the model that produced the scores.
    public let modelID: String

    /// Semantic meaning and numeric range of ``results`` scores.
    public let scoreKind: RerankScoreKind

    /// Structured document results in original or descending relevance order.
    public let results: [RerankedDocument]

    public init(
        modelID: String,
        scoreKind: RerankScoreKind,
        results: [RerankedDocument]
    ) {
        self.modelID = modelID
        self.scoreKind = scoreKind
        self.results = results
    }
}

/// Architecture-neutral interface for local reranker models.
///
/// Use ``scores(query:documents:instruction:options:)`` when scores must stay aligned with
/// the input documents. Use ``rerank(_:options:)`` for sorted retrieval results.
///
/// Rerankers consume document text rather than cached document embeddings. Pairwise
/// cross-encoders jointly encode each query-document pair, while listwise models jointly
/// encode the query and complete candidate set. Independently computed document embeddings
/// therefore cannot reproduce their relevance scores.
public protocol Reranker: Sendable {
    /// Identifier of the loaded model.
    var modelID: String { get }

    /// Semantic meaning and numeric range of returned scores.
    var scoreKind: RerankScoreKind { get }

    /// Score documents while preserving their original order.
    func scores(
        query: String,
        documents: [String],
        instruction: String?,
        options: RerankExecutionOptions
    ) async throws -> RerankResponse
}

extension Reranker {
    /// Score documents without changing their original order.
    public func scores(
        query: String,
        documents: [String],
        instruction: String? = nil,
        options: RerankExecutionOptions = .init()
    ) async throws -> RerankResponse {
        try await scores(
            query: query,
            documents: documents,
            instruction: instruction,
            options: options)
    }

    /// Score and sort documents by descending relevance.
    public func rerank(
        _ request: RerankRequest,
        options: RerankExecutionOptions = .init()
    ) async throws -> RerankResponse {
        try validate(request: request, options: options, scoreKind: scoreKind)
        var response = try await scores(
            query: request.query,
            documents: request.documents,
            instruction: request.instruction,
            options: options)
        try validate(
            response: response,
            documentCount: request.documents.count,
            expectedModelID: modelID,
            expectedScoreKind: scoreKind)

        var results = sortedByDescendingScore(response.results)
        if let minimumScore = request.minimumScore {
            results.removeAll { $0.score < minimumScore }
        }
        if let topK = request.topK {
            results = Array(results.prefix(topK))
        }

        response = RerankResponse(
            modelID: response.modelID, scoreKind: response.scoreKind, results: results)
        return response
    }

    /// Score and sort documents by descending relevance.
    public func rerank(
        query: String,
        documents: [String],
        instruction: String? = nil,
        topK: Int? = nil,
        minimumScore: Double? = nil,
        options: RerankExecutionOptions = .init()
    ) async throws -> RerankResponse {
        try await rerank(
            RerankRequest(
                query: query,
                documents: documents,
                instruction: instruction,
                topK: topK,
                minimumScore: minimumScore),
            options: options)
    }

    /// Score structured documents while preserving their original order.
    public func scores(
        query: String,
        documents: [RerankDocument],
        instruction: String? = nil,
        options: RerankExecutionOptions = .init()
    ) async throws -> RerankDocumentsResponse {
        try validateUniqueDocumentIDs(documents)
        let response = try await scores(
            query: query,
            documents: documents.map(\.text),
            instruction: instruction,
            options: options)
        return try structuredResponse(
            response,
            documents: documents,
            expectedModelID: modelID,
            expectedScoreKind: scoreKind,
            requiresCompleteOriginalOrder: true)
    }

    /// Score and sort structured documents by descending relevance.
    public func rerank(
        query: String,
        documents: [RerankDocument],
        instruction: String? = nil,
        topK: Int? = nil,
        minimumScore: Double? = nil,
        options: RerankExecutionOptions = .init()
    ) async throws -> RerankDocumentsResponse {
        try validateUniqueDocumentIDs(documents)
        let response = try await rerank(
            query: query,
            documents: documents.map(\.text),
            instruction: instruction,
            topK: topK,
            minimumScore: minimumScore,
            options: options)
        return try structuredResponse(
            response,
            documents: documents,
            expectedModelID: modelID,
            expectedScoreKind: scoreKind,
            requiresCompleteOriginalOrder: false)
    }
}

/// A thread-safe, type-erased reranker returned by model factories.
public final class RerankerContainer: Reranker, Sendable {
    /// Architecture-specific scoring operation stored by the type-erased container.
    public typealias ScoreOperation =
        @Sendable (
            _ query: String,
            _ documents: [String],
            _ instruction: String?,
            _ options: RerankExecutionOptions
        ) async throws -> [Double]

    /// Identifier of the loaded model.
    public let modelID: String

    /// Semantic meaning and numeric range of returned scores.
    public let scoreKind: RerankScoreKind
    private let scoreOperation: ScoreOperation

    /// Create a type-erased reranker around a scoring operation.
    ///
    /// The operation must return one finite score per document in original input order.
    public init(
        modelID: String,
        scoreKind: RerankScoreKind,
        scoreOperation: @escaping ScoreOperation
    ) {
        self.modelID = modelID
        self.scoreKind = scoreKind
        self.scoreOperation = scoreOperation
    }

    /// Score documents while preserving their original order.
    public func scores(
        query: String,
        documents: [String],
        instruction: String?,
        options: RerankExecutionOptions
    ) async throws -> RerankResponse {
        let request = RerankRequest(
            query: query, documents: documents, instruction: instruction)
        try validate(request: request, options: options, scoreKind: scoreKind)
        guard !documents.isEmpty else {
            return RerankResponse(modelID: modelID, scoreKind: scoreKind, results: [])
        }

        try Task.checkCancellation()
        let scores = try await scoreOperation(query, documents, instruction, options)
        guard scores.count == documents.count else {
            throw RerankerError.invalidScoreCount(
                expected: documents.count, actual: scores.count)
        }

        let results = try scores.enumerated().map { index, score in
            guard score.isFinite else {
                throw RerankerError.nonFiniteScore(index: index, score: score)
            }
            if let range = scoreKind.validRange, !range.contains(score) {
                throw RerankerError.scoreOutOfRange(index: index, score: score, range: range)
            }
            return RerankResult(index: index, score: score)
        }
        return RerankResponse(modelID: modelID, scoreKind: scoreKind, results: results)
    }
}

/// Tokenized input for an internal reranker model implementation.
package struct RerankerInput: Sendable {
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
package struct RerankerMarkerTokenIds: Sendable {
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
package protocol RerankerInputProcessor: Sendable {
    func encode(
        query: String,
        document: String,
        tokenizer: any Tokenizer,
        maxInputTokens: Int?,
        truncation: RerankTruncationPolicy
    ) throws -> RerankerInput
}

/// Encodes one query with a batch of documents for listwise reranker models.
///
/// Implement this protocol for models whose prompt contains the entire candidate set and
/// whose forward pass returns one score per document.
package protocol ListwiseRerankerInputProcessor: Sendable {
    func encode(
        query: String,
        documents: [String],
        tokenizer: any Tokenizer,
        maxInputTokens: Int?,
        truncation: RerankTruncationPolicy
    ) throws -> RerankerInput
}

/// Models that score a full document list from one reranker prompt.
package protocol ListwiseRerankerModel: BaseLanguageModel {
    func score(input: RerankerInput, documentCount: Int) throws -> [Double]
}

/// Applies a final scalar transform to a reranker logit or logit margin.
///
/// Use ``identity`` when the model already returns calibrated scores, probabilities, or
/// cosine similarities. Use ``sigmoid`` when the raw value is a binary classifier logit or a
/// true-minus-false logit margin.
package enum RerankerScoreTransform: Sendable {
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

/// Raw classifier-token logits used internally by causal-LM rerankers.
package struct RerankerLogits: Sendable {
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

/// Errors thrown while preparing or scoring reranker inputs.
public enum RerankerError: LocalizedError, Sendable {
    case emptyQuery
    case emptyDocument(index: Int)
    case duplicateDocumentID(String)
    case invalidTopK(Int)
    case invalidExecutionOption(name: String, value: Int)
    case invalidMinimumScore(Double, validRange: ClosedRange<Double>?)
    case thresholdUnsupported
    case inputTooLong(actual: Int, maximum: Int)
    case tooManyDocuments(actual: Int, maximum: Int)
    case invalidLogitShape([Int])
    case invalidScoreCount(expected: Int, actual: Int)
    case invalidResultIndex(index: Int, documentCount: Int)
    case duplicateResultIndex(Int)
    case invalidResultOrder(expected: Int, actual: Int)
    case nonFiniteScore(index: Int, score: Double)
    case scoreOutOfRange(index: Int, score: Double, range: ClosedRange<Double>)
    case emptyPrompt
    case missingClassifierToken(String)
    case classifierTokenIsNotSingleToken(String, [Int])
    case missingSpecialToken(String)
    case truncatedRequiredToken(String)
    case tokenLimitTooSmall(maxInputTokens: Int, requiredTemplateTokens: Int)
    case unsupportedModel(String)

    public var errorDescription: String? {
        switch self {
        case .emptyQuery:
            "Reranker query must not be empty."
        case .emptyDocument(let index):
            "Reranker document at index \(index) must not be empty."
        case .duplicateDocumentID(let id):
            "Reranker document identifier '\(id)' is not unique."
        case .invalidTopK(let value):
            "topK must be greater than zero, but received \(value)."
        case .invalidExecutionOption(let name, let value):
            "\(name) must be greater than zero, but received \(value)."
        case .invalidMinimumScore(let score, let validRange):
            if let validRange {
                "minimumScore \(score) is outside the valid range \(validRange)."
            } else {
                "minimumScore must be finite, but received \(score)."
            }
        case .thresholdUnsupported:
            "minimumScore is unavailable for rerankers with unbounded logit scores."
        case .inputTooLong(let actual, let maximum):
            "Reranker input contains \(actual) tokens, exceeding the \(maximum)-token limit."
        case .tooManyDocuments(let actual, let maximum):
            "Reranker received \(actual) documents, exceeding the limit of \(maximum)."
        case .invalidLogitShape(let shape):
            "Reranker logits must have shape [batch, labels], but received \(shape)."
        case .invalidScoreCount(let expected, let actual):
            "Reranker returned \(actual) scores for \(expected) documents."
        case .invalidResultIndex(let index, let documentCount):
            "Reranker returned document index \(index) for a request containing \(documentCount) documents."
        case .duplicateResultIndex(let index):
            "Reranker returned document index \(index) more than once."
        case .invalidResultOrder(let expected, let actual):
            "scores must preserve document order; expected index \(expected), received \(actual)."
        case .nonFiniteScore(let index, let score):
            "Reranker returned non-finite score \(score) for document \(index)."
        case .scoreOutOfRange(let index, let score, let range):
            "Reranker score \(score) for document \(index) is outside \(range)."
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
package struct Qwen3RerankerInputProcessor: RerankerInputProcessor {
    package var instruction: String

    package init(
        instruction: String =
            "Given a web search query, retrieve relevant passages that answer the query"
    ) {
        self.instruction = instruction
    }

    package func encode(
        query: String,
        document: String,
        tokenizer: any Tokenizer,
        maxInputTokens: Int?,
        truncation: RerankTruncationPolicy
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
        if let maxInputTokens, bodyTokens.count + templateTokenCount > maxInputTokens {
            guard truncation == .truncate else {
                throw RerankerError.inputTooLong(
                    actual: bodyTokens.count + templateTokenCount, maximum: maxInputTokens)
            }
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
        "<|im_start|>system\n"
        + "Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n"
        + "<|im_start|>user\n"

    private static let suffix =
        "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
}

/// XLM-RoBERTa sentence-pair processing used by BGE v2 rerankers.
///
/// The encoded sequence has the standard XLM-RoBERTa pair form:
/// `<s> query </s></s> document </s>`.
package struct XLMRobertaRerankerInputProcessor: RerankerInputProcessor {
    package var bosToken: String
    package var eosToken: String

    package init(bosToken: String = "<s>", eosToken: String = "</s>") {
        self.bosToken = bosToken
        self.eosToken = eosToken
    }

    package func encode(
        query: String,
        document: String,
        tokenizer: any Tokenizer,
        maxInputTokens: Int?,
        truncation: RerankTruncationPolicy
    ) throws -> RerankerInput {
        let bosTokenId = try resolveSpecialToken(bosToken, tokenizer: tokenizer)
        let eosTokenId = try resolveSpecialToken(eosToken, tokenizer: tokenizer)

        let queryTokens = tokenizer.encode(text: query, addSpecialTokens: false)
        let documentTokens = tokenizer.encode(text: document, addSpecialTokens: false)
        let truncated = try preparePair(
            first: queryTokens, second: documentTokens,
            maxInputTokens: maxInputTokens, specialTokenCount: 4,
            truncation: truncation)

        let tokenIds =
            [bosTokenId] + truncated.first + [eosTokenId, eosTokenId]
            + truncated.second + [eosTokenId]
        return RerankerInput(
            tokenIds: tokenIds,
            tokenTypeIds: Array(repeating: 0, count: tokenIds.count))
    }
}

/// BERT sentence-pair processing for sequence-classification rerankers.
///
/// The encoded sequence has the standard BERT pair form:
/// `[CLS] query [SEP] document [SEP]`, with token type IDs set to 0 for the query
/// segment and 1 for the document segment.
package struct BERTRerankerInputProcessor: RerankerInputProcessor {
    package var clsToken: String
    package var sepToken: String

    package init(clsToken: String = "[CLS]", sepToken: String = "[SEP]") {
        self.clsToken = clsToken
        self.sepToken = sepToken
    }

    package func encode(
        query: String,
        document: String,
        tokenizer: any Tokenizer,
        maxInputTokens: Int?,
        truncation: RerankTruncationPolicy
    ) throws -> RerankerInput {
        let clsTokenId = try resolveSpecialToken(clsToken, tokenizer: tokenizer)
        let sepTokenId = try resolveSpecialToken(sepToken, tokenizer: tokenizer)

        let queryTokens = tokenizer.encode(text: query, addSpecialTokens: false)
        let documentTokens = tokenizer.encode(text: document, addSpecialTokens: false)
        let truncated = try preparePair(
            first: queryTokens, second: documentTokens,
            maxInputTokens: maxInputTokens, specialTokenCount: 3,
            truncation: truncation)

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
package struct JinaRerankerInputProcessor: ListwiseRerankerInputProcessor {
    package var instruction: String?
    package var queryEmbedToken: String
    package var documentEmbedToken: String

    package init(
        instruction: String? = nil,
        queryEmbedToken: String = "<|rerank_token|>",
        documentEmbedToken: String = "<|embed_token|>"
    ) {
        self.instruction = instruction
        self.queryEmbedToken = queryEmbedToken
        self.documentEmbedToken = documentEmbedToken
    }

    package func encode(
        query: String,
        documents: [String],
        tokenizer: any Tokenizer,
        maxInputTokens: Int?,
        truncation: RerankTruncationPolicy
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
            guard truncation == .truncate else {
                throw RerankerError.inputTooLong(actual: tokenIds.count, maximum: maxInputTokens)
            }
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
        let requiredTemplateTokens = tokenizer.encode(
            text: renderPrompt(
                query: query,
                documents: Array(repeating: "", count: documents.count)),
            addSpecialTokens: false)

        guard maxInputTokens >= requiredTemplateTokens.count else {
            throw RerankerError.tokenLimitTooSmall(
                maxInputTokens: maxInputTokens,
                requiredTemplateTokens: requiredTemplateTokens.count)
        }

        // Tokenizers can merge across passage and marker boundaries. Always encode the complete
        // candidate prompt so truncation produces the same token IDs as normal prompt encoding.
        let maximumDocumentLength = documents.map(\.count).max() ?? 0
        var lowerBound = 0
        var upperBound = maximumDocumentLength
        var bestTokenIds = requiredTemplateTokens
        while lowerBound <= upperBound {
            let maximumCharacters = lowerBound + (upperBound - lowerBound) / 2
            let truncatedDocuments = documents.map {
                String($0.prefix(maximumCharacters))
            }
            let candidateTokenIds = tokenizer.encode(
                text: renderPrompt(query: query, documents: truncatedDocuments),
                addSpecialTokens: false)
            if candidateTokenIds.count <= maxInputTokens {
                bestTokenIds = candidateTokenIds
                lowerBound = maximumCharacters + 1
            } else {
                upperBound = maximumCharacters - 1
            }
        }

        guard bestTokenIds.contains(markerTokenIds.query) else {
            throw RerankerError.truncatedRequiredToken(queryEmbedToken)
        }
        guard bestTokenIds.filter({ $0 == markerTokenIds.document }).count == documents.count else {
            throw RerankerError.truncatedRequiredToken(documentEmbedToken)
        }
        return bestTokenIds
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
        "\n<query>\n\(query)"
    }

    private func renderQuerySuffix() -> String {
        "\n</query><|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    }

    private func sanitize(_ text: String, removing specialTokens: [String]) -> String {
        specialTokens.reduce(text) { result, token in
            result.replacingOccurrences(of: token, with: "")
        }
    }
}

extension ModelContainer {
    /// Score Qwen-style yes/no causal rerankers in document order.
    package func causalRerankerScores(
        query: String,
        documents: [String],
        instruction: String?,
        maxInputTokens: Int,
        options: RerankExecutionOptions
    ) async throws -> [Double] {
        guard !documents.isEmpty else { return [] }
        let instruction =
            instruction
            ?? "Given a web search query, retrieve relevant passages that answer the query"

        return try await perform(values: (query, documents, instruction, maxInputTokens, options)) {
            context, values in
            let (query, documents, instruction, maxInputTokens, options) = values
            let scorer = CausalLMReranker(
                tokenizer: context.tokenizer,
                inputProcessor: Qwen3RerankerInputProcessor(instruction: instruction),
                maxInputTokens: maxInputTokens,
                options: options)

            var encoded = try documents.enumerated().map { index, document in
                try Task.checkCancellation()
                return EncodedCausalDocument(
                    index: index,
                    input: try scorer.encode(query: query, document: document))
            }
            encoded.sort {
                if $0.input.tokenIds.count == $1.input.tokenIds.count {
                    $0.index < $1.index
                } else {
                    $0.input.tokenIds.count < $1.input.tokenIds.count
                }
            }

            var scores = [Double?](repeating: nil, count: documents.count)
            for batch in causalMicroBatches(encoded, options: options) {
                try Task.checkCancellation()
                let batchScores = try scorer.score(batch: batch, model: context.model)
                for (document, score) in zip(batch, batchScores) {
                    scores[document.index] = score
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

    /// Score listwise rerankers in document order.
    package func listwiseRerankerScores(
        query: String,
        documents: [String],
        instruction: String?,
        maxInputTokens: Int,
        maximumDocuments: Int,
        options: RerankExecutionOptions
    ) async throws -> [Double] {
        guard !documents.isEmpty else { return [] }
        guard documents.count <= maximumDocuments else {
            throw RerankerError.tooManyDocuments(
                actual: documents.count, maximum: maximumDocuments)
        }

        return try await perform(
            values: (query, documents, instruction, maxInputTokens, options)
        ) { context, values in
            try Task.checkCancellation()
            let (query, documents, instruction, maxInputTokens, options) = values
            guard let model = context.model as? any ListwiseRerankerModel else {
                throw RerankerError.unsupportedModel(
                    "\(type(of: context.model)) does not expose listwise reranker scores.")
            }
            let input = try JinaRerankerInputProcessor(instruction: instruction).encode(
                query: query,
                documents: documents,
                tokenizer: context.tokenizer,
                maxInputTokens: min(maxInputTokens, options.maxBatchTokens),
                truncation: options.truncation)
            return try model.score(input: input, documentCount: documents.count)
        }
    }
}

private struct CausalLMReranker {
    let tokenizer: any Tokenizer
    let inputProcessor: any RerankerInputProcessor
    let maxInputTokens: Int
    let options: RerankExecutionOptions

    func encode(query: String, document: String) throws -> RerankerInput {
        let input = try inputProcessor.encode(
            query: query,
            document: document,
            tokenizer: tokenizer,
            maxInputTokens: min(maxInputTokens, options.maxBatchTokens),
            truncation: options.truncation)
        guard !input.tokenIds.isEmpty else {
            throw RerankerError.emptyPrompt
        }
        return input
    }

    func score(
        batch: [EncodedCausalDocument], model: any LanguageModel
    ) throws -> [Double] {
        let classifierTokens = try resolveClassifierTokens()
        if batch.count == 1, let input = batch.first?.input {
            let lmInput = LMInput(tokens: MLXArray(input.tokenIds))
            let cache = try model.newCache(parameters: nil)
            let output = try nextTokenLogits(input: lmInput, model: model, cache: cache)
            let logits = output[0..., -1, 0...]
            return [probability(logits: logits, tokens: classifierTokens)]
        }

        let maxLength = batch.map(\.input.tokenIds.count).max() ?? 0
        var inputIDs = [Int]()
        inputIDs.reserveCapacity(batch.count * maxLength)
        for document in batch {
            inputIDs += document.input.tokenIds
            inputIDs += Array(
                repeating: 0,
                count: maxLength - document.input.tokenIds.count)
        }
        let tokens = MLXArray(inputIDs).reshaped(batch.count, maxLength)
        let logits = model(LMInput.Text(tokens: tokens), cache: nil, state: nil).logits
        let scores = batch.enumerated().map { row, document in
            probability(
                logits: logits[row, document.input.tokenIds.count - 1],
                tokens: classifierTokens)
        }
        MLX.eval(logits)
        return scores
    }

    private func nextTokenLogits(
        input: LMInput, model: any LanguageModel, cache: [KVCache]
    ) throws -> MLXArray {
        switch try model.prepare(
            input,
            cache: cache,
            state: nil,
            prefill: .init(stepSize: options.prefillStepSize)
        ) {
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
        let trueTokenId = try resolveClassifierToken("yes")
        let falseTokenId = try resolveClassifierToken("no")
        return (trueTokenId, falseTokenId)
    }

    private func resolveClassifierToken(_ token: String) throws -> Int {
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

    private func probability(
        logits: MLXArray,
        tokens: (trueTokenId: Int, falseTokenId: Int)
    ) -> Double {
        let trueLogit = scalarLogit(logits, tokenId: tokens.trueTokenId)
        let falseLogit = scalarLogit(logits, tokenId: tokens.falseTokenId)
        return RerankerScoreTransform.sigmoid(trueLogit - falseLogit)
    }
}

private struct EncodedCausalDocument {
    var index: Int
    var input: RerankerInput
}

private func causalMicroBatches(
    _ documents: [EncodedCausalDocument],
    options: RerankExecutionOptions
) -> [[EncodedCausalDocument]] {
    var batches = [[EncodedCausalDocument]]()
    var batch = [EncodedCausalDocument]()
    var longest = 0
    for document in documents {
        let nextLongest = max(longest, document.input.tokenIds.count)
        let nextCount = batch.count + 1
        if !batch.isEmpty,
            nextCount > options.maxBatchSize || nextLongest * nextCount > options.maxBatchTokens
        {
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

package func sortedByDescendingScore(_ results: [RerankResult]) -> [RerankResult] {
    results.sorted {
        if $0.score == $1.score {
            return $0.index < $1.index
        }
        return $0.score > $1.score
    }
}

private func validateUniqueDocumentIDs(_ documents: [RerankDocument]) throws {
    var identifiers = Set<String>()
    for document in documents where !identifiers.insert(document.id).inserted {
        throw RerankerError.duplicateDocumentID(document.id)
    }
}

private func structuredResponse(
    _ response: RerankResponse,
    documents: [RerankDocument],
    expectedModelID: String,
    expectedScoreKind: RerankScoreKind,
    requiresCompleteOriginalOrder: Bool
) throws -> RerankDocumentsResponse {
    try validate(
        response: response,
        documentCount: documents.count,
        expectedModelID: expectedModelID,
        expectedScoreKind: expectedScoreKind,
        requiresCompleteOriginalOrder: requiresCompleteOriginalOrder)
    return RerankDocumentsResponse(
        modelID: response.modelID,
        scoreKind: response.scoreKind,
        results: response.results.map { result in
            RerankedDocument(
                index: result.index,
                document: documents[result.index],
                score: result.score)
        })
}

private func validate(
    request: RerankRequest,
    options: RerankExecutionOptions,
    scoreKind: RerankScoreKind
) throws {
    guard !request.query.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
        throw RerankerError.emptyQuery
    }
    if let index = request.documents.firstIndex(where: {
        $0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }) {
        throw RerankerError.emptyDocument(index: index)
    }
    if let topK = request.topK, topK <= 0 {
        throw RerankerError.invalidTopK(topK)
    }
    for (name, value) in [
        ("maxBatchSize", options.maxBatchSize),
        ("maxBatchTokens", options.maxBatchTokens),
        ("prefillStepSize", options.prefillStepSize),
    ] where value <= 0 {
        throw RerankerError.invalidExecutionOption(name: name, value: value)
    }
    if let minimumScore = request.minimumScore {
        guard minimumScore.isFinite else {
            throw RerankerError.invalidMinimumScore(minimumScore, validRange: scoreKind.validRange)
        }
        guard let validRange = scoreKind.validRange else {
            throw RerankerError.thresholdUnsupported
        }
        guard validRange.contains(minimumScore) else {
            throw RerankerError.invalidMinimumScore(minimumScore, validRange: validRange)
        }
    }
}

private func validate(
    response: RerankResponse,
    documentCount: Int,
    expectedModelID: String,
    expectedScoreKind: RerankScoreKind,
    requiresCompleteOriginalOrder: Bool = true
) throws {
    guard response.modelID == expectedModelID, response.scoreKind == expectedScoreKind else {
        throw RerankerError.unsupportedModel(
            "Reranker response metadata does not match the loaded model.")
    }
    guard !requiresCompleteOriginalOrder || response.results.count == documentCount else {
        throw RerankerError.invalidScoreCount(
            expected: documentCount, actual: response.results.count)
    }
    guard response.results.count <= documentCount else {
        throw RerankerError.invalidScoreCount(
            expected: documentCount, actual: response.results.count)
    }
    var seenIndices = Set<Int>()
    for (expectedIndex, result) in response.results.enumerated() {
        guard (0 ..< documentCount).contains(result.index) else {
            throw RerankerError.invalidResultIndex(
                index: result.index, documentCount: documentCount)
        }
        guard seenIndices.insert(result.index).inserted else {
            throw RerankerError.duplicateResultIndex(result.index)
        }
        if requiresCompleteOriginalOrder, result.index != expectedIndex {
            throw RerankerError.invalidResultOrder(
                expected: expectedIndex, actual: result.index)
        }
        guard result.score.isFinite else {
            throw RerankerError.nonFiniteScore(index: result.index, score: result.score)
        }
        if let range = response.scoreKind.validRange, !range.contains(result.score) {
            throw RerankerError.scoreOutOfRange(
                index: result.index, score: result.score, range: range)
        }
    }
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

private func preparePair(
    first: [Int],
    second: [Int],
    maxInputTokens: Int?,
    specialTokenCount: Int,
    truncation: RerankTruncationPolicy
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

    if first.count + second.count > tokenBudget, truncation == .error {
        throw RerankerError.inputTooLong(
            actual: first.count + second.count + specialTokenCount,
            maximum: maxInputTokens)
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
