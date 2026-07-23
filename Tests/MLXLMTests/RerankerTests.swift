// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXEmbedders
@testable import MLXLLM

struct RerankerTests {
    @Test func rerankSortsByScoreDescending() async throws {
        let tokenizer = ByteRerankerTokenizer()
        let container = ModelContainer(
            context: ModelContext(
                configuration: ModelConfiguration(id: "test/reranker"),
                model: SumScoringRerankerModel(
                    trueTokenId: tokenizer.trueTokenId,
                    falseTokenId: tokenizer.falseTokenId,
                    prepareMode: .logits
                ),
                processor: TestInputProcessor(tokenizer: tokenizer),
                tokenizer: tokenizer
            )
        )

        let configuration = CausalLMRerankerConfiguration(
            inputProcessor: MinimalRerankerInputProcessor(),
            trueTokenId: tokenizer.trueTokenId,
            falseTokenId: tokenizer.falseTokenId,
            scoreTransform: .identity
        )

        let results = try await container.rerank(
            query: "swift", documents: ["aa", "zz"], configuration: configuration)

        #expect(results.map(\.document) == ["zz", "aa"])
        #expect(results.map(\.index) == [1, 0])
        #expect(results[0].score > results[1].score)
    }

    @Test func causalRerankLimitsTopKResults() async throws {
        let tokenizer = ByteRerankerTokenizer()
        let container = ModelContainer(
            context: ModelContext(
                configuration: ModelConfiguration(id: "test/reranker"),
                model: SumScoringRerankerModel(
                    trueTokenId: tokenizer.trueTokenId,
                    falseTokenId: tokenizer.falseTokenId,
                    prepareMode: .logits
                ),
                processor: TestInputProcessor(tokenizer: tokenizer),
                tokenizer: tokenizer
            )
        )

        let configuration = CausalLMRerankerConfiguration(
            inputProcessor: MinimalRerankerInputProcessor(),
            trueTokenId: tokenizer.trueTokenId,
            falseTokenId: tokenizer.falseTokenId,
            scoreTransform: .identity
        )

        let results = try await container.rerank(
            query: "swift",
            documents: ["aa", "zz", "mm"],
            topK: 1,
            configuration: configuration)

        #expect(results.map(\.document) == ["zz"])

        let emptyResults = try await container.rerank(
            query: "swift",
            documents: ["aa", "zz"],
            topK: 0,
            configuration: configuration)

        #expect(emptyResults.isEmpty)
    }

    @Test func processPreservesInputOrderByDefault() async throws {
        let tokenizer = ByteRerankerTokenizer()
        let container = ModelContainer(
            context: ModelContext(
                configuration: ModelConfiguration(id: "test/reranker"),
                model: SumScoringRerankerModel(
                    trueTokenId: tokenizer.trueTokenId,
                    falseTokenId: tokenizer.falseTokenId,
                    prepareMode: .logits
                ),
                processor: TestInputProcessor(tokenizer: tokenizer),
                tokenizer: tokenizer
            )
        )

        let results = try await container.process(
            .init(query: "swift", documents: ["aa", "zz"]),
            configuration: CausalLMRerankerConfiguration(
                inputProcessor: MinimalRerankerInputProcessor(),
                trueTokenId: tokenizer.trueTokenId,
                falseTokenId: tokenizer.falseTokenId,
                scoreTransform: .identity
            ))

        #expect(results.map(\.document) == ["aa", "zz"])
        #expect(results[1].score > results[0].score)
    }

    @Test func scoreSupportsPrepareTokensPath() async throws {
        let tokenizer = ByteRerankerTokenizer()
        let container = ModelContainer(
            context: ModelContext(
                configuration: ModelConfiguration(id: "test/reranker"),
                model: SumScoringRerankerModel(
                    trueTokenId: tokenizer.trueTokenId,
                    falseTokenId: tokenizer.falseTokenId,
                    prepareMode: .tokens
                ),
                processor: TestInputProcessor(tokenizer: tokenizer),
                tokenizer: tokenizer
            )
        )

        let result = try await container.score(
            query: "q",
            document: "doc",
            configuration: CausalLMRerankerConfiguration(
                inputProcessor: MinimalRerankerInputProcessor(),
                trueTokenId: tokenizer.trueTokenId,
                falseTokenId: tokenizer.falseTokenId,
                scoreTransform: .sigmoid
            )
        )

        #expect(result.score > 0.5)
        #expect(result.logits?.difference ?? 0 > 0)
    }

    @Test func maxInputTokensKeepsSuffixBudget() async throws {
        let tokenizer = ByteRerankerTokenizer()
        let container = ModelContainer(
            context: ModelContext(
                configuration: ModelConfiguration(id: "test/reranker"),
                model: SumScoringRerankerModel(
                    trueTokenId: tokenizer.trueTokenId,
                    falseTokenId: tokenizer.falseTokenId,
                    prepareMode: .logits
                ),
                processor: TestInputProcessor(tokenizer: tokenizer),
                tokenizer: tokenizer
            )
        )

        let configuration = CausalLMRerankerConfiguration(
            inputProcessor: MinimalRerankerInputProcessor(prefix: "p", suffix: "s"),
            trueTokenId: tokenizer.trueTokenId,
            falseTokenId: tokenizer.falseTokenId,
            maxInputTokens: 8
        )

        let result = try await container.score(
            query: "q",
            document: String(repeating: "z", count: 100),
            configuration: configuration
        )

        #expect(result.score > 0)
    }

    @Test func embedderRerankSortsByScoreDescending() async throws {
        let tokenizer = ByteRerankerTokenizer()
        let model = SumScoringEmbedderRerankerModel()
        let container = EmbedderModelContainer(
            context: EmbedderModelContext(
                configuration: ModelConfiguration(id: "test/embedder-reranker"),
                model: model,
                tokenizer: tokenizer,
                pooling: Pooling(strategy: .none)
            )
        )

        let results = try await container.rerank(
            query: "swift",
            documents: ["aa", "zz"],
            configuration: RerankerConfiguration(
                inputProcessor: XLMRobertaRerankerInputProcessor(),
                scoreTransform: .identity
            )
        )

        #expect(results.map(\.document) == ["zz", "aa"])
        #expect(results[0].score > results[1].score)
        #expect(results[0].logits == nil)
        #expect(model.scoreCallCount == 1)
    }

    @Test func embedderRerankLimitsTopKResults() async throws {
        let tokenizer = ByteRerankerTokenizer()
        let model = SumScoringEmbedderRerankerModel()
        let container = EmbedderModelContainer(
            context: EmbedderModelContext(
                configuration: ModelConfiguration(id: "test/embedder-reranker"),
                model: model,
                tokenizer: tokenizer,
                pooling: Pooling(strategy: .none)
            )
        )

        let results = try await container.rerank(
            query: "swift",
            documents: ["aa", "zz", "mm"],
            topK: 2,
            configuration: RerankerConfiguration(
                inputProcessor: XLMRobertaRerankerInputProcessor(),
                scoreTransform: .identity
            ))

        #expect(results.map(\.document) == ["zz", "mm"])
        #expect(model.scoreCallCount == 1)
    }

    @Test func topResultsLimitsAlreadyOrderedResults() {
        let results = [
            RerankResult(index: 1, document: "b", score: 2),
            RerankResult(index: 0, document: "a", score: 1),
        ]

        #expect(topResults(results, topK: nil).map(\.document) == ["b", "a"])
        #expect(topResults(results, topK: 1).map(\.document) == ["b"])
        #expect(topResults(results, topK: -1).isEmpty)
    }

    @Test func embedderProcessSupportsTwoLogitClassifierHeads() async throws {
        let tokenizer = ByteRerankerTokenizer()
        let container = EmbedderModelContainer(
            context: EmbedderModelContext(
                configuration: ModelConfiguration(id: "test/two-logit-reranker"),
                model: TwoLogitEmbedderRerankerModel(),
                tokenizer: tokenizer,
                pooling: Pooling(strategy: .none)
            )
        )

        let results = try await container.process(
            .init(query: "swift", documents: ["aa", "zz"]),
            configuration: RerankerConfiguration(
                inputProcessor: XLMRobertaRerankerInputProcessor(),
                logitSelection: .logitDifference(positive: 1, negative: 0),
                scoreTransform: .identity
            ))

        #expect(results.map(\.document) == ["aa", "zz"])
        #expect(results[1].score > results[0].score)
        #expect(results[0].rawScore != nil)
    }

    @Test func embedderRerankerCapsInputBudgetToModelPositionLimit() async throws {
        let tokenizer = ByteRerankerTokenizer()
        let model = SumScoringEmbedderRerankerModel(maxPositionEmbeddings: 6)
        let container = EmbedderModelContainer(
            context: EmbedderModelContext(
                configuration: ModelConfiguration(id: "test/capped-encoder-reranker"),
                model: model,
                tokenizer: tokenizer,
                pooling: Pooling(strategy: .none)
            )
        )

        let results = try await container.process(
            .init(query: "q", documents: ["abcdefghij"]),
            configuration: RerankerConfiguration(
                inputProcessor: XLMRobertaRerankerInputProcessor(),
                scoreTransform: .identity,
                maxInputTokens: 8_192
            ))

        #expect(results.count == 1)
        #expect(model.lastInputShape == [1, 6])
        #expect(
            model.lastInputIds == [
                tokenizer._bosTokenId,
                Int("q".utf8.first!) + 10,
                tokenizer._eosTokenId,
                tokenizer._eosTokenId,
                Int("a".utf8.first!) + 10,
                tokenizer._eosTokenId,
            ])
    }

    @Test func xlmRobertaInputProcessorBuildsSentencePair() throws {
        let tokenizer = ByteRerankerTokenizer()
        let input = try XLMRobertaRerankerInputProcessor().encode(
            query: "q",
            document: "d",
            tokenizer: tokenizer,
            maxInputTokens: nil
        )

        #expect(input.tokenIds.first == tokenizer._bosTokenId)
        #expect(input.tokenIds.filter { $0 == tokenizer._eosTokenId }.count == 3)
        #expect(input.tokenTypeIds == nil)
    }

    @Test func xlmRobertaSequenceClassificationConfigScoresSingleLogit() throws {
        let json = """
            {
              "model_type": "xlm-roberta",
              "architectures": ["XLMRobertaForSequenceClassification"],
              "num_labels": 1,
              "vocab_size": 128,
              "hidden_size": 8,
              "num_attention_heads": 2,
              "intermediate_size": 16,
              "num_hidden_layers": 1,
              "type_vocab_size": 1,
              "max_position_embeddings": 16
            }
            """
        let config = try JSONDecoder().decode(BertConfiguration.self, from: Data(json.utf8))
        let model = BertRerankerModel(config)
        let inputIds = MLXArray([3, 10, 4, 4, 11, 4]).reshaped(1, 6)
        let attentionMask = MLXArray.ones(inputIds.shape, dtype: .int32)

        let logits = model.score(
            inputIds,
            positionIds: nil as MLXArray?,
            tokenTypeIds: nil as MLXArray?,
            attentionMask: attentionMask)

        #expect(logits.shape == [1, 1])
    }

    @Test func regularBertRerankThrowsUnsupportedModelInsteadOfCrashing() async throws {
        let json = """
            {
              "model_type": "bert",
              "vocab_size": 128,
              "hidden_size": 8,
              "num_attention_heads": 2,
              "intermediate_size": 16,
              "num_hidden_layers": 1,
              "type_vocab_size": 1,
              "max_position_embeddings": 16
            }
            """
        let config = try JSONDecoder().decode(BertConfiguration.self, from: Data(json.utf8))
        let container = EmbedderModelContainer(
            context: EmbedderModelContext(
                configuration: ModelConfiguration(id: "test/bert-embedder"),
                model: BertModel(config),
                tokenizer: ByteRerankerTokenizer(),
                pooling: Pooling(strategy: .none)
            )
        )

        do {
            _ = try await container.process(
                .init(query: "swift", documents: ["arrays"]),
                configuration: RerankerConfiguration())
            Issue.record("Expected non-classifier BERT reranking to throw unsupportedModel.")
        } catch RerankerError.unsupportedModel(let message) {
            #expect(message.contains("does not expose reranker logits"))
        }
    }

    @Test func xlmRobertaSequenceClassificationConfigCreatesBertRerankerModel() async throws {
        let json = """
            {
              "model_type": "xlm-roberta",
              "architectures": ["XLMRobertaForSequenceClassification"],
              "num_labels": 1,
              "vocab_size": 128,
              "hidden_size": 8,
              "num_attention_heads": 2,
              "intermediate_size": 16,
              "num_hidden_layers": 1,
              "type_vocab_size": 1,
              "max_position_embeddings": 16
            }
            """

        let model = try await EmbedderTypeRegistry.shared.createModel(
            configuration: Data(json.utf8),
            modelType: "xlm-roberta")

        #expect(model is BertRerankerModel)
    }

    @Test func bertSequenceClassificationConfigCreatesPooledRerankerModel() async throws {
        let json = """
            {
              "model_type": "bert",
              "architectures": ["BertForSequenceClassification"],
              "num_labels": 1,
              "vocab_size": 128,
              "hidden_size": 8,
              "num_attention_heads": 2,
              "intermediate_size": 16,
              "num_hidden_layers": 1,
              "type_vocab_size": 2,
              "max_position_embeddings": 16
            }
            """

        let model = try await EmbedderTypeRegistry.shared.createModel(
            configuration: Data(json.utf8),
            modelType: "bert")

        #expect(model is BertSequenceClassificationRerankerModel)
    }

    @Test func bertSequenceClassificationScoresFromPooledOutput() throws {
        let json = """
            {
              "model_type": "bert",
              "architectures": ["BertForSequenceClassification"],
              "num_labels": 1,
              "vocab_size": 128,
              "hidden_size": 8,
              "num_attention_heads": 2,
              "intermediate_size": 16,
              "num_hidden_layers": 1,
              "type_vocab_size": 2,
              "max_position_embeddings": 16
            }
            """
        let config = try JSONDecoder().decode(BertConfiguration.self, from: Data(json.utf8))
        let model = BertSequenceClassificationRerankerModel(config)
        let inputIds = MLXArray([101, 10, 102, 11, 102]).reshaped(1, 5)
        let tokenTypeIds = MLXArray([0, 0, 0, 1, 1]).reshaped(1, 5)
        let attentionMask = MLXArray.ones(inputIds.shape, dtype: .int32)

        let logits = model.score(
            inputIds,
            positionIds: nil as MLXArray?,
            tokenTypeIds: tokenTypeIds,
            attentionMask: attentionMask)

        #expect(logits.shape == [1, 1])
    }

    @Test func bertSequenceClassificationSanitizeKeepsPooledHeadKeys() throws {
        let json = """
            {
              "model_type": "bert",
              "architectures": ["BertForSequenceClassification"],
              "num_labels": 1,
              "vocab_size": 128,
              "hidden_size": 8,
              "num_attention_heads": 2,
              "intermediate_size": 16,
              "num_hidden_layers": 1,
              "type_vocab_size": 2,
              "max_position_embeddings": 16
            }
            """
        let config = try JSONDecoder().decode(BertConfiguration.self, from: Data(json.utf8))
        let model = BertSequenceClassificationRerankerModel(config)

        let sanitized = model.sanitize(weights: [
            "bert.embeddings.word_embeddings.weight": MLXArray.zeros([128, 8]),
            "bert.encoder.layer.0.output.dense.weight": MLXArray.zeros([8, 16]),
            "bert.pooler.dense.weight": MLXArray.zeros([8, 8]),
            "bert.pooler.dense.bias": MLXArray.zeros([8]),
            "classifier.weight": MLXArray.zeros([1, 8]),
            "classifier.bias": MLXArray.zeros([1]),
        ])

        #expect(sanitized["encoder_model.embeddings.word_embeddings.weight"] != nil)
        #expect(sanitized["encoder_model.encoder.layers.0.linear2.weight"] != nil)
        #expect(sanitized["pooler.weight"] != nil)
        #expect(sanitized["pooler.bias"] != nil)
        #expect(sanitized["classifier.weight"] != nil)
        #expect(sanitized["classifier.bias"] != nil)
    }

    @Test func bertRerankerSanitizePrefixesEncoderWeightsOnly() throws {
        let json = """
            {
              "model_type": "xlm-roberta",
              "architectures": ["XLMRobertaForSequenceClassification"],
              "num_labels": 1,
              "vocab_size": 128,
              "hidden_size": 8,
              "num_attention_heads": 2,
              "intermediate_size": 16,
              "num_hidden_layers": 1,
              "type_vocab_size": 1,
              "max_position_embeddings": 16
            }
            """
        let config = try JSONDecoder().decode(BertConfiguration.self, from: Data(json.utf8))
        let model = BertRerankerModel(config)

        let sanitized = model.sanitize(weights: [
            "roberta.embeddings.word_embeddings.weight": MLXArray.zeros([128, 8]),
            "roberta.encoder.layer.0.output.dense.weight": MLXArray.zeros([8, 16]),
            "classifier.out_proj.weight": MLXArray.zeros([1, 8]),
        ])

        #expect(sanitized["encoder_model.embeddings.word_embeddings.weight"] != nil)
        #expect(sanitized["encoder_model.encoder.layers.0.linear2.weight"] != nil)
        #expect(sanitized["classifier.out_proj.weight"] != nil)
    }

    @Test func qwen3JinaForRankingConfigCreatesJinaRerankerModel() async throws {
        let json = """
            {
              "architectures": ["JinaForRanking"],
              "model_type": "qwen3",
              "vocab_size": 128,
              "hidden_size": 8,
              "num_hidden_layers": 1,
              "intermediate_size": 16,
              "num_attention_heads": 2,
              "num_key_value_heads": 1,
              "head_dim": 4,
              "rms_norm_eps": 1e-6,
              "tie_word_embeddings": true
            }
            """

        let model = try await LLMTypeRegistry.shared.createModel(
            configuration: Data(json.utf8),
            modelType: "qwen3")

        #expect(model is JinaRerankerModel)
    }

    @Test func jinaSanitizePrefixesProjectorWeights() throws {
        let json = """
            {
              "model_type": "qwen3",
              "vocab_size": 128,
              "hidden_size": 8,
              "num_hidden_layers": 1,
              "intermediate_size": 16,
              "num_attention_heads": 2,
              "num_key_value_heads": 1,
              "head_dim": 4,
              "rms_norm_eps": 1e-6,
              "tie_word_embeddings": true
            }
            """
        let config = try JSONDecoder().decode(MLXLLM.Qwen3Configuration.self, from: Data(json.utf8))
        let model = JinaRerankerModel(config)

        let sanitized = model.sanitize(weights: [
            "linear1.weight": MLXArray.zeros([512, 8]),
            "linear2.weight": MLXArray.zeros([512, 512]),
        ])

        #expect(sanitized["projector.linear1.weight"] != nil)
        #expect(sanitized["projector.linear2.weight"] != nil)
    }

    @Test func jinaInputProcessorStoresTokenizerResolvedMarkerIds() throws {
        let tokenizer = ByteRerankerTokenizer()
        let input = try JinaRerankerInputProcessor().encode(
            query: "q",
            documents: ["d1", "d2"],
            tokenizer: tokenizer,
            maxInputTokens: nil
        )

        #expect(input.markerTokenIds?.query == tokenizer.rerankTokenId)
        #expect(input.markerTokenIds?.document == tokenizer.embedTokenId)
        #expect(input.tokenIds.filter { $0 == tokenizer.embedTokenId }.count == 2)
    }

    @Test func jinaInputProcessorTruncatesDocumentsWithoutDroppingMarkers() throws {
        let tokenizer = ByteRerankerTokenizer()
        let processor = JinaRerankerInputProcessor()
        let documents = [
            String(repeating: "a", count: 120),
            String(repeating: "b", count: 80),
        ]
        let fullInput = try processor.encode(
            query: "q",
            documents: documents,
            tokenizer: tokenizer,
            maxInputTokens: nil
        )
        guard let queryMarkerIndex = fullInput.tokenIds.firstIndex(of: tokenizer.rerankTokenId)
        else {
            Issue.record("Expected full Jina prompt to contain the query marker.")
            return
        }

        let truncatedInput = try processor.encode(
            query: "q",
            documents: documents,
            tokenizer: tokenizer,
            maxInputTokens: queryMarkerIndex
        )

        #expect(truncatedInput.tokenIds.count <= queryMarkerIndex)
        #expect(truncatedInput.tokenIds.contains(tokenizer.rerankTokenId))
        #expect(
            truncatedInput.tokenIds.filter { $0 == tokenizer.embedTokenId }.count == documents.count
        )
        #expect(truncatedInput.tokenIds.count < fullInput.tokenIds.count)
    }

    @Test func jinaScoreRejectsMissingMarkerTokenIds() throws {
        let json = """
            {
              "model_type": "qwen3",
              "vocab_size": 128,
              "hidden_size": 8,
              "num_hidden_layers": 1,
              "intermediate_size": 16,
              "num_attention_heads": 2,
              "num_key_value_heads": 1,
              "head_dim": 4,
              "rms_norm_eps": 1e-6,
              "tie_word_embeddings": true
            }
            """
        let config = try JSONDecoder().decode(MLXLLM.Qwen3Configuration.self, from: Data(json.utf8))
        let model = JinaRerankerModel(config)
        let input = RerankerInput(tokenIds: [5, 10, 6])

        do {
            _ = try model.score(input: input, documentCount: 1)
            Issue.record("Expected missing marker token IDs to throw.")
        } catch RerankerError.unsupportedModel(let message) {
            #expect(message.contains("missing marker token IDs"))
        }
    }

    @Test func jinaScoreRejectsMultipleQueryMarkers() throws {
        let json = """
            {
              "model_type": "qwen3",
              "vocab_size": 128,
              "hidden_size": 8,
              "num_hidden_layers": 1,
              "intermediate_size": 16,
              "num_attention_heads": 2,
              "num_key_value_heads": 1,
              "head_dim": 4,
              "rms_norm_eps": 1e-6,
              "tie_word_embeddings": true
            }
            """
        let config = try JSONDecoder().decode(MLXLLM.Qwen3Configuration.self, from: Data(json.utf8))
        let model = JinaRerankerModel(config)
        let input = RerankerInput(
            tokenIds: [5, 10, 5, 6],
            markerTokenIds: .init(query: 5, document: 6))

        do {
            _ = try model.score(input: input, documentCount: 1)
            Issue.record("Expected multiple query markers to throw.")
        } catch RerankerError.unsupportedModel(let message) {
            #expect(message.contains("Expected exactly one <|rerank_token|>"))
        }
    }
}

private struct MinimalRerankerInputProcessor: RerankerInputProcessor {
    var prefix = ""
    var suffix = ""

    func encode(
        query: String,
        document: String,
        tokenizer: any Tokenizer,
        maxInputTokens: Int?
    ) throws -> RerankerInput {
        let prefixTokens = tokenizer.encode(text: prefix, addSpecialTokens: false)
        let bodyTokens = tokenizer.encode(text: query + document, addSpecialTokens: false)
        let suffixTokens = tokenizer.encode(text: suffix, addSpecialTokens: false)

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
}

private struct ByteRerankerTokenizer: Tokenizer {
    let trueTokenId = 1
    let falseTokenId = 2
    let _bosTokenId = 3
    let _eosTokenId = 4
    let rerankTokenId = 5
    let embedTokenId = 6

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        var tokenIds = [Int]()
        var remaining = text[...]
        while !remaining.isEmpty {
            if remaining.hasPrefix("<|rerank_token|>") {
                tokenIds.append(rerankTokenId)
                remaining.removeFirst("<|rerank_token|>".count)
            } else if remaining.hasPrefix("<|embed_token|>") {
                tokenIds.append(embedTokenId)
                remaining.removeFirst("<|embed_token|>".count)
            } else {
                tokenIds.append(Int(remaining.removeFirst().asciiValue ?? 0) + 10)
            }
        }
        return tokenIds
    }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        String(decoding: tokenIds.map { UInt8(max(0, $0 - 10)) }, as: UTF8.self)
    }

    func convertTokenToId(_ token: String) -> Int? {
        switch token {
        case "yes": trueTokenId
        case "no": falseTokenId
        case "<s>": _bosTokenId
        case "</s>": _eosTokenId
        case "<|rerank_token|>": rerankTokenId
        case "<|embed_token|>": embedTokenId
        default: nil
        }
    }

    func convertIdToToken(_ id: Int) -> String? {
        switch id {
        case trueTokenId:
            "yes"
        case falseTokenId:
            "no"
        case _bosTokenId:
            "<s>"
        case _eosTokenId:
            "</s>"
        case rerankTokenId:
            "<|rerank_token|>"
        case embedTokenId:
            "<|embed_token|>"
        default:
            nil
        }
    }

    var bosToken: String? { "<s>" }
    var eosToken: String? { "</s>" }
    var unknownToken: String? { nil }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        []
    }
}

private final class SumScoringEmbedderRerankerModel: Module, RerankerModel {
    var vocabularySize: Int { 256 }
    var maxPositionEmbeddings: Int? { _maxPositionEmbeddings }
    var scoreCallCount = 0
    var lastInputIds: [Int]?
    var lastInputShape: [Int]?

    private let _maxPositionEmbeddings: Int?

    init(maxPositionEmbeddings: Int? = nil) {
        _maxPositionEmbeddings = maxPositionEmbeddings
    }

    func callAsFunction(
        _ inputs: MLXArray,
        positionIds: MLXArray?,
        tokenTypeIds: MLXArray?,
        attentionMask: MLXArray?
    ) -> EmbeddingModelOutput {
        EmbeddingModelOutput(hiddenStates: inputs.asType(.float32), pooledOutput: nil)
    }

    func score(
        _ inputs: MLXArray,
        positionIds: MLXArray?,
        tokenTypeIds: MLXArray?,
        attentionMask: MLXArray?
    ) -> MLXArray {
        scoreCallCount += 1
        lastInputShape = inputs.shape
        lastInputIds = inputs.asArray(Int.self)
        return sum(inputs.asType(.float32), axis: 1).reshaped(inputs.dim(0), 1)
    }
}

private final class TwoLogitEmbedderRerankerModel: Module, RerankerModel {
    var vocabularySize: Int { 256 }

    func callAsFunction(
        _ inputs: MLXArray,
        positionIds: MLXArray?,
        tokenTypeIds: MLXArray?,
        attentionMask: MLXArray?
    ) -> EmbeddingModelOutput {
        EmbeddingModelOutput(hiddenStates: inputs.asType(.float32), pooledOutput: nil)
    }

    func score(
        _ inputs: MLXArray,
        positionIds: MLXArray?,
        tokenTypeIds: MLXArray?,
        attentionMask: MLXArray?
    ) -> MLXArray {
        let sums = sum(inputs.asType(.float32), axis: 1)
        return stacked([-sums, sums], axis: 1)
    }
}

private final class SumScoringRerankerModel: Module, LanguageModel {
    enum PrepareMode {
        case logits
        case tokens
    }

    let trueTokenId: Int
    let falseTokenId: Int
    let prepareMode: PrepareMode

    init(trueTokenId: Int, falseTokenId: Int, prepareMode: PrepareMode) {
        self.trueTokenId = trueTokenId
        self.falseTokenId = falseTokenId
        self.prepareMode = prepareMode
    }

    func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
        switch prepareMode {
        case .logits:
            .logits(.init(logits: logits(for: input.text.tokens)))
        case .tokens:
            .tokens(input.text)
        }
    }

    func callAsFunction(_ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?)
        -> LMOutput
    {
        .init(logits: logits(for: input.tokens))
    }

    func newCache(parameters: GenerateParameters?) -> [KVCache] {
        []
    }

    private func logits(for tokens: MLXArray) -> MLXArray {
        let tokenSum = tokens.asArray(Int.self).reduce(0, +)
        let score = Float(tokenSum % 1_000) / 100
        var values = Array(repeating: Float(-100), count: 128)
        values[trueTokenId] = score
        values[falseTokenId] = 0
        return MLXArray(values).reshaped(1, 1, values.count)
    }
}

extension TestInputProcessor {
    fileprivate init(tokenizer: any Tokenizer) {
        self.init(
            tokenizer: tokenizer,
            configuration: ModelConfiguration(id: "test/reranker"),
            messageGenerator: DefaultMessageGenerator()
        )
    }
}
