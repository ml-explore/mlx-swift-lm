// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXEmbedders
@testable import MLXLLM
@testable import MLXRerankers

struct RerankerTests {
    @Test func scoresPreserveDocumentOrder() async throws {
        let reranker = RerankerContainer(
            modelID: "test/model",
            scoreKind: .normalizedRelevance
        ) { _, _, _, _ in [0.2, 0.9, 0.4] }

        let response = try await reranker.scores(
            query: "swift", documents: ["a", "b", "c"])

        #expect(response.modelID == "test/model")
        #expect(response.scoreKind == .normalizedRelevance)
        #expect(response.results.map(\.index) == [0, 1, 2])
        #expect(response.results.map(\.score) == [0.2, 0.9, 0.4])
    }

    @Test func rerankSortsStablyFiltersAndLimitsResults() async throws {
        let reranker = RerankerContainer(
            modelID: "test/model",
            scoreKind: .normalizedRelevance
        ) { _, _, _, _ in [0.8, 0.4, 0.8, 0.9] }

        let response = try await reranker.rerank(
            query: "swift",
            documents: ["a", "b", "c", "d"],
            topK: 3,
            minimumScore: 0.8)

        #expect(response.results.map(\.index) == [3, 0, 2])
    }

    @Test func structuredDocumentsPreserveIdentityAndMetadata() async throws {
        let reranker = makeConstantReranker(
            scoreKind: .normalizedRelevance, scores: [0.2, 0.9])
        let documents = [
            RerankDocument(id: "first", text: "a", metadata: ["source": "one"]),
            RerankDocument(id: "second", text: "b", metadata: ["source": "two"]),
        ]

        let response = try await reranker.rerank(
            query: "q", documents: documents, topK: 1)

        #expect(response.modelID == "test/model")
        #expect(response.results.map(\.document.id) == ["second"])
        #expect(response.results[0].document.metadata == ["source": "two"])
        #expect(response.results[0].index == 1)
        #expect(response.results[0].score == 0.9)
    }

    @Test func structuredDocumentsRequireUniqueIdentifiers() async throws {
        let reranker = makeConstantReranker(
            scoreKind: .normalizedRelevance, scores: [0.2, 0.9])
        let documents = [
            RerankDocument(id: "duplicate", text: "a"),
            RerankDocument(id: "duplicate", text: "b"),
        ]

        await #expect(throws: RerankerError.self) {
            try await reranker.rerank(query: "q", documents: documents)
        }
    }

    @Test func structuredDocumentsRejectInvalidProtocolResponses() async throws {
        let documents = [RerankDocument(id: "only", text: "document")]
        let reranker = StubReranker(
            results: [RerankResult(index: 1, score: 0.5)])

        await #expect(throws: RerankerError.self) {
            try await reranker.scores(query: "q", documents: documents)
        }
    }

    @Test func structuredDocumentsRejectDuplicateProtocolResultIndexes() async throws {
        let documents = [
            RerankDocument(id: "first", text: "a"),
            RerankDocument(id: "second", text: "b"),
        ]
        let reranker = StubReranker(
            results: [
                RerankResult(index: 0, score: 0.9),
                RerankResult(index: 0, score: 0.8),
            ])

        await #expect(throws: RerankerError.self) {
            try await reranker.rerank(query: "q", documents: documents)
        }
    }

    @Test func invalidTopKThrows() async throws {
        let reranker = makeConstantReranker(
            scoreKind: .normalizedRelevance, scores: [0.5])

        await #expect(throws: RerankerError.self) {
            try await reranker.rerank(query: "q", documents: ["d"], topK: 0)
        }
    }

    @Test func thresholdRequiresBoundedScores() async throws {
        let reranker = makeConstantReranker(scoreKind: .logit, scores: [2])

        await #expect(throws: RerankerError.self) {
            try await reranker.rerank(
                query: "q", documents: ["d"], minimumScore: 0.5)
        }
    }

    @Test func nonFiniteAndOutOfRangeScoresThrow() async throws {
        let nonFinite = makeConstantReranker(
            scoreKind: .normalizedRelevance, scores: [.nan])
        let outOfRange = makeConstantReranker(
            scoreKind: .normalizedRelevance, scores: [1.1])

        await #expect(throws: RerankerError.self) {
            try await nonFinite.scores(query: "q", documents: ["d"])
        }
        await #expect(throws: RerankerError.self) {
            try await outOfRange.scores(query: "q", documents: ["d"])
        }
    }

    @Test func cancellationIsObservedBeforeInference() async throws {
        let reranker = makeConstantReranker(
            scoreKind: .normalizedRelevance, scores: [0.5])
        let task = Task {
            withUnsafeCurrentTask { $0?.cancel() }
            return try await reranker.scores(query: "q", documents: ["d"])
        }

        await #expect(throws: CancellationError.self) {
            try await task.value
        }
    }

    @Test func xlmRobertaUsesPaddingAwarePositionIDs() {
        let inputIDs = MLXArray([0, 10, 11, 1, 1]).reshaped(1, 5)

        let positionIDs = bertPositionIDs(
            inputIDs: inputIDs, padTokenID: 1, paddingAware: true)

        #expect(positionIDs.asArray(Int.self) == [2, 3, 4, 1, 1])
    }

    @Test func bertUsesSequentialPositionIDs() {
        let inputIDs = MLXArray([101, 10, 0]).reshaped(1, 3)

        let positionIDs = bertPositionIDs(
            inputIDs: inputIDs, padTokenID: 0, paddingAware: false)

        #expect(positionIDs.asArray(Int.self) == [0, 1, 2])
    }

    @Test func omittedTokenTypeIDsDoNotAddTypeZeroEmbeddings() throws {
        let configuration = try JSONDecoder().decode(
            BertConfiguration.self,
            from: bertConfigurationData(
                modelType: "bert",
                architecture: "BertModel",
                labels: 1,
                padTokenID: 0,
                maxPositionEmbeddings: 8,
                numLayers: 0))
        let model = BertModel(configuration)
        try model.update(
            parameters: ModuleParameters.unflattened([
                "embeddings.word_embeddings.weight": MLXArray.zeros([256, 8]),
                "embeddings.position_embeddings.weight": MLXArray.zeros([8, 8]),
                "embeddings.token_type_embeddings.weight": MLXArray([
                    Float(0), 1, 2, 3, 4, 5, 6, 7,
                ]).reshaped(1, 8),
                "embeddings.norm.weight": MLXArray.ones([8]),
                "embeddings.norm.bias": MLXArray.zeros([8]),
            ]),
            verify: [])

        let inputIDs = MLXArray([10, 11]).reshaped(1, 2)
        let withoutTokenTypes = try #require(model(inputIDs).hiddenStates)
        let withTypeZero = try #require(
            model(inputIDs, tokenTypeIds: MLXArray.zeros([1, 2], dtype: .int32))
                .hiddenStates)
        eval(withoutTokenTypes, withTypeZero)

        #expect(withoutTokenTypes.asArray(Float.self).allSatisfy { abs($0) < 0.0001 })
        #expect(
            zip(
                withoutTokenTypes.asArray(Float.self),
                withTypeZero.asArray(Float.self)
            ).contains { abs($0 - $1) > 0.0001 })
    }

    @Test func bgeConfigurationDerivesPaddingAndContextLimit() throws {
        let configuration = try decodeBertConfiguration(
            modelType: "xlm-roberta",
            architecture: "XLMRobertaForSequenceClassification",
            labels: 1,
            padTokenID: 1,
            maxPositionEmbeddings: 8_194)

        #expect(configuration.padTokenID == 1)
        #expect(configuration.usesPaddingAwarePositionIDs)
        #expect(configuration.encoderRerankerConfiguration.padTokenID == 1)
        #expect(configuration.encoderRerankerConfiguration.maxInputTokens == 8_192)
        #expect(configuration.encoderRerankerConfiguration.scoreKind == .normalizedRelevance)
    }

    @Test func encoderConfigurationUsesDeclaredPositiveLabel() throws {
        let configuration = try decodeBertConfiguration(
            modelType: "xlm-roberta",
            architecture: "XLMRobertaForSequenceClassification",
            labels: 2,
            padTokenID: 1,
            maxPositionEmbeddings: 32,
            idToLabel: [0: "irrelevant", 1: "relevant"])

        guard
            case .softmaxProbability(classIndex: 1)? =
                configuration.encoderRerankerConfiguration.scorePolicy
        else {
            Issue.record("Expected a positive-class softmax policy")
            return
        }
    }

    @Test func encoderConfigurationRejectsAmbiguousLabels() throws {
        let configuration = try decodeBertConfiguration(
            modelType: "bert",
            architecture: "BertForSequenceClassification",
            labels: 2,
            padTokenID: 0,
            maxPositionEmbeddings: 32)

        #expect(configuration.encoderRerankerConfiguration.scorePolicy == nil)
    }

    @Test func encoderConfigurationDecodesSparseLabelToIDMetadata() throws {
        let data = try bertConfigurationData(
            modelType: "bert",
            architecture: "BertForSequenceClassification",
            labels: 1,
            padTokenID: 0,
            maxPositionEmbeddings: 32,
            labelToID: ["irrelevant": 0, "relevant": 2],
            includeNumLabels: false)
        let configuration = try JSONDecoder().decode(BertConfiguration.self, from: data)

        #expect(configuration.numLabels == 3)
        #expect(
            configuration.encoderRerankerConfiguration.scorePolicy
                == .softmaxProbability(classIndex: 2))
    }

    @Test func sequenceClassificationConfigurationCreatesRerankerModel() async throws {
        let data = try bertConfigurationData(
            modelType: "xlm-roberta",
            architecture: "XLMRobertaForSequenceClassification",
            labels: 1,
            padTokenID: 1,
            maxPositionEmbeddings: 32)

        let model = try await EmbedderTypeRegistry.shared.createModel(
            configuration: data, modelType: "xlm-roberta")

        #expect(model is BertRerankerModel)
    }

    @Test func existingBertWeightPathsRemainStable() throws {
        let configuration = try decodeBertConfiguration(
            modelType: "bert",
            architecture: "BertModel",
            labels: 1,
            padTokenID: 0,
            maxPositionEmbeddings: 32)
        let model = BertModel(configuration)

        let weights = model.sanitize(weights: [
            "bert.embeddings.word_embeddings.weight": MLXArray.zeros([256, 8]),
            "bert.encoder.layer.0.output.dense.weight": MLXArray.zeros([8, 16]),
        ])

        #expect(weights["embeddings.word_embeddings.weight"] != nil)
        #expect(weights["encoder.layers.0.linear2.weight"] != nil)
        #expect(weights.keys.allSatisfy { !$0.hasPrefix("encoder_model.") })
    }

    @Test func encoderScoringUsesTokenBudgetedMicroBatches() async throws {
        let tokenizer = ByteRerankerTokenizer()
        let model = TestEncoderRerankerModel(
            configuration: .init(
                inputKind: .xlmRoberta,
                padTokenID: 1,
                maxInputTokens: 128,
                scorePolicy: .singleLogit(transform: .sigmoid),
                scoreKind: .normalizedRelevance))
        let container = EmbedderModelContainer(
            context: EmbedderModelContext(
                configuration: ModelConfiguration(id: "test/encoder"),
                model: model,
                tokenizer: tokenizer,
                pooling: Pooling(strategy: .none)))

        let scores = try await container.rerankerScores(
            query: "q",
            documents: ["a", "bbbb", "cc", "dddddd"],
            options: .init(maxBatchSize: 2, maxBatchTokens: 30))

        #expect(scores.count == 4)
        #expect(model.scoreCallCount == 2)
        #expect(scores.allSatisfy { $0 >= 0 && $0 <= 1 })
    }

    @Test func encoderTokenBudgetTruncatesOversizedSingleton() async throws {
        let model = TestEncoderRerankerModel(
            configuration: .init(
                inputKind: .xlmRoberta,
                padTokenID: 1,
                maxInputTokens: 128,
                scorePolicy: .singleLogit(transform: .sigmoid),
                scoreKind: .normalizedRelevance))
        let container = makeEmbedderContainer(model: model)

        _ = try await container.rerankerScores(
            query: "query",
            documents: [String(repeating: "d", count: 100)],
            options: .init(maxBatchSize: 8, maxBatchTokens: 32))

        #expect(model.scoredShapes == [[1, 32]])
    }

    @Test func encoderScoringRejectsAmbiguousOneDimensionalOutput() async throws {
        let model = TestEncoderRerankerModel(
            configuration: .init(
                inputKind: .xlmRoberta,
                padTokenID: 1,
                maxInputTokens: 128,
                scorePolicy: .singleLogit(transform: .identity),
                scoreKind: .logit),
            output: .oneDimensional)
        let container = makeEmbedderContainer(model: model)

        await #expect(throws: RerankerError.self) {
            try await container.rerankerScores(
                query: "q", documents: ["a", "b"], options: .init())
        }
    }

    @Test func singleLogitRejectsMultipleClassifierOutputs() async throws {
        let model = TestEncoderRerankerModel(
            configuration: .init(
                inputKind: .xlmRoberta,
                padTokenID: 1,
                maxInputTokens: 128,
                scorePolicy: .singleLogit(transform: .sigmoid),
                scoreKind: .normalizedRelevance),
            output: .twoLogits)
        let container = makeEmbedderContainer(model: model)

        await #expect(throws: RerankerError.self) {
            try await container.rerankerScores(
                query: "q", documents: ["a"], options: .init())
        }
    }

    @Test func softmaxProbabilityIsNotTransformedTwice() async throws {
        let model = TestEncoderRerankerModel(
            configuration: .init(
                inputKind: .xlmRoberta,
                padTokenID: 1,
                maxInputTokens: 128,
                scorePolicy: .softmaxProbability(classIndex: 1),
                scoreKind: .normalizedRelevance),
            output: .fixedTwoLogits)
        let container = makeEmbedderContainer(model: model)

        let scores = try await container.rerankerScores(
            query: "q", documents: ["a"], options: .init())

        #expect(abs(scores[0] - 0.880797) < 0.0001)
    }

    @Test func xlmRobertaPairMatchesReferenceTokenLayout() throws {
        let tokenizer = ByteRerankerTokenizer()
        let input = try XLMRobertaRerankerInputProcessor().encode(
            query: "q",
            document: "d",
            tokenizer: tokenizer,
            maxInputTokens: nil,
            truncation: .truncate)

        #expect(input.tokenIds == [3, byteID("q"), 4, 4, byteID("d"), 4])
        #expect(input.tokenTypeIds == [0, 0, 0, 0, 0, 0])
    }

    @Test func qwenPromptIsTokenIdenticalToReferenceTemplate() throws {
        let tokenizer = ByteRerankerTokenizer()
        let instruction = "Find useful passages"
        let input = try Qwen3RerankerInputProcessor(instruction: instruction).encode(
            query: "query",
            document: "document",
            tokenizer: tokenizer,
            maxInputTokens: nil,
            truncation: .truncate)
        let reference =
            "<|im_start|>system\n"
            + "Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n"
            + "<|im_start|>user\n"
            + "<Instruct>: \(instruction)\n<Query>: query\n<Document>: document"
            + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

        #expect(input.tokenIds == tokenizer.encode(text: reference, addSpecialTokens: false))
    }

    @Test func jinaPromptIsTokenIdenticalToReferenceImplementation() throws {
        let tokenizer = ByteRerankerTokenizer()
        let input = try JinaRerankerInputProcessor().encode(
            query: "query",
            documents: ["first", "second"],
            tokenizer: tokenizer,
            maxInputTokens: nil,
            truncation: .truncate)
        let reference = jinaReferencePrompt(
            query: "query", documents: ["first", "second"])

        #expect(input.tokenIds == tokenizer.encode(text: reference, addSpecialTokens: false))
    }

    @Test func jinaTruncationRetokenizesTheCompletePrompt() throws {
        let tokenizer = BoundaryMergingRerankerTokenizer()
        let document = String(repeating: "d", count: 200)
        let limit = tokenizer.encode(
            text: jinaReferencePrompt(query: "query", documents: [String(document.prefix(40))]),
            addSpecialTokens: false
        ).count

        let input = try JinaRerankerInputProcessor().encode(
            query: "query",
            documents: [document],
            tokenizer: tokenizer,
            maxInputTokens: limit,
            truncation: .truncate)
        let expected = tokenizer.encode(
            text: jinaReferencePrompt(query: "query", documents: [String(document.prefix(40))]),
            addSpecialTokens: false)

        #expect(input.tokenIds == expected)
        #expect(input.tokenIds.count <= limit)
    }

    @Test func truncationErrorRejectsOverlongPairs() throws {
        let tokenizer = ByteRerankerTokenizer()

        #expect(throws: RerankerError.self) {
            try XLMRobertaRerankerInputProcessor().encode(
                query: "query",
                document: String(repeating: "d", count: 100),
                tokenizer: tokenizer,
                maxInputTokens: 12,
                truncation: .error)
        }
    }

    @Test func jinaRejectsMoreThanSixtyFourDocuments() async throws {
        let tokenizer = ByteRerankerTokenizer()
        let model = TestCausalRerankerModel(
            trueTokenID: tokenizer.trueTokenID,
            falseTokenID: tokenizer.falseTokenID)
        let container = makeModelContainer(model: model, tokenizer: tokenizer)

        await #expect(throws: RerankerError.self) {
            try await container.listwiseRerankerScores(
                query: "q",
                documents: Array(repeating: "d", count: 65),
                instruction: nil,
                maxInputTokens: 131_072,
                family: .v3,
                options: .init())
        }
    }

    @Test func jinaListwisePromptHonorsForwardPassTokenBudget() async throws {
        let tokenizer = ByteRerankerTokenizer()
        let model = TestCausalRerankerModel(
            trueTokenID: tokenizer.trueTokenID,
            falseTokenID: tokenizer.falseTokenID)
        let container = makeModelContainer(model: model, tokenizer: tokenizer)

        let scores = try await container.listwiseRerankerScores(
            query: "query",
            documents: [String(repeating: "d", count: 1_000)],
            instruction: nil,
            maxInputTokens: 131_072,
            family: .v3,
            options: .init(maxBatchTokens: 800))

        #expect(scores.count == 1)
        #expect(model.listwiseTokenCount == 800)
    }

    @Test func jinaLanguageModelOutputUsesVocabularyDimension() throws {
        let configuration = try decodeQwenConfiguration()
        let model = JinaRerankerModel(configuration)

        let output = model(
            MLXArray([1, 2, 3]).reshaped(1, 3),
            cache: Optional<[KVCache]>.none)

        #expect(output.shape == [1, 3, 128])
    }

    @Test func jinaSanitizeAcceptsBothPackagingsOfTheProjector() throws {
        let configuration = try decodeQwenConfiguration()
        let model = JinaRerankerModel(configuration)
        let linear1 = MLXArray.zeros([512, 1024])
        let linear2 = MLXArray.zeros([512, 512])

        // jinaai/jina-reranker-v3: nn.Sequential(Linear, ReLU, Linear) in `model.safetensors`,
        // so the layers are numbered by position
        let fromSource = model.sanitize(weights: [
            "model.embed_tokens.weight": MLXArray.zeros([151_936, 1024]),
            "projector.0.weight": linear1,
            "projector.2.weight": linear2,
        ])

        // jinaai/jina-reranker-v3-mlx: the same layers renamed, in `projector.safetensors`,
        // without the `projector` prefix
        let fromMLX = model.sanitize(weights: [
            "model.embed_tokens.weight": MLXArray.zeros([151_936, 1024]),
            "linear1.weight": linear1,
            "linear2.weight": linear2,
        ])

        for weights in [fromSource, fromMLX] {
            #expect(weights["projector.linear1.weight"]?.shape == [512, 1024])
            #expect(weights["projector.linear2.weight"]?.shape == [512, 512])
            #expect(weights["model.embed_tokens.weight"] != nil)
            #expect(weights.count == 3, "sanitize must not leave the original spelling behind")
        }
    }

    @Test func jinaDeclaresTheProjectorSidecarNoConventionSelects() throws {
        let configuration = try decodeQwenConfiguration()
        let model = JinaRerankerModel(configuration)

        // `jinaai/jina-reranker-v3-mlx` keeps the projector in `projector.safetensors`, which
        // neither `model*.safetensors` nor its own index selects, so the head is skipped unless
        // the model asks for the file by name (#560). Check the conformance, since that is what
        // `loadWeights` looks for.
        let provider = model as any AdditionalWeightFilesProviding
        #expect(provider.additionalWeightFiles == ["projector.safetensors"])
    }

    @Test func jinaCosineSimilarityRemainsInDeclaredRange() {
        let vector = MLXArray([Float(0.1), 0.2, 0.3]).reshaped(1, 3)
        let scores = jinaCosineSimilarity(vector, vector).asArray(Float.self)

        #expect(scores.count == 1)
        #expect(scores[0] >= -1)
        #expect(scores[0] <= 1)
    }

    @Test(arguments: [DType.bfloat16, .float16, .float32])
    func jinaCosineSimilarityUsesFloat32Reductions(dtype: DType) {
        let query = MLXArray([Float(1), 2, 3]).reshaped(1, 3).asType(dtype)
        let documents = MLXArray([Float(3), 2, 1, -3, -2, -1, 0, 0, 0])
            .reshaped(3, 3).asType(dtype)

        let output = jinaCosineSimilarity(documents, query)
        let scores = output.asArray(Float.self)

        #expect(output.dtype == .float32)
        #expect(abs(scores[0] - 10.0 / 14.0) < 1e-6)
        #expect(abs(scores[1] + 10.0 / 14.0) < 1e-6)
        #expect(scores[2] == 0)
    }

    @Test(arguments: [Float(1e-5), 1e3])
    func jinaCosineSimilarityAvoidsFloat16UnderflowAndOverflow(scale: Float) {
        let vector = (MLXArray([Float(1), 2, 3]) * scale).reshaped(1, 3).asType(.float16)
        let score = jinaCosineSimilarity(vector, vector).item(Float.self)

        #expect(score.isFinite)
        #expect(abs(score - 1) < 1e-6)
    }

    @Test func causalScoringUsesMicroBatchesAndReturnsNormalizedRelevance() async throws {
        let tokenizer = ByteRerankerTokenizer()
        let model = TestCausalRerankerModel(
            trueTokenID: tokenizer.trueTokenID,
            falseTokenID: tokenizer.falseTokenID)
        let container = makeModelContainer(model: model, tokenizer: tokenizer)

        let scores = try await container.causalRerankerScores(
            query: "q",
            documents: ["a", "bbbb", "cc"],
            instruction: nil,
            maxInputTokens: 8_192,
            family: .qwen3, scorePolicy: try qwenScorePolicy(tokenizer),
            options: .init(maxBatchSize: 2, maxBatchTokens: 4_096))

        #expect(scores.count == 3)
        #expect(model.callCount == 2)
        #expect(scores.allSatisfy { $0 >= 0 && $0 <= 1 })

        let singletonScores = try await container.causalRerankerScores(
            query: "q",
            documents: ["a", "bbbb", "cc"],
            instruction: nil,
            maxInputTokens: 8_192,
            family: .qwen3, scorePolicy: try qwenScorePolicy(tokenizer),
            options: .init(maxBatchSize: 1, maxBatchTokens: 4_096))
        #expect(scores == singletonScores)
    }

    @Test func causalTokenBudgetTruncatesOversizedSingleton() async throws {
        let tokenizer = ByteRerankerTokenizer()
        let model = TestCausalRerankerModel(
            trueTokenID: tokenizer.trueTokenID,
            falseTokenID: tokenizer.falseTokenID)
        let container = makeModelContainer(model: model, tokenizer: tokenizer)

        _ = try await container.causalRerankerScores(
            query: "query",
            documents: [String(repeating: "d", count: 1_000)],
            instruction: nil,
            maxInputTokens: 8_192,
            family: .qwen3, scorePolicy: try qwenScorePolicy(tokenizer),
            options: .init(maxBatchTokens: 512))

        #expect(model.callShapes == [[1, 512]])
    }

    @Test(arguments: [false, true], [false, true])
    func qwenFinalTokenLogitsMatchFullProjection(tied: Bool, quantized: Bool) throws {
        let configuration = try JSONDecoder().decode(
            MLXLLM.Qwen3Configuration.self,
            from: Data(
                """
                {
                  "vocab_size": 128, "hidden_size": 64, "num_hidden_layers": 2,
                  "intermediate_size": 128, "num_attention_heads": 4,
                  "num_key_value_heads": 2, "head_dim": 16, "rms_norm_eps": 1e-6,
                  "tie_word_embeddings": \(tied)
                }
                """.utf8))

        for dtype: DType in [.float32, .float16] {
            let model = withRandomState(MLXRandom.RandomState(seed: 42)) {
                MLXLLM.Qwen3Model(configuration)
            }
            model.apply { $0.dtype.isFloatingPoint ? $0.asType(dtype) : $0 }
            if quantized {
                quantize(model: model, groupSize: 32, bits: 4)
            }
            for lengths in [[1], [1, 3, 8], [8, 4, 8]] {
                let width = lengths.max() ?? 1
                let rows = lengths.enumerated().map { row, length in
                    (0 ..< width).map { column in
                        column < length ? (row * 13 + column * 7 + 1) % 128 : 0
                    }
                }
                let tokens = MLXArray(rows.flatMap { $0 }).reshaped(lengths.count, width)
                let full = model(tokens, cache: nil)
                let expected = stacked(
                    lengths.enumerated().map { row, length in full[row, length - 1] })
                let hidden = model.hiddenStates(tokens, cache: nil)
                let actual = model.projectLogits(
                    stacked(lengths.enumerated().map { row, length in hidden[row, length - 1] }))

                #expect(actual.shape == [lengths.count, 128])
                let tolerance: Float = dtype == .float16 ? 0.02 : 0.0001
                #expect(abs(actual - expected).max().item(Float.self) < tolerance)
            }
        }
    }

    @Test func causalScoringProjectsFinalTokensAndKeepsSingletonPrefill() async throws {
        let tokenizer = ByteRerankerTokenizer()
        let container = makeModelContainer(
            model: TestHiddenStateRerankerModel(tokenizer: tokenizer), tokenizer: tokenizer)
        let reference = makeModelContainer(
            model: TestCausalRerankerModel(
                trueTokenID: tokenizer.trueTokenID, falseTokenID: tokenizer.falseTokenID),
            tokenizer: tokenizer)
        let documents = ["bbbb", "a", String(repeating: "c", count: 1_000)]
        let options = RerankExecutionOptions(maxBatchSize: 2, maxBatchTokens: 1_024)

        let actual = try await container.causalRerankerScores(
            query: "q", documents: documents, instruction: nil,
            maxInputTokens: 512,
            family: .qwen3, scorePolicy: try qwenScorePolicy(tokenizer), options: options)
        let expected = try await reference.causalRerankerScores(
            query: "q", documents: documents, instruction: nil,
            maxInputTokens: 512,
            family: .qwen3, scorePolicy: try qwenScorePolicy(tokenizer), options: options)
        let calls = try await container.perform { context in
            let model = try #require(context.model as? TestHiddenStateRerankerModel)
            return (model.hiddenStateShapes, model.projectedShapes, model.fullForwardShapes)
        }

        #expect(actual == expected)
        #expect(calls.0.map(\.first) == [2])
        #expect(calls.1 == [[2, 128]])
        #expect(calls.2 == [[1, 512]])
    }

    @Test func jinaFactoryRegistrationUsesArchitecture() async throws {
        let model = try await LLMTypeRegistry.shared.createModel(
            configuration: try jinaConfigurationData(), modelType: "qwen3")

        #expect(model is JinaRerankerModel)
    }

    @Test func rerankerFactoryRoutesOnlyVerifiedRerankers() throws {
        let bge = try JSONDecoder().decode(
            RerankerDescriptor.self,
            from: bertConfigurationData(
                modelType: "xlm-roberta",
                architecture: "XLMRobertaForSequenceClassification",
                labels: 1,
                padTokenID: 1,
                maxPositionEmbeddings: 8_194))
        let jina = try JSONDecoder().decode(
            RerankerDescriptor.self, from: jinaConfigurationData())
        let qwen = try JSONDecoder().decode(
            RerankerDescriptor.self,
            from: Data(
                """
                {
                  "model_type": "qwen3",
                  "architectures": ["Qwen3ForCausalLM"],
                  "max_position_embeddings": 32768
                }
                """.utf8))
        let unsupported = try JSONDecoder().decode(
            RerankerDescriptor.self,
            from: Data(
                """
                {
                  "model_type": "llama",
                  "architectures": ["LlamaForCausalLM"]
                }
                """.utf8))

        #expect(
            try bge.architecture(modelID: "BAAI/bge-reranker-v2-m3") == .encoder)
        #expect(try jina.architecture(modelID: "jinaai/jina-reranker-v3-mlx") == .jina(.v3))
        #expect(
            try qwen.architecture(modelID: "Qwen/Qwen3-Reranker-0.6B") == .causal(.qwen3))
        #expect(try unsupported.architecture(modelID: "test/model") == nil)

        #expect(throws: RerankerError.self) {
            try qwen.architecture(modelID: "Qwen/Qwen3-0.6B")
        }
        #expect(throws: RerankerError.self) {
            try bge.architecture(modelID: "example/sentiment-classifier")
        }
        #expect(
            try qwen.architecture(
                modelID: "lampo/private-model", allowUnverifiedModel: true) == .causal(.qwen3))
    }
    @Test(arguments: [CausalRerankerFamily.zerank2, .contextual])
    func rawLogitPromptsMatchTheirReference(family: CausalRerankerFamily) throws {
        let tokenizer = ByteRerankerTokenizer()
        let processor = family.inputProcessor(instruction: nil)
        let input = try processor.encode(
            query: "query", document: "document",
            tokenizer: tokenizer, maxInputTokens: nil, truncation: .error)
        let reference =
            family == .zerank2
            ? "<|im_start|>system\nquery<|im_end|>\n<|im_start|>user\ndocument<|im_end|>\n<|im_start|>assistant\n"
            : "Check whether a given document contains information helpful to answer the query.\n<Document> document\n<Query> query ??"
        #expect(input.tokenIds == tokenizer.encode(text: reference, addSpecialTokens: false))
        let instructed = try family.inputProcessor(instruction: "Prefer recent sources")
            .encode(
                query: "query", document: "document",
                tokenizer: tokenizer, maxInputTokens: nil, truncation: .error)
        let instructedReference =
            family == .zerank2
            ? reference
            : "Check whether a given document contains information helpful to answer the query.\n<Document> document\n<Query> query Prefer recent sources ??"
        #expect(
            instructed.tokenIds
                == tokenizer.encode(text: instructedReference, addSpecialTokens: false))
        let truncated = try processor.encode(
            query: "query", document: String(repeating: "d", count: 1_000),
            tokenizer: tokenizer, maxInputTokens: 160, truncation: .truncate)
        #expect(truncated.tokenIds.count <= 160)
        #expect(
            tokenizer.decode(tokenIds: truncated.tokenIds).hasSuffix(
                family == .zerank2 ? "assistant\n" : " ??"))
        #expect(throws: RerankerError.self) {
            try processor.encode(
                query: "query", document: String(repeating: "d", count: 1_000),
                tokenizer: tokenizer, maxInputTokens: 160, truncation: .error)
        }
    }

    @Test func causalMetadataSelectsScoresWithoutGuessingYesAndNo() throws {
        let tokenizer = ByteRerankerTokenizer()
        let qwen = try CausalRerankerFamily.qwen3.scorePolicy(
            metadata: causalMetadata(trueID: 7, falseID: 8), tokenizer: tokenizer,
            vocabularySize: 128)
        #expect(qwen == .binaryMargin(positive: 7, negative: 8))
        let raw = try CausalRerankerFamily.zerank2.scorePolicy(
            metadata: causalMetadata(trueID: 7), tokenizer: tokenizer, vocabularySize: 128)
        #expect(CausalRerankerFamily.zerank2.scoreKind == .logit)
        let values = MLXArray([Float(-9), 3, 4, 5, 6, 7, 8, -2, 0])
        #expect(raw(values) == [-2])
        #expect(abs(qwen(values)[0] - 1 / (1 + exp(2))) < 1e-7)
        for metadata in [
            try causalMetadata(trueID: 7), try causalMetadata(trueID: 7, falseID: 7),
            try causalMetadata(trueID: -1, falseID: 8), try causalMetadata(trueID: 128, falseID: 8),
        ] {
            #expect(throws: RerankerError.self) {
                try CausalRerankerFamily.qwen3.scorePolicy(
                    metadata: metadata, tokenizer: tokenizer, vocabularySize: 128)
            }
        }
        #expect(throws: RerankerError.self) {
            try CausalRerankerFamily.contextual.scorePolicy(
                metadata: causalMetadata(trueID: 0, falseID: 2), tokenizer: tokenizer,
                vocabularySize: 128)
        }
        let contextual = try CausalRerankerFamily.contextual.scorePolicy(
            metadata: nil, tokenizer: tokenizer, vocabularySize: 128)
        #expect(contextual == .logit(tokenID: 0, roundsToBFloat16: true))
    }

    @Test func scorePolicyReadsEveryRowAtOnce() {
        let logits = MLXArray([Float(0), 1, 3, 0, 4, 1]).reshaped(2, 3)
        let margin = CausalRerankerScorePolicy.binaryMargin(positive: 2, negative: 1)
        let expected = [2.0, -3.0].map { 1 / (1 + exp(-$0)) }
        #expect(zip(margin(logits), expected).allSatisfy { abs($0 - $1) < 1e-12 })
        let logit = CausalRerankerScorePolicy.logit(tokenID: 0, roundsToBFloat16: false)
        #expect(logit(logits) == [0, 0])
        #expect(logit(logits.asType(.float16)) == [0, 0])
    }

    @Test func factoryDistinguishesSupportedProtocolsAndRejectsUnknownRerankers() throws {
        let descriptor = try JSONDecoder().decode(
            RerankerDescriptor.self,
            from: Data("{\"model_type\":\"qwen3\",\"architectures\":[\"Qwen3ForCausalLM\"]}".utf8))
        #expect(
            try descriptor.architecture(modelID: "zeroentropy/zerank-2-reranker")
                == .causal(.zerank2))
        #expect(
            try descriptor.architecture(modelID: "zeroentropy/zerank-2")
                == .causal(.zerank2))
        #expect(
            try descriptor.architecture(
                modelID: "ContextualAI/ctxl-rerank-v2-instruct-multilingual-1b")
                == .causal(.contextual))
        #expect(throws: RerankerError.self) {
            try descriptor.architecture(modelID: "example/another-reranker")
        }
        let jina = try JSONDecoder().decode(RerankerDescriptor.self, from: jinaConfigurationData())
        #expect(try jina.architecture(modelID: "jinaai/jina-reranker-v3.5-mlx") == .jina(.v35))
        #expect(throws: RerankerError.self) {
            try jina.architecture(modelID: "jinaai/jina-reranker-v3.50")
        }
    }

    @Test func jinaV35PromptMatchesDualMatchingReference() throws {
        let tokenizer = ByteRerankerTokenizer()
        let input = try JinaRerankerInputProcessor(family: .v35).encode(
            query: "query", documents: ["document"],
            tokenizer: tokenizer, maxInputTokens: nil, truncation: .error)
        let reference = jinaReferencePrompt(query: "query", documents: ["document"])
            .replacingOccurrences(
                of: "relevance to query: query\n",
                with: "relevance to query: query<|rerank_token|>\n"
            )
            .replacingOccurrences(
                of: "</query><|im_end|>",
                with:
                    "</query>\nPlease provide the ranking of all passages based on their relevance to the search query, in descending order of relevance, with each label enclosed in square brackets (e.g., [2] > [1] > [3] > [0]).<|im_end|>"
            )
        #expect(input.tokenIds == tokenizer.encode(text: reference, addSpecialTokens: false))
        #expect(input.markerTokenIds?.queryCount == 2)
        #expect(input.tokenIds.filter { $0 == tokenizer.rerankTokenID }.count == 2)
    }

    @Test func jinaV35FusesQueryEmbeddingsAcrossBlocks() throws {
        let blocks = [
            JinaRerankerEmbeddings(
                documents: MLXArray([Float(1), 0]).reshaped(1, 2),
                query: MLXArray([Float(1), 0]).reshaped(1, 2)),
            JinaRerankerEmbeddings(
                documents: MLXArray([Float(1), 0]).reshaped(1, 2),
                query: MLXArray([Float(0), 1]).reshaped(1, 2)),
        ]
        let scores = jinaFusedScores(blocks)
        #expect(scores.count == 2)
        #expect(scores.allSatisfy { abs($0 - 2 / sqrt(5.0)) < 1e-6 })
        let single = jinaFusedScores([blocks[1]])
        #expect(single == [0])
    }

    @Test func jinaV35BlocksFollowTheReferenceGeometry() throws {
        let tokenizer = ByteRerankerTokenizer()
        let blocking = try #require(JinaRerankerFamily.v35.blocking)
        let short = try blocking.blocks(
            query: "q", documents: Array(repeating: "doc", count: 3), tokenizer: tokenizer,
            blockTokens: 131_072, truncation: .error)
        #expect(short.documents.map(\.count) == [3])
        let long = String(repeating: "d", count: 9_000)
        let packed = try blocking.blocks(
            query: "q", documents: Array(repeating: long, count: 20), tokenizer: tokenizer,
            blockTokens: 131_072, truncation: .truncate)
        // The block closes at 1 + 8_191 * 16 tokens, past 131_072 - 8_192.
        #expect(packed.documents.map(\.count) == [16, 4])
        #expect(packed.documents[0][0].count == 8_191)
        #expect(throws: RerankerError.self) {
            try blocking.blocks(
                query: "q", documents: [long], tokenizer: tokenizer,
                blockTokens: 131_072, truncation: .error)
        }
    }

    @Test(arguments: [1, 3, 7, 1_000])
    func jinaChunkedPrefillMatchesOneForwardPass(stepSize: Int) throws {
        var object = try #require(
            JSONSerialization.jsonObject(with: jinaConfigurationData()) as? [String: Any])
        // Byte tokens reach 137, beyond the fixture's 128-entry vocabulary.
        object["vocab_size"] = 256
        object["num_hidden_layers"] = 2
        object["layer_types"] = ["sliding_attention", "full_attention"]
        object["sliding_window"] = 4
        object["use_sliding_window"] = true
        let configuration = try JSONDecoder().decode(
            MLXLLM.Qwen3Configuration.self, from: JSONSerialization.data(withJSONObject: object))
        let model = withRandomState(MLXRandom.RandomState(seed: 7)) {
            JinaRerankerModel(configuration)
        }
        let tokenizer = ByteRerankerTokenizer()
        let input = try JinaRerankerInputProcessor(family: .v35).encode(
            query: "query", documents: ["first passage", "second"], tokenizer: tokenizer,
            maxInputTokens: nil, truncation: .error)
        let reference = try model.embeddings(input: input, documentCount: 2, stepSize: 1_000_000)
        let chunked = try model.embeddings(input: input, documentCount: 2, stepSize: stepSize)

        #expect(chunked.documents.shape == [2, 512])
        #expect(chunked.query.shape == [1, 512])
        #expect(abs(chunked.documents - reference.documents).max().item(Float.self) < 1e-4)
        #expect(abs(chunked.query - reference.query).max().item(Float.self) < 1e-4)
    }

    @Test func qwenSlidingAttentionMatchesAnIndependentBandedMask() throws {
        let configuration = try qwenAttentionConfiguration(types: ["sliding_attention"], window: 4)
        let model = withRandomState(MLXRandom.RandomState(seed: 42)) {
            MLXLLM.Qwen3Model(configuration)
        }
        let block = MLXLLM.Qwen3TransformerBlock(configuration)
        let prefix = "model.layers.0."
        let weights = Dictionary(
            uniqueKeysWithValues: model.parameters().flattened().compactMap { key, value in
                key.hasPrefix(prefix) ? (String(key.dropFirst(prefix.count)), value) : nil
            })
        try block.update(parameters: ModuleParameters.unflattened(weights))
        for length in [3, 4, 5, 9] {
            let tokens = MLXArray(Array(1 ... length)).reshaped(1, length)
            let mask = MLXArray(
                (0 ..< length).flatMap { row in
                    (0 ..< length).map { column in column <= row && column > row - 4 }
                }
            ).reshaped(length, length)
            let expected = model.model.norm(
                block(model.model.embedTokens(tokens), mask: .array(mask), cache: nil))
            let actual = model.hiddenStates(tokens, cache: nil)
            #expect(abs(actual - expected).max().item(Float.self) < 1e-5)
        }
    }

    @Test func qwenMixedAttentionCachedPrefillMatchesUncachedForward() throws {
        let configuration = try qwenAttentionConfiguration(
            types: ["sliding_attention", "full_attention"], window: 4)
        let model = withRandomState(MLXRandom.RandomState(seed: 42)) {
            MLXLLM.Qwen3Model(configuration)
        }
        let tokens = MLXArray(Array(1 ... 13)).reshaped(1, 13)
        let expected = model(tokens, cache: nil)
        let cache = try model.newCache(parameters: nil)
        #expect(cache[0] is RotatingKVCache)
        #expect(cache[1] is KVCacheSimple)
        for range in [0 ..< 3, 3 ..< 8, 8 ..< 11, 11 ..< 12, 12 ..< 13] {
            let actual = model(tokens[0..., range], cache: cache)
            #expect(abs(actual - expected[0..., range]).max().item(Float.self) < 1e-4)
        }
    }

    @Test func qwenRejectsInvalidAttentionDeclarationsAndPreservesRoundTrips() throws {
        #expect(throws: ModelFactoryError.self) {
            try qwenAttentionConfiguration(types: ["sliding_attention"], window: nil)
        }
        #expect(throws: ModelFactoryError.self) {
            try qwenAttentionConfiguration(types: ["full_attention"], window: 0, layers: 2)
        }
        #expect(throws: DecodingError.self) {
            try qwenAttentionConfiguration(types: ["future_attention"], window: 4)
        }
        let configuration = try qwenAttentionConfiguration(
            types: ["sliding_attention", "full_attention"], window: 4)
        let decoded = try JSONDecoder().decode(
            MLXLLM.Qwen3Configuration.self, from: JSONEncoder().encode(configuration))
        #expect(decoded.layerTypes == configuration.layerTypes)
        #expect(decoded.slidingWindow == 4)
    }

    @Test(arguments: [false, true])
    func factoryRejectsContradictoryMetadataBeforeLoadingWeights(remote: Bool) async throws {
        let directory = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
            .appending(path: "Qwen3-Reranker-test")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory.deletingLastPathComponent()) }
        try Data(
            "{\"model_type\":\"qwen3\",\"architectures\":[\"Qwen3ForCausalLM\"],\"vocab_size\":128}"
                .utf8
        )
        .write(to: directory.appending(path: "config.json"))
        let metadataDirectory = directory.appending(path: "1_LogitScore")
        try FileManager.default.createDirectory(
            at: metadataDirectory, withIntermediateDirectories: true)
        for metadata in [
            "{\"true_token_id\":0,\"false_token_id\":null}",
            "{\"true_token_id\":128,\"false_token_id\":1}",
        ] {
            try Data(metadata.utf8).write(to: metadataDirectory.appending(path: "config.json"))
            await #expect(throws: RerankerError.self) {
                if remote {
                    _ = try await RerankerModelFactory.shared.loadContainer(
                        from: FixtureRerankerDownloader(directory: directory),
                        using: FixtureRerankerTokenizerLoader(), id: "test/Qwen3-Reranker-test",
                        allowUnverifiedModel: true)
                } else {
                    _ = try await RerankerModelFactory.shared.loadContainer(
                        from: directory,
                        using: FixtureRerankerTokenizerLoader(), allowUnverifiedModel: true)
                }
            }
        }
        try Data("{\"true_token_id\":\"invalid\"}".utf8).write(
            to: metadataDirectory.appending(path: "config.json"))
        await #expect(throws: ModelFactoryError.self) {
            _ = try await RerankerModelFactory.shared.loadContainer(
                from: directory, using: FixtureRerankerTokenizerLoader())
        }
    }

    @Test(arguments: ["zerank-2-reranker", "ctxl-rerank-v2-instruct-multilingual-1b"])
    func rawLogitFactoryLoadsAndScoresTinyCheckpoint(name: String) async throws {
        let directory = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
            .appending(path: name)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory.deletingLastPathComponent()) }
        var object = try #require(
            JSONSerialization.jsonObject(with: jinaConfigurationData()) as? [String: Any])
        object["architectures"] = ["Qwen3ForCausalLM"]
        object["vocab_size"] = 256
        let data = try JSONSerialization.data(withJSONObject: object)
        try data.write(to: directory.appending(path: "config.json"))
        let configuration = try JSONDecoder().decode(MLXLLM.Qwen3Configuration.self, from: data)
        let model = withRandomState(MLXRandom.RandomState(seed: 42)) {
            MLXLLM.Qwen3Model(configuration)
        }
        try save(
            arrays: Dictionary(uniqueKeysWithValues: model.parameters().flattened()),
            url: directory.appending(path: "model.safetensors"))
        let metadataDirectory = directory.appending(path: "1_LogitScore")
        try FileManager.default.createDirectory(
            at: metadataDirectory, withIntermediateDirectories: true)
        let classifierID = name == "zerank-2-reranker" ? 7 : 0
        try JSONSerialization.data(withJSONObject: [
            "true_token_id": classifierID, "false_token_id": NSNull(),
        ])
        .write(to: metadataDirectory.appending(path: "config.json"))
        let reranker = try await RerankerModelFactory.shared.loadContainer(
            from: directory, using: FixtureRerankerTokenizerLoader())
        #expect(reranker.scoreKind == .logit)
        let response = try await reranker.scores(query: "query", documents: ["a", "bbbb"])
        let singleton = try await reranker.scores(
            query: "query", documents: ["a", "bbbb"], options: .init(maxBatchSize: 1))
        #expect(response.results.count == 2)
        for (actual, sequential) in zip(response.results, singleton.results) {
            #expect(abs(actual.score - sequential.score) < 1e-5)
            let document = actual.index == 0 ? "a" : "bbbb"
            let prompt =
                name == "zerank-2-reranker"
                ? "<|im_start|>system\nquery<|im_end|>\n<|im_start|>user\n\(document)<|im_end|>\n<|im_start|>assistant\n"
                : "Check whether a given document contains information helpful to answer the query.\n<Document> \(document)\n<Query> query ??"
            let tokens = ByteRerankerTokenizer().encode(text: prompt, addSpecialTokens: false)
            let logit = model(MLXArray(tokens).reshaped(1, -1), cache: nil)[0, -1, classifierID]
            let expected = name == "zerank-2-reranker" ? logit : logit.asType(.bfloat16)
            #expect(abs(actual.score - Double(expected.item(Float.self))) < 1e-5)
        }
    }

    @Test func jinaV35ScoresMoreThanOneBlockInOriginalOrder() async throws {
        let tokenizer = ByteRerankerTokenizer()
        let model = TestCausalRerankerModel(trueTokenID: 1, falseTokenID: 2)
        let container = makeModelContainer(model: model, tokenizer: tokenizer)
        let scores = try await container.listwiseRerankerScores(
            query: "q", documents: Array(repeating: "d", count: 126),
            instruction: nil, maxInputTokens: 131_072, family: .v35, options: .init())
        #expect(model.listwiseBlockCounts == [125, 1])
        #expect(scores.count == 126)
        #expect(scores.allSatisfy { abs($0 - 2 / sqrt(5.0)) < 1e-6 })
    }

    @Test func qwenSlidingConfigurationFollowsTransformers() throws {
        var object = try #require(
            JSONSerialization.jsonObject(with: jinaConfigurationData()) as? [String: Any])
        object["num_hidden_layers"] = 3
        object["use_sliding_window"] = true
        object["sliding_window"] = 4
        object["max_window_layers"] = 2
        func decode() throws -> MLXLLM.Qwen3Configuration {
            try JSONDecoder().decode(
                MLXLLM.Qwen3Configuration.self,
                from: JSONSerialization.data(withJSONObject: object))
        }
        // `max_window_layers` counts the bottom full-attention layers.
        #expect(try decode().layerTypes == [.fullAttention, .fullAttention, .slidingAttention])
        object["layer_types"] = ["sliding_attention", "full_attention", "sliding_attention"]
        #expect(try decode().layerTypes == [.slidingAttention, .fullAttention, .slidingAttention])
        object["use_sliding_window"] = false
        #expect(try decode().layerTypes == [.fullAttention, .fullAttention, .fullAttention])
        object["use_sliding_window"] = true
        object["layer_types"] = nil
        object["sliding_window"] = NSNull()
        #expect(try decode().layerTypes == [.fullAttention, .fullAttention, .fullAttention])
    }

    @Test func explicitFamilyLoadsRenamedCheckpointsAndRejectsMismatches() throws {
        let causal = try JSONDecoder().decode(
            RerankerDescriptor.self,
            from: Data("{\"model_type\":\"qwen3\",\"architectures\":[\"Qwen3ForCausalLM\"]}".utf8))
        let jina = try JSONDecoder().decode(RerankerDescriptor.self, from: jinaConfigurationData())
        var object = try #require(
            JSONSerialization.jsonObject(with: jinaConfigurationData()) as? [String: Any])
        object["use_sliding_window"] = true
        let slidingJina = try JSONDecoder().decode(
            RerankerDescriptor.self, from: JSONSerialization.data(withJSONObject: object))

        #expect(
            try causal.architecture(modelID: "lampo/finetune", family: .zerank2)
                == .causal(.zerank2))
        #expect(
            try causal.architecture(modelID: "mlx-community/zerank-2-4bit") == .causal(.zerank2))
        #expect(
            try jina.architecture(modelID: "lampo/listwise", family: .jinaV35) == .jina(.v35))
        #expect(throws: RerankerError.self) {
            try causal.architecture(modelID: "zeroentropy/zerank-2", family: .jinaV35)
        }
        #expect(throws: RerankerError.self) {
            try jina.architecture(modelID: "lampo/listwise")
        }
        #expect(
            try jina.architecture(modelID: "lampo/listwise", allowUnverifiedModel: true)
                == .jina(.v3))
        #expect(
            try slidingJina.architecture(modelID: "lampo/listwise", allowUnverifiedModel: true)
                == .jina(.v35))
    }
}

private func makeConstantReranker(
    scoreKind: RerankScoreKind,
    scores: [Double]
) -> RerankerContainer {
    RerankerContainer(modelID: "test/model", scoreKind: scoreKind) { _, _, _, _ in scores }
}

private func makeEmbedderContainer(
    model: TestEncoderRerankerModel
) -> EmbedderModelContainer {
    EmbedderModelContainer(
        context: EmbedderModelContext(
            configuration: ModelConfiguration(id: "test/encoder"),
            model: model,
            tokenizer: ByteRerankerTokenizer(),
            pooling: Pooling(strategy: .none)))
}

private func makeModelContainer(
    model: any LanguageModel,
    tokenizer: any Tokenizer
) -> ModelContainer {
    ModelContainer(
        context: ModelContext(
            configuration: ModelConfiguration(id: "test/reranker"),
            model: model,
            processor: TestInputProcessor(tokenizer: tokenizer),
            tokenizer: tokenizer))
}

private func byteID(_ character: Character) -> Int {
    Int(character.asciiValue ?? 0) + 10
}

private func bertConfigurationData(
    modelType: String,
    architecture: String,
    labels: Int,
    padTokenID: Int,
    maxPositionEmbeddings: Int,
    idToLabel: [Int: String] = [:],
    labelToID: [String: Int] = [:],
    includeNumLabels: Bool = true,
    numLayers: Int = 1
) throws -> Data {
    var configuration: [String: Any] = [
        "model_type": modelType,
        "architectures": [architecture],
        "pad_token_id": padTokenID,
        "vocab_size": 256,
        "hidden_size": 8,
        "num_attention_heads": 2,
        "intermediate_size": 16,
        "num_hidden_layers": numLayers,
        "type_vocab_size": 1,
        "max_position_embeddings": maxPositionEmbeddings,
    ]
    if includeNumLabels {
        configuration["num_labels"] = labels
    }
    if !idToLabel.isEmpty {
        configuration["id2label"] = Dictionary(
            uniqueKeysWithValues: idToLabel.map { (String($0.key), $0.value) })
    }
    if !labelToID.isEmpty {
        configuration["label2id"] = labelToID
    }
    return try JSONSerialization.data(withJSONObject: configuration, options: [.sortedKeys])
}

private func decodeBertConfiguration(
    modelType: String,
    architecture: String,
    labels: Int,
    padTokenID: Int,
    maxPositionEmbeddings: Int,
    idToLabel: [Int: String] = [:]
) throws -> BertConfiguration {
    try JSONDecoder().decode(
        BertConfiguration.self,
        from: bertConfigurationData(
            modelType: modelType,
            architecture: architecture,
            labels: labels,
            padTokenID: padTokenID,
            maxPositionEmbeddings: maxPositionEmbeddings,
            idToLabel: idToLabel))
}

private func decodeQwenConfiguration() throws -> MLXLLM.Qwen3Configuration {
    try JSONDecoder().decode(
        MLXLLM.Qwen3Configuration.self, from: jinaConfigurationData())
}

private func jinaConfigurationData() throws -> Data {
    Data(
        """
        {
          "model_type": "qwen3",
          "architectures": ["JinaForRanking"],
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
        """.utf8)
}

private func jinaReferencePrompt(query: String, documents: [String]) -> String {
    "<|im_start|>system\n"
        + "You are a search relevance expert who can determine a ranking of the passages based on how relevant they are to the query. "
        + "If the query is a question, how relevant a passage is depends on how well it answers the question. "
        + "If not, try to analyze the intent of the query and assess how well each passage satisfies the intent. "
        + "If an instruction is provided, you should follow the instruction when determining the ranking."
        + "<|im_end|>\n<|im_start|>user\n"
        + "I will provide you with \(documents.count) passages, each indicated by a numerical identifier. "
        + "Rank the passages based on their relevance to query: \(query)\n"
        + documents.enumerated().map { index, document in
            "<passage id=\"\(index)\">\n\(document)<|embed_token|>\n</passage>"
        }.joined(separator: "\n")
        + "\n<query>\n\(query)<|rerank_token|>\n</query>"
        + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
}

private struct ByteRerankerTokenizer: Tokenizer {
    let trueTokenID = 1
    let falseTokenID = 2
    let bosTokenID = 3
    let eosTokenID = 4
    let rerankTokenID = 5
    let embedTokenID = 6

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        var tokenIDs = [Int]()
        var remaining = text[...]
        while !remaining.isEmpty {
            if remaining.hasPrefix("<|rerank_token|>") {
                tokenIDs.append(rerankTokenID)
                remaining.removeFirst("<|rerank_token|>".count)
            } else if remaining.hasPrefix("<|embed_token|>") {
                tokenIDs.append(embedTokenID)
                remaining.removeFirst("<|embed_token|>".count)
            } else {
                tokenIDs.append(Int(remaining.removeFirst().asciiValue ?? 0) + 10)
            }
        }
        return tokenIDs
    }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        String(decoding: tokenIds.map { UInt8(max(0, $0 - 10)) }, as: UTF8.self)
    }

    func convertTokenToId(_ token: String) -> Int? {
        switch token {
        case "yes": trueTokenID
        case "no": falseTokenID
        case "<s>": bosTokenID
        case "</s>": eosTokenID
        case "<|rerank_token|>": rerankTokenID
        case "<|embed_token|>": embedTokenID
        default: nil
        }
    }

    func convertIdToToken(_ id: Int) -> String? { nil }
    var bosToken: String? { "<s>" }
    var eosToken: String? { "</s>" }
    var unknownToken: String? { nil }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] { [] }
}

/// Models a tokenizer merge across the document-prefix boundary.
private struct BoundaryMergingRerankerTokenizer: Tokenizer {
    private let base = ByteRerankerTokenizer()

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        var tokenIDs = [Int]()
        var remaining = text[...]
        while !remaining.isEmpty {
            if remaining.hasPrefix("<|rerank_token|>") {
                tokenIDs.append(base.rerankTokenID)
                remaining.removeFirst("<|rerank_token|>".count)
            } else if remaining.hasPrefix("<|embed_token|>") {
                tokenIDs.append(base.embedTokenID)
                remaining.removeFirst("<|embed_token|>".count)
            } else if remaining.hasPrefix("\nd") {
                tokenIDs.append(250)
                remaining.removeFirst(2)
            } else {
                tokenIDs.append(Int(remaining.removeFirst().asciiValue ?? 0) + 10)
            }
        }
        return tokenIDs
    }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        base.decode(tokenIds: tokenIds, skipSpecialTokens: skipSpecialTokens)
    }

    func convertTokenToId(_ token: String) -> Int? {
        base.convertTokenToId(token)
    }

    func convertIdToToken(_ id: Int) -> String? { base.convertIdToToken(id) }
    var bosToken: String? { base.bosToken }
    var eosToken: String? { base.eosToken }
    var unknownToken: String? { base.unknownToken }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] { [] }
}

private struct StubReranker: Reranker {
    let modelID = "test/stub"
    let scoreKind = RerankScoreKind.normalizedRelevance
    let results: [RerankResult]

    func scores(
        query: String,
        documents: [String],
        instruction: String?,
        options: RerankExecutionOptions
    ) async throws -> RerankResponse {
        RerankResponse(modelID: modelID, scoreKind: scoreKind, results: results)
    }
}

private final class TestEncoderRerankerModel: Module, RerankerModel, @unchecked Sendable {
    enum Output {
        case oneLogit
        case oneDimensional
        case twoLogits
        case fixedTwoLogits
    }

    let rerankerConfiguration: EncoderRerankerModelConfiguration
    let output: Output
    var scoreCallCount = 0
    var scoredShapes = [[Int]]()
    var vocabularySize: Int { 512 }
    var maxPositionEmbeddings: Int? { rerankerConfiguration.maxInputTokens }

    init(
        configuration: EncoderRerankerModelConfiguration,
        output: Output = .oneLogit
    ) {
        rerankerConfiguration = configuration
        self.output = output
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
        scoredShapes.append(inputs.shape)
        let mask = attentionMask ?? MLXArray.ones(inputs.shape, dtype: .int32)
        let sums = MLX.sum(inputs.asType(.float32) * mask.asType(.float32), axis: 1)
        switch output {
        case .oneLogit:
            return sums.reshaped(inputs.dim(0), 1)
        case .oneDimensional:
            return sums
        case .twoLogits:
            return stacked([-sums, sums], axis: 1)
        case .fixedTwoLogits:
            return tiled(MLXArray([Float(0), Float(2)]), repetitions: [inputs.dim(0), 1])
        }
    }
}

/// Uses the reference logits as hidden states, so projection is the identity.
private final class TestHiddenStateRerankerModel: Module, HiddenStateLanguageModel {
    private let base: TestCausalRerankerModel
    private(set) var hiddenStateShapes = [[Int]]()
    private(set) var projectedShapes = [[Int]]()
    private(set) var fullForwardShapes = [[Int]]()

    init(tokenizer: ByteRerankerTokenizer) {
        base = TestCausalRerankerModel(
            trueTokenID: tokenizer.trueTokenID, falseTokenID: tokenizer.falseTokenID)
    }

    func hiddenStates(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let hidden = base(.init(tokens: inputs), cache: cache, state: nil).logits
        hiddenStateShapes.append(hidden.shape)
        return hidden
    }

    func projectLogits(_ hiddenStates: MLXArray) -> MLXArray {
        projectedShapes.append(hiddenStates.shape)
        return hiddenStates
    }

    func prepare(
        _ input: LMInput, cache: [KVCache], state: LMOutput.State?, prefill: PrefillParameters
    ) throws -> PrepareResult {
        .tokens(input.text)
    }

    func callAsFunction(
        _ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?
    ) -> LMOutput {
        fullForwardShapes.append(input.tokens.shape)
        return base(input, cache: cache, state: state)
    }

    func newCache(parameters: GenerateParameters?) -> [KVCache] { [] }
}

private final class TestCausalRerankerModel: Module, LanguageModel, JinaRerankerEmbeddingModel,
    @unchecked Sendable
{
    let trueTokenID: Int
    let falseTokenID: Int
    var callCount = 0
    var callShapes = [[Int]]()
    var listwiseTokenCount = 0
    var listwiseBlockCounts = [Int]()

    init(trueTokenID: Int, falseTokenID: Int) {
        self.trueTokenID = trueTokenID
        self.falseTokenID = falseTokenID
    }

    func prepare(
        _ input: LMInput,
        cache: [KVCache],
        state: LMOutput.State?,
        prefill: PrefillParameters
    ) throws -> PrepareResult {
        .tokens(input.text)
    }

    func callAsFunction(
        _ input: LMInput.Text,
        cache: [KVCache]?,
        state: LMOutput.State?
    ) -> LMOutput {
        callCount += 1
        var tokens = input.tokens
        if tokens.ndim == 1 {
            tokens = tokens.reshaped(1, -1)
        }
        callShapes.append(tokens.shape)
        let batchSize = tokens.dim(0)
        let sequenceLength = tokens.dim(1)
        let vocabularySize = 128
        let tokenValues = tokens.asArray(Int.self)
        var values = Array(
            repeating: Float(-100),
            count: batchSize * sequenceLength * vocabularySize)
        for row in 0 ..< batchSize {
            var runningTotal = 0
            for column in 0 ..< sequenceLength {
                runningTotal += tokenValues[row * sequenceLength + column] + 1
                let offset = (row * sequenceLength + column) * vocabularySize
                values[offset + trueTokenID] = Float(runningTotal % 100) / 20
                values[offset + falseTokenID] = 0
            }
        }
        return LMOutput(
            logits: MLXArray(values).reshaped(batchSize, sequenceLength, vocabularySize))
    }

    func newCache(parameters: GenerateParameters?) -> [KVCache] { [] }

    func embeddings(input: RerankerInput, documentCount: Int, stepSize: Int) throws
        -> JinaRerankerEmbeddings
    {
        listwiseTokenCount = input.tokenIds.count
        let query = listwiseBlockCounts.isEmpty ? [Float(1), 0] : [Float(0), 1]
        listwiseBlockCounts.append(documentCount)
        return JinaRerankerEmbeddings(
            documents: tiled(
                MLXArray([Float(1), 0]).reshaped(1, 2), repetitions: [documentCount, 1]),
            query: MLXArray(query).reshaped(1, 2))
    }
}

extension TestInputProcessor {
    fileprivate init(tokenizer: any Tokenizer) {
        self.init(
            tokenizer: tokenizer,
            configuration: ModelConfiguration(id: "test/reranker"),
            messageGenerator: DefaultMessageGenerator())
    }
}

private func qwenScorePolicy(_ tokenizer: any Tokenizer) throws -> CausalRerankerScorePolicy {
    try CausalRerankerFamily.qwen3.scorePolicy(
        metadata: nil, tokenizer: tokenizer, vocabularySize: 128)
}

private func causalMetadata(trueID: Int, falseID: Int? = nil) throws -> CausalRerankerMetadata {
    let object: [String: Any] = [
        "true_token_id": trueID, "false_token_id": falseID.map { $0 as Any } ?? NSNull(),
    ]
    return try JSONDecoder().decode(
        CausalRerankerMetadata.self, from: JSONSerialization.data(withJSONObject: object))
}

private func qwenAttentionConfiguration(types: [String], window: Int?, layers: Int? = nil) throws
    -> MLXLLM.Qwen3Configuration
{
    var object = try #require(
        JSONSerialization.jsonObject(with: jinaConfigurationData()) as? [String: Any])
    object["layer_types"] = types
    object["use_sliding_window"] = true
    object["num_hidden_layers"] = layers ?? types.count
    object["sliding_window"] = window.map { $0 as Any } ?? NSNull()
    return try JSONDecoder().decode(
        MLXLLM.Qwen3Configuration.self, from: JSONSerialization.data(withJSONObject: object))
}

private struct FixtureRerankerTokenizerLoader: TokenizerLoader {
    func load(from directory: URL) async throws -> any Tokenizer { ByteRerankerTokenizer() }
}

private struct FixtureRerankerDownloader: Downloader {
    let directory: URL
    func download(
        id: String, revision: String?, matching patterns: [String], useLatest: Bool,
        progressHandler: @Sendable @escaping (Progress) -> Void
    ) async throws -> URL { directory }
}
