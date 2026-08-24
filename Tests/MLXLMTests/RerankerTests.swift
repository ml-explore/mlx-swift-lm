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
            try await reranker.scores(query: "q", documents: ["d"])
        }
        task.cancel()

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
                maximumDocuments: 64,
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
            maximumDocuments: 64,
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
        let provider = try #require(model as? any AdditionalWeightFilesProviding)
        #expect(provider.additionalWeightFiles == ["projector.safetensors"])
    }

    @Test func jinaCosineSimilarityRemainsInDeclaredRange() {
        let vector = MLXArray([Float(0.1), 0.2, 0.3]).reshaped(1, 3)
        let scores = jinaCosineSimilarity(vector, vector).asArray(Float.self)

        #expect(scores.count == 1)
        #expect(scores[0] >= -1)
        #expect(scores[0] <= 1)
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
            options: .init(maxBatchSize: 2, maxBatchTokens: 4_096))

        #expect(scores.count == 3)
        #expect(model.callCount == 2)
        #expect(scores.allSatisfy { $0 >= 0 && $0 <= 1 })
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
            options: .init(maxBatchTokens: 512))

        #expect(model.callShapes == [[1, 512]])
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
        #expect(try jina.architecture(modelID: "jinaai/jina-reranker-v3-mlx") == .jina)
        #expect(
            try qwen.architecture(modelID: "Qwen/Qwen3-Reranker-0.6B") == .qwen3)
        #expect(try unsupported.architecture(modelID: "test/model") == nil)

        #expect(throws: RerankerError.self) {
            try qwen.architecture(modelID: "Qwen/Qwen3-0.6B")
        }
        #expect(throws: RerankerError.self) {
            try bge.architecture(modelID: "example/sentiment-classifier")
        }
        #expect(
            try qwen.architecture(
                modelID: "lampo/private-model", allowUnverifiedModel: true) == .qwen3)
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

private final class TestCausalRerankerModel: Module, LanguageModel, ListwiseRerankerModel,
    @unchecked Sendable
{
    let trueTokenID: Int
    let falseTokenID: Int
    var callCount = 0
    var callShapes = [[Int]]()
    var listwiseTokenCount = 0

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
                runningTotal += tokenValues[row * sequenceLength + column]
                let offset = (row * sequenceLength + column) * vocabularySize
                values[offset + trueTokenID] = Float(runningTotal % 100) / 20
                values[offset + falseTokenID] = 0
            }
        }
        return LMOutput(
            logits: MLXArray(values).reshaped(batchSize, sequenceLength, vocabularySize))
    }

    func newCache(parameters: GenerateParameters?) -> [KVCache] { [] }

    func score(input: RerankerInput, documentCount: Int) throws -> [Double] {
        listwiseTokenCount = input.tokenIds.count
        return Array(repeating: 0.5, count: documentCount)
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
