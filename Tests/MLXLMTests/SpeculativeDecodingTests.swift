// Copyright © 2025 Apple Inc.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import Testing

@Suite(.serialized)
struct SpeculativeDecodingTests {

    let processor: any UserInputProcessor
    let mainContext: ModelContext
    let draftContext: ModelContext

    init() {
        let processor = TestInputProcessor()
        let modelConfig = Gemma3TextConfiguration(
            modelType: "text",
            hiddenSize: 64, hiddenLayers: 8, intermediateSize: 64,
            attentionHeads: 4, headDim: 64,
            rmsNormEps: 0.00001, vocabularySize: 100, kvHeads: 4,
            ropeTheta: 1_000_000, ropeLocalBaseFreq: 10_000,
            ropeTraditional: false, queryPreAttnScalar: 256,
            slidingWindow: 512, slidingWindowPattern: 6,
            maxPositionEmbeddings: 32768
        )

        let mainModel = Gemma3TextModel(modelConfig)

        // on hardware with a NAX, float32 (the default dtype) runs
        // in tf32 in batch mode and float32 in non-batch.  this
        // change in behavior can cause issues with prediction and
        // doesn't match real world behavior (where float32 is not used)
        mainModel.apply {
            if $0.dtype == .float32 {
                $0.asType(.float16)
            } else {
                $0
            }
        }
        let mainContext = ModelContext(
            configuration: processor.configuration,
            model: mainModel,
            processor: processor,
            tokenizer: processor.tokenizer
        )

        let draftModel = Gemma3TextModel(modelConfig)
        draftModel.apply {
            if $0.dtype == .float32 {
                $0.asType(.float16)
            } else {
                $0
            }
        }
        let draftContext = ModelContext(
            configuration: processor.configuration,
            model: draftModel,
            processor: processor,
            tokenizer: processor.tokenizer
        )

        eval(mainModel, draftModel)
        self.processor = processor
        self.mainContext = mainContext
        self.draftContext = draftContext
    }

    @Test(arguments: [2, 8, 48], [false, true])
    func `Speculative decoding matches default token generation for stable logits`(
        numDraftTokens: Int,
        withLogitProcessor: Bool
    ) async throws {
        let vocabularySize = 100
        let tokenizer = TestTokenizer(vocabularySize: vocabularySize)
        let processor = TestInputProcessor(
            tokenizer: tokenizer,
            configuration: ModelConfiguration(id: "stable-transition-test"),
            messageGenerator: DefaultMessageGenerator()
        )
        let model = StableTransitionLanguageModel(vocabularySize: vocabularySize)
        let draftModel = StableTransitionLanguageModel(vocabularySize: vocabularySize)
        let context = ModelContext(
            configuration: processor.configuration,
            model: model,
            processor: processor,
            tokenizer: processor.tokenizer
        )
        let input = LMInput(tokens: MLXArray([92, 85, 2, 95, 55, 7, 94, 42]))
        let parameters = GenerateParameters(
            maxTokens: 32,
            temperature: 0.0,  // Use greedy decoding for deterministic output
            repetitionPenalty: withLogitProcessor ? 1.5 : nil,
            presencePenalty: withLogitProcessor ? 0.5 : nil,
            frequencyPenalty: withLogitProcessor ? 0.2 : nil,
        )

        var normalTokens: [Int] = []
        for await generation in try generateTokens(
            input: input, parameters: parameters, context: context
        ) {
            if let token = generation.token { normalTokens.append(token) }
        }

        var speculativeTokens: [Int] = []
        for await generation in try generateTokens(
            input: input, parameters: parameters, context: context,
            draftModel: draftModel, numDraftTokens: numDraftTokens
        ) {
            if let token = generation.token { speculativeTokens.append(token) }
        }

        #expect(!normalTokens.isEmpty)
        #expect(!speculativeTokens.isEmpty)
        #expect(normalTokens == speculativeTokens)
    }

    @Test(arguments: [2, 8, 48], [false, true])
    func `Speculative decoding Gemma3 smoke test`(
        numDraftTokens: Int,
        withLogitProcessor: Bool
    ) async throws {
        let input = UserInput(prompt: "Input text")
        let modelInput = try await processor.prepare(input: input)
        let maxTokens = 32
        let parameters = GenerateParameters(
            maxTokens: maxTokens,
            temperature: 0.0,
            repetitionPenalty: withLogitProcessor ? 1.5 : nil,
            presencePenalty: withLogitProcessor ? 0.5 : nil,
            frequencyPenalty: withLogitProcessor ? 0.2 : nil,
        )

        var speculativeTokens: [Int] = []
        var telemetry: SpeculativeDecodingTelemetry?
        for await generation in try generateTokens(
            input: modelInput, parameters: parameters, context: mainContext,
            draftModel: draftContext.model, numDraftTokens: numDraftTokens
        ) {
            if let token = generation.token { speculativeTokens.append(token) }
            if let info = generation.info {
                telemetry = info.speculativeDecodingTelemetry
            }
        }

        // Real MLX model kernels may choose different argmaxes for batched
        // verification and token-by-token decoding when logits are close.
        // Keep this as model-path coverage; exact equality belongs to the
        // stable-logit contract test above.
        #expect(speculativeTokens.count == maxTokens)

        let speculativeTelemetry = try #require(telemetry)
        #expect(speculativeTelemetry.roundCount > 0)
        #expect(speculativeTelemetry.draftTokenCount > 0)
        #expect(speculativeTelemetry.targetModelCallCount == speculativeTelemetry.roundCount)
        #expect(speculativeTelemetry.draftModelCallCount == speculativeTelemetry.draftTokenCount)
        #expect(speculativeTelemetry.acceptanceRate >= 0)
        #expect(speculativeTelemetry.acceptanceRate <= 1)
    }

    @Test func `Speculative telemetry emitted count matches generated tokens`() async throws {
        let input = UserInput(prompt: "Input text")
        let modelInput = try await processor.prepare(input: input)
        let parameters = GenerateParameters(
            maxTokens: 1,
            temperature: 0.0
        )

        var tokenCount = 0
        var info: GenerateCompletionInfo?
        for await generation in try generateTokens(
            input: modelInput, parameters: parameters, context: mainContext,
            draftModel: draftContext.model, numDraftTokens: 8
        ) {
            if generation.token != nil {
                tokenCount += 1
            }
            if let generationInfo = generation.info {
                info = generationInfo
            }
        }

        let completionInfo = try #require(info)
        let telemetry = try #require(completionInfo.speculativeDecodingTelemetry)
        #expect(completionInfo.generationTokenCount == tokenCount)
        #expect(telemetry.emittedTokenCount == tokenCount)
        #expect(telemetry.emittedTokenCount == completionInfo.generationTokenCount)
    }

    @Test func `Speculative telemetry emitted count works with direct iterator`() async throws {
        let input = UserInput(prompt: "Input text")
        let modelInput = try await processor.prepare(input: input)
        let parameters = GenerateParameters(
            maxTokens: 3,
            temperature: 0.0
        )

        var iterator = try SpeculativeTokenIterator(
            input: modelInput,
            mainModel: mainContext.model,
            draftModel: draftContext.model,
            parameters: parameters,
            numDraftTokens: 8
        )

        var tokenCount = 0
        while iterator.next() != nil {
            tokenCount += 1
        }

        let telemetry = try #require(iterator.speculativeDecodingTelemetry)
        #expect(tokenCount == 3)
        #expect(telemetry.emittedTokenCount == tokenCount)
    }
}

/// Deterministic causal model for speculative decoding contract tests.
///
/// Each logit row predicts a high-margin transition from the token at the
/// same position. Batched verification and token-by-token decoding therefore
/// execute the same mathematical function, so equality failures point at the
/// speculative iterator rather than at hardware-dependent MLX kernel drift.
private final class StableTransitionLanguageModel: Module, LanguageModel, KVCacheDimensionProvider {
    let vocabularySize: Int
    var kvHeads: [Int] { [] }

    init(vocabularySize: Int) {
        self.vocabularySize = vocabularySize
        super.init()
    }

    func prepare(_ input: LMInput, cache: [KVCache], state _: LMOutput.State?, windowSize: Int?)
        throws -> PrepareResult
    {
        .tokens(input.text)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let tokenIds = inputs.asArray(Int.self)
        var logits = Array(
            repeating: Float(-100),
            count: tokenIds.count * vocabularySize
        )

        for (position, token) in tokenIds.enumerated() {
            logits[position * vocabularySize + nextToken(after: token)] = 100
        }

        return MLXArray(logits, [1, tokenIds.count, vocabularySize])
    }

    private func nextToken(after token: Int) -> Int {
        (token * 31 + 7) % vocabularySize
    }
}
