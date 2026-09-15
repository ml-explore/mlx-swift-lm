// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLMCommon

/// Deterministic verifier whose `prepare` returns `.logits`, the shape of every
/// wired VLM prefill (and of `ChatSession`'s Muse path). Each logit row predicts
/// a high-margin transition from the token at the same position, so batched
/// verification and token-by-token decoding compute the same function and any
/// equality failure points at the speculative plumbing, not at kernel drift.
private final class TransitionOracleModel: Module, LanguageModel, KVCacheDimensionProvider {
    let vocabularySize: Int
    var kvHeads: [Int] { [] }

    init(vocabularySize: Int) {
        self.vocabularySize = vocabularySize
        super.init()
    }

    func prepare(
        _ input: LMInput, cache: [KVCache], state _: LMOutput.State?,
        prefill: PrefillParameters
    ) throws -> PrepareResult {
        let total = input.text.tokens.size
        prefill.progress?(total, total)
        return .logits(LMOutput(logits: callAsFunction(input.text.tokens, cache: nil)))
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let tokenIds = inputs.reshaped(-1).asArray(Int.self)
        var logits = [Float](repeating: -100, count: tokenIds.count * vocabularySize)
        for (position, token) in tokenIds.enumerated() {
            logits[position * vocabularySize + (token * 31 + 7) % vocabularySize] = 100
        }
        return MLXArray(logits, [1, tokenIds.count, vocabularySize])
    }
}

@Suite("Prompt-lookup decoding")
struct PromptLookupDecodingTests {

    private func drafter(
        maxNGram: Int = 4, minNGram: Int = 1, maxLookback: Int? = nil
    ) -> PromptLookupDraftModel {
        PromptLookupDraftModel(
            configuration: .init(
                vocabularySize: 100, maxNGram: maxNGram, minNGram: minNGram,
                maxLookback: maxLookback))
    }

    // MARK: - Token-history cache

    @Test func tokenCacheTracksAppendsAndClampsTrims() {
        let cache = PromptLookupTokenCache()
        #expect(cache.offset == 0)
        #expect(cache.isTrimmable)
        cache.append([1, 2, 3, 4])
        #expect(cache.offset == 4)
        #expect(cache.trim(2) == 2)
        #expect(cache.tokens == [1, 2])
        #expect(cache.trim(5) == 2)
        #expect(cache.offset == 0)
        #expect(cache.trim(-3) == 0)
    }

    @Test func tokenCacheCopyIsIndependent() {
        let cache = PromptLookupTokenCache()
        cache.append([9, 8, 7])
        guard let copy = cache.copy() as? PromptLookupTokenCache else {
            Issue.record("copy() changed the cache type")
            return
        }
        copy.append([6])
        #expect(cache.tokens == [9, 8, 7])
        #expect(copy.tokens == [9, 8, 7, 6])
    }

    @Test func tokenCacheStateRoundTripsThroughMLXArrays() {
        let cache = PromptLookupTokenCache()
        cache.append([5, 6, 7])
        let restored = PromptLookupTokenCache()
        restored.state = cache.state
        #expect(restored.tokens == [5, 6, 7])
        #expect(restored.offset == 3)
    }

    // MARK: - n-gram matching

    @Test func predictsTheContinuationOfTheLongestSuffixMatch() {
        #expect(drafter().predict([1, 2, 3, 4, 7, 5, 1, 2, 3, 4]) == 7)
    }

    @Test func mostRecentOccurrenceWins() {
        // [3, 4] occurs twice with different continuations; the later one is used.
        #expect(drafter(maxNGram: 2).predict([3, 4, 8, 0, 3, 4, 9, 1, 3, 4]) == 9)
    }

    @Test func acceptedPredictionsBlockCopyTheMatchedSpan() {
        let model = drafter()
        var history: [Int32] = [1, 2, 3, 4, 5, 6, 9, 1, 2, 3]
        var predicted = [Int32]()
        for _ in 0 ..< 3 {
            let token = model.predict(history)
            predicted.append(token)
            history.append(token)  // the verifier accepted it
        }
        #expect(predicted == [4, 5, 6])
    }

    @Test func rejectedPredictionTriggersAFreshMatch() {
        let model = drafter()
        #expect(model.predict([1, 2, 3, 4, 7, 5, 1, 2, 3, 4]) == 7)
        // The verifier sampled 5 instead of the predicted 7: the stale block
        // copy must be abandoned and the suffix re-matched from scratch.
        #expect(model.predict([1, 2, 3, 4, 7, 5, 1, 2, 3, 4, 5]) == 1)
    }

    @Test func minNGramSuppressesShortMatches() {
        // Only a 1-gram match exists; with minNGram 2 it must be skipped and
        // the drafter falls back to repeating the last token.
        #expect(drafter(maxNGram: 3, minNGram: 2).predict([1, 5, 2, 9, 5]) == 5)
        #expect(drafter(maxNGram: 3, minNGram: 1).predict([1, 5, 2, 9, 5]) == 2)
    }

    @Test func maxLookbackBoundsTheScan() {
        // The only [7, 8] match starts at position 0, outside a lookback of 4.
        #expect(drafter(maxNGram: 2, maxLookback: 4).predict([7, 8, 3, 0, 1, 2, 7, 8]) == 8)
        #expect(drafter(maxNGram: 2).predict([7, 8, 3, 0, 1, 2, 7, 8]) == 3)
    }

    @Test func emptyHistoryProposesTokenZero() {
        #expect(drafter().predict([]) == 0)
    }

    @Test func configurationRejectsInvalidNGramBounds() {
        // Covered by preconditions at init; here pin the valid boundary.
        let configuration = PromptLookupDraftModel.Configuration(
            vocabularySize: 1, maxNGram: 1, minNGram: 1)
        #expect(configuration.maxNGram == 1)
    }

    // MARK: - one-hot logits

    @Test func prepareEmitsOneHotLogitsForTheMatchedContinuation() throws {
        let model = drafter()
        let cache = try model.newCache(parameters: nil)
        let result = try model.prepare(
            LMInput(tokens: MLXArray([1, 2, 3, 1, 2] as [Int32])),
            cache: cache, state: nil, prefill: .init())
        guard case .logits(let output) = result else {
            Issue.record("prompt-lookup prepare must return logits")
            return
        }
        #expect(output.logits.shape == [1, 1, 100])
        #expect(output.logits.argMax(axis: -1).item(Int.self) == 3)
    }

    // MARK: - configuration surface

    @Test func speculativeConfigExposesPromptLookup() {
        let config = SpeculativeDecodingConfig(
            lookup: .init(vocabularySize: 100), numDraftTokens: 4)
        #expect(config.lookup != nil)
        #expect(config.numDraftTokens == 4)
        #expect(config.estimatedDraftModelBytes == nil)
    }

    // MARK: - end-to-end against a deterministic verifier

    private func generate(
        prompt: [Int32], maxTokens: Int, drafts: Int?
    ) async throws -> (tokens: [Int], telemetry: SpeculativeDecodingTelemetry?) {
        let vocabularySize = 100
        let processor = TestInputProcessor(
            tokenizer: TestTokenizer(vocabularySize: vocabularySize),
            configuration: ModelConfiguration(id: "prompt-lookup-test"),
            messageGenerator: DefaultMessageGenerator())
        let context = ModelContext(
            configuration: processor.configuration,
            model: TransitionOracleModel(vocabularySize: vocabularySize),
            processor: processor,
            tokenizer: processor.tokenizer)
        let input = LMInput(tokens: MLXArray(prompt))
        let parameters = GenerateParameters(maxTokens: maxTokens, temperature: 0.0)

        var tokens = [Int]()
        var telemetry: SpeculativeDecodingTelemetry?
        if let drafts {
            for await generation in try generateTokens(
                input: input, parameters: parameters, context: context,
                draftModel: drafter(), numDraftTokens: drafts
            ) {
                if let token = generation.token { tokens.append(token) }
                if let info = generation.info {
                    telemetry = info.speculativeDecodingTelemetry
                }
            }
        } else {
            for await generation in try generateTokens(
                input: input, parameters: parameters, context: context
            ) {
                if let token = generation.token { tokens.append(token) }
            }
        }
        return (tokens, telemetry)
    }

    @Test(arguments: [2, 3, 8])
    func promptLookupMatchesPlainGenerationOnRepetitiveContext(drafts: Int) async throws {
        // The prompt repeats the transition orbit's opening span, so generated
        // tokens re-enter known context and block copies land.
        let prompt: [Int32] = [0, 7, 24, 51, 88, 35, 0]
        let plain = try await generate(prompt: prompt, maxTokens: 24, drafts: nil)
        let speculative = try await generate(prompt: prompt, maxTokens: 24, drafts: drafts)

        #expect(plain.tokens.count == 24)
        #expect(speculative.tokens == plain.tokens)
        let telemetry = try #require(speculative.telemetry)
        #expect(telemetry.acceptanceRate > 0)
    }

    @Test func promptLookupMatchesPlainGenerationWhenNothingMatches() async throws {
        // No structure to copy: every proposal is the repeat-last fallback and
        // the verifier rejects it, at no cost to correctness.
        let prompt: [Int32] = [92, 85, 2, 95, 55]
        let plain = try await generate(prompt: prompt, maxTokens: 16, drafts: nil)
        let speculative = try await generate(prompt: prompt, maxTokens: 16, drafts: 3)

        #expect(plain.tokens.count == 16)
        #expect(speculative.tokens == plain.tokens)
    }
}
