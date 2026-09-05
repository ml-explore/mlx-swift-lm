// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLMCommon

// MARK: - promptLookupDraft (pure lookup)

@Test func lookupProposesContinuationOfRepeatedNGram() {
    // Trailing [2, 3] previously occurred at index 1; propose what followed.
    let history = [1, 2, 3, 4, 5, 9, 2, 3]
    let proposal = promptLookupDraft(
        history: history, maxNGramLength: 3, minNGramLength: 1, maxTokens: 2)
    #expect(proposal == [4, 5])
}

@Test func lookupPrefersLongerPattern() {
    // Trailing [1, 2] matches at index 0 (→ 7) and trailing [9, 1, 2] at
    // index 3 (→ 8). The longer pattern wins.
    let history = [1, 2, 7, 9, 1, 2, 8, 5, 9, 1, 2]
    let proposal = promptLookupDraft(
        history: history, maxNGramLength: 3, minNGramLength: 1, maxTokens: 1)
    #expect(proposal == [8])
}

@Test func lookupPrefersMostRecentOccurrence() {
    // Trailing [5] occurred at indices 0 (→ 1) and 2 (→ 2): most recent wins.
    let history = [5, 1, 5, 2, 5]
    let proposal = promptLookupDraft(
        history: history, maxNGramLength: 3, minNGramLength: 1, maxTokens: 1)
    #expect(proposal == [2])
}

@Test func lookupReturnsEmptyWithoutRecurrence() {
    let history = [1, 2, 3, 4, 5, 6]
    let proposal = promptLookupDraft(
        history: history, maxNGramLength: 3, minNGramLength: 2, maxTokens: 4)
    #expect(proposal.isEmpty)
}

@Test func lookupClampsAtHistoryEnd() {
    // Match found near the end: only one token of continuation exists.
    let history = [1, 2, 3, 1, 2, 3, 1, 2]
    let proposal = promptLookupDraft(
        history: history, maxNGramLength: 3, minNGramLength: 1, maxTokens: 10)
    #expect(!proposal.isEmpty)
    #expect(proposal.first == 3)
}

@Test func lookupHandlesDegenerateHistories() {
    #expect(
        promptLookupDraft(history: [], maxNGramLength: 3, minNGramLength: 1, maxTokens: 4)
            .isEmpty)
    #expect(
        promptLookupDraft(history: [7], maxNGramLength: 3, minNGramLength: 1, maxTokens: 4)
            .isEmpty)
}

// MARK: - Iterator contract

/// Deterministic causal model whose every logit row predicts a high-margin
/// transition from the token at the same position (same contract as the
/// speculative-decoding test model), with a small cycle so generated text
/// recurs and prompt lookup gets hits. Counts forward calls so tests can
/// assert PLD saves model invocations.
private final class CyclicTransitionLanguageModel: Module, LanguageModel,
    KVCacheDimensionProvider
{
    let vocabularySize: Int
    var kvHeads: [Int] { [] }
    private(set) var forwardCallCount = 0

    /// When set, `newCache` returns a rotating (sliding-window) cache so
    /// tests can exercise the untrimmable-after-wrap fallback.
    var rotatingCacheMaxSize: Int?

    init(vocabularySize: Int, rotatingCacheMaxSize: Int? = nil) {
        self.vocabularySize = vocabularySize
        self.rotatingCacheMaxSize = rotatingCacheMaxSize
        super.init()
    }

    func newCache(parameters: GenerateParameters?) -> [KVCache] {
        if let rotatingCacheMaxSize {
            return [RotatingKVCache(maxSize: rotatingCacheMaxSize)]
        }
        return [KVCacheSimple()]
    }

    func prepare(_ input: LMInput, cache: [KVCache], state _: LMOutput.State?, windowSize: Int?)
        throws -> PrepareResult
    {
        .tokens(input.text)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        forwardCallCount += 1
        let tokenIds = inputs.asArray(Int.self)
        var logits = Array(
            repeating: Float(-100),
            count: tokenIds.count * vocabularySize
        )
        for (position, token) in tokenIds.enumerated() {
            logits[position * vocabularySize + (token + 1) % vocabularySize] = 100
        }
        // Keep cache offsets in sync with the tokens processed.
        if let kvCache = cache?.first {
            let dummy = MLXArray.zeros([1, 1, tokenIds.count, 4], dtype: .float32)
            _ = kvCache.update(keys: dummy, values: dummy)
        }
        return MLXArray(logits, [1, tokenIds.count, vocabularySize])
    }
}

private func drain(_ iterator: inout some TokenIteratorProtocol) -> [Int] {
    var tokens = [Int]()
    while let token = iterator.next() {
        tokens.append(token)
    }
    return tokens
}

@Test func promptLookupMatchesPlainGenerationExactly() throws {
    // The model cycles through an 8-token vocabulary, so the output repeats
    // and lookup rounds fire. At temperature 0 PLD must be a pure
    // accelerator: identical token stream to the plain TokenIterator.
    let parameters = GenerateParameters(maxTokens: 40, temperature: 0)
    let prompt = LMInput(tokens: MLXArray([0, 1, 2]))

    let plainModel = CyclicTransitionLanguageModel(vocabularySize: 8)
    var plain = try TokenIterator(
        input: prompt, model: plainModel, parameters: parameters)
    let plainTokens = drain(&plain)

    let pldModel = CyclicTransitionLanguageModel(vocabularySize: 8)
    var pld = try PromptLookupTokenIterator(
        input: prompt, model: pldModel, parameters: parameters,
        numDraftTokens: 6, maxNGramLength: 3, minNGramLength: 1)
    let pldTokens = drain(&pld)

    #expect(pldTokens == plainTokens)

    // Repetitive text: lookup rounds must fire, accept drafts, and save
    // model forwards versus token-by-token generation.
    let telemetry = try #require(pld.speculativeDecodingTelemetry)
    #expect(telemetry.acceptedDraftTokenCount > 0)
    #expect(pldModel.forwardCallCount < plainModel.forwardCallCount)
}

@Test func promptLookupFallsBackAfterSlidingWindowWraps() throws {
    // With a rotating cache that wraps mid-generation the iterator must
    // stop speculating (the rollback could no longer rewind rejections)
    // and still produce the exact plain-generation stream.
    let parameters = GenerateParameters(maxTokens: 48, temperature: 0)
    let prompt = LMInput(tokens: MLXArray([0, 1, 2]))

    let plainModel = CyclicTransitionLanguageModel(vocabularySize: 8)
    var plain = try TokenIterator(
        input: prompt, model: plainModel, parameters: parameters)
    let plainTokens = drain(&plain)

    let pldModel = CyclicTransitionLanguageModel(vocabularySize: 8, rotatingCacheMaxSize: 16)
    var pld = try PromptLookupTokenIterator(
        input: prompt, model: pldModel, parameters: parameters,
        numDraftTokens: 6, maxNGramLength: 3, minNGramLength: 1)
    let pldTokens = drain(&pld)

    #expect(pldTokens == plainTokens)
}
