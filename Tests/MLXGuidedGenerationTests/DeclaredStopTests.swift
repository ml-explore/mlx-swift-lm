// Copyright © 2026 Apple Inc.

import MLX
import MLXLMCommon
import Testing

@testable import MLXGuidedGeneration

/// Greedy sampler that records the logits it sees.
private final class RecordingSampler: LogitSampler {
    var calls: [[Float]] = []
    var sampled: [Int] = []

    func sample(logits: MLXArray) -> MLXArray {
        calls.append(logits.reshaped(-1).asArray(Float.self))
        let token = argMax(logits, axis: -1)
        sampled.append(Int(token.item(Int32.self)))
        return token
    }
}

/// The grammar knows the stop tokens the loop stops on.
@Suite(.serialized)
struct DeclaredStopTests {

    private static let stringSchema =
        #"{"type":"object","properties":{"s":{"type":"string"}},"required":["s"],"additionalProperties":false}"#

    /// The grammar needs `x` as text, but `x` is a stop id, so the first mask
    /// allows nothing. The run throws before the sampler or the processor runs.
    @Test func emptyFirstMaskThrowsWithoutSampling() throws {
        try Stream.withNewDefaultStream(device: .cpu) {
            let sampler = RecordingSampler()
            let run = try runLoop(
                source: .grammar(#"root ::= "x""#), policy: .prefer([0x78]),
                maxTokens: 8, eosTokenIds: [0x78], sampler: sampler)
            #expect(
                (run.error as? GuidedGenerationError)
                    == .noAllowedToken(stopTokenIds: [0x78, HarnessTokenizer.eosTokenId]))
            #expect(sampler.calls.isEmpty)
            #expect(run.processCalls == 0)
            #expect(run.text.isEmpty)
            #expect(run.fed.isEmpty)
            #expect(run.modelCalls == 1)
        }
    }

    /// The grammar allows only `z` (id 122), but the model's logit row ends at
    /// 100, so the model cannot produce it. The run throws before sampling.
    @Test func maskBeyondLogitRowThrowsWithoutSampling() throws {
        try Stream.withNewDefaultStream(device: .cpu) {
            let sampler = RecordingSampler()
            let run = try runLoop(
                source: .grammar(#"root ::= "z""#), policy: .prefer([0x41]),
                maxTokens: 8, logitDim: 100, sampler: sampler)
            #expect(
                (run.error as? GuidedGenerationError)
                    == .noAllowedToken(stopTokenIds: [HarnessTokenizer.eosTokenId]))
            #expect(sampler.calls.isEmpty)
        }
    }

    /// Once the JSON is complete the grammar allows the secondary stop `x`,
    /// and the loop ends on it as on the EOS.
    @Test func secondaryStopEndsCompleteJSON() throws {
        try Stream.withNewDefaultStream(device: .cpu) {
            let sampler = RecordingSampler()
            let run = try runLoop(
                source: .jsonSchema(Self.stringSchema),
                policy: .then(prefix: #"{"s":"ab"}"#, .prefer([0x78])),
                maxTokens: 64, eosTokenIds: [0x78], sampler: sampler)
            #expect(run.error == nil)
            #expect(run.text == #"{"s":"ab"}"#)
            let last = try #require(sampler.calls.last)
            #expect(last[0x78].isFinite)
            #expect(sampler.sampled.last == 0x78)
            #expect(!run.tokenIDs.contains(0x78))
        }
    }

    /// `z` is the unknown token. The grammar never allows it as text, so the
    /// string does not end on it.
    @Test func unknownTokenInsideStringIsNotText() throws {
        try Stream.withNewDefaultStream(device: .cpu) {
            let run = try runLoop(
                source: .jsonSchema(Self.stringSchema),
                tokenizer: HarnessTokenizer(unknownByte: 0x7A),
                policy: .then(prefix: #"{"s":""#, .prefer([0x7A, 0x22, 0x7D, 0xFF])),
                maxTokens: 64)
            #expect(run.error == nil)
            #expect(run.text == #"{"s":""}"#)
        }
    }

    /// The initializer without `stopTokenIds` declares only the EOS, as before:
    /// another stop id is still text inside a string.
    @Test func eosOnlyInitializerKeepsOtherStopsAsText() throws {
        let grammarTokenizer = try GrammarTokenizer(
            vocab: HarnessTokenizer().grammarVocabulary, vocabType: .byteFallback,
            eosTokenId: 255)
        #expect(grammarTokenizer.stopTokenIDs == [255])

        let constraint = try GrammarConstraint(
            tokenizer: grammarTokenizer, jsonSchema: Self.stringSchema)
        for byte in #"{"s":""#.utf8 {
            _ = try constraint.commitToken(Int32(byte))
        }
        let mask = try constraint.computeMask()
        #expect(GuidedGenerationLoop.maskAllows(mask, tokenId: 0x78, vocabSize: 256))
        #expect(!GuidedGenerationLoop.maskAllows(mask, tokenId: 255, vocabSize: 256))
    }

    /// The declared stops are sorted, without duplicates or ids outside the
    /// vocabulary, and hold the EOS. xgrammar treats each of them as a stop.
    @Test func stopListIsCleaned() throws {
        let grammarTokenizer = try GrammarTokenizer(
            vocab: HarnessTokenizer().grammarVocabulary, vocabType: .byteFallback,
            eosTokenId: 255, stopTokenIds: [0x78, 0x22, 0x78, -1, 256, 1000])
        #expect(grammarTokenizer.stopTokenIDs == [0x22, 0x78, 255])

        // Stops are masked until the grammar can end, then all allowed.
        let constraint = try GrammarConstraint(
            tokenizer: grammarTokenizer, grammar: #"root ::= [a-z"]+"#)
        let first = try constraint.computeMask()
        _ = try constraint.commitToken(0x61)
        let afterA = try constraint.computeMask()
        for id in [0x22, 0x78, 255] {
            #expect(!GuidedGenerationLoop.maskAllows(first, tokenId: id, vocabSize: 256))
            #expect(GuidedGenerationLoop.maskAllows(afterA, tokenId: id, vocabSize: 256))
        }
    }

    /// With the EOS outside the vocabulary and no other stop id, xgrammar still
    /// gets the EOS id, not an empty list that would make it guess stop tokens
    /// from their text: `</s>` stays text.
    @Test func eosOutsideVocabularyKeepsTextTokens() throws {
        let grammarTokenizer = try GrammarTokenizer(
            vocab: ["a", "</s>"], vocabType: .raw, eosTokenId: 5)
        #expect(grammarTokenizer.stopTokenIDs.isEmpty)

        let constraint = try GrammarConstraint(
            tokenizer: grammarTokenizer, grammar: #"root ::= "a" "</s>""#)
        _ = try constraint.commitToken(0)
        let result = try constraint.commitToken(1)
        #expect(!result.isTerminated)
    }

    /// Without fast-forward, `a` and `b` are sampled; then the grammar needs
    /// `x` as text, and the run throws before the sampler or the processor
    /// runs a third time.
    @Test func stopNeededAsTextWithoutFastForwardThrowsBeforeSampling() throws {
        try Stream.withNewDefaultStream(device: .cpu) {
            let sampler = RecordingSampler()
            let run = try runLoop(
                source: .grammar(#"root ::= "abx" [a-z]*"#), policy: .script("abxy"),
                maxTokens: 64, eosTokenIds: [0x78], fastForward: false, sampler: sampler)
            #expect(
                (run.error as? GuidedGenerationError)
                    == .noAllowedToken(stopTokenIds: [0x78, HarnessTokenizer.eosTokenId]))
            #expect(run.text == "ab")
            #expect(sampler.sampled == [0x61, 0x62])
            #expect(run.processCalls == 2)
        }
    }

    /// A grammar that knows only the EOS keeps today's behavior: the loop
    /// stops on `x` inside the string and returns the truncated JSON.
    @Test func eosOnlyGrammarStillEndsOnAnUndeclaredStop() throws {
        try Stream.withNewDefaultStream(device: .cpu) {
            let run = try runLoop(
                source: .jsonSchema(Self.stringSchema),
                policy: .then(prefix: #"{"s":""#, .prefer([0x78, 0x22, 0x7D, 0xFF])),
                maxTokens: 64, eosTokenIds: [0x78],
                grammarTokenizer: { _ in try HarnessTokenizer().makeGrammarTokenizer() })
            #expect(run.error == nil)
            #expect(run.text == #"{"s":""#)
        }
    }

    /// With the EOS outside the grammar's vocabulary no stop is declared, so
    /// even the complete output `ab` cannot end.
    @Test func eosOutsideVocabularyCannotEndACompleteOutput() throws {
        try Stream.withNewDefaultStream(device: .cpu) {
            let run = try runLoop(
                source: .grammar(#"root ::= "ab""#), policy: .script("ab"), maxTokens: 8,
                grammarTokenizer: { _ in
                    try GrammarTokenizer(
                        vocab: Array(HarnessTokenizer().grammarVocabulary.prefix(255)),
                        vocabType: .byteFallback, eosTokenId: 255)
                })
            #expect((run.error as? GuidedGenerationError) == .noAllowedToken(stopTokenIds: []))
            #expect(run.text == "ab")
        }
    }

    /// The recipe in README.md builds the same set as the loop.
    @Test func readmeRecipeMatchesTheLoopStopSet() {
        let tokenizer = HarnessTokenizer(unknownByte: 0x7A)
        var configuration = ModelConfiguration(id: "readme", extraEOSTokens: ["x"])
        configuration.eosTokenIds = [0x79, 300]
        let stopTokenIds =
            Array(configuration.eosTokenIds)
            + [tokenizer.eosTokenId, tokenizer.unknownTokenId].compactMap { $0 }
            + configuration.extraEOSTokens.compactMap { tokenizer.convertTokenToId($0) }
        #expect(
            Set(stopTokenIds)
                == GuidedGenerationLoop.stopTokenIDs(
                    tokenizer: tokenizer, configuration: configuration))
        #expect(Set(stopTokenIds) == [0x78, 0x79, 0x7A, 255, 300])
    }

    @Test func maskAllowsAnyTokenIgnoresBitsBeyondVocab() {
        func mask(_ words: [UInt32]) -> MaskResult {
            MaskResult(
                mask: words.map { Int32(bitPattern: $0) }, isTerminated: false, needsApply: true)
        }
        #expect(!GuidedGenerationLoop.maskAllowsAnyToken(mask([0, 0]), vocabSize: 40))
        #expect(!GuidedGenerationLoop.maskAllowsAnyToken(mask([0, 1 << 8]), vocabSize: 40))
        #expect(GuidedGenerationLoop.maskAllowsAnyToken(mask([0, 1 << 7]), vocabSize: 40))
        #expect(GuidedGenerationLoop.maskAllowsAnyToken(mask([1 << 31, 0]), vocabSize: 40))
        #expect(GuidedGenerationLoop.maskAllowsAnyToken(mask([0, 1 << 31]), vocabSize: 64))
    }
}
