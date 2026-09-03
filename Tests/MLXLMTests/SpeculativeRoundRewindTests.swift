// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLMCommon

/// Deterministic causal model that keeps a real sliding-window ring.
///
/// Logits depend only on the input token (high-margin affine transition), so
/// batched verification and token-by-token decoding compute the same function;
/// the K/V writes are real, so speculative rounds run against a genuine
/// `RotatingKVCache` — including past the wrap, where the ring stops being
/// trimmable and rejected drafts can only be taken back by a staged round.
private final class RingTransitionModel: Module, LanguageModel, KVCacheDimensionProvider {
    let vocabularySize: Int
    let windowSize: Int
    var kvHeads: [Int] { [1] }

    init(vocabularySize: Int, windowSize: Int) {
        self.vocabularySize = vocabularySize
        self.windowSize = windowSize
        super.init()
    }

    func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
        [RotatingKVCache(maxSize: windowSize, keep: 0)]
    }

    func prepare(
        _ input: LMInput, cache: [KVCache], state _: LMOutput.State?, prefill _: PrefillParameters
    ) throws -> PrepareResult {
        .tokens(input.text)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let tokenIds = inputs.reshaped(-1).asArray(Int.self)
        if let first = cache?.first {
            let entry = MLXArray.zeros([1, 1, tokenIds.count, 4])
            _ = first.update(keys: entry, values: entry)
        }
        var logits = [Float](repeating: -100, count: tokenIds.count * vocabularySize)
        for (position, token) in tokenIds.enumerated() {
            logits[position * vocabularySize + (token * 31 + 7) % vocabularySize] = 100
        }
        return MLXArray(logits, [1, tokenIds.count, vocabularySize])
    }
}

/// Cache-less deterministic model; `multiplier`/`increment` select whether it
/// agrees with `RingTransitionModel` (31, 7) or diverges to force rejections.
private final class CachelessTransitionModel: Module, LanguageModel, KVCacheDimensionProvider {
    let vocabularySize: Int
    let multiplier: Int
    let increment: Int
    var kvHeads: [Int] { [] }

    init(vocabularySize: Int, multiplier: Int = 31, increment: Int = 7) {
        self.vocabularySize = vocabularySize
        self.multiplier = multiplier
        self.increment = increment
        super.init()
    }

    func prepare(
        _ input: LMInput, cache: [KVCache], state _: LMOutput.State?, prefill _: PrefillParameters
    ) throws -> PrepareResult {
        .tokens(input.text)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let tokenIds = inputs.reshaped(-1).asArray(Int.self)
        var logits = [Float](repeating: -100, count: tokenIds.count * vocabularySize)
        for (position, token) in tokenIds.enumerated() {
            logits[position * vocabularySize + (token * multiplier + increment) % vocabularySize] =
                100
        }
        return MLXArray(logits, [1, tokenIds.count, vocabularySize])
    }
}

/// Unknown-to-the-round-factory cache that mirrors `RotatingKVCache` through
/// the predictive trimmability query: once `wrapAt` is crossed it can neither
/// be staged (unsupported kind) nor trimmed, which is exactly the topology the
/// iterator's passthrough fallback exists for.
private final class WrapCountingKVCache: KVCache {
    var offset: Int = 0
    let wrapAt: Int

    init(wrapAt: Int) {
        self.wrapAt = wrapAt
    }

    var maxSize: Int? { wrapAt }

    func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        offset += keys.dim(-2)
        return (keys, values)
    }

    var state: [MLXArray] {
        get { [] }
        set {}
    }

    var metaState: [String] {
        get { [] }
        set {}
    }

    var isTrimmable: Bool { isTrimmable(after: 0) }

    func isTrimmable(after positions: Int) -> Bool {
        offset + positions < wrapAt
    }

    @discardableResult
    func trim(_ n: Int) -> Int {
        guard isTrimmable else { return 0 }
        let removed = Swift.min(n, offset)
        offset -= removed
        return removed
    }

    func makeMask(
        n: Int, windowSize: Int?, returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        .none
    }

    func copy() -> any KVCache {
        let copy = WrapCountingKVCache(wrapAt: wrapAt)
        copy.offset = offset
        return copy
    }

    func innerState() -> [MLXArray] { [] }
}

/// `CachelessTransitionModel` variant that owns a `WrapCountingKVCache`.
private final class WrapCachedTransitionModel: Module, LanguageModel, KVCacheDimensionProvider {
    let vocabularySize: Int
    let wrapAt: Int
    let multiplier: Int
    let increment: Int
    var kvHeads: [Int] { [1] }

    init(vocabularySize: Int, wrapAt: Int, multiplier: Int = 31, increment: Int = 7) {
        self.vocabularySize = vocabularySize
        self.wrapAt = wrapAt
        self.multiplier = multiplier
        self.increment = increment
        super.init()
    }

    func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
        [WrapCountingKVCache(wrapAt: wrapAt)]
    }

    func prepare(
        _ input: LMInput, cache: [KVCache], state _: LMOutput.State?, prefill _: PrefillParameters
    ) throws -> PrepareResult {
        .tokens(input.text)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let tokenIds = inputs.reshaped(-1).asArray(Int.self)
        if let first = cache?.first {
            let entry = MLXArray.zeros([1, 1, tokenIds.count, 4])
            _ = first.update(keys: entry, values: entry)
        }
        var logits = [Float](repeating: -100, count: tokenIds.count * vocabularySize)
        for (position, token) in tokenIds.enumerated() {
            logits[position * vocabularySize + (token * multiplier + increment) % vocabularySize] =
                100
        }
        return MLXArray(logits, [1, tokenIds.count, vocabularySize])
    }
}

@Suite("Speculative round rewind", .serialized)
struct SpeculativeRoundRewindTests {

    private let vocabularySize = 100
    /// Opening span of the (t * 31 + 7) % 100 orbit.
    private let prompt: [Int32] = [0, 7, 24, 51, 88, 35, 92]

    private func plainTokens(model: any LanguageModel, maxTokens: Int) async throws -> [Int] {
        let processor = TestInputProcessor(
            tokenizer: TestTokenizer(vocabularySize: vocabularySize),
            configuration: ModelConfiguration(id: "round-rewind-test"),
            messageGenerator: DefaultMessageGenerator())
        let context = ModelContext(
            configuration: processor.configuration,
            model: model,
            processor: processor,
            tokenizer: processor.tokenizer)
        var tokens = [Int]()
        for await generation in try generateTokens(
            input: LMInput(tokens: MLXArray(prompt)),
            parameters: GenerateParameters(maxTokens: maxTokens, temperature: 0.0),
            context: context
        ) {
            if let token = generation.token { tokens.append(token) }
        }
        return tokens
    }

    private func speculativeRun(
        mainModel: any LanguageModel,
        draftModel: any LanguageModel,
        maxTokens: Int,
        numDraftTokens: Int
    ) throws -> (tokens: [Int], iterator: SpeculativeTokenIterator) {
        var iterator = try SpeculativeTokenIterator(
            input: LMInput(tokens: MLXArray(prompt)),
            mainModel: mainModel,
            draftModel: draftModel,
            parameters: GenerateParameters(maxTokens: maxTokens, temperature: 0.0),
            numDraftTokens: numDraftTokens)
        var tokens = [Int]()
        while let token = iterator.next() {
            tokens.append(token)
        }
        return (tokens, iterator)
    }

    @Test(arguments: [2, 3, 4])
    func stagedRoundsMatchPlainGenerationAcrossTheWrap(drafts: Int) async throws {
        // Window 16, prompt 7, 48 generated tokens: the ring wraps early and
        // most rounds run in the regime a trim cannot rewind.
        let main = RingTransitionModel(vocabularySize: vocabularySize, windowSize: 16)
        let plain = try await plainTokens(model: main, maxTokens: 48)
        #expect(plain.count == 48)

        let accepting = try speculativeRun(
            mainModel: main,
            draftModel: CachelessTransitionModel(vocabularySize: vocabularySize),
            maxTokens: 48, numDraftTokens: drafts)
        #expect(accepting.tokens == plain)
        #expect(accepting.iterator.passthroughReason == nil)
        let telemetry = try #require(accepting.iterator.speculativeDecodingTelemetry)
        #expect(telemetry.acceptanceRate > 0.9)
    }

    @Test(arguments: [2, 3, 4])
    func rejectedRoundsRewindExactlyAcrossTheWrap(drafts: Int) async throws {
        // A divergent drafter makes nearly every round commit `retaining` well
        // below what it wrote — past the wrap that rewind is the staged path.
        let main = RingTransitionModel(vocabularySize: vocabularySize, windowSize: 16)
        let plain = try await plainTokens(model: main, maxTokens: 48)

        let rejecting = try speculativeRun(
            mainModel: main,
            draftModel: CachelessTransitionModel(
                vocabularySize: vocabularySize, multiplier: 17, increment: 13),
            maxTokens: 48, numDraftTokens: drafts)
        #expect(rejecting.tokens == plain)
        #expect(rejecting.iterator.passthroughReason == nil)
    }

    @Test func unstageableWrappedMainCacheFallsBackToPassthrough() async throws {
        // An unknown cache type is admitted write-through only while it can
        // still trim; once it wraps the iterator must stop speculating and
        // keep emitting the plain stream.
        let main = WrapCachedTransitionModel(vocabularySize: vocabularySize, wrapAt: 12)
        let plain = try await plainTokens(model: main, maxTokens: 32)
        #expect(plain.count == 32)

        let run = try speculativeRun(
            mainModel: main,
            draftModel: CachelessTransitionModel(vocabularySize: vocabularySize),
            maxTokens: 32, numDraftTokens: 3)
        #expect(run.tokens == plain)
        let reason = try #require(run.iterator.passthroughReason)
        #expect(reason.contains("main KV cache"))
    }

    @Test func draftCacheThatCannotRewindFallsBackToPassthrough() async throws {
        // The drafter's own sliding window wraps; a rejected round then cannot
        // take its drafts back out of the draft cache, so speculation stops
        // rather than propose from a corrupted history.
        let main = CachelessTransitionModel(vocabularySize: vocabularySize)
        let plain = try await plainTokens(model: main, maxTokens: 32)

        let run = try speculativeRun(
            mainModel: main,
            draftModel: WrapCachedTransitionModel(
                vocabularySize: vocabularySize, wrapAt: 10, multiplier: 17, increment: 13),
            maxTokens: 32, numDraftTokens: 3)
        #expect(run.tokens == plain)
        let reason = try #require(run.iterator.passthroughReason)
        #expect(reason.contains("draft KV cache"))
    }

    @Test func cachelessDraftModelsKeepSpeculatingThroughRejections() async throws {
        // `trim` on an empty cache list reports zero; that must not be read as
        // a rewind failure for draft models that keep no KV state at all.
        let main = CachelessTransitionModel(vocabularySize: vocabularySize)
        let plain = try await plainTokens(model: main, maxTokens: 24)

        let run = try speculativeRun(
            mainModel: main,
            draftModel: CachelessTransitionModel(
                vocabularySize: vocabularySize, multiplier: 17, increment: 13),
            maxTokens: 24, numDraftTokens: 3)
        #expect(run.tokens == plain)
        #expect(run.iterator.passthroughReason == nil)
    }
}
