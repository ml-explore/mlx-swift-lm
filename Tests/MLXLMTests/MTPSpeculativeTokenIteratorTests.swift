// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN
import Testing

@_spi(Testing) @testable import MLXLMCommon

// MARK: - Synthetic mocks for iterator plumbing

private let preservedStateKey = LMOutput.Key<Int>("tests.mtp.preservedState")

/// Records `draftBlock(...)` invocations and returns a fixed token pattern
/// so the iterator's draft/verify/accept flow can be exercised without a
/// real drafter.
private final class MockDrafter: Module, StatefulMTPDrafterModel {
    private(set) var draftBlockCallCount = 0
    var draftedTokenValue: Int32
    let requiresGreedySampling: Bool
    /// Per-call record of what the iterator handed `draftBlock`: the
    /// sequence-axis span of each sharedKV entry, and the query offset.
    /// Lets tests assert the state the drafter conditions on, not just the
    /// tokens that come out of it.
    private(set) var receivedSharedKVSpans: [[String: Int]] = []
    private(set) var receivedQueryOffsets: [Int] = []
    private(set) var receivedPositionDeltaValues: [Int?] = []
    private(set) var receivedCacheOffsets: [Int?] = []

    init(draftedTokenValue: Int32 = 7, requiresGreedySampling: Bool = false) {
        self.draftedTokenValue = draftedTokenValue
        self.requiresGreedySampling = requiresGreedySampling
        super.init()
    }

    func makeState(parameters: GenerateParameters?) -> MTPDrafterState {
        MTPDrafterState(cache: [CountingKVCache()])
    }

    func draftBlock(
        target: any LanguageModel,
        lastToken: MLXArray,
        lastHidden: MLXArray,
        sharedKV: [String: (MLXArray, MLXArray)],
        positionDeltas: MLXArray?,
        queryOffset: Int,
        blockSize: Int,
        sampler: any LogitSampler
    ) -> MLXArray {
        var state = makeState(parameters: nil)
        return draftBlock(
            target: target,
            lastToken: lastToken,
            lastHidden: lastHidden,
            sharedKV: sharedKV,
            positionDeltas: positionDeltas,
            queryOffset: queryOffset,
            blockSize: blockSize,
            state: &state,
            sampler: sampler)
    }

    func draftBlock(
        target: any LanguageModel,
        lastToken: MLXArray,
        lastHidden: MLXArray,
        sharedKV: [String: (MLXArray, MLXArray)],
        positionDeltas: MLXArray?,
        queryOffset: Int,
        blockSize: Int,
        state: inout MTPDrafterState,
        sampler: any LogitSampler
    ) -> MLXArray {
        draftBlockCallCount += 1
        receivedSharedKVSpans.append(sharedKV.mapValues { $0.0.dim(-2) })
        receivedQueryOffsets.append(queryOffset)
        receivedPositionDeltaValues.append(positionDeltas?.item(Int.self))
        let mutableCache = state.cache.first as? MutableOffsetKVCache
        receivedCacheOffsets.append(mutableCache?.offset)
        mutableCache?.offset += blockSize - 1
        let batch = lastToken.dim(0)
        let vals = Array(repeating: draftedTokenValue, count: (blockSize - 1) * batch)
        return MLXArray(vals, [batch, blockSize - 1])
    }
}

/// Minimal `LanguageModel` mock that emits MTP state on every call when
/// `mtpEmitFlagKey` is true. Returns shaped logits and a trimmable KV cache.
private final class MockMainModel: Module, LanguageModel, KVCacheDimensionProvider {
    var kvHeads: [Int] { [1] }
    /// Sequence of token values returned in increasing position order. Length
    /// must cover all positions the iterator will sample across the run.
    var nextLogitTokens: [Int32]
    var perPositionIndex: Int = 0
    /// If `true`, omit the MTP state keys from the returned `LMOutput` so the
    /// iterator's passthrough fallback is exercised.
    var omitDrafterState: Bool = false
    var emitEmptySharedKVState: Bool = false
    var maxEmittedSharedKVSpan: Int?
    var returnPreservedStateWhenOmittingDrafterState: Bool = false
    var prepareLogitsStateValue: Int?
    var emittedPositionDelta: Int?

    private(set) var callCount: Int = 0
    private(set) var lastIncomingEmitFlag: Bool? = nil
    private(set) var incomingPreservedStateValues: [Int?] = []
    /// Sequence-axis span of each emitted sharedKV snapshot, in emit order.
    /// Tests assert this against the mock cache offset to pin the mock's
    /// span fidelity (see the emit block below).
    private(set) var emittedSharedKVSpans: [Int] = []

    init(nextLogitTokens: [Int32]) {
        self.nextLogitTokens = nextLogitTokens
        super.init()
    }

    func prepare(
        _ input: LMInput, cache: [KVCache], state _: LMOutput.State?, prefill _: PrefillParameters
    )
        throws -> PrepareResult
    {
        if let prepareLogitsStateValue {
            var state = LMOutput.State()
            state[preservedStateKey] = prepareLogitsStateValue
            if let first = cache.first as? MutableOffsetKVCache {
                first.offset += input.text.cacheSequenceLength
            }
            let logits = makeLogits(positions: input.text.tokens.dim(-1))
            return .logits(LMOutput(logits: logits, state: state))
        }
        // Return `.tokens(...)`; the iterator's `prepare` will follow up with
        // a one-position forward call that primes drafter state.
        return .tokens(input.text)
    }

    /// Returns deterministic one-hot logits at each position so a `softmax/
    /// argmax` sampler picks the planned token sequence.
    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let positions = inputs.dim(-1)
        return makeLogits(positions: positions)
    }

    func callAsFunction(
        _ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?
    ) -> LMOutput {
        callCount += 1
        lastIncomingEmitFlag = state?[mtpEmitFlagKey]
        incomingPreservedStateValues.append(state?[preservedStateKey])

        let positions = input.tokens.dim(-1)
        let logits = makeLogits(positions: positions)

        // Update the mock cache to reflect that `positions` tokens were seen.
        if let first = cache?.first {
            if let mutable = first as? MutableOffsetKVCache {
                mutable.offset += positions
            } else {
                let keys = MLXArray.zeros([1, 1, positions, 4])
                _ = first.update(keys: keys, values: keys)
            }
        }

        if !omitDrafterState, state?[mtpEmitFlagKey] ?? false {
            // Mirror the real emit hook's spans: sharedKV is captured from
            // what the attention layers consumed, so its sequence axis covers
            // the cache's full logical offset (everything processed so far);
            // lastHidden is the forward's own output, so it covers only the
            // current chunk. Span-accurate sharedKV emission is what lets a
            // test distinguish a trimmed snapshot from a stale one — with a
            // fixed-span mock those are indistinguishable.
            let kvOffset = cache?.first?.offset ?? positions
            let kvSpan = maxEmittedSharedKVSpan.map { Swift.min(kvOffset, $0) } ?? kvOffset
            emittedSharedKVSpans.append(kvSpan)
            var out = LMOutput.State()
            out[preservedStateKey] = state?[preservedStateKey]
            out[mtpLastHiddenStatesKey] = MLXArray.zeros([1, positions, 4])
            if emitEmptySharedKVState {
                out[mtpSharedKVStatesKey] = [:]
                out[mtpSharedKVSourceIndicesKey] = [:]
            } else {
                // One cache entry stands in for both layer types here.
                out[mtpSharedKVSourceIndicesKey] = [
                    "full_attention": 0, "sliding_attention": 0,
                ]
                out[mtpSharedKVStatesKey] = [
                    "full_attention": (
                        MLXArray.zeros([1, 1, kvSpan, 4]),
                        MLXArray.zeros([1, 1, kvSpan, 4])
                    ),
                    "sliding_attention": (
                        MLXArray.zeros([1, 1, kvSpan, 4]),
                        MLXArray.zeros([1, 1, kvSpan, 4])
                    ),
                ]
            }
            out[mtpSharedKVOffsetsKey] = ["full_attention": kvOffset]
            if let emittedPositionDelta {
                out[mtpPositionDeltasKey] = MLXArray(emittedPositionDelta)
            }
            return LMOutput(logits: logits, state: out)
        }
        if returnPreservedStateWhenOmittingDrafterState {
            var out = LMOutput.State()
            out[preservedStateKey] = state?[preservedStateKey]
            return LMOutput(logits: logits, state: out)
        }
        return LMOutput(logits: logits)
    }

    func newCache(parameters: GenerateParameters?) -> [KVCache] {
        [CountingKVCache()]
    }

    private func makeLogits(positions: Int) -> MLXArray {
        // [B=1, positions, vocab=20]. One-hot at the planned token for each
        // position, taken from `nextLogitTokens[perPositionIndex...]`.
        let vocab = 20
        var data = [Float](repeating: 0, count: positions * vocab)
        for i in 0 ..< positions {
            let tokIdx = perPositionIndex + i
            let tok = tokIdx < nextLogitTokens.count ? Int(nextLogitTokens[tokIdx]) : 0
            data[i * vocab + tok] = 100
        }
        perPositionIndex += positions
        return MLXArray(data, [1, positions, vocab])
    }
}

/// Minimal `KVCache` that satisfies the protocol's trimmable interface; the
/// mock model adjusts `offset` directly. Inherits the default
/// `ropeOffset = .scalar(offset)` from the `KVCache` protocol extension.
///
/// `wrapAt` is opt-in and defaults to nil (unbounded, always trimmable), which
/// is the behavior every pre-existing test relies on. Setting it mirrors
/// `RotatingKVCache` through the predictive trimmability query, so a mock stream
/// can cross the window the same way a real sliding-window cache does.
private protocol MutableOffsetKVCache: KVCache, AnyObject {
    var offset: Int { get set }
}

private class CountingKVCache: MutableOffsetKVCache {
    var offset: Int = 0
    let wrapAt: Int?
    init(wrapAt: Int? = nil) {
        self.wrapAt = wrapAt
    }
    var maxSize: Int? { wrapAt }
    func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        (keys, values)
    }
    var state: [MLXArray] {
        get { [] }
        set {}
    }
    var metaState: [String] {
        get { [] }
        set {}
    }
    var isTrimmable: Bool {
        isTrimmable(after: 0)
    }
    func isTrimmable(after positions: Int) -> Bool {
        guard let wrapAt else { return true }
        return offset + positions < wrapAt
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
        let c = CountingKVCache(wrapAt: wrapAt)
        c.offset = offset
        return c
    }
    func innerState() -> [MLXArray] { [] }
}

private final class NonTrimmableCountingKVCache: CountingKVCache {
    override var isTrimmable: Bool { false }

    @discardableResult
    override func trim(_ n: Int) -> Int { 0 }

    override func copy() -> any KVCache {
        let c = NonTrimmableCountingKVCache()
        c.offset = offset
        return c
    }
}

// MARK: - Smallest-unit-of-work smoke test

@Suite("MTP KV-cache configuration")
struct MTPKVCacheConfigurationTests {
    @Test func legacyTurboSchemeUsesTypedDispatcher() throws {
        let main = MockMainModel(nextLogitTokens: [0, 0, 7])
        let drafter = MockDrafter(draftedTokenValue: 7)
        let input = LMInput(tokens: MLXArray([Int32(1), 2, 3]))
        let iterator = try MTPSpeculativeTokenIterator(
            input: input,
            mainModel: main,
            drafter: drafter,
            parameters: GenerateParameters(kvScheme: "turbo0v4"),
            blockSize: 2)
        let simple = KVCacheSimple()
        simple.offset = 1
        var cache: [KVCache] = [simple]

        iterator.kvCachePlan.apply(to: &cache)

        #expect(cache[0] is TurboQuantKVCache)
    }
}

@Test
func testMTPSpeculateRoundSmokeWithSynthetics() throws {
    // Plan: prompt of 3 tokens [1, 2, 3], main model is rigged so that it
    // samples bonus token 7 at prefill, then on the verify pass samples
    // [7, 7, 7, 9] (3 matching drafts, 1 correction). Drafter returns
    // [7, 7, 7] so all three drafts match. Total tokens yielded across
    // the bonus drain + one speculation round: [bonus=7, draft=7,
    // draft=7, draft=7, correction=9].

    let mainLogitTokens: [Int32] = [
        // Prefill follow-up call (length 3): only the final position is
        // sampled (iterator takes `logits[-1]`); positions 0/1 can hold any
        // value < vocab=20. Using 0 as an inert placeholder.
        0, 0, 7,
        // Verify pass (length 4): [7, 7, 7, 9]
        7, 7, 7, 9,
    ]
    let main = MockMainModel(nextLogitTokens: mainLogitTokens)
    let drafter = MockDrafter(draftedTokenValue: 7)
    let promptTokens = MLXArray([Int32(1), 2, 3])
    let input = LMInput(tokens: promptTokens)

    var iter = try MTPSpeculativeTokenIterator(
        input: input,
        mainModel: main,
        drafter: drafter,
        mainCache: nil,
        parameters: GenerateParameters(maxTokens: 8),
        blockSize: 4
    )

    // First token drained from `next()` is the prepare-time bonus; the
    // speculation round runs on the second call.
    let t0 = iter.next()
    #expect(t0 == 7)

    // Drain the speculation round's pending buffer.
    let t1 = iter.next()
    let t2 = iter.next()
    let t3 = iter.next()
    let t4 = iter.next()
    #expect(t1 == 7)
    #expect(t2 == 7)
    #expect(t3 == 7)
    #expect(t4 == 9)

    // 1 prepare bonus + one round of 4 tokens (3 accepted + 1 correction).
    #expect(iter.tokenCount == 5)
    #expect(drafter.draftBlockCallCount == 1)
    // proposedCount = numDraft = 3; accepted = 3.
    #expect(iter.proposedCount == 3)
    #expect(iter.acceptedCount == 3)
    let telemetry = try #require(iter.speculativeDecodingTelemetry)
    #expect(telemetry.roundCount == 1)
    #expect(telemetry.draftTokenCount == 3)
    #expect(telemetry.acceptedDraftTokenCount == 3)
    #expect(telemetry.targetModelCallCount == 1)
    #expect(telemetry.draftModelCallCount == 1)
    #expect(telemetry.targetVerifiedTokenCount == 4)
    #expect(telemetry.emittedTokenCount == iter.tokenCount)
    // Verify the main model received emit=true on every call after prefill.
    #expect(main.lastIncomingEmitFlag == true)
}

@Test
func testMTPIteratorForwardsPositionDeltasToDrafter() throws {
    let mainLogitTokens: [Int32] = [
        0, 0, 7,
        7, 7, 7, 9,
    ]
    let main = MockMainModel(nextLogitTokens: mainLogitTokens)
    main.emittedPositionDelta = 11
    let drafter = MockDrafter(draftedTokenValue: 7)
    let input = LMInput(tokens: MLXArray([Int32(1), 2, 3]))

    var iter = try MTPSpeculativeTokenIterator(
        input: input,
        mainModel: main,
        drafter: drafter,
        mainCache: nil,
        parameters: GenerateParameters(maxTokens: 8),
        blockSize: 4
    )

    _ = iter.next()
    _ = iter.next()

    #expect(drafter.receivedPositionDeltaValues == [11])
}

// MARK: - Passthrough fallback when state is absent

@Test
func testMTPIteratorMissingStateFallsBackToPassthrough() throws {
    // Main model with `omitDrafterState=true` never populates the MTP keys,
    // so the iterator switches to passthrough on the first `speculateRound`
    // call. Drafter must not be invoked.
    //
    // With maxTokens=3, the iterator yields 1 prepare-time bonus + 2
    // passthrough tokens before hitting the budget. The third passthrough
    // position is never reached.
    let mainLogitTokens: [Int32] = [
        // Prefill follow-up: length 3, final position picks bonus 5
        // (positions 0/1 are not sampled and must be < vocab=20; using 0 as
        // a placeholder).
        0, 0, 5,
        // Passthrough single-token rounds (length-1 calls): the iterator
        // takes 2 of these inside the 3-token budget after yielding the
        // bonus first.
        11, 12,
    ]
    let main = MockMainModel(nextLogitTokens: mainLogitTokens)
    main.omitDrafterState = true
    let drafter = MockDrafter()
    let input = LMInput(tokens: MLXArray([Int32(1), 2, 3]))

    var iter = try MTPSpeculativeTokenIterator(
        input: input, mainModel: main, drafter: drafter, mainCache: nil,
        parameters: GenerateParameters(maxTokens: 3), blockSize: 4
    )

    let tokens = [iter.next(), iter.next(), iter.next(), iter.next()]
    // [bonus from prepare, 2 passthrough tokens, nil].
    #expect(tokens[0] == 5)
    #expect(tokens[1] == 11)
    #expect(tokens[2] == 12)
    #expect(tokens[3] == nil)
    // Drafter was never invoked for an actual round.
    #expect(drafter.draftBlockCallCount == 0)
}

@Test
func testMTPIteratorEmptySharedKVFallsBackToPassthrough() throws {
    // Qwen35 can still return the sharedKV state key when the underlying
    // full-attention cache no longer exposes regular K/V, e.g. after cache
    // quantization. An empty dictionary must be treated as missing drafter
    // state, otherwise the iterator drafts with no usable K/V anchor.
    let mainLogitTokens: [Int32] = [
        0, 0, 5,  // prefill follow-up picks bonus 5
        // passthrough tokens after empty sharedKV is detected
        11, 12,
    ]
    let main = MockMainModel(nextLogitTokens: mainLogitTokens)
    main.emitEmptySharedKVState = true
    let drafter = MockDrafter()
    let input = LMInput(tokens: MLXArray([Int32(1), 2, 3]))

    var iter = try MTPSpeculativeTokenIterator(
        input: input, mainModel: main, drafter: drafter, mainCache: nil,
        parameters: GenerateParameters(maxTokens: 3), blockSize: 4
    )

    #expect(iter.next() == 5)
    #expect(iter.next() == 11)
    #expect(iter.next() == 12)
    #expect(iter.next() == nil)
    #expect(drafter.draftBlockCallCount == 0)
    #expect(iter.passthroughReason == "main model did not emit shared target K/V")
}

// MARK: - Pending buffer drain order

@Test
func testMTPIteratorPendingBufferDrainOrder() throws {
    // Drafter returns [5, 5, 5]; main verifies [5, 5, 7, 9].
    // After the prepare-time bonus (5) is yielded first, speculateRound's
    // accept-prefix is positions 0..1 → [5, 5], then correction at position
    // 2 = 7. The pendingTokens order inside the round should be [5, 5, 7] —
    // main-model sequence order — not the drafter's [5, 5, 5]. Total
    // stream: [bonus=5, draft=5, draft=5, correction=7].
    let mainLogitTokens: [Int32] = [
        0, 0, 5,  // prefill follow-up picks bonus 5 (positions 0/1 unused, < vocab=20)
        5, 5, 7, 9,  // verify positions
    ]
    let main = MockMainModel(nextLogitTokens: mainLogitTokens)
    let drafter = MockDrafter(draftedTokenValue: 5)
    let input = LMInput(tokens: MLXArray([Int32(1), 2, 3]))

    var iter = try MTPSpeculativeTokenIterator(
        input: input, mainModel: main, drafter: drafter, mainCache: nil,
        parameters: GenerateParameters(maxTokens: 8), blockSize: 4
    )

    let t0 = iter.next()
    let t1 = iter.next()
    let t2 = iter.next()
    let t3 = iter.next()
    #expect(t0 == 5)  // bonus from prepare
    #expect(t1 == 5)  // first accepted draft
    #expect(t2 == 5)  // second accepted draft
    #expect(t3 == 7)  // correction at the rejected position
    // proposedCount = 3 (numDraft); accepted = 2.
    #expect(iter.proposedCount == 3)
    #expect(iter.acceptedCount == 2)
}

@Test
func testMTPIteratorUsesSingleStepWhenOnlyOneTokenRemains() throws {
    // With maxTokens=2, the prepare-time bonus consumes the first output slot.
    // Only one slot remains, so a speculative round would be unable to emit
    // both an accepted draft and the verifier's correction/bonus token. The
    // iterator must take one normal main-model step instead.
    let mainLogitTokens: [Int32] = [
        0, 0, 5,  // prefill follow-up picks bonus 5
        11,  // single-token tail step
    ]
    let main = MockMainModel(nextLogitTokens: mainLogitTokens)
    let drafter = MockDrafter(draftedTokenValue: 11)
    let input = LMInput(tokens: MLXArray([Int32(1), 2, 3]))
    let cache = CountingKVCache()

    var iter = try MTPSpeculativeTokenIterator(
        input: input, mainModel: main, drafter: drafter, mainCache: [cache],
        parameters: GenerateParameters(maxTokens: 2), blockSize: 4
    )

    let tokens = [iter.next(), iter.next(), iter.next()]

    #expect(tokens[0] == 5)
    #expect(tokens[1] == 11)
    #expect(tokens[2] == nil)
    #expect(iter.tokenCount == 2)
    #expect(drafter.draftBlockCallCount == 0)
    #expect(iter.proposedCount == 0)
    #expect(iter.acceptedCount == 0)
    #expect(cache.offset == 4)
}

// MARK: - LogitProcessor emit-only invariant

/// Reference-backed processor proving verifier logic does not rely on an
/// existential assignment being a deep copy.
private final class EmissionLog: LogitProcessor {
    var recordedTokens: [Int] = []

    func prompt(_ prompt: MLXArray) {}

    func process(logits: MLXArray) -> MLXArray { logits }

    func didSample(token: MLXArray) {
        recordedTokens.append(token.item(Int.self))
    }

    func copy() -> Self {
        let copy = EmissionLog()
        copy.recordedTokens = recordedTokens
        return copy as! Self
    }
}

/// The canonical processor advances only for emitted tokens. In particular,
/// a class conformer must not observe verifier rows after the first mismatch.
///
/// Test scenario: bs=4 (numDraft=3), drafter proposes [5, 5, 5], main
/// verifies and samples [5, 9, 1, 2] — only position 0 matches the draft.
/// accepted=1, correction=9, emitted=[bonus=5, draft=5, correction=9].
/// The processor's `didSample` should fire exactly twice (for emitted [5, 9])
/// — never for [1, 2]. The probe processor is installed AFTER init so the
/// prepare-time bonus is not recorded; the test asserts on speculation-
/// round emissions only.
@Test
func testMTPVerifyLoopMutatesClassProcessorForEmittedTokensOnly() throws {
    let mainLogitTokens: [Int32] = [
        0, 0, 5,  // prefill follow-up picks bonus 5 (positions 0/1 unused, < vocab=20)
        5, 9, 1, 2,  // verify positions: only position 0 matches draft
    ]
    let main = MockMainModel(nextLogitTokens: mainLogitTokens)
    let drafter = MockDrafter(draftedTokenValue: 5)
    let input = LMInput(tokens: MLXArray([Int32(1), 2, 3]))

    // maxTokens larger than the test's emit budget so `speculateRound`'s
    // `numDraft = min(remaining, blockSize - 1)` doesn't get capped — we
    // need the full numDraft=3 verify pass to exercise the invariant
    // (4 verify-position samples vs only 2 emitted tokens). Control the
    // round count by manual `next()` calls instead of draining.
    var iter = try MTPSpeculativeTokenIterator(
        input: input, mainModel: main, drafter: drafter, mainCache: nil,
        parameters: GenerateParameters(maxTokens: 8), blockSize: 4
    )

    // Install probe AFTER init so the prepare-time bonus didSample (which
    // happens inside init's call to prepare()) hits the parameters-derived
    // processor (nil here, since no penalties were configured) rather than
    // the EmissionLog. The probe records speculation-round emissions only.
    let emissionLog = EmissionLog()
    iter._setProcessorForTesting(emissionLog)

    // Manual drain — exactly 3 calls to cover prepare bonus + 1 accepted
    // draft + correction. Stopping here avoids triggering a second round
    // (which would need more `mainLogitTokens` data and would just retest
    // the same invariant redundantly).
    let t0 = iter.next()
    let t1 = iter.next()
    let t2 = iter.next()
    #expect(t0 == 5, "prepare bonus")
    #expect(t1 == 5, "first accepted draft")
    #expect(t2 == 9, "correction at the first rejected position")
    #expect(iter.proposedCount == 3, "numDraft=3 verify samples expected")
    #expect(iter.acceptedCount == 1, "only draft[0] matched")

    #expect(iter._processorForTesting as? EmissionLog === emissionLog)
    #expect(
        emissionLog.recordedTokens == [5, 9],
        "recordedTokens=\(emissionLog.recordedTokens) — expected one accepted draft and one correction only"
    )
}

@Test
func testQwenStyleMTPRequiresGreedySampling() throws {
    let main = MockMainModel(nextLogitTokens: [0, 0, 5, 6])
    let drafter = MockDrafter(draftedTokenValue: 5, requiresGreedySampling: true)
    let input = LMInput(tokens: MLXArray([Int32(1), 2, 3]))
    var iter = try MTPSpeculativeTokenIterator(
        input: input, mainModel: main, drafter: drafter,
        parameters: GenerateParameters(maxTokens: 2, temperature: 0.6), blockSize: 2)

    #expect(iter.next() == 5)
    #expect(iter.next() == 6)
    #expect(drafter.draftBlockCallCount == 0)
    #expect(
        iter.passthroughReason
            == "Qwen MTP currently requires temperature == 0; generating without speculation")
}

// MARK: - sharedKV span across partial acceptance

/// The drafter-conditioning observable that output-equivalence testing
/// cannot see: the verify pass emits `mtpSharedKVStatesKey` spanning the
/// full verify chunk, before acceptance is known. After a partial
/// acceptance the snapshot must be rewound in lockstep with the main
/// cache, or the next round's `draftBlock` cross-attends over K/V rows of
/// tokens the main model just rejected. Byte-identity suites verify the
/// target path only; the state the drafter consumes has to be asserted
/// directly. (PR #308 review: discussion_r3391133046,
/// discussion_r3391147261.)
@Test
func testMTPSharedKVSpanTrimmedAfterPartialAcceptance() throws {
    // Round 1: drafter proposes [5, 5, 5]; main verifies [5, 7, ...] —
    // draft_1 accepted, mismatch at draft_2, so rejected = 2. True
    // sequence after round 1: 3 prompt + 1 bonus + 1 accepted = 5.
    // Round 2's draftBlock must receive sharedKV spanning 5 (== cache
    // offset == queryOffset), not the stale verify-chunk span 7.
    let mainLogitTokens: [Int32] = [
        0, 0, 5,  // prefill follow-up picks bonus 5 (placeholders < vocab=20)
        5, 7, 0, 0,  // round-1 verify: accept draft_1, reject at draft_2
        0, 0, 0, 0,  // round-2 verify: placeholders; round 2 only needs to run
    ]
    let main = MockMainModel(nextLogitTokens: mainLogitTokens)
    let drafter = MockDrafter(draftedTokenValue: 5)
    let input = LMInput(tokens: MLXArray([Int32(1), 2, 3]))

    var iter = try MTPSpeculativeTokenIterator(
        input: input, mainModel: main, drafter: drafter, mainCache: nil,
        parameters: GenerateParameters(maxTokens: 12), blockSize: 4
    )

    // Drain the prepare bonus and round 1, then check round-1 accounting
    // before triggering round 2.
    let t0 = iter.next()
    let t1 = iter.next()
    let t2 = iter.next()
    #expect(t0 == 5, "prepare bonus")
    #expect(t1 == 5, "round-1 accepted draft")
    #expect(t2 == 7, "round-1 correction at the first rejected position")
    #expect(iter.acceptedCount == 1, "round 1 must accept exactly 1 of 3")

    // One more `next()` starts round 2 — the draftBlock call under test.
    _ = iter.next()

    // Mock-fidelity gate: every emitted sharedKV span equals the mock
    // cache offset at emit time (prefill 3, round-1 verify 7, round-2
    // verify 9). If this fails, the mock's emission lost span accuracy
    // and the assertions below would be vacuous.
    #expect(main.emittedSharedKVSpans == [3, 7, 9])

    #expect(drafter.draftBlockCallCount == 2)
    // Round 1 drafts from the prefill emission: span 3 at offset 3.
    #expect(
        drafter.receivedSharedKVSpans.first == [
            "full_attention": 3, "sliding_attention": 3,
        ])
    #expect(drafter.receivedQueryOffsets.first == 3)
    #expect(drafter.receivedCacheOffsets.first == 0)
    // Round 2 is the regression surface: the round-1 verify emission
    // spanned 7; after the rewind trims the 2 rejected rows the drafter
    // must see span 5 == cache offset == queryOffset.
    #expect(drafter.receivedQueryOffsets.last == 5)
    #expect(
        drafter.receivedCacheOffsets.last == 1,
        "drafter cache should keep only the one accepted round-1 draft step")
    #expect(
        drafter.receivedSharedKVSpans.last == [
            "full_attention": 5, "sliding_attention": 5,
        ],
        "drafter received spans \(drafter.receivedSharedKVSpans.last ?? [:]) — expected 5 (= cache offset). Span 7 means the emitted snapshot crossed the cache rewind untrimmed."
    )
}

@Test
func testMTPQueryOffsetUsesAbsoluteOffsetWhenSharedKVSpanIsCapped() throws {
    // RotatingKVCache caps the K/V tensor's sequence axis once maxKVSize is
    // reached, but the drafter RoPE position must continue from the absolute
    // cache offset. The target publishes that absolute offset separately.
    let mainLogitTokens: [Int32] = [
        0, 0, 0, 0, 0, 5,  // prefill follow-up picks bonus 5
        5, 9,  // one-token draft round: accept draft, emit correction 9
    ]
    let main = MockMainModel(nextLogitTokens: mainLogitTokens)
    main.maxEmittedSharedKVSpan = 4
    let drafter = MockDrafter(draftedTokenValue: 5)
    let input = LMInput(tokens: MLXArray([Int32(1), 2, 3, 4, 5, 6]))

    var iter = try MTPSpeculativeTokenIterator(
        input: input, mainModel: main, drafter: drafter,
        mainCache: [RotatingKVCache(maxSize: 4, keep: 0)],
        parameters: GenerateParameters(maxTokens: 4), blockSize: 2
    )

    #expect(iter.next() == 5)
    #expect(iter.next() == 5)

    #expect(main.emittedSharedKVSpans.starts(with: [4, 4]))
    #expect(drafter.draftBlockCallCount == 1)
    #expect(drafter.receivedSharedKVSpans == [["full_attention": 4, "sliding_attention": 4]])
    #expect(drafter.receivedQueryOffsets == [6])
}

@Test
func testMTPCarriesMainStateThroughPrimeAndVerify() throws {
    // Qwen VLM stores multimodal RoPE bookkeeping in LMOutput.State. The MTP
    // iterator must keep that state while adding mtpEmitFlagKey.
    let mainLogitTokens: [Int32] = [
        0, 0, 5,  // prepare logits; sample first bonus 5
        5,  // prime follow-up; sample next bonus 5
        5, 7, 0, 0,  // verify accepts one draft then emits correction 7
    ]
    let main = MockMainModel(nextLogitTokens: mainLogitTokens)
    main.prepareLogitsStateValue = 42
    let drafter = MockDrafter(draftedTokenValue: 5)
    let cache = CountingKVCache()
    let input = LMInput(tokens: MLXArray([Int32(1), 2, 3]))

    var iter = try MTPSpeculativeTokenIterator(
        input: input,
        mainModel: main,
        drafter: drafter,
        mainCache: [cache],
        parameters: GenerateParameters(maxTokens: 6),
        blockSize: 4
    )

    _ = iter.next()
    _ = iter.next()
    _ = iter.next()

    #expect(main.incomingPreservedStateValues == [42, 42])
}

@Test
func testMTPPassthroughCarriesMainStateAfterFallback() throws {
    // Qwen VLM still needs its RoPE bookkeeping state after MTP falls back to
    // plain autoregressive generation.
    let mainLogitTokens: [Int32] = [
        0, 0, 5,  // prepare logits; sample first bonus 5
        6,  // prime follow-up; sample next bonus 6
        7,  // passthrough after missing MTP state
    ]
    let main = MockMainModel(nextLogitTokens: mainLogitTokens)
    main.prepareLogitsStateValue = 42
    main.omitDrafterState = true
    main.returnPreservedStateWhenOmittingDrafterState = true
    let drafter = MockDrafter(draftedTokenValue: 5)
    let cache = CountingKVCache()
    let input = LMInput(tokens: MLXArray([Int32(1), 2, 3]))

    var iter = try MTPSpeculativeTokenIterator(
        input: input,
        mainModel: main,
        drafter: drafter,
        mainCache: [cache],
        parameters: GenerateParameters(maxTokens: 3),
        blockSize: 4
    )

    #expect(iter.next() == 5)
    #expect(iter.next() == 6)
    #expect(iter.next() == 7)
    #expect(drafter.draftBlockCallCount == 0)
    #expect(main.incomingPreservedStateValues == [42, 42])
}

// MARK: - reconcileSharedKVState contract

/// Builds a state carrying a sharedKV snapshot with distinguishable content
/// (values encode their flat index) so prefix byte-identity is checkable
/// after a trim, plus a hidden entry that the trim must never touch. The
/// two layer types get different head/dim shapes to catch a slice applied
/// to the wrong axis.
private func makeSharedKVState(span: Int) -> LMOutput.State {
    func arange(_ shape: [Int]) -> MLXArray {
        let count = shape.reduce(1, *)
        return MLXArray(Array(Int32(0) ..< Int32(count)), shape)
    }
    var state = LMOutput.State()
    state[mtpLastHiddenStatesKey] = arange([1, 2, 4])
    state[mtpSharedKVSourceIndicesKey] = ["full_attention": 0, "sliding_attention": 1]
    state[mtpSharedKVStatesKey] = [
        "full_attention": (arange([1, 1, span, 4]), arange([1, 1, span, 4])),
        "sliding_attention": (arange([1, 2, span, 2]), arange([1, 2, span, 2])),
    ]
    state[mtpSharedKVOffsetsKey] = [
        "full_attention": span,
        "sliding_attention": span,
    ]
    return state
}

/// A snapshot with no source map is one the consumer cannot place against cache entries.
private func makeSourcelessSharedKVState(span: Int) -> LMOutput.State {
    var state = makeSharedKVState(span: span)
    state[mtpSharedKVSourceIndicesKey] = nil
    return state
}

@Test
func testTrimSharedKVStateZeroTokensIsNoOp() throws {
    var state: LMOutput.State? = makeSharedKVState(span: 6)
    reconcileSharedKVState(&state, discarding: 0, lengths: { _ in Int.max })

    let kv = try #require(state?[mtpSharedKVStatesKey])
    let original = try #require(makeSharedKVState(span: 6)[mtpSharedKVStatesKey])
    let offsets = try #require(state?[mtpSharedKVOffsetsKey])
    #expect(Set(kv.keys) == Set(original.keys))
    for (layerType, pair) in kv {
        let origPair = try #require(original[layerType])
        #expect(pair.0.shape == origPair.0.shape)
        #expect(allClose(pair.0, origPair.0, rtol: 0, atol: 0).item(Bool.self))
        #expect(allClose(pair.1, origPair.1, rtol: 0, atol: 0).item(Bool.self))
    }
    #expect(offsets == ["full_attention": 6, "sliding_attention": 6])
}

@Test
func testTrimSharedKVStateTrimsTrailingRowsPreservingPrefix() throws {
    var state: LMOutput.State? = makeSharedKVState(span: 6)
    reconcileSharedKVState(&state, discarding: 2, lengths: { _ in Int.max })

    let kv = try #require(state?[mtpSharedKVStatesKey])
    let original = try #require(makeSharedKVState(span: 6)[mtpSharedKVStatesKey])
    let offsets = try #require(state?[mtpSharedKVOffsetsKey])
    #expect(Set(kv.keys) == Set(original.keys))
    for (layerType, pair) in kv {
        let origPair = try #require(original[layerType])
        // Span reduced by exactly the trim on the sequence axis only.
        #expect(pair.0.dim(-2) == 4)
        #expect(pair.1.dim(-2) == 4)
        #expect(pair.0.shape.dropLast(2).elementsEqual(origPair.0.shape.dropLast(2)))
        // Retained prefix is byte-identical to the original's prefix.
        #expect(
            allClose(pair.0, origPair.0[.ellipsis, ..<4, 0...], rtol: 0, atol: 0)
                .item(Bool.self))
        #expect(
            allClose(pair.1, origPair.1[.ellipsis, ..<4, 0...], rtol: 0, atol: 0)
                .item(Bool.self))
    }
    #expect(offsets == ["full_attention": 4, "sliding_attention": 4])

    // The hidden entry rides along untouched — it needs no analogous trim
    // (the accepted-index slice selects by position) and the helper must
    // not disturb it.
    let hidden = try #require(state?[mtpLastHiddenStatesKey])
    let origHidden = try #require(makeSharedKVState(span: 6)[mtpLastHiddenStatesKey])
    #expect(hidden.shape == origHidden.shape)
    #expect(allClose(hidden, origHidden, rtol: 0, atol: 0).item(Bool.self))
}

@Test
func testTrimSharedKVStateNoOpOnNilStateAndAbsentKey() {
    // Nil state: must not crash, must stay nil.
    var nilState: LMOutput.State? = nil
    reconcileSharedKVState(&nilState, discarding: 3, lengths: { _ in Int.max })
    #expect(nilState == nil)

    // Present state without the sharedKV key (e.g. the quantization-onset
    // round): must not crash, must not invent the key, must leave sibling
    // keys alone.
    var keylessState: LMOutput.State? = LMOutput.State()
    keylessState?[mtpEmitFlagKey] = true
    reconcileSharedKVState(&keylessState, discarding: 3, lengths: { _ in Int.max })
    #expect(keylessState?[mtpSharedKVStatesKey] == nil)
    #expect(keylessState?[mtpEmitFlagKey] == true)
}

// MARK: - Speculating past the sliding window
//
// MTP's accept/reject step used to rewind the main cache, and a `RotatingKVCache` cannot be
// rewound once its ring has wrapped, so the iterator stood down to plain decoding as soon as the
// stream's total context crossed the window (512 on Gemma 4 E2B/E4B). With a chat template and
// any real system prompt that happened at prefill, so the drafter never fired at all.
//
// Rounds now run against `MTPCacheOverlay`: staged K/V beside the ring, only the accepted prefix
// committed, nothing ever trimmed. These checks pin the behavior that replaced the stand-down --
// speculation survives the crossing, and the stream stays token-identical to a plain
// autoregressive run on both sides of it.
//
// Coverage boundary: these are bookkeeping tests. The mock's predictions are a function of
// position, not of K/V content, so they pin offsets, spans, commit counts and token identity --
// not whether the committed rows hold the right numbers. That is `MTPCacheOverlayTests`, which
// drives real position-encoded K/V through the cache and compares attention against a
// logical-order reference.

/// A target whose prediction at every position is a pure function of that absolute position, so a
/// speculative run and a greedy run must agree token for token no matter how the drafts land.
/// (The older `MockMainModel` scripts by forward-call index, which only stays in step when every
/// draft is accepted -- too weak to test an accept/reject path across a wrap.)
///
/// It writes real K/V through whatever caches it is handed and emits `sharedKV` from what those
/// caches returned, clamped per layer type exactly as the Gemma 4 emit hook does. So the cache
/// machinery under test is exercised rather than simulated.
private final class PositionScriptedMainModel: Module, LanguageModel, KVCacheDimensionProvider {
    static let vocabularySize = 32

    /// Which of `prepare`'s two shapes this model returns, and whether it emits drafter state
    /// while doing so. The iterator reconciles the emitted snapshot at a different site in each
    /// case, and the re-prime forward exists only for the middle one.
    enum PrefillStyle {
        /// `prepare` hands back the prompt; the iterator runs the final position itself.
        case tokens
        /// `prepare` evaluates the prompt and returns logits, carrying no drafter state -- so the
        /// iterator follows up with a re-prime forward to obtain some.
        case logits
        /// `prepare` evaluates the prompt and threads drafter state through, so no re-prime.
        case logitsWithDrafterState
    }

    var kvHeads: [Int] { Array(repeating: 1, count: wideSlidingWindow == nil ? 2 : 3) }
    let script: [Int32]
    let slidingWindow: Int
    let wideSlidingWindow: Int?
    let omitDrafterState: Bool
    /// Emit `mtpSharedKVStatesKey` without the source map that says which entry each tuple came
    /// from -- a snapshot the consumer cannot place against the cache.
    let omitSharedKVSources: Bool
    let prefillStyle: PrefillStyle

    private(set) var emittedSharedKVSpans: [[String: Int]] = []
    private(set) var verifyChunkLengths: [Int] = []
    private(set) var forwardCallCount = 0

    init(
        script: [Int32], slidingWindow: Int, wideSlidingWindow: Int? = nil,
        omitDrafterState: Bool = false, omitSharedKVSources: Bool = false,
        prefillStyle: PrefillStyle = .tokens
    ) {
        self.script = script
        self.slidingWindow = slidingWindow
        self.wideSlidingWindow = wideSlidingWindow
        self.omitDrafterState = omitDrafterState
        self.omitSharedKVSources = omitSharedKVSources
        self.prefillStyle = prefillStyle
        super.init()
    }

    /// Layer 0 is global, layer 1 slides -- Gemma 4's topology in miniature. `wideSlidingWindow`
    /// adds a third layer sliding at a different width, so a timeline between the two widths has
    /// one ring wrapped and the other not.
    func newCache(parameters: GenerateParameters?) -> [KVCache] {
        var cache: [KVCache] = [KVCacheSimple(), RotatingKVCache(maxSize: slidingWindow, keep: 0)]
        if let wideSlidingWindow {
            cache.append(RotatingKVCache(maxSize: wideSlidingWindow, keep: 0))
        }
        return cache
    }

    func prepare(
        _ input: LMInput, cache: [KVCache], state _: LMOutput.State?, prefill _: PrefillParameters
    ) throws -> PrepareResult {
        switch prefillStyle {
        case .tokens:
            return .tokens(input.text)
        case .logits:
            return .logits(callAsFunction(input.text, cache: cache, state: nil))
        case .logitsWithDrafterState:
            var emitting = LMOutput.State()
            emitting[mtpEmitFlagKey] = true
            return .logits(callAsFunction(input.text, cache: cache, state: emitting))
        }
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        makeLogits(from: cache?.first?.offset ?? 0, positions: inputs.dim(-1))
    }

    func callAsFunction(
        _ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?
    ) -> LMOutput {
        forwardCallCount += 1
        let positions = input.tokens.dim(-1)
        // Read the base position before the caches move; every entry is at the same offset.
        let base = cache?.first?.offset ?? 0
        let logits = makeLogits(from: base, positions: positions)

        guard let cache else { return LMOutput(logits: logits) }
        if positions > 1 { verifyChunkLengths.append(positions) }

        let (keys, values) = Self.positionedKV(base ..< (base + positions))
        let presented = cache.map { $0.update(keys: keys, values: values) }

        guard !omitDrafterState, state?[mtpEmitFlagKey] ?? false else {
            return LMOutput(logits: logits)
        }

        // Emitted verbatim: presenting a sliding layer more than its window is what the write
        // path does, and clamping it back is the consumer's job now, not the target's.
        var sharedKV = [String: (MLXArray, MLXArray)]()
        sharedKV["full_attention"] = presented[0]
        sharedKV["sliding_attention"] = presented[1]
        emittedSharedKVSpans.append(sharedKV.mapValues { $0.0.dim(-2) })

        var out = LMOutput.State()
        out[mtpLastHiddenStatesKey] = MLXArray.zeros([1, positions, 4])
        if !omitSharedKVSources {
            out[mtpSharedKVSourceIndicesKey] = ["full_attention": 0, "sliding_attention": 1]
        }
        out[mtpSharedKVStatesKey] = sharedKV
        return LMOutput(logits: logits, state: out)
    }

    private func makeLogits(from base: Int, positions: Int) -> MLXArray {
        let vocab = Self.vocabularySize
        var data = [Float](repeating: 0, count: positions * vocab)
        for i in 0 ..< positions {
            data[i * vocab + Int(prediction(at: base + i))] = 100
        }
        return MLXArray(data, [1, positions, vocab])
    }

    /// The token this model predicts will follow absolute position `p`.
    private func prediction(at p: Int) -> Int32 { script[p % script.count] }

    private static func positionedKV(_ positions: Range<Int>) -> (MLXArray, MLXArray) {
        let count = positions.count
        let keys = MLXArray(
            positions.flatMap { Array(repeating: Float($0), count: 4) }, [1, 1, count, 4])
        return (keys, keys * -1)
    }

}

/// A script that mixes the drafted token with others, so rounds land on a spread of acceptance
/// counts rather than always-accept or always-reject.
private func mixedAcceptanceScript(drafted: Int32, length: Int = 512) -> [Int32] {
    (0 ..< length).map { position in
        // Runs of the drafted value separated by other tokens: acceptance varies round to round
        // as the phase drifts against the block size.
        (position % 7) < 3 ? drafted : Int32((position % 11) + 12)
    }
}

private func runMTP(
    model: PositionScriptedMainModel, drafter: MockDrafter, prompt: [Int32], maxTokens: Int,
    blockSize: Int = 4
) throws -> (tokens: [Int], iterator: MTPSpeculativeTokenIterator) {
    var iter = try MTPSpeculativeTokenIterator(
        input: LMInput(tokens: MLXArray(prompt)),
        mainModel: model,
        drafter: drafter,
        mainCache: model.newCache(parameters: nil),
        parameters: GenerateParameters(maxTokens: maxTokens, temperature: 0),
        blockSize: blockSize
    )
    var tokens = [Int]()
    while let token = iter.next() { tokens.append(token) }
    return (tokens, iter)
}

private func runGreedy(
    model: PositionScriptedMainModel, prompt: [Int32], maxTokens: Int
) throws -> [Int] {
    var iter = try TokenIterator(
        input: LMInput(tokens: MLXArray(prompt)),
        model: model,
        cache: model.newCache(parameters: nil),
        parameters: GenerateParameters(maxTokens: maxTokens, temperature: 0)
    )
    var tokens = [Int]()
    while let token = iter.next() { tokens.append(token) }
    return tokens
}

// MARK: - A snapshot the consumer cannot place is refused at prefill too
//
// Prefill emits, and it emits at three sites depending on which shape the target's `prepare`
// returns. All three reconcile the snapshot against the cache, and all three have to act on a
// refusal: the round-level `assert` that would otherwise catch it is compiled out in release,
// leaving the drafter attending an unreconciled sliding snapshot.

/// Shared body: a target that emits a snapshot with no source map must stand the drafter down
/// during `init`, and the stream must still come out token-identical to plain greedy decoding.
private func expectSourcelessPrefillStandsDown(
    _ style: PositionScriptedMainModel.PrefillStyle,
    sourceLocation: SourceLocation = #_sourceLocation
) throws {
    let window = 8
    let prompt = Array(1 ... Int32(window * 3))
    func makeModel() -> PositionScriptedMainModel {
        PositionScriptedMainModel(
            script: mixedAcceptanceScript(drafted: 7), slidingWindow: window,
            omitSharedKVSources: true, prefillStyle: style)
    }

    let drafter = MockDrafter(draftedTokenValue: 7)
    let (tokens, iter) = try runMTP(
        model: makeModel(), drafter: drafter, prompt: prompt, maxTokens: 24)

    #expect(
        iter.passthroughReason == MTPSpeculativeTokenIterator.missingSharedKVSourcesReason,
        "a sourceless snapshot at prefill did not stand the drafter down",
        sourceLocation: sourceLocation)
    #expect(
        drafter.draftBlockCallCount == 0, "the drafter ran against an unreconciled snapshot",
        sourceLocation: sourceLocation)
    #expect(
        tokens.count == 24, "the stream did not complete in passthrough",
        sourceLocation: sourceLocation)

    let greedy = try runGreedy(model: makeModel(), prompt: prompt, maxTokens: 24)
    #expect(
        tokens == greedy, "standing down at prefill changed the emitted stream",
        sourceLocation: sourceLocation)
}

@Test func testSourcelessPrefillStandsDownWhenPrepareReturnsTokens() throws {
    try expectSourcelessPrefillStandsDown(.tokens)
}

@Test func testSourcelessPrefillStandsDownWhenPrepareReturnsLogits() throws {
    try expectSourcelessPrefillStandsDown(.logits)
}

@Test func testSourcelessPrefillStandsDownWhenPrepareEmitsDrafterState() throws {
    try expectSourcelessPrefillStandsDown(.logitsWithDrafterState)
}

/// The re-prime forward exists only to obtain drafter state. Once prefill's own state has been
/// refused there is nothing to obtain, so it is skipped rather than run and thrown away.
@Test func testSourcelessPrefillSkipsTheRePrimeForward() throws {
    let window = 8
    let prompt = Array(1 ... Int32(window * 3))

    let refused = PositionScriptedMainModel(
        script: mixedAcceptanceScript(drafted: 7), slidingWindow: window,
        omitSharedKVSources: true, prefillStyle: .logitsWithDrafterState)
    var refusedIterator = try MTPSpeculativeTokenIterator(
        input: LMInput(tokens: MLXArray(prompt)),
        mainModel: refused, drafter: MockDrafter(draftedTokenValue: 7),
        mainCache: refused.newCache(parameters: nil),
        parameters: GenerateParameters(maxTokens: 4, temperature: 0), blockSize: 4)
    #expect(
        refused.forwardCallCount == 1,
        "prefill ran \(refused.forwardCallCount) forwards; the re-prime should have been skipped")
    #expect(
        refusedIterator.passthroughReason
            == MTPSpeculativeTokenIterator.missingSharedKVSourcesReason)
    #expect(refusedIterator.next() != nil)
}

/// The regression that started this: a prompt longer than the sliding window used to stand the
/// drafter down at init, so it never fired for the whole stream.
@Test
func testMTPIteratorSpeculatesWhenThePromptExceedsTheSlidingWindow() throws {
    let window = 8
    let drafter = MockDrafter(draftedTokenValue: 7)
    let model = PositionScriptedMainModel(
        script: mixedAcceptanceScript(drafted: 7), slidingWindow: window)

    let (tokens, iter) = try runMTP(
        model: model, drafter: drafter, prompt: Array(1 ... Int32(window * 3)), maxTokens: 16)

    #expect(tokens.count == 16)
    #expect(iter.passthroughReason == nil, "the drafter stood down on a long prompt")
    #expect(drafter.draftBlockCallCount > 0, "the drafter never fired")
    #expect(iter.proposedCount > 0)
}

/// ...and a stream that crosses the window mid-generation used to stand down at the crossing.
@Test
func testMTPIteratorKeepsSpeculatingAcrossTheWindowCrossing() throws {
    let window = 8
    let drafter = MockDrafter(draftedTokenValue: 7)
    let model = PositionScriptedMainModel(
        script: mixedAcceptanceScript(drafted: 7), slidingWindow: window)

    // Prompt sits under the window; generation runs several multiples past it.
    let (tokens, iter) = try runMTP(
        model: model, drafter: drafter, prompt: [1, 2, 3], maxTokens: window * 5)

    #expect(tokens.count == window * 5)
    #expect(iter.passthroughReason == nil, "the drafter stood down at the crossing")

    // Rounds kept running after the crossing rather than stopping at it.
    let roundsBeforeCrossing = (window - 3) / 4 + 1
    #expect(
        drafter.draftBlockCallCount > roundsBeforeCrossing,
        "speculation stopped at the window: \(drafter.draftBlockCallCount) rounds")
    #expect(iter.acceptedCount > 0, "no draft was ever accepted")
}

/// Invariant: speculation is an optimization, so the emitted stream has to be the one a plain
/// autoregressive run produces -- before, across, and long after the window crossing.
@Test(arguments: [
    // (prompt length, generated tokens) with window 8: below, at, above, and far past.
    (3, 40), (7, 40), (8, 40), (9, 40), (20, 64),
])
func testMTPIteratorMatchesGreedyAcrossTheWindow(promptLength: Int, maxTokens: Int) throws {
    let window = 8
    let script = mixedAcceptanceScript(drafted: 7)
    let prompt = Array(1 ... Int32(promptLength))

    let model = PositionScriptedMainModel(script: script, slidingWindow: window)
    let drafter = MockDrafter(draftedTokenValue: 7)
    let (mtpTokens, iter) = try runMTP(
        model: model, drafter: drafter, prompt: prompt, maxTokens: maxTokens)

    let greedyTokens = try runGreedy(
        model: PositionScriptedMainModel(script: script, slidingWindow: window),
        prompt: prompt, maxTokens: maxTokens)

    #expect(iter.passthroughReason == nil, "the run fell back instead of speculating")
    #expect(iter.acceptedCount > 0, "the run never accepted a draft, so it proves little")
    #expect(
        iter.proposedCount > iter.acceptedCount,
        "the run never rejected a draft, so the commit path is untested here")
    #expect(
        mtpTokens == greedyTokens,
        "prompt \(promptLength): MTP diverged from greedy\n  \(mtpTokens)\n  \(greedyTokens)")
}

/// The drafter attends over `sharedKV` bidirectionally and preconditions on the sliding entry
/// fitting its window (`Gemma4Assistant.forwardHidden`). A round presents the target up to
/// `window + blockSize - 1` sliding entries, so the hand-off has to be clamped -- this is the
/// check that the clamp is actually reaching the drafter.
@Test
func testDrafterSeesExactlyTheWindowItCanAttendOver() throws {
    let window = 8
    let drafter = MockDrafter(draftedTokenValue: 7)
    let model = PositionScriptedMainModel(
        script: mixedAcceptanceScript(drafted: 7), slidingWindow: window)

    let (_, iter) = try runMTP(
        model: model, drafter: drafter, prompt: [1, 2, 3], maxTokens: window * 6)
    #expect(iter.passthroughReason == nil)
    #expect(drafter.receivedSharedKVSpans.count > 1)

    // Exact, not merely bounded: the commit that decided what the cache kept also clamped the
    // snapshot, so the drafter sees the whole window rather than the window minus that round's
    // rejects. `queryOffset` is the timeline the iterator handed the drafter.
    for (round, spans) in drafter.receivedSharedKVSpans.enumerated() {
        let sliding = try #require(spans["sliding_attention"])
        let timeline = drafter.receivedQueryOffsets[round]
        #expect(
            sliding == Swift.min(timeline, window),
            "round \(round): drafter got \(sliding) sliding entries at timeline \(timeline)")
        #expect(try #require(spans["full_attention"]) == timeline, "round \(round): global span")
    }
}

/// A cache the overlay cannot stage or rewind is refused at init rather than silently producing
/// a wrong stream.
@Test
func testMTPIteratorRejectsACacheItCannotStage() {
    let model = PositionScriptedMainModel(
        script: mixedAcceptanceScript(drafted: 7), slidingWindow: 8)
    #expect(throws: KVCacheError.self) {
        _ = try MTPSpeculativeTokenIterator(
            input: LMInput(tokens: MLXArray([Int32(1), 2, 3])),
            mainModel: model,
            drafter: MockDrafter(),
            mainCache: [MambaCache()],
            parameters: GenerateParameters(maxTokens: 4, temperature: 0),
            blockSize: 4)
    }
}

/// If an entry stops being stageable mid-stream the round falls back stickily, and the stream
/// still completes.
@Test
func testMTPIteratorFallsBackWhenACacheStopsBeingStageable() throws {
    let cache = CountingKVCache(wrapAt: 8)
    let main = MockMainModel(nextLogitTokens: [0, 0, 7, 7, 7, 7, 9, 11, 12, 13, 14])
    let drafter = MockDrafter(draftedTokenValue: 7)

    var iter = try MTPSpeculativeTokenIterator(
        input: LMInput(tokens: MLXArray([Int32(1), 2, 3])),
        mainModel: main, drafter: drafter, mainCache: [cache],
        parameters: GenerateParameters(maxTokens: 8, temperature: 0), blockSize: 4)

    var tokens = [Int]()
    while let token = iter.next() { tokens.append(token) }

    #expect(tokens.count == 8, "the stream did not complete")
    #expect(
        iter.passthroughReason
            == "main KV cache cannot stage a speculative round; continuing without speculation")
}

/// The block size is clamped to what the narrowest sliding layer can present rather than
/// trapping, because it is a tuning knob rather than a correctness input.
@Test
func testBlockSizeIsClampedToTheNarrowestSlidingWindow() throws {
    let cache: [KVCache] = [KVCacheSimple(), RotatingKVCache(maxSize: 8, keep: 0)]
    #expect(MTPSpeculativeTokenIterator.maximumBlockSize(for: cache) == 9)
    #expect(
        MTPSpeculativeTokenIterator.maximumBlockSize(for: [KVCacheSimple()]) == .max,
        "nothing slides, so nothing bounds the round")
    #expect(
        MTPSpeculativeTokenIterator.maximumBlockSize(for: [RotatingKVCache(maxSize: 1)]) == 2,
        "the clamp must never fall below the blockSize >= 2 precondition")

    let model = PositionScriptedMainModel(
        script: mixedAcceptanceScript(drafted: 7), slidingWindow: 8)
    let (tokens, iter) = try runMTP(
        model: model, drafter: MockDrafter(draftedTokenValue: 7), prompt: [1, 2, 3],
        maxTokens: 24, blockSize: 64)
    #expect(iter.blockSize == 9, "blockSize was not clamped to the window")
    #expect(tokens.count == 24)
}

/// A round wider than the window still decodes correctly; the clamp exists for the stats, not
/// for correctness, so bypassing it must not change the stream.
@Test
func testStreamMatchesGreedyWhenTheRoundIsWiderThanTheWindow() throws {
    let window = 4
    let script = mixedAcceptanceScript(drafted: 7)
    let prompt: [Int32] = [1, 2, 3]

    let model = PositionScriptedMainModel(script: script, slidingWindow: window)
    let (mtpTokens, iter) = try runMTP(
        model: model, drafter: MockDrafter(draftedTokenValue: 7), prompt: prompt,
        maxTokens: 32, blockSize: 5)
    #expect(iter.blockSize == 5, "this case is only interesting at blockSize == window + 1")

    let greedyTokens = try runGreedy(
        model: PositionScriptedMainModel(script: script, slidingWindow: window),
        prompt: prompt, maxTokens: 32)
    #expect(mtpTokens == greedyTokens)
}

/// The head clamp is the other half of reconciliation: a sliding layer is presented more than its
/// window on purpose, and what the drafter may attend over is bounded by what the layer can still
/// describe.
@Test
func testReconcileSharedKVStateClampsTheHeadToWhatTheLeafDescribes() throws {
    var state: LMOutput.State? = makeSharedKVState(span: 10)
    // Leaf 0 is global and unbounded; leaf 1 slides with a 4-entry window.
    reconcileSharedKVState(&state, discarding: 0, lengths: { $0 == 0 ? Int.max : 4 })

    let sharedKV = try #require(state?[mtpSharedKVStatesKey])
    #expect(try #require(sharedKV["full_attention"]).0.dim(-2) == 10, "global entry was clamped")
    #expect(try #require(sharedKV["sliding_attention"]).0.dim(-2) == 4)
}

/// Tail first, then head. Clamping first would leave a sliding entry short by the rejected count
/// every round, which is a silent acceptance-rate loss rather than a failure.
@Test
func testReconcileSharedKVStateDropsTheTailBeforeClampingTheHead() throws {
    var state: LMOutput.State? = makeSharedKVState(span: 10)
    reconcileSharedKVState(&state, discarding: 3, lengths: { $0 == 0 ? Int.max : 4 })

    let sharedKV = try #require(state?[mtpSharedKVStatesKey])
    #expect(try #require(sharedKV["full_attention"]).0.dim(-2) == 7)
    // 10 - 3 rejected = 7 true rows, of which the window keeps the last 4 — not 4 - 3 = 1.
    #expect(try #require(sharedKV["sliding_attention"]).0.dim(-2) == 4)
}

/// Without a source map the snapshot cannot be placed against cache entries, so reconciliation
/// refuses rather than guessing a bound. The iterator reads that as a reason to stop speculating.
@Test
func testReconcileSharedKVStateRefusesWithoutSourceIndices() throws {
    var state: LMOutput.State? = makeSourcelessSharedKVState(span: 6)
    #expect(reconcileSharedKVState(&state, discarding: 2, lengths: { _ in Int.max }) == false)

    let sharedKV = try #require(state?[mtpSharedKVStatesKey])
    #expect(
        try #require(sharedKV["full_attention"]).0.dim(-2) == 6,
        "a refused reconciliation must leave the snapshot untouched")
}

/// An absent snapshot is not a refusal — the quantization-onset round emits none.
@Test
func testReconcileSharedKVStateAcceptsAnAbsentSnapshot() throws {
    var state: LMOutput.State? = LMOutput.State()
    state?[mtpEmitFlagKey] = true
    #expect(reconcileSharedKVState(&state, discarding: 3, lengths: { _ in Int.max }))
    #expect(state?[mtpSharedKVStatesKey] == nil)
    #expect(state?[mtpEmitFlagKey] == true)
}

// MARK: - The timeline tracks what was emitted
//
// `KVCacheStorage.processedTokenCount` is the authoritative position, and `ChatSession`
// reconciles its token ledger against it rather than against per-entry offsets. So it has to
// describe exactly the tokens the consumer saw — including when the consumer stops partway
// through a round's accepted prefix, which is where the old rewind silently gave up past a
// sliding window.
//
// The `- 1` is not slack: the last verifier sample is never cached in its own round. It is the
// input to the next one.

@Test
func testEveryRoundLeavesTheLeavesInStepWithTheTimeline() throws {
    let window = 8
    let prompt: [Int32] = [1, 2, 3]
    let model = PositionScriptedMainModel(
        script: mixedAcceptanceScript(drafted: 7), slidingWindow: window)

    var iter = try MTPSpeculativeTokenIterator(
        input: LMInput(tokens: MLXArray(prompt)),
        mainModel: model, drafter: MockDrafter(draftedTokenValue: 7),
        mainCache: model.newCache(parameters: nil),
        parameters: GenerateParameters(maxTokens: window * 6, temperature: 0), blockSize: 4)

    var emitted = 0
    while iter.next() != nil {
        emitted += 1
        #expect(
            iter.mainCacheStorage.nativeAttentionOffsetsAreAligned,
            "leaves drifted from the timeline after token \(emitted)")
    }
    iter.finalizeGeneration()

    #expect(iter.passthroughReason == nil)
    #expect(emitted == window * 6)
    #expect(iter.mainCacheStorage.processedTokenCount == prompt.count + emitted - 1)
    #expect(iter.mainCacheStorage.nativeAttentionOffsetsAreAligned)
}

/// A consumer that stops partway through a round — a stop token, a stop string, cancellation —
/// leaves committed positions undrained. They have to come back out, past the wrap too.
@Test(arguments: [1, 2, 3, 5, 8, 13])
func testStoppingEarlyLeavesTheTimelineAtWhatWasEmitted(stopAfter: Int) throws {
    let window = 8
    let prompt: [Int32] = [1, 2, 3]
    let model = PositionScriptedMainModel(
        script: mixedAcceptanceScript(drafted: 7), slidingWindow: window)

    var iter = try MTPSpeculativeTokenIterator(
        input: LMInput(tokens: MLXArray(prompt)),
        mainModel: model, drafter: MockDrafter(draftedTokenValue: 7),
        mainCache: model.newCache(parameters: nil),
        parameters: GenerateParameters(maxTokens: window * 6, temperature: 0), blockSize: 4)

    var emitted = 0
    while emitted < stopAfter, iter.next() != nil {
        emitted += 1
    }
    iter.finalizeGeneration()

    #expect(emitted == stopAfter)
    // The `- 1` applies only when the last token emitted was a verifier sample, whose K/V is
    // deliberately not cached in its own round. Stopping mid-buffer ends on an accepted draft
    // instead, which *is* cached — so the timeline lands on one of two adjacent values. The half
    // that matters is the upper bound: describing more than the consumer saw is the failure the
    // old rewind produced silently past the window.
    let timeline = iter.mainCacheStorage.processedTokenCount
    #expect(
        timeline <= prompt.count + emitted,
        "timeline \(timeline) describes tokens beyond the \(emitted) the consumer saw")
    #expect(
        timeline >= prompt.count + emitted - 1,
        "timeline \(timeline) dropped tokens the consumer did see")
    #expect(iter.mainCacheStorage.nativeAttentionOffsetsAreAligned)
}

/// The same stop, on a topology whose sliding layers have different widths. At a timeline between
/// them the narrow ring has wrapped and the wide one has not, so one commit produces a restore
/// point for the first and a plain trim record for the second — and the trim has to reach the live
/// ring rather than the round's staged wrapper, whose `trim(_:)` is a deliberate no-op.
@Test(arguments: [8, 11, 13])
func testStoppingEarlyOnMixedWidthSlidingLayers(stopAfter: Int) throws {
    let prompt: [Int32] = [1, 2, 3]
    let model = PositionScriptedMainModel(
        script: mixedAcceptanceScript(drafted: 7), slidingWindow: 8, wideSlidingWindow: 64)

    var iter = try MTPSpeculativeTokenIterator(
        input: LMInput(tokens: MLXArray(prompt)),
        mainModel: model, drafter: MockDrafter(draftedTokenValue: 7),
        mainCache: model.newCache(parameters: nil),
        parameters: GenerateParameters(maxTokens: 48, temperature: 0), blockSize: 4)

    var emitted = 0
    while emitted < stopAfter, iter.next() != nil { emitted += 1 }
    iter.finalizeGeneration()

    #expect(emitted == stopAfter)

    // The premise, asserted rather than assumed: these stop points are past the narrow window and
    // well inside the wide one, so the two layers really did take different commit records.
    let leaves = iter.mainCacheStorage.cache
    #expect(leaves.count == 3)
    #expect(
        (leaves[1] as? RotatingKVCache)?.isTrimmable == false,
        "the narrow ring should have wrapped by \(stopAfter) tokens")
    #expect(
        (leaves[2] as? RotatingKVCache)?.isTrimmable == true,
        "the wide ring should not have wrapped by \(stopAfter) tokens")

    let timeline = iter.mainCacheStorage.processedTokenCount
    #expect(
        timeline <= prompt.count + emitted,
        "timeline \(timeline) describes tokens beyond the \(emitted) the consumer saw")
    #expect(timeline >= prompt.count + emitted - 1)
    #expect(
        iter.mainCacheStorage.nativeAttentionOffsetsAreAligned,
        "the wide sliding layer stayed ahead of the timeline")
}

/// `discardGeneratedToken` is telemetry only: a stop token that is not delivered was still
/// returned by `next()` and is genuinely in the cache, so it must not move the retain count.
@Test
func testDiscardingAGeneratedTokenDoesNotMoveTheTimeline() throws {
    let window = 8
    let prompt: [Int32] = [1, 2, 3]
    let model = PositionScriptedMainModel(
        script: mixedAcceptanceScript(drafted: 7), slidingWindow: window)

    var iter = try MTPSpeculativeTokenIterator(
        input: LMInput(tokens: MLXArray(prompt)),
        mainModel: model, drafter: MockDrafter(draftedTokenValue: 7),
        mainCache: model.newCache(parameters: nil),
        parameters: GenerateParameters(maxTokens: window * 4, temperature: 0), blockSize: 4)

    var emitted = 0
    while emitted < 11, iter.next() != nil { emitted += 1 }
    iter.discardGeneratedToken()
    iter.finalizeGeneration()

    // Same two-sided bound as above, and the point is that discarding did not shift it: the
    // discarded token was returned by `next()` and is genuinely part of the model's context.
    let timeline = iter.mainCacheStorage.processedTokenCount
    #expect(timeline <= prompt.count + emitted)
    #expect(timeline >= prompt.count + emitted - 1)
    #expect(iter.mainCacheStorage.nativeAttentionOffsetsAreAligned)
}
