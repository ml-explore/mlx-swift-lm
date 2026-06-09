// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN
import Testing

@_spi(Testing) @testable import MLXLMCommon

// MARK: - Synthetic mocks for iterator plumbing

/// Records `draftBlock(...)` invocations and returns a fixed token pattern
/// so the iterator's draft/verify/accept flow can be exercised without a
/// real drafter.
private final class MockDrafter: Module, MTPDrafterModel {
    private(set) var draftBlockCallCount = 0
    var draftedTokenValue: Int32

    init(draftedTokenValue: Int32 = 7) {
        self.draftedTokenValue = draftedTokenValue
        super.init()
    }

    func draftBlock(
        target: any LanguageModel,
        lastToken: MLXArray,
        lastHidden: MLXArray,
        sharedKV: [String: (MLXArray, MLXArray)],
        positionIds: MLXArray,
        blockSize: Int,
        sampler: any LogitSampler
    ) -> MLXArray {
        draftBlockCallCount += 1
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

    private(set) var callCount: Int = 0
    private(set) var lastIncomingEmitFlag: Bool? = nil

    init(nextLogitTokens: [Int32]) {
        self.nextLogitTokens = nextLogitTokens
        super.init()
    }

    func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
        // Return `.tokens(...)`; the iterator's `prepare` will follow up with
        // a one-position forward call that primes drafter state.
        .tokens(input.text)
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

        let positions = input.tokens.dim(-1)
        let logits = makeLogits(positions: positions)

        // Update the mock cache to reflect that `positions` tokens were seen.
        if let cache, let first = cache.first as? CountingKVCache {
            first.offset += positions
        }

        if !omitDrafterState, state?[mtpEmitFlagKey] ?? false {
            var out = LMOutput.State()
            out[mtpLastHiddenStatesKey] = MLXArray.zeros([1, positions, 4])
            out[mtpSharedKVStatesKey] = [
                "full_attention": (
                    MLXArray.zeros([1, 1, positions, 4]),
                    MLXArray.zeros([1, 1, positions, 4])
                ),
                "sliding_attention": (
                    MLXArray.zeros([1, 1, positions, 4]),
                    MLXArray.zeros([1, 1, positions, 4])
                ),
            ]
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
private final class CountingKVCache: KVCache {
    var offset: Int = 0
    var maxSize: Int? { nil }
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
    var isTrimmable: Bool { true }
    @discardableResult
    func trim(_ n: Int) -> Int {
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
        let c = CountingKVCache()
        c.offset = offset
        return c
    }
    func innerState() -> [MLXArray] { [] }
}

// MARK: - Smallest-unit-of-work smoke test

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
    // Verify the main model received emit=true on every call after prefill.
    #expect(main.lastIncomingEmitFlag == true)
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

// MARK: - LogitProcessor emit-only invariant

/// Records `didSample(token:)` calls so a test can verify which tokens the
/// processor actually observed. Pure value semantics — Swift struct value
/// copies (e.g., `var verifyProcessorCopy = processor` in `speculateRound`)
/// produce a separate `recordedTokens` backing via array copy-on-write.
private struct EmissionLog: LogitProcessor {
    var recordedTokens: [Int] = []

    mutating func prompt(_ prompt: MLXArray) {}

    func process(logits: MLXArray) -> MLXArray { logits }

    mutating func didSample(token: MLXArray) {
        recordedTokens.append(token.item(Int.self))
    }
}

/// Locks in the value-semantics invariant of `speculateRound`'s verify
/// loop: `var verifyProcessorCopy = processor` makes a Swift struct copy,
/// so `verifyProcessorCopy.didSample(...)` calls mutate the local copy
/// and do NOT propagate back to `self.processor`. The canonical processor
/// state at `self.processor` is updated only by the accept loop, which
/// runs over the actually-emitted tokens (accepted drafts + correction).
///
/// Test scenario: bs=4 (numDraft=3), drafter proposes [5, 5, 5], main
/// verifies and samples [5, 9, 1, 2] — only position 0 matches the draft.
/// accepted=1, correction=9, emitted=[bonus=5, draft=5, correction=9].
/// Verify loop's `didSample` fires four times (on the copy) for [5, 9, 1, 2].
/// Self.processor's `didSample` should fire exactly twice (for emitted [5, 9])
/// — NOT four times. The probe processor is installed AFTER init so the
/// prepare-time bonus is not recorded; the test asserts on speculation-
/// round emissions only.
@Test
func testMTPVerifyLoopDidSampleStaysScopedToLocalCopy() throws {
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
    iter._setProcessorForTesting(EmissionLog())

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

    let log = iter._processorForTesting as? EmissionLog
    #expect(log != nil, "probe processor lost between install and drain")

    // self.processor's didSample fired exactly twice — for the accepted
    // draft and the correction — NOT for the three other verify-position
    // samples (9, 1, 2) which happened on the local copy. If a regression
    // ever removes the local-copy idiom, log.recordedTokens would gain
    // entries [9, 1, 2] from the rejected verify positions.
    #expect(
        log?.recordedTokens == [5, 9],
        "self.processor.recordedTokens=\(log?.recordedTokens ?? []) — expected [5, 9] (1 accepted draft + 1 correction). Verify-loop didSample is leaking from the copy into the canonical processor."
    )
}
