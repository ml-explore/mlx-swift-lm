// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLMCommon

// MARK: - Synthetic mocks for iterator plumbing

/// Records `bind(target:)` and `draftBlock(...)` invocations and returns a
/// fixed token pattern so the iterator's draft/verify/accept flow can be
/// exercised without a real drafter.
private final class MockDrafter: Module, MTPDrafterModel {
    private(set) var bindCallCount = 0
    private(set) var draftBlockCallCount = 0
    var draftedTokenValue: Int32

    init(draftedTokenValue: Int32 = 7) {
        self.draftedTokenValue = draftedTokenValue
        super.init()
    }

    func bind(target: any LanguageModel) {
        bindCallCount += 1
    }

    func draftBlock(
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
    // [7, 7, 7, 9] (3 matching drafts, 1 correction). Drafter returns [7, 7, 7]
    // so all three drafts match and we get a single correction bonus of 9.
    // Expected pendingTokens after one round: [7, 7, 7, 9].

    let mainLogitTokens: [Int32] = [
        // Prefill follow-up call (length 3): final position picks 7
        99, 99, 7,
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

    // `bind` called exactly once in init.
    #expect(drafter.bindCallCount == 1)

    // Draw the first token from `next()`; this triggers `speculateRound`.
    let t0 = iter.next()
    #expect(t0 == 7)

    // Drain the rest of the round's pending buffer.
    let t1 = iter.next()
    let t2 = iter.next()
    let t3 = iter.next()
    #expect(t1 == 7)
    #expect(t2 == 7)
    #expect(t3 == 9)

    // One round produced 4 tokens (3 accepted + 1 bonus correction).
    #expect(iter.tokenCount == 4)
    #expect(drafter.draftBlockCallCount == 1)
    // proposedCount = numDraft = 3; accepted = 3.
    #expect(iter.proposedCount == 3)
    #expect(iter.acceptedCount == 3)
    // Verify the main model received emit=true on every call after prefill.
    #expect(main.lastIncomingEmitFlag == true)
}

// MARK: - Bind-once invariant

@Test
func testMTPIteratorInitCallsBindOnce() throws {
    let main = MockMainModel(nextLogitTokens: [0, 0, 0])
    let drafter = MockDrafter()
    let input = LMInput(tokens: MLXArray([Int32(1), 2, 3]))

    let iter = try MTPSpeculativeTokenIterator(
        input: input, mainModel: main, drafter: drafter, mainCache: nil,
        parameters: GenerateParameters(maxTokens: 0), blockSize: 4
    )
    #expect(drafter.bindCallCount == 1)
    #expect(iter.maxTokens == 0)
}

// MARK: - Passthrough fallback when state is absent

@Test
func testMTPIteratorMissingStateFallsBackToPassthrough() throws {
    // Main model with `omitDrafterState=true` never populates the MTP keys,
    // so the iterator switches to passthrough on the first `speculateRound`
    // call. Drafter must not be invoked.
    let mainLogitTokens: [Int32] = [
        // Prefill follow-up: length 3, final position picks 5
        99, 99, 5,
        // Passthrough single-token rounds: each is a length-1 call picking
        // 11, then 12, then 13.
        11, 12, 13,
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
    // 3 tokens generated, then nil.
    #expect(tokens[0] == 11)
    #expect(tokens[1] == 12)
    #expect(tokens[2] == 13)
    #expect(tokens[3] == nil)
    // Drafter was never invoked for an actual round.
    #expect(drafter.draftBlockCallCount == 0)
}

// MARK: - Pending buffer drain order

@Test
func testMTPIteratorPendingBufferDrainOrder() throws {
    // Drafter returns [5, 5, 5]; main verifies [5, 5, 7, 9].
    // Expected accept-prefix is positions 0..1 -> [5, 5], then correction at
    // position 2 = 7. The pendingTokens order should be [5, 5, 7] — main-model
    // sequence order — not the drafter's [5, 5, 5].
    let mainLogitTokens: [Int32] = [
        99, 99, 5,  // prefill follow-up picks bonus 5
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
    #expect(t0 == 5)
    #expect(t1 == 5)
    #expect(t2 == 7)
    // proposedCount = 3 (numDraft); accepted = 2.
    #expect(iter.proposedCount == 3)
    #expect(iter.acceptedCount == 2)
}
