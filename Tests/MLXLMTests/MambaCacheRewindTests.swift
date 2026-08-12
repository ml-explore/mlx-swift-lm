import Foundation
import MLX
import Testing

@testable import MLXLMCommon

// Rewindability for `MambaCache`.
//
// A recurrent state cannot be trimmed by dropping rows, so `MambaCache` reports
// `isTrimmable == false` and that disables speculative decoding for every hybrid
// SSM/attention model: `canTrimPromptCache` is an `allSatisfy`, and
// `MTPSpeculativeTokenIterator.init` throws without it.
//
// These tests pin the opt-in rewind path, and — just as importantly — pin that
// the default path is untouched.

/// Distinct, cheap slot payloads so a restored snapshot is identifiable by value.
private func slots(_ step: Int) -> (conv: MLXArray, ssm: MLXArray) {
    (conv: MLXArray([Float(step), Float(step) + 0.5]), ssm: MLXArray([Float(step) * 10]))
}

/// Advance a cache one decode step, writing both slots.
private func step(_ cache: MambaCache, _ n: Int) {
    let s = slots(n)
    cache[0] = s.conv
    cache[1] = s.ssm
    cache.checkpoint(advancing: 1)
}

private func expectSlots(_ cache: MambaCache, matches n: Int, _ label: String = "") {
    let want = slots(n)
    #expect(cache[0]?.asArray(Float.self) == want.conv.asArray(Float.self), "conv slot \(label)")
    #expect(cache[1]?.asArray(Float.self) == want.ssm.asArray(Float.self), "ssm slot \(label)")
}

// MARK: - the default path must not change

@Test func testMambaCacheIsNotTrimmableByDefault() {
    let cache = MambaCache()
    for n in 1 ... 4 { step(cache, n) }
    #expect(cache.rewindDepth == 0, "rewinding must be opt-in")
    #expect(!cache.isTrimmable, "the historical default is untrimmable and must stay that way")
    #expect(cache.trim(1) == 0, "a disabled rewind must refuse, not guess")
    expectSlots(cache, matches: 4, "after a refused trim")
}

@Test func testHybridCacheListIsUntrimmableUntilEveryMambaOptsIn() {
    // The actual gate: one `false` in the layer list disables speculation for
    // the whole model, which is why this defaults matter.
    let attention = KVCacheSimple()
    let mamba = MambaCache()
    #expect(canTrimPromptCache([attention]), "attention caches are trimmable")
    #expect(!canTrimPromptCache([attention, mamba]), "one untrimmable entry poisons the list")

    mamba.rewindDepth = 1
    step(mamba, 1)
    #expect(canTrimPromptCache([attention, mamba]), "opting in unlocks the list")
}

// MARK: - the property that makes speculation sound

@Test func testRewindRestoresStateExactly() {
    let cache = MambaCache()
    cache.rewindDepth = 2
    for n in 1 ... 5 { step(cache, n) }

    #expect(cache.isTrimmable)
    #expect(cache.trim(2) == 2, "an in-history rewind returns the amount asked for")
    expectSlots(cache, matches: 3, "state must be exactly as it was two tokens ago")

    // And the cache is usable afterwards: the next step continues from there.
    step(cache, 42)
    expectSlots(cache, matches: 42, "cache must accept writes after a rewind")
}

@Test func testRewindIsRepeatable() {
    let cache = MambaCache()
    cache.rewindDepth = 3
    for n in 1 ... 4 { step(cache, n) }

    #expect(cache.trim(1) == 1)
    expectSlots(cache, matches: 3, "first rewind")
    #expect(cache.trim(1) == 1, "history must survive an earlier rewind")
    expectSlots(cache, matches: 2, "second rewind")
}

// MARK: - refusal, not approximation

@Test func testTrimBeyondHistoryRefusesAndLeavesStateUntouched() {
    let cache = MambaCache()
    cache.rewindDepth = 1
    for n in 1 ... 3 { step(cache, n) }

    // depth 1 retains 2 snapshots, so a 2-token rewind has no exact target.
    #expect(cache.trim(2) == 0, "an out-of-history rewind must refuse")
    expectSlots(cache, matches: 3, "a refused trim must not mutate state")
    #expect(cache.trim(1) == 1, "and must not damage the history it does have")
    expectSlots(cache, matches: 2, "in-history rewind still works")
}

@Test func testTrimBeforeTheStartRefuses() {
    let cache = MambaCache()
    cache.rewindDepth = 4
    step(cache, 1)
    #expect(cache.trim(5) == 0, "cannot rewind past position zero")
    expectSlots(cache, matches: 1, "state untouched")
}

@Test func testNonPositiveTrimIsANoOp() {
    let cache = MambaCache()
    cache.rewindDepth = 2
    for n in 1 ... 2 { step(cache, n) }
    #expect(cache.trim(0) == 0)
    #expect(cache.trim(-3) == 0)
    expectSlots(cache, matches: 2, "state untouched")
}

@Test func testRewindCannotLandInsideAPrefill() {
    // A prefill of S tokens is one snapshot at position S. Rewinding to a
    // position interior to it has no exact target, so it must refuse rather
    // than silently restore the wrong state.
    let cache = MambaCache()
    cache.rewindDepth = 4
    let prefill = slots(100)
    cache[0] = prefill.conv
    cache[1] = prefill.ssm
    cache.checkpoint(advancing: 8)
    step(cache, 9)

    #expect(cache.trim(1) == 1, "rewinding the decode step is exact")
    expectSlots(cache, matches: 100, "back to the end of the prefill")
    #expect(cache.trim(3) == 0, "a position inside the prefill is not a valid target")
    expectSlots(cache, matches: 100, "state untouched")
    // The prefill snapshot is the earliest one there is: nothing recorded the
    // state before it, so position 0 is not a rewind target either. The floor
    // is the oldest retained snapshot, not zero.
    #expect(cache.trim(8) == 0, "cannot rewind past the earliest snapshot")
    expectSlots(cache, matches: 100, "state untouched")
}

// MARK: - bounds and bookkeeping

@Test func testHistoryIsBoundedByRewindDepth() {
    let cache = MambaCache()
    cache.rewindDepth = 2
    for n in 1 ... 50 { step(cache, n) }
    // Bounded memory is the whole reason `rewindDepth` exists, so assert the
    // bound behaviourally: depth is available, depth+1 is not.
    #expect(cache.trim(2) == 2, "the full declared depth must be available")
    expectSlots(cache, matches: 48)
    for n in 51 ... 100 { step(cache, n) }
    #expect(cache.trim(3) == 0, "history must not grow past the declared depth")
}

@Test func testRewindPreservesEmptySlots() {
    // `state` drops empty slots, so a snapshot taken through `state` would
    // resize a two-slot cache to one. Models with `convKernelSize == 1` never
    // populate slot 0, so this is a real configuration, not a synthetic one.
    let cache = MambaCache()
    cache.rewindDepth = 1
    cache[1] = MLXArray([Float(1)])
    cache.checkpoint(advancing: 1)
    cache[1] = MLXArray([Float(2)])
    cache.checkpoint(advancing: 1)

    #expect(cache.state.count == 1, "only one slot is populated")
    #expect(cache.trim(1) == 1)
    #expect(cache[0] == nil, "the empty slot must stay empty, not vanish")
    #expect(cache[1]?.asArray(Float.self) == [1], "the populated slot rewinds")
    // The slot count itself must survive, or the next `cache[1]` write traps.
    cache[1] = MLXArray([Float(3)])
    #expect(cache[1]?.asArray(Float.self) == [3])
}

@Test func testCopyPreservesRewindability() throws {
    let cache = MambaCache()
    cache.rewindDepth = 2
    for n in 1 ... 3 { step(cache, n) }

    let copied = try #require(cache.copy() as? MambaCache)
    #expect(copied.rewindDepth == 2, "a copy must stay rewindable")
    #expect(copied.isTrimmable)
    #expect(copied.trim(2) == 2, "the copy carries its own history")
    expectSlots(copied, matches: 1, "copy rewinds to the right state")
    // The original is independent.
    expectSlots(cache, matches: 3, "rewinding a copy must not disturb the original")
}

@Test func testTrimPromptCacheDrivesTheRewind() {
    // End to end through the public helper speculative decoding actually calls.
    let mamba = MambaCache()
    mamba.rewindDepth = 2
    for n in 1 ... 4 { step(mamba, n) }
    let caches: [KVCache] = [mamba]
    #expect(canTrimPromptCache(caches))
    #expect(trimPromptCache(caches, numTokens: 2) == 2)
    expectSlots(mamba, matches: 2, "restored through trimPromptCache")
}
