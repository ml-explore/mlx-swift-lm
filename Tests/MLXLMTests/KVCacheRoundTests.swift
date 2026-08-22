// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import Testing

@testable import MLXLMCommon

// MARK: - Fixtures

/// K/V whose every element encodes its own sequence position: keys hold `+p`, values `-p`.
/// Any reordering, duplication, or K/V mix-up changes the numbers.
private func positionedKV(
    _ positions: Range<Int>, headDim: Int = 2
) -> (MLXArray, MLXArray) {
    let count = positions.count
    let keys = MLXArray(
        positions.flatMap { Array(repeating: Float($0), count: headDim) },
        [1, 1, count, headDim])
    let values = MLXArray(
        positions.flatMap { Array(repeating: Float(-$0), count: headDim) },
        [1, 1, count, headDim])
    return (keys, values)
}

/// Recover the sequence positions encoded by `positionedKV` from a `[B, H, S, D]` key array.
private func encodedPositions(_ keys: MLXArray) -> [Int] {
    keys[0, 0, 0..., 0].asArray(Float.self).map { Int($0) }
}

/// The round width these cases open with unless they are probing the width itself.
private let defaultRoundWidth = 8

/// A storage and its open round, paired so a case can commit without threading both. Rounds are
/// opened on a storage because committing has to advance the processed-token timeline in the same
/// operation; this is only sugar over that.
private final class StagedRound {
    let storage: KVCacheStorage
    let round: KVCacheRound

    var caches: [KVCache] { round.caches }

    init?(_ storage: KVCacheStorage, width: Int = defaultRoundWidth) {
        guard let round = storage.beginRound(maximumPositions: width) else { return nil }
        self.storage = storage
        self.round = round
    }

    convenience init?(_ caches: [KVCache], width: Int = defaultRoundWidth) {
        self.init(KVCacheStorage(caches, plan: .disabled), width: width)
    }

    @discardableResult
    func commit(accepted: Int) -> KVCacheRoundCommit {
        storage.commit(round, retaining: accepted)
    }

    func rollback() { storage.rollback(round) }
}

/// Write `positions` to every cache in the round, as one verify forward would.
@discardableResult
private func stageRound(
    _ staged: StagedRound, _ positions: Range<Int>
) -> [(MLXArray, MLXArray)] {
    let (keys, values) = positionedKV(positions)
    return staged.caches.map { $0.update(keys: keys, values: values) }
}

/// A deep-enough snapshot of a rotating cache's physical state to detect any mutation.
private struct RingSnapshot {
    let arrays: [MLXArray]
    let offset: Int
    let metaState: [String]

    init(_ cache: RotatingKVCache) {
        // Materialize, so the snapshot cannot alias buffers the cache may later write in place.
        arrays = cache.innerState().map { MLXArray($0.asArray(Float.self), $0.shape) }
        offset = cache.offset
        metaState = cache.metaState
    }

    func expectUnchanged(_ cache: RotatingKVCache, _ note: String) {
        let now = cache.innerState()
        #expect(now.count == arrays.count, "innerState array count changed — \(note)")
        for (index, (before, after)) in zip(arrays, now).enumerated() {
            #expect(before.shape == after.shape, "innerState[\(index)] reshaped — \(note)")
            #expect(
                allClose(before, after, rtol: 0, atol: 0).item(Bool.self),
                "innerState[\(index)] mutated — \(note)")
        }
        #expect(cache.offset == offset, "offset moved — \(note)")
        #expect(cache.metaState == metaState, "idx/offset bookkeeping moved — \(note)")
    }
}

// MARK: - Invariant 1: the live ring is physically untouched until commit

@Test func testOverlayLeavesTheLiveRingUntouchedBeforeCommit() throws {
    let live = RotatingKVCache(maxSize: 8, keep: 0)
    for p in 0 ..< 20 {
        let (k, v) = positionedKV(p ..< (p + 1))
        _ = live.update(keys: k, values: v)
    }

    let snapshot = RingSnapshot(live)
    let overlay = try #require(StagedRound([live]))

    stageRound(overlay, 20 ..< 24)
    snapshot.expectUnchanged(live, "after staging a full round")

    overlay.rollback()
    snapshot.expectUnchanged(live, "after rolling the round back")
}

@Test func testOverlayWriteThroughLayerIsRestoredByRollback() throws {
    let live = KVCacheSimple()
    for p in 0 ..< 6 {
        let (k, v) = positionedKV(p ..< (p + 1))
        _ = live.update(keys: k, values: v)
    }

    let overlay = try #require(StagedRound([live]))
    stageRound(overlay, 6 ..< 10)
    #expect(live.offset == 10, "an append-only layer is written through, not staged")

    overlay.rollback()
    #expect(live.offset == 6, "rollback did not clamp the append-only layer back")
}

@Test func testVarianceNormalizedRoundCommitsExactlyBelowTileBoundary() throws {
    let live = VarianceNormalizedKVCache(
        tileSize: 32, keyBits: 4, valueBits: 4, sinkhornIterations: 2)
    let initial = positionedKV(0 ..< 20, headDim: 32)
    _ = live.update(keys: initial.0, values: initial.1)

    let staged = try #require(StagedRound([live], width: 4))
    let provisional = positionedKV(20 ..< 24, headDim: 32)
    _ = staged.caches[0].update(keys: provisional.0, values: provisional.1)
    staged.commit(accepted: 2)

    let empty = MLXArray.zeros([1, 1, 0, 32])
    let materialized = live.update(keys: empty, values: empty)
    let expected = positionedKV(0 ..< 22, headDim: 32)
    eval(materialized.0, materialized.1)

    #expect(live.offset == 22)
    #expect(allClose(materialized.0, expected.0, rtol: 0, atol: 1e-5).item(Bool.self))
    #expect(allClose(materialized.1, expected.1, rtol: 0, atol: 1e-5).item(Bool.self))
}

@Test func testVarianceNormalizedRoundRefusesCompressionBoundary() {
    let live = VarianceNormalizedKVCache(
        tileSize: 32, keyBits: 4, valueBits: 4, sinkhornIterations: 2)
    let initial = positionedKV(0 ..< 30, headDim: 32)
    _ = live.update(keys: initial.0, values: initial.1)

    #expect(live.isTrimmable(after: 1))
    #expect(!live.isTrimmable(after: 2))
    #expect(StagedRound([live], width: 2) == nil)
    #expect(live.offset == 30, "refusing the round must leave the live cache untouched")
}

// MARK: - Invariant 2: the presentation is the live write path's, exactly

/// The overlay's whole correctness argument: what the adapter returns is what the live cache
/// would have returned had it written the same rows, so the model cannot tell the difference.
@Test func testOverlayPresentationEqualsTheLiveWritePath() throws {
    let window = 8

    let staged = RotatingKVCache(maxSize: window, keep: 0)
    let written = RotatingKVCache(maxSize: window, keep: 0)
    for p in 0 ..< 20 {
        let (k, v) = positionedKV(p ..< (p + 1))
        _ = staged.update(keys: k, values: v)
        _ = written.update(keys: k, values: v)
    }

    let overlay = try #require(StagedRound([staged]))
    let adapter = try #require(overlay.caches.first as? RotatingStagedKVCache)

    // Masks are built before the layer writes, on both paths.
    let overlayMask = adapter.makeMask(n: 4, windowSize: window, returnArray: false)
    let liveMask = written.makeMask(n: 4, windowSize: window, returnArray: false)

    let (newKeys, newValues) = positionedKV(20 ..< 24)
    let presented = adapter.update(keys: newKeys, values: newValues)
    let reference = written.update(keys: newKeys, values: newValues)

    #expect(
        encodedPositions(presented.0) == encodedPositions(reference.0),
        "overlay presented a different sequence than the live write path")
    // Shape first: `allClose` on mismatched shapes is a fatal error inside MLX, which would take
    // the whole bundle down and hide every test after this one.
    guard presented.0.shape == reference.0.shape, presented.1.shape == reference.1.shape else {
        Issue.record(
            "presentation shape \(presented.0.shape) != write-path shape \(reference.0.shape)")
        return
    }
    #expect(allClose(presented.0, reference.0, rtol: 0, atol: 0).item(Bool.self))
    #expect(allClose(presented.1, reference.1, rtol: 0, atol: 0).item(Bool.self))

    guard case .array(let overlayMaskArray) = overlayMask,
        case .array(let liveMaskArray) = liveMask
    else {
        Issue.record("expected array masks past the window, got \(overlayMask) / \(liveMask)")
        return
    }
    #expect(overlayMaskArray.shape == liveMaskArray.shape)
    #expect(
        overlayMaskArray.dim(-1) == presented.0.dim(2),
        """
        mask key axis \(overlayMaskArray.dim(-1)) does not span the presentation \
        (\(presented.0.dim(2)) keys) — attention would broadcast-fail
        """)
    if overlayMaskArray.shape == liveMaskArray.shape {
        #expect(allClose(overlayMaskArray, liveMaskArray, rtol: 0, atol: 0).item(Bool.self))
    }
}

/// Attention through the overlay, past a wrap, against a reference that never rotated.
@Test func testOverlayAttentionPastWrapMatchesLogicalOrderReference() throws {
    MLXRandom.seed(0)

    let window = 8
    let history = 20
    let queries = 4
    let heads = 2
    let headDim = 4
    let scale = 1.0 / Float(headDim).squareRoot()

    let allKeys = MLXRandom.normal([1, heads, history + queries, headDim])
    let allValues = MLXRandom.normal([1, heads, history + queries, headDim])
    let q = MLXRandom.normal([1, heads, queries, headDim])

    let live = RotatingKVCache(maxSize: window, keep: 0)
    for p in 0 ..< history {
        _ = live.update(
            keys: allKeys[0..., 0..., p ..< (p + 1), 0...],
            values: allValues[0..., 0..., p ..< (p + 1), 0...])
    }

    let overlay = try #require(StagedRound([live]))
    let adapter = try #require(overlay.caches.first as? RotatingStagedKVCache)

    let mask = adapter.makeMask(n: queries, windowSize: window, returnArray: false)
    let (keys, values) = adapter.update(
        keys: allKeys[0..., 0..., history ..< (history + queries), 0...],
        values: allValues[0..., 0..., history ..< (history + queries), 0...])
    let out = MLXFast.scaledDotProductAttention(
        queries: q, keys: keys, values: values, scale: scale, mask: mask)

    let queryPositions = MLXArray(Int32(history) ..< Int32(history + queries))[0..., .newAxis]
    let keyPositions = MLXArray(Int32(0) ..< Int32(history + queries))[.newAxis]
    let referenceMask =
        (queryPositions .>= keyPositions) & (queryPositions .< keyPositions + Int32(window))
    let expected = MLXFast.scaledDotProductAttention(
        queries: q, keys: allKeys, values: allValues, scale: scale, mask: .array(referenceMask))

    #expect(out.shape == expected.shape)
    #expect(
        allClose(out, expected, rtol: 1e-5, atol: 1e-5).item(Bool.self),
        "overlay attention diverged from the logical-order reference")
}

// MARK: - Invariant 3: commit equivalence

/// After a schedule of rounds, a cache driven through the overlay must hold exactly what a cache
/// that only ever saw the accepted tokens holds. Sliding attention only reads the window, so
/// equal windows and equal offsets is equality for every subsequent query.
private func expectCommitEquivalence(
    window: Int, keep: Int = 0, rounds: [(written: Int, accepted: Int)], prefill: Int,
    sourceLocation: SourceLocation = #_sourceLocation
) throws {
    let live = RotatingKVCache(maxSize: window, keep: keep)
    let reference = RotatingKVCache(maxSize: window, keep: keep)

    if prefill > 0 {
        let (k, v) = positionedKV(0 ..< prefill)
        _ = live.update(keys: k, values: v)
        _ = reference.update(keys: k, values: v)
    }

    var next = prefill
    for (index, round) in rounds.enumerated() {
        let overlay = try #require(StagedRound([live]), sourceLocation: sourceLocation)
        stageRound(overlay, next ..< (next + round.written))
        overlay.commit(accepted: round.accepted)

        if round.accepted > 0 {
            let (k, v) = positionedKV(next ..< (next + round.accepted))
            _ = reference.update(keys: k, values: v)
        }
        next += round.accepted

        #expect(
            live.offset == reference.offset,
            "round \(index): offset \(live.offset) != \(reference.offset)",
            sourceLocation: sourceLocation)

        let liveWindow = try #require(
            live.logicalView(tail: window), sourceLocation: sourceLocation)
        let referenceWindow = try #require(
            reference.logicalView(tail: window), sourceLocation: sourceLocation)
        #expect(
            encodedPositions(liveWindow.0) == encodedPositions(referenceWindow.0),
            """
            round \(index): window diverged — \
            \(encodedPositions(liveWindow.0)) vs \(encodedPositions(referenceWindow.0))
            """,
            sourceLocation: sourceLocation)
        #expect(
            encodedPositions(liveWindow.1).map { -$0 } == encodedPositions(referenceWindow.0),
            "round \(index): values diverged from keys",
            sourceLocation: sourceLocation)
    }
}

@Test func testOverlayCommitAcceptsNothing() throws {
    try expectCommitEquivalence(
        window: 8, rounds: Array(repeating: (written: 4, accepted: 0), count: 3), prefill: 6)
}

@Test func testOverlayCommitAcceptsEverything() throws {
    try expectCommitEquivalence(
        window: 8, rounds: Array(repeating: (written: 4, accepted: 4), count: 6), prefill: 6)
}

@Test func testOverlayCommitAcceptsPartialPrefixesAcrossTheWrap() throws {
    // Prefill sits just under the window, so the first rounds cross it and the rest run far past.
    try expectCommitEquivalence(
        window: 8,
        rounds: [
            (4, 1), (4, 4), (4, 0), (4, 2), (4, 3), (4, 1), (4, 4), (4, 2), (4, 0), (4, 3),
        ],
        prefill: 7)
}

@Test func testOverlayCommitEquivalenceHoldsWithAPinnedPrefix() throws {
    try expectCommitEquivalence(
        window: 8, keep: 2,
        rounds: [(4, 2), (4, 4), (4, 1), (4, 3), (4, 0), (4, 4)],
        prefill: 6)
}

@Test func testOverlayCommitEquivalenceFromAnEmptyCache() throws {
    try expectCommitEquivalence(
        window: 8, rounds: [(4, 3), (4, 4), (4, 1), (4, 4), (4, 2)], prefill: 0)
}

// MARK: - Invariant 6: rounds at or beyond the window

/// A round wider than the window is not reachable at a sane block size, but the API accepts an
/// arbitrary one. It has to stay correct rather than silently corrupt the ring.
@Test func testOverlayCommitEquivalenceWhenTheRoundExceedsTheWindow() throws {
    try expectCommitEquivalence(
        window: 4, rounds: [(6, 2), (6, 6), (6, 0), (6, 3)], prefill: 3)
    try expectCommitEquivalence(
        window: 4, rounds: [(8, 5), (8, 8), (8, 1)], prefill: 0)
}

// MARK: - Mixed topologies and the processed-token timeline

/// `KVCacheStorage` audits that every attention entry's offset equals the model-wide
/// processed-token count, so a commit has to advance sliding and global layers in lockstep.
@Test func testOverlayCommitAdvancesEveryLayerByExactlyTheAcceptedCount() throws {
    let rotating = RotatingKVCache(maxSize: 8, keep: 0)
    let simple = KVCacheSimple()
    let cache: [KVCache] = [simple, rotating, KVCacheSimple(), RotatingKVCache(maxSize: 8)]

    let (k, v) = positionedKV(0 ..< 7)
    cache.forEach { _ = $0.update(keys: k, values: v) }
    #expect(Set(cache.map(\.offset)) == [7])

    for (round, accepted) in [(0, 2), (1, 4), (2, 0), (3, 3)] {
        let overlay = try #require(StagedRound(cache))
        let before = try #require(cache.first?.offset)
        stageRound(overlay, 100 ..< 104)
        overlay.commit(accepted: accepted)

        let offsets = Set(cache.map(\.offset))
        #expect(
            offsets == [before + accepted],
            "round \(round): layers diverged — \(cache.map(\.offset))")
    }
}

// MARK: - begin: what the overlay refuses

@Test func testOverlayRefusesRecurrentState() {
    #expect(StagedRound([MambaCache()]) == nil)
    #expect(StagedRound([ArraysCache(size: 2)]) == nil)
    #expect(
        StagedRound([KVCacheSimple(), MambaCache()]) == nil,
        "one recurrent entry has to disqualify the whole array")
}

@Test func testOverlayRefusesNestedCacheLists() {
    let nested = CacheList(KVCacheSimple(), RotatingKVCache(maxSize: 8))
    #expect(
        StagedRound([nested]) == nil,
        "nested topologies are not supported yet and must fall back, not misalign")
}

@Test func testOverlayAcceptsTheGemma4Topology() throws {
    // Gemma 4's newCache: a standard cache for full-attention layers, a rotating one per sliding
    // layer, in a 4:1 pattern.
    let cache: [KVCache] = (0 ..< 10).map { index in
        index % 5 == 4 ? KVCacheSimple() : RotatingKVCache(maxSize: 512, keep: 0)
    }
    let overlay = try #require(StagedRound(cache))
    #expect(overlay.caches.count == cache.count)

    for (index, (live, presented)) in zip(cache, overlay.caches).enumerated() {
        if live is RotatingKVCache {
            #expect(presented is RotatingStagedKVCache, "sliding layer \(index) was not staged")
        } else {
            #expect(
                presented as AnyObject === live as AnyObject,
                "global layer \(index) should be written through")
        }
    }
}

@Test func testOverlayAcceptsQuantizedLayersAsWriteThrough() throws {
    let overlay = try #require(StagedRound([QuantizedKVCache()]))
    #expect(overlay.caches.first is QuantizedKVCache, "quantized layers are written through")
}

/// A cache type the overlay has never heard of is judged on the contract it advertises, not on
/// whether it appears in a list: write-through plus a clamp is sound exactly when `trim(_:)`
/// undoes the appended positions, which is what `isTrimmable` promises.
@Test func testOverlayJudgesUnknownCacheTypesByTrimmability() throws {
    let trimmable = ProbeCache(isTrimmable: true)
    let overlay = try #require(StagedRound([trimmable]))
    #expect(overlay.caches.first as AnyObject === trimmable as AnyObject)

    let (k, v) = positionedKV(0 ..< 4)
    _ = overlay.caches[0].update(keys: k, values: v)
    overlay.commit(accepted: 1)
    #expect(trimmable.offset == 1, "commit did not clamp an unknown write-through layer")

    #expect(
        StagedRound([ProbeCache(isTrimmable: false)]) == nil,
        "a cache that cannot be rewound cannot be written through")
}

/// A minimal `KVCache` conformer that is none of the types the overlay knows by name.
private final class ProbeCache: KVCache {
    var offset: Int = 0
    let trimmable: Bool

    init(isTrimmable: Bool) { self.trimmable = isTrimmable }

    var maxSize: Int? { nil }
    var isTrimmable: Bool { trimmable }

    func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        offset += keys.dim(2)
        return (keys, values)
    }

    @discardableResult
    func trim(_ n: Int) -> Int {
        guard trimmable else { return 0 }
        let removed = Swift.min(n, offset)
        offset -= removed
        return removed
    }

    var state: [MLXArray] {
        get { [] }
        set {}
    }
    var metaState: [String] {
        get { [] }
        set {}
    }
    func makeMask(
        n: Int, windowSize: Int?, returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode { .none }
    func copy() -> any KVCache { ProbeCache(isTrimmable: trimmable) }
    func innerState() -> [MLXArray] { [] }
}

// MARK: - Adapter surface

@Test func testFirstStagedPresentationPreservesInputDTypeAndShape() throws {
    let live = RotatingKVCache(maxSize: 8, keep: 0)
    let overlay = try #require(StagedRound([live]))
    let adapter = try #require(overlay.caches.first as? RotatingStagedKVCache)
    let keys = MLXArray.ones([1, 2, 3, 4], dtype: .bfloat16)
    let values = MLXArray.zeros([1, 2, 3, 4], dtype: .bfloat16)

    let presented = adapter.update(keys: keys, values: values)

    #expect(presented.0.dtype == .bfloat16)
    #expect(presented.1.dtype == .bfloat16)
    #expect(presented.0.shape == keys.shape)
    #expect(presented.1.shape == values.shape)
    #expect(live.offset == 0, "staging the first rows must not initialize the live ring")
}

@Test func testOverlayAdapterReportsStagedPositionsInItsOffset() throws {
    let live = RotatingKVCache(maxSize: 8, keep: 0)
    let (k, v) = positionedKV(0 ..< 5)
    _ = live.update(keys: k, values: v)

    let overlay = try #require(StagedRound([live]))
    let adapter = try #require(overlay.caches.first as? RotatingStagedKVCache)

    #expect(adapter.offset == 5, "before staging the adapter must read as the live cache")
    #expect(adapter.maxSize == 8)

    stageRound(overlay, 5 ..< 9)
    #expect(adapter.stagedCount == 4)
    #expect(adapter.offset == 9)
    #expect(live.offset == 5, "staging must not move the live offset")

    overlay.commit(accepted: 2)
    #expect(adapter.stagedCount == 0, "commit must clear staging")
    #expect(live.offset == 7)
}

@Test func testOverlayAdapterIsNotTrimmable() throws {
    let live = RotatingKVCache(maxSize: 8, keep: 0)
    let (k, v) = positionedKV(0 ..< 5)
    _ = live.update(keys: k, values: v)

    let overlay = try #require(StagedRound([live]))
    let adapter = try #require(overlay.caches.first as? RotatingStagedKVCache)

    #expect(adapter.isTrimmable == false)
    #expect(adapter.trim(3) == 0, "staged rows are dropped by commit, never trimmed")
    #expect(live.offset == 5)
}

@Test func testOverlayAdapterInnerStateExposesStagedArraysForEval() throws {
    let live = RotatingKVCache(maxSize: 8, keep: 0)
    let (k, v) = positionedKV(0 ..< 5)
    _ = live.update(keys: k, values: v)

    let overlay = try #require(StagedRound([live]))
    let adapter = try #require(overlay.caches.first as? RotatingStagedKVCache)
    let liveArrays = adapter.innerState().count

    stageRound(overlay, 5 ..< 9)
    #expect(
        adapter.innerState().count == liveArrays + 2,
        "staged K/V must be reachable from innerState or eval would miss them")
}

// MARK: - Oracle: a staged round vs an accepted-only replay
//
// The tests above check the overlay against hand-computed expectations. This one checks it
// against the other implementation of the same idea: take a deep snapshot before the round, run
// the round, then replay *only* the accepted tokens into the snapshot through the ordinary write
// path. Snapshot-and-replay is correct by construction and slow; the overlay is fast and has to
// agree with it exactly. `KVCacheStorage.copy()` is the snapshot.

/// Everything that has to match between a storage driven through staged rounds and one that only
/// ever saw the accepted tokens.
private func expectStoragesEquivalent(
    _ staged: KVCacheStorage, _ replayed: KVCacheStorage, _ note: String,
    sourceLocation: SourceLocation = #_sourceLocation
) {
    #expect(
        staged.processedTokenCount == replayed.processedTokenCount,
        "\(note): timeline \(staged.processedTokenCount) != \(replayed.processedTokenCount)",
        sourceLocation: sourceLocation)
    #expect(
        staged.nativeAttentionOffsetsAreAligned,
        "\(note): staged leaves drifted from the timeline", sourceLocation: sourceLocation)
    #expect(
        replayed.nativeAttentionOffsetsAreAligned,
        "\(note): replayed leaves drifted from the timeline", sourceLocation: sourceLocation)

    #expect(staged.cache.count == replayed.cache.count, sourceLocation: sourceLocation)
    for (index, (a, b)) in zip(staged.cache, replayed.cache).enumerated() {
        #expect(
            a.offset == b.offset, "\(note): leaf \(index) offset", sourceLocation: sourceLocation)

        if let a = a as? RotatingKVCache, let b = b as? RotatingKVCache {
            guard let aView = a.logicalView(tail: a.maxSize ?? 0),
                let bView = b.logicalView(tail: b.maxSize ?? 0)
            else {
                #expect(
                    a.logicalView(tail: 1) == nil && b.logicalView(tail: 1) == nil,
                    "\(note): leaf \(index) — one ring is empty and the other is not",
                    sourceLocation: sourceLocation)
                continue
            }
            #expect(
                encodedPositions(aView.0) == encodedPositions(bView.0),
                """
                \(note): leaf \(index) window \(encodedPositions(aView.0)) \
                != \(encodedPositions(bView.0))
                """,
                sourceLocation: sourceLocation)
            // Physical layout too: `idx` and the ring's rotation are invisible to the logical
            // view, and a divergence there would surface later as a wrong single-token write.
            #expect(
                a.metaState == b.metaState,
                "\(note): leaf \(index) ring layout \(a.metaState) != \(b.metaState)",
                sourceLocation: sourceLocation)
        } else {
            let aState = a.state
            let bState = b.state
            #expect(
                aState.count == bState.count, "\(note): leaf \(index) state arity",
                sourceLocation: sourceLocation)
            for (arrayIndex, (x, y)) in zip(aState, bState).enumerated() {
                guard x.shape == y.shape else {
                    Issue.record(
                        "\(note): leaf \(index) state[\(arrayIndex)] \(x.shape) != \(y.shape)",
                        sourceLocation: sourceLocation)
                    continue
                }
                #expect(
                    allClose(x, y, rtol: 0, atol: 0).item(Bool.self),
                    "\(note): leaf \(index) state[\(arrayIndex)] values",
                    sourceLocation: sourceLocation)
            }
        }
    }
}

private func makeGemmaShapedStorage(window: Int, keep: Int) -> KVCacheStorage {
    KVCacheStorage(
        [KVCacheSimple(), RotatingKVCache(maxSize: window, keep: keep)], plan: .disabled)
}

private func expectRoundMatchesAcceptedOnlyReplay(
    window: Int, keep: Int = 0, prefill: Int, rounds: [(written: Int, retaining: Int)],
    sourceLocation: SourceLocation = #_sourceLocation
) throws {
    let live = makeGemmaShapedStorage(window: window, keep: keep)

    if prefill > 0 {
        let (k, v) = positionedKV(0 ..< prefill)
        for leaf in live.cache { _ = leaf.update(keys: k, values: v) }
        live.commitProcessedTokens(prefill)
    }

    var next = prefill
    for (index, round) in rounds.enumerated() {
        // The oracle has to be taken before the round: `copy()` refuses to run inside one.
        let oracle = live.copy()

        let overlay = try #require(
            StagedRound(live), sourceLocation: sourceLocation)
        let (writtenKeys, writtenValues) = positionedKV(next ..< (next + round.written))
        for presented in overlay.caches {
            _ = presented.update(keys: writtenKeys, values: writtenValues)
        }
        overlay.commit(accepted: round.retaining)

        if round.retaining > 0 {
            let (k, v) = positionedKV(next ..< (next + round.retaining))
            for leaf in oracle.cache { _ = leaf.update(keys: k, values: v) }
            oracle.commitProcessedTokens(round.retaining)
        }
        next += round.retaining

        expectStoragesEquivalent(live, oracle, "round \(index)", sourceLocation: sourceLocation)
    }
}

@Test func testStagedRoundMatchesAcceptedOnlyReplayFromEmpty() throws {
    try expectRoundMatchesAcceptedOnlyReplay(
        window: 8, prefill: 0, rounds: [(4, 3), (4, 4), (4, 1), (4, 4), (4, 2)])
}

@Test func testStagedRoundMatchesAcceptedOnlyReplayAcrossTheWrap() throws {
    try expectRoundMatchesAcceptedOnlyReplay(
        window: 8, prefill: 7,
        rounds: [(4, 1), (4, 4), (4, 0), (4, 2), (4, 3), (4, 1), (4, 4), (4, 2), (4, 0), (4, 3)])
}

@Test func testStagedRoundMatchesAcceptedOnlyReplayWithAPinnedPrefix() throws {
    try expectRoundMatchesAcceptedOnlyReplay(
        window: 8, keep: 2, prefill: 6, rounds: [(4, 2), (4, 4), (4, 1), (4, 3), (4, 0), (4, 4)])
}

@Test func testStagedRoundMatchesAcceptedOnlyReplayWhenWiderThanTheWindow() throws {
    try expectRoundMatchesAcceptedOnlyReplay(
        window: 4, prefill: 3, rounds: [(6, 2), (6, 6), (6, 0), (6, 3)])
}

/// The timeline half of the oracle, without the replay: a round leaves the storage's
/// processed-token count and its leaves in step. (The `roundIsOpen` refusals this checks
/// against are inert until the transaction is hosted on the storage itself.)
@Test func testStagedRoundLeavesTheTimelineAligned() throws {
    let live = makeGemmaShapedStorage(window: 8, keep: 0)
    let (k, v) = positionedKV(0 ..< 5)
    for leaf in live.cache { _ = leaf.update(keys: k, values: v) }
    live.commitProcessedTokens(5)

    #expect(live.roundIsOpen == false)
    let overlay = try #require(StagedRound(live))
    stageRound(overlay, 5 ..< 9)
    overlay.commit(accepted: 2)
    #expect(live.processedTokenCount == 7)
    #expect(live.nativeAttentionOffsetsAreAligned)
}

// MARK: - Taking back committed-but-unemitted positions
//
// A round commits its accepted prefix before the consumer has drained it, so a stop token partway
// through leaves positions in the cache that were never emitted. Past a wrapped ring a plain trim
// cannot take them back — the append already evicted what the rewind would restore — so the
// commit keeps a restore point. Same oracle as above: the answer has to equal a cache that only
// ever saw the positions that were actually emitted.

private func expectRewindMatchesEmittedOnlyReplay(
    window: Int, keep: Int = 0, prefill: Int, written: Int, retaining: Int, rewinding: Int,
    sourceLocation: SourceLocation = #_sourceLocation
) throws {
    let live = makeGemmaShapedStorage(window: window, keep: keep)
    if prefill > 0 {
        let (k, v) = positionedKV(0 ..< prefill)
        for leaf in live.cache { _ = leaf.update(keys: k, values: v) }
        live.commitProcessedTokens(prefill)
    }

    let oracle = live.copy()

    let staged = try #require(
        StagedRound(live, width: written), sourceLocation: sourceLocation)
    stageRound(staged, prefill ..< (prefill + written))
    staged.commit(accepted: retaining)

    let rewound = live.rewindLastRound(rewinding)
    #expect(
        rewound == Swift.min(rewinding, retaining),
        "rewind took back \(rewound), expected \(Swift.min(rewinding, retaining))",
        sourceLocation: sourceLocation)

    // The oracle only ever sees the positions that survived both steps.
    let emitted = retaining - rewound
    if emitted > 0 {
        let (k, v) = positionedKV(prefill ..< (prefill + emitted))
        for leaf in oracle.cache { _ = leaf.update(keys: k, values: v) }
        oracle.commitProcessedTokens(emitted)
    }

    expectStoragesEquivalent(
        live, oracle, "rewind \(rewinding) of \(retaining)", sourceLocation: sourceLocation)
}

@Test(arguments: [1, 2, 3, 4])
func testRewindPastTheWrapMatchesEmittedOnlyReplay(rewinding: Int) throws {
    // Prefill alone clears the window, so every ring here has already wrapped and a plain trim
    // would silently no-op.
    try expectRewindMatchesEmittedOnlyReplay(
        window: 8, prefill: 20, written: 4, retaining: 4, rewinding: rewinding)
}

@Test(arguments: [1, 2, 3])
func testRewindBelowTheWrapMatchesEmittedOnlyReplay(rewinding: Int) throws {
    // Below the window the commit needs no restore point; the exact trim path handles it.
    try expectRewindMatchesEmittedOnlyReplay(
        window: 32, prefill: 4, written: 4, retaining: 4, rewinding: rewinding)
}

@Test func testRewindPastTheWrapWithAPinnedPrefix() throws {
    try expectRewindMatchesEmittedOnlyReplay(
        window: 8, keep: 2, prefill: 20, written: 4, retaining: 3, rewinding: 2)
}

@Test func testRewindingEverythingTheRoundCommittedPastTheWrap() throws {
    try expectRewindMatchesEmittedOnlyReplay(
        window: 8, prefill: 20, written: 4, retaining: 3, rewinding: 3)
}

/// The record is only good for the most recent commit: anything written after it invalidates the
/// pre-commit contents the restore point refers to.
@Test func testRewindIsRefusedAfterAnotherWrite() throws {
    let live = makeGemmaShapedStorage(window: 8, keep: 0)
    let (k, v) = positionedKV(0 ..< 20)
    for leaf in live.cache { _ = leaf.update(keys: k, values: v) }
    live.commitProcessedTokens(20)

    let staged = try #require(StagedRound(live, width: 4))
    stageRound(staged, 20 ..< 24)
    staged.commit(accepted: 4)
    #expect(live.processedTokenCount == 24)

    // A single-token write outside a round: exactly what passthrough decoding does.
    let (k2, v2) = positionedKV(24 ..< 25)
    for leaf in live.cache { _ = leaf.update(keys: k2, values: v2) }
    live.commitProcessedTokens(1)

    // The rotating ring has wrapped, so the fallback trim reports that it took nothing back
    // rather than corrupting the ring.
    #expect(live.rewindLastRound(2) == 0)
    #expect(live.processedTokenCount == 25)
    #expect(live.nativeAttentionOffsetsAreAligned)
}

/// Two sliding layers at different widths: at a timeline between them one has wrapped and one has
/// not, so the same commit produces a restore point for the first and a plain trim record for the
/// second. The trim has to reach the *live* ring — the round presented a staged wrapper whose
/// `trim(_:)` is deliberately a no-op, and rewinding through that moves the timeline while the
/// leaf stays put.
@Test func testRewindReachesTheLiveRingWhenOnlySomeLayersWrapped() throws {
    let narrow = RotatingKVCache(maxSize: 4, keep: 0)
    let wide = RotatingKVCache(maxSize: 16, keep: 0)
    let live = KVCacheStorage([narrow, wide], plan: .disabled)

    let (k, v) = positionedKV(0 ..< 6)
    for leaf in live.cache { _ = leaf.update(keys: k, values: v) }
    live.commitProcessedTokens(6)
    #expect(narrow.isTrimmable == false, "the narrow ring must have wrapped")
    #expect(wide.isTrimmable, "the wide ring must not have wrapped")

    let oracle = live.copy()

    let staged = try #require(StagedRound(live, width: 3))
    stageRound(staged, 6 ..< 9)
    staged.commit(accepted: 3)

    #expect(live.rewindLastRound(2) == 2)

    let (ok, ov) = positionedKV(6 ..< 7)
    for leaf in oracle.cache { _ = leaf.update(keys: ok, values: ov) }
    oracle.commitProcessedTokens(1)

    expectStoragesEquivalent(live, oracle, "mixed-width rewind")
}

/// Dynamic compression can replace an entry between a commit and a rewind. The record refers to a
/// cache slot, not to the object the commit happened to hold, so a replacement invalidates it and
/// the rewind falls back to the plain trim rather than restoring into a leaf the storage no longer
/// owns.
@Test func testRewindIsRefusedAfterTheCacheIsReplaced() throws {
    let original = RotatingKVCache(maxSize: 8, keep: 0)
    let live = KVCacheStorage([original], plan: .disabled)
    let (k, v) = positionedKV(0 ..< 20)
    _ = original.update(keys: k, values: v)
    live.commitProcessedTokens(20)

    let staged = try #require(StagedRound(live, width: 4))
    stageRound(staged, 20 ..< 24)
    staged.commit(accepted: 4)
    #expect(live.processedTokenCount == 24)

    // Stand in for a compression pass: an equivalent leaf, a different object.
    let replacement = try #require(original.copy() as? RotatingKVCache)
    let replacementBefore = RingSnapshot(replacement)
    live.replace(with: [replacement])

    // The ring has wrapped, so the fallback trim takes nothing back rather than restoring a
    // snapshot into a cache that is no longer the one at this slot.
    #expect(live.rewindLastRound(2) == 0)
    #expect(live.processedTokenCount == 24)
    #expect(live.nativeAttentionOffsetsAreAligned)
    replacementBefore.expectUnchanged(replacement, "after a rewind against a replaced entry")
}
