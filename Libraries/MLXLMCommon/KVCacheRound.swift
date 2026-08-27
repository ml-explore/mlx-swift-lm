// Copyright © 2026 Apple Inc.

import Foundation
import MLX

/// How one cache leaf takes part in a staged round.
///
/// Speculative decoding writes several candidate positions and keeps only a prefix. Undoing the
/// rest by writing everything and then trimming works for an append-only cache, whose `trim(_:)`
/// is pure bookkeeping, but not for a ``RotatingKVCache``: once its ring has wrapped, the write a
/// rewind would undo has already evicted the entries that rewind would have to restore. So the
/// leaf chooses: provisional tail and a cursor commit, or staged K/V and a chronological view.
///
/// Selection is fail-closed. ``KVCacheStorage/beginRound(maximumPositions:)`` classifies every
/// leaf before constructing anything, and a single leaf that cannot commit exactly abandons the
/// round with nothing touched.
package protocol KVCacheRoundStrategy: AnyObject {
    /// Which entry of the storage's array this leaf occupies.
    ///
    /// The slot is the leaf's stable identity, and the only one worth recording: ``presented`` is
    /// round-scoped for a staged leaf, and dynamic compression can replace the live object at any
    /// point outside a round. Anything that has to find this leaf again looks it up here.
    var slot: Int { get }

    /// The cache to hand the model in this leaf's slot for the duration of the round.
    var presented: KVCache { get }

    /// Positions written through ``presented`` so far this round.
    var writtenPositions: Int { get }

    /// The live entry's own position. Reads as the pre-round value until the round commits.
    var liveOffset: Int { get }

    /// The window the live entry attends over, or `nil` when it is unbounded.
    var attentionBound: Int? { get }

    /// Keep the first `retaining` written positions and drop the rest. Called at most once.
    func commit(retaining: Int)

    /// Enough to put this leaf back should the committed positions turn out not to have been
    /// emitted after all, or `nil` when a later `trim(_:)` would undo them exactly.
    ///
    /// Called immediately before ``commit(retaining:)``, because that is the last moment the
    /// pre-commit contents still exist.
    func makeRestorePoint(retaining: Int) -> (any KVCacheLeafRestorePoint)?
}

extension KVCacheRoundStrategy {
    /// Most leaves need nothing: their `trim(_:)` is bookkeeping and undoes an append exactly.
    func makeRestorePoint(retaining: Int) -> (any KVCacheLeafRestorePoint)? { nil }
}

/// Enough to put one leaf back the way it was before a commit, and replay part of it.
///
/// A generation can stop with positions committed but never emitted -- a stop token partway
/// through a round's accepted prefix, a consumer that walked away. Those have to come back out of
/// the cache, and for a wrapped ring `trim(_:)` cannot do it: the append already evicted the
/// entries the rewind would need to restore.
///
/// Deliberately holds no reference to the cache it was made from. The leaf is resolved from the
/// storage at rewind time and passed in, so a record can never write into an object the storage
/// has stopped owning.
package protocol KVCacheLeafRestorePoint {
    /// Whether `leaf` is something this record can be applied to.
    func canRestore(into leaf: KVCache) -> Bool

    /// Restore `leaf`'s pre-commit contents, then re-apply the first `retaining` committed
    /// positions. Returns `false` if `leaf` is not a cache this record was made from, in which
    /// case nothing is written.
    @discardableResult
    func restore(into leaf: KVCache, retaining: Int) -> Bool
}

extension KVCacheRoundStrategy {
    /// How much of the emitted sequence this leaf can still describe after a commit: everything
    /// for a global layer, the trailing window for a sliding one.
    var emittedLength: Int {
        Swift.min(liveOffset, attentionBound ?? .max)
    }
}

/// An append-only leaf: written through, then clamped back over the rejected rows.
///
/// This is what the rewind already did, and it is deliberately cheaper than staging — a global
/// layer's presentation is its whole history, so staging one would cost a concatenation across
/// the full context on every round.
final class AppendOnlyRoundStrategy: KVCacheRoundStrategy {
    let slot: Int
    let live: KVCache
    private let startOffset: Int

    init(slot: Int, live: KVCache) {
        self.slot = slot
        self.live = live
        self.startOffset = live.offset
    }

    var presented: KVCache { live }
    var writtenPositions: Int { live.offset - startOffset }
    var liveOffset: Int { live.offset }
    var attentionBound: Int? { live.maxSize }

    func commit(retaining: Int) {
        let written = writtenPositions
        let keep = Swift.min(Swift.max(retaining, 0), written)
        if written > keep {
            live.trim(written - keep)
        }
    }
}

/// A rotating leaf: staged beside the ring, which stays physically untouched until commit.
final class RotatingRoundStrategy: KVCacheRoundStrategy {
    let slot: Int
    let live: RotatingKVCache
    let staged: RotatingStagedKVCache

    init(slot: Int, live: RotatingKVCache) {
        self.slot = slot
        self.live = live
        self.staged = RotatingStagedKVCache(live: live)
    }

    var presented: KVCache { staged }
    var writtenPositions: Int { staged.stagedCount }
    var liveOffset: Int { live.offset }
    var attentionBound: Int? { live.maxSize }

    func commit(retaining: Int) {
        staged.commit(retaining: retaining)
    }

    func makeRestorePoint(retaining: Int) -> (any KVCacheLeafRestorePoint)? {
        // Below the window a trim is exact bookkeeping, so a snapshot would be dead weight on
        // every round of every short generation. Ask about the post-commit position, since that
        // is the state a later rewind would be undoing.
        guard !live.isTrimmable(after: retaining) else { return nil }
        let snapshot = live.state
        guard snapshot.count == 2 else { return nil }
        return RotatingRestorePoint(
            // Detached the same way `RotatingKVCache.copy()` detaches: a full-range slice shares
            // the node but is a new object, and an in-place write rebinds the *receiver*. Holding
            // `live.state` directly would hold the very objects the commit is about to rebind.
            state: snapshot.map { $0[.ellipsis] },
            metaState: live.metaState,
            replay: staged.stagedArrays)
    }
}

/// Restores a rotating leaf by putting its ring back and replaying the accepted rows.
///
/// Capture is cheap -- a full-range slice reuses the underlying node -- so the price is deferred:
/// while the snapshot is alive the next ring write cannot donate its buffer. It is dropped as
/// soon as another round opens.
struct RotatingRestorePoint: KVCacheLeafRestorePoint {
    let state: [MLXArray]
    let metaState: [String]
    let replay: (MLXArray, MLXArray)?

    func canRestore(into leaf: KVCache) -> Bool { leaf is RotatingKVCache }

    @discardableResult
    func restore(into leaf: KVCache, retaining: Int) -> Bool {
        guard let live = leaf as? RotatingKVCache else { return false }
        // Order matters and mirrors `copy()`: the state setter deliberately leaves `offset`
        // alone, because `idx` and `offset` both arrive with the metadata.
        live.state = state
        live.metaState = metaState
        guard retaining > 0, let replay else { return true }
        _ = live.update(
            keys: replay.0[.ellipsis, ..<retaining, 0...],
            values: replay.1[.ellipsis, ..<retaining, 0...])
        return true
    }
}

/// Chooses a strategy per leaf, or refuses.
///
/// Deliberately not a requirement on ``KVCache`` itself: that protocol is public, so a `package`
/// requirement could not be added to it, and a public one would be new API. The consequence is a
/// ceiling worth naming — a conformer from outside this package cannot opt into staging. It gets
/// write-through if it can prove it stays trimmable, and refusal otherwise.
enum KVCacheRoundStrategyFactory {
    /// - Parameter slot: the leaf's index in the storage's array, which the strategy carries so a
    ///   later rewind can resolve the live entry rather than trust a captured object.
    static func make(
        for leaf: KVCacheLeaf, slot: Int, maximumPositions: Int
    ) -> (any KVCacheRoundStrategy)? {
        // Recurrent state has no rewindable position and no staged form, and deliberately does
        // not participate in the shared timeline at all.
        guard leaf.isAttentionCache else { return nil }

        switch leaf.kind {
        case .rotating(let rotating):
            return RotatingRoundStrategy(slot: slot, live: rotating)
        case .recurrent:
            return nil
        case .simple, .affine, .turboQuant, .varianceNormalized, .unsupported:
            // Write-through plus a clamp is sound exactly when a later trim undoes the appended
            // positions -- asked with the round's real width, because the entry that matters is
            // one that is trimmable now and would stop being trimmable during the round.
            guard leaf.cache.isTrimmable(after: maximumPositions) else { return nil }
            return AppendOnlyRoundStrategy(slot: slot, live: leaf.cache)
        }
    }
}

/// An open staged round: the caches to run the model against, and the per-leaf strategies that
/// will commit them.
///
/// Deliberately has no `commit` of its own. Committing means advancing the model-wide
/// processed-token timeline by exactly the retained count, and that timeline lives on
/// ``KVCacheStorage`` — so commit does too, and the two cannot be separated or reordered.
package final class KVCacheRound {
    let strategies: [any KVCacheRoundStrategy]

    /// Positionally matches the storage's live array.
    package let caches: [KVCache]

    /// The width the round was opened for. Leaves were admitted on the promise of absorbing it.
    package let maximumPositions: Int

    init(strategies: [any KVCacheRoundStrategy], maximumPositions: Int) {
        self.strategies = strategies
        self.caches = strategies.map(\.presented)
        self.maximumPositions = maximumPositions
    }

    /// Positions this round wrote, which every leaf must agree on.
    var writtenPositions: Int {
        guard let first = strategies.first else { return 0 }
        let written = first.writtenPositions
        assert(
            strategies.allSatisfy { $0.writtenPositions == written },
            "leaves disagree on how much this round wrote: "
                + "\(strategies.map(\.writtenPositions))")
        return written
    }
}

/// What a commit retained, and what the leaves can still describe afterwards.
package struct KVCacheRoundCommit {
    /// Positions kept. The timeline advanced by exactly this much.
    package let committedPositions: Int

    /// Positions written and then dropped — the rejected tail.
    package let discardedPositions: Int

    /// Per leaf, how much of the emitted sequence it still describes: the whole stream for a
    /// global layer, the trailing window for a sliding one.
    package let emittedLengths: [Int]
}
