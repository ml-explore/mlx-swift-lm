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
    let live: KVCache
    private let startOffset: Int

    init(live: KVCache) {
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
    let live: RotatingKVCache
    let staged: RotatingStagedKVCache

    init(live: RotatingKVCache) {
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
}

/// Chooses a strategy per leaf, or refuses.
///
/// Deliberately not a requirement on ``KVCache`` itself: that protocol is public, so a `package`
/// requirement could not be added to it, and a public one would be new API. The consequence is a
/// ceiling worth naming — a conformer from outside this package cannot opt into staging. It gets
/// write-through if it can prove it stays trimmable, and refusal otherwise.
enum KVCacheRoundStrategyFactory {
    static func make(
        for leaf: KVCacheLeaf, maximumPositions: Int
    ) -> (any KVCacheRoundStrategy)? {
        // Recurrent state has no rewindable position and no staged form, and deliberately does
        // not participate in the shared timeline at all.
        guard leaf.isAttentionCache else { return nil }

        switch leaf.kind {
        case .rotating(let rotating):
            return RotatingRoundStrategy(live: rotating)
        case .recurrent:
            return nil
        case .simple, .affine, .turboQuant, .unsupported:
            // Write-through plus a clamp is sound exactly when a later trim undoes the appended
            // positions -- asked with the round's real width, because the entry that matters is
            // one that is trimmable now and would stop being trimmable during the round.
            guard leaf.cache.isTrimmable(after: maximumPositions) else { return nil }
            return AppendOnlyRoundStrategy(live: leaf.cache)
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
