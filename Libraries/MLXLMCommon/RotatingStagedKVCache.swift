// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN

/// A ``RotatingKVCache`` presented with uncommitted rows appended.
///
/// `update(keys:values:)` stages its arguments instead of writing them and returns the trailing
/// window of the live ring followed by everything staged so far -- the same arrays, in the same
/// chronological order, that the live cache would have returned had it written them.
final class RotatingStagedKVCache: KVCache {
    let live: RotatingKVCache

    private var stagedKeys: MLXArray?
    private var stagedValues: MLXArray?

    init(live: RotatingKVCache) {
        self.live = live
    }

    /// Positions staged but not yet committed.
    var stagedCount: Int { stagedKeys?.dim(2) ?? 0 }

    /// The staged rows, for a caller that needs to replay them after a restore. Valid until
    /// ``commit(retaining:)`` clears them.
    var stagedArrays: (MLXArray, MLXArray)? {
        guard let stagedKeys, let stagedValues else { return nil }
        return (stagedKeys, stagedValues)
    }

    /// The live ring's position plus whatever is staged.
    ///
    /// Masks are built before the layer writes, so at mask time this reads as the live offset --
    /// which is exactly what makes delegating ``makeMask(n:windowSize:returnArray:)`` correct.
    var offset: Int { live.offset + stagedCount }

    var maxSize: Int? { live.maxSize }

    func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        stagedKeys = stagedKeys.map { concatenated([$0, keys], axis: 2) } ?? keys
        stagedValues = stagedValues.map { concatenated([$0, values], axis: 2) } ?? values
        return presentation()
    }

    /// The live cache's own mask.
    ///
    /// For a multi-token write ``RotatingKVCache`` builds a causal-intersect-window mask over
    /// `min(maxSize - 1, offset) + n` chronological key positions, and that is precisely the
    /// length of the presentation below. Delegating means the model sees the mask it would have
    /// seen without a staged round, rather than one this type had to reconstruct.
    func makeMask(
        n: Int, windowSize: Int?, returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        live.makeMask(n: n, windowSize: windowSize, returnArray: returnArray)
    }

    /// Append the first `retaining` staged positions to the ring through the normal write path and
    /// drop the rest. The ring is untouched until this point.
    func commit(retaining: Int) {
        defer {
            stagedKeys = nil
            stagedValues = nil
        }
        guard let stagedKeys, let stagedValues else { return }
        let keep = Swift.min(Swift.max(retaining, 0), stagedKeys.dim(2))
        guard keep > 0 else { return }
        _ = live.update(
            keys: stagedKeys[.ellipsis, ..<keep, 0...],
            values: stagedValues[.ellipsis, ..<keep, 0...])
    }

    private func presentation() -> (MLXArray, MLXArray) {
        guard let stagedKeys, let stagedValues else {
            return live.logicalView(tail: live.offset) ?? emptyPresentation()
        }
        // `maxSize - 1` is the same bound the multi-token write path front-trims to, so the first
        // staged query still reaches back a full window.
        let bound = Swift.min(live.offset, (live.maxSize ?? Int.max) - 1)
        guard bound > 0, let view = live.logicalView(tail: bound) else {
            return (stagedKeys, stagedValues)
        }
        return (
            concatenated([view.0, stagedKeys], axis: 2),
            concatenated([view.1, stagedValues], axis: 2)
        )
    }

    private func emptyPresentation() -> (MLXArray, MLXArray) {
        let empty = MLXArray.zeros([1, 1, 0, 1])
        return (empty, empty)
    }

    func innerState() -> [MLXArray] {
        live.innerState() + [stagedKeys, stagedValues].compactMap { $0 }
    }

    // MARK: - Round-scoped: not serializable, not rewindable, not copyable

    /// This never outlives its round, and prompt caches are saved between generations, so the live
    /// ring is the whole answer here. Restoring into one is a programming error.
    var state: [MLXArray] {
        get { live.state }
        set(newState) {
            fatalError(
                "RotatingStagedKVCache is round-scoped and cannot be restored from state "
                    + "(\(newState.count) arrays)")
        }
    }

    var metaState: [String] {
        get { live.metaState }
        set(newMetaState) {
            fatalError(
                "RotatingStagedKVCache is round-scoped and cannot be restored from metaState "
                    + "(\(newMetaState.count) values)")
        }
    }

    /// Staged rows are dropped wholesale by ``commit(retaining:)``, never trimmed. Reporting
    /// `false` keeps a caller from mistaking this for something it can rewind.
    var isTrimmable: Bool { false }

    @discardableResult
    func trim(_ n: Int) -> Int { 0 }

    func copy() -> any KVCache {
        fatalError("RotatingStagedKVCache is round-scoped and cannot be copied")
    }

    func prepare(lengths: [Int]?) { live.prepare(lengths: lengths) }

    func prepare(lengths: MLXArray?) { live.prepare(lengths: lengths) }

    func finalize() { live.finalize() }
}
