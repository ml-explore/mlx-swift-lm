// Copyright © 2026 Apple Inc.

import Foundation
import MLX

/// A sliding-window KV cache for a batch of left-padded sequences.
///
/// Port of mlx-lm's `BatchRotatingKVCache`: the left padding of ``BatchKVCache`` with
/// the window of ``RotatingKVCache``. Every row keeps its last ``maxSize`` positions.
/// Unlike ``RotatingKVCache`` it has no pinned prefix (`keep`) and no rewind reserve.
public final class BatchRotatingKVCache: BaseKVCache, BatchPositionedKVCache {
    /// Each row's left padding, shape `[B]`: the padding positions the buffer still holds.
    /// Writes consume the oldest positions first, so it falls to zero as the window fills
    /// (mlx-lm lets it go negative; this keeps a count, so a later ragged prefill or
    /// extension adds its padding to the real figure).
    public private(set) var leftPadding: MLXArray

    /// Each row's position, shape `[B]`. It starts at `-leftPadding`.
    public private(set) var batchOffset: MLXArray

    var keys: MLXArray?
    var values: MLXArray?

    private var window: Int
    /// The buffer position the next token is written to.
    private var idx = 0
    /// The tokens written to the buffer so far.
    private var written = 0
    /// Whether writes have wrapped around the buffer.
    private var rotated = false
    /// Each row's end during a ragged prefill, from `prepare` until `finalize`.
    private var lengths: MLXArray?

    public var step = 256

    /// Whether the window came from the model or from a requested capacity.
    package var capacityOrigin = RotatingKVCache.CapacityOrigin.modelNative

    public override var maxSize: Int? { window }

    /// The position of the longest row, as ``KVCache/offset`` counts it: every
    /// token written, not only the window's. Rows keep their own positions in
    /// ``batchOffset``, so the batch as a whole cannot be moved.
    public override var offset: Int {
        get { batchSize > 0 ? Swift.max(0, Int(batchOffset.max().item(Int32.self))) : 0 }
        set { preconditionFailure("BatchRotatingKVCache positions are per row; see batchOffset") }
    }

    /// - Parameters:
    ///   - maxSize: the window
    ///   - leftPadding: the left padding of each row
    public init(maxSize: Int, leftPadding: [Int]) {
        self.window = maxSize
        self.leftPadding = MLXArray(leftPadding.map { Int32($0) })
        self.batchOffset = MLXArray(leftPadding.map { -Int32($0) })
        super.init()
    }

    /// The number of rows.
    public var batchSize: Int { leftPadding.dim(0) }

    /// Per-row positions, also when the cache is typed as `KVCache`. No copy: every
    /// change replaces the array rather than mutating it.
    public override var ropeOffset: RoPEOffset { .batch(batchOffset) }

    public override func innerState() -> [MLXArray] {
        [keys, values].compactMap { $0 }
    }

    /// Writes one decode token in place, and everything else (prefill, ragged steps)
    /// by concatenation.
    public override func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        if keys.dim(2) == 1 && lengths == nil {
            return updateInPlace(keys: keys, values: values)
        }
        return updateConcat(keys: keys, values: values)
    }

    private func updateConcat(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        if self.keys == nil {
            self.keys = keys
            self.values = values
        } else {
            temporalOrder()
            if self.keys!.dim(2) > idx {
                self.keys = self.keys![.ellipsis, ..<idx, 0...]
                self.values = self.values![.ellipsis, ..<idx, 0...]
            }

            // Roll right-padded rows so the trim below keeps their tokens.
            if let lengths {
                let roll = MLX.maximum(MLXArray(Int32(0)), batchOffset - lengths)
                self.keys = dynamicRoll(self.keys!, shifts: roll[0..., .newAxis], axis: 2)
                self.values = dynamicRoll(self.values!, shifts: roll[0..., .newAxis], axis: 2)
                leftPadding = leftPadding + roll
                batchOffset = batchOffset - roll
            }

            // Keep `window - 1` old positions, so every new token sees a full window. A ragged
            // prefill keeps a full window: its completed rows were just rolled behind their
            // padding, and they get padding, not tokens, from this chunk. The buffer thus
            // stays within a window and a chunk, whatever the longest row's length.
            let trimSize = idx - window + (lengths == nil ? 1 : 0)
            if trimSize > 0 {
                leftPadding = consumePadding(trimSize)
                self.keys = concatenated([self.keys![.ellipsis, trimSize..., 0...], keys], axis: 2)
                self.values = concatenated(
                    [self.values![.ellipsis, trimSize..., 0...], values], axis: 2)
            } else {
                self.keys = concatenated([self.keys!, keys], axis: 2)
                self.values = concatenated([self.values!, values], axis: 2)
            }
        }

        batchOffset = batchOffset + Int32(keys.dim(2))
        written += keys.dim(2)
        idx = self.keys!.dim(2)
        return (self.keys!, self.values!)
    }

    private func updateInPlace(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        let count = keys.dim(2)
        let previous = written

        // Grow the buffer until it holds the window.
        if self.keys == nil || (previous >= self.keys!.dim(2) && self.keys!.dim(2) < window) {
            let size = min(step, window - previous)
            let newKeys = MLXArray.zeros(
                [keys.dim(0), keys.dim(1), size, keys.dim(3)], dtype: keys.dtype)
            let newValues = MLXArray.zeros(
                [values.dim(0), values.dim(1), size, values.dim(3)], dtype: values.dtype)
            if let currentKeys = self.keys, let currentValues = self.values {
                self.keys = concatenated([currentKeys, newKeys], axis: 2)
                self.values = concatenated([currentValues, newValues], axis: 2)
            } else {
                self.keys = newKeys
                self.values = newValues
            }
            idx = previous
        }

        // A buffer longer than the window drops its oldest positions.
        let trimSize = self.keys!.dim(2) - window
        if trimSize > 0 {
            self.keys = self.keys![.ellipsis, trimSize..., 0...]
            self.values = self.values![.ellipsis, trimSize..., 0...]
            idx = window
            leftPadding = consumePadding(trimSize)
        }

        if idx == window {
            rotated = true
            idx = 0
        }
        if rotated {
            leftPadding = consumePadding(count)
        }

        self.keys![.ellipsis, idx ..< idx + count, 0...] = keys
        self.values![.ellipsis, idx ..< idx + count, 0...] = values
        written += count
        batchOffset = batchOffset + Int32(count)
        idx += count

        if written < window {
            return (
                self.keys![.ellipsis, ..<written, 0...], self.values![.ellipsis, ..<written, 0...]
            )
        }
        return (self.keys!, self.values!)
    }

    /// The padding each row keeps after its `count` oldest positions are overwritten or
    /// dropped: padding goes first, and none is left to count below zero.
    private func consumePadding(_ count: Int) -> MLXArray {
        MLX.maximum(leftPadding - Int32(count), MLXArray(Int32(0)))
    }

    /// Puts a wrapped buffer back in temporal order.
    private func temporalOrder() {
        guard rotated, let keys, let values else { return }
        if idx < keys.dim(2) {
            if idx < written {
                self.keys = MLX.roll(keys, shift: -idx, axis: 2)
                self.values = MLX.roll(values, shift: -idx, axis: 2)
            } else {
                self.keys = keys[.ellipsis, ..<idx, 0...]
                self.values = values[.ellipsis, ..<idx, 0...]
            }
        }
        idx = self.keys!.dim(2)
        rotated = false
    }

    public override var state: [MLXArray] {
        get {
            guard let keys, let values else { return [] }
            if written < keys.dim(2) {
                return [
                    keys[.ellipsis, ..<written, 0...], values[.ellipsis, ..<written, 0...],
                    batchOffset, leftPadding,
                ]
            }
            return [keys, values, batchOffset, leftPadding]
        }
        set {
            guard newValue.count == 4 else {
                fatalError("BatchRotatingKVCache state must have 4 arrays, got \(newValue.count)")
            }
            keys = newValue[0]
            values = newValue[1]
            batchOffset = newValue[2]
            leftPadding = newValue[3]
        }
    }

    /// `[maxSize, written, idx, rotated]`, as in mlx-lm.
    public override var metaState: [String] {
        get { [String(window), String(written), String(idx), String(rotated)] }
        set {
            guard newValue.count == 4, let window = Int(newValue[0]),
                let written = Int(newValue[1]), let idx = Int(newValue[2]),
                let rotated = Bool(newValue[3])
            else {
                fatalError("Invalid BatchRotatingKVCache metaState \(newValue)")
            }
            self.window = window
            self.written = written
            self.idx = idx
            self.rotated = rotated
        }
    }

    public override var isTrimmable: Bool { isTrimmable(after: 0) }

    /// A row that has evicted history cannot rewind exactly: replaying a trimmed token would
    /// see a window without the positions it saw. A merged row can hold less than a window
    /// and still have evicted, so `written` alone does not say.
    public override func isTrimmable(after positions: Int) -> Bool {
        written + positions < window && evictedTokenCount == 0
    }

    /// What ``trim(_:)`` removes, while the window has not filled.
    public override var maxTrimCount: Int { isTrimmable ? trimLimit : 0 }

    /// No row can lose more tokens than the shortest row holds.
    private var trimLimit: Int {
        let shortest = batchSize > 0 ? Int(batchOffset.min().item(Int32.self)) : 0
        return min(written, max(0, shortest))
    }

    /// The most positions any row no longer retains. A row retains the buffer's width
    /// less its own padding.
    public override var evictedTokenCount: Int {
        guard batchSize > 0 else { return 0 }
        let width = rotated ? (keys?.dim(2) ?? 0) : idx
        let evicted = batchOffset + leftPadding - Int32(width)
        return Swift.max(0, Int(evicted.max().item(Int32.self)))
    }

    /// Rewinds the newest `n` positions, at most ``maxTrimCount``: nothing once the window has
    /// filled, because the ring then holds no history to give back.
    @discardableResult
    public override func trim(_ n: Int) -> Int {
        let trimmed = min(maxTrimCount, n)
        written -= trimmed
        idx -= trimmed
        batchOffset = batchOffset - Int32(trimmed)
        return trimmed
    }

    /// Prepares a right-padded prefill of rows with the given lengths.
    public override func prepare(lengths: [Int]?) {
        guard let lengths, let longest = lengths.max(), lengths.contains(where: { $0 < longest })
        else { return }
        self.lengths = MLXArray(lengths.map { Int32($0) }) + batchOffset
    }

    public override func prepare(lengths: MLXArray?) {
        prepare(lengths: lengths?.asArray(Int.self))
    }

    /// Moves the right padding of a prefill to the left, where the mask excludes it.
    public override func finalize() {
        guard let lengths else { return }
        let roll = MLX.maximum(MLXArray(Int32(0)), batchOffset - lengths)
        if let keys, let values {
            self.keys = dynamicRoll(keys, shifts: roll[0..., .newAxis], axis: 2)
            self.values = dynamicRoll(values, shifts: roll[0..., .newAxis], axis: 2)
        }
        leftPadding = leftPadding + roll
        batchOffset = batchOffset - roll
        self.lengths = nil
    }

    /// Keeps only the rows at `batchIndices`, in that order.
    public override func filter(batchIndices: MLXArray) {
        let rows = batchIndices.asArray(Int32.self)
        guard !rows.isEmpty else {
            keys = nil
            values = nil
            lengths = nil
            idx = 0
            written = 0
            rotated = false
            leftPadding = MLXArray([Int32]())
            batchOffset = MLXArray([Int32]())
            return
        }
        let indices = MLXArray(rows)
        keys = keys?[indices]
        values = values?[indices]
        batchOffset = batchOffset[indices]
        leftPadding = leftPadding[indices]
        lengths = lengths?[indices]
    }

    /// Appends the rows of `other`, which must be prefilled or empty like this cache's and
    /// share the window, step and capacity origin.
    public override func extend(other: any KVCache) {
        guard let other = other as? BatchRotatingKVCache else {
            preconditionFailure(
                "BatchRotatingKVCache can only be extended with a BatchRotatingKVCache")
        }
        extend(other: other)
    }

    public override func validateExtension(other: any KVCache) throws {
        guard let other = other as? BatchRotatingKVCache,
            other.window == window, other.step == step, other.capacityOrigin == capacityOrigin
        else {
            throw KVCacheError(
                message: "Rotating batches must share one type, window, step and origin")
        }
        guard lengths == nil, other.lengths == nil else {
            throw KVCacheError(message: "Rotating batch extension requires finalized rows")
        }
        for row in [self, other] {
            guard row.leftPadding.ndim == 1, row.batchOffset.ndim == 1,
                row.batchOffset.size == row.leftPadding.size
            else {
                throw KVCacheError(message: "Rotating batch metadata must agree on the row count")
            }
        }
        guard
            other.batchSize == 0
                || (other.keys != nil || keys == nil)
                    && (keys != nil || batchSize == 0 || other.keys == nil)
        else {
            throw KVCacheError(message: "Rotating batches cannot mix empty and prefilled rows")
        }
        try validateBatchKVExtension(
            keys: keys, values: values, batchSize: batchSize, end: idx,
            otherKeys: other.keys, otherValues: other.values, otherBatchSize: other.batchSize,
            otherEnd: other.idx)
    }

    public func extend(other: BatchRotatingKVCache) {
        // A batch with no rows changes nothing. Otherwise prefilled rows join prefilled rows
        // and empty rows join empty rows; a mix would need all-padding rows inside the window.
        guard other.batchSize > 0 else { return }
        precondition(
            lengths == nil && other.lengths == nil,
            "BatchRotatingKVCache.extend takes finalized rows; finalize a ragged prefill first")
        precondition(
            other.keys != nil || keys == nil,
            "BatchRotatingKVCache.extend cannot add empty rows to prefilled rows")
        precondition(
            keys != nil || batchSize == 0 || other.keys == nil,
            "BatchRotatingKVCache.extend cannot add prefilled rows to empty rows")
        precondition(
            other.window == window && other.step == step
                && other.capacityOrigin == capacityOrigin,
            "BatchRotatingKVCache.extend requires matching maxSize, step and capacityOrigin")
        guard keys != nil, other.keys != nil else {
            if keys == nil, other.keys == nil {
                leftPadding = concatenated([leftPadding, other.leftPadding])
                batchOffset = concatenated([batchOffset, other.batchOffset])
            } else if let otherKeys = other.keys, let otherValues = other.values {
                // New array objects, so writes to either cache leave the other unchanged.
                keys = otherKeys[.ellipsis]
                values = otherValues[.ellipsis]
                idx = other.idx
                written = other.written
                rotated = other.rotated
                leftPadding = other.leftPadding
                batchOffset = other.batchOffset
            }
            return
        }

        if rotated != other.rotated || idx != other.idx {
            temporalOrder()
            other.temporalOrder()
        }
        let end = max(idx, other.idx)
        let width = max(keys!.dim(2), other.keys!.dim(2))
        let own = rightJustify(keys!, values!, end: idx, to: end, width: width)
        let added = rightJustify(other.keys!, other.values!, end: other.idx, to: end, width: width)
        keys = concatenated([own.keys, added.keys])
        values = concatenated([own.values, added.values])
        leftPadding = concatenated([
            leftPadding + Int32(end - idx), other.leftPadding + Int32(end - other.idx),
        ])
        batchOffset = concatenated([batchOffset, other.batchOffset])
        idx = end
        written = max(written, other.written)
    }

    /// The row at `index` in temporal order without its left padding, as a
    /// ``RotatingKVCache`` with this cache's window, step and capacity origin.
    public override func extract(_ index: Int) -> any KVCache {
        let cache = RotatingKVCache(maxSize: window, step: step)
        let padding = Int(leftPadding[index].item(Int32.self))
        let rowOffset = max(0, Int(batchOffset[index].item(Int32.self)))
        var rowIdx = min(rowOffset, window)

        if let keys, let values {
            var rowKeys = keys[index ..< index + 1]
            var rowValues = values[index ..< index + 1]
            if rotated {
                rowKeys = MLX.roll(rowKeys, shift: -idx, axis: 2)
                rowValues = MLX.roll(rowValues, shift: -idx, axis: 2)
            }
            let end = rotated ? window : idx
            rowKeys = MLX.contiguous(rowKeys[.ellipsis, padding ..< end, 0...])
            rowValues = MLX.contiguous(rowValues[.ellipsis, padding ..< end, 0...])
            cache.state = [rowKeys, rowValues]
            // A concatenated buffer can hold more than a window, in temporal order.
            rowIdx = min(rowOffset, rowKeys.dim(2))
        }

        // RotatingKVCache's metadata: keep, maxSize, step, offset, idx, origin, wrapped.
        cache.metaState = [
            "0", String(window), String(step), String(rowOffset), String(rowIdx),
            capacityOrigin.rawValue, "false",
        ]
        return cache
    }

    /// A batch of one row holding the window of `cache`.
    ///
    /// - Throws: `KVCacheError` when `cache` has a pinned prefix or a rewind reserve,
    ///   which a batch cannot hold.
    fileprivate convenience init(row cache: RotatingKVCache) throws {
        guard cache.preservedPrefixTokens == 0, cache.rewindCapacity == 0,
            let window = cache.maxSize
        else {
            throw KVCacheError(
                message:
                    "A RotatingKVCache with a pinned prefix or a rewind reserve cannot be batched"
            )
        }
        self.init(maxSize: window, leftPadding: [0])
        capacityOrigin = cache.capacityOrigin
        step = cache.allocationStep
        if let (keys, values) = cache.logicalView(tail: window) {
            // New array objects, so writes to the batch leave `cache` unchanged.
            self.keys = keys[.ellipsis]
            self.values = values[.ellipsis]
            idx = keys.dim(2)
            written = keys.dim(2)
            batchOffset = MLXArray([Int32(cache.offset)])
        }
    }

    /// The mask for `n` new tokens: the window, the causal order and each row's left
    /// padding, laid out like the keys the matching update returns.
    public override func makeMask(
        n: Int, windowSize: Int?, returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        let windowSize = windowSize ?? window
        // The matching update concatenates: it puts a wrapped buffer in order and then
        // trims it to `window - 1` positions, or to a window during a ragged prefill.
        // Predict that trim.
        let retained = rotated ? (keys?.dim(2) ?? 0) : idx
        let concatenates = n != 1 || lengths != nil
        let trimSize =
            lengths != nil
            ? Swift.max(0, retained - window)
            : (concatenates && rotated ? window : idx) - window + (concatenates ? 1 : 0)
        // A ragged prefill presents every position it keeps; otherwise at most the window.
        let offset = lengths == nil ? min(window - 1, written) : retained - trimSize
        let rinds = MLXArray(Int32(0) ..< Int32(offset + n))[.newAxis]
        let linds = MLXArray(Int32(offset) ..< Int32(offset + n))[0..., .newAxis]
        var mask = (linds .>= rinds) & (linds .< rinds + Int32(windowSize))

        var padding = leftPadding
        if trimSize > 0 {
            padding = consumePadding(trimSize)
        }

        // The matching update writes in place and wraps, over the oldest position.
        let wraps = !concatenates && (rotated || idx >= window)
        if wraps {
            padding = MLX.maximum(padding - 1, MLXArray(Int32(0)))
        }
        mask = mask & (rinds .>= padding[0..., .newAxis, .newAxis, .newAxis])
        if wraps {
            let position = idx >= window ? 0 : idx
            mask = MLX.roll(mask, shift: position + 1, axis: -1)
        }
        return .array(mask)
    }

    public override func copy() -> any KVCache {
        let copy = BatchRotatingKVCache(maxSize: window, leftPadding: [])
        copy.leftPadding = leftPadding[0...]
        copy.batchOffset = batchOffset[0...]
        copy.keys = keys?[.ellipsis]
        copy.values = values?[.ellipsis]
        copy.idx = idx
        copy.written = written
        copy.rotated = rotated
        copy.lengths = lengths?[0...]
        copy.step = step
        copy.capacityOrigin = capacityOrigin
        return copy
    }
}

extension RotatingKVCache {
    /// The rows of `caches` as one batch, each keeping its chronological window and
    /// absolute position. Port of mlx-lm's `RotatingKVCache.merge`.
    ///
    /// - Throws: `KVCacheError` when a cache keeps a prefix or a rewind reserve,
    ///   which a batch cannot hold, or when the rows mix empty and prefilled caches.
    public static func merge(_ caches: [RotatingKVCache]) throws -> BatchRotatingKVCache {
        precondition(!caches.isEmpty, "merge needs a cache")
        let first = caches[0]
        guard
            caches.allSatisfy({
                $0.maxSize == first.maxSize && $0.allocationStep == first.allocationStep
                    && $0.capacityOrigin == first.capacityOrigin
            })
        else {
            throw KVCacheError(
                message: "RotatingKVCache.merge needs rows that share one window, step and origin"
            )
        }
        // Rows join by `extend`, which takes all-empty or all-prefilled rows; a
        // mix would need an all-padding row inside the window.
        let prefilled = caches.filter { $0.offset > 0 }.count
        guard prefilled == 0 || prefilled == caches.count else {
            throw KVCacheError(
                message: "RotatingKVCache.merge needs rows that are all empty or all prefilled")
        }
        let rows = try caches.map { try BatchRotatingKVCache(row: $0) }
        guard prefilled > 0 else {
            // Empty rows have nothing to join; size the batch directly.
            let batch = BatchRotatingKVCache(
                maxSize: rows[0].maxSize ?? 0, leftPadding: Array(repeating: 0, count: rows.count))
            batch.step = rows[0].step
            batch.capacityOrigin = rows[0].capacityOrigin
            return batch
        }
        let batch = rows[0]
        for row in rows.dropFirst() {
            batch.extend(other: row)
        }
        return batch
    }
}
