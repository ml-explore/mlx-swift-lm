// Copyright © 2026 Apple Inc.

import Foundation
import MLX

/// A KV cache for a batch of left-padded sequences.
///
/// Port of mlx-lm's `BatchKVCache`, with the fixes mlx-vlm's port of it carries
/// (`mlx_vlm/models/cache.py` at mlx-vlm dcea787, noted where they apply). Rows are
/// left-padded so that sequences of
/// different lengths end at the same position. Prompts `[1, 3, 5]`, `[7]` and
/// `[2, 6, 8, 9]` are stored as
/// ```
/// [0, 1, 3, 5]
/// [0, 0, 0, 7]
/// [2, 6, 8, 9]
/// ```
/// with ``leftPadding`` `[1, 3, 0]`. ``KVCache/offset`` is the width of the buffer in use.
///
/// Row positions and padding live on the host. The arrays the model reads are
/// built from them on demand, so a decode step advances positions without a
/// kernel per layer, and a batch whose rows carry no padding decodes one token
/// with no mask at all.
public final class BatchKVCache: BaseKVCache, BatchPositionedKVCache {
    /// Each row's left padding, shape `[B]`.
    public private(set) var leftPadding: MLXArray

    /// Each row's left padding on the host; `leftPadding` mirrors it.
    private var leftPaddingValues: [Int32]

    /// Each row's position on the host. It starts at `-leftPadding`.
    private var positions: [Int32]

    /// `positions` as an array, built on first use after every change.
    private var positionsArray: MLXArray?

    /// Each row's position, shape `[B]`. It starts at `-leftPadding`.
    public var batchOffset: MLXArray {
        if let positionsArray { return positionsArray }
        let array = MLXArray(positions)
        positionsArray = array
        return array
    }

    var keys: MLXArray?
    var values: MLXArray?

    /// The right padding of a ragged prefill, from `prepare` until `finalize`.
    private var rightPadding: MLXArray?
    private var rightPaddingValues: [Int32]?

    public var step = 256

    /// - Parameter leftPadding: the left padding of each row
    public init(leftPadding: [Int]) {
        let padding = leftPadding.map { Int32($0) }
        self.leftPaddingValues = padding
        self.leftPadding = MLXArray(padding)
        self.positions = padding.map { -$0 }
        super.init()
    }

    /// The number of rows.
    public var batchSize: Int { positions.count }

    /// Whether any row carries left padding.
    private var hasLeftPadding: Bool { leftPaddingValues.contains { $0 != 0 } }

    private func setPositions(_ new: [Int32]) {
        positions = new
        positionsArray = nil
    }

    private func setLeftPadding(_ new: [Int32]) {
        leftPaddingValues = new
        leftPadding = MLXArray(new)
    }

    private func setRightPadding(_ new: [Int32]?) {
        rightPaddingValues = new
        rightPadding = new.map { MLXArray($0) }
    }

    /// Per-row positions, also when the cache is typed as `KVCache`.
    ///
    /// Rows at one shared position, such as a batch of one, get the scalar
    /// form, so no array is built per layer per step. Otherwise no copy:
    /// every change replaces the array rather than mutating it.
    public override var ropeOffset: RoPEOffset {
        if let first = positions.first, positions.allSatisfy({ $0 == first }) {
            return .scalar(Int(first))
        }
        return .batch(batchOffset)
    }

    public override func innerState() -> [MLXArray] {
        [keys, values].compactMap { $0 }
    }

    public override func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        let previous = offset
        if self.keys == nil || previous + keys.dim(2) > self.keys!.dim(2) {
            let length = (step + keys.dim(2) - 1) / step * step
            let newKeys = MLXArray.zeros(
                [keys.dim(0), keys.dim(1), length, keys.dim(3)], dtype: keys.dtype)
            let newValues = MLXArray.zeros(
                [values.dim(0), values.dim(1), length, values.dim(3)], dtype: values.dtype)
            if var currentKeys = self.keys, var currentValues = self.values {
                if previous % step != 0 {
                    currentKeys = currentKeys[.ellipsis, ..<previous, 0...]
                    currentValues = currentValues[.ellipsis, ..<previous, 0...]
                }
                self.keys = concatenated([currentKeys, newKeys], axis: 2)
                self.values = concatenated([currentValues, newValues], axis: 2)
            } else {
                self.keys = newKeys
                self.values = newValues
            }
        }

        let n = Int32(keys.dim(2))
        offset += keys.dim(2)
        setPositions(positions.map { $0 + n })
        self.keys![.ellipsis, previous ..< offset, 0...] = keys
        self.values![.ellipsis, previous ..< offset, 0...] = values
        return (self.keys![.ellipsis, ..<offset, 0...], self.values![.ellipsis, ..<offset, 0...])
    }

    public override var state: [MLXArray] {
        get {
            guard let keys, let values else { return [batchOffset, leftPadding] }
            return [
                keys[.ellipsis, ..<offset, 0...], values[.ellipsis, ..<offset, 0...],
                batchOffset, leftPadding,
            ]
        }
        set {
            switch newValue.count {
            case 2:
                keys = nil
                values = nil
                offset = 0
                setPositions(newValue[0].asArray(Int32.self))
                setLeftPadding(newValue[1].asArray(Int32.self))
            case 4:
                keys = newValue[0]
                values = newValue[1]
                offset = newValue[0].dim(2)
                setPositions(newValue[2].asArray(Int32.self))
                setLeftPadding(newValue[3].asArray(Int32.self))
            default:
                fatalError("BatchKVCache state must have 2 or 4 arrays, got \(newValue.count)")
            }
        }
    }

    public override var isTrimmable: Bool { true }

    public override func isTrimmable(after positions: Int) -> Bool {
        // A round can discard its whole write only after every row has consumed its padding.
        positions >= 0
            && (positions == 0
                || (!self.positions.isEmpty && self.positions.allSatisfy { $0 >= 0 }))
    }

    /// No row can lose more tokens than the shortest row holds.
    public override var maxTrimCount: Int {
        let shortest = Int(positions.min() ?? 0)
        return min(offset, max(0, shortest))
    }

    @discardableResult
    public override func trim(_ n: Int) -> Int {
        let trimmed = min(maxTrimCount, n)
        offset -= trimmed
        setPositions(positions.map { $0 - Int32(trimmed) })
        return trimmed
    }

    /// No mask for one token when no row is padded and the window, if any,
    /// covers every retained key; otherwise an explicit mask, so left padding
    /// and the window stay masked. This cache never evicts, so a window
    /// smaller than the history still needs the mask.
    public override func makeMask(
        n: Int, windowSize: Int?, returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        if n == 1, !hasLeftPadding, windowSize.map({ $0 >= offset + n }) ?? true {
            return .none
        }
        return .array(
            createCausalMask(
                n: n, offset: offset, windowSize: windowSize, leftPadding: leftPadding))
    }

    /// Prepares a right-padded prefill of rows with the given lengths.
    public override func prepare(lengths: [Int]?) {
        guard let lengths, let longest = lengths.max(), lengths.contains(where: { $0 < longest })
        else { return }
        setRightPadding(lengths.map { Int32(longest - $0) })
    }

    public override func prepare(lengths: MLXArray?) {
        prepare(lengths: lengths?.asArray(Int.self))
    }

    /// Moves the right padding of a prefill to the left, where the mask excludes it.
    public override func finalize() {
        guard let rightPadding, let rightPaddingValues else { return }
        // A circular roll of the whole buffer lands each row's tokens at
        // `padding ..< offset`; what wraps around is masked or unused.
        if let keys, let values {
            self.keys = dynamicRoll(keys, shifts: rightPadding[0..., .newAxis], axis: 2)
            self.values = dynamicRoll(values, shifts: rightPadding[0..., .newAxis], axis: 2)
        }
        setPositions(zip(positions, rightPaddingValues).map { $0 - $1 })
        setLeftPadding(zip(leftPaddingValues, rightPaddingValues).map { $0 + $1 })
        setRightPadding(nil)
    }

    /// Keeps only the rows at `batchIndices`, in that order.
    public override func filter(batchIndices: MLXArray) {
        let rows = batchIndices.asArray(Int32.self).map { Int($0) }
        guard !rows.isEmpty else {
            keys = nil
            values = nil
            setRightPadding(nil)
            offset = 0
            setLeftPadding([])
            setPositions([])
            return
        }
        let indices = MLXArray(rows.map { Int32($0) })
        keys = keys?[indices]
        values = values?[indices]
        setPositions(rows.map { positions[$0] })
        var padding = rows.map { leftPaddingValues[$0] }
        // A ragged prefill's right padding goes with its rows, as in mlx-vlm:
        // https://github.com/Blaizzy/mlx-vlm/blob/dcea787/mlx_vlm/models/cache.py#L1673-L1674
        setRightPadding(rightPaddingValues.map { values in rows.map { values[$0] } })

        // Drop the left padding that every remaining row has, but no more than the buffer
        // holds: a chunked prefill can be filtered before all of its padding is written.
        // mlx-vlm bounds `min_left_pad` by `_idx` the same way:
        // https://github.com/Blaizzy/mlx-vlm/blob/dcea787/mlx_vlm/models/cache.py#L1677
        let shared = min(padding.min() ?? 0, Int32(offset))
        if shared > 0, keys != nil {
            keys = keys?[.ellipsis, Int(shared)..., 0...]
            values = values?[.ellipsis, Int(shared)..., 0...]
            offset -= Int(shared)
            padding = padding.map { $0 - shared }
        }
        setLeftPadding(padding)
    }

    public override func extend(other: any KVCache) {
        guard let other = other as? BatchKVCache else {
            preconditionFailure("BatchKVCache can only be extended with a BatchKVCache")
        }
        extend(other: other)
    }

    public override func validateExtension(other: any KVCache) throws {
        guard let other = other as? BatchKVCache else {
            throw KVCacheError(message: "BatchKVCache can only append a BatchKVCache")
        }
        guard rightPadding == nil, other.rightPadding == nil else {
            throw KVCacheError(message: "BatchKVCache extension requires finalized rows")
        }
        for row in [self, other] {
            guard row.leftPaddingValues.count == row.batchSize,
                row.leftPadding.ndim == 1, row.leftPadding.size == row.batchSize,
                zip(row.positions, row.leftPaddingValues).allSatisfy({
                    Int($0) + Int($1) == row.offset
                })
            else {
                throw KVCacheError(message: "BatchKVCache metadata must agree on the row count")
            }
        }
        try validateBatchKVExtension(
            keys: keys, values: values, batchSize: batchSize, end: offset,
            otherKeys: other.keys, otherValues: other.values, otherBatchSize: other.batchSize,
            otherEnd: other.offset)
    }

    /// Appends the rows of `other`. Port of mlx-lm's `BatchKVCache.extend`: rows without keys
    /// yet join as all-padding rows, so empty and prefilled rows can share a batch, and a
    /// batch with no rows changes nothing. Both batches must be finalized: a ragged prefill in
    /// progress cannot take or give rows.
    public func extend(other: BatchKVCache) {
        guard other.batchSize > 0 else { return }
        // Right padding of a prefill in progress is not combined: finalize first.
        precondition(
            rightPadding == nil && other.rightPadding == nil,
            "BatchKVCache.extend takes finalized rows; finalize a ragged prefill first")
        guard batchSize > 0 else {
            // New array objects, so writes to either cache leave the other unchanged.
            keys = other.keys?[.ellipsis]
            values = other.values?[.ellipsis]
            offset = other.offset
            setLeftPadding(other.leftPaddingValues)
            setPositions(other.positions)
            setRightPadding(other.rightPaddingValues)
            return
        }
        guard let templateKeys = keys ?? other.keys, let templateValues = values ?? other.values
        else {
            setLeftPadding(leftPaddingValues + other.leftPaddingValues)
            setPositions(positions + other.positions)
            return
        }
        // Rows without keys take zero-width arrays, which the padding below fills.
        func arrays(_ cache: BatchKVCache) -> (MLXArray, MLXArray) {
            if let keys = cache.keys, let values = cache.values { return (keys, values) }
            return (
                MLXArray.zeros(
                    [cache.batchSize, templateKeys.dim(1), 0, templateKeys.dim(3)],
                    dtype: templateKeys.dtype),
                MLXArray.zeros(
                    [cache.batchSize, templateValues.dim(1), 0, templateValues.dim(3)],
                    dtype: templateValues.dtype)
            )
        }
        let (ownKeys, ownValues) = arrays(self)
        let (otherKeys, otherValues) = arrays(other)

        let end = max(offset, other.offset)
        let width = max(ownKeys.dim(2), otherKeys.dim(2))
        let own = rightJustify(ownKeys, ownValues, end: offset, to: end, width: width)
        let added = rightJustify(otherKeys, otherValues, end: other.offset, to: end, width: width)
        keys = concatenated([own.keys, added.keys])
        values = concatenated([own.values, added.values])
        let ownShift = Int32(end - offset)
        let addedShift = Int32(end - other.offset)
        setLeftPadding(
            leftPaddingValues.map { $0 + ownShift }
                + other.leftPaddingValues.map { $0 + addedShift })
        setPositions(positions + other.positions)
        offset = end
    }

    /// The row at `index` without its left padding, as a ``KVCacheSimple``.
    public override func extract(_ index: Int) -> any KVCache {
        // mlx-vlm checks the row index first too:
        // https://github.com/Blaizzy/mlx-vlm/blob/dcea787/mlx_vlm/models/cache.py#L1730-L1736
        precondition(
            leftPaddingValues.indices.contains(index), "BatchKVCache row index out of range")
        let cache = KVCacheSimple()
        let padding = Int(leftPaddingValues[index])
        if let keys, let values {
            cache.keys = MLX.contiguous(keys[index ..< index + 1, 0..., padding ..< offset, 0...])
            cache.values = MLX.contiguous(
                values[index ..< index + 1, 0..., padding ..< offset, 0...])
            cache.offset = offset - padding
        }
        return cache
    }

    /// The rows of `caches` as one batch, left-padded to the longest row.
    fileprivate convenience init(merging caches: [KVCacheSimple]) {
        let longest = caches.map(\.offset).max() ?? 0
        self.init(leftPadding: caches.map { longest - $0.offset })
        // An all-empty merge is an empty batch; the first update sizes it. mlx-vlm returns
        // an empty batch for it too:
        // https://github.com/Blaizzy/mlx-vlm/blob/dcea787/mlx_vlm/models/cache.py#L1750-L1751
        guard let template = caches.first(where: { $0.keys != nil }),
            let templateKeys = template.keys, let templateValues = template.values
        else { return }
        var keys: [MLXArray] = []
        var values: [MLXArray] = []
        for cache in caches {
            if let rowKeys = cache.keys, let rowValues = cache.values, cache.offset > 0 {
                // Slice off allocation slack, then pad on the left to `longest`.
                let row = rightJustify(
                    rowKeys[.ellipsis, ..<cache.offset, 0...],
                    rowValues[.ellipsis, ..<cache.offset, 0...],
                    end: cache.offset, to: longest, width: longest)
                keys.append(row.keys)
                values.append(row.values)
            } else {
                keys.append(
                    MLXArray.zeros(
                        [1, templateKeys.dim(1), longest, templateKeys.dim(3)],
                        dtype: templateKeys.dtype))
                values.append(
                    MLXArray.zeros(
                        [1, templateValues.dim(1), longest, templateValues.dim(3)],
                        dtype: templateValues.dtype))
            }
        }
        self.keys = concatenated(keys)
        self.values = concatenated(values)
        offset = longest
        setPositions(caches.map { Int32($0.offset) })
    }

    public override func copy() -> any KVCache {
        let copy = BatchKVCache(leftPadding: [])
        copy.setLeftPadding(leftPaddingValues)
        copy.setPositions(positions)
        copy.keys = keys?[.ellipsis]
        copy.values = values?[.ellipsis]
        copy.setRightPadding(rightPaddingValues)
        copy.offset = offset
        copy.step = step
        return copy
    }
}

extension KVCacheSimple {
    /// The rows of `caches` as one batch, left-padded so every row ends at the
    /// longest row's position. Port of mlx-lm's `KVCache.merge`. Every cache
    /// must be exactly a `KVCacheSimple`; empty caches become empty rows.
    public static func merge(_ caches: [KVCacheSimple]) -> BatchKVCache {
        precondition(!caches.isEmpty, "merge needs a cache")
        // Exact type: a ChunkedKVCache can report an offset beyond the arrays it holds.
        precondition(
            caches.allSatisfy { type(of: $0) == KVCacheSimple.self },
            "KVCacheSimple.merge takes exactly KVCacheSimple rows")
        return BatchKVCache(merging: caches)
    }
}

/// `keys` and `values`, whose rows end at position `end`, padded to end at `newEnd`
/// in a buffer `width` wide.
func rightJustify(
    _ keys: MLXArray, _ values: MLXArray, end: Int, to newEnd: Int, width: Int
) -> (keys: MLXArray, values: MLXArray) {
    var keys = keys
    var values = values
    let left = newEnd - end
    var right = width - keys.dim(2) - left
    if right < 0 {
        keys = keys[.ellipsis, ..<(keys.dim(2) + right), 0...]
        values = values[.ellipsis, ..<(values.dim(2) + right), 0...]
        right = 0
    }
    if left != 0 || right != 0 {
        let widths: [IntOrPair] = [0, 0, .init((left, right)), 0]
        keys = MLX.padded(keys, widths: widths)
        values = MLX.padded(values, widths: widths)
    }
    return (keys, values)
}

/// Rolls each row of `x` along `axis` by its own shift. Port of mlx-lm's `dynamic_roll`.
func dynamicRoll(_ x: MLXArray, shifts: MLXArray, axis: Int) -> MLXArray {
    let axis = axis >= 0 ? axis : x.ndim + axis
    let n = x.dim(axis)
    var positionShape = [Int](repeating: 1, count: x.ndim)
    positionShape[axis] = n
    var shiftShape = [Int](repeating: 1, count: x.ndim)
    for dimension in 0 ..< min(shifts.ndim, x.ndim) {
        shiftShape[dimension] = shifts.dim(dimension)
    }
    let positions = MLXArray(Int32(0) ..< Int32(n)).reshaped(positionShape)
    let size = MLXArray(Int32(n))
    // Adding `n` keeps the index non-negative for negative shifts.
    let indices = ((positions - shifts.reshaped(shiftShape)) % size + size) % size
    return takeAlong(x, indices.asType(.int32), axis: axis)
}
