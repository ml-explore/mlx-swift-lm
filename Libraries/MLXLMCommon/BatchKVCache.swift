// Port of mlx_lm.models.cache.BatchKVCache.
// https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/cache.py

import Foundation
import MLX

/// A `KVCache` that supports continuous-batching primitives (in-place row
/// filtering and concatenation). Both `BatchKVCache` (for full-attention
/// layers) and `ArraysCache` (for SSM-style layers like Qwen 3.5's
/// GatedDeltaNet) conform.
public protocol BatchedCache: KVCache {
    /// In-place keep only the rows at the given batch indices.
    func filterBatched(batchIndices: MLXArray)

    /// In-place append `other`'s rows. The runtime types must match.
    func extendBatched(_ other: any BatchedCache)

    /// Prepare cache metadata before ragged prompt prefill.
    func prepareBatched(leftPadding: [Int]?, lengths: [Int]?, rightPadding: [Int]?)

    /// Finalize cache metadata after ragged prompt prefill.
    func finalizeBatched()

    /// Extract one row as its corresponding single-request cache.
    func extractBatched(_ idx: Int) -> any KVCache

    /// Advance chunk-local metadata after a chunked prefill step.
    func advanceBatched(_ n: Int)
}

/// Continuous-batching KV cache.
///
/// Storage is right-justified along axis=2: for each row `b`, real keys
/// live at `[..., leftPadding[b]..._idx, :]` and the leading `leftPadding[b]`
/// slots are zero. Per-row position offsets are exposed via `batchOffset`
/// for RoPE dispatch through `BatchPositionedKVCache`.
///
/// Not thread-safe; the `BatchGenerator` mutates this from a single task.
public final class BatchKVCache: BaseKVCache, BatchPositionedKVCache, BatchedCache {

    public func filterBatched(batchIndices: MLXArray) {
        filter(batchIndices: batchIndices)
    }

    public func extendBatched(_ other: any BatchedCache) {
        guard let other = other as? BatchKVCache else {
            preconditionFailure("BatchKVCache.extendBatched requires another BatchKVCache")
        }
        extend(other)
    }

    public func prepareBatched(leftPadding: [Int]?, lengths: [Int]?, rightPadding: [Int]?) {
        prepare(leftPadding: leftPadding, lengths: lengths, rightPadding: rightPadding)
    }

    public func finalizeBatched() {
        finalize()
    }

    public func extractBatched(_ idx: Int) -> any KVCache {
        extract(idx)
    }

    public func advanceBatched(_: Int) {}

    /// Allocation chunk size along the time axis.
    public static let allocationStep = 256

    /// `[B, kvHeads, T, headDim]`, nil until the first `update`.
    public private(set) var keys: MLXArray?

    /// `[B, kvHeads, T, headValueDim]`, nil until the first `update`.
    public private(set) var values: MLXArray?

    /// Per-row position counter `[B]`. Starts at `-leftPadding[b]`; advances
    /// by `keys.dim(2)` per `update`. Read by RoPE via `BatchPositionedKVCache`.
    public private(set) var batchOffset: MLXArray

    /// Per-row left padding `[B]`. Slots `[..., 0..<leftPadding[b], :]` are
    /// zero and the mask blocks them.
    public private(set) var leftPadding: MLXArray

    /// Rightmost-valid slot. Shared scalar across rows because they're kept
    /// right-aligned. Slots past `_idx` are pre-allocated capacity.
    private var _idx: Int = 0

    /// Right-padding applied at `finalize()`. Set when chunked prefill needs
    /// to roll rows shorter than the prefill window into right-aligned
    /// position.
    private var _rightPadding: MLXArray?

    /// Scalar offset for the legacy `KVCache` API. Returns `_idx` (the
    /// rightmost trailing edge); only `makeMask` consumes it on this path,
    /// since `applyRotaryPosition` dispatches to `batchOffset` instead.
    public override var offset: Int {
        get { _idx }
        set { _idx = newValue }
    }

    public override var maxSize: Int? { nil }
    public override var isTrimmable: Bool { true }

    // MARK: - Init

    /// Construct an empty cache for a batch of `leftPadding.count` rows.
    ///
    /// The cache expects inputs to be left-padded. For these prompts:
    /// ```
    /// [1, 3, 5]
    /// [7]
    /// [2, 6, 8, 9]
    /// ```
    /// the effective batched input is right-aligned to:
    /// ```
    /// [0, 1, 3, 5]
    /// [0, 0, 0, 7]
    /// [2, 6, 8, 9]
    /// ```
    /// and `leftPadding = [1, 3, 0]`.
    public init(leftPadding: [Int]) {
        self.leftPadding = MLXArray(leftPadding.map { Int32($0) })
        self.batchOffset = MLXArray(leftPadding.map { Int32(-$0) })
        super.init()
    }

    private init(
        keys: MLXArray?,
        values: MLXArray?,
        offset: MLXArray,
        leftPadding: MLXArray,
        idx: Int
    ) {
        self.keys = keys
        self.values = values
        self.batchOffset = offset
        self.leftPadding = leftPadding
        super.init()
        self._idx = idx
    }

    /// Append `[B, kvHeads, T, D]` keys/values and return the full populated
    /// keys/values (`[B, kvHeads, _idx, D]`). Storage grows in
    /// `allocationStep` chunks when capacity is exceeded.
    public override func update(
        keys: MLXArray, values: MLXArray
    ) -> (MLXArray, MLXArray) {
        let prev = _idx
        let stepCount = keys.dim(2)
        let needGrow: Bool = {
            guard let storedKeys = self.keys else { return true }
            return (prev + stepCount) > storedKeys.dim(2)
        }()

        if needGrow {
            let B = keys.dim(0)
            let nKVHeads = keys.dim(1)
            let kHeadDim = keys.dim(3)
            let vHeadDim = values.dim(3)
            let nSteps = (Self.allocationStep + stepCount - 1) / Self.allocationStep
            let kShape = [B, nKVHeads, nSteps * Self.allocationStep, kHeadDim]
            let vShape = [B, nKVHeads, nSteps * Self.allocationStep, vHeadDim]
            let newK = MLXArray.zeros(kShape, dtype: keys.dtype)
            let newV = MLXArray.zeros(vShape, dtype: values.dtype)

            if var currentK = self.keys, var currentV = self.values {
                if prev % Self.allocationStep != 0 {
                    currentK = currentK[.ellipsis, ..<prev, 0...]
                    currentV = currentV[.ellipsis, ..<prev, 0...]
                }
                self.keys = concatenated([currentK, newK], axis: 2)
                self.values = concatenated([currentV, newV], axis: 2)
            } else {
                self.keys = newK
                self.values = newV
            }
        }

        batchOffset = batchOffset + Int32(stepCount)
        _idx += stepCount
        self.keys?[.ellipsis, prev ..< _idx, 0...] = keys
        self.values?[.ellipsis, prev ..< _idx, 0...] = values

        return (
            self.keys![.ellipsis, ..<_idx, 0...],
            self.values![.ellipsis, ..<_idx, 0...]
        )
    }

    // MARK: - prefill helpers

    /// Prepare for chunked prefill. `leftPadding` may only be added to an
    /// empty cache. `rightPadding` is recorded and applied by `finalize()`
    /// after the prompt forward pass completes.
    public func prepare(
        leftPadding additionalLeftPadding: [Int]? = nil,
        lengths _: [Int]? = nil,
        rightPadding: [Int]? = nil
    ) {
        if let additionalLeftPadding {
            precondition(
                keys == nil,
                "prepare() with leftPadding can only be called on an empty BatchKVCache"
            )
            let additional = MLXArray(additionalLeftPadding.map { Int32($0) })
            leftPadding = leftPadding + additional
            batchOffset = batchOffset - additional
        }

        if let rightPadding, rightPadding.contains(where: { $0 > 0 }) {
            _rightPadding = MLXArray(rightPadding.map { Int32($0) })
        }
    }

    /// Roll each row right by its pending right-padding so all rows are
    /// right-justified along the time axis.
    public func finalize() {
        guard let pending = _rightPadding else { return }
        guard let storedK = keys, let storedV = values else {
            _rightPadding = nil
            return
        }

        let shifts = pending[0..., .newAxis]
        keys = dynamicRoll(storedK, shifts: shifts, axis: 2)
        values = dynamicRoll(storedV, shifts: shifts, axis: 2)
        batchOffset = batchOffset - pending
        leftPadding = leftPadding + pending
        _rightPadding = nil
    }

    /// In-place: keep only rows at `batchIndices` and shave any common
    /// left-padding from the front of the storage.
    public func filter(batchIndices: MLXArray) {
        if keys != nil {
            keys = take(keys!, batchIndices, axis: 0)
            values = take(values!, batchIndices, axis: 0)
        }
        batchOffset = take(batchOffset, batchIndices, axis: 0)
        leftPadding = take(leftPadding, batchIndices, axis: 0)

        let minLeftPad = leftPadding.min().item(Int32.self)
        if minLeftPad > 0 {
            let minPadInt = Int(minLeftPad)
            if let storedK = keys, let storedV = values {
                keys = storedK[.ellipsis, minPadInt..., 0...]
                values = storedV[.ellipsis, minPadInt..., 0...]
            }
            _idx -= minPadInt
            leftPadding = leftPadding - Int32(minPadInt)
        }
    }

    // MARK: - extend (in-place admission)

    /// In-place concatenation of another batched cache's rows onto this one.
    /// Both caches are padded to be right-justified and same time-axis size,
    /// then concatenated along the batch axis.
    public func extend(_ other: BatchKVCache) {
        // Both empty: just concat the metadata.
        if keys == nil && other.keys == nil {
            leftPadding = concatenated([leftPadding, other.leftPadding], axis: 0)
            batchOffset = concatenated([batchOffset, other.batchOffset], axis: 0)
            return
        }

        let maxIdx = max(_idx, other._idx)
        var L1 = 0
        var L2 = 0
        var H = 0
        var Dk = 0
        var Dv = 0

        if let storedK = keys {
            L1 = storedK.dim(2)
            H = storedK.dim(1)
            Dk = storedK.dim(3)
            Dv = values!.dim(3)
        }
        if let storedK = other.keys {
            L2 = storedK.dim(2)
            H = storedK.dim(1)
            Dk = storedK.dim(3)
            Dv = other.values!.dim(3)
        }
        let maxSize = max(L1, L2)

        // Pad each cache so its keys/values share the same shape and are
        // right-justified with the rightmost index `maxIdx`.
        func pad(
            _ cacheKeys: MLXArray?,
            _ cacheValues: MLXArray?,
            cacheIdx: Int,
            cacheOffset: MLXArray,
            cacheLeftPadding: MLXArray
        ) -> (MLXArray, MLXArray, MLXArray, MLXArray) {
            var k: MLXArray
            var v: MLXArray
            if let cacheKeys, let cacheValues {
                k = cacheKeys
                v = cacheValues
            } else {
                let Bc = cacheOffset.dim(0)
                k = MLXArray.zeros([Bc, H, 0, Dk], dtype: keys?.dtype ?? .float32)
                v = MLXArray.zeros([Bc, H, 0, Dv], dtype: values?.dtype ?? .float32)
            }

            let leftPad = maxIdx - cacheIdx
            var rightPad = maxSize - k.dim(2) - leftPad
            if rightPad < 0 {
                k = k[.ellipsis, ..<(k.dim(2) + rightPad), 0...]
                v = v[.ellipsis, ..<(v.dim(2) + rightPad), 0...]
                rightPad = 0
            }
            if leftPad != 0 || rightPad != 0 {
                let widths: [IntOrPair] = [
                    .init(0), .init(0), .init((leftPad, rightPad)), .init(0),
                ]
                k = padded(k, widths: widths)
                v = padded(v, widths: widths)
            }
            return (k, v, cacheOffset, cacheLeftPadding + Int32(leftPad))
        }

        let (k1, v1, o1, lp1) = pad(
            keys, values, cacheIdx: _idx,
            cacheOffset: batchOffset, cacheLeftPadding: leftPadding
        )
        let (k2, v2, o2, lp2) = pad(
            other.keys, other.values, cacheIdx: other._idx,
            cacheOffset: other.batchOffset, cacheLeftPadding: other.leftPadding
        )

        keys = concatenated([k1, k2], axis: 0)
        values = concatenated([v1, v2], axis: 0)
        batchOffset = concatenated([o1, o2], axis: 0)
        leftPadding = concatenated([lp1, lp2], axis: 0)
        _idx = maxIdx
    }

    /// Slice row `idx` out as a standalone single-row `KVCacheSimple`. The
    /// slice is materialized so it owns its storage and survives subsequent
    /// `filter` / `extend` mutations on this batched cache.
    public func extract(_ idx: Int) -> KVCacheSimple {
        let cache = KVCacheSimple()
        guard let storedK = keys, let storedV = values else {
            return cache
        }
        let leftPad = Int(leftPadding[idx].item(Int32.self))
        let kSlice = storedK[idx ..< (idx + 1), 0..., leftPad ..< _idx, 0...]
        let vSlice = storedV[idx ..< (idx + 1), 0..., leftPad ..< _idx, 0...]
        eval(kSlice, vSlice)
        cache.state = [kSlice, vSlice]
        return cache
    }

    /// Build a `BatchKVCache` from a list of single-row caches by padding
    /// each to the longest length and right-justifying.
    public static func merge(_ caches: [KVCacheSimple]) -> BatchKVCache {
        let lengths = caches.map { $0.offset }
        let maxLength = lengths.max() ?? 0

        if maxLength == 0 {
            return BatchKVCache(leftPadding: Array(repeating: 0, count: caches.count))
        }

        let leftPaddings = lengths.map { maxLength - $0 }
        let B = caches.count

        guard let template = caches.first(where: { $0.state.count >= 2 }) else {
            return BatchKVCache(leftPadding: leftPaddings)
        }
        let templateK = template.state[0]
        let templateV = template.state[1]
        let H = templateK.dim(1)
        let Dk = templateK.dim(3)
        let Dv = templateV.dim(3)
        let dtype = templateK.dtype

        let keys = MLXArray.zeros([B, H, maxLength, Dk], dtype: dtype)
        let values = MLXArray.zeros([B, H, maxLength, Dv], dtype: dtype)

        for (i, (pad, cache)) in zip(leftPaddings, caches).enumerated() {
            guard cache.state.count >= 2 else { continue }
            let k = cache.state[0]
            let v = cache.state[1]
            let len = cache.offset
            keys[i ..< (i + 1), 0..., pad ..< (pad + len), 0...] = k[.ellipsis, ..<len, 0...]
            values[i ..< (i + 1), 0..., pad ..< (pad + len), 0...] = v[.ellipsis, ..<len, 0...]
        }

        let result = BatchKVCache(leftPadding: leftPaddings)
        result.keys = keys
        result.values = values
        result.batchOffset = result.batchOffset + Int32(maxLength)
        result._idx = maxLength
        return result
    }

    /// Causal mask that also blocks each row's own left-padded slots.
    /// Always materialized to an array because per-row left-padding can't
    /// be expressed via the symbolic `.causal` mode.
    public override func makeMask(
        n: Int, windowSize: Int?, returnArray _: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        let mask = createCausalMask(
            n: n,
            offset: _idx,
            windowSize: windowSize,
            leftPadding: leftPadding
        )
        return .array(mask)
    }

    @discardableResult
    public override func trim(_ n: Int) -> Int {
        let trimmed = min(_idx, n)
        _idx -= trimmed
        batchOffset = batchOffset - Int32(trimmed)
        return trimmed
    }

    public func size() -> Int { _idx }
    public func isEmpty() -> Bool { keys == nil }

    public override var state: [MLXArray] {
        get {
            guard let storedK = keys, let storedV = values else {
                return [batchOffset, leftPadding]
            }
            let kClipped: MLXArray
            let vClipped: MLXArray
            if _idx < storedK.dim(2) {
                kClipped = storedK[.ellipsis, ..<_idx, 0...]
                vClipped = storedV[.ellipsis, ..<_idx, 0...]
            } else {
                kClipped = storedK
                vClipped = storedV
            }
            return [kClipped, vClipped, batchOffset, leftPadding]
        }
        set {
            if newValue.count >= 4 {
                keys = newValue[0]
                values = newValue[1]
                batchOffset = newValue[2]
                leftPadding = newValue[3]
                _idx = newValue[0].dim(2)
            } else if newValue.count == 2 {
                batchOffset = newValue[0]
                leftPadding = newValue[1]
                keys = nil
                values = nil
                _idx = 0
            } else {
                fatalError("BatchKVCache.state setter expects 2 or 4 arrays")
            }
        }
    }

    public override func innerState() -> [MLXArray] {
        state
    }

    public override func copy() -> any KVCache {
        BatchKVCache(
            keys: keys.map { $0[.ellipsis] },
            values: values.map { $0[.ellipsis] },
            offset: batchOffset[0...],
            leftPadding: leftPadding[0...],
            idx: _idx
        )
    }
}

/// Sliding-window batched KV cache.
///
/// This cache preserves the per-row position and left-padding semantics of
/// `BatchKVCache`, while trimming stored keys/values to `maxSize` for sliding
/// attention layers.
public final class BatchRotatingKVCache: BaseKVCache, BatchPositionedKVCache, BatchedCache {
    public static let allocationStep = 256

    public private(set) var keys: MLXArray?
    public private(set) var values: MLXArray?
    public private(set) var batchOffset: MLXArray
    public private(set) var leftPadding: MLXArray

    private let maxCacheSize: Int
    private var _idx: Int = 0
    private var _rightPadding: MLXArray?

    public init(maxSize: Int, leftPadding: [Int]) {
        self.maxCacheSize = maxSize
        self.leftPadding = MLXArray(leftPadding.map { Int32($0) })
        self.batchOffset = MLXArray(leftPadding.map { Int32(-$0) })
        super.init()
    }

    private init(
        maxSize: Int,
        keys: MLXArray?,
        values: MLXArray?,
        offset: MLXArray,
        leftPadding: MLXArray,
        idx: Int
    ) {
        self.maxCacheSize = maxSize
        self.keys = keys
        self.values = values
        self.batchOffset = offset
        self.leftPadding = leftPadding
        self._idx = idx
        super.init()
    }

    public override var maxSize: Int? { maxCacheSize }
    public override var isTrimmable: Bool { _idx < maxCacheSize }

    public override var offset: Int {
        get { _idx }
        set { _idx = newValue }
    }

    public func filterBatched(batchIndices: MLXArray) {
        filter(batchIndices: batchIndices)
    }

    public func extendBatched(_ other: any BatchedCache) {
        guard let other = other as? BatchRotatingKVCache else {
            preconditionFailure(
                "BatchRotatingKVCache.extendBatched requires another BatchRotatingKVCache")
        }
        extend(other)
    }

    public func prepareBatched(leftPadding: [Int]?, lengths: [Int]?, rightPadding: [Int]?) {
        prepare(leftPadding: leftPadding, lengths: lengths, rightPadding: rightPadding)
    }

    public func finalizeBatched() {
        finalize()
    }

    public func extractBatched(_ idx: Int) -> any KVCache {
        extract(idx)
    }

    public func advanceBatched(_: Int) {}

    public override func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        let stepCount = keys.dim(2)

        if self.keys == nil {
            self.keys = keys
            self.values = values
            _idx = stepCount
        } else {
            if stepCount > 1 {
                // Multi-token prefill must keep enough temporary context for
                // every query in this call. Match RotatingKVCache's concat
                // path: trim old context before appending, but do not trim the
                // newly returned prefill block.
                let trimSize = _idx - maxCacheSize + 1
                if trimSize > 0 {
                    self.keys = self.keys![.ellipsis, trimSize..., 0...]
                    self.values = self.values![.ellipsis, trimSize..., 0...]
                    leftPadding = leftPadding - Int32(trimSize)
                    _idx -= trimSize
                }
            }

            self.keys = concatenated([self.keys!, keys], axis: 2)
            self.values = concatenated([self.values!, values], axis: 2)
            _idx += stepCount
        }

        batchOffset = batchOffset + Int32(stepCount)

        if stepCount == 1, _idx > maxCacheSize {
            let trimSize = _idx - maxCacheSize
            self.keys = self.keys![.ellipsis, trimSize..., 0...]
            self.values = self.values![.ellipsis, trimSize..., 0...]
            leftPadding = leftPadding - Int32(trimSize)
            _idx = maxCacheSize
        }

        return (self.keys!, self.values!)
    }

    public func prepare(
        leftPadding additionalLeftPadding: [Int]? = nil,
        lengths _: [Int]? = nil,
        rightPadding: [Int]? = nil
    ) {
        if let additionalLeftPadding {
            precondition(
                keys == nil,
                "prepare() with leftPadding can only be called on an empty BatchRotatingKVCache"
            )
            let additional = MLXArray(additionalLeftPadding.map { Int32($0) })
            leftPadding = leftPadding + additional
            batchOffset = batchOffset - additional
        }

        if let rightPadding, rightPadding.contains(where: { $0 > 0 }) {
            _rightPadding = MLXArray(rightPadding.map { Int32($0) })
        }
    }

    public func finalize() {
        guard let pending = _rightPadding else { return }
        guard let storedK = keys, let storedV = values else {
            _rightPadding = nil
            return
        }

        let shifts = pending[0..., .newAxis]
        keys = dynamicRoll(storedK, shifts: shifts, axis: 2)
        values = dynamicRoll(storedV, shifts: shifts, axis: 2)
        batchOffset = batchOffset - pending
        leftPadding = leftPadding + pending
        _rightPadding = nil
    }

    public func filter(batchIndices: MLXArray) {
        if keys != nil {
            keys = take(keys!, batchIndices, axis: 0)
            values = take(values!, batchIndices, axis: 0)
        }
        batchOffset = take(batchOffset, batchIndices, axis: 0)
        leftPadding = take(leftPadding, batchIndices, axis: 0)
    }

    public func extend(_ other: BatchRotatingKVCache) {
        precondition(
            maxCacheSize == other.maxCacheSize,
            "BatchRotatingKVCache can only extend caches with the same maximum size"
        )

        if keys == nil && other.keys == nil {
            leftPadding = concatenated([leftPadding, other.leftPadding], axis: 0)
            batchOffset = concatenated([batchOffset, other.batchOffset], axis: 0)
            return
        }

        let maxIdx = max(_idx, other._idx)
        var L1 = 0
        var L2 = 0
        var H = 0
        var Dk = 0
        var Dv = 0

        if let storedK = keys {
            L1 = storedK.dim(2)
            H = storedK.dim(1)
            Dk = storedK.dim(3)
            Dv = values!.dim(3)
        }
        if let storedK = other.keys {
            L2 = storedK.dim(2)
            H = storedK.dim(1)
            Dk = storedK.dim(3)
            Dv = other.values!.dim(3)
        }
        let maxSize = max(L1, L2)

        func pad(
            _ cacheKeys: MLXArray?,
            _ cacheValues: MLXArray?,
            cacheIdx: Int,
            cacheOffset: MLXArray,
            cacheLeftPadding: MLXArray
        ) -> (MLXArray, MLXArray, MLXArray, MLXArray) {
            var k: MLXArray
            var v: MLXArray
            if let cacheKeys, let cacheValues {
                k = cacheKeys
                v = cacheValues
            } else {
                let Bc = cacheOffset.dim(0)
                k = MLXArray.zeros([Bc, H, 0, Dk], dtype: keys?.dtype ?? .float32)
                v = MLXArray.zeros([Bc, H, 0, Dv], dtype: values?.dtype ?? .float32)
            }

            let leftPad = maxIdx - cacheIdx
            var rightPad = maxSize - k.dim(2) - leftPad
            if rightPad < 0 {
                k = k[.ellipsis, ..<(k.dim(2) + rightPad), 0...]
                v = v[.ellipsis, ..<(v.dim(2) + rightPad), 0...]
                rightPad = 0
            }
            if leftPad != 0 || rightPad != 0 {
                let widths: [IntOrPair] = [
                    .init(0), .init(0), .init((leftPad, rightPad)), .init(0),
                ]
                k = padded(k, widths: widths)
                v = padded(v, widths: widths)
            }
            return (k, v, cacheOffset, cacheLeftPadding + Int32(leftPad))
        }

        let (k1, v1, o1, lp1) = pad(
            keys, values, cacheIdx: _idx,
            cacheOffset: batchOffset, cacheLeftPadding: leftPadding
        )
        let (k2, v2, o2, lp2) = pad(
            other.keys, other.values, cacheIdx: other._idx,
            cacheOffset: other.batchOffset, cacheLeftPadding: other.leftPadding
        )

        keys = concatenated([k1, k2], axis: 0)
        values = concatenated([v1, v2], axis: 0)
        batchOffset = concatenated([o1, o2], axis: 0)
        leftPadding = concatenated([lp1, lp2], axis: 0)
        _idx = maxIdx
    }

    public func extract(_ idx: Int) -> RotatingKVCache {
        let cache = RotatingKVCache(maxSize: maxCacheSize, keep: 0)
        guard let storedK = keys, let storedV = values else {
            return cache
        }

        let leftPad = max(0, Int(leftPadding[idx].item(Int32.self)))
        let kSlice = storedK[idx ..< (idx + 1), 0..., leftPad ..< _idx, 0...]
        let vSlice = storedV[idx ..< (idx + 1), 0..., leftPad ..< _idx, 0...]
        eval(kSlice, vSlice)
        cache.state = [kSlice, vSlice]

        let absoluteOffset = Int(batchOffset[idx].item(Int32.self))
        cache.metaState = [
            "0", "\(maxCacheSize)", "\(Self.allocationStep)", "\(absoluteOffset)",
            "\(kSlice.dim(2))",
        ]
        return cache
    }

    public override func makeMask(
        n: Int, windowSize: Int?, returnArray _: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        let actualWindowSize = windowSize ?? maxCacheSize
        let maskOffset = min(maxCacheSize - 1, _idx)
        let mask = createCausalMask(
            n: n,
            offset: maskOffset,
            windowSize: actualWindowSize,
            leftPadding: leftPadding
        )
        return .array(mask)
    }

    @discardableResult
    public override func trim(_ n: Int) -> Int {
        let trimmed = min(_idx, n)
        _idx -= trimmed
        batchOffset = batchOffset - Int32(trimmed)
        return trimmed
    }

    public func size() -> Int { _idx }
    public func isEmpty() -> Bool { keys == nil }

    public override var state: [MLXArray] {
        get {
            guard let storedK = keys, let storedV = values else {
                return [batchOffset, leftPadding]
            }
            return [storedK, storedV, batchOffset, leftPadding]
        }
        set {
            if newValue.count >= 4 {
                keys = newValue[0]
                values = newValue[1]
                batchOffset = newValue[2]
                leftPadding = newValue[3]
                _idx = newValue[0].dim(2)
            } else if newValue.count == 2 {
                batchOffset = newValue[0]
                leftPadding = newValue[1]
                keys = nil
                values = nil
                _idx = 0
            } else {
                fatalError("BatchRotatingKVCache.state setter expects 2 or 4 arrays")
            }
        }
    }

    public override func innerState() -> [MLXArray] {
        state
    }

    public override func copy() -> any KVCache {
        BatchRotatingKVCache(
            maxSize: maxCacheSize,
            keys: keys.map { $0[.ellipsis] },
            values: values.map { $0[.ellipsis] },
            offset: batchOffset[0...],
            leftPadding: leftPadding[0...],
            idx: _idx
        )
    }
}

// MARK: - ArraysCache conformance

extension ArraysCache: BatchedCache {
    public func filterBatched(batchIndices: MLXArray) {
        filter(batchIndices: batchIndices)
    }

    public func extendBatched(_ other: any BatchedCache) {
        guard let other = other as? ArraysCache else {
            preconditionFailure("ArraysCache.extendBatched requires another ArraysCache")
        }
        extend(other: other)
    }

    public func prepareBatched(leftPadding _: [Int]?, lengths: [Int]?, rightPadding _: [Int]?) {
        prepare(lengths: lengths)
    }

    public func finalizeBatched() {
        finalize()
    }

    public func extractBatched(_ idx: Int) -> any KVCache {
        extract(idx)
    }

    public func advanceBatched(_ n: Int) {
        advance(n)
    }
}

/// Per-row roll of `x` along `axis`. `shifts` broadcasts on the leading
/// axes (typically rank `axis+1` -- the per-row shift count).
@inline(__always)
public func dynamicRoll(
    _ x: MLXArray, shifts: MLXArray, axis: Int
) -> MLXArray {
    let n = x.dim(axis)
    var arangeShape = Array(repeating: 1, count: x.ndim)
    arangeShape[axis] = n
    var shiftShape = Array(repeating: 1, count: x.ndim)
    for i in 0 ..< min(shifts.ndim, axis) {
        shiftShape[i] = shifts.dim(i)
    }

    let arange = MLXArray(Int32(0) ..< Int32(n)).reshaped(arangeShape)
    let reshapedShifts = shifts.reshaped(shiftShape)
    let idx = (arange - reshapedShifts) % Int32(n)
    return takeAlong(x, idx, axis: axis)
}
