// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXNN

// MARK: - BatchKVCache

/// Batch-aware KV cache with left-padding strategy for continuous batching.
///
/// Ported from Python mlx-lm's `BatchKVCache`. The cache expects inputs to be
/// left-padded so that variable-length sequences align on the right.
///
/// For example, prompts `[1, 3, 5]`, `[7]`, and `[2, 6, 8, 9]` are padded:
/// ```
/// [0, 1, 3, 5]
/// [0, 0, 0, 7]
/// [2, 6, 8, 9]
/// ```
/// With `leftPadding = [1, 3, 0]`.
public class BatchKVCache: BaseKVCache, BatchPositionedKVCache {

    /// Per-sequence left-padding amounts as an MLXArray of shape `[B]`.
    public internal(set) var leftPadding: MLXArray

    /// Per-sequence offset as an MLXArray of shape `[B]`.
    /// Starts negative (equal to `-leftPadding`) and advances with each update.
    public internal(set) var batchOffsets: MLXArray

    /// Internal buffer index tracking how far into the keys/values buffer we've written.
    internal var _idx: Int = 0

    /// Keys buffer: `[B, H, S_buf, D_k]`
    internal var keys: MLXArray?

    /// Values buffer: `[B, H, S_buf, D_v]`
    internal var values: MLXArray?

    /// Step size for buffer allocation (grow in chunks of this size).
    public var step: Int = 256

    /// The scalar offset (not meaningful for batch caches, returns `_idx`).
    public override var offset: Int {
        get { _idx }
        set { _idx = newValue }
    }

    /// Initialize a BatchKVCache with the given left-padding per sequence.
    ///
    /// - Parameter leftPadding: Array of integers specifying the left-padding for each sequence.
    public init(leftPadding: [Int]) {
        self.leftPadding = MLXArray(leftPadding.map { Int32($0) })
        self.batchOffsets = MLXArray(leftPadding.map { -Int32($0) })
        super.init()
    }

    /// Internal initializer for creating empty batch caches with pre-built MLXArrays.
    internal init(leftPaddingArray: MLXArray, batchOffsetsArray: MLXArray) {
        self.leftPadding = leftPaddingArray
        self.batchOffsets = batchOffsetsArray
        super.init()
    }

    // MARK: - KVCache Protocol

    public override func innerState() -> [MLXArray] {
        [self.keys, self.values].compactMap { $0 }
    }

    /// Update the cache with new keys and values.
    ///
    /// Keys/values have shape `[B, H, S, D]` where `S` is the number of new tokens.
    /// The cache buffer grows in steps of `step` size.
    public override func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        let prev = _idx

        let reset: Bool
        if let currentKeys = self.keys, (prev + keys.dim(2)) <= currentKeys.dim(2) {
            reset = false
        } else {
            reset = true
        }

        if reset {
            let B = keys.dim(0)
            let kvHeads = keys.dim(1)
            let kHeadDim = keys.dim(3)
            let vHeadDim = values.dim(3)

            let nSteps = (step + keys.dim(2) - 1) / step
            let kShape = [B, kvHeads, nSteps * step, kHeadDim]
            let vShape = [B, kvHeads, nSteps * step, vHeadDim]
            let newK = MLXArray.zeros(kShape, dtype: keys.dtype)
            let newV = MLXArray.zeros(vShape, dtype: values.dtype)

            if var currentKeys = self.keys, var currentValues = self.values {
                if prev % step != 0 {
                    currentKeys = currentKeys[.ellipsis, ..<prev, 0...]
                    currentValues = currentValues[.ellipsis, ..<prev, 0...]
                }
                self.keys = concatenated([currentKeys, newK], axis: 2)
                self.values = concatenated([currentValues, newV], axis: 2)
            } else {
                self.keys = newK
                self.values = newV
            }
        }

        batchOffsets = batchOffsets + Int32(keys.dim(2))
        _idx += keys.dim(2)

        self.keys?[.ellipsis, prev ..< _idx, 0...] = keys
        self.values?[.ellipsis, prev ..< _idx, 0...] = values

        let returnedKeys = self.keys![.ellipsis, ..<_idx, 0...]
        let returnedValues = self.values![.ellipsis, ..<_idx, 0...]

        return (returnedKeys, returnedValues)
    }

    // MARK: - State Serialization

    public override var state: [MLXArray] {
        get {
            guard let keys = self.keys, let values = self.values else { return [] }
            let k: MLXArray
            let v: MLXArray
            if _idx < keys.dim(2) {
                k = keys[.ellipsis, ..<_idx, 0...]
                v = values[.ellipsis, ..<_idx, 0...]
            } else {
                k = keys
                v = values
            }
            return [k, v, batchOffsets, leftPadding]
        }
        set {
            guard newValue.count == 4 else {
                fatalError("BatchKVCache state must have exactly 4 arrays (keys, values, offset, leftPadding)")
            }
            self.keys = newValue[0]
            self.values = newValue[1]
            self.batchOffsets = newValue[2]
            self.leftPadding = newValue[3]
            self._idx = self.keys!.dim(2)
        }
    }

    public override var metaState: [String] {
        get { [String(_idx)] }
        set {
            guard newValue.count == 1 else {
                fatalError("BatchKVCache metaState must have exactly 1 value")
            }
            self._idx = Int(newValue[0]) ?? 0
        }
    }

    public override var isTrimmable: Bool { true }

    @discardableResult
    public override func trim(_ n: Int) -> Int {
        let trimmed = min(_idx, n)
        _idx -= trimmed
        batchOffsets = batchOffsets - Int32(trimmed)
        return trimmed
    }

    /// The batch size (number of sequences).
    public var batchSize: Int {
        leftPadding.dim(0)
    }

    /// Whether the cache is empty (no keys/values stored).
    public var isEmpty: Bool {
        keys == nil
    }

    // MARK: - BatchPositionedKVCache Conformance

    /// Per-sequence position offsets as an MLXArray of shape `[B]`.
    ///
    /// This is an alias for `batchOffsets`, providing the per-sequence position
    /// offsets needed for batch-aware RoPE application via `applyRotaryPosition()`.
    public var batchOffset: MLXArray {
        batchOffsets
    }

    // MARK: - Batch Operations

    /// In-place filter to keep only the sequences at the given batch indices.
    ///
    /// After filtering, the minimum left-padding is subtracted from all sequences
    /// and the buffer is trimmed accordingly (shift left to reduce padding).
    ///
    /// - Parameter batchIndices: Array of batch indices to keep.
    public func filter(batchIndices: [Int]) {
        // Handle empty filter -> produce valid empty state
        guard !batchIndices.isEmpty else {
            keys = nil
            values = nil
            leftPadding = MLXArray([Int32]())
            batchOffsets = MLXArray([Int32]())
            _idx = 0
            return
        }

        let indices = MLXArray(batchIndices.map { Int32($0) })

        // Filter along batch dimension (dim 0)
        keys = keys?[indices]
        values = values?[indices]
        batchOffsets = batchOffsets[indices]
        leftPadding = leftPadding[indices]

        // Shift left to reduce padding
        let minLeftPad = leftPadding.min().item(Int32.self)
        if minLeftPad > 0 {
            let padInt = Int(minLeftPad)
            keys = keys?[.ellipsis, padInt..., 0...]
            values = values?[.ellipsis, padInt..., 0...]
            _idx -= padInt
            leftPadding = leftPadding - minLeftPad
        }
    }

    /// In-place extend this cache with another BatchKVCache.
    ///
    /// The caches are right-justified: the shorter cache gets additional left-padding
    /// to align with the longer one along the sequence dimension.
    ///
    /// - Parameter other: The other BatchKVCache to merge into this one.
    public func extend(other: BatchKVCache) {
        guard let selfKeys = self.keys, let otherKeys = other.keys else {
            // If self is empty, take the other's state
            if other.keys != nil {
                self.keys = other.keys
                self.values = other.values
                self.batchOffsets = other.batchOffsets
                self.leftPadding = other.leftPadding
                self._idx = other._idx
            }
            return
        }

        let maxIdx = max(self._idx, other._idx)
        let maxSize = max(selfKeys.dim(2), otherKeys.dim(2))

        // Inner function to pad a cache's keys/values for right-justification.
        func pad(
            _ cache: BatchKVCache
        ) -> (MLXArray, MLXArray, MLXArray, MLXArray) {
            let left = maxIdx - cache._idx
            var right = maxSize - cache.keys!.dim(2) - left

            var k = cache.keys!
            var v = cache.values!

            if right < 0 {
                k = k[.ellipsis, ..<(k.dim(2) + right), 0...]
                v = v[.ellipsis, ..<(v.dim(2) + right), 0...]
                right = 0
            }

            if left != 0 || right != 0 {
                let padWidths: [IntOrPair] = [0, 0, .init((left, right)), 0]
                k = MLX.padded(k, widths: padWidths)
                v = MLX.padded(v, widths: padWidths)
            }

            let adjustedLeftPadding = cache.leftPadding + Int32(left)

            return (k, v, cache.batchOffsets, adjustedLeftPadding)
        }

        let (selfK, selfV, selfOff, selfLP) = pad(self)
        let (otherK, otherV, otherOff, otherLP) = pad(other)

        self.keys = concatenated([selfK, otherK], axis: 0)
        self.values = concatenated([selfV, otherV], axis: 0)
        self.batchOffsets = concatenated([selfOff, otherOff], axis: 0)
        self.leftPadding = concatenated([selfLP, otherLP], axis: 0)
        self._idx = maxIdx
    }

    /// Extract a single sequence from the batch as a `KVCacheSimple`.
    ///
    /// The returned cache has the left-padding stripped and contains only the
    /// valid (non-padded) key/value data.
    ///
    /// - Parameter idx: The batch index of the sequence to extract.
    /// - Returns: A `KVCacheSimple` with the extracted sequence data.
    public func extract(idx: Int) -> KVCacheSimple {
        let cache = KVCacheSimple()
        let padding = Int(leftPadding[idx].item(Int32.self))

        if let k = keys, let v = values {
            cache.keys = MLX.contiguous(k[idx ..< (idx + 1), 0..., padding ..< _idx, 0...])
            cache.values = MLX.contiguous(v[idx ..< (idx + 1), 0..., padding ..< _idx, 0...])
            cache.offset = cache.keys!.dim(2)
        }

        return cache
    }

    /// Create a BatchKVCache by merging multiple individual KVCache instances.
    ///
    /// Each cache is right-justified in the batch: shorter caches receive left-padding
    /// to match the longest sequence.
    ///
    /// - Parameter caches: An array of `KVCache` instances (typically `KVCacheSimple`).
    /// - Returns: A new `BatchKVCache` containing all sequences.
    public class func merge(_ caches: [KVCache]) -> BatchKVCache {
        let lengths = caches.map { $0.offset }
        let maxLength = lengths.max() ?? 0
        let padding = lengths.map { maxLength - $0 }
        let B = caches.count

        // Find dimensions from first non-empty cache
        var H = 0
        var Dk = 0
        var Dv = 0
        var dt: DType = .float16

        for c in caches {
            if let simple = c as? KVCacheSimple, let k = simple.keys {
                H = k.dim(1)
                Dk = k.dim(3)
                Dv = simple.values!.dim(3)
                dt = k.dtype
                break
            }
        }

        guard H > 0 else {
            // All caches are empty
            return BatchKVCache(leftPadding: padding)
        }

        let keysArr = MLXArray.zeros([B, H, maxLength, Dk], dtype: dt)
        let valuesArr = MLXArray.zeros([B, H, maxLength, Dv], dtype: dt)

        for (i, (p, c)) in zip(padding, caches).enumerated() {
            if let simple = c as? KVCacheSimple, let k = simple.keys, let v = simple.values {
                let seqLen = c.offset
                keysArr[i ..< (i + 1), 0..., p ..< (p + seqLen), 0...] =
                    k[.ellipsis, ..<seqLen, 0...]
                valuesArr[i ..< (i + 1), 0..., p ..< (p + seqLen), 0...] =
                    v[.ellipsis, ..<seqLen, 0...]
            }
        }

        let cache = BatchKVCache(leftPadding: padding)
        cache.keys = keysArr
        cache.values = valuesArr
        // After merge, offset should advance by maxLength for all sequences
        cache.batchOffsets = cache.batchOffsets + Int32(maxLength)
        cache._idx = maxLength

        return cache
    }

    /// Create a batch-1 BatchKVCache from a single KVCacheSimple.
    ///
    /// The resulting cache has `leftPadding = [0]` and identical data.
    ///
    /// - Parameter cache: A single `KVCacheSimple` to wrap.
    /// - Returns: A new `BatchKVCache` with batch size 1.
    public class func fromSingle(_ cache: KVCacheSimple) -> BatchKVCache {
        let batchCache = BatchKVCache(leftPadding: [0])

        if let k = cache.keys, let v = cache.values {
            batchCache.keys = k
            batchCache.values = v
            batchCache._idx = cache.offset
            batchCache.batchOffsets = MLXArray([Int32(cache.offset)])
        }

        return batchCache
    }

    /// Convert a batch-1 BatchKVCache back to a KVCacheSimple.
    ///
    /// - Returns: A `KVCacheSimple` with the single sequence data.
    public func toSingle() -> KVCacheSimple {
        precondition(batchSize == 1, "toSingle() requires batch size of 1")
        return extract(idx: 0)
    }

    // MARK: - Mask Creation

    /// Create an attention mask for this batch cache.
    ///
    /// Unlike non-batch caches which return `.none` for `n=1`, batch caches
    /// MUST always produce a mask that excludes left-padded positions. This
    /// ensures that during single-token decode steps, padded positions are
    /// still correctly masked out.
    ///
    /// - Parameters:
    ///   - n: The sequence length for the new tokens
    ///   - windowSize: Optional sliding window size
    ///   - returnArray: Force return of array mask instead of symbolic
    /// - Returns: Attention mask mode for scaled dot product attention
    public override func makeMask(
        n: Int, windowSize: Int?, returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        // Batch caches always need an explicit mask to handle left-padding,
        // even for n=1 decode steps.
        return .array(
            createCausalMask(
                n: n, offset: _idx - n, windowSize: windowSize, leftPadding: leftPadding
            )
        )
    }

    public var debugDescription: String {
        "BatchKVCache batchSize: \(batchSize), _idx: \(_idx), keys: \(keys?.shape.description ?? "-"), values: \(values?.shape.description ?? "-")"
    }
}
