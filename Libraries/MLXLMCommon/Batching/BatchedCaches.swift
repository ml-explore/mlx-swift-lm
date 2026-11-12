// Copyright © 2026 Apple Inc.

import Foundation
import MLX

/// A cache that holds the rows of a batch, with the row operations continuous batching needs.
///
/// Port of the row operations of mlx-lm's batch caches (`filter`, `extend`, `extract` and
/// `merge`). Single-sequence caches such as ``KVCacheSimple`` do not conform:
/// ``mergeCaches(_:)`` turns their rows into the batched cache of their kind, which does.
public protocol BatchableKVCache: KVCache {
    /// Keep only the rows at `batchIndices`, in that order.
    func filter(batchIndices: MLXArray)

    /// Append the rows of `other`, a batch of this type that ``validateExtension(other:)``
    /// accepted.
    func extend(other: any KVCache)

    /// Check that ``extend(other:)`` can append `other` without changing either cache.
    ///
    /// Admission validates every layer before extending any layer or sampling tokens.
    /// - Throws: An error when the types, settings, or storage layouts are incompatible.
    func validateExtension(other: any KVCache) throws

    /// The row at `index` as a single-sequence cache.
    ///
    /// The batch must be finalized: extracting during a ragged prefill, before `finalize()`
    /// moves its right padding, stops with a precondition failure.
    func extract(_ index: Int) -> any KVCache

    /// The rows of `caches`, single-sequence caches of one layer, as a batch of this type.
    ///
    /// - Throws: `KVCacheError` when there are no rows, when a row is not exactly the class
    ///   this batch takes, or when a row cannot be batched.
    static func merge(_ caches: [any KVCache]) throws -> any BatchableKVCache
}

/// The rows of `caches`, single-sequence caches of one layer, as one batched cache.
///
/// Port of mlx-lm's `_merge_caches` for one layer. A cache that is its own batch merges
/// through its own class; a ``KVCacheSimple`` row merges into a ``BatchKVCache`` and a
/// ``RotatingKVCache`` row into a ``BatchRotatingKVCache``. Each batch
/// takes rows of exactly its class, so a `ChunkedKVCache` or an unsupported compressed cache
/// is refused rather than treated as the class it subclasses. Empty caches merge into an
/// empty batch with one row each.
///
/// - Throws: `KVCacheError` when the type has no batched form, when the rows do not share
///   one type, or when a rotating cache keeps a prefix or a rewind reserve.
public func mergeCaches(_ caches: [any KVCache]) throws -> any BatchableKVCache {
    guard let first = caches.first else {
        throw KVCacheError(message: "merge needs a cache")
    }
    guard caches.allSatisfy({ type(of: $0) == type(of: first) }) else {
        throw KVCacheError(message: "Rows of a batch must share one cache type")
    }
    let batch: any BatchableKVCache.Type
    switch first {
    case let own as any BatchableKVCache: batch = type(of: own)
    case is KVCacheSimple: batch = BatchKVCache.self
    case is RotatingKVCache: batch = BatchRotatingKVCache.self
    default: throw KVCacheError(message: "\(type(of: first)) has no batched form")
    }
    return try batch.merge(caches)
}

/// `caches` as rows of exactly `T`, for a `merge` that takes only its own class.
func batchRows<T: KVCache>(_ caches: [any KVCache], of _: T.Type) throws -> [T] {
    guard !caches.isEmpty else {
        throw KVCacheError(message: "merge needs a cache")
    }
    return try caches.map { cache in
        guard type(of: cache) == T.self, let row = cache as? T else {
            throw KVCacheError(message: "\(type(of: cache)) has no batched form")
        }
        return row
    }
}

/// Check that `rows`, single-sequence rows of one attention layer with `end` positions
/// written, can be stacked into one batch: each holds paired 4-D keys and values with one row,
/// and every row that holds keys matches the heads, widths and dtype of the first one that
/// does. Reads only array metadata.
func validateBatchKVRows(_ rows: [(keys: MLXArray?, values: MLXArray?, end: Int)]) throws {
    let reference = rows.first { $0.keys != nil } ?? (nil, nil, 0)
    for row in rows {
        try validateBatchKVExtension(
            keys: reference.keys, values: reference.values, batchSize: 1, end: reference.end,
            otherKeys: row.keys, otherValues: row.values, otherBatchSize: 1, otherEnd: row.end)
    }
}

/// Check concatenation dimensions without evaluating either cache's arrays.
func validateBatchKVExtension(
    keys: MLXArray?, values: MLXArray?, batchSize: Int, end: Int,
    otherKeys: MLXArray?, otherValues: MLXArray?, otherBatchSize: Int, otherEnd: Int
) throws {
    for (keys, values, count, end) in [
        (keys, values, batchSize, end), (otherKeys, otherValues, otherBatchSize, otherEnd),
    ] {
        guard (keys == nil) == (values == nil), end >= 0 else {
            throw KVCacheError(message: "Batched attention needs paired key/value storage")
        }
        if let keys, let values {
            guard keys.ndim == 4, values.ndim == 4,
                keys.dim(0) == count, values.dim(0) == count,
                keys.dim(1) == values.dim(1), keys.dim(2) == values.dim(2), end <= keys.dim(2)
            else {
                throw KVCacheError(message: "Batched attention has an invalid key/value layout")
            }
        } else if end != 0 {
            throw KVCacheError(message: "An empty attention batch cannot hold written positions")
        }
    }
    for (own, added) in [(keys, otherKeys), (values, otherValues)] {
        if let own, let added,
            own.dtype != added.dtype || own.dim(1) != added.dim(1) || own.dim(3) != added.dim(3)
        {
            throw KVCacheError(
                message: "Attention batches must share key/value heads, widths and dtypes")
        }
    }
}
