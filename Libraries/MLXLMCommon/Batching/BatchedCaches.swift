// Copyright © 2026 Apple Inc.

import Foundation
import MLX

/// The rows of `caches`, single-sequence caches of one layer, as one batched cache.
///
/// Port of mlx-lm's `_merge_caches` for one layer: the rows merge on their exact class, so a
/// `ChunkedKVCache` or an unsupported compressed cache is refused rather than treated as
/// the class it subclasses. Empty caches merge into an empty batch with one row each.
///
/// - Throws: `KVCacheError` when the type has no batched form, when the rows do not share
///   one type, or when a rotating cache keeps a prefix or a rewind reserve.
public func mergeCaches(_ caches: [any KVCache]) throws -> any KVCache {
    guard let first = caches.first else {
        throw KVCacheError(message: "merge needs a cache")
    }
    guard caches.allSatisfy({ type(of: $0) == type(of: first) }) else {
        throw KVCacheError(message: "Rows of a batch must share one cache type")
    }
    return try type(of: first).merge(caches)
}

/// `caches` as rows of exactly `T`, for a `merge` that takes only its own class.
func batchRows<T: KVCache>(_ caches: [any KVCache], of _: T.Type) throws -> [T] {
    try caches.map { cache in
        guard type(of: cache) == T.self, let row = cache as? T else {
            throw KVCacheError(message: "\(type(of: cache)) has no batched form")
        }
        return row
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
