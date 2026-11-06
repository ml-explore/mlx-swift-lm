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

// MARK: - Recurrent caches

extension ArraysCache {
    /// Check merge storage without reading tensors or excluding prepared metadata.
    static func validateMerge(_ caches: [ArraysCache]) throws {
        guard let first = caches.first else {
            throw KVCacheError(message: "merge needs a cache")
        }
        guard caches.allSatisfy({ $0.slotCount == first.slotCount }) else {
            throw KVCacheError(message: "Recurrent rows must share one slot count")
        }
        var layouts = [(dtype: DType, shape: [Int])?](repeating: nil, count: first.slotCount)
        for cache in caches {
            let state = (0 ..< cache.slotCount).map { cache[$0] }
            let metadata = [cache.leftPadding, cache.lengths]
            guard state.allSatisfy({ $0 == nil || $0!.ndim > 0 }),
                metadata.allSatisfy({ $0 == nil || $0!.ndim == 1 })
            else {
                throw KVCacheError(message: "Recurrent rows need row-shaped state and metadata")
            }
            let count = cache.batchSize
            guard count >= 0,
                state.allSatisfy({ $0 == nil || $0!.dim(0) == count }),
                metadata.allSatisfy({ $0 == nil || $0!.size == count })
            else {
                throw KVCacheError(message: "Recurrent row state must agree on its batch size")
            }
            for (slot, array) in state.enumerated() {
                guard let array else { continue }
                let shape = Array(array.shape.dropFirst())
                if let layout = layouts[slot] {
                    guard array.dtype == layout.dtype, shape == layout.shape else {
                        throw KVCacheError(
                            message: "Recurrent rows must share each state slot's layout")
                    }
                } else {
                    layouts[slot] = (array.dtype, shape)
                }
            }
        }
    }

    /// The rows of `caches` as one cache. Port of mlx-lm's `ArraysCache.merge`.
    ///
    /// Recurrent state has no padding to align, so this is the first row extended by the
    /// rest; a batch of `MambaCache` stays a `MambaCache`. When any row holds state, the
    /// empty rows get zero state of the same shape first, so every row counts. An all-empty
    /// merge stays empty and takes its row count from the first update.
    /// Inputs must share their slot counts and populated state layouts.
    public static func merge(_ caches: [ArraysCache]) -> ArraysCache {
        precondition(!caches.isEmpty, "merge needs a cache")
        do {
            try validateMerge(caches)
        } catch {
            preconditionFailure(error.localizedDescription)
        }
        func isEmpty(_ cache: ArraysCache) -> Bool {
            (0 ..< cache.slotCount).allSatisfy { cache[$0] == nil }
        }
        let template = caches.first { !isEmpty($0) }
        let rows: [ArraysCache] = caches.map { cache in
            guard let row = cache.copy() as? ArraysCache else {
                preconditionFailure("An ArraysCache copy must stay an ArraysCache")
            }
            guard let template, isEmpty(row) else { return row }
            // An empty cache may already stand for several rows (its padding says how many).
            let rowCount = row.batchSize
            for slot in 0 ..< row.slotCount {
                if let example = template[slot] {
                    row[slot] = MLXArray.zeros(
                        [rowCount] + Array(example.shape.dropFirst()), dtype: example.dtype)
                }
            }
            return row
        }
        let merged = rows[0]
        for row in rows.dropFirst() {
            merged.extend(other: row)
        }
        // An all-empty merge has no arrays to count rows by, so it records how many it holds
        // (mlx-lm gives every row zero padding; see `emptyRowCount`).
        if template == nil {
            merged.emptyRowCount = caches.reduce(0) { $0 + $1.batchSize }
        }
        // Recurrent state keeps one position counter for the batch: the longest row's.
        merged.offset = caches.map(\.offset).max() ?? 0
        return merged
    }
}

// MARK: - Composite caches

extension CacheList {
    /// The rows of `caches`, composite caches of one layout, as one composite whose children
    /// are batched. Each child position merges on its own type. Port of mlx-lm's
    /// `CacheList.merge`.
    ///
    /// - Throws: `KVCacheError` when the layouts differ or a child has no batched form.
    public static func merge(_ caches: [CacheList]) throws -> CacheList {
        precondition(!caches.isEmpty, "merge needs a cache")
        let width = caches[0].children.count
        guard caches.allSatisfy({ $0.children.count == width }) else {
            throw KVCacheError(message: "Rows of a CacheList batch must share one layout")
        }
        let merged = try (0 ..< width).map { index in
            try mergeCaches(caches.map { $0.children[index] })
        }
        return CacheList(caches: merged)
    }
}
