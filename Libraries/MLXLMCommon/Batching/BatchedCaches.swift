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
/// Port of mlx-lm's `_merge_caches` for one layer. A cache that is its own batch
/// (``ArraysCache`` and ``MambaCache``, ``CacheList``) merges through its own class; a ``KVCacheSimple`` row merges into a ``BatchKVCache`` and a
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

// MARK: - Recurrent caches

extension ArraysCache: BatchableKVCache {
    /// Every slot's state, `nil` where a slot holds none.
    private var slots: [MLXArray?] { (0 ..< slotCount).map { self[$0] } }

    public func extend(other: any KVCache) {
        guard let other = other as? ArraysCache else {
            preconditionFailure("\(Self.self) can only be extended with an ArraysCache")
        }
        extend(other: other)
    }

    public func validateExtension(other: any KVCache) throws {
        guard type(of: self) == ArraysCache.self || type(of: self) == MambaCache.self,
            type(of: self) == type(of: other), let other = other as? ArraysCache,
            slotCount == other.slotCount
        else {
            throw KVCacheError(
                message: "Recurrent batches must share one exact type and slot count")
        }
        for row in [self, other] {
            let state = row.slots
            guard row.lengths == nil, state.allSatisfy({ $0 == nil || $0!.ndim > 0 }) else {
                throw KVCacheError(message: "Recurrent batches need finalized, row-shaped state")
            }
            let count = row.batchSize
            guard state.allSatisfy({ $0 == nil || $0!.dim(0) == count }),
                row.leftPadding.map({ $0.ndim == 1 && $0.size == count }) ?? true
            else {
                throw KVCacheError(message: "Recurrent batch state must agree on the row count")
            }
        }
        for (own, added) in zip(slots, other.slots) {
            if let own, let added,
                own.dtype != added.dtype || own.shape.dropFirst() != added.shape.dropFirst()
            {
                throw KVCacheError(message: "Recurrent batches must share each state slot's layout")
            }
        }
    }

    /// The row at `index`, as a ``MambaCache`` when this is one.
    public func extract(_ index: Int) -> any KVCache {
        let row = self is MambaCache ? MambaCache() : ArraysCache(size: slotCount)
        for slot in 0 ..< slotCount {
            row[slot] = self[slot]?[index ..< index + 1]
        }
        row.offset = offset
        row.leftPadding = leftPadding?[index ..< index + 1]
        row.lengths = lengths?[index ..< index + 1]
        return row
    }

    /// ``merge(_:)-swift.type.method`` for rows typed as `KVCache`; rows of exactly this class.
    public static func merge(_ caches: [any KVCache]) throws -> any BatchableKVCache {
        guard Self.self == ArraysCache.self || Self.self == MambaCache.self else {
            throw KVCacheError(message: "\(Self.self) must implement its own batched form")
        }
        let rows = try batchRows(caches, of: Self.self)
        try ArraysCache.validateMerge(rows)
        return merge(rows)
    }

    /// Check merge storage without reading tensors or excluding prepared metadata.
    ///
    /// Rows must be all exactly `ArraysCache` or all exactly `MambaCache`. A subclass such as
    /// LFM2's `RewindableConvolutionCache` keeps state this merge would drop, so it needs a
    /// batched form of its own.
    static func validateMerge(_ caches: [ArraysCache]) throws {
        guard let first = caches.first else {
            throw KVCacheError(message: "merge needs a cache")
        }
        let kind = type(of: first)
        guard kind == ArraysCache.self || kind == MambaCache.self else {
            throw KVCacheError(message: "\(kind) must implement its own batched form")
        }
        guard caches.allSatisfy({ type(of: $0) == kind }) else {
            throw KVCacheError(message: "Recurrent rows must all be \(kind)")
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
    /// Inputs must be all exactly `ArraysCache` or all exactly `MambaCache`, and share their
    /// slot counts and populated state layouts.
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
            // mlx-vlm zero-fills an empty row's slots the same way:
            // https://github.com/Blaizzy/mlx-vlm/blob/dcea787/mlx_vlm/models/cache.py#L1306-L1317
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
        // (mlx-lm and mlx-vlm give every row zero padding; see `emptyRowCount`).
        if template == nil {
            merged.emptyRowCount = caches.reduce(0) { $0 + $1.batchSize }
        }
        // Recurrent state keeps one position counter for the batch: the longest row's.
        merged.offset = caches.map(\.offset).max() ?? 0
        return merged
    }
}

// MARK: - Composite caches

extension CacheList: BatchableKVCache {
    /// The children as batched caches, as ``merge(_:)-swift.type.method`` makes them.
    /// ``validateExtension(other:)`` refuses a child that is not one by throwing.
    private var batchedChildren: [any BatchableKVCache] {
        children.map { child in
            guard let child = child as? any BatchableKVCache else {
                preconditionFailure(
                    "\(type(of: child)) in a CacheList is not a batched cache; merge rows first")
            }
            return child
        }
    }

    /// Keeps only the rows at `batchIndices`, in that order, in every child.
    public func filter(batchIndices: MLXArray) {
        for child in batchedChildren {
            child.filter(batchIndices: batchIndices)
        }
    }

    /// Appends the rows of `other`, a `CacheList` of the same layout, child by child.
    public func extend(other: any KVCache) {
        guard let other = other as? CacheList, other.children.count == children.count else {
            preconditionFailure(
                "A CacheList can only be extended with a CacheList of the same layout")
        }
        for (child, added) in zip(batchedChildren, other.children) {
            child.extend(other: added)
        }
    }

    public func validateExtension(other: any KVCache) throws {
        guard type(of: self) == CacheList.self, type(of: other) == CacheList.self,
            let other = other as? CacheList, other.children.count == children.count
        else {
            throw KVCacheError(message: "Composite batches must share one exact type and layout")
        }
        for (child, added) in zip(children, other.children) {
            guard let child = child as? any BatchableKVCache else {
                throw KVCacheError(
                    message: "\(type(of: child)) in a CacheList is not a batched cache")
            }
            try child.validateExtension(other: added)
        }
    }

    /// The row at `index` as a composite of single-sequence caches.
    public func extract(_ index: Int) -> any KVCache {
        CacheList(caches: batchedChildren.map { $0.extract(index) })
    }

    /// ``merge(_:)-swift.type.method`` for rows typed as `KVCache`.
    public static func merge(_ caches: [any KVCache]) throws -> any BatchableKVCache {
        try merge(try batchRows(caches, of: CacheList.self))
    }

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
