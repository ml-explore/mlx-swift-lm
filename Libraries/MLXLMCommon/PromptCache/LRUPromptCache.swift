// Copyright © 2025 Apple Inc.

import Foundation

/// Cross-request LRU prompt cache: keeps recently-used KV caches keyed by
/// `(model, tokens)` and, for a new prompt, returns the nearest reusable cache
/// plus the token remainder still to be processed.
///
/// Direct port of Python `mlx_lm`'s `LRUPromptCache` (`mlx_lm/models/cache.py`).
/// It also carries fixes for two bugs still open in the Python implementation,
/// reported in ml-explore/mlx-lm#1495 and fixed there by ml-explore/mlx-lm#1496:
///
/// - a value stored at a single-token prefix is now returned
///   (``PromptTrie/search(model:tokens:)`` uses `lastIndex >= 0`).
/// - fetching a cache refreshes its LRU recency, so the store is truly
///   least-recently-*used* rather than FIFO. See ``fetchNearestCache(model:tokens:)``.
///
/// Eviction is type-aware: entries are ordered `assistant` → `user` → `system`,
/// and the eviction policy sheds the most-populous higher-churn class first, so
/// system prompts survive longest.
///
/// This is a plain non-`Sendable` `final class`, mirroring Python's
/// single-server-thread assumption. Cross-task consumers must wrap an instance
/// in a `SerialAccessContainer` (see `Utilities/SerialAccessContainer.swift`).
public final class LRUPromptCache<Model: Hashable> {

    /// One stored cache plus its byte size and type.
    public struct CacheEntry {
        public var promptCache: [KVCache]
        public var nbytes: Int
        public var cacheType: String
    }

    /// Per-type sequence statistics returned by ``statsByType()``.
    public struct TypeStats {
        public let sequences: Int
        public let bytes: Int
    }

    /// Type-partitioned recency lists. Newest is appended; oldest is `first`.
    private final class CacheOrder {
        let ordering: [String]
        var lrus: [String: [(Model, [Int])]]

        init(ordering: [String] = ["assistant", "user", "system"]) {
            self.ordering = ordering
            self.lrus = Dictionary(uniqueKeysWithValues: ordering.map { ($0, []) })
        }

        var count: Int { lrus.values.reduce(0) { $0 + $1.count } }

        func count(ofType cacheType: String) -> Int { lrus[cacheType]?.count ?? 0 }

        func push(model: Model, tokens: [Int], cacheType: String = "assistant") {
            lrus[cacheType, default: []].append((model, tokens))
        }

        func remove(model: Model, tokens: [Int]) {
            for cacheType in ordering {
                if let idx = lrus[cacheType]!.firstIndex(where: {
                    $0.0 == model && $0.1 == tokens
                }) {
                    lrus[cacheType]!.remove(at: idx)
                    return
                }
            }
        }

        /// Evict the least-recently-used entry from the most-populous
        /// higher-churn class (assistant before user before system).
        func pop() -> (Model, [Int]) {
            var i = 0
            while i + 1 < ordering.count {
                let a = lrus[ordering[i]]!
                let b = lrus[ordering[i + 1]]!
                if !a.isEmpty && a.count >= b.count {
                    return lrus[ordering[i]]!.removeFirst()
                }
                i += 1
            }
            return lrus[ordering[i]]!.removeFirst()
        }
    }

    public let maxSize: Int
    public let maxBytes: Int

    private let trie = PromptTrie<Model, CacheEntry>()
    private let lru = CacheOrder()
    private var totalBytes = 0
    private var bytesByType: [String: Int]

    public init(maxSize: Int = 10, maxBytes: Int = Int.max) {
        self.maxSize = maxSize
        self.maxBytes = maxBytes
        self.bytesByType = Dictionary(uniqueKeysWithValues: lru.ordering.map { ($0, 0) })
    }

    /// Number of stored sequences.
    public var count: Int { lru.count }

    /// Total bytes held across all stored caches.
    public var nbytes: Int { totalBytes }

    /// Fetch the nearest reusable cache for `tokens`, returning a deep copy plus
    /// the token remainder the caller still needs to process. Returns `(nil,
    /// tokens)` on a total miss.
    ///
    /// Every hit refreshes the matched entry's LRU recency (upstream fix ml-explore/mlx-lm#1496).
    public func fetchNearestCache(model: Model, tokens: [Int]) -> (
        cache: [KVCache]?, remainder: [Int]
    ) {
        let result = trie.search(model: model, tokens: tokens)

        if let exact = result.exact {
            let entry = trie.get(model: result.model, tokens: exact)
            refreshRecency(model: result.model, tokens: exact, cacheType: entry.cacheType)
            return (entry.promptCache.map { $0.copy() }, [])
        }

        let shortLength = result.shorter?.count ?? 0
        if let longer = result.longer, result.commonPrefix > shortLength {
            let entry = trie.get(model: result.model, tokens: longer)
            if canTrimPromptCache(entry.promptCache) {
                let cache = entry.promptCache.map { $0.copy() }
                let prefix = min(tokens.count - 1, result.commonPrefix)
                let numToTrim = longer.count - prefix
                trimPromptCache(cache, numTokens: numToTrim)
                refreshRecency(model: result.model, tokens: longer, cacheType: entry.cacheType)
                return (cache, Array(tokens[prefix...]))
            }
        }

        if shortLength > 0, let shorter = result.shorter {
            let entry = trie.get(model: result.model, tokens: shorter)
            refreshRecency(model: result.model, tokens: shorter, cacheType: entry.cacheType)
            return (entry.promptCache.map { $0.copy() }, Array(tokens[shortLength...]))
        }

        return (nil, tokens)
    }

    /// Insert `cache` at `tokens` for `model`, updating byte accounting, purging
    /// now-redundant prefixes of a trimmable cache, and evicting to satisfy the
    /// size/byte limits.
    public func insertCache(
        model: Model, tokens: [Int], cache: [KVCache], cacheType: String = "assistant"
    ) {
        let entry = CacheEntry(
            promptCache: cache,
            nbytes: cache.reduce(0) { $0 + $1.nbytes },
            cacheType: cacheType)

        totalBytes += entry.nbytes
        bytesByType[cacheType, default: 0] += entry.nbytes
        if let prev = trie.add(model: model, tokens: tokens, value: entry) {
            totalBytes -= prev.nbytes
            bytesByType[prev.cacheType, default: 0] -= prev.nbytes
            lru.remove(model: model, tokens: tokens)
        }
        lru.push(model: model, tokens: tokens, cacheType: cacheType)

        // A trimmable cache subsumes its own prefixes, so those just take space.
        if canTrimPromptCache(cache) {
            for (prefixLen, purged) in trie.popPrefixes(model: model, tokens: tokens) {
                totalBytes -= purged.nbytes
                bytesByType[purged.cacheType, default: 0] -= purged.nbytes
                lru.remove(model: model, tokens: Array(tokens[0 ..< prefixLen]))
            }
        }

        // One insert adds at most one net entry, so a single size-eviction restores it.
        if lru.count > maxSize {
            evictOne()
        }
        while totalBytes > maxBytes {
            evictOne()
        }
    }

    /// Trim down to at most `sequences` entries and/or `bytes` total bytes.
    /// A `nil` bound means "unbounded" for that dimension.
    public func trimTo(sequences: Int? = nil, bytes: Int? = nil) {
        let nSequences = sequences.map { max(0, $0) } ?? Int.max
        let nBytes = bytes.map { max(0, $0) } ?? Int.max
        while lru.count > nSequences {
            evictOne()
        }
        while totalBytes > nBytes {
            evictOne()
        }
    }

    /// Per-type `(sequences, bytes)` breakdown.
    public func statsByType() -> [String: TypeStats] {
        var result: [String: TypeStats] = [:]
        for cacheType in lru.ordering {
            result[cacheType] = TypeStats(
                sequences: lru.count(ofType: cacheType),
                bytes: bytesByType[cacheType] ?? 0)
        }
        return result
    }

    private func refreshRecency(model: Model, tokens: [Int], cacheType: String) {
        lru.remove(model: model, tokens: tokens)
        lru.push(model: model, tokens: tokens, cacheType: cacheType)
    }

    /// Evict a single least-recently-used entry and update byte accounting.
    @discardableResult
    private func evictOne() -> CacheEntry {
        let (model, tokens) = lru.pop()
        let popped = trie.pop(model: model, tokens: tokens)
        totalBytes -= popped.nbytes
        bytesByType[popped.cacheType, default: 0] -= popped.nbytes
        return popped
    }
}
