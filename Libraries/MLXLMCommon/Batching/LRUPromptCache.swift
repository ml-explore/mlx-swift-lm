// Copyright © 2024 Apple Inc.

import Foundation
import MLX

// MARK: - LRUPromptCache

/// Trie-based LRU cache storing KV caches keyed by token sequences.
///
/// Ported from Python mlx-lm's `LRUPromptCache`. Supports exact, shorter-prefix,
/// and longer-prefix lookups. Fetch always returns a deep copy (independent of
/// stored cache). Model isolation ensures caches from different models don't
/// cross-contaminate.
///
/// Thread safety is ensured via `NSLock`-based serialization.
///
/// Key operations:
/// - `insertCache(model:tokens:promptCache:)` — store a KV cache for a token sequence
/// - `fetchNearestCache(model:tokens:)` — find the best matching cached prefix
/// - `trimTo(nSequences:nBytes:)` — memory-aware eviction
public final class LRUPromptCache: @unchecked Sendable {

    // MARK: - Types

    /// A single entry stored at a trie leaf.
    final class CacheEntry {
        let promptCache: [KVCache]
        let nbytes: Int

        init(promptCache: [KVCache], nbytes: Int) {
            self.promptCache = promptCache
            self.nbytes = nbytes
        }
    }

    /// A node in the trie. Children are keyed by token ID.
    final class TrieNode {
        var children: [Int32: TrieNode] = [:]
        var cache: CacheEntry?
    }

    /// LRU order tracking with support for checkpoint vs regular entries.
    final class CacheOrder {
        /// Regular LRU entries (most-recently-used at the back).
        private var lru: [(model: String, tokens: [Int])] = []
        /// Checkpoint LRU entries (most-recently-used at the back).
        private var lruCheckpoints: [(model: String, tokens: [Int])] = []

        var count: Int { lru.count + lruCheckpoints.count }

        func push(model: String, tokens: [Int], checkpoint: Bool = false) {
            if checkpoint {
                lruCheckpoints.append((model, tokens))
            } else {
                lru.append((model, tokens))
            }
        }

        func remove(model: String, tokens: [Int]) {
            if let idx = lru.firstIndex(where: { $0.model == model && $0.tokens == tokens }) {
                lru.remove(at: idx)
            } else if let idx = lruCheckpoints.firstIndex(where: {
                $0.model == model && $0.tokens == tokens
            }) {
                lruCheckpoints.remove(at: idx)
            }
        }

        /// Pop the least-recently-used entry. Pops from the longer list first
        /// (matching the Python behavior which pops from whichever deque is longer).
        func pop() -> (model: String, tokens: [Int])? {
            if lru.count >= lruCheckpoints.count {
                return lru.isEmpty ? nil : lru.removeFirst()
            } else {
                return lruCheckpoints.isEmpty ? nil : lruCheckpoints.removeFirst()
            }
        }
    }

    /// Result of a trie search.
    private struct SearchResult {
        let model: String
        /// Non-nil if an exact match was found.
        let exact: [Int]?
        /// Non-nil if a shorter prefix with a cached entry was found.
        let shorter: [Int]?
        /// Non-nil if a longer cached entry reachable from the query's path was found.
        let longer: [Int]?
        /// How many tokens of the query matched trie edges (may exceed cached depth).
        let commonPrefix: Int
    }

    // MARK: - Properties

    /// Maximum number of cached entries.
    public let maxSize: Int

    /// Maximum total bytes across all cached entries.
    public let maxBytes: Int

    /// Root trie nodes keyed by model identifier.
    private var cache: [String: TrieNode] = [:]

    /// LRU order tracker.
    private let lru = CacheOrder()

    /// Total byte size of all cached entries.
    private var _nBytes: Int = 0

    /// Lock for thread safety.
    private let lock = NSLock()

    // MARK: - Initializer

    /// Create a new LRUPromptCache.
    ///
    /// - Parameters:
    ///   - maxSize: Maximum number of cached entries (default: 10).
    ///   - maxBytes: Maximum total bytes across all entries (default: `Int.max`).
    public init(maxSize: Int = 10, maxBytes: Int = Int.max) {
        self.maxSize = maxSize
        self.maxBytes = maxBytes
    }

    // MARK: - Public API

    /// The number of cached entries.
    public var count: Int {
        lock.lock()
        defer { lock.unlock() }
        return lru.count
    }

    /// The total byte size of all cached entries.
    public var nbytes: Int {
        lock.lock()
        defer { lock.unlock() }
        return _nBytes
    }

    /// Fetch the nearest matching KV cache for the given token sequence.
    ///
    /// Returns a deep copy of the matched cache (mutations don't affect stored cache)
    /// and the remainder tokens that still need processing.
    ///
    /// Match priority:
    /// 1. **Exact match** — returns cache with empty remainder.
    /// 2. **Longer prefix** — if a cached entry covers more tokens than the query
    ///    and the cache is trimmable, returns a deep-copied and trimmed cache.
    /// 3. **Shorter prefix** — returns the deepest cached prefix with remainder tokens.
    ///
    /// - Parameters:
    ///   - model: Model identifier for isolation.
    ///   - tokens: The token sequence to look up.
    /// - Returns: A tuple of (cache, remainderTokens). Cache is nil if no match found;
    ///   remainder is the full token array if no match.
    public func fetchNearestCache(model: String, tokens: [Int]) -> ([KVCache]?, [Int]) {
        lock.lock()
        defer { lock.unlock() }
        return _fetchNearestCache(model: model, tokens: tokens)
    }

    /// Insert a KV cache for the given token sequence.
    ///
    /// If the cache is trimmable and a shorter prefix is encountered during insertion,
    /// it is removed (the new, longer cache supersedes it). After insertion, LRU and
    /// memory-based eviction is triggered if limits are exceeded.
    ///
    /// - Parameters:
    ///   - model: Model identifier for isolation.
    ///   - tokens: The token sequence this cache covers.
    ///   - promptCache: The KV cache layers to store.
    ///   - checkpoint: Whether this is a checkpoint entry (affects eviction priority).
    public func insertCache(
        model: String, tokens: [Int], promptCache: [KVCache], checkpoint: Bool = false
    ) {
        lock.lock()
        defer { lock.unlock() }
        _insertCache(model: model, tokens: tokens, promptCache: promptCache, checkpoint: checkpoint)
    }

    /// Evict entries until the cache is within the given limits.
    ///
    /// - Parameters:
    ///   - nSequences: Maximum number of entries to keep (nil = no limit).
    ///   - nBytes: Maximum total bytes to keep (nil = no limit).
    public func trimTo(nSequences: Int? = nil, nBytes: Int? = nil) {
        lock.lock()
        defer { lock.unlock() }

        let seqLimit = nSequences.map { max(0, $0) } ?? Int.max
        let byteLimit = nBytes.map { max(0, $0) } ?? Int.max

        while lru.count > seqLimit {
            guard let evicted = lru.pop() else { break }
            _delete(model: evicted.model, tokens: evicted.tokens)
        }
        while _nBytes > byteLimit {
            guard let evicted = lru.pop() else { break }
            _delete(model: evicted.model, tokens: evicted.tokens)
        }
    }

    // MARK: - Private Implementation

    /// Search the trie for the best match.
    private func _search(model: String, tokens: [Int]) -> SearchResult {
        guard let root = cache[model] else {
            return SearchResult(
                model: model, exact: nil, shorter: nil, longer: nil, commonPrefix: 0)
        }

        var current = root
        var lastCacheIndex = -1
        var index = 0

        while index < tokens.count, let next = current.children[Int32(tokens[index])] {
            current = next
            if current.cache != nil {
                lastCacheIndex = index
            }
            index += 1
        }

        // Exact match: the deepest cached node is at the last token
        if lastCacheIndex == tokens.count - 1 {
            return SearchResult(
                model: model, exact: tokens, shorter: nil, longer: nil, commonPrefix: 0)
        }

        // Shorter prefix
        var shorter: [Int]?
        if lastCacheIndex >= 0 {
            shorter = Array(tokens[...lastCacheIndex])
        }

        // Longer prefix: search for the shortest cached descendant from `current`
        var longer: [Int]?
        let commonPrefix = index
        if index > 0 {
            var best: [Int]?
            var stack: [(node: TrieNode, extra: [Int])] = [(current, [])]
            while !stack.isEmpty {
                let (node, extra) = stack.removeLast()
                if node.cache != nil {
                    if best == nil || extra.count < best!.count {
                        best = extra
                    }
                } else {
                    for (tok, child) in node.children {
                        stack.append((child, extra + [Int(tok)]))
                    }
                }
            }
            if let best {
                longer = Array(tokens[..<index]) + best
            }
        }

        return SearchResult(
            model: model, exact: nil, shorter: shorter, longer: longer,
            commonPrefix: commonPrefix)
    }

    /// Get the cache entry at the given path.
    private func _get(model: String, tokens: [Int]) -> CacheEntry {
        var current = cache[model]!
        for tok in tokens {
            current = current.children[Int32(tok)]!
        }
        return current.cache!
    }

    /// Delete a cache entry from the trie.
    private func _delete(model: String, tokens: [Int]) {
        guard let root = cache[model] else { return }

        var path = [root]
        for tok in tokens {
            guard let next = path.last!.children[Int32(tok)] else { return }
            path.append(next)
        }

        guard let entry = path.last?.cache else { return }
        _nBytes -= entry.nbytes
        path.last!.cache = nil

        // Clean up empty nodes from the bottom
        for i in stride(from: tokens.count - 1, through: 0, by: -1) {
            let child = path[i + 1]
            if child.children.isEmpty && child.cache == nil {
                path[i].children.removeValue(forKey: Int32(tokens[i]))
            } else {
                break
            }
        }
    }

    /// Deep-copy a KV cache by reading and writing its state.
    private func _deepCopy(_ promptCache: [KVCache]) -> [KVCache] {
        promptCache.map { original in
            var copy: KVCache
            if original is KVCacheSimple {
                copy = KVCacheSimple()
            } else if let rotating = original as? RotatingKVCache {
                copy = RotatingKVCache(maxSize: rotating.maxSize ?? 0)
            } else {
                // Fallback: KVCacheSimple for unknown types
                copy = KVCacheSimple()
            }
            let originalState = original.state
            // Only restore state if the cache has data (non-empty state).
            // Empty state means keys/values are nil (e.g., mock model didn't
            // populate the cache), and setting empty state would crash.
            if !originalState.isEmpty {
                copy.state = originalState
            }
            copy.metaState = original.metaState
            return copy
        }
    }

    /// Refresh LRU recency for the given entry (move to most-recently-used).
    private func _touch(model: String, tokens: [Int]) {
        lru.remove(model: model, tokens: tokens)
        lru.push(model: model, tokens: tokens)
    }

    /// Internal fetch without locking.
    private func _fetchNearestCache(model: String, tokens: [Int]) -> ([KVCache]?, [Int]) {
        let result = _search(model: model, tokens: tokens)

        // Exact match
        if let exact = result.exact {
            let entry = _get(model: result.model, tokens: exact)
            _touch(model: result.model, tokens: exact)
            return (_deepCopy(entry.promptCache), [])
        }

        let shortLength = result.shorter?.count ?? 0

        // Longer prefix: if the cached entry is longer than the query and trimmable
        if let longer = result.longer, result.commonPrefix > shortLength {
            let entry = _get(model: result.model, tokens: longer)
            if canTrimPromptCache(entry.promptCache) {
                let copy = _deepCopy(entry.promptCache)
                let prefix = min(tokens.count, result.commonPrefix)
                let numToTrim = longer.count - prefix
                trimPromptCache(copy, numTokens: numToTrim)
                let remainder = prefix < tokens.count ? Array(tokens[prefix...]) : []
                _touch(model: result.model, tokens: longer)
                return (copy, remainder)
            }
        }

        // Shorter prefix
        if shortLength > 0 {
            let entry = _get(model: result.model, tokens: result.shorter!)
            _touch(model: result.model, tokens: result.shorter!)
            return (_deepCopy(entry.promptCache), Array(tokens[shortLength...]))
        }

        // No match
        return (nil, tokens)
    }

    /// Internal insert without locking.
    private func _insertCache(
        model: String, tokens: [Int], promptCache: [KVCache], checkpoint: Bool
    ) {
        let isTrimmable = canTrimPromptCache(promptCache)

        if cache[model] == nil {
            cache[model] = TrieNode()
        }
        var current = cache[model]!

        for i in 0 ..< tokens.count {
            let tok = Int32(tokens[i])
            if current.children[tok] == nil {
                current.children[tok] = TrieNode()
            }
            // If inserting a trimmable cache and we pass through an existing cached node,
            // remove it (the new longer cache supersedes the shorter one).
            if isTrimmable, current.cache != nil {
                _nBytes -= current.cache!.nbytes
                current.cache = nil
                lru.remove(model: model, tokens: Array(tokens[..<i]))
            }
            current = current.children[tok]!
        }

        if current.cache != nil {
            // Update existing entry: remove from LRU and reinsert
            lru.remove(model: model, tokens: tokens)
        } else {
            let cacheBytes = promptCache.reduce(0) { $0 + $1.state.reduce(0) { $0 + $1.nbytes } }
            current.cache = CacheEntry(promptCache: promptCache, nbytes: cacheBytes)
            _nBytes += cacheBytes
        }

        lru.push(model: model, tokens: tokens, checkpoint: checkpoint)

        // Evict if over maxSize
        if lru.count > maxSize {
            if let evicted = lru.pop() {
                _delete(model: evicted.model, tokens: evicted.tokens)
            }
        }

        // Evict if over maxBytes
        while _nBytes > maxBytes {
            guard let evicted = lru.pop() else { break }
            _delete(model: evicted.model, tokens: evicted.tokens)
        }
    }
}
