// Copyright © 2025 Apple Inc.

import Foundation

/// Cross-request LRU prompt cache: keeps recently-used KV caches, with the model state that
/// belongs with them, keyed by `(model, tokens)` and, for a new prompt, returns the nearest
/// reusable cache plus the token remainder still to be processed.
///
/// Port of Python `mlx_lm`'s `LRUPromptCache` (`mlx_lm/models/cache.py`).
///
/// `Model` identifies what produced a cache; entries for different models never match. A
/// Hugging Face model id (`String`) is typical. Include anything else that changes the
/// cache contents for the same tokens, such as an adapter or a KV-cache configuration.
///
/// `tokens` are the model's input token ids (`LMInput.text.tokens`). The store is text-only:
/// those ids do not identify image, video, or audio content, so two inputs with different media
/// but the same placeholder tokens are indistinguishable. Do not insert or fetch entries whose
/// input carried media.
///
/// Caches are copied on insert and on fetch, so a stored entry never shares a ``KVCache``
/// instance with a caller. Model state (``LMOutput/State``) is stored and returned with its cache.
/// An entry that carries state is never trimmed, since the state is anchored to the position it
/// was captured at; it is reused only when the new prompt extends its tokens.
///
/// Eviction is type-aware: entries are evicted `tool` → `assistant` → `user` → `system`, and the
/// eviction policy sheds the most-populous higher-churn class first, so system prompts survive
/// longest.
///
/// This is a plain non-`Sendable` `final class`, mirroring Python's
/// single-server-thread assumption. Callers that share an instance across tasks
/// must serialize access to it.
public final class LRUPromptCache<Model: Hashable> {

    /// Per-type sequence statistics returned by ``statsByType()``.
    public struct TypeStats {
        public let sequences: Int
        public let bytes: Int
    }

    public typealias Role = Chat.Message.Role

    private struct CacheEntry {
        let cache: [KVCache]
        let state: LMOutput.State?
        let nbytes: Int
        let role: Role

        func snapshot() -> PromptCacheSnapshot {
            PromptCacheSnapshot(cache: cache.map { $0.copy() }, state: state)
        }
    }

    /// Role-partitioned recency lists. Newest is appended; oldest is `first`.
    private final class CacheOrder {
        static var evictionOrder: [Role] { [.tool, .assistant, .user, .system] }

        var lrus: [Role: [(Model, [Int])]] = [:]

        var count: Int { lrus.values.reduce(0) { $0 + $1.count } }

        func count(of role: Role) -> Int { lrus[role]?.count ?? 0 }

        func push(model: Model, tokens: [Int], role: Role) {
            lrus[role, default: []].append((model, tokens))
        }

        func remove(model: Model, tokens: [Int], role: Role) {
            if let idx = lrus[role]?.firstIndex(where: { $0.0 == model && $0.1 == tokens }) {
                lrus[role]!.remove(at: idx)
            }
        }

        /// Evict the least-recently-used entry from the most-populous
        /// higher-churn class, in ``evictionOrder``.
        func pop() -> (Model, [Int]) {
            let order = Self.evictionOrder
            for (role, next) in zip(order, order.dropFirst()) {
                let n = count(of: role)
                if n > 0 && n >= count(of: next) {
                    return lrus[role]!.removeFirst()
                }
            }
            return lrus[order.last!]!.removeFirst()
        }
    }

    public let maxSize: Int
    public let maxBytes: Int

    private let trie = PromptTrie<Model, CacheEntry>()
    private let lru = CacheOrder()
    private var totalBytes = 0
    private var bytesByRole: [Role: Int] = [:]

    public init(maxSize: Int = 10, maxBytes: Int = Int.max) {
        self.maxSize = maxSize
        self.maxBytes = maxBytes
    }

    /// Number of stored sequences.
    public var count: Int { lru.count }

    /// Total bytes held across all stored caches.
    public var nbytes: Int { totalBytes }

    /// Fetch the nearest reusable cache for `tokens`, returning a copy of the cache and its model
    /// state plus the token remainder the caller still needs to process. Returns `(nil, tokens)`
    /// on a total miss.
    ///
    /// The remainder is never empty for a non-empty `tokens`: an exact match is returned trimmed
    /// by one token, or, when that entry cannot be trimmed, the nearest shorter entry is used.
    /// An empty `tokens` always misses.
    ///
    /// Every hit refreshes the matched entry's LRU recency.
    public func fetchNearestCache(model: Model, tokens: [Int]) -> (
        snapshot: PromptCacheSnapshot?, remainder: [Int]
    ) {
        guard !tokens.isEmpty else { return (nil, tokens) }
        return fetchNearest(model: model, tokens: tokens, keepingOneToken: true)
    }

    /// With `keepingOneToken`, the returned remainder is non-empty; without it, `tokens` may be
    /// matched in full.
    private func fetchNearest(model: Model, tokens: [Int], keepingOneToken: Bool) -> (
        snapshot: PromptCacheSnapshot?, remainder: [Int]
    ) {
        guard let last = tokens.last else { return (nil, tokens) }
        let result = trie.search(model: model, tokens: tokens)

        if let exact = result.exact {
            let entry = trie.get(model: result.model, tokens: exact)
            if !keepingOneToken {
                refreshRecency(model: result.model, tokens: exact, role: entry.role)
                return (entry.snapshot(), [])
            }
            guard isTrimmable(entry) else {
                let fallback = fetchNearest(
                    model: model, tokens: Array(tokens.dropLast()), keepingOneToken: false)
                return (fallback.snapshot, fallback.remainder + [last])
            }
            let snapshot = entry.snapshot()
            trimPromptCache(snapshot.cache, numTokens: 1)
            refreshRecency(model: result.model, tokens: exact, role: entry.role)
            return (snapshot, [last])
        }

        let shortLength = result.shorter?.count ?? 0
        if let longer = result.longer, result.commonPrefix > shortLength {
            let entry = trie.get(model: result.model, tokens: longer)
            if isTrimmable(entry) {
                let snapshot = entry.snapshot()
                let prefix = min(tokens.count - (keepingOneToken ? 1 : 0), result.commonPrefix)
                trimPromptCache(snapshot.cache, numTokens: longer.count - prefix)
                refreshRecency(model: result.model, tokens: longer, role: entry.role)
                return (snapshot, Array(tokens[prefix...]))
            }
        }

        if shortLength > 0, let shorter = result.shorter {
            let entry = trie.get(model: result.model, tokens: shorter)
            refreshRecency(model: result.model, tokens: shorter, role: entry.role)
            return (entry.snapshot(), Array(tokens[shortLength...]))
        }

        return (nil, tokens)
    }

    /// Insert a copy of `cache`, with the model `state` produced alongside it, at `tokens` for
    /// `model`, updating byte accounting, purging now-redundant prefixes of a trimmable cache,
    /// and evicting to satisfy the size/byte limits.
    ///
    /// - Parameters:
    ///   - model: the model that produced `cache`.
    ///   - tokens: the input token ids `cache` holds.
    ///   - cache: a cache holding exactly `tokens`, i.e. at offset `tokens.count`.
    ///   - state: the model state that belongs with `cache`, e.g. ``TokenIteratorProtocol/state``.
    ///     Pass it whenever the model produced one: models that need it to continue a warm
    ///     cache throw ``ContinuationStateError`` when it is missing.
    ///   - role: the role of the last message `tokens` covers; it selects the eviction class.
    public func insertCache(
        model: Model, tokens: [Int], cache: [KVCache], state: LMOutput.State? = nil,
        role: Role = .assistant
    ) {
        let copied = cache.map { $0.copy() }
        let entry = CacheEntry(
            cache: copied,
            state: state,
            nbytes: copied.reduce(0) { $0 + $1.nbytes },
            role: role)

        totalBytes += entry.nbytes
        bytesByRole[role, default: 0] += entry.nbytes
        if let prev = trie.add(model: model, tokens: tokens, value: entry) {
            totalBytes -= prev.nbytes
            bytesByRole[prev.role, default: 0] -= prev.nbytes
            lru.remove(model: model, tokens: tokens, role: prev.role)
        }
        lru.push(model: model, tokens: tokens, role: role)

        // A trimmable cache subsumes its own prefixes, so those just take space.
        if isTrimmable(entry) {
            for (prefixLen, purged) in trie.popPrefixes(model: model, tokens: tokens) {
                totalBytes -= purged.nbytes
                bytesByRole[purged.role, default: 0] -= purged.nbytes
                lru.remove(model: model, tokens: Array(tokens[0 ..< prefixLen]), role: purged.role)
            }
        }

        // One insert adds at most one net entry, so a single size-eviction restores it.
        if lru.count > maxSize {
            evictOne()
        }
        while totalBytes > maxBytes && lru.count > 0 {
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

    /// Per-role `(sequences, bytes)` breakdown.
    public func statsByType() -> [Role: TypeStats] {
        var result: [Role: TypeStats] = [:]
        for role in CacheOrder.evictionOrder {
            result[role] = TypeStats(
                sequences: lru.count(of: role), bytes: bytesByRole[role] ?? 0)
        }
        return result
    }

    private func isTrimmable(_ entry: CacheEntry) -> Bool {
        entry.state == nil && canTrimPromptCache(entry.cache)
    }

    private func refreshRecency(model: Model, tokens: [Int], role: Role) {
        lru.remove(model: model, tokens: tokens, role: role)
        lru.push(model: model, tokens: tokens, role: role)
    }

    /// Evict a single least-recently-used entry and update byte accounting.
    private func evictOne() {
        let (model, tokens) = lru.pop()
        let popped = trie.pop(model: model, tokens: tokens)
        totalBytes -= popped.nbytes
        bytesByRole[popped.role, default: 0] -= popped.nbytes
    }
}
