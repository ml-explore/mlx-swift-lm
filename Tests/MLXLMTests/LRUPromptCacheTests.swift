// Copyright © 2025 Apple Inc.

import Foundation
import MLX
import Testing

@testable import MLXLMCommon

/// Swift analogue of the Python `MockCache` in `mlx_lm`'s `test_server.py`:
/// a `BaseKVCache` whose `nbytes` is the value length and whose `copy()`
/// preserves the value (so a fetched copy compares equal). Overriding `nbytes`
/// exercises the `open var` dispatch path added on `BaseKVCache`.
private final class MockCache: BaseKVCache {
    let value: String
    private let trimmable: Bool

    init(_ value: String, isTrimmable: Bool = true) {
        self.value = value
        self.trimmable = isTrimmable
        super.init()
    }

    override var nbytes: Int { value.utf8.count }
    override var isTrimmable: Bool { trimmable }

    @discardableResult
    override func trim(_ n: Int) -> Int { n }

    override func copy() -> any KVCache { MockCache(value, isTrimmable: trimmable) }
}

@Suite("LRUPromptCache")
struct LRUPromptCacheTests {

    private func mockValue(_ c: [KVCache]?) -> String? {
        (c?.first as? MockCache)?.value
    }

    private func flattenEqualsInts(_ a: MLXArray, _ expected: [Int]) -> Bool {
        let flat = a.reshaped([-1]).asType(.int32)
        guard flat.size == expected.count else { return false }
        let e = MLXArray(expected.map { Int32($0) })
        return (flat .== e).all().item(Bool.self)
    }

    // MARK: - test_caching (real KV caches)

    @Test func caching() {
        let cache = LRUPromptCache<String>(maxSize: 10)

        func getKV(_ n: Int) -> (MLXArray, MLXArray) {
            let keys = MLXArray((0 ..< n).map { Int32($0) }).reshaped([1, 1, n, 1])
            return (keys, keys)
        }

        let model = "test"
        var tokens = Array(repeating: 10, count: 24)

        let (c0, t0) = cache.fetchNearestCache(model: model, tokens: tokens)
        #expect(c0 == nil)
        #expect(t0 == tokens)

        let c = KVCacheSimple()
        let kv24 = getKV(24)
        _ = c.update(keys: kv24.0, values: kv24.1)
        cache.insertCache(model: model, tokens: t0, cache: [c])

        // Fetching a strict prefix does not evict it.
        tokens = tokens + Array(repeating: 20, count: 5)
        let (c1, t1) = cache.fetchNearestCache(model: model, tokens: tokens)
        let s1 = c1![0].state
        #expect(flattenEqualsInts(s1[0], Array(0 ..< 24)))
        #expect(t1 == Array(repeating: 20, count: 5))
        #expect(cache.count == 1)

        // Inserting a trimmable cache with a shared prefix purges the prefix.
        tokens = tokens + Array(repeating: 30, count: 3)
        let kv8a = getKV(8)
        _ = c1![0].update(keys: kv8a.0, values: kv8a.1)
        cache.insertCache(model: model, tokens: tokens, cache: c1!)
        #expect(cache.count == 1)

        // Fetching a shared-prefix (diverging) query returns a trimmed copy.
        tokens = Array(tokens[0 ..< 26]) + Array(repeating: 40, count: 8)
        let (c2, t2) = cache.fetchNearestCache(model: model, tokens: tokens)
        let s2 = c2![0].state
        #expect(flattenEqualsInts(s2[0], Array(0 ..< 24) + [0, 1]))
        #expect(t2 == Array(repeating: 40, count: 8))
        #expect(cache.count == 1)

        // Inserting a diverged cache creates a second entry.
        let kv8b = getKV(8)
        _ = c2![0].update(keys: kv8b.0, values: kv8b.1)
        cache.insertCache(model: model, tokens: tokens, cache: c2!)
        #expect(cache.count == 2)
    }

    // MARK: - test_lru (type-aware eviction)

    @Test func lru() {
        let cache = LRUPromptCache<String>(maxSize: 2)
        let model = "test"
        cache.insertCache(model: model, tokens: [1, 2], cache: [MockCache("test1")])
        cache.insertCache(model: model, tokens: [2, 3], cache: [MockCache("test2")])

        #expect(mockValue(cache.fetchNearestCache(model: model, tokens: [1, 2]).cache) == "test1")
        #expect(cache.fetchNearestCache(model: model, tokens: [1, 2]).remainder == [])

        var r = cache.fetchNearestCache(model: model, tokens: [1])
        #expect(mockValue(r.cache) == "test1")
        #expect(r.remainder == [1])

        r = cache.fetchNearestCache(model: model, tokens: [1, 3, 4])
        #expect(mockValue(r.cache) == "test1")
        #expect(r.remainder == [3, 4])

        r = cache.fetchNearestCache(model: model, tokens: [2, 3, 4])
        #expect(mockValue(r.cache) == "test2")
        #expect(r.remainder == [4])

        r = cache.fetchNearestCache(model: model, tokens: [2, 4, 5])
        #expect(mockValue(r.cache) == "test2")
        #expect(r.remainder == [4, 5])

        cache.insertCache(model: model, tokens: [1, 2], cache: [MockCache("test1")])
        cache.insertCache(model: model, tokens: [2, 3], cache: [MockCache("test2")])
        cache.insertCache(model: model, tokens: [3, 4], cache: [MockCache("test3")])

        #expect(cache.fetchNearestCache(model: model, tokens: [1, 2]).cache == nil)
        #expect(mockValue(cache.fetchNearestCache(model: model, tokens: [2, 3]).cache) == "test2")
        #expect(mockValue(cache.fetchNearestCache(model: model, tokens: [3, 4]).cache) == "test3")

        cache.insertCache(
            model: model, tokens: [4, 5], cache: [MockCache("test4")], cacheType: "user")
        #expect(cache.fetchNearestCache(model: model, tokens: [2, 3]).cache == nil)
        #expect(mockValue(cache.fetchNearestCache(model: model, tokens: [3, 4]).cache) == "test3")
        #expect(mockValue(cache.fetchNearestCache(model: model, tokens: [4, 5]).cache) == "test4")

        cache.insertCache(model: model, tokens: [5, 6], cache: [MockCache("test5")])
        cache.insertCache(model: model, tokens: [6, 7], cache: [MockCache("test6")])
        #expect(cache.fetchNearestCache(model: model, tokens: [5, 6]).cache == nil)
        #expect(mockValue(cache.fetchNearestCache(model: model, tokens: [6, 7]).cache) == "test6")
        // The `user`-typed entry survives longest.
        #expect(mockValue(cache.fetchNearestCache(model: model, tokens: [4, 5]).cache) == "test4")
    }

    // MARK: - prefix / empty-token behaviors

    @Test func insertTrimmableCacheRemovesImmediatePrefix() {
        let cache = LRUPromptCache<String>(maxSize: 10)
        let model = "test"

        cache.insertCache(model: model, tokens: [1, 2], cache: [MockCache("ab")])
        #expect(cache.count == 1)
        #expect(cache.nbytes == 2)

        cache.insertCache(model: model, tokens: [1, 2, 3], cache: [MockCache("abc")])
        #expect(cache.count == 1)
        #expect(cache.nbytes == 3)
    }

    @Test func insertEmptyTokensDoesNotSelfDestruct() {
        let cache = LRUPromptCache<String>(maxSize: 10)
        let model = "test"

        cache.insertCache(model: model, tokens: [], cache: [MockCache("root")])
        #expect(cache.count == 1)
        #expect(cache.nbytes == 4)

        let (c, t) = cache.fetchNearestCache(model: model, tokens: [])
        #expect(c != nil)
        #expect(t == [])
    }

    @Test func fetchEmptyTokensAfterRootEviction() {
        let cache = LRUPromptCache<String>(maxSize: 10)
        let model = "test"

        cache.insertCache(model: model, tokens: [], cache: [MockCache("root")])
        cache.insertCache(model: model, tokens: [1], cache: [MockCache("a")])

        let (c, t) = cache.fetchNearestCache(model: model, tokens: [])
        #expect(c == nil)
        #expect(t == [])
    }

    @Test func lruBytes() {
        let cache = LRUPromptCache<String>(maxSize: 100, maxBytes: 10)
        let model = "test"

        cache.insertCache(model: model, tokens: [1, 2], cache: [MockCache("aaa")])
        cache.insertCache(model: model, tokens: [3, 4], cache: [MockCache("bbb")])
        cache.insertCache(model: model, tokens: [4, 5], cache: [MockCache("ccc")])
        cache.insertCache(model: model, tokens: [6, 7], cache: [MockCache("ddd")])

        #expect(cache.count == 3)
        #expect(cache.nbytes == 9)

        cache.trimTo(bytes: 7)
        #expect(cache.count == 2)
        #expect(cache.nbytes == 6)

        #expect(cache.fetchNearestCache(model: model, tokens: [1, 2]).cache == nil)
        #expect(cache.fetchNearestCache(model: model, tokens: [3, 4]).cache == nil)
    }

    // MARK: - Regression: upstream mlx-lm #1496

    /// Fetching a cache must refresh its LRU recency (upstream ships FIFO). With
    /// A, B inserted then A fetched, inserting C past `maxSize` must evict B (the
    /// truly least-recently-used) and keep A. Reverting the recency refresh in
    /// `fetchNearestCache` evicts A instead, failing the first expectation.
    @Test func regressionFetchRefreshesRecency_1496() {
        let cache = LRUPromptCache<String>(maxSize: 2)
        let model = "m"
        cache.insertCache(model: model, tokens: [1, 1], cache: [MockCache("A")])
        cache.insertCache(model: model, tokens: [2, 2], cache: [MockCache("B")])

        // Touch A so it becomes most-recently-used.
        #expect(mockValue(cache.fetchNearestCache(model: model, tokens: [1, 1]).cache) == "A")

        cache.insertCache(model: model, tokens: [3, 3], cache: [MockCache("C")])

        #expect(mockValue(cache.fetchNearestCache(model: model, tokens: [1, 1]).cache) == "A")
        #expect(cache.fetchNearestCache(model: model, tokens: [2, 2]).cache == nil)
        #expect(mockValue(cache.fetchNearestCache(model: model, tokens: [3, 3]).cache) == "C")
    }
}
