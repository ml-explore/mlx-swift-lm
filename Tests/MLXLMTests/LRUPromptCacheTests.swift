// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import XCTest

@testable import MLXLMCommon

// MARK: - LRUPromptCacheTests

final class LRUPromptCacheTests: XCTestCase {

    // MARK: - Helpers

    /// Create a mock KVCacheSimple with a given number of tokens.
    /// The cache will report `offset == seqLen` and hold synthetic keys/values.
    private func makeMockCache(seqLen: Int, heads: Int = 2, headDim: Int = 4) -> KVCacheSimple {
        let cache = KVCacheSimple()
        if seqLen > 0 {
            let keys = MLXArray.ones([1, heads, seqLen, headDim])
            let values = MLXArray.ones([1, heads, seqLen, headDim])
            _ = cache.update(keys: keys, values: values)
        }
        return cache
    }

    /// Create a multi-layer mock prompt cache (array of KVCacheSimple).
    private func makeMockPromptCache(
        layers: Int = 2, seqLen: Int, heads: Int = 2, headDim: Int = 4
    ) -> [KVCache] {
        (0 ..< layers).map { _ in makeMockCache(seqLen: seqLen, heads: heads, headDim: headDim) }
    }

    // MARK: - VAL-PCACHE-001: Empty cache returns nil

    func testEmptyCacheReturnsNil() throws {
        try skipIfMetalUnavailable()

        let cache = LRUPromptCache(maxSize: 10)
        let (result, remainder) = cache.fetchNearestCache(model: "model1", tokens: [1, 2, 3])

        XCTAssertNil(result, "Empty cache should return nil")
        XCTAssertEqual(remainder, [1, 2, 3], "Remainder should be the full token array")
    }

    // MARK: - VAL-PCACHE-002: Single insertion and exact retrieval

    func testSingleInsertionExactRetrieval() throws {
        try skipIfMetalUnavailable()

        let cache = LRUPromptCache(maxSize: 10)
        let promptCache = makeMockPromptCache(seqLen: 3)

        cache.insertCache(model: "model1", tokens: [1, 2, 3], promptCache: promptCache)

        let (result, remainder) = cache.fetchNearestCache(model: "model1", tokens: [1, 2, 3])

        XCTAssertNotNil(result, "Should find exact match")
        XCTAssertEqual(result!.count, 2, "Should have 2 layers")
        XCTAssertEqual(remainder, [], "Exact match should have empty remainder")
    }

    // MARK: - VAL-PCACHE-003: Shorter prefix match returns cached prefix and remainder

    func testShorterPrefixMatch() throws {
        try skipIfMetalUnavailable()

        let cache = LRUPromptCache(maxSize: 10)
        let promptCache = makeMockPromptCache(seqLen: 3)

        cache.insertCache(model: "model1", tokens: [1, 2, 3], promptCache: promptCache)

        let (result, remainder) = cache.fetchNearestCache(
            model: "model1", tokens: [1, 2, 3, 4, 5])

        XCTAssertNotNil(result, "Should find shorter prefix match")
        XCTAssertEqual(remainder, [4, 5], "Remainder should be uncached suffix")
    }

    // MARK: - VAL-PCACHE-004: Longest available prefix selected

    func testLongestPrefixSelected() throws {
        try skipIfMetalUnavailable()

        let cache = LRUPromptCache(maxSize: 10)
        let shortCache = makeMockPromptCache(seqLen: 2)
        let longCache = makeMockPromptCache(seqLen: 3)

        cache.insertCache(model: "model1", tokens: [1, 2], promptCache: shortCache)
        cache.insertCache(model: "model1", tokens: [1, 2, 3], promptCache: longCache)

        let (result, remainder) = cache.fetchNearestCache(
            model: "model1", tokens: [1, 2, 3, 4])

        XCTAssertNotNil(result, "Should find longest prefix match")
        XCTAssertEqual(remainder, [4], "Remainder should be [4] (matched [1,2,3])")
    }

    // MARK: - VAL-PCACHE-005: LRU eviction triggered at maxSize

    func testLRUEvictionAtMaxSize() throws {
        try skipIfMetalUnavailable()

        let cache = LRUPromptCache(maxSize: 3)

        // Insert 3 entries
        cache.insertCache(
            model: "model1", tokens: [1], promptCache: makeMockPromptCache(seqLen: 1))
        cache.insertCache(
            model: "model1", tokens: [2], promptCache: makeMockPromptCache(seqLen: 1))
        cache.insertCache(
            model: "model1", tokens: [3], promptCache: makeMockPromptCache(seqLen: 1))
        XCTAssertEqual(cache.count, 3)

        // 4th insertion should evict the least-recently-used (tokens: [1])
        cache.insertCache(
            model: "model1", tokens: [4], promptCache: makeMockPromptCache(seqLen: 1))
        XCTAssertEqual(cache.count, 3, "Should still have maxSize entries after eviction")

        // The oldest entry [1] should be evicted
        let (result1, _) = cache.fetchNearestCache(model: "model1", tokens: [1])
        XCTAssertNil(result1, "Evicted entry should not be found")

        // More recent entries should still be present
        let (result2, _) = cache.fetchNearestCache(model: "model1", tokens: [2])
        XCTAssertNotNil(result2, "Entry [2] should still be present")
        let (result3, _) = cache.fetchNearestCache(model: "model1", tokens: [3])
        XCTAssertNotNil(result3, "Entry [3] should still be present")
        let (result4, _) = cache.fetchNearestCache(model: "model1", tokens: [4])
        XCTAssertNotNil(result4, "Entry [4] should still be present")
    }

    // MARK: - VAL-PCACHE-006: Memory-aware eviction by bytes

    func testMemoryAwareEviction() throws {
        try skipIfMetalUnavailable()

        // Each mock cache with seqLen=1, 2 layers, 2 heads, headDim=4 uses some bytes.
        // We'll insert a few caches and set a maxBytes that triggers eviction.
        let promptCache1 = makeMockPromptCache(seqLen: 5)
        let bytes1 = promptCache1.reduce(0) { $0 + $1.state.reduce(0) { $0 + $1.nbytes } }

        // Set maxBytes just above 2 entries' worth
        let cache = LRUPromptCache(maxSize: 100, maxBytes: bytes1 * 2 + 1)

        cache.insertCache(
            model: "model1", tokens: [1], promptCache: makeMockPromptCache(seqLen: 5))
        cache.insertCache(
            model: "model1", tokens: [2], promptCache: makeMockPromptCache(seqLen: 5))
        XCTAssertEqual(cache.count, 2)

        // 3rd insertion should trigger byte-based eviction
        cache.insertCache(
            model: "model1", tokens: [3], promptCache: makeMockPromptCache(seqLen: 5))

        // At least one entry should have been evicted
        XCTAssertLessThanOrEqual(cache.nbytes, bytes1 * 2 + 1)
    }

    // MARK: - VAL-PCACHE-011: Concurrent access safety

    func testConcurrentAccessSafety() throws {
        try skipIfMetalUnavailable()

        let cache = LRUPromptCache(maxSize: 100)
        let iterations = 50
        let expectation = XCTestExpectation(description: "Concurrent access")
        expectation.expectedFulfillmentCount = iterations * 2

        let queue = DispatchQueue(label: "test.concurrent", attributes: .concurrent)

        // Local helper to avoid capturing `self` in @Sendable closure
        @Sendable func makeCache(seqLen: Int) -> [KVCache] {
            let c = KVCacheSimple()
            if seqLen > 0 {
                let keys = MLXArray.ones([1, 2, seqLen, 4])
                let values = MLXArray.ones([1, 2, seqLen, 4])
                _ = c.update(keys: keys, values: values)
            }
            return [c, KVCacheSimple()]
        }

        // Concurrent inserts
        for i in 0 ..< iterations {
            queue.async {
                let promptCache = makeCache(seqLen: i + 1)
                cache.insertCache(
                    model: "model1", tokens: Array(0 ... i), promptCache: promptCache)
                expectation.fulfill()
            }
        }

        // Concurrent fetches
        for i in 0 ..< iterations {
            queue.async {
                let _ = cache.fetchNearestCache(model: "model1", tokens: Array(0 ... i))
                expectation.fulfill()
            }
        }

        wait(for: [expectation], timeout: 10.0)

        // Verify cache is in a valid state
        XCTAssertGreaterThan(cache.count, 0, "Cache should have entries after concurrent inserts")
    }

    // MARK: - VAL-PCACHE-012: Model isolation

    func testModelIsolation() throws {
        try skipIfMetalUnavailable()

        let cache = LRUPromptCache(maxSize: 10)
        let promptCache = makeMockPromptCache(seqLen: 3)

        cache.insertCache(model: "modelA", tokens: [1, 2, 3], promptCache: promptCache)

        // Fetch from a different model should return nil
        let (result, remainder) = cache.fetchNearestCache(model: "modelB", tokens: [1, 2, 3])
        XCTAssertNil(result, "Cross-model lookup should return nil")
        XCTAssertEqual(remainder, [1, 2, 3], "Remainder should be full tokens for cross-model")

        // Fetch from same model should work
        let (resultA, remainderA) = cache.fetchNearestCache(model: "modelA", tokens: [1, 2, 3])
        XCTAssertNotNil(resultA, "Same model lookup should succeed")
        XCTAssertEqual(remainderA, [], "Same model exact match should have empty remainder")
    }

    // MARK: - VAL-PCACHE-013: Longer cached prefix returns trimmed cache

    func testLongerCachedPrefixReturnsTrimmed() throws {
        try skipIfMetalUnavailable()

        let cache = LRUPromptCache(maxSize: 10)
        let promptCache = makeMockPromptCache(seqLen: 5)

        cache.insertCache(model: "model1", tokens: [1, 2, 3, 4, 5], promptCache: promptCache)

        // Query is shorter than cached entry
        let (result, remainder) = cache.fetchNearestCache(model: "model1", tokens: [1, 2, 3])

        XCTAssertNotNil(result, "Should find longer prefix and return trimmed cache")
        // After trimming, the cache should cover the full query (3 tokens).
        // prefix = min(tokens.count, commonPrefix) = min(3, 3) = 3
        // numToTrim = longer.count - prefix = 5 - 3 = 2
        // After trimming 2 tokens from a 5-token cache: offset = 3
        if let result {
            for layer in result {
                XCTAssertEqual(layer.offset, 3, "Trimmed cache should have offset 3")
            }
            XCTAssertEqual(remainder, [], "Remainder should be empty (all query tokens covered)")
        }
    }

    // MARK: - Additional tests

    func testFetchReturnsDeepCopy() throws {
        try skipIfMetalUnavailable()

        let cache = LRUPromptCache(maxSize: 10)
        let promptCache = makeMockPromptCache(seqLen: 3)

        cache.insertCache(model: "model1", tokens: [1, 2, 3], promptCache: promptCache)

        let (result1, _) = cache.fetchNearestCache(model: "model1", tokens: [1, 2, 3])
        let (result2, _) = cache.fetchNearestCache(model: "model1", tokens: [1, 2, 3])

        XCTAssertNotNil(result1)
        XCTAssertNotNil(result2)

        // Mutate result1 by trimming — result2 should be unaffected
        if let r1 = result1, let r2 = result2 {
            r1[0].trim(1)
            XCTAssertNotEqual(
                r1[0].offset, r2[0].offset,
                "Deep copies should be independent after mutation")
        }
    }

    func testTrimToNSequences() throws {
        try skipIfMetalUnavailable()

        let cache = LRUPromptCache(maxSize: 100)

        for i in 1 ... 5 {
            cache.insertCache(
                model: "model1", tokens: [i], promptCache: makeMockPromptCache(seqLen: 1))
        }
        XCTAssertEqual(cache.count, 5)

        cache.trimTo(nSequences: 2)
        XCTAssertEqual(cache.count, 2, "Should have trimmed down to 2 entries")
    }

    func testTrimToNBytes() throws {
        try skipIfMetalUnavailable()

        let cache = LRUPromptCache(maxSize: 100)

        for i in 1 ... 5 {
            cache.insertCache(
                model: "model1", tokens: [i], promptCache: makeMockPromptCache(seqLen: 5))
        }

        cache.trimTo(nBytes: 0)
        XCTAssertEqual(cache.count, 0, "Trimming to 0 bytes should remove all entries")
        XCTAssertEqual(cache.nbytes, 0, "Byte count should be 0 after full trim")
    }

    func testInsertUpdatesSameKey() throws {
        try skipIfMetalUnavailable()

        let cache = LRUPromptCache(maxSize: 10)
        let promptCache1 = makeMockPromptCache(seqLen: 3)
        let promptCache2 = makeMockPromptCache(seqLen: 5)

        cache.insertCache(model: "model1", tokens: [1, 2, 3], promptCache: promptCache1)
        XCTAssertEqual(cache.count, 1)

        // Re-inserting same key should update, not add
        cache.insertCache(model: "model1", tokens: [1, 2, 3], promptCache: promptCache2)
        XCTAssertEqual(cache.count, 1, "Re-insertion should not increase count")
    }

    func testNoMatchForDifferentPrefix() throws {
        try skipIfMetalUnavailable()

        let cache = LRUPromptCache(maxSize: 10)
        cache.insertCache(
            model: "model1", tokens: [1, 2, 3], promptCache: makeMockPromptCache(seqLen: 3))

        // Different starting token
        let (result, remainder) = cache.fetchNearestCache(model: "model1", tokens: [5, 6, 7])
        XCTAssertNil(result, "Completely different prefix should not match")
        XCTAssertEqual(remainder, [5, 6, 7])
    }

    func testTrimmableShorterPrefixEvictionOnInsert() throws {
        try skipIfMetalUnavailable()

        let cache = LRUPromptCache(maxSize: 10)

        // Insert a shorter prefix
        cache.insertCache(
            model: "model1", tokens: [1, 2], promptCache: makeMockPromptCache(seqLen: 2))

        // Now insert a longer sequence through the same path — the shorter should be evicted
        cache.insertCache(
            model: "model1", tokens: [1, 2, 3], promptCache: makeMockPromptCache(seqLen: 3))

        // Since KVCacheSimple is trimmable, the shorter entry should have been removed
        // The longer entry should exist
        let (result, remainder) = cache.fetchNearestCache(model: "model1", tokens: [1, 2, 3])
        XCTAssertNotNil(result, "Longer entry should exist")
        XCTAssertEqual(remainder, [], "Should be exact match")

        // Count should be 1 (shorter was evicted)
        XCTAssertEqual(cache.count, 1)
    }

    func testMultipleModels() throws {
        try skipIfMetalUnavailable()

        let cache = LRUPromptCache(maxSize: 10)
        cache.insertCache(
            model: "modelA", tokens: [1, 2], promptCache: makeMockPromptCache(seqLen: 2))
        cache.insertCache(
            model: "modelB", tokens: [1, 2], promptCache: makeMockPromptCache(seqLen: 2))

        XCTAssertEqual(cache.count, 2, "Two entries for different models")

        let (resultA, _) = cache.fetchNearestCache(model: "modelA", tokens: [1, 2])
        let (resultB, _) = cache.fetchNearestCache(model: "modelB", tokens: [1, 2])

        XCTAssertNotNil(resultA)
        XCTAssertNotNil(resultB)
    }

    // MARK: - Regression: Bug 1 — Single-token prefix miss

    func testSingleTokenPrefixMatch() throws {
        try skipIfMetalUnavailable()

        let cache = LRUPromptCache(maxSize: 10)
        cache.insertCache(
            model: "model1", tokens: [42], promptCache: makeMockPromptCache(seqLen: 1))

        // Query extends beyond the single cached token
        let (result, remainder) = cache.fetchNearestCache(
            model: "model1", tokens: [42, 100, 200])

        XCTAssertNotNil(result, "Single-token cached prefix must be found")
        XCTAssertEqual(
            remainder, [100, 200], "Remainder should be tokens after the single-token prefix")
    }

    func testSingleTokenExactMatch() throws {
        try skipIfMetalUnavailable()

        let cache = LRUPromptCache(maxSize: 10)
        cache.insertCache(
            model: "model1", tokens: [42], promptCache: makeMockPromptCache(seqLen: 1))

        // Exact single-token query
        let (result, remainder) = cache.fetchNearestCache(model: "model1", tokens: [42])

        XCTAssertNotNil(result, "Single-token exact match must be found")
        XCTAssertEqual(remainder, [], "Exact match remainder should be empty")
    }

    // MARK: - Regression: Bug 2 — Longer-prefix under-trim

    func testLongerPrefixTrimAlignedToQueryLength() throws {
        try skipIfMetalUnavailable()

        let cache = LRUPromptCache(maxSize: 10)
        // Cached entry covers 10 tokens
        cache.insertCache(
            model: "model1", tokens: Array(1 ... 10),
            promptCache: makeMockPromptCache(seqLen: 10))

        // Query covers the first 5 tokens
        let (result, remainder) = cache.fetchNearestCache(
            model: "model1", tokens: Array(1 ... 5))

        XCTAssertNotNil(result, "Longer prefix should return trimmed cache")
        if let result {
            for layer in result {
                // prefix = min(5, 5) = 5, numToTrim = 10 - 5 = 5
                // After trimming 5 tokens from 10: offset = 5
                XCTAssertEqual(
                    layer.offset, 5, "Trimmed cache should have offset equal to query length")
            }
        }
        // All query tokens are covered — remainder should be empty
        XCTAssertEqual(remainder, [], "All query tokens are covered by the longer cached entry")
    }

    func testLongerPrefixTrimPartialQueryMatch() throws {
        try skipIfMetalUnavailable()

        let cache = LRUPromptCache(maxSize: 10)
        // Cached entry: [1, 2, 3, 4, 5]
        cache.insertCache(
            model: "model1", tokens: [1, 2, 3, 4, 5],
            promptCache: makeMockPromptCache(seqLen: 5))

        // Query [1, 2, 3, 6, 7] diverges at index 3
        // commonPrefix = 3, longer prefix = [1,2,3,4,5] (found via DFS)
        let (result, remainder) = cache.fetchNearestCache(
            model: "model1", tokens: [1, 2, 3, 6, 7])

        XCTAssertNotNil(result, "Should find longer prefix from diverging query")
        if let result {
            for layer in result {
                // prefix = min(5, 3) = 3, numToTrim = 5 - 3 = 2
                XCTAssertEqual(layer.offset, 3, "Trimmed cache should cover common prefix")
            }
        }
        XCTAssertEqual(remainder, [6, 7], "Remainder should be the diverging suffix")
    }

    // MARK: - Regression: Bug 3 — LRU recency not refreshed on fetch

    func testFetchRefreshesLRURecency() throws {
        try skipIfMetalUnavailable()

        let cache = LRUPromptCache(maxSize: 3)

        // Insert 3 entries in order: [1], [2], [3]
        cache.insertCache(
            model: "model1", tokens: [1], promptCache: makeMockPromptCache(seqLen: 1))
        cache.insertCache(
            model: "model1", tokens: [2], promptCache: makeMockPromptCache(seqLen: 1))
        cache.insertCache(
            model: "model1", tokens: [3], promptCache: makeMockPromptCache(seqLen: 1))

        // Fetch [1] to refresh its recency — it becomes the most-recently-used
        let (fetched, _) = cache.fetchNearestCache(model: "model1", tokens: [1])
        XCTAssertNotNil(fetched, "[1] should still be present before eviction")

        // Insert [4], which must evict the LRU entry.
        // Without the fix, [1] would be evicted (insertion order).
        // With the fix, [2] should be evicted (least recently used after [1] was fetched).
        cache.insertCache(
            model: "model1", tokens: [4], promptCache: makeMockPromptCache(seqLen: 1))
        XCTAssertEqual(cache.count, 3)

        // [1] should survive because it was recently fetched
        let (result1, _) = cache.fetchNearestCache(model: "model1", tokens: [1])
        XCTAssertNotNil(result1, "[1] should survive eviction because fetch refreshed its recency")

        // [2] should be evicted (oldest unfetched entry)
        let (result2, _) = cache.fetchNearestCache(model: "model1", tokens: [2])
        XCTAssertNil(result2, "[2] should be evicted as least-recently-used")

        // [3] and [4] should still be present
        let (result3, _) = cache.fetchNearestCache(model: "model1", tokens: [3])
        XCTAssertNotNil(result3, "[3] should still be present")
        let (result4, _) = cache.fetchNearestCache(model: "model1", tokens: [4])
        XCTAssertNotNil(result4, "[4] should still be present")
    }

    func testFetchRefreshesLRURecencyShorterPrefix() throws {
        try skipIfMetalUnavailable()

        let cache = LRUPromptCache(maxSize: 3)

        // Insert 3 entries
        cache.insertCache(
            model: "model1", tokens: [10, 20],
            promptCache: makeMockPromptCache(seqLen: 2))
        cache.insertCache(
            model: "model1", tokens: [30],
            promptCache: makeMockPromptCache(seqLen: 1))
        cache.insertCache(
            model: "model1", tokens: [40],
            promptCache: makeMockPromptCache(seqLen: 1))

        // Fetch [10, 20, 99] which triggers shorter-prefix match on [10, 20]
        let (fetched, rem) = cache.fetchNearestCache(
            model: "model1", tokens: [10, 20, 99])
        XCTAssertNotNil(fetched, "Should find shorter prefix [10,20]")
        XCTAssertEqual(rem, [99])

        // Insert [50] — this should evict [30] (LRU), not [10,20]
        cache.insertCache(
            model: "model1", tokens: [50],
            promptCache: makeMockPromptCache(seqLen: 1))

        let (r1020, _) = cache.fetchNearestCache(model: "model1", tokens: [10, 20])
        XCTAssertNotNil(r1020, "[10,20] should survive because fetch refreshed its recency")

        let (r30, _) = cache.fetchNearestCache(model: "model1", tokens: [30])
        XCTAssertNil(r30, "[30] should be evicted as least-recently-used")
    }

    // MARK: - Regression: Bug 4 — maxBytes eviction stops at 1 entry

    func testMaxBytesEvictsLastOversizedEntry() throws {
        try skipIfMetalUnavailable()

        // Set maxBytes to 0: every entry should be evicted immediately after insertion
        let cache = LRUPromptCache(maxSize: 100, maxBytes: 0)

        cache.insertCache(
            model: "model1", tokens: [1], promptCache: makeMockPromptCache(seqLen: 5))

        // With the bug (lru.count > 1), the single entry would stay.
        // With the fix, it should be evicted since its bytes > maxBytes(0).
        XCTAssertEqual(
            cache.count, 0, "Single oversized entry should be evicted when exceeding maxBytes")
        XCTAssertEqual(cache.nbytes, 0, "Byte count should be 0 after evicting oversized entry")
    }

    func testMaxBytesEvictsDownToLimit() throws {
        try skipIfMetalUnavailable()

        let promptCache = makeMockPromptCache(seqLen: 5)
        let bytesPerEntry = promptCache.reduce(0) { $0 + $1.state.reduce(0) { $0 + $1.nbytes } }

        // Set maxBytes to fit exactly 1 entry
        let cache = LRUPromptCache(maxSize: 100, maxBytes: bytesPerEntry)

        cache.insertCache(
            model: "model1", tokens: [1], promptCache: makeMockPromptCache(seqLen: 5))
        cache.insertCache(
            model: "model1", tokens: [2], promptCache: makeMockPromptCache(seqLen: 5))

        // After inserting 2nd entry, total bytes = 2 * bytesPerEntry > maxBytes.
        // Should evict down until within budget. Only 1 entry should remain.
        XCTAssertEqual(cache.count, 1, "Should evict down to 1 entry to stay within maxBytes")
        XCTAssertLessThanOrEqual(cache.nbytes, bytesPerEntry)

        // The surviving entry should be [2] (most recently inserted)
        let (result1, _) = cache.fetchNearestCache(model: "model1", tokens: [1])
        XCTAssertNil(result1, "[1] should be evicted (LRU)")
        let (result2, _) = cache.fetchNearestCache(model: "model1", tokens: [2])
        XCTAssertNotNil(result2, "[2] should survive (most recent)")
    }
}
