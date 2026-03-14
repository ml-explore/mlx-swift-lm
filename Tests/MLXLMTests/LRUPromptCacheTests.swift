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
        // After trimming, the cache should cover the common prefix (3 tokens)
        // and remainder should be the tokens after the prefix match point
        if let result {
            for layer in result {
                // Each layer's offset should be 2 (trimmed from 5 to prefix=2)
                // Python: prefix = min(len(tokens)-1, commonPrefix) = min(2, 3) = 2
                // numToTrim = len(longer) - prefix = 5 - 2 = 3
                // After trimming 3 tokens from a 5-token cache: offset = 2
                XCTAssertEqual(layer.offset, 2, "Trimmed cache should have offset 2")
            }
            XCTAssertEqual(remainder, [3], "Remainder should start from prefix point")
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
}
