// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import XCTest

@testable import MLXLMCommon

// MARK: - BatchKVCacheTests

final class BatchKVCacheTests: XCTestCase {

    // MARK: - Helpers

    /// Create keys/values with known content for testing.
    /// Shape: [B, H, S, D]
    private func makeKV(
        batchSize B: Int, heads H: Int, seqLen S: Int, headDim D: Int, value: Float = 1.0
    ) -> (MLXArray, MLXArray) {
        let keys = MLXArray.ones([B, H, S, D]) * value
        let values = MLXArray.ones([B, H, S, D]) * (value + 1)
        return (keys, values)
    }

    /// Create keys/values with per-batch unique content (batch i gets value i+1).
    private func makeDistinctKV(
        batchSize B: Int, heads H: Int, seqLen S: Int, headDim D: Int
    ) -> (MLXArray, MLXArray) {
        var keysList: [MLXArray] = []
        var valuesList: [MLXArray] = []
        for i in 0 ..< B {
            keysList.append(MLXArray.ones([1, H, S, D]) * Float(i + 1))
            valuesList.append(MLXArray.ones([1, H, S, D]) * Float(i + 1) * 10)
        }
        return (concatenated(keysList, axis: 0), concatenated(valuesList, axis: 0))
    }

    // MARK: - VAL-CACHE-001: Init with left-padding

    func testInitWithLeftPadding() throws {
        try skipIfMetalUnavailable()

        let cache = BatchKVCache(leftPadding: [1, 3, 0])

        // leftPadding stored correctly
        XCTAssertEqual(cache.leftPadding.shape, [3])
        XCTAssertEqual(cache.leftPadding[0].item(Int32.self), 1)
        XCTAssertEqual(cache.leftPadding[1].item(Int32.self), 3)
        XCTAssertEqual(cache.leftPadding[2].item(Int32.self), 0)

        // offset = -leftPadding
        XCTAssertEqual(cache.batchOffsets[0].item(Int32.self), -1)
        XCTAssertEqual(cache.batchOffsets[1].item(Int32.self), -3)
        XCTAssertEqual(cache.batchOffsets[2].item(Int32.self), 0)

        // Keys and values are nil initially
        XCTAssertNil(cache.keys)
        XCTAssertNil(cache.values)

        // _idx starts at 0
        XCTAssertEqual(cache._idx, 0)
    }

    // MARK: - VAL-CACHE-002: First update stores keys/values and advances offset

    func testFirstUpdate() throws {
        try skipIfMetalUnavailable()

        let cache = BatchKVCache(leftPadding: [1, 3, 0])
        let B = 3
        let H = 4
        let S = 5
        let D = 8

        let (keys, values) = makeKV(batchSize: B, heads: H, seqLen: S, headDim: D)
        let (retK, retV) = cache.update(keys: keys, values: values)

        // Returned shape correct
        XCTAssertEqual(retK.shape, [B, H, S, D])
        XCTAssertEqual(retV.shape, [B, H, S, D])

        // Offset advanced by sequence length
        XCTAssertEqual(cache.batchOffsets[0].item(Int32.self), -1 + Int32(S))
        XCTAssertEqual(cache.batchOffsets[1].item(Int32.self), -3 + Int32(S))
        XCTAssertEqual(cache.batchOffsets[2].item(Int32.self), 0 + Int32(S))

        // _idx advanced
        XCTAssertEqual(cache._idx, S)

        // Keys/values are not nil
        XCTAssertNotNil(cache.keys)
        XCTAssertNotNil(cache.values)
    }

    // MARK: - VAL-CACHE-003: Filter retains only selected batch indices

    func testFilterRetainsIndices() throws {
        try skipIfMetalUnavailable()

        let cache = BatchKVCache(leftPadding: [1, 3, 0])
        let B = 3
        let H = 2
        let S = 4
        let D = 4

        let (keys, values) = makeDistinctKV(batchSize: B, heads: H, seqLen: S, headDim: D)
        _ = cache.update(keys: keys, values: values)

        // Keep only batch 0 and 2
        cache.filter(batchIndices: [0, 2])

        // Batch dimension reduced
        XCTAssertEqual(cache.keys!.dim(0), 2)
        XCTAssertEqual(cache.values!.dim(0), 2)
        XCTAssertEqual(cache.batchOffsets.dim(0), 2)
        XCTAssertEqual(cache.leftPadding.dim(0), 2)
    }

    // MARK: - VAL-CACHE-004: Filter shifts left to reduce padding

    func testFilterShiftsPadding() throws {
        try skipIfMetalUnavailable()

        let cache = BatchKVCache(leftPadding: [2, 4, 0])
        let B = 3
        let H = 2
        let S = 6
        let D = 4

        let (keys, values) = makeKV(batchSize: B, heads: H, seqLen: S, headDim: D)
        _ = cache.update(keys: keys, values: values)

        let idxBefore = cache._idx
        // Keep only batch 0 (padding=2) and batch 1 (padding=4)
        cache.filter(batchIndices: [0, 1])

        let minPad = 2  // min of [2, 4]
        XCTAssertEqual(cache._idx, idxBefore - minPad)
        XCTAssertEqual(cache.leftPadding[0].item(Int32.self), 0)  // 2 - 2
        XCTAssertEqual(cache.leftPadding[1].item(Int32.self), 2)  // 4 - 2
    }

    // MARK: - VAL-CACHE-005: Extend merges two caches along batch dimension

    func testExtendMergesBatch() throws {
        try skipIfMetalUnavailable()

        let cacheA = BatchKVCache(leftPadding: [0, 0])
        let cacheB = BatchKVCache(leftPadding: [0])

        let H = 2
        let S = 3
        let D = 4

        let (keysA, valuesA) = makeKV(batchSize: 2, heads: H, seqLen: S, headDim: D, value: 1.0)
        let (keysB, valuesB) = makeKV(batchSize: 1, heads: H, seqLen: S, headDim: D, value: 5.0)

        _ = cacheA.update(keys: keysA, values: valuesA)
        _ = cacheB.update(keys: keysB, values: valuesB)

        cacheA.extend(other: cacheB)

        // Combined batch size
        XCTAssertEqual(cacheA.keys!.dim(0), 3)
        XCTAssertEqual(cacheA.values!.dim(0), 3)
        XCTAssertEqual(cacheA.batchOffsets.dim(0), 3)
        XCTAssertEqual(cacheA.leftPadding.dim(0), 3)
    }

    // MARK: - VAL-CACHE-006: Extend right-justifies different lengths

    func testExtendRightJustifies() throws {
        try skipIfMetalUnavailable()

        let cacheA = BatchKVCache(leftPadding: [0])
        let cacheB = BatchKVCache(leftPadding: [0])

        let H = 2
        let D = 4

        // Cache A has 5 tokens
        let (keysA, valuesA) = makeKV(batchSize: 1, heads: H, seqLen: 5, headDim: D, value: 1.0)
        _ = cacheA.update(keys: keysA, values: valuesA)

        // Cache B has 3 tokens (shorter)
        let (keysB, valuesB) = makeKV(batchSize: 1, heads: H, seqLen: 3, headDim: D, value: 2.0)
        _ = cacheB.update(keys: keysB, values: valuesB)

        cacheA.extend(other: cacheB)

        // _idx should be max(5, 3) = 5
        XCTAssertEqual(cacheA._idx, 5)

        // Shorter cache (B) gets left-padding of 2
        XCTAssertEqual(cacheA.leftPadding[1].item(Int32.self), 2)  // 5 - 3

        // Longer cache (A) keeps leftPadding of 0
        XCTAssertEqual(cacheA.leftPadding[0].item(Int32.self), 0)
    }

    // MARK: - VAL-CACHE-007: Extract returns single-sequence KVCacheSimple

    func testExtractReturnsKVCacheSimple() throws {
        try skipIfMetalUnavailable()

        let cache = BatchKVCache(leftPadding: [2, 0])
        let H = 2
        let S = 4
        let D = 4

        let (keys, values) = makeDistinctKV(batchSize: 2, heads: H, seqLen: S, headDim: D)
        _ = cache.update(keys: keys, values: values)

        let extracted = cache.extract(idx: 1)

        // extract(idx:) returns KVCacheSimple — verify it has the expected properties
        XCTAssertEqual(String(describing: type(of: extracted)), "KVCacheSimple")

        // Batch dimension is 1
        XCTAssertEqual(extracted.keys!.dim(0), 1)
        XCTAssertEqual(extracted.values!.dim(0), 1)
    }

    // MARK: - VAL-CACHE-008: Extract strips left-padding

    func testExtractStripsPadding() throws {
        try skipIfMetalUnavailable()

        let cache = BatchKVCache(leftPadding: [2, 0])
        let H = 2
        let S = 5
        let D = 4

        let (keys, values) = makeDistinctKV(batchSize: 2, heads: H, seqLen: S, headDim: D)
        _ = cache.update(keys: keys, values: values)

        // Extract batch 0 which has padding=2
        let extracted = cache.extract(idx: 0)

        // Sequence length should be S - padding = 5 - 2 = 3
        XCTAssertEqual(extracted.keys!.dim(2), S - 2)
        XCTAssertEqual(extracted.values!.dim(2), S - 2)

        // Offset should be 3
        XCTAssertEqual(extracted.offset, S - 2)
    }

    // MARK: - VAL-CACHE-009: Merge creates BatchKVCache from individual caches

    func testMergeFromIndividuals() throws {
        try skipIfMetalUnavailable()

        let H = 2
        let D = 4

        let cacheA = KVCacheSimple()
        let cacheB = KVCacheSimple()
        let cacheC = KVCacheSimple()

        let (kA, vA) = makeKV(batchSize: 1, heads: H, seqLen: 5, headDim: D, value: 1.0)
        let (kB, vB) = makeKV(batchSize: 1, heads: H, seqLen: 3, headDim: D, value: 2.0)
        let (kC, vC) = makeKV(batchSize: 1, heads: H, seqLen: 7, headDim: D, value: 3.0)

        _ = cacheA.update(keys: kA, values: vA)
        _ = cacheB.update(keys: kB, values: vB)
        _ = cacheC.update(keys: kC, values: vC)

        let batchCache = BatchKVCache.merge([cacheA, cacheB, cacheC])

        // Batch size is 3
        XCTAssertEqual(batchCache.batchSize, 3)
        XCTAssertEqual(batchCache.keys!.dim(0), 3)
    }

    // MARK: - VAL-CACHE-010: Merge left-pads shorter sequences

    func testMergeLeftPads() throws {
        try skipIfMetalUnavailable()

        let H = 2
        let D = 4

        let cacheA = KVCacheSimple()
        let cacheB = KVCacheSimple()
        let cacheC = KVCacheSimple()

        let (kA, vA) = makeKV(batchSize: 1, heads: H, seqLen: 5, headDim: D, value: 1.0)
        let (kB, vB) = makeKV(batchSize: 1, heads: H, seqLen: 3, headDim: D, value: 2.0)
        let (kC, vC) = makeKV(batchSize: 1, heads: H, seqLen: 7, headDim: D, value: 3.0)

        _ = cacheA.update(keys: kA, values: vA)
        _ = cacheB.update(keys: kB, values: vB)
        _ = cacheC.update(keys: kC, values: vC)

        let batchCache = BatchKVCache.merge([cacheA, cacheB, cacheC])

        // maxLength = 7, padding = [2, 4, 0]
        XCTAssertEqual(batchCache.leftPadding[0].item(Int32.self), 2)
        XCTAssertEqual(batchCache.leftPadding[1].item(Int32.self), 4)
        XCTAssertEqual(batchCache.leftPadding[2].item(Int32.self), 0)
    }

    // MARK: - VAL-CACHE-016: fromSingle creates batch-1 cache

    func testFromSingle() throws {
        try skipIfMetalUnavailable()

        let simple = KVCacheSimple()
        let H = 2
        let D = 4
        let S = 5

        let (k, v) = makeKV(batchSize: 1, heads: H, seqLen: S, headDim: D)
        _ = simple.update(keys: k, values: v)

        let batchCache = BatchKVCache.fromSingle(simple)

        XCTAssertEqual(batchCache.batchSize, 1)
        XCTAssertEqual(batchCache.leftPadding[0].item(Int32.self), 0)
        XCTAssertNotNil(batchCache.keys)
        XCTAssertEqual(batchCache._idx, S)
        XCTAssertEqual(batchCache.batchOffsets[0].item(Int32.self), Int32(S))
    }

    // MARK: - VAL-CACHE-017: Batch-1 equivalence

    func testBatch1Equivalence() throws {
        try skipIfMetalUnavailable()

        let H = 2
        let D = 4
        let S = 5

        let (keys, values) = makeKV(batchSize: 1, heads: H, seqLen: S, headDim: D)

        // Use KVCacheSimple
        let simpleCache = KVCacheSimple()
        let (simpleK, simpleV) = simpleCache.update(keys: keys, values: values)

        // Use BatchKVCache with batch size 1
        let batchCache = BatchKVCache(leftPadding: [0])
        let (batchK, batchV) = batchCache.update(keys: keys, values: values)

        // Results should be identical
        XCTAssertEqual(simpleK.shape, batchK.shape)
        XCTAssertEqual(simpleV.shape, batchV.shape)

        let kDiff = abs(simpleK - batchK).sum().item(Float.self)
        let vDiff = abs(simpleV - batchV).sum().item(Float.self)
        XCTAssertEqual(kDiff, 0.0)
        XCTAssertEqual(vDiff, 0.0)
    }

    // MARK: - VAL-CACHE-018: Merge-extract round-trip preserves data

    func testMergeExtractRoundTrip() throws {
        try skipIfMetalUnavailable()

        let H = 2
        let D = 4

        let cacheA = KVCacheSimple()
        let cacheB = KVCacheSimple()

        let (kA, vA) = makeKV(batchSize: 1, heads: H, seqLen: 3, headDim: D, value: 1.0)
        let (kB, vB) = makeKV(batchSize: 1, heads: H, seqLen: 5, headDim: D, value: 2.0)

        _ = cacheA.update(keys: kA, values: vA)
        _ = cacheB.update(keys: kB, values: vB)

        // Merge
        let batchCache = BatchKVCache.merge([cacheA, cacheB])

        // Extract
        let extractedA = batchCache.extract(idx: 0)
        let extractedB = batchCache.extract(idx: 1)

        // Check offsets
        XCTAssertEqual(extractedA.offset, 3)
        XCTAssertEqual(extractedB.offset, 5)

        // Check key shapes
        XCTAssertEqual(extractedA.keys!.dim(2), 3)
        XCTAssertEqual(extractedB.keys!.dim(2), 5)

        // Check values match
        let diffAKeys = abs(extractedA.keys![.ellipsis, ..<3, 0...] - kA).sum().item(Float.self)
        let diffBKeys = abs(extractedB.keys![.ellipsis, ..<5, 0...] - kB).sum().item(Float.self)
        XCTAssertEqual(diffAKeys, 0.0)
        XCTAssertEqual(diffBKeys, 0.0)

        let diffAValues =
            abs(extractedA.values![.ellipsis, ..<3, 0...] - vA).sum().item(Float.self)
        let diffBValues =
            abs(extractedB.values![.ellipsis, ..<5, 0...] - vB).sum().item(Float.self)
        XCTAssertEqual(diffAValues, 0.0)
        XCTAssertEqual(diffBValues, 0.0)
    }

    // MARK: - VAL-CACHE-019: Successive filter-extend cycles

    func testSuccessiveFilterExtendCycles() throws {
        try skipIfMetalUnavailable()

        let H = 2
        let D = 4

        let cacheA = KVCacheSimple()
        let cacheB = KVCacheSimple()
        let cacheC = KVCacheSimple()

        let (kA, vA) = makeKV(batchSize: 1, heads: H, seqLen: 3, headDim: D, value: 1.0)
        let (kB, vB) = makeKV(batchSize: 1, heads: H, seqLen: 4, headDim: D, value: 2.0)
        let (kC, vC) = makeKV(batchSize: 1, heads: H, seqLen: 5, headDim: D, value: 3.0)

        _ = cacheA.update(keys: kA, values: vA)
        _ = cacheB.update(keys: kB, values: vB)
        _ = cacheC.update(keys: kC, values: vC)

        let batchCache = BatchKVCache.merge([cacheA, cacheB, cacheC])
        XCTAssertEqual(batchCache.batchSize, 3)

        // Cycle 1: filter out batch 1
        batchCache.filter(batchIndices: [0, 2])
        XCTAssertEqual(batchCache.batchSize, 2)

        // Add a new sequence
        let cacheD = KVCacheSimple()
        let (kD, vD) = makeKV(batchSize: 1, heads: H, seqLen: 6, headDim: D, value: 4.0)
        _ = cacheD.update(keys: kD, values: vD)
        let newBatch = BatchKVCache.merge([cacheD])
        batchCache.extend(other: newBatch)
        XCTAssertEqual(batchCache.batchSize, 3)

        // Cycle 2: filter out first
        batchCache.filter(batchIndices: [1, 2])
        XCTAssertEqual(batchCache.batchSize, 2)

        // Cycle 3: add another
        let cacheE = KVCacheSimple()
        let (kE, vE) = makeKV(batchSize: 1, heads: H, seqLen: 2, headDim: D, value: 5.0)
        _ = cacheE.update(keys: kE, values: vE)
        let newBatch2 = BatchKVCache.merge([cacheE])
        batchCache.extend(other: newBatch2)
        XCTAssertEqual(batchCache.batchSize, 3)

        // Verify we can still extract
        let ex0 = batchCache.extract(idx: 0)
        let ex1 = batchCache.extract(idx: 1)
        let ex2 = batchCache.extract(idx: 2)

        XCTAssertGreaterThan(ex0.offset, 0)
        XCTAssertGreaterThan(ex1.offset, 0)
        XCTAssertGreaterThan(ex2.offset, 0)
    }

    // MARK: - VAL-CACHE-021: Filter to empty batch

    func testFilterToEmptyBatch() throws {
        try skipIfMetalUnavailable()

        let cache = BatchKVCache(leftPadding: [1, 2, 0])
        let H = 2
        let S = 3
        let D = 4

        let (keys, values) = makeKV(batchSize: 3, heads: H, seqLen: S, headDim: D)
        _ = cache.update(keys: keys, values: values)

        cache.filter(batchIndices: [])

        XCTAssertNil(cache.keys)
        XCTAssertNil(cache.values)
        XCTAssertEqual(cache._idx, 0)
        XCTAssertEqual(cache.leftPadding.dim(0), 0)
        XCTAssertEqual(cache.batchOffsets.dim(0), 0)
    }

    // MARK: - Additional tests

    func testToSingle() throws {
        try skipIfMetalUnavailable()

        let simple = KVCacheSimple()
        let H = 2
        let D = 4
        let S = 5

        let (k, v) = makeKV(batchSize: 1, heads: H, seqLen: S, headDim: D, value: 7.0)
        _ = simple.update(keys: k, values: v)

        let batchCache = BatchKVCache.fromSingle(simple)
        let backToSingle = batchCache.toSingle()

        XCTAssertEqual(backToSingle.offset, S)
        XCTAssertEqual(backToSingle.keys!.dim(0), 1)
        XCTAssertEqual(backToSingle.keys!.dim(2), S)
    }

    func testMultipleUpdates() throws {
        try skipIfMetalUnavailable()

        let cache = BatchKVCache(leftPadding: [0, 0])
        let H = 2
        let D = 4

        let (k1, v1) = makeKV(batchSize: 2, heads: H, seqLen: 3, headDim: D, value: 1.0)
        let (retK1, _) = cache.update(keys: k1, values: v1)
        XCTAssertEqual(retK1.shape, [2, H, 3, D])
        XCTAssertEqual(cache._idx, 3)

        let (k2, v2) = makeKV(batchSize: 2, heads: H, seqLen: 1, headDim: D, value: 2.0)
        let (retK2, _) = cache.update(keys: k2, values: v2)
        XCTAssertEqual(retK2.shape, [2, H, 4, D])
        XCTAssertEqual(cache._idx, 4)
    }

    func testFilterSingleIndex() throws {
        try skipIfMetalUnavailable()

        let cache = BatchKVCache(leftPadding: [0, 2, 1])
        let H = 2
        let S = 4
        let D = 4

        let (keys, values) = makeDistinctKV(batchSize: 3, heads: H, seqLen: S, headDim: D)
        _ = cache.update(keys: keys, values: values)

        cache.filter(batchIndices: [1])

        XCTAssertEqual(cache.batchSize, 1)
        XCTAssertEqual(cache.leftPadding[0].item(Int32.self), 0)
    }

    func testExtendEmptyWithNonEmpty() throws {
        try skipIfMetalUnavailable()

        let emptyCache = BatchKVCache(leftPadding: [])
        let filledCache = BatchKVCache(leftPadding: [0])

        let H = 2
        let D = 4
        let (k, v) = makeKV(batchSize: 1, heads: H, seqLen: 3, headDim: D)
        _ = filledCache.update(keys: k, values: v)

        emptyCache.extend(other: filledCache)

        XCTAssertNotNil(emptyCache.keys)
        XCTAssertEqual(emptyCache._idx, 3)
        XCTAssertEqual(emptyCache.batchSize, 1)
    }

    func testStateSerialization() throws {
        try skipIfMetalUnavailable()

        let cache = BatchKVCache(leftPadding: [1, 0])
        let H = 2
        let S = 3
        let D = 4

        let (keys, values) = makeKV(batchSize: 2, heads: H, seqLen: S, headDim: D)
        _ = cache.update(keys: keys, values: values)

        let savedState = cache.state
        let savedMeta = cache.metaState

        let newCache = BatchKVCache(leftPadding: [0, 0])
        newCache.state = savedState
        newCache.metaState = savedMeta

        XCTAssertEqual(newCache._idx, cache._idx)
        XCTAssertNotNil(newCache.keys)
        XCTAssertNotNil(newCache.values)
    }

    func testIsTrimmable() throws {
        try skipIfMetalUnavailable()

        let cache = BatchKVCache(leftPadding: [0])
        XCTAssertTrue(cache.isTrimmable)
    }

    func testTrim() throws {
        try skipIfMetalUnavailable()

        let cache = BatchKVCache(leftPadding: [0])
        let (k, v) = makeKV(batchSize: 1, heads: 2, seqLen: 5, headDim: 4)
        _ = cache.update(keys: k, values: v)

        let trimmed = cache.trim(2)
        XCTAssertEqual(trimmed, 2)
        XCTAssertEqual(cache._idx, 3)
    }

    // MARK: - State round-trip for fresh (empty) cache

    func testStateRoundTripFreshCache() throws {
        try skipIfMetalUnavailable()

        let cache = BatchKVCache(leftPadding: [2, 5, 0])

        // Fresh cache — keys/values are nil
        XCTAssertNil(cache.keys)
        XCTAssertNil(cache.values)

        let savedState = cache.state
        let savedMeta = cache.metaState

        // State should contain batchOffsets + leftPadding (2 arrays)
        XCTAssertEqual(savedState.count, 2)

        // Round-trip into a new cache
        let restored = BatchKVCache(leftPadding: [0])
        restored.state = savedState
        restored.metaState = savedMeta

        // Verify round-trip preserves offsets and padding
        XCTAssertNil(restored.keys)
        XCTAssertNil(restored.values)
        XCTAssertEqual(restored._idx, 0)
        XCTAssertEqual(restored.batchOffsets.shape, [3])
        XCTAssertEqual(restored.leftPadding.shape, [3])
        XCTAssertEqual(restored.batchOffsets[0].item(Int32.self), -2)
        XCTAssertEqual(restored.batchOffsets[1].item(Int32.self), -5)
        XCTAssertEqual(restored.batchOffsets[2].item(Int32.self), 0)
        XCTAssertEqual(restored.leftPadding[0].item(Int32.self), 2)
        XCTAssertEqual(restored.leftPadding[1].item(Int32.self), 5)
        XCTAssertEqual(restored.leftPadding[2].item(Int32.self), 0)
    }

    // MARK: - State round-trip for cache emptied by filter([])

    func testStateRoundTripFilteredEmptyCache() throws {
        try skipIfMetalUnavailable()

        let cache = BatchKVCache(leftPadding: [1, 2, 0])
        let H = 2
        let S = 3
        let D = 4

        let (keys, values) = makeKV(batchSize: 3, heads: H, seqLen: S, headDim: D)
        _ = cache.update(keys: keys, values: values)

        // Empty the cache via filter
        cache.filter(batchIndices: [])

        XCTAssertNil(cache.keys)
        XCTAssertNil(cache.values)
        XCTAssertEqual(cache._idx, 0)

        let savedState = cache.state
        let savedMeta = cache.metaState

        // State should contain batchOffsets + leftPadding (2 arrays, both empty)
        XCTAssertEqual(savedState.count, 2)

        // Round-trip into a new cache
        let restored = BatchKVCache(leftPadding: [99])
        restored.state = savedState
        restored.metaState = savedMeta

        // Verify round-trip preserves empty state
        XCTAssertNil(restored.keys)
        XCTAssertNil(restored.values)
        XCTAssertEqual(restored._idx, 0)
        XCTAssertEqual(restored.batchOffsets.dim(0), 0)
        XCTAssertEqual(restored.leftPadding.dim(0), 0)
    }

    // MARK: - makeMask uses pre-update offset (real call order)

    func testMakeMaskBeforeUpdate() throws {
        try skipIfMetalUnavailable()

        // Simulate the real model call order: makeMask THEN update.
        // After prefill of S=4, _idx=4. Then for a decode step with n=1,
        // makeMask should produce a mask spanning columns 0..<(4+1)=5
        // (the 4 cached tokens plus the 1 new token).
        let cache = BatchKVCache(leftPadding: [1, 0])
        let B = 2
        let H = 2
        let S = 4
        let D = 4

        // Prefill
        let (keys, values) = makeKV(batchSize: B, heads: H, seqLen: S, headDim: D)
        _ = cache.update(keys: keys, values: values)
        XCTAssertEqual(cache._idx, S)

        // Now simulate a decode step: makeMask is called BEFORE update
        let n = 1
        let mask = cache.makeMask(n: n, windowSize: nil, returnArray: false)

        // The mask should cover offset=_idx=4 columns of history + n=1 new token = 5 columns total.
        // createCausalMask(n:1, offset:4) produces shape [1, 5].
        switch mask {
        case .array(let arr):
            // Row dimension = n = 1, column dimension = _idx + n = 5
            XCTAssertEqual(arr.dim(arr.ndim - 1), S + n)  // 5 columns
            XCTAssertEqual(arr.dim(arr.ndim - 2), n)  // 1 row
        default:
            XCTFail("Expected .array mask from batch cache")
        }

        // Now update (after makeMask, as models do)
        let (k2, v2) = makeKV(batchSize: B, heads: H, seqLen: n, headDim: D, value: 2.0)
        _ = cache.update(keys: k2, values: v2)
        XCTAssertEqual(cache._idx, S + n)
    }

    // MARK: - makeMask masks left-padding in decode step

    func testMakeMaskLeftPaddingDecode() throws {
        try skipIfMetalUnavailable()

        // Sequence 0 has leftPadding=2, sequence 1 has leftPadding=0.
        // After prefill of S=4 tokens, _idx=4. Decode step n=1.
        // For sequence 0, columns 0 and 1 (padded) must be False.
        // For sequence 1, all 5 columns should follow normal causal pattern.
        let cache = BatchKVCache(leftPadding: [2, 0])
        let B = 2
        let H = 2
        let S = 4
        let D = 4

        let (keys, values) = makeKV(batchSize: B, heads: H, seqLen: S, headDim: D)
        _ = cache.update(keys: keys, values: values)

        let n = 1
        let mask = cache.makeMask(n: n, windowSize: nil, returnArray: false)

        switch mask {
        case .array(let arr):
            // Shape: [B, 1, n, _idx+n] = [2, 1, 1, 5]
            XCTAssertEqual(arr.dim(arr.ndim - 1), S + n)  // 5 columns

            // Sequence 0 (leftPadding=2): columns 0,1 should be False
            let seq0Mask = arr[0]
            let col0 = seq0Mask[0..., 0..., 0].item(Bool.self)
            let col1 = seq0Mask[0..., 0..., 1].item(Bool.self)
            let col2 = seq0Mask[0..., 0..., 2].item(Bool.self)
            XCTAssertFalse(col0, "Padded column 0 should be masked out")
            XCTAssertFalse(col1, "Padded column 1 should be masked out")
            XCTAssertTrue(col2, "Valid column 2 should be unmasked")

            // Sequence 1 (leftPadding=0): all columns through the causal position should be True
            let seq1Mask = arr[1]
            let seq1col0 = seq1Mask[0..., 0..., 0].item(Bool.self)
            XCTAssertTrue(seq1col0, "Sequence 1 column 0 should be unmasked")
        default:
            XCTFail("Expected .array mask from batch cache")
        }
    }
}
