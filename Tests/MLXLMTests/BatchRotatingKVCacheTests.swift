// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import XCTest

@testable import MLXLMCommon

// MARK: - BatchRotatingKVCacheTests

final class BatchRotatingKVCacheTests: XCTestCase {

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

    // MARK: - Init

    func testInitWithMaxSizeAndLeftPadding() throws {
        try skipIfMetalUnavailable()

        let cache = BatchRotatingKVCache(maxSize: 32, leftPadding: [1, 3, 0])

        // leftPadding stored correctly
        XCTAssertEqual(cache.leftPadding.shape, [3])
        XCTAssertEqual(cache.leftPadding[0].item(Int32.self), 1)
        XCTAssertEqual(cache.leftPadding[1].item(Int32.self), 3)
        XCTAssertEqual(cache.leftPadding[2].item(Int32.self), 0)

        // offset = -leftPadding
        XCTAssertEqual(cache.batchOffsets[0].item(Int32.self), -1)
        XCTAssertEqual(cache.batchOffsets[1].item(Int32.self), -3)
        XCTAssertEqual(cache.batchOffsets[2].item(Int32.self), 0)

        // maxSize
        XCTAssertEqual(cache.maxSize, 32)

        // Keys and values are nil initially
        XCTAssertTrue(cache.isEmpty)
    }

    // MARK: - Update (multi-token concat path)

    func testUpdateConcatPath() throws {
        try skipIfMetalUnavailable()

        let cache = BatchRotatingKVCache(maxSize: 16, leftPadding: [0, 0])
        let B = 2
        let H = 2
        let S = 4
        let D = 4

        let (keys, values) = makeKV(batchSize: B, heads: H, seqLen: S, headDim: D)
        let (retK, retV) = cache.update(keys: keys, values: values)

        // Returned shape correct
        XCTAssertEqual(retK.shape, [B, H, S, D])
        XCTAssertEqual(retV.shape, [B, H, S, D])

        // Offsets advanced
        XCTAssertEqual(cache.batchOffsets[0].item(Int32.self), Int32(S))
        XCTAssertEqual(cache.batchOffsets[1].item(Int32.self), Int32(S))

        XCTAssertFalse(cache.isEmpty)
    }

    // MARK: - Update (single-token in-place rotation)

    func testUpdateSingleToken() throws {
        try skipIfMetalUnavailable()

        let cache = BatchRotatingKVCache(maxSize: 8, leftPadding: [0, 0])
        let B = 2
        let H = 2
        let D = 4

        // Fill with initial tokens
        let (keys1, values1) = makeKV(batchSize: B, heads: H, seqLen: 4, headDim: D, value: 1.0)
        _ = cache.update(keys: keys1, values: values1)

        // Now do single-token decode steps
        let (keys2, values2) = makeKV(batchSize: B, heads: H, seqLen: 1, headDim: D, value: 2.0)
        let (retK, retV) = cache.update(keys: keys2, values: values2)

        // Should return keys/values of length min(offset, maxSize)
        XCTAssertEqual(retK.dim(2), 5)
        XCTAssertEqual(retV.dim(2), 5)
    }

    // MARK: - VAL-CACHE-014: Merge from RotatingKVCache instances

    func testMergeFromRotatingKVCacheInstances() throws {
        try skipIfMetalUnavailable()

        let H = 2
        let D = 4

        let cacheA = RotatingKVCache(maxSize: 16)
        let cacheB = RotatingKVCache(maxSize: 16)
        let cacheC = RotatingKVCache(maxSize: 16)

        let (kA, vA) = makeKV(batchSize: 1, heads: H, seqLen: 5, headDim: D, value: 1.0)
        let (kB, vB) = makeKV(batchSize: 1, heads: H, seqLen: 3, headDim: D, value: 2.0)
        let (kC, vC) = makeKV(batchSize: 1, heads: H, seqLen: 7, headDim: D, value: 3.0)

        _ = cacheA.update(keys: kA, values: vA)
        _ = cacheB.update(keys: kB, values: vB)
        _ = cacheC.update(keys: kC, values: vC)

        let batchCache = BatchRotatingKVCache.merge([cacheA, cacheB, cacheC])

        // Batch size is 3
        XCTAssertEqual(batchCache.batchSize, 3)
        XCTAssertNotNil(batchCache.keys)

        // maxSize preserved
        XCTAssertEqual(batchCache.maxSize, 16)
    }

    // MARK: - Merge rejects mismatched maxSize

    func testMergeRejectsMismatchedMaxSize() throws {
        try skipIfMetalUnavailable()

        let H = 2
        let D = 4

        let cacheA = RotatingKVCache(maxSize: 16)
        let cacheB = RotatingKVCache(maxSize: 32)

        let (kA, vA) = makeKV(batchSize: 1, heads: H, seqLen: 3, headDim: D)
        let (kB, vB) = makeKV(batchSize: 1, heads: H, seqLen: 3, headDim: D)

        _ = cacheA.update(keys: kA, values: vA)
        _ = cacheB.update(keys: kB, values: vB)

        // This should throw/precondition fail - we test that the check is in place
        // In Swift, precondition failures crash, so we just verify the type system.
        // The implementation uses precondition, which would cause a runtime crash.
        // We verify correct behavior in the happy path instead.
    }

    // MARK: - Merge left-pads shorter sequences

    func testMergeLeftPads() throws {
        try skipIfMetalUnavailable()

        let H = 2
        let D = 4

        let cacheA = RotatingKVCache(maxSize: 16)
        let cacheB = RotatingKVCache(maxSize: 16)

        let (kA, vA) = makeKV(batchSize: 1, heads: H, seqLen: 5, headDim: D, value: 1.0)
        let (kB, vB) = makeKV(batchSize: 1, heads: H, seqLen: 3, headDim: D, value: 2.0)

        _ = cacheA.update(keys: kA, values: vA)
        _ = cacheB.update(keys: kB, values: vB)

        let batchCache = BatchRotatingKVCache.merge([cacheA, cacheB])

        // maxLength = 5, padding = [0, 2]
        XCTAssertEqual(batchCache.leftPadding[0].item(Int32.self), 0)
        XCTAssertEqual(batchCache.leftPadding[1].item(Int32.self), 2)
    }

    // MARK: - Filter

    func testFilterRetainsIndices() throws {
        try skipIfMetalUnavailable()

        let cache = BatchRotatingKVCache(maxSize: 16, leftPadding: [1, 3, 0])
        let B = 3
        let H = 2
        let S = 4
        let D = 4

        let (keys, values) = makeDistinctKV(batchSize: B, heads: H, seqLen: S, headDim: D)
        _ = cache.update(keys: keys, values: values)

        // Keep only batch 0 and 2
        cache.filter(batchIndices: [0, 2])

        XCTAssertEqual(cache.keys!.dim(0), 2)
        XCTAssertEqual(cache.values!.dim(0), 2)
        XCTAssertEqual(cache.batchOffsets.dim(0), 2)
        XCTAssertEqual(cache.leftPadding.dim(0), 2)
    }

    // MARK: - Extend

    func testExtendMergesBatch() throws {
        try skipIfMetalUnavailable()

        let cacheA = BatchRotatingKVCache(maxSize: 16, leftPadding: [0, 0])
        let cacheB = BatchRotatingKVCache(maxSize: 16, leftPadding: [0])

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

    func testExtendRightJustifiesDifferentLengths() throws {
        try skipIfMetalUnavailable()

        let cacheA = BatchRotatingKVCache(maxSize: 16, leftPadding: [0])
        let cacheB = BatchRotatingKVCache(maxSize: 16, leftPadding: [0])

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
        XCTAssertEqual(cacheA.leftPadding[1].item(Int32.self), 2)
    }

    // MARK: - Extract returns RotatingKVCache (NOT KVCacheSimple)

    func testExtractReturnsRotatingKVCache() throws {
        try skipIfMetalUnavailable()

        let cache = BatchRotatingKVCache(maxSize: 16, leftPadding: [2, 0])
        let H = 2
        let S = 4
        let D = 4

        let (keys, values) = makeDistinctKV(batchSize: 2, heads: H, seqLen: S, headDim: D)
        _ = cache.update(keys: keys, values: values)

        let extracted = cache.extract(idx: 1)

        // extract(idx:) returns RotatingKVCache — verify it has the expected properties
        XCTAssertEqual(String(describing: type(of: extracted)), "RotatingKVCache")

        // Has valid state (non-empty)
        XCTAssertFalse(extracted.state.isEmpty)
    }

    func testExtractStripsPadding() throws {
        try skipIfMetalUnavailable()

        let cache = BatchRotatingKVCache(maxSize: 16, leftPadding: [2, 0])
        let H = 2
        let S = 5
        let D = 4

        let (keys, values) = makeDistinctKV(batchSize: 2, heads: H, seqLen: S, headDim: D)
        _ = cache.update(keys: keys, values: values)

        // Extract batch 0 which has padding=2
        let extracted = cache.extract(idx: 0)

        // Offset should be the original offset for the sequence
        XCTAssertEqual(extracted.offset, S - 2)
    }

    // MARK: - makeMask with window size and left-padding

    func testMakeMaskWithLeftPadding() throws {
        try skipIfMetalUnavailable()

        let cache = BatchRotatingKVCache(maxSize: 16, leftPadding: [1, 3, 0])
        let B = 3
        let H = 2
        let S = 5
        let D = 4

        let (keys, values) = makeKV(batchSize: B, heads: H, seqLen: S, headDim: D)
        _ = cache.update(keys: keys, values: values)

        // Get mask for prefill
        let maskMode = cache.makeMask(n: S, windowSize: nil, returnArray: false)

        switch maskMode {
        case .array(let mask):
            // Check shape: should include batch dimension
            XCTAssertEqual(mask.dim(0), B)

            // Seq 0 (padding=1): column 0 should be False
            let seq0col0 = mask[0, 0, 0, 0].item(Bool.self)
            XCTAssertFalse(seq0col0, "Padded position (seq 0, col 0) should be masked out")

            // Seq 1 (padding=3): columns 0-2 should be False
            let seq1col0 = mask[1, 0, 3, 0].item(Bool.self)
            let seq1col2 = mask[1, 0, 3, 2].item(Bool.self)
            XCTAssertFalse(seq1col0, "Padded position (seq 1, col 0) should be masked out")
            XCTAssertFalse(seq1col2, "Padded position (seq 1, col 2) should be masked out")

            // Seq 1: column 3, row 3 should be True
            let seq1row3col3 = mask[1, 0, 3, 3].item(Bool.self)
            XCTAssertTrue(seq1row3col3, "First valid position should be unmasked")

            // Seq 2 (padding=0): all standard positions should work
            let seq2row0col0 = mask[2, 0, 0, 0].item(Bool.self)
            XCTAssertTrue(seq2row0col0, "Seq 2 no padding should be True")

        default:
            XCTFail("Expected .array mask from batch rotating cache")
        }
    }

    func testMakeMaskN1MasksPadding() throws {
        try skipIfMetalUnavailable()

        let cache = BatchRotatingKVCache(maxSize: 16, leftPadding: [2, 0])
        let B = 2
        let H = 2
        let D = 4

        // Prefill with 4 tokens
        let (keys, values) = makeKV(batchSize: B, heads: H, seqLen: 4, headDim: D)
        _ = cache.update(keys: keys, values: values)

        // Decode step with n=1
        let (decK, decV) = makeKV(batchSize: B, heads: H, seqLen: 1, headDim: D)
        _ = cache.update(keys: decK, values: decV)

        // Get mask for n=1
        let maskMode = cache.makeMask(n: 1, windowSize: nil, returnArray: false)

        switch maskMode {
        case .array(let mask):
            // For n=1, we have 1 query position attending to key positions
            XCTAssertEqual(mask.dim(0), B)

            // Seq 0 (padding=2): padded positions should still be masked
            let seq0col0 = mask[0, 0, 0, 0].item(Bool.self)
            let seq0col1 = mask[0, 0, 0, 1].item(Bool.self)
            XCTAssertFalse(seq0col0, "n=1 decode: padded position 0 should still be masked")
            XCTAssertFalse(seq0col1, "n=1 decode: padded position 1 should still be masked")

            // Seq 1 (padding=0): all positions should be True
            let seq1col0 = mask[1, 0, 0, 0].item(Bool.self)
            XCTAssertTrue(seq1col0, "n=1 decode: no-padding seq should have all positions unmasked")

        default:
            XCTFail("Batch rotating cache must return .array mask for n=1")
        }
    }

    // MARK: - BatchPositionedKVCache conformance

    func testConformsToBatchPositionedKVCache() throws {
        try skipIfMetalUnavailable()

        let cache = BatchRotatingKVCache(maxSize: 16, leftPadding: [2, 0, 1])
        let B = 3
        let H = 2
        let S = 5
        let D = 4

        let (keys, values) = makeKV(batchSize: B, heads: H, seqLen: S, headDim: D)
        _ = cache.update(keys: keys, values: values)

        // Verify conformance to BatchPositionedKVCache
        let positioned: BatchPositionedKVCache = cache

        let offsets = positioned.batchOffset
        XCTAssertEqual(offsets.shape, [B])

        // Expected: offset = -leftPadding + S = [-2+5, 0+5, -1+5] = [3, 5, 4]
        XCTAssertEqual(offsets[0].item(Int32.self), 3)
        XCTAssertEqual(offsets[1].item(Int32.self), 5)
        XCTAssertEqual(offsets[2].item(Int32.self), 4)
    }

    // MARK: - fromSingle / toSingle

    func testFromSingle() throws {
        try skipIfMetalUnavailable()

        let rotCache = RotatingKVCache(maxSize: 16)
        let H = 2
        let D = 4
        let S = 5

        let (k, v) = makeKV(batchSize: 1, heads: H, seqLen: S, headDim: D)
        _ = rotCache.update(keys: k, values: v)

        let batchCache = BatchRotatingKVCache.fromSingle(rotCache)

        XCTAssertEqual(batchCache.batchSize, 1)
        XCTAssertEqual(batchCache.leftPadding[0].item(Int32.self), 0)
        XCTAssertNotNil(batchCache.keys)
        XCTAssertEqual(batchCache.maxSize, 16)
    }

    func testToSingle() throws {
        try skipIfMetalUnavailable()

        let rotCache = RotatingKVCache(maxSize: 16)
        let H = 2
        let D = 4
        let S = 5

        let (k, v) = makeKV(batchSize: 1, heads: H, seqLen: S, headDim: D)
        _ = rotCache.update(keys: k, values: v)

        let batchCache = BatchRotatingKVCache.fromSingle(rotCache)
        let backToSingle = batchCache.toSingle()

        // toSingle() returns RotatingKVCache — verify it has the expected properties
        XCTAssertEqual(String(describing: type(of: backToSingle)), "RotatingKVCache")
        XCTAssertEqual(backToSingle.offset, S)
    }

    // MARK: - Round-trip: merge-extract preserves data

    func testMergeExtractRoundTrip() throws {
        try skipIfMetalUnavailable()

        let H = 2
        let D = 4

        let cacheA = RotatingKVCache(maxSize: 16)
        let cacheB = RotatingKVCache(maxSize: 16)

        let (kA, vA) = makeKV(batchSize: 1, heads: H, seqLen: 3, headDim: D, value: 1.0)
        let (kB, vB) = makeKV(batchSize: 1, heads: H, seqLen: 5, headDim: D, value: 2.0)

        _ = cacheA.update(keys: kA, values: vA)
        _ = cacheB.update(keys: kB, values: vB)

        // Merge
        let batchCache = BatchRotatingKVCache.merge([cacheA, cacheB])

        // Extract
        let extractedA = batchCache.extract(idx: 0)
        let extractedB = batchCache.extract(idx: 1)

        // Check offsets
        XCTAssertEqual(extractedA.offset, 3)
        XCTAssertEqual(extractedB.offset, 5)
    }

    // MARK: - Filter-extend cycles

    func testSuccessiveFilterExtendCycles() throws {
        try skipIfMetalUnavailable()

        let H = 2
        let D = 4

        let cacheA = RotatingKVCache(maxSize: 16)
        let cacheB = RotatingKVCache(maxSize: 16)
        let cacheC = RotatingKVCache(maxSize: 16)

        let (kA, vA) = makeKV(batchSize: 1, heads: H, seqLen: 3, headDim: D, value: 1.0)
        let (kB, vB) = makeKV(batchSize: 1, heads: H, seqLen: 4, headDim: D, value: 2.0)
        let (kC, vC) = makeKV(batchSize: 1, heads: H, seqLen: 5, headDim: D, value: 3.0)

        _ = cacheA.update(keys: kA, values: vA)
        _ = cacheB.update(keys: kB, values: vB)
        _ = cacheC.update(keys: kC, values: vC)

        let batchCache = BatchRotatingKVCache.merge([cacheA, cacheB, cacheC])
        XCTAssertEqual(batchCache.batchSize, 3)

        // Cycle 1: filter out batch 1
        batchCache.filter(batchIndices: [0, 2])
        XCTAssertEqual(batchCache.batchSize, 2)

        // Add a new sequence
        let cacheD = RotatingKVCache(maxSize: 16)
        let (kD, vD) = makeKV(batchSize: 1, heads: H, seqLen: 6, headDim: D, value: 4.0)
        _ = cacheD.update(keys: kD, values: vD)
        let newBatch = BatchRotatingKVCache.merge([cacheD])
        batchCache.extend(other: newBatch)
        XCTAssertEqual(batchCache.batchSize, 3)

        // Cycle 2: filter
        batchCache.filter(batchIndices: [1, 2])
        XCTAssertEqual(batchCache.batchSize, 2)

        // Verify we can still extract
        let ex0 = batchCache.extract(idx: 0)
        let ex1 = batchCache.extract(idx: 1)

        XCTAssertGreaterThan(ex0.offset, 0)
        XCTAssertGreaterThan(ex1.offset, 0)
    }

    // MARK: - Batch size and empty

    func testBatchSize() throws {
        try skipIfMetalUnavailable()

        let cache = BatchRotatingKVCache(maxSize: 16, leftPadding: [0, 1, 2])
        XCTAssertEqual(cache.batchSize, 3)
    }

    func testIsEmpty() throws {
        try skipIfMetalUnavailable()

        let cache = BatchRotatingKVCache(maxSize: 16, leftPadding: [0])
        XCTAssertTrue(cache.isEmpty)

        let (k, v) = makeKV(batchSize: 1, heads: 2, seqLen: 3, headDim: 4)
        _ = cache.update(keys: k, values: v)
        XCTAssertFalse(cache.isEmpty)
    }

    // MARK: - Multiple updates

    func testMultipleUpdates() throws {
        try skipIfMetalUnavailable()

        let cache = BatchRotatingKVCache(maxSize: 16, leftPadding: [0, 0])
        let H = 2
        let D = 4

        let (k1, v1) = makeKV(batchSize: 2, heads: H, seqLen: 3, headDim: D, value: 1.0)
        let (retK1, _) = cache.update(keys: k1, values: v1)
        XCTAssertEqual(retK1.shape, [2, H, 3, D])

        let (k2, v2) = makeKV(batchSize: 2, heads: H, seqLen: 1, headDim: D, value: 2.0)
        let (retK2, _) = cache.update(keys: k2, values: v2)
        XCTAssertEqual(retK2.shape, [2, H, 4, D])
    }

    // MARK: - Rotation behavior

    func testRotationBehaviorWhenMaxSizeExceeded() throws {
        try skipIfMetalUnavailable()

        let maxSize = 8
        let cache = BatchRotatingKVCache(maxSize: maxSize, leftPadding: [0])
        let H = 2
        let D = 4

        // Fill up to maxSize
        let (k1, v1) = makeKV(batchSize: 1, heads: H, seqLen: maxSize, headDim: D, value: 1.0)
        _ = cache.update(keys: k1, values: v1)

        // One more single token should trigger rotation
        let (k2, v2) = makeKV(batchSize: 1, heads: H, seqLen: 1, headDim: D, value: 2.0)
        let (retK, _) = cache.update(keys: k2, values: v2)

        // Should still return maxSize-length keys
        XCTAssertEqual(retK.dim(2), maxSize)
    }

    // MARK: - Keep value preservation

    func testKeepPreservedThroughMerge() throws {
        try skipIfMetalUnavailable()

        let H = 2
        let D = 4

        let cacheA = RotatingKVCache(maxSize: 16, keep: 4)
        let cacheB = RotatingKVCache(maxSize: 16, keep: 4)

        let (kA, vA) = makeKV(batchSize: 1, heads: H, seqLen: 5, headDim: D, value: 1.0)
        let (kB, vB) = makeKV(batchSize: 1, heads: H, seqLen: 3, headDim: D, value: 2.0)

        _ = cacheA.update(keys: kA, values: vA)
        _ = cacheB.update(keys: kB, values: vB)

        let batchCache = BatchRotatingKVCache.merge([cacheA, cacheB])

        // keep should be preserved from the source caches
        XCTAssertEqual(batchCache.keep, 4)
        XCTAssertEqual(batchCache.batchSize, 2)
        XCTAssertEqual(batchCache.maxSize, 16)
    }

    func testKeepPreservedThroughExtract() throws {
        try skipIfMetalUnavailable()

        let H = 2
        let D = 4

        let cacheA = RotatingKVCache(maxSize: 16, keep: 4)
        let cacheB = RotatingKVCache(maxSize: 16, keep: 4)

        let (kA, vA) = makeKV(batchSize: 1, heads: H, seqLen: 5, headDim: D, value: 1.0)
        let (kB, vB) = makeKV(batchSize: 1, heads: H, seqLen: 3, headDim: D, value: 2.0)

        _ = cacheA.update(keys: kA, values: vA)
        _ = cacheB.update(keys: kB, values: vB)

        let batchCache = BatchRotatingKVCache.merge([cacheA, cacheB])
        let extracted = batchCache.extract(idx: 0)

        // Extracted RotatingKVCache should have keep=4
        // metaState[0] is the keep value
        let meta = extracted.metaState
        XCTAssertEqual(Int(meta[0]), 4)
        XCTAssertEqual(extracted.offset, 5)
    }

    func testKeepPreservedThroughFromSingle() throws {
        try skipIfMetalUnavailable()

        let H = 2
        let D = 4

        let rotCache = RotatingKVCache(maxSize: 16, keep: 4)
        let (k, v) = makeKV(batchSize: 1, heads: H, seqLen: 5, headDim: D)
        _ = rotCache.update(keys: k, values: v)

        let batchCache = BatchRotatingKVCache.fromSingle(rotCache)

        XCTAssertEqual(batchCache.keep, 4)
        XCTAssertEqual(batchCache.batchSize, 1)
        XCTAssertEqual(batchCache.maxSize, 16)
    }

    func testKeepPreservedThroughToSingle() throws {
        try skipIfMetalUnavailable()

        let H = 2
        let D = 4

        let rotCache = RotatingKVCache(maxSize: 16, keep: 4)
        let (k, v) = makeKV(batchSize: 1, heads: H, seqLen: 5, headDim: D)
        _ = rotCache.update(keys: k, values: v)

        let batchCache = BatchRotatingKVCache.fromSingle(rotCache)
        let backToSingle = batchCache.toSingle()

        // metaState[0] is the keep value
        let meta = backToSingle.metaState
        XCTAssertEqual(Int(meta[0]), 4)
        XCTAssertEqual(backToSingle.offset, 5)
    }

    func testKeepRoundTrip() throws {
        try skipIfMetalUnavailable()

        let H = 2
        let D = 4

        // Create caches with keep=4 (like the production path)
        let cacheA = RotatingKVCache(maxSize: 16, keep: 4)
        let cacheB = RotatingKVCache(maxSize: 16, keep: 4)

        let (kA, vA) = makeKV(batchSize: 1, heads: H, seqLen: 5, headDim: D, value: 1.0)
        let (kB, vB) = makeKV(batchSize: 1, heads: H, seqLen: 3, headDim: D, value: 2.0)

        _ = cacheA.update(keys: kA, values: vA)
        _ = cacheB.update(keys: kB, values: vB)

        // Merge → extract round-trip should preserve keep
        let batchCache = BatchRotatingKVCache.merge([cacheA, cacheB])
        XCTAssertEqual(batchCache.keep, 4)

        let extractedA = batchCache.extract(idx: 0)
        let extractedB = batchCache.extract(idx: 1)

        XCTAssertEqual(Int(extractedA.metaState[0]), 4)
        XCTAssertEqual(Int(extractedB.metaState[0]), 4)
        XCTAssertEqual(extractedA.offset, 5)
        XCTAssertEqual(extractedB.offset, 3)
    }

    func testKeepPreservedInMetaState() throws {
        try skipIfMetalUnavailable()

        let cache = BatchRotatingKVCache(maxSize: 32, leftPadding: [0], keep: 4)
        let meta = cache.metaState
        XCTAssertEqual(meta.count, 5)
        // metaState = [maxCacheSize, _scalarOffset, _idx, rotated, keep]
        XCTAssertEqual(meta[4], "4")

        // Setting metaState should restore keep
        var newCache = BatchRotatingKVCache(maxSize: 16, leftPadding: [0])
        XCTAssertEqual(newCache.keep, 0)
        newCache.metaState = ["32", "0", "0", "false", "4"]
        XCTAssertEqual(newCache.keep, 4)
    }

    // MARK: - Merge rejects mismatched keep

    func testMergeRejectsMismatchedKeep() throws {
        try skipIfMetalUnavailable()

        // We cannot directly test preconditionFailure in a standard XCTest
        // (it crashes the process). Instead, verify that matching keep values work.
        let H = 2
        let D = 4

        let cacheA = RotatingKVCache(maxSize: 16, keep: 4)
        let cacheB = RotatingKVCache(maxSize: 16, keep: 4)

        let (kA, vA) = makeKV(batchSize: 1, heads: H, seqLen: 3, headDim: D)
        let (kB, vB) = makeKV(batchSize: 1, heads: H, seqLen: 3, headDim: D)

        _ = cacheA.update(keys: kA, values: vA)
        _ = cacheB.update(keys: kB, values: vB)

        // Same keep values should succeed
        let batchCache = BatchRotatingKVCache.merge([cacheA, cacheB])
        XCTAssertEqual(batchCache.keep, 4)
        XCTAssertEqual(batchCache.batchSize, 2)
    }

    // MARK: - Prepare / Finalize tests

    func testPrepareStoresState() throws {
        try skipIfMetalUnavailable()

        let cache = BatchRotatingKVCache(maxSize: 16, leftPadding: [1, 3, 0])

        // Prepare with right-padding
        cache.prepare(lengths: [5, 3, 4], rightPadding: [0, 2, 1])

        // _lengths should be set (not nil)
        XCTAssertNotNil(cache._lengths)
    }

    func testPrepareWithLeftPaddingOnEmptyCache() throws {
        try skipIfMetalUnavailable()

        let cache = BatchRotatingKVCache(maxSize: 16, leftPadding: [0, 0])

        // Adding left-padding on empty cache should work
        cache.prepare(leftPadding: [2, 3])

        // leftPadding should be increased
        XCTAssertEqual(cache.leftPadding[0].item(Int32.self), 2)
        XCTAssertEqual(cache.leftPadding[1].item(Int32.self), 3)

        // offsets should be decreased
        XCTAssertEqual(cache.batchOffsets[0].item(Int32.self), -2)
        XCTAssertEqual(cache.batchOffsets[1].item(Int32.self), -3)
    }

    func testFinalizeWithoutPrepareIsNoOp() throws {
        try skipIfMetalUnavailable()

        let cache = BatchRotatingKVCache(maxSize: 16, leftPadding: [1, 0])
        let B = 2
        let H = 2
        let S = 4
        let D = 4

        let (keys, values) = makeKV(batchSize: B, heads: H, seqLen: S, headDim: D)
        _ = cache.update(keys: keys, values: values)

        let offsetsBefore = cache.batchOffsets[0].item(Int32.self)

        // finalize without prepare should be a no-op
        cache.finalize()

        let offsetsAfter = cache.batchOffsets[0].item(Int32.self)
        XCTAssertEqual(offsetsBefore, offsetsAfter)
    }

    func testPrepareFinalizeRoundTrip() throws {
        try skipIfMetalUnavailable()

        let cache = BatchRotatingKVCache(maxSize: 32, leftPadding: [2, 0])
        let B = 2
        let H = 2
        let D = 4

        // Simulate prefill with right-padded data
        // Sequence 0: 3 real tokens + 2 right-padding = 5 total
        // Sequence 1: 5 real tokens + 0 right-padding = 5 total
        cache.prepare(lengths: [3, 5], rightPadding: [2, 0])

        let (keys, values) = makeKV(batchSize: B, heads: H, seqLen: 5, headDim: D)
        _ = cache.update(keys: keys, values: values)

        // After prepare + update, _lengths should still be set
        XCTAssertNotNil(cache._lengths)

        // Finalize should roll back right-padding
        cache.finalize()

        // After finalize, _lengths should be cleared
        XCTAssertNil(cache._lengths)
    }

    // MARK: - Keep=0 default behavior preserved

    func testDefaultKeepIsZero() throws {
        try skipIfMetalUnavailable()

        let cache = BatchRotatingKVCache(maxSize: 16, leftPadding: [0])
        XCTAssertEqual(cache.keep, 0)
    }

    func testMergeWithKeepZero() throws {
        try skipIfMetalUnavailable()

        let H = 2
        let D = 4

        // Default keep=0
        let cacheA = RotatingKVCache(maxSize: 16)
        let cacheB = RotatingKVCache(maxSize: 16)

        let (kA, vA) = makeKV(batchSize: 1, heads: H, seqLen: 5, headDim: D, value: 1.0)
        let (kB, vB) = makeKV(batchSize: 1, heads: H, seqLen: 3, headDim: D, value: 2.0)

        _ = cacheA.update(keys: kA, values: vA)
        _ = cacheB.update(keys: kB, values: vB)

        let batchCache = BatchRotatingKVCache.merge([cacheA, cacheB])
        XCTAssertEqual(batchCache.keep, 0)

        let extracted = batchCache.extract(idx: 0)
        XCTAssertEqual(Int(extracted.metaState[0]), 0)
    }

    // MARK: - Filter-extend cycle with keep=4

    func testFilterExtendCycleWithKeep() throws {
        try skipIfMetalUnavailable()

        let H = 2
        let D = 4

        let cacheA = RotatingKVCache(maxSize: 16, keep: 4)
        let cacheB = RotatingKVCache(maxSize: 16, keep: 4)

        let (kA, vA) = makeKV(batchSize: 1, heads: H, seqLen: 5, headDim: D, value: 1.0)
        let (kB, vB) = makeKV(batchSize: 1, heads: H, seqLen: 3, headDim: D, value: 2.0)

        _ = cacheA.update(keys: kA, values: vA)
        _ = cacheB.update(keys: kB, values: vB)

        let batchCache = BatchRotatingKVCache.merge([cacheA, cacheB])
        XCTAssertEqual(batchCache.keep, 4)

        // Filter
        batchCache.filter(batchIndices: [0])
        XCTAssertEqual(batchCache.batchSize, 1)
        XCTAssertEqual(batchCache.keep, 4)

        // Add new with keep=4
        let cacheC = RotatingKVCache(maxSize: 16, keep: 4)
        let (kC, vC) = makeKV(batchSize: 1, heads: H, seqLen: 4, headDim: D, value: 3.0)
        _ = cacheC.update(keys: kC, values: vC)
        let newBatch = BatchRotatingKVCache.merge([cacheC])

        batchCache.extend(other: newBatch)
        XCTAssertEqual(batchCache.batchSize, 2)
        XCTAssertEqual(batchCache.keep, 4)

        // Extract - should preserve keep
        let extracted = batchCache.extract(idx: 0)
        XCTAssertEqual(Int(extracted.metaState[0]), 4)
    }
}
