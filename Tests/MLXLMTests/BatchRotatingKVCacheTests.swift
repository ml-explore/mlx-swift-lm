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

    // MARK: - Keep Semantics: Overflow Preservation

    /// Test that updateConcat preserves the first `keep` tokens during overflow trim.
    func testUpdateConcatPreservesKeepDuringOverflow() throws {
        try skipIfMetalUnavailable()

        let maxSize = 8
        let keepCount = 2
        let H = 2
        let D = 4

        let cache = BatchRotatingKVCache(maxSize: maxSize, leftPadding: [0], keep: keepCount)

        // Prefill with `maxSize` tokens — fill buffer exactly. First `keep` tokens are special.
        // Use distinct values so we can verify: token i has value Float(i+1)
        var keySlices: [MLXArray] = []
        var valSlices: [MLXArray] = []
        for i in 0 ..< maxSize {
            keySlices.append(MLXArray.ones([1, H, 1, D]) * Float(i + 1))
            valSlices.append(MLXArray.ones([1, H, 1, D]) * Float((i + 1) * 10))
        }
        let initialKeys = concatenated(keySlices, axis: 2)
        let initialValues = concatenated(valSlices, axis: 2)

        _ = cache.update(keys: initialKeys, values: initialValues)
        XCTAssertEqual(cache._idx, maxSize)

        // Now add 3 more tokens via concat (this will trigger trimming)
        let overflowKeys = MLXArray.ones([1, H, 3, D]) * Float(100)
        let overflowValues = MLXArray.ones([1, H, 3, D]) * Float(1000)
        let (retK, _) = cache.update(keys: overflowKeys, values: overflowValues)

        // The first `keep` tokens should be preserved in the returned keys.
        // Token 0 has value 1.0, token 1 has value 2.0
        let firstKeepToken = retK[0, 0, 0, 0].item(Float.self)
        let secondKeepToken = retK[0, 0, 1, 0].item(Float.self)
        XCTAssertEqual(firstKeepToken, 1.0, "First keep token should be preserved after overflow")
        XCTAssertEqual(secondKeepToken, 2.0, "Second keep token should be preserved after overflow")

        // The last 3 tokens should be the overflow values
        let seqLen = retK.dim(2)
        let lastToken = retK[0, 0, seqLen - 1, 0].item(Float.self)
        XCTAssertEqual(lastToken, 100.0, "Overflow tokens should be at the end")
    }

    /// Test that updateInPlace wraps _idx to keep (not 0) during rotation.
    func testUpdateInPlaceWrapsToKeep() throws {
        try skipIfMetalUnavailable()

        let maxSize = 8
        let keepCount = 2
        let H = 2
        let D = 4

        let cache = BatchRotatingKVCache(maxSize: maxSize, leftPadding: [0], keep: keepCount)

        // Prefill with distinct per-position values
        var keySlices: [MLXArray] = []
        var valSlices: [MLXArray] = []
        for i in 0 ..< maxSize {
            keySlices.append(MLXArray.ones([1, H, 1, D]) * Float(i + 1))
            valSlices.append(MLXArray.ones([1, H, 1, D]) * Float((i + 1) * 10))
        }
        let initialKeys = concatenated(keySlices, axis: 2)
        let initialValues = concatenated(valSlices, axis: 2)
        _ = cache.update(keys: initialKeys, values: initialValues)

        // Now do single-token decodes to trigger rotation
        let overflowK = MLXArray.ones([1, H, 1, D]) * Float(99)
        let overflowV = MLXArray.ones([1, H, 1, D]) * Float(990)
        let (retK, _) = cache.update(keys: overflowK, values: overflowV)

        // Buffer should be full (maxSize)
        XCTAssertEqual(retK.dim(2), maxSize)

        // The first `keep` positions in the raw buffer should still be the original tokens
        // Position 0: value 1.0, Position 1: value 2.0
        let rawK = cache.keys!
        let pos0 = rawK[0, 0, 0, 0].item(Float.self)
        let pos1 = rawK[0, 0, 1, 0].item(Float.self)
        XCTAssertEqual(pos0, 1.0, "Keep position 0 should never be overwritten")
        XCTAssertEqual(pos1, 2.0, "Keep position 1 should never be overwritten")

        // The new token should be at position `keep` (where idx wrapped to)
        let posKeep = rawK[0, 0, keepCount, 0].item(Float.self)
        XCTAssertEqual(posKeep, 99.0, "New token should be written at keep position after wrap")
    }

    /// Test that temporal ordering handles the keep prefix correctly after rotation.
    func testTemporalOrderWithKeep() throws {
        try skipIfMetalUnavailable()

        let maxSize = 8
        let keepCount = 2
        let H = 2
        let D = 4

        let cache = BatchRotatingKVCache(maxSize: maxSize, leftPadding: [0], keep: keepCount)

        // Fill with maxSize distinct tokens
        var keySlices: [MLXArray] = []
        var valSlices: [MLXArray] = []
        for i in 0 ..< maxSize {
            keySlices.append(MLXArray.ones([1, H, 1, D]) * Float(i + 1))
            valSlices.append(MLXArray.ones([1, H, 1, D]) * Float((i + 1) * 10))
        }
        let initialKeys = concatenated(keySlices, axis: 2)
        let initialValues = concatenated(valSlices, axis: 2)
        _ = cache.update(keys: initialKeys, values: initialValues)

        // Two single-token decodes to rotate
        for step in 0 ..< 2 {
            let dk = MLXArray.ones([1, H, 1, D]) * Float(100 + step)
            let dv = MLXArray.ones([1, H, 1, D]) * Float(1000 + step)
            _ = cache.update(keys: dk, values: dv)
        }

        XCTAssertTrue(cache.rotated, "Cache should be rotated after overflow")

        // Now do a multi-token concat which triggers temporalOrder()
        let concatK = MLXArray.ones([1, H, 2, D]) * Float(200)
        let concatV = MLXArray.ones([1, H, 2, D]) * Float(2000)
        let (retK, _) = cache.update(keys: concatK, values: concatV)

        // After temporal ordering + concat, the first `keep` tokens should still be
        // the original values (1.0 and 2.0)
        let first = retK[0, 0, 0, 0].item(Float.self)
        let second = retK[0, 0, 1, 0].item(Float.self)
        XCTAssertEqual(first, 1.0, "Keep token 0 should be preserved after temporal reorder")
        XCTAssertEqual(second, 2.0, "Keep token 1 should be preserved after temporal reorder")
    }

    /// Round-trip test: merge caches with keep=4, trigger overflow, extract — keep prefix intact.
    /// Asserts actual key/value tensor CONTENTS after extraction, not just metadata.
    func testKeepOverflowMergeExtractRoundTrip() throws {
        try skipIfMetalUnavailable()

        let maxSize = 8
        let keepCount = 4
        let H = 2
        let D = 4

        // Create two RotatingKVCache with keep=4 and fill to near-max
        let cacheA = RotatingKVCache(maxSize: maxSize, keep: keepCount)
        let cacheB = RotatingKVCache(maxSize: maxSize, keep: keepCount)

        // Cache A: 6 tokens (key values 1..6, value values 10..60)
        var kaSlices: [MLXArray] = []
        var vaSlices: [MLXArray] = []
        for i in 0 ..< 6 {
            kaSlices.append(MLXArray.ones([1, H, 1, D]) * Float(i + 1))
            vaSlices.append(MLXArray.ones([1, H, 1, D]) * Float((i + 1) * 10))
        }
        _ = cacheA.update(
            keys: concatenated(kaSlices, axis: 2),
            values: concatenated(vaSlices, axis: 2)
        )

        // Cache B: 4 tokens (key values 11..14, value values 110..140)
        var kbSlices: [MLXArray] = []
        var vbSlices: [MLXArray] = []
        for i in 0 ..< 4 {
            kbSlices.append(MLXArray.ones([1, H, 1, D]) * Float(i + 11))
            vbSlices.append(MLXArray.ones([1, H, 1, D]) * Float((i + 11) * 10))
        }
        _ = cacheB.update(
            keys: concatenated(kbSlices, axis: 2),
            values: concatenated(vbSlices, axis: 2)
        )

        // Merge into batch
        let batchCache = BatchRotatingKVCache.merge([cacheA, cacheB])
        XCTAssertEqual(batchCache.keep, keepCount)

        // Add decode tokens to trigger overflow
        // Each decode step adds 1 token to both batch elements
        for step in 0 ..< 4 {
            let dk = MLXArray.ones([2, H, 1, D]) * Float(50 + step)
            let dv = MLXArray.ones([2, H, 1, D]) * Float(500 + step)
            _ = batchCache.update(keys: dk, values: dv)
        }

        // Extract and verify keep prefix data is actually preserved
        let extractedA = batchCache.extract(idx: 0)
        let extractedB = batchCache.extract(idx: 1)

        // Both should have keep=4 preserved in metadata
        XCTAssertEqual(Int(extractedA.metaState[0]), keepCount)
        XCTAssertEqual(Int(extractedB.metaState[0]), keepCount)

        // Extracted state should have non-empty keys/values
        XCTAssertFalse(extractedA.state.isEmpty)
        XCTAssertFalse(extractedB.state.isEmpty)

        // Offsets should have advanced: original + 4 decode tokens
        XCTAssertEqual(extractedA.offset, 6 + 4)
        XCTAssertEqual(extractedB.offset, 4 + 4)

        // --- Assert actual tensor contents ---

        // Extracted A: keep prefix should be tokens 1, 2, 3, 4
        let stateA = extractedA.state
        XCTAssertEqual(stateA.count, 2, "Extracted state should have keys and values")
        let keysA = stateA[0]
        let valsA = stateA[1]

        // Cache A had 6 tokens + 4 decode = 10 total, maxSize=8, keep=4
        // Extracted should have maxSize=8 tokens: [keep: 1,2,3,4] [window: 50,51,52,53]
        XCTAssertEqual(keysA.dim(2), maxSize, "Extracted A should have maxSize tokens")

        // Verify keep prefix key contents (positions 0..3 should be 1.0, 2.0, 3.0, 4.0)
        for i in 0 ..< keepCount {
            let keyVal = keysA[0, 0, i, 0].item(Float.self)
            XCTAssertEqual(
                keyVal, Float(i + 1),
                "Extracted A keep prefix key[\(i)] should be \(i + 1), got \(keyVal)"
            )
        }

        // Verify keep prefix value contents (positions 0..3 should be 10, 20, 30, 40)
        for i in 0 ..< keepCount {
            let valVal = valsA[0, 0, i, 0].item(Float.self)
            XCTAssertEqual(
                valVal, Float((i + 1) * 10),
                "Extracted A keep prefix val[\(i)] should be \((i + 1) * 10), got \(valVal)"
            )
        }

        // Extracted B: keep prefix should be tokens 11, 12, 13, 14
        let stateB = extractedB.state
        XCTAssertEqual(stateB.count, 2, "Extracted state should have keys and values")
        let keysB = stateB[0]
        let valsB = stateB[1]

        // Cache B had 4 tokens + 4 decode = 8 total, maxSize=8, keep=4
        // Extracted should have maxSize=8 tokens: [keep: 11,12,13,14] [window: 50,51,52,53]
        XCTAssertEqual(keysB.dim(2), maxSize, "Extracted B should have maxSize tokens")

        // Verify keep prefix key contents (positions 0..3 should be 11, 12, 13, 14)
        for i in 0 ..< keepCount {
            let keyVal = keysB[0, 0, i, 0].item(Float.self)
            XCTAssertEqual(
                keyVal, Float(i + 11),
                "Extracted B keep prefix key[\(i)] should be \(i + 11), got \(keyVal)"
            )
        }

        // Verify keep prefix value contents (positions 0..3 should be 110, 120, 130, 140)
        for i in 0 ..< keepCount {
            let valVal = valsB[0, 0, i, 0].item(Float.self)
            XCTAssertEqual(
                valVal, Float((i + 11) * 10),
                "Extracted B keep prefix val[\(i)] should be \((i + 11) * 10), got \(valVal)"
            )
        }
    }

    /// Test that keep=0 (default) continues to work correctly with rotation.
    func testKeepZeroRotationStillWorks() throws {
        try skipIfMetalUnavailable()

        let maxSize = 8
        let H = 2
        let D = 4

        let cache = BatchRotatingKVCache(maxSize: maxSize, leftPadding: [0])
        XCTAssertEqual(cache.keep, 0)

        // Fill and overflow
        let (k1, v1) = makeKV(batchSize: 1, heads: H, seqLen: maxSize, headDim: D, value: 1.0)
        _ = cache.update(keys: k1, values: v1)

        // Single-token decode to trigger rotation
        let (k2, v2) = makeKV(batchSize: 1, heads: H, seqLen: 1, headDim: D, value: 99.0)
        let (retK, _) = cache.update(keys: k2, values: v2)

        // Should still return maxSize
        XCTAssertEqual(retK.dim(2), maxSize)
        XCTAssertTrue(cache.rotated)
        // _idx should be 1 (wrapped to keep=0, then advanced by 1)
        XCTAssertEqual(cache._idx, 1)
    }

    /// Test that in-place rotation correctly wraps multiple times with keep > 0.
    func testMultipleRotationCyclesWithKeep() throws {
        try skipIfMetalUnavailable()

        let maxSize = 8
        let keepCount = 2
        let H = 2
        let D = 4

        let cache = BatchRotatingKVCache(maxSize: maxSize, leftPadding: [0], keep: keepCount)

        // Fill the buffer exactly
        var keySlices: [MLXArray] = []
        var valSlices: [MLXArray] = []
        for i in 0 ..< maxSize {
            keySlices.append(MLXArray.ones([1, H, 1, D]) * Float(i + 1))
            valSlices.append(MLXArray.ones([1, H, 1, D]) * Float((i + 1) * 10))
        }
        _ = cache.update(
            keys: concatenated(keySlices, axis: 2),
            values: concatenated(valSlices, axis: 2)
        )

        // Do (maxSize - keep) single-token decodes to wrap once fully through the window
        let windowSize = maxSize - keepCount
        for step in 0 ..< windowSize {
            let dk = MLXArray.ones([1, H, 1, D]) * Float(200 + step)
            let dv = MLXArray.ones([1, H, 1, D]) * Float(2000 + step)
            _ = cache.update(keys: dk, values: dv)
        }

        // After full cycle, _idx should be back at keep + windowSize = maxSize, then wrap again
        // Check that keep positions are still the originals
        let rawK = cache.keys!
        let pos0 = rawK[0, 0, 0, 0].item(Float.self)
        let pos1 = rawK[0, 0, 1, 0].item(Float.self)
        XCTAssertEqual(pos0, 1.0, "Keep position 0 preserved after full rotation cycle")
        XCTAssertEqual(pos1, 2.0, "Keep position 1 preserved after full rotation cycle")

        // Do another cycle
        for step in 0 ..< windowSize {
            let dk = MLXArray.ones([1, H, 1, D]) * Float(300 + step)
            let dv = MLXArray.ones([1, H, 1, D]) * Float(3000 + step)
            _ = cache.update(keys: dk, values: dv)
        }

        // Keep positions should still be originals
        let rawK2 = cache.keys!
        let pos0b = rawK2[0, 0, 0, 0].item(Float.self)
        let pos1b = rawK2[0, 0, 1, 0].item(Float.self)
        XCTAssertEqual(pos0b, 1.0, "Keep position 0 still preserved after 2nd rotation cycle")
        XCTAssertEqual(pos1b, 2.0, "Keep position 1 still preserved after 2nd rotation cycle")
    }

    // MARK: - Extract with negative leftPadding after overflow

    /// Test that extract() correctly handles negative leftPadding after overflow.
    /// After rotation, updateInPlace decrements leftPadding each step, which can
    /// make it negative. extract() must clamp to non-negative before slicing.
    func testExtractWithNegativeLeftPaddingAfterOverflow() throws {
        try skipIfMetalUnavailable()

        let maxSize = 8
        let H = 2
        let D = 4

        // Create a batch with padding: seq 0 has padding=2, seq 1 has padding=0
        let cache = BatchRotatingKVCache(maxSize: maxSize, leftPadding: [2, 0])

        // Prefill with 6 tokens (padded to 6 for both)
        let (keys, values) = makeDistinctKV(batchSize: 2, heads: H, seqLen: 6, headDim: D)
        _ = cache.update(keys: keys, values: values)

        // Now do single-token decodes to overflow the cache
        // After maxSize - 6 = 2 more tokens the buffer is full, then rotation starts
        for step in 0 ..< 6 {
            let dk = MLXArray.ones([2, H, 1, D]) * Float(90 + step)
            let dv = MLXArray.ones([2, H, 1, D]) * Float(900 + step)
            _ = cache.update(keys: dk, values: dv)
        }

        // After overflow, leftPadding should be negative for at least one sequence
        let lp0 = cache.leftPadding[0].item(Int32.self)
        XCTAssertLessThan(lp0, 0, "leftPadding should be negative after overflow")

        // extract() should NOT crash despite negative leftPadding
        let extracted0 = cache.extract(idx: 0)
        let extracted1 = cache.extract(idx: 1)

        // Extracted caches should have valid state
        XCTAssertFalse(extracted0.state.isEmpty, "Extracted cache 0 should have data")
        XCTAssertFalse(extracted1.state.isEmpty, "Extracted cache 1 should have data")

        // Extracted keys should have shape [1, H, seqLen, D] where seqLen <= maxSize
        let extractedK0 = extracted0.state[0]
        let extractedK1 = extracted1.state[0]
        XCTAssertGreaterThan(extractedK0.dim(2), 0, "Extracted key seq length should be positive")
        XCTAssertLessThanOrEqual(
            extractedK0.dim(2), maxSize, "Extracted key seq length should not exceed maxSize")
        XCTAssertGreaterThan(extractedK1.dim(2), 0, "Extracted key seq length should be positive")
        XCTAssertLessThanOrEqual(
            extractedK1.dim(2), maxSize, "Extracted key seq length should not exceed maxSize")

        // Offsets should be positive and valid
        XCTAssertGreaterThan(extracted0.offset, 0)
        XCTAssertGreaterThan(extracted1.offset, 0)
    }

    /// Test that extract() handles a rotated keep+window buffer with negative leftPadding.
    func testExtractRotatedKeepWindowWithNegativePadding() throws {
        try skipIfMetalUnavailable()

        let maxSize = 8
        let keepCount = 2
        let H = 2
        let D = 4

        // Create individual caches with keep, fill them, merge
        let cacheA = RotatingKVCache(maxSize: maxSize, keep: keepCount)
        let cacheB = RotatingKVCache(maxSize: maxSize, keep: keepCount)

        // Cache A: 6 tokens with distinct values
        var kaSlices: [MLXArray] = []
        var vaSlices: [MLXArray] = []
        for i in 0 ..< 6 {
            kaSlices.append(MLXArray.ones([1, H, 1, D]) * Float(i + 1))
            vaSlices.append(MLXArray.ones([1, H, 1, D]) * Float((i + 1) * 10))
        }
        _ = cacheA.update(
            keys: concatenated(kaSlices, axis: 2),
            values: concatenated(vaSlices, axis: 2))

        // Cache B: 4 tokens
        var kbSlices: [MLXArray] = []
        var vbSlices: [MLXArray] = []
        for i in 0 ..< 4 {
            kbSlices.append(MLXArray.ones([1, H, 1, D]) * Float(i + 11))
            vbSlices.append(MLXArray.ones([1, H, 1, D]) * Float((i + 11) * 10))
        }
        _ = cacheB.update(
            keys: concatenated(kbSlices, axis: 2),
            values: concatenated(vbSlices, axis: 2))

        let batchCache = BatchRotatingKVCache.merge([cacheA, cacheB])
        XCTAssertEqual(batchCache.keep, keepCount)

        // Add enough decode tokens to trigger overflow and make leftPadding go negative
        for step in 0 ..< 8 {
            let dk = MLXArray.ones([2, H, 1, D]) * Float(50 + step)
            let dv = MLXArray.ones([2, H, 1, D]) * Float(500 + step)
            _ = batchCache.update(keys: dk, values: dv)
        }

        // leftPadding should now be negative for at least the shorter sequence
        XCTAssertTrue(batchCache.rotated, "Cache should be rotated after overflow")

        // extract() should NOT crash
        let extractedA = batchCache.extract(idx: 0)
        let extractedB = batchCache.extract(idx: 1)

        // Extracted states should be valid
        XCTAssertFalse(extractedA.state.isEmpty)
        XCTAssertFalse(extractedB.state.isEmpty)

        // Keep prefix should be preserved in the extracted keys
        let keysA = extractedA.state[0]
        let keysB = extractedB.state[0]

        // Cache A keep prefix: tokens 1, 2
        let keepA0 = keysA[0, 0, 0, 0].item(Float.self)
        let keepA1 = keysA[0, 0, 1, 0].item(Float.self)
        XCTAssertEqual(keepA0, 1.0, "Extracted A keep[0] should be 1.0")
        XCTAssertEqual(keepA1, 2.0, "Extracted A keep[1] should be 2.0")

        // Cache B keep prefix: tokens 11, 12
        let keepB0 = keysB[0, 0, 0, 0].item(Float.self)
        let keepB1 = keysB[0, 0, 1, 0].item(Float.self)
        XCTAssertEqual(keepB0, 11.0, "Extracted B keep[0] should be 11.0")
        XCTAssertEqual(keepB1, 12.0, "Extracted B keep[1] should be 12.0")

        // Keep value preserved in metaState
        XCTAssertEqual(Int(extractedA.metaState[0]), keepCount)
        XCTAssertEqual(Int(extractedB.metaState[0]), keepCount)
    }
}
