// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXNN
@testable import MLXLMCommon
import XCTest

// MARK: - BatchMaskingAndPositionTests

final class BatchMaskingAndPositionTests: XCTestCase {

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

    // MARK: - VAL-CACHE-012: createCausalMask with leftPadding masks padding positions

    func testCreateCausalMaskWithLeftPadding() throws {
        try skipIfMetalUnavailable()

        // 2 sequences: sequence 0 has 1 padding position, sequence 1 has 2
        let leftPadding = MLXArray([Int32(1), Int32(2)])
        let n = 4
        let offset = 0

        let mask = createCausalMask(
            n: n, offset: offset, leftPadding: leftPadding
        )

        // mask shape should be [2, 1, 4, 4] (B=2, broadcast over heads, n=4, total_len=4)
        XCTAssertEqual(mask.ndim, 4)
        XCTAssertEqual(mask.dim(0), 2)  // batch
        XCTAssertEqual(mask.dim(2), n)  // query sequence
        XCTAssertEqual(mask.dim(3), n)  // key sequence

        // For sequence 0 (leftPadding=1): column 0 should be masked (False)
        // Position 0 is padded, so mask[0, :, :, 0] should be False
        let seq0col0 = mask[0, 0, 0, 0].item(Bool.self)
        XCTAssertFalse(seq0col0, "Padded position (seq 0, col 0) should be masked out")

        // For sequence 0: column 1 at row 1 should be True (valid position, causal ok)
        let seq0row1col1 = mask[0, 0, 1, 1].item(Bool.self)
        XCTAssertTrue(seq0row1col1, "Valid position (seq 0, row 1, col 1) should be unmasked")

        // For sequence 1 (leftPadding=2): columns 0 and 1 should be masked (False)
        let seq1col0 = mask[1, 0, 0, 0].item(Bool.self)
        let seq1col1 = mask[1, 0, 0, 1].item(Bool.self)
        XCTAssertFalse(seq1col0, "Padded position (seq 1, col 0) should be masked out")
        XCTAssertFalse(seq1col1, "Padded position (seq 1, col 1) should be masked out")

        // For sequence 1: column 2 at row 2 should be True (valid, causal ok)
        let seq1row2col2 = mask[1, 0, 2, 2].item(Bool.self)
        XCTAssertTrue(seq1row2col2, "Valid position (seq 1, row 2, col 2) should be unmasked")
    }

    // MARK: - VAL-CACHE-013: createCausalMask backward compatible without leftPadding

    func testCreateCausalMaskBackwardCompatible() throws {
        try skipIfMetalUnavailable()

        let n = 4
        let offset = 2

        // Call without leftPadding (should be identical to before)
        let maskWithout = createCausalMask(n: n, offset: offset)

        // Call with leftPadding explicitly nil
        let maskWithNil = createCausalMask(n: n, offset: offset, leftPadding: nil)

        // Results should be identical
        XCTAssertEqual(maskWithout.shape, maskWithNil.shape)

        let diff = abs(maskWithout.asType(.float32) - maskWithNil.asType(.float32)).sum().item(
            Float.self)
        XCTAssertEqual(diff, 0.0, "Masks should be identical when leftPadding is nil")

        // Verify the standard causal structure:
        // With offset=2, total columns = offset + n = 6, query rows = n = 4
        // Row i (query position offset+i) can attend to columns 0..offset+i
        XCTAssertEqual(maskWithout.dim(-1), offset + n)  // 6 columns
        XCTAssertEqual(maskWithout.dim(-2), n)  // 4 rows
    }

    // MARK: - VAL-CACHE-011: makeMask generates correct causal mask with left-padding

    func testBatchKVCacheMakeMaskWithLeftPadding() throws {
        try skipIfMetalUnavailable()

        let cache = BatchKVCache(leftPadding: [1, 3, 0])
        let B = 3
        let H = 2
        let S = 5
        let D = 4

        let (keys, values) = makeKV(batchSize: B, heads: H, seqLen: S, headDim: D)
        _ = cache.update(keys: keys, values: values)

        // Now cache._idx = 5. Ask for mask with n=5 (full prefill)
        let maskMode = cache.makeMask(n: S, windowSize: nil, returnArray: false)

        // Should always return .array for batch caches
        switch maskMode {
        case .array(let mask):
            // Check shape: should be [B, 1, n, S_total]
            XCTAssertEqual(mask.dim(0), B)
            XCTAssertEqual(mask.dim(2), S)
            XCTAssertEqual(mask.dim(3), S)

            // Seq 0 (padding=1): column 0 should be False for all rows
            let seq0col0 = mask[0, 0, 0, 0].item(Bool.self)
            XCTAssertFalse(seq0col0, "Seq 0 padded col 0 should be masked")

            // Seq 0: column 1, row 1 should be True
            let seq0row1col1 = mask[0, 0, 1, 1].item(Bool.self)
            XCTAssertTrue(seq0row1col1, "Seq 0 valid position should be unmasked")

            // Seq 1 (padding=3): columns 0-2 should be False
            let seq1col0 = mask[1, 0, 3, 0].item(Bool.self)
            let seq1col1 = mask[1, 0, 3, 1].item(Bool.self)
            let seq1col2 = mask[1, 0, 3, 2].item(Bool.self)
            XCTAssertFalse(seq1col0, "Seq 1 padded col 0 should be masked")
            XCTAssertFalse(seq1col1, "Seq 1 padded col 1 should be masked")
            XCTAssertFalse(seq1col2, "Seq 1 padded col 2 should be masked")

            // Seq 1: column 3, row 3 should be True (first non-padded position)
            let seq1row3col3 = mask[1, 0, 3, 3].item(Bool.self)
            XCTAssertTrue(seq1row3col3, "Seq 1 first valid position should be unmasked")

            // Seq 2 (padding=0): all standard causal positions should work
            let seq2row0col0 = mask[2, 0, 0, 0].item(Bool.self)
            XCTAssertTrue(seq2row0col0, "Seq 2 no padding, (0,0) should be True")

        default:
            XCTFail("Expected .array mask from batch cache, got \(maskMode)")
        }
    }

    // MARK: - VAL-CACHE-020: BatchKVCache makeMask with n=1 masks left-padding during decode

    func testBatchKVCacheMakeMaskN1MasksPadding() throws {
        try skipIfMetalUnavailable()

        let cache = BatchKVCache(leftPadding: [2, 0])
        let B = 2
        let H = 2
        let D = 4

        // First, do a prefill with 4 tokens
        let (keys, values) = makeKV(batchSize: B, heads: H, seqLen: 4, headDim: D)
        _ = cache.update(keys: keys, values: values)

        // Now do a decode step with n=1
        let (decK, decV) = makeKV(batchSize: B, heads: H, seqLen: 1, headDim: D)
        _ = cache.update(keys: decK, values: decV)

        // Get mask for n=1 (single token decode)
        let maskMode = cache.makeMask(n: 1, windowSize: nil, returnArray: false)

        switch maskMode {
        case .array(let mask):
            // For n=1, we have 1 query position attending to 5 key positions (_idx=5)
            // Mask shape: [B, 1, 1, 5]
            XCTAssertEqual(mask.dim(0), B)
            XCTAssertEqual(mask.dim(2), 1)
            XCTAssertEqual(mask.dim(3), 5)

            // Seq 0 (padding=2): columns 0,1 should be False
            let seq0col0 = mask[0, 0, 0, 0].item(Bool.self)
            let seq0col1 = mask[0, 0, 0, 1].item(Bool.self)
            XCTAssertFalse(seq0col0, "n=1 decode: padded position 0 should still be masked")
            XCTAssertFalse(seq0col1, "n=1 decode: padded position 1 should still be masked")

            // Seq 0: columns 2-4 should be True
            let seq0col2 = mask[0, 0, 0, 2].item(Bool.self)
            let seq0col4 = mask[0, 0, 0, 4].item(Bool.self)
            XCTAssertTrue(seq0col2, "n=1 decode: valid position 2 should be unmasked")
            XCTAssertTrue(seq0col4, "n=1 decode: valid position 4 should be unmasked")

            // Seq 1 (padding=0): all columns should be True
            let seq1col0 = mask[1, 0, 0, 0].item(Bool.self)
            let seq1col4 = mask[1, 0, 0, 4].item(Bool.self)
            XCTAssertTrue(seq1col0, "n=1 decode: no-padding seq should have all positions unmasked")
            XCTAssertTrue(seq1col4, "n=1 decode: no-padding seq col 4 should be unmasked")

        default:
            XCTFail("Batch cache must return .array mask for n=1, not .none")
        }
    }

    // MARK: - VAL-CACHE-015: BatchPositionedKVCache protocol provides per-sequence offsets

    func testBatchPositionedKVCacheOffsets() throws {
        try skipIfMetalUnavailable()

        let cache = BatchKVCache(leftPadding: [2, 0, 1])
        let B = 3
        let H = 2
        let S = 5
        let D = 4

        let (keys, values) = makeKV(batchSize: B, heads: H, seqLen: S, headDim: D)
        _ = cache.update(keys: keys, values: values)

        // Verify conformance to BatchPositionedKVCache
        let positioned: BatchPositionedKVCache = cache

        // batchOffset should be per-sequence offsets
        let offsets = positioned.batchOffset
        XCTAssertEqual(offsets.shape, [B])

        // Expected: offset = -leftPadding + S = [-2+5, 0+5, -1+5] = [3, 5, 4]
        XCTAssertEqual(offsets[0].item(Int32.self), 3)
        XCTAssertEqual(offsets[1].item(Int32.self), 5)
        XCTAssertEqual(offsets[2].item(Int32.self), 4)
    }

    // MARK: - VAL-CACHE-022: CacheList and MambaCache detected as batch-incompatible

    func testCacheListBatchIncompatible() {
        let cacheList = CacheList(KVCacheSimple(), KVCacheSimple())
        XCTAssertFalse(
            isBatchCompatible([cacheList]),
            "CacheList should be detected as batch-incompatible"
        )
    }

    func testMambaCacheBatchIncompatible() {
        let mambaCache = MambaCache()
        XCTAssertFalse(
            isBatchCompatible([mambaCache]),
            "MambaCache should be detected as batch-incompatible"
        )
    }

    func testQuantizedKVCacheBatchIncompatible() {
        let quantizedCache = QuantizedKVCache()
        XCTAssertFalse(
            isBatchCompatible([quantizedCache]),
            "QuantizedKVCache should be detected as batch-incompatible"
        )
    }

    func testKVCacheSimpleBatchCompatible() {
        let cache = KVCacheSimple()
        XCTAssertTrue(
            isBatchCompatible([cache]),
            "KVCacheSimple should be batch-compatible"
        )
    }

    func testRotatingKVCacheBatchCompatible() {
        let cache = RotatingKVCache(maxSize: 32)
        XCTAssertTrue(
            isBatchCompatible([cache]),
            "RotatingKVCache should be batch-compatible"
        )
    }

    func testEmptyCacheBatchCompatible() {
        XCTAssertTrue(
            isBatchCompatible([]),
            "Empty cache array should be batch-compatible"
        )
    }

    func testMixedCacheBatchIncompatible() {
        let caches: [KVCache] = [KVCacheSimple(), MambaCache()]
        XCTAssertFalse(
            isBatchCompatible(caches),
            "Mixed caches with MambaCache should be batch-incompatible"
        )
    }

    // MARK: - VAL-MODEL-002: applyRotaryPosition backward compatible with KVCacheSimple

    func testApplyRotaryPositionWithKVCacheSimple() throws {
        try skipIfMetalUnavailable()

        let rope = RoPE(dimensions: 8)
        let x = MLXArray.ones([1, 4, 3, 8])  // [B, H, S, D]

        let cache = KVCacheSimple()
        _ = cache.update(
            keys: MLXArray.ones([1, 4, 3, 8]),
            values: MLXArray.ones([1, 4, 3, 8])
        )

        // Apply via helper
        let result = applyRotaryPosition(rope, to: x, cache: cache)

        // Apply directly (old pattern)
        let expected = rope(x, offset: cache.offset)

        // Results should be identical
        XCTAssertEqual(result.shape, expected.shape)

        let diff = abs(result - expected).sum().item(Float.self)
        XCTAssertEqual(diff, 0.0, "applyRotaryPosition with KVCacheSimple should match direct call")
    }

    // MARK: - VAL-MODEL-003: applyRotaryPosition supports BatchPositionedKVCache

    func testApplyRotaryPositionWithBatchPositionedKVCache() throws {
        try skipIfMetalUnavailable()

        let rope = RoPE(dimensions: 8)
        let x = MLXArray.ones([2, 4, 3, 8])  // [B=2, H=4, S=3, D=8]

        let cache = BatchKVCache(leftPadding: [1, 0])
        _ = cache.update(
            keys: MLXArray.ones([2, 4, 3, 8]),
            values: MLXArray.ones([2, 4, 3, 8])
        )

        // Apply via helper with batch cache
        let result = applyRotaryPosition(rope, to: x, cache: cache)

        // Should use batchOffset (MLXArray offsets)
        let expected = rope(x, offset: cache.batchOffset)

        XCTAssertEqual(result.shape, expected.shape)

        let diff = abs(result - expected).sum().item(Float.self)
        XCTAssertEqual(
            diff, 0.0, "applyRotaryPosition with BatchKVCache should use per-sequence offsets")
    }

    // MARK: - VAL-MODEL-004: applyRotaryPosition handles nil cache

    func testApplyRotaryPositionWithNilCache() throws {
        try skipIfMetalUnavailable()

        let rope = RoPE(dimensions: 8)
        let x = MLXArray.ones([1, 4, 3, 8])

        // Apply with nil cache
        let result = applyRotaryPosition(rope, to: x, cache: nil)

        // Should be equivalent to offset=0
        let expected = rope(x, offset: 0)

        XCTAssertEqual(result.shape, expected.shape)

        let diff = abs(result - expected).sum().item(Float.self)
        XCTAssertEqual(diff, 0.0, "applyRotaryPosition with nil cache should use offset=0")
    }

    // MARK: - Additional mask tests

    func testCreateCausalMaskWithWindowSizeAndLeftPadding() throws {
        try skipIfMetalUnavailable()

        // Verify that windowSize and leftPadding work together
        let leftPadding = MLXArray([Int32(1)])
        let n = 4
        let offset = 0
        let windowSize = 3

        let mask = createCausalMask(
            n: n, offset: offset, windowSize: windowSize, leftPadding: leftPadding
        )

        // Should have shape [1, 1, 4, 4]
        XCTAssertEqual(mask.dim(0), 1)
        XCTAssertEqual(mask.dim(2), n)
        XCTAssertEqual(mask.dim(3), n)

        // Column 0 should be masked (padded)
        let col0 = mask[0, 0, 0, 0].item(Bool.self)
        XCTAssertFalse(col0, "Padded position should be masked even with window")
    }

    func testBatchKVCacheMakeMaskMultipleDecodeSteps() throws {
        try skipIfMetalUnavailable()

        // Verify that mask remains correct across multiple decode steps
        let cache = BatchKVCache(leftPadding: [1, 0])
        let B = 2
        let H = 2
        let D = 4

        // Prefill with 3 tokens
        let (keys, values) = makeKV(batchSize: B, heads: H, seqLen: 3, headDim: D)
        _ = cache.update(keys: keys, values: values)

        // First decode step
        let (d1k, d1v) = makeKV(batchSize: B, heads: H, seqLen: 1, headDim: D)
        _ = cache.update(keys: d1k, values: d1v)

        // Second decode step
        let (d2k, d2v) = makeKV(batchSize: B, heads: H, seqLen: 1, headDim: D)
        _ = cache.update(keys: d2k, values: d2v)

        // Mask for n=1 at _idx=5
        let maskMode = cache.makeMask(n: 1, windowSize: nil, returnArray: false)

        switch maskMode {
        case .array(let mask):
            // Seq 0 (padding=1): column 0 should still be False
            let seq0col0 = mask[0, 0, 0, 0].item(Bool.self)
            XCTAssertFalse(seq0col0, "After multiple decode steps, padding should still be masked")

            // Seq 0: all other positions should be True
            let seq0col1 = mask[0, 0, 0, 1].item(Bool.self)
            XCTAssertTrue(seq0col1, "Valid positions should be unmasked")

        default:
            XCTFail("Batch cache must return .array mask")
        }
    }

    func testNonBatchCacheMakeMaskN1ReturnsNone() throws {
        try skipIfMetalUnavailable()

        // Verify that the existing non-batch behavior (BaseKVCache) returns .none for n=1
        let cache = KVCacheSimple()
        _ = cache.update(
            keys: MLXArray.ones([1, 2, 3, 4]),
            values: MLXArray.ones([1, 2, 3, 4])
        )

        let maskMode = cache.makeMask(n: 1, windowSize: nil, returnArray: false)

        switch maskMode {
        case .none:
            break  // Expected
        default:
            XCTFail("Non-batch cache should return .none for n=1, got \(maskMode)")
        }
    }
}
