// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import XCTest

@testable import MLXLMCommon

/// Unlimited-OCR's R-SWA cache: prefill retains the full prompt KV; decode fills
/// then overwrites a ring while the absolute offset keeps growing.
final class RingSlidingKVCacheTests: XCTestCase {

    private func token(value: Float, heads: Int = 2, dim: Int = 4) -> (MLXArray, MLXArray) {
        let keys = MLXArray.full([1, heads, 1, dim], values: MLXArray(value))
        let values = MLXArray.full([1, heads, 1, dim], values: MLXArray(value + 100))
        return (keys, values)
    }

    private func prefill(length: Int, value: Float = 1) -> (MLXArray, MLXArray) {
        let keys = MLXArray.full([1, 2, length, 4], values: MLXArray(value))
        let values = MLXArray.full([1, 2, length, 4], values: MLXArray(value + 100))
        return (keys, values)
    }

    func testPrefillRetainsFullPromptThenDecodeAppendsUntilRingFull() {
        let window = 4
        let cache = RingSlidingKVCache(windowSize: window)
        let prefillLen = 6

        let (pk, pv) = prefill(length: prefillLen)
        let (k0, v0) = cache.update(keys: pk, values: pv)
        XCTAssertNil(cache.prefillLength)
        XCTAssertEqual(cache.offset, prefillLen)
        XCTAssertEqual(k0.dim(2), prefillLen)
        XCTAssertEqual(v0.dim(2), prefillLen)

        for i in 0 ..< window {
            let (dk, dv) = token(value: Float(10 + i))
            let (k, v) = cache.update(keys: dk, values: dv)
            if i == 0 {
                XCTAssertEqual(cache.prefillLength, prefillLen)
            }
            XCTAssertEqual(cache.offset, prefillLen + i + 1)
            XCTAssertEqual(k.dim(2), prefillLen + i + 1)
            XCTAssertEqual(v.dim(2), prefillLen + i + 1)
        }

        XCTAssertEqual(cache.offset, prefillLen + window)
        XCTAssertEqual(cache.maxSize, prefillLen + window)
    }

    func testDecodeRingOverwritesWhileOffsetKeepsGrowing() {
        let window = 3
        let prefillLen = 5
        let cache = RingSlidingKVCache(windowSize: window)

        let (pk, pv) = prefill(length: prefillLen, value: 1)
        _ = cache.update(keys: pk, values: pv)

        for i in 0 ..< window {
            let (dk, dv) = token(value: Float(10 + i))
            _ = cache.update(keys: dk, values: dv)
        }
        XCTAssertEqual(cache.offset, prefillLen + window)

        let (ok, ov) = token(value: 99)
        let (k, v) = cache.update(keys: ok, values: ov)
        XCTAssertEqual(cache.offset, prefillLen + window + 1)
        XCTAssertEqual(k.dim(2), prefillLen + window)
        XCTAssertEqual(v.dim(2), prefillLen + window)

        XCTAssertEqual(k[0, 0, 0, 0].item(Float.self), 1)
        XCTAssertEqual(k[0, 0, prefillLen, 0].item(Float.self), 99)
        XCTAssertEqual(k[0, 0, prefillLen + 1, 0].item(Float.self), 11)
        XCTAssertEqual(k[0, 0, prefillLen + 2, 0].item(Float.self), 12)

        let (ok2, ov2) = token(value: 88)
        let (k2, _) = cache.update(keys: ok2, values: ov2)
        XCTAssertEqual(cache.offset, prefillLen + window + 2)
        XCTAssertEqual(k2.dim(2), prefillLen + window)
        XCTAssertEqual(k2[0, 0, prefillLen, 0].item(Float.self), 99)
        XCTAssertEqual(k2[0, 0, prefillLen + 1, 0].item(Float.self), 88)
        XCTAssertEqual(k2[0, 0, prefillLen + 2, 0].item(Float.self), 12)
    }

    func testMakeCachesReturnsRingSlidingWhenSlidingWindowSizeSet() {
        let withWindow = makeCaches(numLayers: 3, slidingWindowSize: 128)
        XCTAssertEqual(withWindow.count, 3)
        for cache in withWindow {
            let ring = cache as? RingSlidingKVCache
            XCTAssertNotNil(ring)
            XCTAssertEqual(ring?.windowSize, 128)
        }

        let without = makeCaches(numLayers: 2, slidingWindowSize: nil)
        XCTAssertEqual(without.count, 2)
        for cache in without {
            XCTAssertTrue(cache is KVCacheSimple)
            XCTAssertFalse(cache is RingSlidingKVCache)
        }
    }

    func testSerializationRoundTripPreservesRingMetadata() throws {
        let cache = RingSlidingKVCache(windowSize: 4)
        let (pk, pv) = prefill(length: 3)
        _ = cache.update(keys: pk, values: pv)
        for i in 0 ..< 5 {
            let (dk, dv) = token(value: Float(20 + i))
            _ = cache.update(keys: dk, values: dv)
        }

        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("safetensors")
        defer { try? FileManager.default.removeItem(at: url) }

        try savePromptCache(url: url, cache: [cache], metadata: [:])
        let (loaded, _) = try loadPromptCache(url: url)
        let restored = try XCTUnwrap(loaded[0] as? RingSlidingKVCache)

        XCTAssertEqual(restored.windowSize, cache.windowSize)
        XCTAssertEqual(restored.prefillLength, cache.prefillLength)
        XCTAssertEqual(restored.offset, cache.offset)
        XCTAssertEqual(restored.metaState, cache.metaState)
        XCTAssertEqual(restored.state.count, cache.state.count)
        if let a = restored.state.first, let b = cache.state.first {
            XCTAssertEqual(a.shape, b.shape)
            XCTAssertTrue(allClose(a, b).item(Bool.self))
        }
    }

    private func tokenBatch(values seqValues: [Float], heads: Int = 2, dim: Int = 4) -> (
        MLXArray, MLXArray
    ) {
        let keys = concatenated(
            seqValues.map { MLXArray.full([1, heads, 1, dim], values: MLXArray($0)) }, axis: 2)
        let values = concatenated(
            seqValues.map { MLXArray.full([1, heads, 1, dim], values: MLXArray($0 + 100)) },
            axis: 2)
        return (keys, values)
    }

    /// A decode step with seqLen > 1 that crosses the fill→ring boundary must
    /// give every new token its full window (concat path), then retain exactly
    /// the last `windowSize` decode tokens.
    func testMultiTokenDecodeCrossingWindowBoundary() {
        let window = 4
        let prefillLen = 3
        let cache = RingSlidingKVCache(windowSize: window)

        let (pk, pv) = prefill(length: prefillLen, value: 1)
        _ = cache.update(keys: pk, values: pv)

        // First single-token step marks prefill; offset = 4, capacity = 7.
        let (dk, dv) = token(value: 10)
        _ = cache.update(keys: dk, values: dv)
        XCTAssertEqual(cache.prefillLength, prefillLen)

        // 5-token step: offset 4 + 5 > capacity 7 — crosses into the ring.
        let (bk, bv) = tokenBatch(values: [20, 21, 22, 23, 24])
        let (k, v) = cache.update(keys: bk, values: bv)

        // Wide return: [prefix(3) | kept decode(1) | new(5)] = 9, temporal order.
        XCTAssertEqual(k.dim(2), 9)
        XCTAssertEqual(v.dim(2), 9)
        XCTAssertEqual(k[0, 0, 2, 0].item(Float.self), 1)
        XCTAssertEqual(k[0, 0, 3, 0].item(Float.self), 10)
        XCTAssertEqual(k[0, 0, 4, 0].item(Float.self), 20)
        XCTAssertEqual(k[0, 0, 8, 0].item(Float.self), 24)
        XCTAssertEqual(cache.offset, 9)

        // Retained: prefix + last `window` decode tokens (21, 22, 23, 24).
        let state = cache.state
        XCTAssertEqual(state[0].dim(2), prefillLen + window)
        XCTAssertEqual(state[0][0, 0, prefillLen, 0].item(Float.self), 21)
        XCTAssertEqual(state[0][0, 0, prefillLen + 3, 0].item(Float.self), 24)

        // Next single-token step overwrites the oldest retained slot (21).
        let (nk, nv) = token(value: 30)
        let (k2, _) = cache.update(keys: nk, values: nv)
        XCTAssertEqual(cache.offset, 10)
        XCTAssertEqual(k2.dim(2), prefillLen + window)
        XCTAssertEqual(k2[0, 0, prefillLen, 0].item(Float.self), 30)
        XCTAssertEqual(k2[0, 0, prefillLen + 1, 0].item(Float.self), 22)
        XCTAssertEqual(k2[0, 0, prefillLen + 3, 0].item(Float.self), 24)
    }

    /// Exercises the wrapped-ring reorder: after the ring has wrapped
    /// (ringPos > 0), a multi-token step must return the retained window in
    /// temporal order via the two-slice reorder, then re-linearize retention.
    func testMultiTokenDecodeAfterRingWrapReordersWindow() {
        let window = 4
        let prefillLen = 3
        let cache = RingSlidingKVCache(windowSize: window)

        let (pk, pv) = prefill(length: prefillLen, value: 1)
        _ = cache.update(keys: pk, values: pv)

        // Fill: abs 3...6 (values 10...13) — offset reaches capacity 7.
        // Wrap twice: abs 7 (14) -> slot 3, abs 8 (15) -> slot 4; ringPos = 2.
        for i in 0 ..< 6 {
            let (dk, dv) = token(value: Float(10 + i))
            _ = cache.update(keys: dk, values: dv)
        }
        XCTAssertEqual(cache.offset, 9)

        // Ring slots hold [14, 15, 12, 13]; temporal order is 12, 13, 14, 15.
        // 2-token step: wide = [prefix(3) | kept(3): 13, 14, 15 | new: 16, 17].
        let (bk, bv) = tokenBatch(values: [16, 17])
        let (k, _) = cache.update(keys: bk, values: bv)
        XCTAssertEqual(k.dim(2), 8)
        XCTAssertEqual(k[0, 0, 2, 0].item(Float.self), 1)
        XCTAssertEqual(k[0, 0, 3, 0].item(Float.self), 13)
        XCTAssertEqual(k[0, 0, 4, 0].item(Float.self), 14)
        XCTAssertEqual(k[0, 0, 5, 0].item(Float.self), 15)
        XCTAssertEqual(k[0, 0, 6, 0].item(Float.self), 16)
        XCTAssertEqual(k[0, 0, 7, 0].item(Float.self), 17)
        XCTAssertEqual(cache.offset, 11)

        // Retained: prefix + last `window` in temporal order (14, 15, 16, 17).
        let state = cache.state
        XCTAssertEqual(state[0].dim(2), prefillLen + window)
        XCTAssertEqual(state[0][0, 0, prefillLen, 0].item(Float.self), 14)
        XCTAssertEqual(state[0][0, 0, prefillLen + 3, 0].item(Float.self), 17)

        // Next single token overwrites the oldest retained slot (14).
        let (nk, nv) = token(value: 18)
        let (k2, _) = cache.update(keys: nk, values: nv)
        XCTAssertEqual(k2.dim(2), prefillLen + window)
        XCTAssertEqual(k2[0, 0, prefillLen, 0].item(Float.self), 18)
        XCTAssertEqual(k2[0, 0, prefillLen + 1, 0].item(Float.self), 15)
        XCTAssertEqual(k2[0, 0, prefillLen + 3, 0].item(Float.self), 17)
    }

    /// Trimming back to (or below) the marked reference prefix must unmark it —
    /// otherwise a later long step computes a mask narrower than the returned
    /// keys (prompt-cache reuse trims exactly like this).
    func testTrimBelowPrefillUnmarksReferencePrefix() {
        let window = 4
        let cache = RingSlidingKVCache(windowSize: window)

        let (pk, pv) = prefill(length: 4, value: 1)
        _ = cache.update(keys: pk, values: pv)
        let (dk, dv) = token(value: 10)
        _ = cache.update(keys: dk, values: dv)
        XCTAssertEqual(cache.prefillLength, 4)
        XCTAssertTrue(cache.isTrimmable)

        // Trim below the prefix boundary (offset 5 -> 2).
        XCTAssertEqual(cache.trim(3), 3)
        XCTAssertEqual(cache.offset, 2)
        XCTAssertNil(cache.prefillLength)

        // A long follow-up step is plain prefill growth again: symbolic causal
        // mask, and update returns the matching full width.
        let mode = cache.makeMask(n: 7, windowSize: nil, returnArray: false)
        if case .array = mode {
            XCTFail("Expected symbolic causal mask after trim, got \(mode)")
        }
        let (bk, bv) = tokenBatch(values: [20, 21, 22, 23, 24, 25, 26])
        let (k, _) = cache.update(keys: bk, values: bv)
        XCTAssertEqual(k.dim(2), 9)
        XCTAssertEqual(cache.offset, 9)
        XCTAssertNil(cache.prefillLength)
    }

    /// The mask for a multi-token ring step must match the wide temporal
    /// layout: full prefix visible, decode columns limited to each query's
    /// trailing window.
    func testMultiTokenRingMaskMatchesTemporalLayout() throws {
        let window = 4
        let prefillLen = 3
        let cache = RingSlidingKVCache(windowSize: window)

        let (pk, pv) = prefill(length: prefillLen, value: 1)
        _ = cache.update(keys: pk, values: pv)
        let (dk, dv) = token(value: 10)
        _ = cache.update(keys: dk, values: dv)

        // Mask is created before update: n = 5, columns = prefix(3) + keep(1) + 5.
        let mode = cache.makeMask(n: 5, windowSize: nil, returnArray: false)
        guard case .array(let mask) = mode else {
            XCTFail("Expected array mask for multi-token ring step, got \(mode)")
            return
        }
        XCTAssertEqual(mask.shape, [5, 9])

        // Column absolute positions: [0, 1, 2 | 3 | 4, 5, 6, 7, 8]; rows 4...8.
        // Row 0 (q=4): causal cuts future decode columns.
        XCTAssertTrue(mask[0, 3].item(Bool.self))
        XCTAssertTrue(mask[0, 4].item(Bool.self))
        XCTAssertFalse(mask[0, 5].item(Bool.self))
        // Row 4 (q=8): window (4) excludes decode positions 3 and 4...
        XCTAssertFalse(mask[4, 3].item(Bool.self))
        XCTAssertFalse(mask[4, 4].item(Bool.self))
        XCTAssertTrue(mask[4, 5].item(Bool.self))
        XCTAssertTrue(mask[4, 8].item(Bool.self))
        // ...but the reference prefix is always visible.
        XCTAssertTrue(mask[4, 0].item(Bool.self))
        XCTAssertTrue(mask[4, 2].item(Bool.self))

        // Mask width matches what update() returns for the same step.
        let (bk, bv) = tokenBatch(values: [20, 21, 22, 23, 24])
        let (k, _) = cache.update(keys: bk, values: bv)
        XCTAssertEqual(k.dim(2), mask.dim(1))
    }

    /// KV quantization must leave the ring cache alone: quantizing would freeze
    /// the ring slots and break in-place overwrite (the cache is deliberately
    /// not a `KVCacheSimple`).
    func testMaybeQuantizeLeavesRingCacheUntouched() {
        let heads = 2
        let dim = 32  // quantizable head dim — a plain cache would convert
        let cache = RingSlidingKVCache(windowSize: 4)
        let pk = MLXArray.full([1, heads, 6, dim], values: MLXArray(Float(1)))
        let pv = MLXArray.full([1, heads, 6, dim], values: MLXArray(Float(2)))
        _ = cache.update(keys: pk, values: pv)

        var caches: [KVCache] = [cache]
        XCTAssertFalse(caches[0] is KVCacheSimple)
        maybeQuantizeKVCache(cache: &caches, kvBits: 4, kvGroupSize: 32, quantizedKVStart: 0)
        XCTAssertTrue(caches[0] is RingSlidingKVCache)
        XCTAssertFalse(caches[0] is QuantizedKVCacheProtocol)

        // Control: an actual KVCacheSimple with the same contents does convert.
        let simple = KVCacheSimple()
        _ = simple.update(keys: pk, values: pv)
        var simpleCaches: [KVCache] = [simple]
        maybeQuantizeKVCache(
            cache: &simpleCaches, kvBits: 4, kvGroupSize: 32, quantizedKVStart: 0)
        XCTAssertTrue(simpleCaches[0] is QuantizedKVCache)
    }

    func testSteadyStateDecodeMaskIsNone() {
        let cache = RingSlidingKVCache(windowSize: 2)
        let (pk, pv) = prefill(length: 2)
        _ = cache.update(keys: pk, values: pv)
        for _ in 0 ..< 2 {
            let (dk, dv) = token(value: 3)
            _ = cache.update(keys: dk, values: dv)
        }
        let (dk, dv) = token(value: 4)
        _ = cache.update(keys: dk, values: dv)

        let mask = cache.makeMask(n: 1, windowSize: nil, returnArray: false)
        if case .none = mask {
            // expected
        } else {
            XCTFail("Expected .none mask for steady-state ring decode, got \(mask)")
        }
    }
}
