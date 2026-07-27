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
