// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXVLM
import XCTest

/// Plain DeepSeek-OCR packs (no R-SWA window) must build their caches through
/// the shared attention-cache constructor so `GenerateParameters.maxKVSize` is
/// honored. `newCache` once dropped `parameters` and always returned unbounded
/// `KVCacheSimple` layers, so a requested capacity was silently ignored.
final class DeepseekOCRCacheTests: XCTestCase {

    func testNewCacheHonorsMaxKVSizeOnPlainDeepseekPath() throws {
        let model = try makeModel()
        let parameters = GenerateParameters(maxKVSize: 64)

        let caches = try model.newCache(parameters: parameters)
        XCTAssertEqual(caches.count, 2)
        for cache in caches {
            let rotating = try XCTUnwrap(cache as? RotatingKVCache)
            XCTAssertEqual(rotating.maxSize, 64)
        }

        let status = try model.cacheStatus(parameters: parameters)
        XCTAssertEqual(status.capacityDisposition, .fullyApplied)
        XCTAssertNoThrow(
            try validateKVCacheCompatibility(
                caches, configuration: KVCacheConfiguration(capacity: try .init(maxTokens: 64))))
    }

    func testNewCacheWithoutParametersIsUnboundedOnPlainDeepseekPath() throws {
        let model = try makeModel()

        let caches = try model.newCache(parameters: nil)
        XCTAssertEqual(caches.count, 2)
        for cache in caches {
            XCTAssertTrue(cache is KVCacheSimple)
            XCTAssertNil(cache.maxSize)
        }
    }

    func testNewCacheRejectsInvalidMaxKVSizeOnPlainDeepseekPath() throws {
        let model = try makeModel()

        XCTAssertThrowsError(try model.newCache(parameters: GenerateParameters(maxKVSize: 0))) {
            error in
            XCTAssertEqual(error as? KVCacheConfigurationError, .invalidCapacity(0))
        }
    }

    /// Python resolves the window with `sliding_window_size or sliding_window`,
    /// so `0` means unset and the plain attention path is taken.
    func testZeroSlidingWindowSizeMeansUnsetOnPlainDeepseekPath() throws {
        let model = try makeModel(slidingWindowSize: 0)

        let caches = try model.newCache(parameters: nil)
        XCTAssertEqual(caches.count, 2)
        for cache in caches {
            XCTAssertTrue(cache is KVCacheSimple)
        }
    }

    private func makeModel(slidingWindowSize: Int? = nil) throws -> DeepseekOCR {
        let config = try JSONDecoder().decode(
            DeepseekOCRConfiguration.self,
            from: Data(Self.configJSON(slidingWindowSize: slidingWindowSize).utf8))
        XCTAssertNil(config.resolvedSlidingWindowSize)
        return DeepseekOCR(config)
    }

    /// Minimal DeepSeek-OCR config, optionally carrying a top-level `sliding_window_size`.
    private static func configJSON(slidingWindowSize: Int?) -> String {
        let window = slidingWindowSize.map { "\"sliding_window_size\": \($0)," } ?? ""
        return #"""
            {
              "model_type": "deepseekocr",
              \#(window)
              "vision_config": {
                "hidden_size": 32,
                "output_channels": 8,
                "num_hidden_layers": 1,
                "num_attention_heads": 4,
                "image_size": 32,
                "patch_size": 16,
                "window_size": 2,
                "global_attn_indexes": [0],
                "mlp_dim": 64
              },
              "language_config": {
                "vocab_size": 32,
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 4,
                "max_position_embeddings": 32
              }
            }
            """#
    }
}
