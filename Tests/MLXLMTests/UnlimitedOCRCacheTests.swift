// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXVLM
import XCTest

/// TASK-010: UnlimitedOCR wires R-SWA caches; L4 memory bound =
/// `prefill_length + sliding_window_size`.
final class UnlimitedOCRCacheTests: XCTestCase {

    func testNewCacheDefaultsToRingWindow128() async throws {
        let model = try await VLMTypeRegistry.shared.createModel(
            configuration: Self.configWithoutWindow.data(using: .utf8)!,
            modelType: "unlimited-ocr")
        let unlimited = try XCTUnwrap(model as? UnlimitedOCR)
        XCTAssertNil(unlimited.config.resolvedSlidingWindowSize)

        let caches = unlimited.newCache(parameters: nil)
        XCTAssertEqual(caches.count, 12)
        for cache in caches {
            let ring = try XCTUnwrap(cache as? RingSlidingKVCache)
            XCTAssertEqual(ring.windowSize, 128)
        }
    }

    func testL4LongDecodeKVBoundedAtPrefillPlusWindow() {
        let window = 128
        let prefillLen = 64
        let decodeSteps = 400
        let cache = RingSlidingKVCache(windowSize: window)

        let keys = MLXArray.full([1, 2, prefillLen, 4], values: MLXArray(Float(1)))
        let values = MLXArray.full([1, 2, prefillLen, 4], values: MLXArray(Float(2)))
        _ = cache.update(keys: keys, values: values)

        for i in 0 ..< decodeSteps {
            let dk = MLXArray.full([1, 2, 1, 4], values: MLXArray(Float(10 + i)))
            let dv = MLXArray.full([1, 2, 1, 4], values: MLXArray(Float(20 + i)))
            let (k, v) = cache.update(keys: dk, values: dv)
            let bound = prefillLen + window
            XCTAssertLessThanOrEqual(k.dim(2), bound)
            XCTAssertLessThanOrEqual(v.dim(2), bound)
            if i + 1 >= window {
                XCTAssertEqual(cache.maxSize, bound)
                XCTAssertEqual(k.dim(2), bound)
            }
        }

        XCTAssertEqual(cache.offset, prefillLen + decodeSteps)
        XCTAssertEqual(cache.maxSize, prefillLen + window)
        XCTAssertEqual(cache.state[0].dim(2), prefillLen + window)
    }

    /// Minimal Unlimited config omitting sliding_window_* so default-128 path is exercised.
    private static let configWithoutWindow = #"""
        {
         "architectures": ["UnlimitedOCRForCausalLM"],
         "language_config": {
          "hidden_size": 1280,
          "intermediate_size": 6848,
          "lm_head": true,
          "max_position_embeddings": 8192,
          "moe_intermediate_size": 896,
          "n_group": 1,
          "n_routed_experts": 64,
          "n_shared_experts": 2,
          "num_attention_heads": 10,
          "num_experts_per_tok": 6,
          "num_hidden_layers": 12,
          "num_key_value_heads": 10,
          "vocab_size": 129280
         },
         "model_type": "unlimited-ocr",
         "vision_config": {
          "image_size": 1024,
          "model_type": "vision",
          "width": {
           "sam_vit_b": {
            "downsample_channels": [512, 1024],
            "global_attn_indexes": [2, 5, 8, 11],
            "heads": 12,
            "layers": 12,
            "width": 768
           }
          }
         },
         "vocab_size": 129280
        }
        """#
}
