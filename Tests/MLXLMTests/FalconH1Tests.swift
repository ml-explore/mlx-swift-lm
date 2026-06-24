// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import XCTest

final class FalconH1Tests: XCTestCase {

    private func tinyConfiguration(numLogitsToKeep: Int = 1) throws -> FalconH1Configuration {
        let json = """
            {
                "model_type": "falcon_h1",
                "hidden_size": 32,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 8,
                "vocab_size": 100,
                "mamba_d_ssm": 16,
                "mamba_d_state": 8,
                "mamba_d_head": 8,
                "mamba_n_heads": 2,
                "mamba_n_groups": 1,
                "mamba_d_conv": 4,
                "mamba_chunk_size": 64,
                "attention_in_multiplier": 2.0,
                "key_multiplier": 0.5,
                "num_logits_to_keep": \(numLogitsToKeep)
            }
            """.data(using: .utf8)!
        return try JSONDecoder.json5().decode(FalconH1Configuration.self, from: json)
    }

    // MARK: - Key scaling

    func testSanitizeScalesQKVEquivalently() throws {
        let config = try tinyConfiguration()
        let model = FalconH1Model(config)

        // The sanitizer needs the conv1d weight of the first layer to decide whether
        // the checkpoint is in the original (transposed) layout.
        let convShape: [Int] = [32, 4, 32]
        let weights: [String: MLXArray] = [
            "model.layers.0.mamba.conv1d.weight": MLXArray.ones(convShape, dtype: .float32),
            "model.layers.0.self_attn.q_proj.weight": MLXArray.ones([32, 32], dtype: .float32),
            "model.layers.0.self_attn.k_proj.weight": MLXArray.ones([16, 32], dtype: .float32),
            "model.layers.0.self_attn.v_proj.weight": MLXArray.ones([16, 32], dtype: .float32),
        ]

        let sanitized = model.sanitize(weights: weights)

        // The dead `key_proj.weight` branch must no longer exist, so `k_proj` is
        // scaled exactly like `q_proj` and `v_proj`. The H1R-specific `key_multiplier`
        // is applied at runtime in `FalconH1Attention`.
        XCTAssertEqual(
            sanitized["model.layers.0.self_attn.q_proj.weight"]!.mean().item(Float.self), 2.0,
            accuracy: 1e-5)
        XCTAssertEqual(
            sanitized["model.layers.0.self_attn.k_proj.weight"]!.mean().item(Float.self), 2.0,
            accuracy: 1e-5)
        XCTAssertEqual(
            sanitized["model.layers.0.self_attn.v_proj.weight"]!.mean().item(Float.self), 2.0,
            accuracy: 1e-5)
    }

    // MARK: - Logits-to-keep

    func testDirectCallKeepsAllLogits() throws {
        let config = try tinyConfiguration()
        let model = FalconH1Model(config)

        let inputs = MLXArray([0, 1, 2, 3, 4]).reshaped(1, 5)
        let logits = model(inputs)

        // Direct model calls are used by training/evaluation code and keep the
        // standard full-sequence logits contract shared by other LLM models.
        XCTAssertEqual(logits.shape, [1, 5, 100])
    }

    func testGenerationCallUsesConfiguredLogitsToKeep() throws {
        let config = try tinyConfiguration()
        let model = FalconH1Model(config)

        let inputs = MLXArray([0, 1, 2, 3, 4]).reshaped(1, 5)
        let output = model(LMInput.Text(tokens: inputs), cache: nil, state: nil)
        let logits = output.logits

        // Generation follows H1R's `num_logits_to_keep` default and avoids the
        // full vocabulary projection for prompt positions that will be discarded.
        XCTAssertEqual(logits.shape, [1, 1, 100])
    }

    func testExplicitLogitsToKeep() throws {
        let config = try tinyConfiguration()
        let model = FalconH1Model(config)

        let inputs = MLXArray([0, 1, 2, 3, 4]).reshaped(1, 5)
        let logits = model(inputs, cache: nil, logitsToKeep: 3)

        XCTAssertEqual(logits.shape, [1, 3, 100])
    }

    // MARK: - Cache construction

    func testNewCacheHonorsMaxKVSize() throws {
        let config = try tinyConfiguration()
        let model = FalconH1Model(config)
        let params = GenerateParameters(maxKVSize: 8)

        let cache = model.newCache(parameters: params)
        XCTAssertEqual(cache.count, 2)

        for entry in cache {
            let list = try XCTUnwrap(entry as? CacheList)
            XCTAssertTrue(list[1] is RotatingKVCache, "attention cache should rotate")
            XCTAssertTrue(list[0] is MambaCache, "mamba cache should remain unbounded")
        }
    }

    func testNewCacheCreatesQuantizedAttentionCache() throws {
        let config = try tinyConfiguration()
        let model = FalconH1Model(config)
        let params = GenerateParameters(kvBits: 4, kvGroupSize: 32, quantizedKVStart: 0)

        let cache = model.newCache(parameters: params)
        XCTAssertEqual(cache.count, 2)

        for entry in cache {
            let list = try XCTUnwrap(entry as? CacheList)
            XCTAssertTrue(list[1] is QuantizedKVCache, "attention cache should be quantized")
            XCTAssertTrue(list[0] is MambaCache, "mamba cache should never be quantized")
        }
    }

    func testDynamicQuantizerRecursesIntoCacheList() throws {
        let simple = KVCacheSimple()
        _ = simple.update(
            keys: MLXArray.ones([1, 2, 8, 64], dtype: .float32),
            values: MLXArray.ones([1, 2, 8, 64], dtype: .float32)
        )
        var cache: [any KVCache] = [CacheList(MambaCache(), simple)]

        maybeQuantizeKVCache(
            cache: &cache,
            kvBits: 4,
            kvGroupSize: 64,
            quantizedKVStart: 0
        )

        let list = try XCTUnwrap(cache.first as? CacheList)
        XCTAssertTrue(list[1] is QuantizedKVCache)
        XCTAssertTrue(list[0] is MambaCache)
    }

    // MARK: - Conversion metadata

    func testConversionMetadataSkipsResanitizingConvertedWeights() throws {
        let config = try tinyConfiguration()
        let model = FalconH1Model(config)

        let weights: [String: MLXArray] = [
            "model.layers.0.mamba.conv1d.weight": MLXArray.ones([32, 4, 32], dtype: .float32),
            "model.layers.0.self_attn.q_proj.weight": MLXArray.ones([32, 32], dtype: .float32),
        ]

        let metadata = model.modelConversionMetadata
        let sanitized = model.sanitize(weights: weights, metadata: metadata)

        XCTAssertEqual(
            sanitized["model.layers.0.self_attn.q_proj.weight"]!.mean().item(Float.self), 1.0,
            accuracy: 1e-5)
        XCTAssertEqual(sanitized["model.layers.0.mamba.conv1d.weight"]!.shape, [32, 4, 32])
    }
}
