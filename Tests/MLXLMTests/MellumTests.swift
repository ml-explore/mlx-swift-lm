// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXLLM

final class MellumTests: XCTestCase {

    private func configuration(
        layerTypes: String = #"["sliding_attention", "sliding_attention", "full_attention"]"#,
        mlpLayerTypes: String = #"["sparse", "dense", "sparse"]"#,
        ropeParameters: String = """
        {
            "full_attention": {
                "rope_type": "yarn", "rope_theta": 500000.0, "factor": 4.0,
                "original_max_position_embeddings": 32, "beta_fast": 32.0, "beta_slow": 1.0,
                "attention_factor": 1.1386294361119891
            },
            "sliding_attention": {"rope_type": "default", "rope_theta": 500000.0}
        }
        """
    ) throws -> MellumConfiguration {
        let json = """
            {
                "model_type": "mellum", "hidden_size": 32, "num_hidden_layers": 3,
                "intermediate_size": 48, "num_attention_heads": 4, "num_key_value_heads": 2,
                "head_dim": 32, "num_experts": 4, "num_experts_per_tok": 2,
                "moe_intermediate_size": 16, "norm_topk_prob": true, "rms_norm_eps": 1e-6,
                "vocab_size": 64, "max_position_embeddings": 128, "sliding_window": 4,
                "tie_word_embeddings": false, "attention_bias": false,
                "layer_types": \(layerTypes), "mlp_layer_types": \(mlpLayerTypes),
                "rope_parameters": \(ropeParameters)
            }
            """
        return try JSONDecoder().decode(MellumConfiguration.self, from: Data(json.utf8))
    }

    private func model(dtype: DType = .bfloat16, seed: UInt64 = 7) throws -> MellumModel {
        MLXRandom.seed(seed)
        let model = MellumModel(try configuration())
        model.update(parameters: model.parameters().mapValues { $0.asType(dtype) })
        eval(model)
        return model
    }

    func testConfigurationBuildsPerLayerAttentionAndFeedForward() throws {
        let configuration = try configuration()
        XCTAssertNoThrow(try configuration.validateModelConfiguration())
        let model = MellumModel(configuration)
        let layers = model.model.layers

        XCTAssertEqual(layers.map(\.kind), [.sliding, .sliding, .full])
        XCTAssertTrue(layers[0].mlp is MellumSparseMoeBlock)
        XCTAssertTrue(layers[1].mlp is MellumMLP)
        XCTAssertTrue(layers[0].selfAttn.rope is RoPE)
        XCTAssertTrue(layers[2].selfAttn.rope is YarnRoPE)
        XCTAssertEqual(model.reasoningConfig, QwenReasoningProtocol.tagged)
        XCTAssertNil(model.toolCallFormat)
    }

    func testDefaultsMatchTheReferenceConfiguration() throws {
        let json = """
            {
                "hidden_size": 32, "num_hidden_layers": 2, "intermediate_size": 48,
                "num_attention_heads": 4, "num_key_value_heads": 2, "num_experts": 4,
                "num_experts_per_tok": 2, "moe_intermediate_size": 16, "vocab_size": 64
            }
            """
        let configuration = try JSONDecoder().decode(
            MellumConfiguration.self, from: Data(json.utf8))
        XCTAssertNoThrow(try configuration.validateModelConfiguration())
        XCTAssertEqual(configuration.headDim, 8)
        XCTAssertEqual(configuration.layerTypes, [.full, .full])
        XCTAssertEqual(configuration.mlpLayerTypes, [.sparse, .sparse])
        XCTAssertEqual(configuration.rope(for: .full).theta, 500_000)
        XCTAssertEqual(configuration.rope(for: .sliding).theta, 10_000)
    }

    func testValidationRejectsInconsistentSchedules() throws {
        XCTAssertThrowsError(
            try configuration(layerTypes: #"["full_attention"]"#).validateModelConfiguration())
        XCTAssertThrowsError(
            try configuration(mlpLayerTypes: #"["sparse"]"#).validateModelConfiguration())
        XCTAssertThrowsError(
            try configuration(
                ropeParameters: #"{"full_attention": {"rope_theta": 10000.0}}"#
            ).validateModelConfiguration())
        XCTAssertThrowsError(
            try configuration(layerTypes: #"["sliding_attention", "global", "full_attention"]"#))
    }

    func testSlidingLayersGetWindowedCaches() throws {
        let caches = try model().newCache(parameters: nil)
        XCTAssertEqual(caches.map { $0.maxSize }, [4, 4, nil])
        XCTAssertTrue(caches[0] is RotatingKVCache)
        XCTAssertFalse(caches[2] is RotatingKVCache)
    }

    func testSanitizeStacksPerExpertProjections() throws {
        let model = try model()
        var weights: [String: MLXArray] = [:]
        for expert in 0 ..< 4 {
            for projection in ["gate_proj", "up_proj", "down_proj"] {
                weights["model.layers.0.mlp.experts.\(expert).\(projection).weight"] =
                    MLXArray.ones([3, 5]) * Float(expert)
            }
        }

        let sanitized = model.sanitize(weights: weights)

        XCTAssertEqual(sanitized.count, 3)
        let gate = try XCTUnwrap(sanitized["model.layers.0.mlp.switch_mlp.gate_proj.weight"])
        XCTAssertEqual(gate.shape, [4, 3, 5])
        XCTAssertEqual(gate[0..., 0, 0].asArray(Float.self), [0, 1, 2, 3])
    }

    func testIncrementalDecodeMatchesFullForward() throws {
        let model = try model(dtype: .float32)
        // Eight positions wrap the four-token sliding window.
        let tokens = MLXArray([Int32(3), 9, 27, 5, 11, 2, 40, 17]).reshaped(1, 8)

        let full = model(tokens, cache: try model.newCache(parameters: nil))

        let cache = try model.newCache(parameters: nil)
        let prefill = model(tokens[0..., ..<3], cache: cache)
        let steps = (3 ..< 8).map { model(tokens[0..., $0 ..< ($0 + 1)], cache: cache) }
        let incremental = concatenated([prefill] + steps, axis: 1)

        XCTAssertEqual(cache.map(\.offset), [8, 8, 8])
        XCTAssertTrue(allClose(incremental, full, rtol: 1e-4, atol: 1e-4).item(Bool.self))
    }

    func testQuantizedKVCacheOnFullAttention() throws {
        let model = try model(dtype: .float32)
        let cache = try model.newCache(parameters: nil).map { cache -> KVCache in
            cache is RotatingKVCache ? cache : QuantizedKVCache(groupSize: 32, bits: 8)
        }

        let logits = model(MLXArray([Int32(1), 2]).reshaped(1, 2), cache: cache)
        let next = model(MLXArray([Int32(3)]).reshaped(1, 1), cache: cache)

        XCTAssertEqual(logits.shape, [1, 2, 64])
        XCTAssertEqual(next.shape, [1, 1, 64])
        XCTAssertEqual(cache.map(\.offset), [3, 3, 3])
    }

    func testRegisteredModelType() async throws {
        XCTAssertTrue(
            LLMRegistry.shared.contains(id: "mlx-community/Mellum2.1-12B-A2.5B-Thinking-6bit"))
        let json = Data(
            """
            {
                "model_type": "mellum", "hidden_size": 32, "num_hidden_layers": 1,
                "intermediate_size": 48, "num_attention_heads": 4, "num_key_value_heads": 2,
                "num_experts": 4, "num_experts_per_tok": 2, "moe_intermediate_size": 16,
                "vocab_size": 64
            }
            """.utf8)
        let model = try await LLMTypeRegistry.shared.createModel(
            configuration: json, modelType: "mellum")
        XCTAssertTrue(model is MellumModel)
    }
}
