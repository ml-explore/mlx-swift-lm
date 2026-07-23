// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import XCTest

/// Regression coverage for the Hunyuan dense V1 architecture (`hunyuan_v1_dense`),
/// used by Hunyuan-MT-7B and Hy-MT2-7B.
public class HunyuanTests: XCTestCase {

    // MARK: - Configuration decoding

    /// A Hunyuan config is a flat `HunYuanDenseV1ForCausalLM` config (no nested `text_config`).
    func testConfigurationDecodingFromFlatConfig() throws {
        // Values mirror tencent/Hunyuan-MT-7B.
        let json = """
            {
                "model_type": "hunyuan_v1_dense",
                "architectures": ["HunYuanDenseV1ForCausalLM"],
                "hidden_size": 4096,
                "num_hidden_layers": 32,
                "intermediate_size": 14336,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "head_dim": 128,
                "rms_norm_eps": 1e-5,
                "vocab_size": 128256,
                "rope_theta": 10000.0,
                "attention_bias": false,
                "use_qk_norm": true,
                "tie_word_embeddings": true,
                "max_position_embeddings": 32768,
                "rope_scaling": {
                    "alpha": 100000.0,
                    "factor": 1.0,
                    "type": "dynamic"
                }
            }
            """

        let config = try JSONDecoder().decode(
            HunyuanConfiguration.self, from: json.data(using: .utf8)!)

        XCTAssertEqual(config.hiddenSize, 4096)
        XCTAssertEqual(config.hiddenLayers, 32)
        XCTAssertEqual(config.intermediateSize, 14336)
        XCTAssertEqual(config.attentionHeads, 32)
        XCTAssertEqual(config.kvHeads, 8)
        XCTAssertEqual(config.headDim, 128)
        XCTAssertEqual(config.vocabularySize, 128256)
        XCTAssertEqual(config.ropeTheta, 10000.0)
        XCTAssertTrue(config.useQkNorm)
        XCTAssertTrue(config.tieWordEmbeddings)
        XCTAssertFalse(config.attentionBias)
        XCTAssertEqual(config.ropeScaling?["alpha"]?.asFloat(), 100000.0)
    }

    /// Hy-MT2-7B configs omit `head_dim` and only declare `attention_head_dim`. Use a value
    /// that differs from `hidden_size / num_attention_heads` so the test fails if the alias
    /// is dropped and we silently fall back to the computed default.
    func testConfigurationReadsAttentionHeadDimAlias() throws {
        let json = """
            {
                "model_type": "hunyuan_v1_dense",
                "hidden_size": 8,
                "num_hidden_layers": 1,
                "intermediate_size": 16,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "attention_head_dim": 6,
                "rms_norm_eps": 1e-5,
                "vocab_size": 32,
                "tie_word_embeddings": true,
                "rope_scaling": {"alpha": 1000.0, "type": "dynamic", "factor": 1.0}
            }
            """
        let config = try JSONDecoder().decode(
            HunyuanConfiguration.self, from: json.data(using: .utf8)!)
        // 6 from attention_head_dim, not hidden_size/heads = 4.
        XCTAssertEqual(config.headDim, 6)
    }

    // MARK: - Weight sanitization

    /// Hunyuan ties `lm_head` to the token embeddings, so the checkpoint ships no
    /// `lm_head.weight` and `sanitize` must drop any that appears.
    func testSanitizeDropsTiedLmHead() throws {
        let config = try Self.tinyConfig(tied: true)
        let model = HunyuanModel(config)

        let weights: [String: MLXArray] = [
            "model.embed_tokens.weight":
                MLXArray.zeros([config.vocabularySize, config.hiddenSize]),
            "lm_head.weight": MLXArray.zeros([config.vocabularySize, config.hiddenSize]),
        ]

        let sanitized = model.sanitize(weights: weights)
        XCTAssertNil(sanitized["lm_head.weight"])
        XCTAssertNotNil(sanitized["model.embed_tokens.weight"])
    }

    // MARK: - Forward pass

    /// A tiny tied model should run end-to-end and produce one logit row per input
    /// token over the full vocabulary (exercising qk-norm, GQA and dynamic RoPE).
    func testTinyModelForwardProducesVocabLogits() throws {
        let config = try Self.tinyConfig(tied: true)
        let model = HunyuanModel(config)

        let tokens = MLXArray([1, 2, 3] as [Int32]).reshaped(1, 3)
        let logits = model(tokens, cache: nil)

        XCTAssertEqual(logits.dim(0), 1)
        XCTAssertEqual(logits.dim(1), 3)
        XCTAssertEqual(logits.dim(2), config.vocabularySize)
        // Force materialization to surface any compute errors.
        XCTAssertTrue(logits.sum().item(Float.self).isFinite)
    }

    // MARK: - Dynamic NTK RoPE

    /// The `alpha` factor must actually rescale the RoPE base: at a non-zero position
    /// the rotated output differs from plain RoPE (alpha = 1).
    func testDynamicNTKAlphaRoPEAppliesAlpha() {
        let plain = DynamicNTKAlphaRoPE(dimensions: 4, base: 10000, scalingAlpha: 1.0)
        let scaled = DynamicNTKAlphaRoPE(dimensions: 4, base: 10000, scalingAlpha: 1000.0)

        // Shape [B, H, L, D] with L = 2 so position 1 carries a non-trivial rotation.
        let x = MLXArray((1 ... 8).map { Float($0) }).reshaped(1, 1, 2, 4)

        let a = plain(x, offset: 0)
        let b = scaled(x, offset: 0)

        let maxDiff = abs(a - b).max().item(Float.self)
        XCTAssertGreaterThan(maxDiff, 1e-3, "alpha rescale should change the rotation")
    }

    // MARK: - Registry presets

    func testPresetsResolveToExpectedIds() {
        let expected: [(ModelConfiguration, String)] = [
            (LLMRegistry.hunyuan_mt_7b_4bit, "mlx-community/Hunyuan-MT-7B-4bit"),
            (LLMRegistry.hunyuan_mt_7b_8bit, "mlx-community/Hunyuan-MT-7B-8bit"),
            (LLMRegistry.hy_mt2_7b_4bit, "mlx-community/Hy-MT2-7B-4bit"),
            (LLMRegistry.hy_mt2_7b_8bit, "mlx-community/Hy-MT2-7B-8bit"),
        ]
        for (configuration, id) in expected {
            XCTAssertEqual(configuration.name, id)
        }
    }

    // MARK: - Helpers

    /// Minimal config decoded from JSON (the struct only has a `Decodable` initializer).
    private static func tinyConfig(tied: Bool) throws -> HunyuanConfiguration {
        let json = """
            {
                "model_type": "hunyuan_v1_dense",
                "hidden_size": 8,
                "num_hidden_layers": 1,
                "intermediate_size": 16,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 4,
                "rms_norm_eps": 1e-5,
                "vocab_size": 32,
                "rope_theta": 10000.0,
                "use_qk_norm": true,
                "tie_word_embeddings": \(tied),
                "rope_scaling": {"alpha": 1000.0, "factor": 1.0, "type": "dynamic"}
            }
            """
        return try JSONDecoder().decode(
            HunyuanConfiguration.self, from: json.data(using: .utf8)!)
    }
}
