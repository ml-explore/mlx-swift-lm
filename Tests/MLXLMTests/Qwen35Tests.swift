// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import XCTest

@testable import MLXLLM

final class Qwen35Tests: XCTestCase {

    private func makeConfiguration(
        modelType: String, numExperts: Int = 0, numExpertsPerTok: Int = 0
    )
        throws -> Qwen35Configuration
    {
        let json = """
            {
                "model_type": "\(modelType)",
                "text_config": {
                    "model_type": "\(modelType)",
                    "hidden_size": 16,
                    "num_hidden_layers": 1,
                    "intermediate_size": 32,
                    "num_attention_heads": 4,
                    "num_key_value_heads": 2,
                    "linear_num_value_heads": 4,
                    "linear_num_key_heads": 2,
                    "linear_key_head_dim": 4,
                    "linear_value_head_dim": 4,
                    "linear_conv_kernel_dim": 2,
                    "rms_norm_eps": 0.000001,
                    "vocab_size": 32,
                    "rope_theta": 100000.0,
                    "partial_rotary_factor": 0.25,
                    "max_position_embeddings": 64,
                    "full_attention_interval": 1,
                    "tie_word_embeddings": false,
                    "attention_bias": false,
                    "num_experts": \(numExperts),
                    "num_experts_per_tok": \(numExpertsPerTok),
                    "decoder_sparse_step": 1,
                    "shared_expert_intermediate_size": 16,
                    "moe_intermediate_size": 16,
                    "norm_topk_prob": true
                }
            }
            """
        return try JSONDecoder().decode(Qwen35Configuration.self, from: Data(json.utf8))
    }

    private func assertSanitizeIdempotence(
        model: Qwen35Model,
        hfMTPKey: String,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        let hfNormKey = "model.language_model.layers.0.input_layernorm.weight"
        let mlxNormKey = "language_model.model.layers.0.input_layernorm.weight"
        let base = MLXArray(0 ..< 8).asType(DType.float32)

        let converted = model.sanitize(weights: [
            hfNormKey: base,
            hfMTPKey: MLXArray.zeros([1], dtype: DType.float32),
        ])

        XCTAssertNotNil(converted[mlxNormKey], file: file, line: line)
        XCTAssertTrue(
            arrayEqual(converted[mlxNormKey]!, base + MLXArray(1.0, dtype: DType.float32)).item(
                Bool.self),
            file: file,
            line: line
        )
        XCTAssertFalse(converted.keys.contains { $0.contains("mtp.") }, file: file, line: line)

        let loaded = model.sanitize(weights: converted)
        XCTAssertTrue(
            arrayEqual(loaded[mlxNormKey]!, converted[mlxNormKey]!).item(Bool.self),
            file: file,
            line: line
        )
    }

    func testPreciseSiLUMultiplyMatchesFloat32ReferenceAndPreservesDType() {
        let eps: Float = 1e-6
        let layer = Qwen3NextRMSNormGated(dimensions: 4, eps: eps)
        let hiddenStates = MLXArray([1.0 as Float, -2.0, 3.0, -4.0], [1, 4]).asType(.bfloat16)
        let gate = MLXArray([0.5 as Float, -1.0, 2.0, -3.0], [1, 4]).asType(.bfloat16)

        let normalized = MLXFast.rmsNorm(hiddenStates, weight: layer.weight, eps: eps)
        let expected = preciseSiLUMultiply(hiddenStates, gate: gate, normalized: normalized)
        let actual = layer(hiddenStates, gate: gate)

        eval(expected, actual)

        XCTAssertEqual(actual.dtype, hiddenStates.dtype)
        XCTAssertTrue(
            allClose(
                actual.asType(DType.float32), expected.asType(DType.float32), rtol: 1e-3,
                atol: 1e-3
            ).item(Bool.self)
        )
    }

    func testQwen35SanitizeIsIdempotent() throws {
        let model = Qwen35Model(try makeConfiguration(modelType: "qwen3_5"))
        assertSanitizeIdempotence(model: model, hfMTPKey: "mtp.fc.weights")
    }

    func testQwen35MoESanitizeIsIdempotent() throws {
        let model = Qwen35MoEModel(
            try makeConfiguration(modelType: "qwen3_5_moe", numExperts: 2, numExpertsPerTok: 1))
        assertSanitizeIdempotence(model: model, hfMTPKey: "mtp.fc.weight")
    }
}
