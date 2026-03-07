// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN
import XCTest

@testable import MLXVLM

final class Qwen35VLMTests: XCTestCase {

    private func makeConfiguration() throws -> Qwen35Configuration {
        let json = """
            {
                "model_type": "qwen3_5",
                "text_config": {
                    "model_type": "qwen3_5",
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
                    "full_attention_interval": 4,
                    "tie_word_embeddings": false,
                    "attention_bias": false
                },
                "vision_config": {
                    "model_type": "qwen3_vl",
                    "depth": 1,
                    "hidden_size": 8,
                    "intermediate_size": 16,
                    "out_hidden_size": 8,
                    "num_heads": 1,
                    "patch_size": 2,
                    "spatial_merge_size": 1,
                    "temporal_patch_size": 1,
                    "num_position_embeddings": 16
                }
            }
            """

        return try JSONDecoder().decode(Qwen35Configuration.self, from: Data(json.utf8))
    }

    func testRMSNormGatedMatchesCurrentPythonVLMBehavior() {
        let eps: Float = 1e-6
        let layer = Qwen35Language.RMSNormGated(dimensions: 4, eps: eps)
        let hiddenStates = MLXArray([1.0 as Float, -2.0, 3.0, -4.0], [1, 4]).asType(.bfloat16)
        let gate = MLXArray([0.5 as Float, -1.0, 2.0, -3.0], [1, 4]).asType(.bfloat16)

        let normalized = MLXFast.rmsNorm(hiddenStates, weight: layer.weight, eps: eps)
        let expected = silu(gate) * normalized
        let actual = layer(hiddenStates, gate: gate)

        eval(expected, actual)

        XCTAssertEqual(actual.dtype, hiddenStates.dtype)
        XCTAssertTrue(
            allClose(
                actual.asType(.float32), expected.asType(.float32), rtol: 1e-3, atol: 1e-3
            ).item(Bool.self)
        )
    }

    func testSanitizeMatchesCurrentPythonVLMBehavior() throws {
        let config = try makeConfiguration()
        let model = Qwen35(config)

        let hfNormKey = "model.language_model.model.layers.0.input_layernorm.weight"
        let convertedNormKey = "language_model.model.layers.0.input_layernorm.weight"
        let hfConvKey = "model.language_model.model.layers.0.linear_attn.conv1d.weight"
        let convertedConvKey = "language_model.model.layers.0.linear_attn.conv1d.weight"
        let hfCorrectConvKey = "model.language_model.model.layers.1.linear_attn.conv1d.weight"
        let convertedCorrectConvKey = "language_model.model.layers.1.linear_attn.conv1d.weight"

        let keyDim =
            config.textConfiguration.linearNumKeyHeads
            * config.textConfiguration.linearKeyHeadDim
        let valueDim =
            config.textConfiguration.linearNumValueHeads
            * config.textConfiguration.linearValueHeadDim
        let convDim = (keyDim * 2) + valueDim
        let kernelSize = config.textConfiguration.linearConvKernelDim

        let base = MLXArray(0 ..< 8).asType(.float32)
        let convNeedsTranspose = MLXArray(0 ..< (convDim * kernelSize))
            .asType(.float32)
            .reshaped(convDim, 1, kernelSize)
        let convAlreadyCorrect = MLXArray(0 ..< (convDim * kernelSize))
            .asType(.float32)
            .reshaped(convDim, kernelSize, 1)

        let sanitized = model.sanitize(weights: [
            hfNormKey: base,
            hfConvKey: convNeedsTranspose,
            hfCorrectConvKey: convAlreadyCorrect,
            "mtp.fc.weight": MLXArray.zeros([1], dtype: .float32),
        ])

        eval(
            sanitized[convertedNormKey]!,
            sanitized[convertedConvKey]!,
            sanitized[convertedCorrectConvKey]!
        )

        XCTAssertFalse(sanitized.keys.contains { $0.contains("mtp.") })
        XCTAssertTrue(
            arrayEqual(
                sanitized[convertedNormKey]!,
                base + MLXArray(1.0, dtype: .float32)
            ).item(Bool.self)
        )
        XCTAssertTrue(
            arrayEqual(
                sanitized[convertedConvKey]!,
                convNeedsTranspose.movedAxis(source: 2, destination: 1)
            ).item(Bool.self)
        )
        XCTAssertTrue(
            arrayEqual(sanitized[convertedCorrectConvKey]!, convAlreadyCorrect).item(Bool.self)
        )
    }

    func testSanitizeBypassesMLXFormattedWeights() throws {
        let model = Qwen35(try makeConfiguration())
        let key = "language_model.model.layers.0.input_layernorm.weight"
        let weight = MLXArray(0 ..< 8).asType(.float32)

        let sanitized = model.sanitize(weights: [key: weight], metadata: ["format": "mlx"])

        eval(sanitized[key]!)

        XCTAssertTrue(arrayEqual(sanitized[key]!, weight).item(Bool.self))
    }
}
