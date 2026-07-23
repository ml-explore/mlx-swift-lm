// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@_spi(Testing) @testable import MLXVLM

final class DeepseekOCRLanguageTests: XCTestCase {

    func testMoEGateUsesGroupedSigmoidRouting() throws {
        let model = try makeMoEModel()
        let gateWeights = concatenated([MLX.eye(4), zeros([4, 28], type: Float.self)], axis: 1)
        let correctionBias = MLXArray([Float](arrayLiteral: 0, 0, 1, 0))

        try model.update(
            parameters: ModuleParameters.unflattened([
                "model.layers.0.mlp.gate.weight": gateWeights,
                "model.layers.0.mlp.gate.e_score_correction_bias": correctionBias,
            ]),
            verify: [])

        let hiddenPrefix = MLXArray([Float](arrayLiteral: 2, 1, 0, -1)).reshaped(1, 1, 4)
        let hidden = concatenated([hiddenPrefix, zeros([1, 1, 28], type: Float.self)], axis: -1)
        let routed = try XCTUnwrap(model.routeLayerForTesting(hidden, layerIndex: 0))

        let indices = routed.0.reshaped(-1).asArray(Int32.self).map(Int.init)
        let scores = routed.1.reshaped(-1).asArray(Float.self)

        // Selection follows bias-corrected grouped scores (group {2,3} wins on
        // sigmoid(0) + 1 = 1.5), but the returned weights come from the raw
        // sigmoid scores — the Python reference gathers from `raw_flat`.
        XCTAssertEqual(Set(indices), [2, 3])
        XCTAssertEqual(scores.max() ?? .nan, 0.5, accuracy: 1e-4)
        XCTAssertEqual(scores.min() ?? .nan, 1 / (1 + expf(1)), accuracy: 1e-4)
    }

    func testSanitizeStacksQuantizedExpertWeightsIntoSwitchMLP() throws {
        let model = try makeMoEModel()
        let weights: [String: MLXArray] = [
            "model.layers.0.mlp.experts.0.gate_proj.weight": zeros([64, 32], type: UInt32.self),
            "model.layers.0.mlp.experts.1.gate_proj.weight": ones([64, 32], type: UInt32.self),
            "model.layers.0.mlp.experts.0.gate_proj.scales": zeros([64, 1], type: Float.self),
            "model.layers.0.mlp.experts.1.gate_proj.scales": ones([64, 1], type: Float.self),
            "model.layers.0.mlp.experts.0.gate_proj.biases": zeros([64, 1], type: Float.self),
            "model.layers.0.mlp.experts.1.gate_proj.biases": ones([64, 1], type: Float.self),
            "model.layers.0.mlp.experts.0.up_proj.weight": zeros([64, 32], type: UInt32.self),
            "model.layers.0.mlp.experts.1.up_proj.weight": ones([64, 32], type: UInt32.self),
            "model.layers.0.mlp.experts.0.down_proj.weight": zeros([32, 64], type: UInt32.self),
            "model.layers.0.mlp.experts.1.down_proj.weight": ones([32, 64], type: UInt32.self),
        ]

        let sanitized = model.sanitize(weights: weights)

        XCTAssertEqual(
            sanitized["model.layers.0.mlp.switch_mlp.gate_proj.weight"]?.shape, [2, 64, 32])
        XCTAssertEqual(
            sanitized["model.layers.0.mlp.switch_mlp.gate_proj.scales"]?.shape, [2, 64, 1])
        XCTAssertEqual(
            sanitized["model.layers.0.mlp.switch_mlp.gate_proj.biases"]?.shape, [2, 64, 1])
        XCTAssertEqual(
            sanitized["model.layers.0.mlp.switch_mlp.up_proj.weight"]?.shape, [2, 64, 32])
        XCTAssertEqual(
            sanitized["model.layers.0.mlp.switch_mlp.down_proj.weight"]?.shape, [2, 32, 64])
    }

    func testSingleTokenDecodeRunsWithQuantizedSwitchMLPWeights() throws {
        let source = try makeMoEModel()
        let target = try makeMoEModel()

        quantize(model: source) { path, _ in
            path.contains("model.layers.0.mlp.switch_mlp") ? (32, 4, .affine) : nil
        }
        quantize(model: target) { path, _ in
            path.contains("model.layers.0.mlp.switch_mlp") ? (32, 4, .affine) : nil
        }

        let quantizedWeights = Dictionary(
            uniqueKeysWithValues: source.parameters().flattened().filter {
                $0.0.contains("model.layers.0.mlp.switch_mlp")
            })

        try target.update(parameters: ModuleParameters.unflattened(quantizedWeights), verify: [])
        eval(target)

        let output = target(
            LMInput.Text(tokens: MLXArray([Int32(1)]).reshaped(1, 1)), cache: nil, state: nil)
        XCTAssertEqual(output.logits.shape, [1, 1, 32])
    }

    private func makeMoEModel() throws -> DeepseekOCR {
        let config = try JSONDecoder().decode(
            DeepseekOCRConfiguration.self,
            from: Data(Self.configJSON.utf8))
        return DeepseekOCR(config)
    }

    private static let configJSON = #"""
        {
          "model_type": "deepseekocr",
          "vision_config": {
            "hidden_size": 32,
            "output_channels": 8,
            "num_hidden_layers": 12,
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
            "moe_intermediate_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "max_position_embeddings": 32,
            "n_routed_experts": 4,
            "num_experts_per_tok": 2,
            "moe_layer_freq": 1,
            "first_k_dense_replace": 0,
            "norm_topk_prob": false,
            "n_group": 2,
            "topk_group": 1,
            "routed_scaling_factor": 1.0,
            "scoring_func": "sigmoid",
            "topk_method": "noaux_tc"
          }
        }
        """#
}
