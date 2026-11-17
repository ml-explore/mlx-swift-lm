import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXLLM

/// The DeepSeek-V3 MoE output must follow the activation dtype. Checkpoints
/// such as mlx-community/DeepSeek-V3.1-4bit store `e_score_correction_bias` as
/// float32 while the rest of the model is bfloat16. The router scores then
/// become float32, and without a cast back the expert sum promotes the residual
/// stream to float32 from the first MoE layer on.
final class DeepseekV3DtypeTests: XCTestCase {

    // Layer 0 is dense, layers 1 and 2 are MoE: 4 routed experts in 2 groups.
    // Layer 2 attention sees the residual stream after the first MoE layer.
    private func tinyConfig() throws -> DeepseekV3Configuration {
        try JSONDecoder().decode(
            DeepseekV3Configuration.self,
            from: Data(
                """
                {
                    "model_type": "deepseek_v3", "vocab_size": 32, "hidden_size": 8,
                    "intermediate_size": 16, "moe_intermediate_size": 8,
                    "num_hidden_layers": 3, "num_attention_heads": 2,
                    "num_key_value_heads": 2, "n_routed_experts": 4, "n_shared_experts": 1,
                    "num_experts_per_tok": 2, "n_group": 2, "topk_group": 1,
                    "norm_topk_prob": true, "routed_scaling_factor": 2.5,
                    "first_k_dense_replace": 1, "moe_layer_freq": 1,
                    "q_lora_rank": 4, "kv_lora_rank": 4, "qk_rope_head_dim": 2,
                    "qk_nope_head_dim": 2, "v_head_dim": 4, "rms_norm_eps": 1e-6,
                    "rope_theta": 10000.0, "max_position_embeddings": 128,
                    "attention_bias": false
                }
                """.utf8))
    }

    /// A bfloat16 model whose correction bias is float32 when `float32Bias`,
    /// as the checkpoint ships it.
    private func bf16Model(float32Bias: Bool) throws -> DeepseekV3Model {
        let model = DeepseekV3Model(try tinyConfig())
        let parameters = model.parameters().flattened().map { key, value in
            if key.hasSuffix("gate.weight") {
                return (key, MLXRandom.normal(value.shape).asType(.bfloat16))
            }
            if key.hasSuffix("e_score_correction_bias") {
                let bias = MLXRandom.normal(value.shape) * 0.1
                return (key, float32Bias ? bias : bias.asType(.bfloat16))
            }
            return (key, value.asType(.bfloat16))
        }
        model.update(parameters: ModuleParameters.unflattened(parameters))
        eval(model)
        return model
    }

    private func moeBlock(_ model: DeepseekV3Model) throws -> DeepseekV3MoE {
        try XCTUnwrap(model.model.layers[1].mlp as? DeepseekV3MoE)
    }

    func testMoEBlockKeepsBFloat16WithFloat32CorrectionBias() throws {
        let model = try bf16Model(float32Bias: true)
        let moe = try moeBlock(model)
        XCTAssertEqual(moe.gate.e_score_correction_bias.dtype, .float32)

        let x = MLXRandom.normal([1, 3, 8]).asType(.bfloat16)
        let y = moe(x)
        eval(y)
        XCTAssertEqual(y.dtype, .bfloat16)
    }

    func testForwardKeepsResidualStreamInInputDType() throws {
        for float32Bias in [true, false] {
            let model = try bf16Model(float32Bias: float32Bias)
            let cache = try model.newCache(parameters: nil)
            let inputs = MLXArray(Array(Int32(1) ... Int32(5))).reshaped(1, 5)

            let hidden = model.model(inputs, cache: cache)
            eval(hidden)
            XCTAssertEqual(hidden.dtype, .bfloat16, "prefill, float32Bias=\(float32Bias)")

            let step = model(MLXArray([Int32(6)]).reshaped(1, 1), cache: cache)
            eval(step)
            XCTAssertEqual(step.dtype, .bfloat16, "decode logits, float32Bias=\(float32Bias)")

            // Every layer's cache keeps its keys in the input dtype.
            for (layer, layerCache) in cache.enumerated() {
                XCTAssertEqual(
                    layerCache.state.first?.dtype, .bfloat16,
                    "cache layer \(layer), float32Bias=\(float32Bias)")
            }
        }
    }
}
