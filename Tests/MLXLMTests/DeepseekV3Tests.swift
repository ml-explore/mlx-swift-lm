import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXLLM

final class DeepseekV3Tests: XCTestCase {

    // Tiny MoE config: 4 routed experts in 2 groups, top-1 group, top-2 experts.
    private func makeConfig(numHiddenLayers: Int = 2) throws -> DeepseekV3Configuration {
        let json = """
            {
                "model_type": "deepseek_v3",
                "vocab_size": 32,
                "hidden_size": 8,
                "intermediate_size": 16,
                "moe_intermediate_size": 8,
                "num_hidden_layers": \(numHiddenLayers),
                "num_attention_heads": 2,
                "num_key_value_heads": 2,
                "n_routed_experts": 4,
                "n_shared_experts": 1,
                "num_experts_per_tok": 2,
                "n_group": 2,
                "topk_group": 1,
                "norm_topk_prob": true,
                "routed_scaling_factor": 1.0,
                "first_k_dense_replace": 1,
                "moe_layer_freq": 1,
                "q_lora_rank": 4,
                "kv_lora_rank": 4,
                "qk_rope_head_dim": 2,
                "qk_nope_head_dim": 2,
                "v_head_dim": 4,
                "rms_norm_eps": 1e-6,
                "rope_theta": 10000.0,
                "max_position_embeddings": 128,
                "attention_bias": false
            }
            """
        return try JSONDecoder().decode(DeepseekV3Configuration.self, from: Data(json.utf8))
    }

    /// Attention must update the KV cache exactly once per forward pass.
    ///
    /// The attention path builds `keys`/`values` and hands them to
    /// `attentionWithCacheUpdate`, which performs the cache update itself. An
    /// additional explicit `cache.update(...)` beforehand appended the returned
    /// history a second time, so a prompt of L tokens left the cache at 2L and
    /// every subsequent token attended over duplicated keys.
    func testForwardPassUpdatesCacheExactlyOnce() throws {
        let model = DeepseekV3Model(try makeConfig())
        let cache = try model.newCache(parameters: nil)
        let tokenCount = 5
        let inputs = MLXArray(Array(Int32(1) ... Int32(tokenCount))).reshaped(1, tokenCount)

        let logits = model(inputs, cache: cache)
        eval(logits)

        for (i, layerCache) in cache.enumerated() {
            XCTAssertEqual(
                layerCache.offset, tokenCount,
                "layer \(i) cache advanced by \(layerCache.offset) for \(tokenCount) tokens")
        }
    }

    func testForwardPassProducesLogitsShape() throws {
        let model = DeepseekV3Model(try makeConfig())
        let inputs = MLXArray([1, 2, 3] as [Int32]).reshaped(1, 3)

        let logits = model(inputs, cache: nil)
        eval(logits)

        XCTAssertEqual(logits.shape, [1, 3, 32])
    }

    func testSanitizeDropsPredictionLayerAfterDeepseekV3Stack() throws {
        let model = DeepseekV3Model(try makeConfig(numHiddenLayers: 61))
        let sanitized = model.sanitize(weights: [
            "model.layers.60.self_attn.q_proj.weight": MLXArray.zeros([1]),
            "model.layers.61.self_attn.q_proj.weight": MLXArray.zeros([1]),
        ])

        XCTAssertNotNil(sanitized["model.layers.60.self_attn.q_proj.weight"])
        XCTAssertNil(sanitized["model.layers.61.self_attn.q_proj.weight"])
    }

    func testSanitizeUsesConfiguredLayerCount() throws {
        let model = DeepseekV3Model(try makeConfig(numHiddenLayers: 78))
        let sanitized = model.sanitize(weights: [
            "model.layers.61.self_attn.q_proj.weight": MLXArray.zeros([1]),
            "model.layers.77.self_attn.q_proj.weight": MLXArray.zeros([1]),
            "model.layers.78.self_attn.q_proj.weight": MLXArray.zeros([1]),
        ])

        XCTAssertNotNil(sanitized["model.layers.61.self_attn.q_proj.weight"])
        XCTAssertNotNil(sanitized["model.layers.77.self_attn.q_proj.weight"])
        XCTAssertNil(sanitized["model.layers.78.self_attn.q_proj.weight"])
    }

    /// Safetensors fp8 loads as raw bytes, so the block dequantization must decode them first,
    /// as mlx-lm's `from_fp8` does, and return bfloat16.
    func testSanitizeDecodesFP8BeforeScaling() throws {
        let model = DeepseekV3Model(try makeConfig())
        let values = sin(MLXArray(0 ..< (130 * 200)).asType(.float32) * Float(0.37)) * Float(3)
        let fp8 = toFP8(values.reshaped(130, 200))
        let scaleInv = MLXArray([0.5, 2.0, 4.0, 0.25] as [Float]).reshaped(2, 2)
        let key = "model.layers.0.self_attn.o_proj.weight"
        let sanitized = model.sanitize(weights: [key: fp8, "\(key)_scale_inv": scaleInv])

        let blockScales = repeated(repeated(scaleInv, count: 128, axis: 0), count: 128, axis: 1)[
            ..<130, ..<200]
        let expected = (fromFP8(fp8, dtype: .bfloat16) * blockScales).asType(.bfloat16)
        let weight = try XCTUnwrap(sanitized[key])
        XCTAssertEqual(weight.dtype, .bfloat16)
        XCTAssertEqual(weight.shape, [130, 200])
        XCTAssertTrue(arrayEqual(weight, expected).item(Bool.self))
        XCTAssertNil(sanitized["\(key)_scale_inv"])
    }

    /// fp8 experts are dequantized before they are stacked, and no per-expert key is left.
    func testSanitizeStacksDequantizedFP8Experts() throws {
        let model = DeepseekV3Model(try makeConfig())
        var weights = [String: MLXArray]()
        var expected = [MLXArray]()
        for expert in 0 ..< 4 {
            let values = cos(MLXArray(0 ..< 64).asType(.float32) * Float(expert + 1)).reshaped(8, 8)
            let key = "model.layers.1.mlp.experts.\(expert).gate_proj.weight"
            weights[key] = toFP8(values)
            weights["\(key)_scale_inv"] = MLXArray([Float(expert + 1)]).reshaped(1, 1)
            expected.append(
                (fromFP8(toFP8(values), dtype: .bfloat16) * Float(expert + 1))
                    .asType(.bfloat16))
        }
        let sanitized = model.sanitize(weights: weights)
        XCTAssertEqual(Array(sanitized.keys), ["model.layers.1.mlp.switch_mlp.gate_proj.weight"])
        let stackedWeight = try XCTUnwrap(
            sanitized["model.layers.1.mlp.switch_mlp.gate_proj.weight"])
        XCTAssertEqual(stackedWeight.dtype, .bfloat16)
        XCTAssertTrue(arrayEqual(stackedWeight, stacked(expected)).item(Bool.self))
    }
}
