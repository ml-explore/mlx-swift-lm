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
}
