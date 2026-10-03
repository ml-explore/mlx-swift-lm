import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import Testing

struct GLM4MoELiteTests {
    @Test("GLM4MoELite decodes through a quantized KV cache")
    func quantizedKVCacheDecodesAfterPrefill() throws {
        let model = GLM4MoELiteModel(try Self.configuration())
        eval(model)

        var cache: [KVCache] = try model.newCache(parameters: nil)
        let promptLogits = model(MLXArray([1, 2, 3]).reshaped([1, 3]), cache: cache)
        eval(promptLogits)
        #expect(promptLogits.shape == [1, 3, 32])

        maybeQuantizeKVCache(cache: &cache, kvBits: 8, kvGroupSize: 64, quantizedKVStart: 0)
        let quantizedCache = try #require(cache.first as? QuantizedKVCache)
        #expect(quantizedCache.groupSize == 64)

        let nextLogits = model(MLXArray([4]).reshaped([1, 1]), cache: cache)
        eval(nextLogits)
        #expect(nextLogits.shape == [1, 1, 32])
        #expect(nextLogits.asType(.float32).sum().item(Float.self).isFinite)
    }

    /// The attention caches `[kvLatent, kPe]` as keys and `kvLatent` as values, so keys are
    /// wider than values and a single KV head serves every query head.
    private static func configuration() throws -> GLM4MoELiteConfiguration {
        let json = """
            {
              "model_type": "glm4_moe_lite",
              "vocab_size": 32,
              "hidden_size": 32,
              "intermediate_size": 64,
              "moe_intermediate_size": 64,
              "num_hidden_layers": 2,
              "num_attention_heads": 2,
              "num_key_value_heads": 2,
              "n_shared_experts": null,
              "n_routed_experts": null,
              "routed_scaling_factor": 1.0,
              "kv_lora_rank": 128,
              "q_lora_rank": 16,
              "qk_rope_head_dim": 64,
              "qk_nope_head_dim": 64,
              "v_head_dim": 32,
              "norm_topk_prob": true,
              "n_group": 1,
              "topk_group": 1,
              "num_experts_per_tok": 1,
              "first_k_dense_replace": 2,
              "max_position_embeddings": 128,
              "rms_norm_eps": 1e-6,
              "rope_theta": 10000.0,
              "attention_bias": false,
              "partial_rotary_factor": 1.0,
              "tie_word_embeddings": true,
              "num_nextn_predict_layers": 0
            }
            """
        return try JSONDecoder().decode(GLM4MoELiteConfiguration.self, from: Data(json.utf8))
    }
}
