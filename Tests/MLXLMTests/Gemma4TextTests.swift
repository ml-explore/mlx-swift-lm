import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import Testing

struct Gemma4TextTests {
    @Test("Gemma4Text handles quantized KV cache in shared full attention")
    func quantizedKVCacheSupportsSharedFullAttention() throws {
        let model = Gemma4TextModel(try Self.configuration(attentionKEqV: false))
        eval(model)

        var cache: [KVCache] = model.newCache(parameters: nil)
        let promptLogits = model(MLXArray([1, 2, 3]).reshaped([1, 3]), cache: cache)
        eval(promptLogits)
        #expect(promptLogits.shape == [1, 3, 32])

        maybeQuantizeKVCache(cache: &cache, kvBits: 4, kvGroupSize: 64, quantizedKVStart: 0)
        #expect(cache.contains { $0 is QuantizedKVCache })

        let nextLogits = model(MLXArray([4]).reshaped([1, 1]), cache: cache)
        eval(nextLogits)
        #expect(nextLogits.shape == [1, 1, 32])
    }

    @Test("Gemma4Text handles K-equals-V full attention before and after KV quantization")
    func quantizedKVCacheSupportsKEqVFullAttention() throws {
        let model = Gemma4TextModel(try Self.configuration(attentionKEqV: true))
        eval(model)

        var cache: [KVCache] = model.newCache(parameters: nil)
        let promptLogits = model(MLXArray([1, 2, 3]).reshaped([1, 3]), cache: cache)
        eval(promptLogits)
        #expect(promptLogits.shape == [1, 3, 32])

        maybeQuantizeKVCache(cache: &cache, kvBits: 4, kvGroupSize: 64, quantizedKVStart: 0)
        #expect(cache.contains { $0 is QuantizedKVCache })

        let nextLogits = model(MLXArray([4]).reshaped([1, 1]), cache: cache)
        eval(nextLogits)
        #expect(nextLogits.shape == [1, 1, 32])
    }

    private static func configuration(attentionKEqV: Bool) throws -> Gemma4TextConfiguration {
        let json = """
            {
              "model_type": "gemma4_text",
              "hidden_size": 16,
              "num_hidden_layers": 2,
              "intermediate_size": 32,
              "num_attention_heads": 2,
              "head_dim": 8,
              "global_head_dim": 8,
              "global_partial_rotary_factor": 0.25,
              "rms_norm_eps": 0.000001,
              "vocab_size": 32,
              "vocab_size_per_layer_input": 32,
              "num_key_value_heads": 1,
              "num_global_key_value_heads": 1,
              "num_kv_shared_layers": 1,
              "hidden_size_per_layer_input": 0,
              "sliding_window": 8,
              "sliding_window_pattern": 1,
              "max_position_embeddings": 64,
              "attention_k_eq_v": \(attentionKEqV),
              "final_logit_softcapping": 30.0,
              "use_double_wide_mlp": false,
              "layer_types": ["full_attention", "full_attention"],
              "tie_word_embeddings": true
            }
            """
        return try JSONDecoder().decode(
            Gemma4TextConfiguration.self, from: Data(json.utf8))
    }
}
