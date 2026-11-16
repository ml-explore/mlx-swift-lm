import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@_spi(GemmaEncoder) @testable import MLXLLM

struct Gemma4TextTests {
    @Test(
        "Gemma4Text softcap preserves prefill and decode logits",
        arguments: [DType.float16, .bfloat16, .float32], [false, true]
    )
    func softcapPreservesLogits(dtype: DType, tiedEmbeddings: Bool) throws {
        for quantized in [false, true] {
            var config = try Self.configuration(attentionKEqV: true)
            config.hiddenSize = 64
            config.intermediateSize = 128
            config.tieWordEmbeddings = tiedEmbeddings
            config.finalLogitSoftcapping = 7.25
            let model = Gemma4TextModel(config)
            model.apply { $0.dtype.isFloatingPoint ? $0.asType(dtype) : $0 }
            if quantized { quantize(model: model, groupSize: 32, bits: 4) }
            eval(model)
            let actualCache = try model.newCache(parameters: nil)
            let expectedCache = try model.newCache(parameters: nil)

            for tokens in [[1, 2, 3], [4]] {
                let input = MLXArray(tokens).reshaped(1, tokens.count)
                let actual = model(input, cache: actualCache)
                let hidden = model.model(input, cache: expectedCache)
                let logits = model.lmHead?(hidden) ?? model.model.embedTokens.asLinear(hidden)
                let expected =
                    tanh(logits / config.finalLogitSoftcapping)
                    * config.finalLogitSoftcapping
                #expect(actual.dtype == expected.dtype)
                #expect(actual.shape == expected.shape)
                #expect(
                    actual.asType(.float32).asArray(Float.self).map(\.bitPattern)
                        == expected.asType(.float32).asArray(Float.self).map(\.bitPattern))
                #expect(actualCache.map(\.offset) == expectedCache.map(\.offset))
            }
        }
    }

    /// Gemma 4 declares its reasoning protocol through `ChatConventionsProviding`,
    /// the seam a model uses for conventions it knows about itself.
    @Test("Gemma4Text declares the labeled-channel reasoning config")
    func declaresChannelReasoningConfig() throws {
        let model = Gemma4TextModel(try Self.configuration(attentionKEqV: false))
        let config = try #require(model.reasoningConfig)
        #expect(config == .gemma4)
        #expect(config.channel == .gemma4)
        #expect(config.startDelimiter == "<|channel>")
        #expect(config.endDelimiter == "<channel|>")
    }

    @Test("Gemma4Text handles quantized KV cache in shared full attention")
    func quantizedKVCacheSupportsSharedFullAttention() throws {
        let model = Gemma4TextModel(try Self.configuration(attentionKEqV: false))
        eval(model)

        var cache: [KVCache] = try model.newCache(parameters: nil)
        let promptLogits = model(MLXArray([1, 2, 3]).reshaped([1, 3]), cache: cache)
        eval(promptLogits)
        #expect(promptLogits.shape == [1, 3, 32])

        maybeQuantizeKVCache(cache: &cache, kvBits: 4, kvGroupSize: 64, quantizedKVStart: 0)
        #expect(cache.contains { $0 is QuantizedKVCache })
        let quantizedCache = try #require(cache.first as? QuantizedKVCache)
        #expect(quantizedCache.groupSize == 32)

        let nextLogits = model(MLXArray([4]).reshaped([1, 1]), cache: cache)
        eval(nextLogits)
        #expect(nextLogits.shape == [1, 1, 32])
    }

    @Test("Gemma4Text handles K-equals-V full attention before and after KV quantization")
    func quantizedKVCacheSupportsKEqVFullAttention() throws {
        let model = Gemma4TextModel(try Self.configuration(attentionKEqV: true))
        eval(model)

        var cache: [KVCache] = try model.newCache(parameters: nil)
        let promptLogits = model(MLXArray([1, 2, 3]).reshaped([1, 3]), cache: cache)
        eval(promptLogits)
        #expect(promptLogits.shape == [1, 3, 32])

        maybeQuantizeKVCache(cache: &cache, kvBits: 4, kvGroupSize: 64, quantizedKVStart: 0)
        #expect(cache.contains { $0 is QuantizedKVCache })
        let quantizedCache = try #require(cache.first as? QuantizedKVCache)
        #expect(quantizedCache.groupSize == 32)

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
              "head_dim": 32,
              "global_head_dim": 32,
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
