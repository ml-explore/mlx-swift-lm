import Foundation
import MLX
import MLXLMCommon
import XCTest

@testable import MLXLLM

/// Cached-forward smoke tests for models with a bespoke `newCache` / hybrid cache layout.
///
/// Each test builds a tiny model from an inline JSON config (no downloads), runs one
/// prefill, and asserts every attention layer's cache advanced by exactly the token count
/// (SSM/Mamba caches are checked for populated recurrent state instead).
final class CachedForwardSmokeTests: XCTestCase {

    /// Prefill `tokenCount` tokens and assert each KV cache advanced exactly once.
    ///
    /// SSM/Mamba caches carry recurrent state rather than a token offset, so they are
    /// checked for populated state instead.
    private func assertCacheAdvancesOnce(
        _ model: any LanguageModel,
        tokenCount: Int = 5,
        expectedLayers: Int,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        let cache = model.newCache(parameters: nil)
        XCTAssertEqual(
            cache.count, expectedLayers,
            "newCache must return one cache per layer", file: file, line: line)

        let inputs = MLXArray(Array(Int32(1) ... Int32(tokenCount))).reshaped(1, tokenCount)
        let logits = model(inputs, cache: cache)
        eval(logits)

        for (i, layerCache) in cache.enumerated() {
            switch layerCache {
            case let list as CacheList:
                // Composite layout (e.g. BaichuanM1): sub-cache 0 holds recurrent
                // conv state, sub-cache 1 is the KV cache that tracks the offset.
                XCTAssertEqual(
                    list[1].offset, tokenCount,
                    "layer \(i) KV sub-cache advanced by \(list[1].offset) for \(tokenCount) tokens",
                    file: file, line: line)
            case is MambaCache:
                XCTAssertFalse(
                    layerCache.innerState().isEmpty,
                    "layer \(i) SSM cache was never written", file: file, line: line)
            default:
                XCTAssertEqual(
                    layerCache.offset, tokenCount,
                    "layer \(i) cache advanced by \(layerCache.offset) for \(tokenCount) tokens",
                    file: file, line: line)
            }
        }
    }

    private func config<C: Decodable>(_ type: C.Type, _ json: String) throws -> C {
        try JSONDecoder().decode(C.self, from: Data(json.utf8))
    }

    // MARK: - Olmo3 (sliding-window / full-attention hybrid)

    private func olmo3Config() throws -> Olmo3Configuration {
        try config(
            Olmo3Configuration.self,
            """
            {
                "hidden_size": 8, "num_hidden_layers": 4, "intermediate_size": 16,
                "num_attention_heads": 2, "head_dim": 4, "num_key_value_heads": 1,
                "rms_norm_eps": 1e-6, "vocab_size": 32, "max_position_embeddings": 128,
                "sliding_window": 8, "rope_theta": 10000.0,
                "layer_types": ["sliding_attention", "sliding_attention",
                                "full_attention", "sliding_attention"],
                "tie_word_embeddings": true
            }
            """)
    }

    func testOlmo3ForwardPassUpdatesCacheExactlyOnce() throws {
        try assertCacheAdvancesOnce(Olmo3Model(olmo3Config()), expectedLayers: 4)
    }

    /// Each sliding-window layer must get a `RotatingKVCache` (bounded to the window) and
    /// each full-attention layer a `KVCacheSimple`.
    func testOlmo3SlidingLayersGetRotatingCache() throws {
        let config = try olmo3Config()
        let model: any LanguageModel = Olmo3Model(config)
        let cache = model.newCache(parameters: nil)

        for (i, layerType) in config.layerTypes.enumerated() {
            if layerType == "full_attention" {
                XCTAssertTrue(
                    cache[i] is KVCacheSimple,
                    "layer \(i) is full_attention but got \(type(of: cache[i]))")
            } else {
                XCTAssertTrue(
                    cache[i] is RotatingKVCache,
                    "layer \(i) is \(layerType) but got \(type(of: cache[i])) — "
                        + "the sliding window is not enforced")
            }
        }
    }

    // MARK: - Jamba (Mamba + attention + MoE)

    func testJambaForwardPassUpdatesCacheExactlyOnce() throws {
        // attn_layer_period 2 / offset 1 ⇒ layers 1 and 3 are attention, 0 and 2 mamba.
        let config = try config(
            JambaConfiguration.self,
            """
            {
                "model_type": "jamba", "hidden_size": 8, "intermediate_size": 16,
                "num_hidden_layers": 4, "num_attention_heads": 2, "num_key_value_heads": 1,
                "attn_layer_offset": 1, "attn_layer_period": 2,
                "expert_layer_offset": 1, "expert_layer_period": 2,
                "mamba_d_conv": 4, "mamba_d_state": 8, "mamba_expand": 2,
                "num_experts": 4, "num_experts_per_tok": 2,
                "rms_norm_eps": 1e-6, "max_position_embeddings": 128, "vocab_size": 32,
                "tie_word_embeddings": true
            }
            """)
        try assertCacheAdvancesOnce(JambaModel(config), expectedLayers: 4)
    }

    // MARK: - GraniteMoeHybrid (Mamba2 + attention + MoE)

    func testGraniteMoeHybridForwardPassUpdatesCacheExactlyOnce() throws {
        let config = try config(
            GraniteMoeHybridConfiguration.self,
            """
            {
                "model_type": "granitemoehybrid", "vocab_size": 32, "hidden_size": 8,
                "intermediate_size": 16, "num_hidden_layers": 4,
                "max_position_embeddings": 128, "num_attention_heads": 2,
                "num_key_value_heads": 1, "attention_bias": false,
                "embedding_multiplier": 1.0, "attention_multiplier": 1.0,
                "logits_scaling": 1.0, "residual_multiplier": 1.0,
                "layer_types": ["mamba", "attention", "mamba", "attention"],
                "rms_norm_eps": 1e-6, "rope_theta": 10000.0,
                "num_local_experts": 4, "num_experts_per_tok": 2,
                "shared_intermediate_size": 8,
                "mamba_n_heads": 2, "mamba_d_head": 4, "mamba_d_state": 8,
                "mamba_d_conv": 4, "mamba_n_groups": 1,
                "mlp_bias": false, "position_embedding_type": "rope",
                "tie_word_embeddings": true
            }
            """)
        try assertCacheAdvancesOnce(GraniteMoeHybridModel(config), expectedLayers: 4)
    }

    // MARK: - Qwen3Next (gated-delta linear attention + full attention + MoE)

    func testQwen3NextForwardPassUpdatesCacheExactlyOnce() throws {
        // full_attention_interval 2 ⇒ layers 0 and 2 are linear, 1 and 3 full attention.
        let config = try config(
            Qwen3NextConfiguration.self,
            """
            {
                "hidden_size": 16, "num_hidden_layers": 4, "intermediate_size": 32,
                "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 8,
                "linear_num_value_heads": 2, "linear_num_key_heads": 2,
                "linear_key_head_dim": 8, "linear_value_head_dim": 8,
                "linear_conv_kernel_dim": 4,
                "num_experts": 4, "num_experts_per_tok": 2, "decoder_sparse_step": 1,
                "shared_expert_intermediate_size": 16, "mlp_only_layers": [],
                "moe_intermediate_size": 16, "rms_norm_eps": 1e-6, "vocab_size": 32,
                "rope_theta": 10000.0, "partial_rotary_factor": 0.25,
                "max_position_embeddings": 128, "norm_topk_prob": true,
                "tie_word_embeddings": true, "attention_bias": false,
                "full_attention_interval": 2
            }
            """)
        try assertCacheAdvancesOnce(Qwen3NextModel(config), expectedLayers: 4)
    }

    // MARK: - GPTOSS (sliding-window rotating cache + attention sinks + MoE)

    func testGPTOSSForwardPassUpdatesCacheExactlyOnce() throws {
        let config = try config(
            GPTOSSConfiguration.self,
            """
            {
                "model_type": "gpt_oss", "num_hidden_layers": 4, "num_local_experts": 4,
                "num_experts_per_tok": 2, "vocab_size": 32, "rms_norm_eps": 1e-6,
                "hidden_size": 16, "intermediate_size": 16, "head_dim": 8,
                "num_attention_heads": 2, "num_key_value_heads": 1,
                "sliding_window": 8, "rope_theta": 10000.0,
                "layer_types": ["sliding_attention", "full_attention",
                                "sliding_attention", "full_attention"]
            }
            """)
        try assertCacheAdvancesOnce(GPTOSSModel(config), expectedLayers: 4)
    }

    // MARK: - BaichuanM1 (per-layer CacheList: conv MambaCache + sliding/full KV)

    func testBaichuanM1ForwardPassUpdatesCacheExactlyOnce() throws {
        // slidingWindowLayers [0, 2] ⇒ layers 0 and 2 are SWA (RotatingKVCache),
        // layers 1 and 3 full (KVCacheSimple), each wrapped in a CacheList with a
        // MambaCache holding the short convolution state.
        let config = try config(
            BaichuanM1Configuration.self,
            """
            {
                "vocab_size": 32, "hidden_size": 8, "intermediate_size": 16,
                "num_hidden_layers": 4, "num_attention_heads": 2, "num_key_value_heads": 1,
                "rope_theta": 10000.0, "sliding_window": 8, "sliding_window_layers": [0, 2],
                "conv_window": 2, "rms_norm_eps": 1e-6, "tie_word_embeddings": true
            }
            """)
        try assertCacheAdvancesOnce(BaichuanM1Model(config), expectedLayers: 4)
    }

    // MARK: - MiMoV2Flash (hybrid sliding/full attention + attention sinks)

    func testMiMoV2FlashForwardPassUpdatesCacheExactlyOnce() throws {
        // hybrid_layer_pattern [1,0,1,0] ⇒ layers 0,2 sliding, 1,3 full.
        // moe_layer_freq all 0 keeps every layer dense (cache split is the target).
        let config = try config(
            MiMoV2FlashConfiguration.self,
            """
            {
                "model_type": "mimo_v2_flash", "num_experts_per_tok": 2,
                "hybrid_layer_pattern": [1, 0, 1, 0], "moe_layer_freq": [0, 0, 0, 0],
                "add_swa_attention_sink_bias": true, "add_full_attention_sink_bias": true,
                "sliding_window_size": 8, "vocab_size": 32, "hidden_size": 8,
                "intermediate_size": 16, "moe_intermediate_size": 8, "num_hidden_layers": 4,
                "num_attention_heads": 2, "num_key_value_heads": 1,
                "topk_method": "greedy", "scoring_func": "softmax", "norm_topk_prob": true,
                "n_group": 1, "topk_group": 1, "max_position_embeddings": 128,
                "layernorm_epsilon": 1e-6, "rope_theta": 10000.0, "swa_rope_theta": 10000.0,
                "swa_num_attention_heads": 2, "swa_num_key_value_heads": 1,
                "head_dim": 4, "v_head_dim": 4, "swa_head_dim": 4, "swa_v_head_dim": 4,
                "partial_rotary_factor": 1.0
            }
            """)
        try assertCacheAdvancesOnce(MiMoV2FlashModel(config), expectedLayers: 4)
    }

    // MARK: - AfMoE (layer_types sliding/full + MoE routing)

    func testAfMoEForwardPassUpdatesCacheExactlyOnce() throws {
        // num_dense_layers 2 ⇒ layers 0,1 dense, 2,3 MoE. layer_types drive the cache split.
        let config = try config(
            AfMoEConfiguration.self,
            """
            {
                "model_type": "afmoe", "vocab_size": 32, "hidden_size": 8,
                "intermediate_size": 16, "moe_intermediate_size": 8, "num_hidden_layers": 4,
                "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 4,
                "rms_norm_eps": 1e-6, "num_experts": 4, "num_experts_per_tok": 2,
                "num_shared_experts": 1, "num_dense_layers": 2, "n_group": 1, "topk_group": 1,
                "layer_types": ["sliding_attention", "full_attention",
                                "sliding_attention", "full_attention"],
                "sliding_window": 8, "tie_word_embeddings": true
            }
            """)
        try assertCacheAdvancesOnce(AfMoEModel(config), expectedLayers: 4)
    }

    // MARK: - Exaone4 (sliding_window_pattern string → local/global cache)

    func testExaone4ForwardPassUpdatesCacheExactlyOnce() throws {
        // pattern "LLLG" ⇒ layers 0,1,2 local (RotatingKVCache), layer 3 global (StandardKVCache).
        let config = try config(
            Exaone4Configuration.self,
            """
            {
                "hidden_size": 8, "num_hidden_layers": 4, "intermediate_size": 16,
                "num_attention_heads": 2, "rms_norm_eps": 1e-6, "vocab_size": 32,
                "num_key_value_heads": 1, "max_position_embeddings": 128,
                "rope_theta": 10000.0, "head_dim": 4, "tie_word_embeddings": true,
                "sliding_window": 8, "sliding_window_pattern": "LLLG"
            }
            """)
        try assertCacheAdvancesOnce(Exaone4Model(config), expectedLayers: 4)
    }

    // MARK: - Mistral3Text (layer_types sliding/full)

    func testMistral3TextForwardPassUpdatesCacheExactlyOnce() throws {
        let config = try config(
            Mistral3TextConfiguration.self,
            """
            {
                "model_type": "ministral3", "hidden_size": 8, "num_hidden_layers": 4,
                "intermediate_size": 16, "num_attention_heads": 2, "rms_norm_eps": 1e-6,
                "vocab_size": 32, "num_key_value_heads": 1, "head_dim": 4,
                "rope_theta": 10000.0, "tie_word_embeddings": true, "sliding_window": 8,
                "layer_types": ["sliding_attention", "full_attention",
                                "sliding_attention", "full_attention"]
            }
            """)
        try assertCacheAdvancesOnce(Mistral3TextModel(config), expectedLayers: 4)
    }
}
