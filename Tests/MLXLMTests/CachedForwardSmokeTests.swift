import Foundation
import MLX
import MLXLMCommon
import XCTest

@testable import MLXLLM

/// Cached-forward smoke tests for models with a bespoke `newCache` / hybrid cache layout.
///
/// Each test builds a tiny model from an inline JSON config (no downloads), runs one
/// prefill through `newCache(parameters:)`, and asserts every attention layer's cache
/// advanced by exactly the token count. This catches the two structural failures that
/// shipped undetected in DeepSeek-V3: a cache array whose length disagrees with the
/// layer count (index-out-of-range), and a double KV update (offset == 2 * tokens).
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
            if layerCache is MambaCache {
                XCTAssertFalse(
                    layerCache.innerState().isEmpty,
                    "layer \(i) SSM cache was never written", file: file, line: line)
            } else {
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

    /// Regression: `newCache` must be reachable through ``LanguageModel``.
    ///
    /// Olmo3 declared `newCache(parameters: GenerateParameters)` — a non-optional
    /// parameter, which does *not* satisfy the protocol requirement
    /// `newCache(parameters: GenerateParameters?)`. The model also conforms to
    /// ``KVCacheDimensionProvider``, so the protocol picked up the generic default
    /// (all-`KVCacheSimple`) and the sliding-window branch never ran: sliding layers
    /// got an unbounded cache and attended past the window during decode.
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
}
