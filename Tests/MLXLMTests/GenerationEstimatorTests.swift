// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLLM
import Testing

@testable import MLXLMCommon
@testable import MLXVLM

@Suite struct GenerationEstimatorTests {

    private func decode<C: Decodable>(_ type: C.Type, _ json: String) throws -> C {
        try JSONDecoder().decode(C.self, from: Data(json.utf8))
    }

    /// Six full-attention layers, so TurboQuant protects two layers at each end.
    private func qwen3() throws -> Qwen3Model {
        Qwen3Model(
            try decode(
                Qwen3Configuration.self,
                """
                {
                    "hidden_size": 64, "num_hidden_layers": 6, "intermediate_size": 64,
                    "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 64,
                    "rms_norm_eps": 1e-6, "vocab_size": 32, "rope_theta": 10000.0,
                    "tie_word_embeddings": true
                }
                """))
    }

    /// Sliding-window and full-attention layers.
    private func gptOSS() throws -> GPTOSSModel {
        GPTOSSModel(
            try decode(
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
                """))
    }

    /// Gated-delta recurrent layers and full-attention layers.
    private func qwen3Next() throws -> Qwen3NextModel {
        Qwen3NextModel(
            try decode(
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
                """))
    }

    /// Multi-head latent attention, with different key and value head dimensions.
    private func deepseekV2() throws -> DeepseekV2Model {
        DeepseekV2Model(
            try decode(
                DeepseekV2Configuration.self,
                """
                {
                    "model_type": "deepseek_v2", "vocab_size": 32, "hidden_size": 8,
                    "intermediate_size": 16, "moe_intermediate_size": 8,
                    "num_hidden_layers": 2, "num_attention_heads": 2, "num_key_value_heads": 2,
                    "n_routed_experts": 4, "n_shared_experts": 1, "num_experts_per_tok": 2,
                    "n_group": 2, "topk_group": 1, "topk_method": "group_limited_greedy",
                    "routed_scaling_factor": 1.0, "first_k_dense_replace": 1,
                    "moe_layer_freq": 1, "q_lora_rank": null, "kv_lora_rank": 4,
                    "qk_rope_head_dim": 2, "qk_nope_head_dim": 2, "v_head_dim": 4,
                    "rms_norm_eps": 1e-6, "rope_theta": 10000.0,
                    "max_position_embeddings": 128,
                    "rope_scaling": {"type": "yarn", "factor": 1.0, "mscale_all_dim": 0}
                }
                """))
    }

    /// The last two layers reuse earlier layers' keys and values.
    private func gemma4KVShared() throws -> Gemma4TextModel {
        Gemma4TextModel(
            try decode(
                Gemma4TextConfiguration.self,
                """
                {
                    "model_type": "gemma4_text", "hidden_size": 8, "num_hidden_layers": 4,
                    "intermediate_size": 16, "num_attention_heads": 2, "num_key_value_heads": 1,
                    "head_dim": 4, "global_head_dim": 8, "vocab_size": 12,
                    "vocab_size_per_layer_input": 12, "num_kv_shared_layers": 2,
                    "hidden_size_per_layer_input": 0, "sliding_window": 8,
                    "sliding_window_pattern": 2, "max_position_embeddings": 32,
                    "rms_norm_eps": 1e-6, "rope_traditional": false,
                    "use_double_wide_mlp": false, "enable_moe_block": false,
                    "attention_k_eq_v": false,
                    "layer_types": ["sliding_attention", "full_attention",
                                    "sliding_attention", "full_attention"],
                    "rope_parameters": {}, "tie_word_embeddings": true
                }
                """))
    }

    /// A vision-language model with gated-delta layers and compiled decode.
    private func qwen35VLM() throws -> Qwen35 {
        Qwen35(
            try decode(
                Qwen35Configuration.self,
                """
                {
                    "model_type": "qwen3_5_vl", "image_token_id": 500, "video_token_id": 501,
                    "vision_start_token_id": 502, "vision_end_token_id": 503,
                    "vocab_size": 512,
                    "text_config": {
                        "model_type": "qwen3_5", "hidden_size": 64, "num_hidden_layers": 4,
                        "intermediate_size": 128, "num_attention_heads": 4,
                        "num_key_value_heads": 2, "head_dim": 32, "vocab_size": 512,
                        "full_attention_interval": 2, "linear_num_value_heads": 4,
                        "linear_num_key_heads": 2, "linear_key_head_dim": 32,
                        "linear_value_head_dim": 32, "linear_conv_kernel_dim": 4,
                        "max_position_embeddings": 4096,
                        "rope_parameters": {
                            "type": "default", "mrope_section": [8, 4, 4],
                            "rope_theta": 100000.0, "partial_rotary_factor": 1.0
                        }
                    },
                    "vision_config": {
                        "model_type": "qwen3_vl", "depth": 2, "hidden_size": 32,
                        "intermediate_size": 64, "out_hidden_size": 64, "num_heads": 2,
                        "patch_size": 16, "spatial_merge_size": 2, "temporal_patch_size": 2,
                        "num_position_embeddings": 64
                    }
                }
                """))
    }

    /// The cache after a real prefill and one decode step, `tokens` positions in total.
    private func generatedCache(
        _ model: any LanguageModel, parameters: GenerateParameters, tokens: Int
    ) throws -> [KVCache] {
        let prompt = MLXArray((0 ..< tokens - 1).map { Int32($0 % 32) })
        var iterator = try TokenIterator(
            input: LMInput(tokens: prompt), model: model, parameters: parameters)
        _ = iterator.next()
        let cache = iterator.cache
        eval(cache)
        return cache
    }

    /// Check the estimate against every layer of a generated cache.
    ///
    /// Affine caches grow by one step on their first decode write, so they may hold
    /// up to 256 tokens more than the estimate.
    @discardableResult
    private func expectEstimateMatchesGeneration(
        _ model: any LanguageModel, parameters: GenerateParameters,
        allowingStepSlack: Bool = false, tokens: Int = 512
    ) throws -> GenerationEstimator.Cache {
        let estimate = try model.estimator(parameters: parameters).cache
        let leaves = KVCacheTree.leaves(
            in: try generatedCache(model, parameters: parameters, tokens: tokens))
        #expect(estimate.layers.map(\.path) == leaves.map(\.path))
        for (layer, leaf) in zip(estimate.layers, leaves) {
            let actual = leaf.cache.innerState().reduce(0) { $0 + $1.nbytes }
            let slack = actual - layer.bytes(tokens: tokens)
            let allowed = allowingStepSlack ? 0 ..< 256 * layer.bytesPerToken + 1 : 0 ..< 1
            #expect(allowed.contains(slack), "layer \(layer.path) holds \(slack) more bytes")
        }
        return estimate
    }

    @Test func fullPrecisionMatchesGeneration() throws {
        let estimate = try expectEstimateMatchesGeneration(qwen3(), parameters: .init())
        for layer in estimate.layers {
            // One KV head of 64 float32 values for keys and values.
            #expect(layer.bytesPerToken == 2 * 64 * 4)
            #expect(layer.fixedBytes == 0)
            #expect(layer.capacity == nil)
            #expect(layer.strategy == .fullPrecision)
        }
    }

    @Test func slidingWindowLayersStopGrowing() throws {
        let estimate = try expectEstimateMatchesGeneration(gptOSS(), parameters: .init())
        #expect(estimate.layers.map(\.capacity) == [8, nil, 8, nil])
    }

    @Test func recurrentStateDoesNotGrow() throws {
        let estimate = try expectEstimateMatchesGeneration(qwen3Next(), parameters: .init())
        let recurrent = estimate.layers.filter { $0.kind == .stateSpace }
        #expect(recurrent.count == 2)
        for layer in recurrent {
            #expect(layer.bytesPerToken == 0)
            #expect(layer.fixedBytes > 0)
            #expect(layer.strategy == nil)
        }
    }

    @Test func latentAttentionMatchesGeneration() throws {
        try expectEstimateMatchesGeneration(deepseekV2(), parameters: .init())
    }

    @Test func sharedKVLayersHaveNoCache() throws {
        let estimate = try expectEstimateMatchesGeneration(gemma4KVShared(), parameters: .init())
        #expect(estimate.layers.allSatisfy { $0.bytesPerToken > 0 })
        #expect(estimate.layers.count == 2)
    }

    @Test func probeLeavesCompiledDecodeUntouched() throws {
        let model = try qwen35VLM()
        let estimate = try model.estimator(parameters: nil).cache
        #expect(model.languageModel.model.compiledDecodeSegmentCount == 0)
        #expect(estimate.layers.map(\.kind.isAttention) == [false, true, false, true])
        try expectEstimateMatchesGeneration(model, parameters: .init())
    }

    @Test func requestedCapacityIncludesRewindReserve() throws {
        let parameters = GenerateParameters(
            kvCache: KVCacheConfiguration(
                capacity: try .init(maxTokens: 256), rewind: try .init(maxTokens: 64)))
        let estimate = try qwen3().estimator(parameters: parameters).cache
        #expect(estimate.layers.allSatisfy { $0.capacity == 320 })
        #expect(estimate.maxTokens(fittingIn: .max) == nil)
    }

    @Test func affineMatchesGeneration() throws {
        let parameters = GenerateParameters(
            kvCache: KVCacheConfiguration(strategy: .affine(.fourBit)))
        let estimate = try expectEstimateMatchesGeneration(
            qwen3(), parameters: parameters, allowingStepSlack: true)
        #expect(estimate.layers.allSatisfy { $0.strategy == .affine })
        #expect(estimate.layers.allSatisfy { $0.fixedBytes == 0 })
        let fullPrecision = try qwen3().estimator(parameters: .init()).cache
        #expect(estimate.bytes(tokens: 4096) < fullPrecision.bytes(tokens: 4096) / 3)
    }

    @Test func turboQuantMatchesGeneration() throws {
        let parameters = GenerateParameters(
            kvCache: KVCacheConfiguration(strategy: .turboQuant(.memoryFirst)))
        let estimate = try expectEstimateMatchesGeneration(
            qwen3(), parameters: parameters, allowingStepSlack: true)
        #expect(
            estimate.layers.map(\.strategy) == [
                .affine, .affine, .turboQuant, .turboQuant, .affine, .affine,
            ])
    }

    @Test func varianceNormalizedMatchesGeneration() throws {
        let parameters = GenerateParameters(
            kvCache: KVCacheConfiguration(strategy: .varianceNormalized(.balanced)))
        let estimate = try expectEstimateMatchesGeneration(qwen3(), parameters: parameters)
        #expect(estimate.layers.allSatisfy { $0.strategy == .varianceNormalized })
    }

    @Test func compressionStartDoesNotChangeTheEstimate() throws {
        let late = GenerateParameters(
            kvCache: KVCacheConfiguration(
                strategy: .affine(try .init(bits: 4, compressionStart: 100_000))))
        let early = GenerateParameters(
            kvCache: KVCacheConfiguration(strategy: .affine(.fourBit)))
        #expect(
            try qwen3().estimator(parameters: late).cache
                == qwen3().estimator(parameters: early).cache)
    }

    @Test func maxTokensFittingInBudget() {
        let estimate = GenerationEstimator.Cache(layers: [
            .init(
                path: [0], kind: .attention(maxSize: nil), strategy: .fullPrecision,
                bytesPerToken: 10, fixedBytes: 0, capacity: nil),
            .init(
                path: [1], kind: .rotatingAttention(maxSize: 100, keep: 0),
                strategy: .fullPrecision, bytesPerToken: 5, fixedBytes: 0, capacity: 100),
            .init(
                path: [2], kind: .stateSpace, strategy: nil,
                bytesPerToken: 0, fixedBytes: 1_000, capacity: nil),
        ])
        #expect(estimate.bytes(tokens: 0) == 1_000)
        #expect(estimate.bytes(tokens: 100) == 2_500)
        #expect(estimate.bytes(tokens: 200) == 3_500)
        #expect(estimate.maxTokens(fittingIn: 3_509) == 200)
        #expect(estimate.maxTokens(fittingIn: 3_510) == 201)
        #expect(estimate.maxTokens(fittingIn: 1_014) == 0)
        #expect(estimate.maxTokens(fittingIn: 999) == 0)
        #expect(estimate.maxTokens(fittingIn: .max) == (Int.max - 1_500) / 10)

        let bounded = GenerationEstimator.Cache(layers: Array(estimate.layers.dropFirst()))
        #expect(bounded.maxTokens(fittingIn: 1_500) == nil)
        #expect(bounded.maxTokens(fittingIn: 1_499) == 99)
    }

    @Test func cacheGrowsByItsBytesPerToken() {
        let estimate = GenerationEstimator.Cache(layers: [
            .init(
                path: [0], kind: .attention(maxSize: nil), strategy: .fullPrecision,
                bytesPerToken: 10, fixedBytes: 0, capacity: nil),
            .init(
                path: [1], kind: .rotatingAttention(maxSize: 100, keep: 0),
                strategy: .fullPrecision, bytesPerToken: 5, fixedBytes: 0, capacity: 100),
        ])
        #expect(estimate.bytesPerToken == 15)
        #expect(estimate.bytes(tokens: 200) - estimate.bytes(tokens: 100) == 10 * 100)
    }

    @Test func decodeReadsTheWeightsAndTheCache() {
        let estimator = GenerationEstimator(
            cache: .init(layers: [
                .init(
                    path: [0], kind: .attention(maxSize: nil), strategy: .fullPrecision,
                    bytesPerToken: 10, fixedBytes: 0, capacity: nil),
                .init(
                    path: [1], kind: .stateSpace, strategy: nil,
                    bytesPerToken: 0, fixedBytes: 1_000, capacity: nil),
            ]))
        #expect(estimator.decode.bytes(tokens: 100, weightBytes: 2_000) == 4_000)
        #expect(estimator.decode.bytes(tokens: 0, weightBytes: 2_000) == 3_000)
    }

    @Test func decodeFollowsTheCacheOfTheModel() throws {
        let estimator = try qwen3().estimator(parameters: nil)
        let cache = estimator.cache
        #expect(estimator.decode.bytes(tokens: 512, weightBytes: 0) == cache.bytes(tokens: 512))
        #expect(
            estimator.decode.bytes(tokens: 512, weightBytes: 1_000) == cache.bytes(tokens: 512)
                + 1_000)
    }

    @Test func decodeDoesNotReadTheRewindReserve() {
        let estimator = GenerationEstimator(
            cache: .init(layers: [
                .init(
                    path: [0], kind: .rotatingAttention(maxSize: 100, keep: 0),
                    strategy: .fullPrecision, bytesPerToken: 5, fixedBytes: 0, capacity: 160)
            ]))
        // The cache holds the window and the reserve, a step attends to the window.
        #expect(estimator.cache.bytes(tokens: 200) == 800)
        #expect(estimator.decode.bytes(tokens: 200, weightBytes: 0) == 500)
    }

    @Test func decodeCombinesTheLayersOfSeveralModels() throws {
        let target = try qwen3().estimator(parameters: nil)
        let drafter = try qwen3().estimator(parameters: nil)
        let combined = GenerationEstimator(
            cache: .init(layers: target.cache.layers + drafter.cache.layers))
        #expect(
            combined.decode.bytes(tokens: 512, weightBytes: 0) == 2
                * target.cache.bytes(tokens: 512))
    }
}
