// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import MLXVLM
import Testing

// MARK: - Config decode (no MLX kernels required)

@Test
func testGemma4AssistantConfigurationDecodesSyntheticJSON() throws {
    let json = """
        {
          "model_type": "gemma4_assistant",
          "backbone_hidden_size": 5376,
          "tie_word_embeddings": true,
          "use_ordered_embeddings": false,
          "num_centroids": 2048,
          "centroid_intermediate_top_k": 32,
          "text_config": {
            "model_type": "gemma4_text",
            "hidden_size": 1024,
            "num_hidden_layers": 4,
            "num_attention_heads": 32,
            "num_key_value_heads": 16,
            "head_dim": 256,
            "global_head_dim": 512,
            "vocab_size": 262144,
            "num_kv_shared_layers": 0,
            "hidden_size_per_layer_input": 0,
            "sliding_window": 512,
            "sliding_window_pattern": 5,
            "max_position_embeddings": 262144,
            "rms_norm_eps": 1e-6,
            "rope_traditional": false,
            "use_double_wide_mlp": false,
            "enable_moe_block": false,
            "attention_k_eq_v": true,
            "intermediate_size": 8192,
            "layer_types": ["sliding_attention", "sliding_attention", "sliding_attention", "full_attention"],
            "rope_parameters": {},
            "tie_word_embeddings": true
          }
        }
        """
    let data = Data(json.utf8)
    let cfg = try JSONDecoder().decode(Gemma4AssistantConfiguration.self, from: data)
    #expect(cfg.modelType == "gemma4_assistant")
    #expect(cfg.backboneHiddenSize == 5376)
    #expect(cfg.tieWordEmbeddings == true)
    #expect(cfg.useOrderedEmbeddings == false)
    #expect(cfg.numCentroids == 2048)
    #expect(cfg.centroidIntermediateTopK == 32)
    #expect(cfg.textConfiguration.hiddenSize == 1024)
    #expect(cfg.textConfiguration.hiddenLayers == 4)
    #expect(cfg.textConfiguration.vocabularySize == 262_144)
    #expect(cfg.textConfiguration.layerTypes.count == 4)
}

// MARK: - Sanitize behavior (no MLX kernels — uses MLXArray scalar metadata only)

@Test
func testSanitizeDropsLmHeadWhenTied() {
    let cfg = syntheticConfig(tieWordEmbeddings: true)
    let model = Gemma4AssistantDraftModel(cfg)
    let weights: [String: MLXArray] = [
        "model.embed_tokens.weight": MLXArray.zeros([10, 4]),
        "lm_head.weight": MLXArray.zeros([10, 4]),
        "pre_projection.weight": MLXArray.zeros([4, 8]),
    ]
    let sanitized = model.sanitize(weights: weights)
    #expect(sanitized["lm_head.weight"] == nil)
    #expect(sanitized["model.embed_tokens.weight"] != nil)
    #expect(sanitized["pre_projection.weight"] != nil)
}

@Test
func testSanitizeKeepsLmHeadWhenNotTied() {
    let cfg = syntheticConfig(tieWordEmbeddings: false)
    let model = Gemma4AssistantDraftModel(cfg)
    let weights: [String: MLXArray] = [
        "model.embed_tokens.weight": MLXArray.zeros([10, 4]),
        "lm_head.weight": MLXArray.zeros([10, 4]),
    ]
    let sanitized = model.sanitize(weights: weights)
    #expect(sanitized["lm_head.weight"] != nil)
}

// MARK: - Synthetic shape test (no checkpoint needed)

@Test
func testGemma4AssistantDraftModelInstantiatesAndShape() {
    let cfg = syntheticConfig(tieWordEmbeddings: true)
    let model = Gemma4AssistantDraftModel(cfg)
    #expect(model.config.modelType == "gemma4_assistant")
    #expect(model.config.backboneHiddenSize == 4)
    #expect(model.config.tieWordEmbeddings == true)
    // The inner Embedding and Linears are constructed at init.
    // We don't run inference here (would need actual weights + metal kernels).
}

// MARK: - Helpers

private func syntheticConfig(tieWordEmbeddings: Bool) -> Gemma4AssistantConfiguration {
    // Build a tiny synthetic config sufficient for sanitize / instantiation tests.
    // Uses JSON round-trip to avoid manually constructing Gemma4TextConfiguration
    // (which has many defaulted fields and a custom decoder).
    let textJSON =
        """
        {
          "model_type": "gemma4_text",
          "hidden_size": 4,
          "num_hidden_layers": 1,
          "num_attention_heads": 2,
          "num_key_value_heads": 1,
          "head_dim": 2,
          "global_head_dim": 2,
          "vocab_size": 10,
          "num_kv_shared_layers": 0,
          "hidden_size_per_layer_input": 0,
          "sliding_window": 4,
          "sliding_window_pattern": 1,
          "max_position_embeddings": 16,
          "rms_norm_eps": 1e-6,
          "rope_traditional": false,
          "use_double_wide_mlp": false,
          "enable_moe_block": false,
          "attention_k_eq_v": true,
          "intermediate_size": 8,
          "layer_types": ["full_attention"],
          "rope_parameters": {},
          "tie_word_embeddings": \(tieWordEmbeddings)
        }
        """
    let json =
        """
        {
          "model_type": "gemma4_assistant",
          "backbone_hidden_size": 4,
          "tie_word_embeddings": \(tieWordEmbeddings),
          "use_ordered_embeddings": false,
          "num_centroids": 2,
          "centroid_intermediate_top_k": 1,
          "text_config": \(textJSON)
        }
        """
    return try! JSONDecoder().decode(
        Gemma4AssistantConfiguration.self, from: Data(json.utf8))
}

// MARK: - gemma4_unified_assistant (the 12B drafter)

/// Decode pin for the unified-assistant config shape: mirrors
/// `mlx-community/gemma-4-12B-it-assistant-4bit`'s `config.json`
/// (`model_type: gemma4_unified_assistant`, `text_config.model_type:
/// gemma4_unified_text`, `attention_k_eq_v`, `num_global_key_value_heads`).
@Test
func testGemma4UnifiedAssistantConfigurationDecodes() throws {
    let json = """
        {
          "model_type": "gemma4_unified_assistant",
          "backbone_hidden_size": 3840,
          "tie_word_embeddings": true,
          "use_ordered_embeddings": false,
          "num_centroids": 2048,
          "centroid_intermediate_top_k": 32,
          "text_config": {
            "model_type": "gemma4_unified_text",
            "hidden_size": 1024,
            "num_hidden_layers": 4,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "num_global_key_value_heads": 1,
            "head_dim": 256,
            "global_head_dim": 512,
            "vocab_size": 262144,
            "num_kv_shared_layers": 4,
            "hidden_size_per_layer_input": 0,
            "sliding_window": 1024,
            "max_position_embeddings": 262144,
            "rms_norm_eps": 1e-6,
            "attention_k_eq_v": true,
            "intermediate_size": 8192,
            "layer_types": ["sliding_attention", "sliding_attention", "sliding_attention", "full_attention"],
            "rope_parameters": {},
            "tie_word_embeddings": true
          }
        }
        """
    let cfg = try JSONDecoder().decode(
        Gemma4AssistantConfiguration.self, from: Data(json.utf8))
    #expect(cfg.modelType == "gemma4_unified_assistant")
    #expect(cfg.backboneHiddenSize == 3840)
    #expect(cfg.useOrderedEmbeddings == false)
    #expect(cfg.textConfiguration.modelType == "gemma4_unified_text")
    #expect(cfg.textConfiguration.attentionKEqV == true)
    #expect(cfg.textConfiguration.globalKVHeads == 1)
    #expect(cfg.textConfiguration.hiddenSize == 1024)

    // Instantiation must succeed with the unified text config.
    let model = Gemma4AssistantDraftModel(cfg)
    #expect(model.config.modelType == "gemma4_unified_assistant")
}

/// End-to-end synthetic pin for the `gemma4_unified` target path: a tiny
/// `Gemma4Unified` (a) emits drafter state through the MTP-aware
/// `callAsFunction(_:cache:state:)` entry point, and (b) is accepted by
/// `draftBlock` as the target (the pre-fix guard only recognized `Gemma4`
/// and trapped on `Gemma4Unified`, which the iterator surfaced as sticky
/// passthrough — MTP silently disabled for the 12B).
@Test
func testDraftBlockAcceptsGemma4UnifiedTarget() throws {
    let unifiedJSON = """
        {
          "model_type": "gemma4_unified",
          "vocab_size": 32,
          "image_token_id": 31,
          "text_config": {
            "model_type": "gemma4_unified_text",
            "hidden_size": 8,
            "num_hidden_layers": 2,
            "intermediate_size": 16,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "num_global_key_value_heads": 1,
            "head_dim": 8,
            "global_head_dim": 8,
            "vocab_size": 32,
            "num_kv_shared_layers": 0,
            "hidden_size_per_layer_input": 0,
            "sliding_window": 8,
            "attention_k_eq_v": true,
            "layer_types": ["sliding_attention", "full_attention"],
            "rope_parameters": {},
            "tie_word_embeddings": true
          },
          "vision_config": null,
          "audio_config": null
        }
        """
    let targetConfig = try JSONDecoder.json5().decode(
        Gemma4UnifiedConfiguration.self, from: Data(unifiedJSON.utf8))
    let target = Gemma4Unified(targetConfig)

    // Prime drafter state through the MTP entry point (what the iterator does).
    let cache = target.newCache(parameters: nil)
    var emitState = LMOutput.State()
    emitState[mtpEmitFlagKey] = true
    let tokens = MLXArray((0 ..< 8).map { Int32($0 % 32) }).reshaped([1, 8])
    let out = target(LMInput.Text(tokens: tokens), cache: cache, state: emitState)
    eval(out.logits)

    #expect(out.state != nil, "Gemma4Unified must emit drafter state when the flag is set")
    guard let state = out.state,
        let lastHidden = state[mtpLastHiddenStatesKey],
        let sharedKV = state[mtpSharedKVStatesKey]
    else {
        Issue.record("missing drafter state keys on Gemma4Unified LMOutput")
        return
    }
    #expect(Set(sharedKV.keys) == ["full_attention", "sliding_attention"])

    // Tiny drafter with geometry matching the target's KV (cross-attention
    // consumes the target's sharedKV pool) and backbone hidden size.
    let drafterJSON = """
        {
          "model_type": "gemma4_unified_assistant",
          "backbone_hidden_size": 8,
          "tie_word_embeddings": true,
          "use_ordered_embeddings": false,
          "num_centroids": 2,
          "centroid_intermediate_top_k": 1,
          "text_config": {
            "model_type": "gemma4_unified_text",
            "hidden_size": 8,
            "num_hidden_layers": 2,
            "intermediate_size": 16,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "num_global_key_value_heads": 1,
            "head_dim": 8,
            "global_head_dim": 8,
            "vocab_size": 32,
            "num_kv_shared_layers": 2,
            "hidden_size_per_layer_input": 0,
            "sliding_window": 8,
            "attention_k_eq_v": true,
            "layer_types": ["sliding_attention", "full_attention"],
            "rope_parameters": {},
            "tie_word_embeddings": true
          }
        }
        """
    let drafterConfig = try JSONDecoder().decode(
        Gemma4AssistantConfiguration.self, from: Data(drafterJSON.utf8))
    let drafter = Gemma4AssistantDraftModel(drafterConfig)

    let lastHiddenSlice = lastHidden[0..., (-1)..., 0...]
    let proposed = drafter.draftBlock(
        target: target,
        lastToken: MLXArray([Int32(3)]),
        lastHidden: lastHiddenSlice,
        sharedKV: sharedKV,
        queryOffset: cache.first?.offset ?? 8,
        blockSize: 3,
        sampler: ArgMaxSampler()
    )
    eval(proposed)
    #expect(proposed.shape == [1, 2])
}
