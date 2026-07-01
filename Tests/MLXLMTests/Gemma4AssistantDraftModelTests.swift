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

// MARK: - MaskedEmbedder forward (use_ordered_embeddings)

/// Independent correctness pin for the centroid-routed sparse LM head.
///
/// With an identity `token_ordering` (cluster `c` owns tokens `[c·vsc, …)`), the
/// masked head must, at every *selected* vocab position, reproduce the plain dense
/// logit `hidden · embᵀ`; exactly `topK·vsc` positions are selected and every other
/// position must carry `mask_value`. The dense reference is computed independently
/// of the embedder, so a wrong gather/scatter/matmul fails this — no checkpoint or
/// Python fixture required.
@Test
func testMaskedEmbedderSelectedLogitsMatchDense() throws {
    let (h, v, c, k) = (4, 8, 4, 2)  // vsc = v/c = 2, N = k·vsc = 4
    let vsc = v / c
    let n = k * vsc

    let emb = Gemma4AssistantMaskedEmbedder(config: orderedSyntheticConfig())
    // Identity ordering so canonical token IDs equal vocab positions.
    try emb.update(
        parameters: ModuleParameters.unflattened([
            "token_ordering": MLXArray((0 ..< v).map { Int32($0) })
        ]), verify: [])

    let hidden = MLXArray([0.5, -1.0, 2.0, 0.25] as [Float], [1, 1, h])
    let lmHead = (MLXArray(0 ..< (v * h)).asType(.float32) * 0.1 - 1.0).reshaped([v, h])

    let masked = emb(hidden, tiedEmbedding: Embedding(weight: lmHead)).reshaped([v])
    let dense = matmul(hidden, lmHead.swappedAxes(-1, -2)).reshaped([v])

    let maskValue = masked.min()
    let selected = masked .!= maskValue
    // Exactly N positions survive the centroid pruning.
    #expect(selected.sum().item(Int.self) == n)
    // Every surviving position equals its dense logit.
    let agree = MLX.where(selected, abs(masked - dense) .< 1e-4, MLXArray(true))
    #expect(agree.all().item(Bool.self))
}

/// Regression pin for #383: the tied embedding of a *quantized* E-series drafter
/// is a `QuantizedEmbedding` whose `.weight` is packed `[vocab, H·bits/32]` — the
/// old forward fed that raw weight into a gather/reshape-to-H and crashed. Passing
/// the module (not `.weight`) gathers through its dequantizing forward. Reference
/// is the dense logit against the *dequantized* rows, so selected positions must
/// match exactly (identical dequantized rows, identical matmul).
@Test
func testMaskedEmbedderQuantizedTiedEmbedding() throws {
    // Hidden must be a multiple of the quant group size (64).
    let (h, v, c, k) = (64, 128, 8, 2)  // vsc = v/c = 16, N = k·vsc = 32
    let vsc = v / c
    let n = k * vsc

    let emb = Gemma4AssistantMaskedEmbedder(
        config: orderedSyntheticConfig(hidden: h, vocab: v, centroids: c, topK: k))
    try emb.update(
        parameters: ModuleParameters.unflattened([
            "token_ordering": MLXArray((0 ..< v).map { Int32($0) })
        ]), verify: [])

    let hidden = (MLXArray(0 ..< h).asType(.float32) * 0.01 - 0.3).reshaped([1, 1, h])
    let lmHead = (MLXArray(0 ..< (v * h)).asType(.float32) * 0.001 - 0.05).reshaped([v, h])
    let quantEmb = QuantizedEmbedding(weight: lmHead, groupSize: 64, bits: 4)

    let masked = emb(hidden, tiedEmbedding: quantEmb).reshaped([v])
    // Dense reference over dequantized rows (the head must reproduce these).
    let dequant = quantEmb(MLXArray(0 ..< v))  // [v, h]
    let dense = matmul(hidden, dequant.swappedAxes(-1, -2)).reshaped([v])

    let maskValue = masked.min()
    let selected = masked .!= maskValue
    #expect(selected.sum().item(Int.self) == n)
    let agree = MLX.where(selected, abs(masked - dense) .< 1e-3, MLXArray(true))
    #expect(agree.all().item(Bool.self))
}

// MARK: - Helpers

/// Tiny `use_ordered_embeddings=true` config. Defaults (hidden 4, vocab 8, 4
/// centroids, top-K 2) suit the dense pin; the quantized pin bumps hidden to a
/// multiple of the quantization group size.
private func orderedSyntheticConfig(
    hidden: Int = 4, vocab: Int = 8, centroids: Int = 4, topK: Int = 2
) -> Gemma4AssistantConfiguration {
    let textJSON =
        """
        {
          "model_type": "gemma4_text", "hidden_size": \(hidden), "num_hidden_layers": 1,
          "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 2,
          "global_head_dim": 2, "vocab_size": \(vocab), "num_kv_shared_layers": 0,
          "hidden_size_per_layer_input": 0, "sliding_window": 4, "sliding_window_pattern": 1,
          "max_position_embeddings": 16, "rms_norm_eps": 1e-6, "rope_traditional": false,
          "use_double_wide_mlp": false, "enable_moe_block": false, "attention_k_eq_v": true,
          "intermediate_size": 8, "layer_types": ["full_attention"], "rope_parameters": {},
          "tie_word_embeddings": true
        }
        """
    let json =
        """
        {
          "model_type": "gemma4_assistant", "backbone_hidden_size": \(hidden),
          "tie_word_embeddings": true, "use_ordered_embeddings": true,
          "num_centroids": \(centroids), "centroid_intermediate_top_k": \(topK),
          "text_config": \(textJSON)
        }
        """
    return try! JSONDecoder().decode(Gemma4AssistantConfiguration.self, from: Data(json.utf8))
}

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
