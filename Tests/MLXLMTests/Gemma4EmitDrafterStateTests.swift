// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXVLM

// MARK: - Emit-hook plumbing tests for Gemma4TextLanguageModel
//
// These tests exercise the new `emitDrafterState: Bool = false` parameter on
// `Gemma4TextLanguageModel.callAsFunction` (which threads into
// `Gemma4TextBackbone.callAsFunction`) and the resulting population of
// `mtpLastHiddenStatesKey` / `mtpSharedKVStatesKey` on `LMOutput.state`.
//
// Construction uses small synthetic `Gemma4TextConfiguration` JSON to avoid
// loading real Gemma 4 weights — random-initialized modules are sufficient
// for verifying state shapes and key presence.

@Test
func testGemma4TextEmitFalseReturnsNoStateBySynthetic() {
    let model = makeSyntheticGemma4TextLanguageModel(layerTypes: [
        "full_attention", "sliding_attention",
    ])
    let inputs = MLXArray((0 ..< 8).map { Int32($0) }).reshaped([1, 8])

    let out = model(inputs, cache: nil, emitDrafterState: false)

    #expect(out.state == nil)
    eval(out.logits)
    #expect(out.logits.shape == [1, 8, 10])
}

@Test
func testGemma4TextEmitTrueWithBothLayerTypesPopulatesStateBySynthetic() {
    let model = makeSyntheticGemma4TextLanguageModel(layerTypes: [
        "full_attention", "sliding_attention",
    ])
    let cache = model.newCache(parameters: nil)
    let inputs = MLXArray((0 ..< 8).map { Int32($0) }).reshaped([1, 8])

    let out = model(inputs, cache: cache, emitDrafterState: true)

    #expect(out.state != nil, "emit=true should populate LMOutput.state")
    guard let state = out.state else { return }
    #expect(state[mtpLastHiddenStatesKey] != nil)
    #expect(state[mtpSharedKVStatesKey] != nil)
    guard
        let lastHidden = state[mtpLastHiddenStatesKey],
        let sharedKV = state[mtpSharedKVStatesKey]
    else { return }
    eval(lastHidden)
    #expect(lastHidden.shape == [1, 8, 4])
    #expect(Set(sharedKV.keys) == ["full_attention", "sliding_attention"])
    let full = sharedKV["full_attention"]!
    let sliding = sharedKV["sliding_attention"]!
    eval(full.0, full.1, sliding.0, sliding.1)
    #expect(full.0.shape.count == 4)
    #expect(full.1.shape.count == 4)
    #expect(sliding.0.shape.count == 4)
    #expect(sliding.1.shape.count == 4)
}

@Test
func testGemma4TextEmitTrueWithMissingLayerTypeReturnsNilSharedKV() {
    let model = makeSyntheticGemma4TextLanguageModel(layerTypes: ["full_attention"])
    let cache = model.newCache(parameters: nil)
    let inputs = MLXArray((0 ..< 4).map { Int32($0) }).reshaped([1, 4])

    let out = model(inputs, cache: cache, emitDrafterState: true)

    // With only `full_attention` layers, the emit-walk cannot satisfy both
    // target layer types and the backbone returns sharedKV: nil — which the
    // language-model layer reads as "no state to emit". The whole state stays
    // nil so the iterator falls back to single-token generation.
    #expect(out.state == nil)
    eval(out.logits)
}

@Test
func testGemma4TextEmitDisabledIsBitIdenticalRegressionBySynthetic() {
    let model = makeSyntheticGemma4TextLanguageModel(layerTypes: [
        "full_attention", "sliding_attention",
    ])
    let inputs = MLXArray((0 ..< 6).map { Int32($0) }).reshaped([1, 6])

    let outDefault = model(inputs, cache: nil)
    let outExplicitFalse = model(inputs, cache: nil, emitDrafterState: false)

    eval(outDefault.logits, outExplicitFalse.logits)
    // Bit-identical: emit=false MUST take the same code path as the default.
    #expect(
        allClose(outDefault.logits, outExplicitFalse.logits, rtol: 0, atol: 0)
            .item(Bool.self),
        "emit=false path must be bit-identical to the default code path"
    )
}

// MARK: - Helpers

/// Build a small 2-layer Gemma4TextLanguageModel with random-initialized
/// weights. Sufficient for emit-hook plumbing tests; not a correctness
/// reference for the model itself.
private func makeSyntheticGemma4TextLanguageModel(
    layerTypes: [String]
) -> Gemma4TextLanguageModel {
    let layersJSON = layerTypes.map { "\"\($0)\"" }.joined(separator: ", ")
    let textJSON =
        """
        {
          "model_type": "gemma4_text",
          "hidden_size": 4,
          "num_hidden_layers": \(layerTypes.count),
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
          "layer_types": [\(layersJSON)],
          "rope_parameters": {},
          "tie_word_embeddings": true
        }
        """
    let cfg = try! JSONDecoder().decode(
        Gemma4TextConfiguration.self, from: Data(textJSON.utf8))
    return Gemma4TextLanguageModel(cfg)
}
