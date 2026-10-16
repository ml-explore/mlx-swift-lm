// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXVLM

// `lastPositionOnly` (used by the multimodal prompt step) must give exactly the last row of the full logits.

@Test
func testGemma4LastPositionOnlyMatchesLastRowOfFullLogits() {
    let model = makeLastPositionTestModel(layerTypes: [
        "full_attention", "sliding_attention",
    ])
    let inputs = MLXArray((0 ..< 8).map { Int32($0) }).reshaped([1, 8])

    let full = model(inputs, cache: nil).logits
    let last = model(inputs, cache: nil, lastPositionOnly: true).logits
    eval(full, last)

    #expect(full.shape == [1, 8, 10])
    #expect(last.shape == [1, 1, 10])
    #expect(allClose(last, full[0..., 7..., 0...]).item(Bool.self))
}

@Test
func testGemma4LastPositionOnlyWithOnePositionIsUnchanged() {
    let model = makeLastPositionTestModel(layerTypes: ["full_attention"])
    let inputs = MLXArray([Int32(3)]).reshaped([1, 1])

    let full = model(inputs, cache: nil).logits
    let last = model(inputs, cache: nil, lastPositionOnly: true).logits
    eval(full, last)

    #expect(last.shape == [1, 1, 10])
    #expect(allClose(last, full).item(Bool.self))
}

// MARK: - Helpers

/// A small Gemma4TextLanguageModel with random-initialized weights (as in Gemma4EmitDrafterStateTests).
private func makeLastPositionTestModel(
    layerTypes: [String],
    numKvSharedLayers: Int = 0
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
          "num_kv_shared_layers": \(numKvSharedLayers),
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
