// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXVLM

// MARK: - Regression: KV-shared layers must not declare local K/V projections
//
// Gemma 4's tail layers (`layer_idx >= num_hidden_layers - num_kv_shared_layers`)
// reuse an earlier same-type layer's K/V and own no `k_proj` / `v_proj` / `k_norm`.
// `Gemma4TextLanguageModel.sanitize` drops those tensors from the checkpoint, and
// the runtime forward feeds the shared layers a `sharedKV` instead of computing
// their own. If the module tree still *constructs* those projections (the default
// `kvSharedOnly: false`), loading any real Gemma 4 checkpoint fails under the
// strict loader verify with e.g.
//
//   keyNotFound(path: ["language_model","model","layers","24","self_attn",
//                      "v_proj","weight"], modules: [... "Gemma4TextAttention","Linear"])
//
// (`mlx-community/gemma-4-e4b-it-4bit`: num_hidden_layers=42, num_kv_shared_layers=18
//  → the first shared layer is 24.) These tests pin the construction side to the
// sanitize side so the two can't drift apart again.

struct Gemma4KVSharedLoadTests {

    /// The module tree of a model with KV sharing must not contain
    /// `k_proj` / `v_proj` / `k_norm` for the shared tail layers, while the
    /// non-shared head layers must still own them.
    @Test("KV-shared layers own no local k_proj/v_proj/k_norm")
    func kvSharedLayersOwnNoLocalKVProjections() {
        // 4 layers, 2 KV-shared → layers 0,1 own K/V; layers 2,3 share it.
        let model = Gemma4TextLanguageModel(Self.textConfig(numKVSharedLayers: 2))
        let keys = Set(model.parameters().flattened().map(\.0))

        // Head (KV-owning) layers keep their projections.
        for owning in [0, 1] {
            #expect(keys.contains { $0.contains("layers.\(owning).self_attn.k_proj") })
            #expect(keys.contains { $0.contains("layers.\(owning).self_attn.v_proj") })
        }
        // Tail (KV-shared) layers must own none of them.
        for shared in [2, 3] {
            #expect(!keys.contains { $0.contains("layers.\(shared).self_attn.k_proj") })
            #expect(!keys.contains { $0.contains("layers.\(shared).self_attn.v_proj") })
            #expect(!keys.contains { $0.contains("layers.\(shared).self_attn.k_norm") })
        }
    }

    /// End-to-end reproduction of the production load failure: build a
    /// checkpoint that carries K/V projections on every layer (a dense export),
    /// sanitize it the way real Gemma 4 checkpoints are sanitized (drop the
    /// KV-shared tail's `k_proj`/`v_proj`/`k_norm`), then load it into the
    /// KV-sharing model through the exact call the production loader uses.
    /// Before the fix this threw `keyNotFound(… self_attn.v_proj.weight …)`.
    @Test("Sanitized Gemma 4 checkpoint loads into a KV-sharing model")
    func sanitizedCheckpointLoadsWithoutKeyNotFound() throws {
        // A "dense" sibling (no KV sharing) supplies weights for every layer,
        // independent of how the KV-sharing model constructs its own tree.
        let dense = Gemma4TextLanguageModel(Self.textConfig(numKVSharedLayers: 0))
        eval(dense)

        let firstKVSharedLayer = 4 - 2
        var checkpoint = [String: MLXArray]()
        for (key, value) in dense.parameters().flattened() {
            if let layerIdx = Self.layerIndex(in: key), layerIdx >= firstKVSharedLayer,
                key.contains(".self_attn.k_proj")
                    || key.contains(".self_attn.v_proj")
                    || key.contains(".self_attn.k_norm")
            {
                continue  // dropped by sanitize for KV-shared layers
            }
            checkpoint[key] = value
        }

        let model = Gemma4TextLanguageModel(Self.textConfig(numKVSharedLayers: 2))
        // `[.all]` is what `MLXLMCommon.loadWeights` applies — it throws on any
        // module parameter that has no matching weight (the original bug).
        try model.update(
            parameters: ModuleParameters.unflattened(checkpoint), verify: [.all])
        eval(model)
    }

    // MARK: - Helpers

    private static func layerIndex(in key: String) -> Int? {
        guard let range = key.range(of: "layers.") else { return nil }
        return Int(key[range.upperBound...].prefix { $0.isNumber })
    }

    /// Tiny all-`full_attention` text config. `use_double_wide_mlp` is off so the
    /// dense and KV-sharing models differ *only* in the shared layers' K/V tensors,
    /// keeping every other weight shape-compatible across the two.
    private static func textConfig(numKVSharedLayers: Int) -> Gemma4TextConfiguration {
        let json = """
            {
              "model_type": "gemma4_text",
              "hidden_size": 8,
              "num_hidden_layers": 4,
              "intermediate_size": 16,
              "num_attention_heads": 2,
              "num_key_value_heads": 1,
              "head_dim": 4,
              "global_head_dim": 4,
              "vocab_size": 12,
              "vocab_size_per_layer_input": 12,
              "num_kv_shared_layers": \(numKVSharedLayers),
              "hidden_size_per_layer_input": 0,
              "sliding_window": 8,
              "sliding_window_pattern": 1,
              "max_position_embeddings": 32,
              "rms_norm_eps": 1e-6,
              "rope_traditional": false,
              "use_double_wide_mlp": false,
              "enable_moe_block": false,
              "attention_k_eq_v": false,
              "layer_types": [
                "full_attention", "full_attention", "full_attention", "full_attention"
              ],
              "rope_parameters": {},
              "tie_word_embeddings": true
            }
            """
        return try! JSONDecoder().decode(
            Gemma4TextConfiguration.self, from: Data(json.utf8))
    }
}
