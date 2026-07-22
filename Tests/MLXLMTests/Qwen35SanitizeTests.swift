// Copyright © 2026 Apple Inc.
//
// Verifies the #143 fix: Qwen35 VLM sanitize must remap bare `model.*`
// weight keys to `language_model.model.*`. Pre-fix the keys fell through
// unchanged and `language_model` lost its tensors at load time.

import Foundation
import MLX
import MLXLMCommon
import MLXVLM
import XCTest

final class Qwen35SanitizeTests: XCTestCase {

    private func makeMinimalConfig() throws -> Qwen35Configuration {
        // Minimum-viable config with small dims so module init stays cheap.
        // Only the fields without defaults need values; everything else
        // falls back to the public defaults (which we never exercise in the
        // sanitize-only test).
        let json = """
            {
                "model_type": "qwen3_5_moe_vl",
                "text_config": {
                    "hidden_size": 8,
                    "num_hidden_layers": 1,
                    "intermediate_size": 16,
                    "num_attention_heads": 1,
                    "num_key_value_heads": 1,
                    "linear_num_value_heads": 1,
                    "linear_num_key_heads": 1,
                    "linear_key_head_dim": 8,
                    "linear_value_head_dim": 8,
                    "linear_conv_kernel_dim": 4,
                    "vocab_size": 32,
                    "full_attention_interval": 2,
                    "num_experts": 0,
                    "num_experts_per_tok": 0
                },
                "vision_config": {
                    "model_type": "qwen3_5_moe_vl",
                    "depth": 1,
                    "hidden_size": 8,
                    "intermediate_size": 16,
                    "out_hidden_size": 8,
                    "num_heads": 1,
                    "patch_size": 16,
                    "spatial_merge_size": 1,
                    "temporal_patch_size": 1,
                    "num_position_embeddings": 8
                }
            }
            """
        return try JSONDecoder().decode(
            Qwen35Configuration.self, from: Data(json.utf8))
    }

    /// Pre-fix, weights with `model.*` paths (no `language_model` prefix and
    /// no `visual` prefix) fell through `sanitize` unchanged, so the
    /// `language_model` submodule received no weights and load failed with
    /// `keyNotFound`.
    func testBareModelKeysAreRemapped() throws {
        let config = try makeMinimalConfig()
        let model = Qwen35(config)

        // 2D dummy avoids tripping vision-side sanitize transposes for any
        // weights that flow through the Qwen3VL vision model's sanitize at
        // the end of Qwen35.sanitize — only language-side keys are tested.
        let dummy = MLXArray.zeros([1, 1])
        let weights: [String: MLXArray] = [
            // Path that issue #143 calls out: bare `model.layers.*`.
            "model.layers.0.mlp.up_proj.weight": dummy,
            "model.layers.0.self_attn.q_proj.weight": dummy,
            "model.embed_tokens.weight": dummy,
            "model.norm.weight": dummy,
            // Already-namespaced path — verify the existing rename branch
            // still fires.
            "model.language_model.layers.0.mlp.up_proj.weight": dummy,
            // Top-level path the existing logic remaps.
            "lm_head.weight": dummy,
        ]

        let sanitized = model.sanitize(weights: weights)

        // Bare `model.*` keys are now under `language_model.model.*`.
        XCTAssertNotNil(
            sanitized["language_model.model.layers.0.mlp.up_proj.weight"],
            "bare `model.layers.0.mlp.up_proj.weight` must be remapped")
        XCTAssertNotNil(
            sanitized["language_model.model.layers.0.self_attn.q_proj.weight"])
        XCTAssertNotNil(sanitized["language_model.model.embed_tokens.weight"])
        XCTAssertNotNil(sanitized["language_model.model.norm.weight"])

        // The `lm_head` rename branch is preserved.
        XCTAssertNotNil(sanitized["language_model.lm_head.weight"])

        // None of the bare `model.*` keys remain in the sanitized dict.
        for key in sanitized.keys {
            XCTAssertFalse(
                key.hasPrefix("model.layers.")
                    || key == "model.embed_tokens.weight"
                    || key == "model.norm.weight",
                "bare model.* key leaked through sanitize: \(key)")
        }
    }

    /// A pre-converted MLX checkpoint (conv1d already sanitized, trailing dim
    /// == 1, no MTP tensors) already stores RMSNorm weights in the shifted
    /// convention. Sanitize must NOT add the `+1` again: double-shifting every
    /// layernorm decoheres the whole model and produces garbage tokens. This is
    /// the VLM/LLM sanitize divergence — the LLM path gated the shift, the VLM
    /// path did not.
    func testPreConvertedNormWeightsAreNotShifted() throws {
        let config = try makeMinimalConfig()
        let model = Qwen35(config)

        let weights: [String: MLXArray] = [
            // Sanitized conv1d (trailing dim == 1) marks an already-converted
            // checkpoint, so the norm shift must be suppressed.
            "language_model.model.layers.0.linear_attn.conv1d.weight": MLXArray.zeros([8, 4, 1]),
            "language_model.model.norm.weight": MLXArray.zeros([8]),
        ]

        let sanitized = model.sanitize(weights: weights)

        let norm = try XCTUnwrap(sanitized["language_model.model.norm.weight"])
        XCTAssertEqual(
            norm.sum().item(Float.self), 0.0, accuracy: 1e-6,
            "pre-converted norm weight must not be +1 shifted (double-shift => garbage)")
    }

    /// A raw HF checkpoint (unsanitized conv1d, trailing dim != 1) stores
    /// RMSNorm weights un-shifted, so sanitize must add the `+1`.
    func testRawCheckpointNormWeightsAreShifted() throws {
        let config = try makeMinimalConfig()
        let model = Qwen35(config)

        let weights: [String: MLXArray] = [
            // Unsanitized conv1d (trailing dim != 1) marks a raw checkpoint.
            "language_model.model.layers.0.linear_attn.conv1d.weight": MLXArray.zeros([8, 1, 4]),
            "language_model.model.norm.weight": MLXArray.zeros([8]),
        ]

        let sanitized = model.sanitize(weights: weights)

        let norm = try XCTUnwrap(sanitized["language_model.model.norm.weight"])
        XCTAssertEqual(
            norm.sum().item(Float.self), 8.0, accuracy: 1e-6,
            "raw-checkpoint norm weight must be +1 shifted (8 zeros => sum 8)")
    }
}
