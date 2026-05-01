// Copyright © 2026 Apple Inc.
//
// Verifies the #231 fix: Gemma 4 Text attention with `attention_k_eq_v=true`
// must produce keys and values with matching shapes. Pre-fix the code took
// `v = k` AFTER `k_norm` + transpose + RoPE had been applied to k, then ran
// v through its own `vNorm` + transpose, so values ended up in
// `(B, L, H, D)` while keys were in `(B, H, L, D)` — `scaledDotProductAttention`
// crashes with `broadcast_shapes (B, H, L, D) vs (B, L, H, D)`.

import Foundation
import MLX
@testable import MLXLLM
import MLXLMCommon
import XCTest

final class Gemma4KeqVTests: XCTestCase {

    /// Build a tiny Gemma 4 Text config (1 layer, hidden_size=64) with
    /// `attention_k_eq_v: true`. Defaults handle every other field.
    private func makeConfig() throws -> Gemma4TextConfiguration {
        let json = """
            {
                "model_type": "gemma4_text",
                "hidden_size": 64,
                "num_hidden_layers": 1,
                "intermediate_size": 128,
                "num_attention_heads": 4,
                "head_dim": 16,
                "global_head_dim": 32,
                "num_key_value_heads": 2,
                "vocab_size": 64,
                "vocab_size_per_layer_input": 64,
                "hidden_size_per_layer_input": 16,
                "num_kv_shared_layers": 0,
                "sliding_window": 16,
                "sliding_window_pattern": 2,
                "max_position_embeddings": 128,
                "tie_word_embeddings": true,
                "use_double_wide_mlp": true,
                "attention_k_eq_v": true,
                "layer_types": ["full_attention"]
            }
            """
        return try JSONDecoder().decode(
            Gemma4TextConfiguration.self, from: Data(json.utf8))
    }

    /// The bug manifested as a `broadcast_shapes` fatal during the very first
    /// forward pass when `attention_k_eq_v=true`. A successful forward + a
    /// non-NaN logits output is sufficient evidence the k / v shapes now
    /// agree.
    func testForwardWithKeqV() throws {
        let config = try makeConfig()
        let model = Gemma4TextModel(config)

        // Single-token batch decode — this is the path that crashed.
        let input = MLXArray([Int32(1)], [1, 1])
        let logits = model(input, cache: nil)

        XCTAssertEqual(logits.dim(0), 1)
        XCTAssertEqual(logits.dim(1), 1)
        XCTAssertEqual(logits.dim(2), config.vocabSize)
        // Sanity: forward returned without crashing — k/v shapes are now
        // consistent under `attention_k_eq_v=true`.
        XCTAssertEqual(logits.ndim, 3)
    }

    /// Same forward at sequence length > 1 (prefill path). The shape mismatch
    /// also fired here.
    func testPrefillWithKeqV() throws {
        let config = try makeConfig()
        let model = Gemma4TextModel(config)

        let input = MLXArray([Int32](1...8), [1, 8])
        let logits = model(input, cache: nil)

        XCTAssertEqual(logits.dim(1), 8)
        XCTAssertEqual(logits.dim(2), config.vocabSize)
    }
}
