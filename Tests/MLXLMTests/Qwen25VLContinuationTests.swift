// Copyright © 2026 Apple Inc.
//
// Equivalence tests for Qwen2.5-VL and Qwen2-VL windowed prefill and warm
// (cached-prefix) continuation, on tiny random-weight models so they run in CI
// without downloads. The invariant under test mirrors Qwen35ContinuationTests:
// however a prompt reaches the KV cache — one shot, windowed chunks, or split
// across a warm continuation — the next-token logits must match, because M-RoPE
// positions must be anchored at the cache offset (plus the carried rope delta),
// never restarted at zero.

import Foundation
import MLX
import MLXLMCommon
import MLXVLM
import XCTest

final class Qwen25VLContinuationTests: XCTestCase {

    // MARK: - Tiny models

    private func makeTinyQwen25VL() throws -> Qwen25VL {
        let json = """
            {
                "model_type": "qwen2_5_vl",
                "hidden_size": 64,
                "num_hidden_layers": 2,
                "intermediate_size": 128,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "rms_norm_eps": 1e-6,
                "vocab_size": 512,
                "max_position_embeddings": 4096,
                "rope_theta": 1000000.0,
                "rope_traditional": false,
                "tie_word_embeddings": false,
                "sliding_window": 32768,
                "use_sliding_window": false,
                "max_window_layers": 2,
                "image_token_id": 500,
                "video_token_id": 501,
                "vision_start_token_id": 502,
                "vision_end_token_id": 503,
                "vision_token_id": 504,
                "rope_scaling": {
                    "type": "mrope",
                    "mrope_section": [2, 3, 3]
                },
                "vision_config": {
                    "depth": 2,
                    "hidden_size": 32,
                    "intermediate_size": 64,
                    "out_hidden_size": 64,
                    "num_heads": 2,
                    "patch_size": 16,
                    "spatial_patch_size": 16,
                    "spatial_merge_size": 2,
                    "temporal_patch_size": 2,
                    "window_size": 64,
                    "fullatt_block_indexes": [1],
                    "tokens_per_second": 2,
                    "in_chans": 3
                }
            }
            """
        let config = try JSONDecoder().decode(
            Qwen25VLConfiguration.self, from: Data(json.utf8))
        return Qwen25VL(config)
    }

    private func makeTinyQwen2VL() throws -> Qwen2VL {
        let json = """
            {
                "model_type": "qwen2_vl",
                "hidden_size": 64,
                "num_hidden_layers": 2,
                "intermediate_size": 128,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "rms_norm_eps": 1e-6,
                "vocab_size": 512,
                "max_position_embeddings": 4096,
                "rope_theta": 1000000.0,
                "rope_traditional": false,
                "tie_word_embeddings": false,
                "image_token_id": 500,
                "video_token_id": 501,
                "rope_scaling": {
                    "type": "mrope",
                    "mrope_section": [2, 3, 3]
                },
                "vision_config": {
                    "depth": 2,
                    "embed_dim": 32,
                    "hidden_size": 64,
                    "num_heads": 2,
                    "patch_size": 16,
                    "mlp_ratio": 2.0,
                    "spatial_patch_size": 16,
                    "spatial_merge_size": 2,
                    "temporal_patch_size": 2,
                    "in_channels": 3
                }
            }
            """
        let config = try JSONDecoder().decode(
            Qwen2VLConfiguration.self, from: Data(json.utf8))
        return Qwen2VL(config)
    }

    // MARK: - Shared assertions

    /// The Qwen2 family and Qwen2.5-VL share the tiny-model token ids.
    private let continuation = ContinuationAssertions(
        imageTokenId: 500, visionStartTokenId: 502)

    // MARK: - Qwen2.5-VL

    func testQwen25VLWarmTextContinuationMatchesFullPrefill() throws {
        try continuation.assertWarmTextContinuation(makeTinyQwen25VL())
    }

    func testQwen25VLWarmImageContinuationMatchesFullPrefill() throws {
        try continuation.assertWarmImageContinuation(makeTinyQwen25VL())
    }

    func testQwen25VLImageMidContinuationResumeState() throws {
        try continuation.assertImageMidContinuationResumeState(makeTinyQwen25VL())
    }

    func testQwen25VLWindowedPrefillMatchesSingleShot() throws {
        try continuation.assertWindowedTextPrefill(makeTinyQwen25VL())
    }

    func testQwen25VLWindowedImagePrefillMatchesSingleShot() throws {
        try continuation.assertWindowedImagePrefill(makeTinyQwen25VL())
    }

    // MARK: - Qwen2-VL

    func testQwen2VLWarmTextContinuationMatchesFullPrefill() throws {
        try continuation.assertWarmTextContinuation(makeTinyQwen2VL())
    }

    func testQwen2VLWarmImageContinuationMatchesFullPrefill() throws {
        try continuation.assertWarmImageContinuation(makeTinyQwen2VL())
    }

    func testQwen2VLImageMidContinuationResumeState() throws {
        try continuation.assertImageMidContinuationResumeState(makeTinyQwen2VL())
    }

    func testQwen2VLWindowedPrefillMatchesSingleShot() throws {
        try continuation.assertWindowedTextPrefill(makeTinyQwen2VL())
    }

    func testQwen2VLWindowedImagePrefillMatchesSingleShot() throws {
        try continuation.assertWindowedImagePrefill(makeTinyQwen2VL())
    }
}
