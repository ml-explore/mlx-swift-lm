// Copyright © 2026 Apple Inc.
//
// Continuation equivalences for Qwen3-VL-MoE, on a tiny random-weight model so
// they run without downloads. The model resumes a warm cache from the rope delta
// it carries, in the same offset-relative frame as Qwen3-VL, so a cache trimmed
// back to a shorter prefix must resume from the rewound delta.

import Foundation
import MLX
import MLXLMCommon
import MLXVLM
import XCTest

final class Qwen3VLMoEContinuationTests: XCTestCase {

    private let continuation = ContinuationAssertions(
        imageTokenId: 500, visionStartTokenId: 502)

    private func makeTinyModel() throws -> Qwen3VLMoE {
        let json = """
            {
                "model_type": "qwen3_vl_moe",
                "image_token_id": 500,
                "video_token_id": 501,
                "vision_start_token_id": 502,
                "vision_end_token_id": 503,
                "vocab_size": 512,
                "text_config": {
                    "model_type": "qwen3_vl_moe",
                    "hidden_size": 64,
                    "num_hidden_layers": 2,
                    "intermediate_size": 128,
                    "num_attention_heads": 4,
                    "num_key_value_heads": 2,
                    "head_dim": 16,
                    "num_experts": 4,
                    "num_experts_per_tok": 2,
                    "moe_intermediate_size": 32,
                    "vocab_size": 512,
                    "max_position_embeddings": 4096,
                    "rms_norm_eps": 1e-6,
                    "rope_theta": 100000.0,
                    "rope_scaling": {
                        "type": "default",
                        "mrope_section": [4, 2, 2]
                    }
                },
                "vision_config": {
                    "model_type": "qwen3_vl_moe",
                    "depth": 2,
                    "hidden_size": 32,
                    "intermediate_size": 64,
                    "out_hidden_size": 64,
                    "num_heads": 2,
                    "patch_size": 16,
                    "spatial_merge_size": 2,
                    "temporal_patch_size": 2,
                    "num_position_embeddings": 64,
                    "deepstack_visual_indexes": [0, 1]
                }
            }
            """
        let config = try JSONDecoder().decode(
            Qwen3VLMoEConfiguration.self, from: Data(json.utf8))
        return withRandomState(MLXRandom.RandomState(seed: 1)) { Qwen3VLMoE(config) }
    }

    func testWarmTextContinuationMatchesFullPrefill() throws {
        try continuation.assertWarmTextContinuation(try makeTinyModel())
    }

    /// A cache trimmed back past an image must resume from the delta of the prefix it kept.
    func testRewoundStateContinuationMatchesFullPrefill() throws {
        try continuation.assertRewoundStateContinuation(try makeTinyModel())
    }

    /// A cache trimmed back past text, keeping its image, must keep the delta it carried.
    func testRewoundStateKeepsThePrefixMediaDelta() throws {
        try continuation.assertRewoundStateKeepsThePrefixMediaDelta(try makeTinyModel())
    }
}
