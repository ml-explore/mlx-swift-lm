// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import MLXVLM

/// A prompt whose image-token count disagrees with the patch count the vision tower
/// produced must raise an error the caller can catch.
struct VLMImageTokenMismatchTests {

    /// Tiny Mistral3. `image_token_index` is 10, and `spatial_merge_size` 1 keeps the
    /// patch count equal to the grid, so a test can predict it.
    private static func makeTinyMistral3() throws -> Mistral3VLM {
        let json = """
            {
                "model_type": "mistral3",
                "image_token_index": 10,
                "spatial_merge_size": 1,
                "vision_feature_layer": -1,
                "text_config": {
                    "model_type": "mistral",
                    "hidden_size": 32,
                    "num_hidden_layers": 2,
                    "intermediate_size": 64,
                    "num_attention_heads": 2,
                    "num_key_value_heads": 1,
                    "head_dim": 16,
                    "rms_norm_eps": 1e-5,
                    "vocab_size": 100,
                    "max_position_embeddings": 512,
                    "rope_parameters": { "rope_theta": 10000.0 }
                },
                "vision_config": {
                    "model_type": "pixtral",
                    "hidden_size": 32,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 2,
                    "intermediate_size": 64,
                    "patch_size": 8,
                    "image_size": 32
                }
            }
            """
        let config = try JSONDecoder().decode(
            Mistral3VLMConfiguration.self, from: Data(json.utf8))
        return withRandomState(MLXRandom.RandomState(seed: 42)) {
            Mistral3VLM(config)
        }
    }

    /// One 16x16 image against a `patch_size` of 8 gives a 2x2 grid, so the vision
    /// tower produces 4 patches. The prompt carries a single image token, which is the
    /// mismatch under test.
    @Test("Mistral3 raises rather than aborting when the counts disagree")
    func mistral3RaisesOnMismatch() throws {
        let model = try Self.makeTinyMistral3()
        let tokens = MLXArray([Int32(1), 10, 2]).expandedDimensions(axis: 0)
        let pixels = MLXRandom.normal([1, 3, 16, 16]).asType(.float32)
        let input = LMInput(
            text: .init(tokens: tokens),
            image: .init(pixels: pixels, frames: [.init(1, 16, 16)]))

        #expect(throws: VLMError.self) {
            _ = try model.prepare(
                input, cache: try model.newCache(parameters: nil), state: nil,
                prefill: .init(stepSize: 512))
        }
    }

    /// Tiny LFM2VL. `image_token_id` is 396 by default, so the prompt below uses it.
    private static func makeTinyLFM2VL() throws -> LFM2VL {
        let json = """
            {
                "model_type": "lfm2-vl",
                "downsample_factor": 1,
                "projector_hidden_size": 32,
                "projector_use_layernorm": false,
                "text_config": {
                    "model_type": "lfm2",
                    "hidden_size": 32,
                    "num_hidden_layers": 2,
                    "num_attention_heads": 2,
                    "num_key_value_heads": 1,
                    "vocab_size": 500
                },
                "vision_config": {
                    "model_type": "siglip2_navit",
                    "hidden_size": 32,
                    "intermediate_size": 64,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 2,
                    "patch_size": 8,
                    "image_size": 32
                }
            }
            """
        let config = try JSONDecoder().decode(
            LFM2VLConfiguration.self, from: Data(json.utf8))
        return withRandomState(MLXRandom.RandomState(seed: 42)) {
            LFM2VL(config)
        }
    }

    @Test("LFM2VL raises rather than aborting when the counts disagree")
    func lfm2vlRaisesOnMismatch() throws {
        let model = try Self.makeTinyLFM2VL()
        let tokens = MLXArray([Int32(1), 396, 2]).expandedDimensions(axis: 0)
        // A 2x2 patch grid, so the tower yields 4 features against the prompt's 1 token.
        let pixels = MLXRandom.normal([1, 4, 3 * 8 * 8]).asType(.float32)
        let input = LMInput(
            text: .init(tokens: tokens),
            image: .init(pixels: pixels, frames: [.init(1, 2, 2)]))

        #expect(throws: VLMError.self) {
            _ = try model.prepare(
                input, cache: try model.newCache(parameters: nil), state: nil,
                prefill: .init(stepSize: 512))
        }
    }
}
