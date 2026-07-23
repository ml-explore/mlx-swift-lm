// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import XCTest

@_spi(Testing) @testable import MLXVLM

final class DeepseekOCRVisionTests: XCTestCase {

    func testSAMEncoderMatchesExpectedFeatureShape() throws {
        let model = try makeModel()
        let pixels = zeros([1, 1024, 1024, 3], type: Float.self)

        let features = model.samFeaturesForTesting(pixels)

        XCTAssertEqual(features.shape, [1, 16, 16, 1024])
    }

    func testClipFusionProducesExpectedPreProjectorShape() throws {
        let model = try makeModel()
        let pixels = zeros([1, 1024, 1024, 3], type: Float.self)

        let features = model.fusedVisionFeaturesForTesting(pixels)

        XCTAssertEqual(features.shape, [1, 256, 2048])
    }

    func testProjectorOutputsLanguageHiddenSize() throws {
        let model = try makeModel()
        let pixels = zeros([1, 1024, 1024, 3], type: Float.self)

        let features = model.projectedImageFeaturesForTesting(pixels)

        XCTAssertEqual(features.shape, [1, 256, 1280])
    }

    func testSanitizeRemapsSAMWeightsAndPreservesShapes() throws {
        let model = try makeModel()
        let weights: [String: MLXArray] = [
            "model.sam_model.patch_embed.proj.weight": zeros([768, 3, 16, 16], type: Float.self),
            "model.sam_model.blocks.0.attn.qkv.weight": zeros([2304, 768], type: Float.self),
            "model.sam_model.neck.0.weight": zeros([256, 768, 1, 1], type: Float.self),
            "model.sam_model.neck.2.weight": zeros([256, 256, 3, 3], type: Float.self),
        ]

        let sanitized = model.sanitize(weights: weights)

        XCTAssertEqual(sanitized["sam_model.patch_embed.proj.weight"]?.shape, [768, 16, 16, 3])
        XCTAssertEqual(sanitized["sam_model.layers.0.attn.qkv.weight"]?.shape, [2304, 768])
        XCTAssertEqual(sanitized["sam_model.neck.conv1.weight"]?.shape, [256, 1, 1, 768])
        XCTAssertEqual(sanitized["sam_model.neck.conv2.weight"]?.shape, [256, 3, 3, 256])
    }

    func testSanitizeRemapsClipAndProjectorWeights() throws {
        let model = try makeModel()
        let weights: [String: MLXArray] = [
            "model.vision_model.embeddings.class_embedding": zeros([1024], type: Float.self),
            "model.vision_model.pre_layrnorm.weight": zeros([1024], type: Float.self),
            "model.projector.layers.weight": zeros([1280, 2048], type: Float.self),
            "model.projector.layers.bias": zeros([1280], type: Float.self),
        ]

        let sanitized = model.sanitize(weights: weights)

        XCTAssertEqual(sanitized["vision_model.embeddings.classEmbedding"]?.shape, [1024])
        XCTAssertEqual(sanitized["vision_model.pre_layrnorm.weight"]?.shape, [1024])
        XCTAssertEqual(sanitized["projector.layers.weight"]?.shape, [1280, 2048])
        XCTAssertEqual(sanitized["projector.layers.bias"]?.shape, [1280])
    }

    private func makeModel() throws -> DeepseekOCR {
        let config = try JSONDecoder().decode(
            DeepseekOCRConfiguration.self,
            from: Data(Self.configJSON.utf8))
        return DeepseekOCR(config)
    }

    private static let configJSON = #"""
        {
         "model_type": "deepseekocr",
         "vision_config": {
          "hidden_size": 768,
          "output_channels": 256,
          "num_hidden_layers": 12,
          "num_attention_heads": 12,
          "image_size": 1024,
          "patch_size": 16,
          "global_attn_indexes": [2, 5, 8, 11],
          "mlp_dim": 3072
         },
         "language_config": {
          "vocab_size": 129280,
          "hidden_size": 1280,
          "intermediate_size": 6848,
          "num_hidden_layers": 12,
          "num_attention_heads": 10,
          "num_key_value_heads": 10,
          "max_position_embeddings": 8192
         }
        }
        """#
}
