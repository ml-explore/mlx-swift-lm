// Copyright © 2026

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import XCTest

public class Gemma4AssistantTests: XCTestCase {

    // MARK: - Configuration decoding

    func testConfigurationDecodingFromJSON() throws {
        let json = """
            {
                "model_type": "gemma4_assistant",
                "backbone_hidden_size": 1536,
                "use_ordered_embeddings": true,
                "num_centroids": 2048,
                "centroid_intermediate_top_k": 32,
                "tie_word_embeddings": true,
                "block_size": 4,
                "text_config": {
                    "model_type": "gemma4_text",
                    "hidden_size": 256,
                    "num_hidden_layers": 4,
                    "intermediate_size": 1024,
                    "num_attention_heads": 8,
                    "head_dim": 64,
                    "global_head_dim": 64,
                    "vocab_size": 262144,
                    "num_key_value_heads": 1,
                    "num_kv_shared_layers": 0,
                    "sliding_window": 512,
                    "sliding_window_pattern": 5,
                    "tie_word_embeddings": true,
                    "use_double_wide_mlp": false,
                    "hidden_size_per_layer_input": 0,
                    "rms_norm_eps": 1.0e-6
                }
            }
            """.data(using: .utf8)!

        let config = try JSONDecoder().decode(Gemma4AssistantConfiguration.self, from: json)
        XCTAssertEqual(config.modelType, "gemma4_assistant")
        XCTAssertEqual(config.backboneHiddenSize, 1536)
        XCTAssertTrue(config.useOrderedEmbeddings)
        XCTAssertEqual(config.numCentroids, 2048)
        XCTAssertEqual(config.centroidIntermediateTopK, 32)
        XCTAssertTrue(config.tieWordEmbeddings)
        XCTAssertEqual(config.blockSize, 4)
        // text_config decoded; concrete fields validated indirectly via model
        // construction in subsequent tests (sanitize / draft).
    }

    // MARK: - Sanitize

    func testSanitizeDropsTiedLMHead() throws {
        let config = try makeTinyConfig(useOrdered: false, tied: true)
        let drafter = Gemma4AssistantDraftModel(config)
        let weights: [String: MLXArray] = [
            "model.embed_tokens.weight": MLXArray.zeros([1024, 64]),
            "lm_head.weight": MLXArray.zeros([1024, 64]),
            "pre_projection.weight": MLXArray.zeros([64, 256]),
            "post_projection.weight": MLXArray.zeros([128, 64]),
        ]
        let sanitized = drafter.sanitize(weights: weights)
        XCTAssertNil(
            sanitized["lm_head.weight"],
            "lm_head.weight must be dropped when tie_word_embeddings is true")
        XCTAssertNotNil(sanitized["model.embed_tokens.weight"])
        XCTAssertNotNil(sanitized["pre_projection.weight"])
        XCTAssertNotNil(sanitized["post_projection.weight"])
    }

    func testSanitizeKeepsLMHeadWhenUntied() throws {
        let config = try makeTinyConfig(useOrdered: false, tied: false)
        let drafter = Gemma4AssistantDraftModel(config)
        let weights: [String: MLXArray] = [
            "model.embed_tokens.weight": MLXArray.zeros([1024, 64]),
            "lm_head.weight": MLXArray.zeros([1024, 64]),
        ]
        let sanitized = drafter.sanitize(weights: weights)
        XCTAssertNotNil(
            sanitized["lm_head.weight"],
            "lm_head.weight must survive when tie_word_embeddings is false")
    }

    func testSanitizeCastsTokenOrderingToInt32() throws {
        let config = try makeTinyConfig(useOrdered: true, tied: true)
        let drafter = Gemma4AssistantDraftModel(config)
        let int64Buffer = MLXArray.zeros([1024], dtype: .int64)
        let weights: [String: MLXArray] = [
            "masked_embedding.token_ordering": int64Buffer
        ]
        let sanitized = drafter.sanitize(weights: weights)
        XCTAssertEqual(
            sanitized["masked_embedding.token_ordering"]?.dtype, .int32,
            "token_ordering must be cast to int32 (loaded as int64 from HF)")
    }

    // MARK: - Helpers

    private func makeTinyConfig(useOrdered: Bool, tied: Bool) throws
        -> Gemma4AssistantConfiguration
    {
        let json = """
            {
                "model_type": "gemma4_assistant",
                "backbone_hidden_size": 128,
                "use_ordered_embeddings": \(useOrdered),
                "num_centroids": 32,
                "centroid_intermediate_top_k": 4,
                "tie_word_embeddings": \(tied),
                "block_size": 4,
                "text_config": {
                    "model_type": "gemma4_text",
                    "hidden_size": 64,
                    "num_hidden_layers": 2,
                    "intermediate_size": 256,
                    "num_attention_heads": 4,
                    "head_dim": 16,
                    "global_head_dim": 16,
                    "vocab_size": 1024,
                    "num_key_value_heads": 1,
                    "num_kv_shared_layers": 0,
                    "sliding_window": 16,
                    "sliding_window_pattern": 5,
                    "tie_word_embeddings": true,
                    "use_double_wide_mlp": false,
                    "hidden_size_per_layer_input": 0,
                    "rms_norm_eps": 1.0e-6
                }
            }
            """.data(using: .utf8)!
        return try JSONDecoder().decode(Gemma4AssistantConfiguration.self, from: json)
    }
}
