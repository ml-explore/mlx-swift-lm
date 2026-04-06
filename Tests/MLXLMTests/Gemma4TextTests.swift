// Copyright © 2025 Apple Inc.

import Foundation
import MLX
import MLXNN
import XCTest

@testable import MLXLLM
@testable import MLXLMCommon

public class Gemma4TextTests: XCTestCase {

    /// Minimal Gemma 4 text config for unit testing (tiny dimensions).
    private func makeTestConfig() throws -> Gemma4TextConfiguration {
        let json = """
            {
                "model_type": "gemma4_text",
                "hidden_size": 64,
                "num_hidden_layers": 10,
                "intermediate_size": 128,
                "num_attention_heads": 4,
                "head_dim": 16,
                "global_head_dim": 32,
                "rms_norm_eps": 1e-6,
                "vocab_size": 100,
                "num_key_value_heads": 1,
                "sliding_window": 64,
                "sliding_window_pattern": 5,
                "max_position_embeddings": 256,
                "final_logit_softcapping": 30.0,
                "num_kv_shared_layers": 4,
                "use_double_wide_mlp": true,
                "tie_word_embeddings": true,
                "attention_k_eq_v": false,
                "enable_moe_block": false,
                "rope_parameters": {
                    "full_attention": {
                        "partial_rotary_factor": 0.25,
                        "rope_theta": 1000000.0,
                        "rope_type": "proportional"
                    },
                    "sliding_attention": {
                        "rope_theta": 10000.0,
                        "rope_type": "default"
                    }
                }
            }
            """
        return try JSONDecoder().decode(
            Gemma4TextConfiguration.self, from: json.data(using: .utf8)!)
    }

    func testConfigDecoding() throws {
        let config = try makeTestConfig()
        XCTAssertEqual(config.hiddenSize, 64)
        XCTAssertEqual(config.hiddenLayers, 10)
        XCTAssertEqual(config.numKvSharedLayers, 4)
        XCTAssertEqual(config.globalHeadDim, 32)
        XCTAssertTrue(config.useDoubleWideMlp)
        XCTAssertTrue(config.tieWordEmbeddings)
        XCTAssertFalse(config.attentionKEqV)
    }

    func testLayerTypeGeneration() throws {
        let config = try makeTestConfig()
        let types = config.resolvedLayerTypes
        XCTAssertEqual(types.count, 10)
        // Pattern: 4 sliding + 1 full, repeated
        XCTAssertEqual(types[0], "sliding_attention")
        XCTAssertEqual(types[4], "full_attention")
        XCTAssertEqual(types[9], "full_attention")
    }

    func testModelInstantiation() throws {
        let config = try makeTestConfig()
        let model = Gemma4TextModel(config)
        XCTAssertEqual(model.vocabularySize, 100)
    }

    func testCacheCreation() throws {
        let config = try makeTestConfig()
        let model = Gemma4TextModel(config)
        let cache = model.newCache(parameters: nil)
        // All 10 layers get caches
        XCTAssertEqual(cache.count, 10)
    }

    func testForwardPass() throws {
        let config = try makeTestConfig()
        let model = Gemma4TextModel(config)
        quantize(model: model, groupSize: 64, bits: 4)

        let input = MLXArray([1, 2, 3])[.newAxis]  // [1, 3]
        let cache = model.newCache(parameters: nil)
        let output = model(input, cache: cache)

        // Output shape: [1, 3, vocab_size]
        XCTAssertEqual(output.shape[0], 1)
        XCTAssertEqual(output.shape[1], 3)
        XCTAssertEqual(output.shape[2], 100)
    }

    func testVLMConfigNesting() throws {
        // Gemma 4 VLM models wrap text config under "text_config"
        let json = """
            {
                "model_type": "gemma4",
                "text_config": {
                    "model_type": "gemma4_text",
                    "hidden_size": 64,
                    "num_hidden_layers": 10,
                    "intermediate_size": 128,
                    "num_attention_heads": 4,
                    "head_dim": 16,
                    "global_head_dim": 32,
                    "vocab_size": 100,
                    "num_key_value_heads": 1,
                    "sliding_window": 64,
                    "sliding_window_pattern": 5,
                    "num_kv_shared_layers": 4,
                    "use_double_wide_mlp": true,
                    "tie_word_embeddings": true
                }
            }
            """
        let config = try JSONDecoder().decode(
            Gemma4TextConfiguration.self, from: json.data(using: .utf8)!)
        XCTAssertEqual(config.hiddenSize, 64)
        XCTAssertEqual(config.numKvSharedLayers, 4)
    }
}
