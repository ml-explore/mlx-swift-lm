// Copyright © 2025 Apple Inc.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import XCTest

public class Gemma4Tests: XCTestCase {

    /// Test that Gemma4TextModel can forward-pass with a minimal configuration
    func testGemma4Eval() throws {
        // Minimal config matching E4B structure but tiny
        let configJSON = """
            {
                "model_type": "gemma4_text",
                "hidden_size": 64,
                "num_hidden_layers": 6,
                "intermediate_size": 128,
                "num_attention_heads": 2,
                "head_dim": 32,
                "global_head_dim": 64,
                "rms_norm_eps": 1e-6,
                "vocab_size": 100,
                "num_key_value_heads": 1,
                "sliding_window": 32,
                "layer_types": [
                    "sliding_attention", "sliding_attention", "sliding_attention",
                    "sliding_attention", "sliding_attention", "full_attention"
                ],
                "max_position_embeddings": 1024,
                "final_logit_softcapping": 30.0,
                "num_kv_shared_layers": 0,
                "rope_parameters": {
                    "full_attention": {
                        "rope_theta": 1000000.0,
                        "rope_type": "proportional",
                        "partial_rotary_factor": 0.25
                    },
                    "sliding_attention": {
                        "rope_theta": 10000.0,
                        "rope_type": "default"
                    }
                },
                "vocab_size_per_layer_input": 0,
                "hidden_size_per_layer_input": 0,
                "enable_moe_block": false,
                "attention_k_eq_v": false,
                "use_double_wide_mlp": false,
                "tie_word_embeddings": true
            }
            """
        let configData = configJSON.data(using: .utf8)!
        let config = try JSONDecoder().decode(Gemma4TextConfiguration.self, from: configData)

        let model = Gemma4TextModel(config)

        let input = MLXArray([1, 2, 3, 4, 5])[.newAxis, .ellipsis]
        let output = model.callAsFunction(input, cache: nil)

        XCTAssertEqual(output.shape, [1, 5, 100])
    }

    /// Test with KV cache sharing enabled
    func testGemma4WithKVSharing() throws {
        let configJSON = """
            {
                "model_type": "gemma4_text",
                "hidden_size": 64,
                "num_hidden_layers": 6,
                "intermediate_size": 128,
                "num_attention_heads": 2,
                "head_dim": 32,
                "global_head_dim": 64,
                "rms_norm_eps": 1e-6,
                "vocab_size": 100,
                "num_key_value_heads": 1,
                "sliding_window": 32,
                "layer_types": [
                    "sliding_attention", "sliding_attention", "full_attention",
                    "sliding_attention", "sliding_attention", "full_attention"
                ],
                "max_position_embeddings": 1024,
                "final_logit_softcapping": 30.0,
                "num_kv_shared_layers": 2,
                "rope_parameters": {
                    "full_attention": {
                        "rope_theta": 1000000.0,
                        "rope_type": "proportional",
                        "partial_rotary_factor": 0.25
                    },
                    "sliding_attention": {
                        "rope_theta": 10000.0,
                        "rope_type": "default"
                    }
                },
                "vocab_size_per_layer_input": 0,
                "hidden_size_per_layer_input": 0,
                "enable_moe_block": false,
                "attention_k_eq_v": false,
                "use_double_wide_mlp": false,
                "tie_word_embeddings": true
            }
            """
        let configData = configJSON.data(using: .utf8)!
        let config = try JSONDecoder().decode(Gemma4TextConfiguration.self, from: configData)

        let model = Gemma4TextModel(config)
        let caches = model.newCache()

        // Should only have 4 caches (6 layers - 2 shared)
        XCTAssertEqual(caches.count, 4)

        let input = MLXArray([1, 2, 3, 4, 5])[.newAxis, .ellipsis]
        let output = model.callAsFunction(input, cache: caches)

        XCTAssertEqual(output.shape, [1, 5, 100])
    }

    /// Test with per-layer embeddings enabled (like E4B)
    func testGemma4WithPerLayerEmbeddings() throws {
        let configJSON = """
            {
                "model_type": "gemma4_text",
                "hidden_size": 64,
                "num_hidden_layers": 6,
                "intermediate_size": 128,
                "num_attention_heads": 2,
                "head_dim": 32,
                "global_head_dim": 64,
                "rms_norm_eps": 1e-6,
                "vocab_size": 100,
                "num_key_value_heads": 1,
                "sliding_window": 32,
                "layer_types": [
                    "sliding_attention", "sliding_attention", "sliding_attention",
                    "sliding_attention", "sliding_attention", "full_attention"
                ],
                "max_position_embeddings": 1024,
                "final_logit_softcapping": 30.0,
                "num_kv_shared_layers": 0,
                "rope_parameters": {
                    "full_attention": {
                        "rope_theta": 1000000.0,
                        "rope_type": "proportional",
                        "partial_rotary_factor": 0.25
                    },
                    "sliding_attention": {
                        "rope_theta": 10000.0,
                        "rope_type": "default"
                    }
                },
                "vocab_size_per_layer_input": 100,
                "hidden_size_per_layer_input": 16,
                "enable_moe_block": false,
                "attention_k_eq_v": false,
                "use_double_wide_mlp": false,
                "tie_word_embeddings": true
            }
            """
        let configData = configJSON.data(using: .utf8)!
        let config = try JSONDecoder().decode(Gemma4TextConfiguration.self, from: configData)

        let model = Gemma4TextModel(config)

        let input = MLXArray([1, 2, 3, 4, 5])[.newAxis, .ellipsis]
        let output = model.callAsFunction(input, cache: nil)

        XCTAssertEqual(output.shape, [1, 5, 100])
    }

    /// Test config parsing from a real Gemma 4 E4B config
    func testGemma4ConfigParsing() throws {
        let configJSON = """
            {
                "model_type": "gemma4",
                "text_config": {
                    "model_type": "gemma4_text",
                    "hidden_size": 2560,
                    "num_hidden_layers": 42,
                    "intermediate_size": 10240,
                    "num_attention_heads": 8,
                    "head_dim": 256,
                    "global_head_dim": 512,
                    "rms_norm_eps": 1e-6,
                    "vocab_size": 262144,
                    "num_key_value_heads": 2,
                    "sliding_window": 512,
                    "layer_types": ["sliding_attention", "sliding_attention", "sliding_attention",
                                    "sliding_attention", "sliding_attention", "full_attention"],
                    "max_position_embeddings": 131072,
                    "final_logit_softcapping": 30.0,
                    "num_kv_shared_layers": 18,
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
                    },
                    "vocab_size_per_layer_input": 262144,
                    "hidden_size_per_layer_input": 256,
                    "enable_moe_block": false,
                    "attention_k_eq_v": false,
                    "use_double_wide_mlp": false,
                    "tie_word_embeddings": true
                }
            }
            """
        let configData = configJSON.data(using: .utf8)!
        let config = try JSONDecoder().decode(Gemma4TextConfiguration.self, from: configData)

        // Verify config parsed correctly by instantiating the model
        // (properties are internal but construction validates parsing)
        let model = Gemma4TextModel(config)
        XCTAssertEqual(model.vocabularySize, 262144)
    }
}
