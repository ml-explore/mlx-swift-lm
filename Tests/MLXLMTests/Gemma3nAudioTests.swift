import MLX
import MLXLMCommon
import MLXNN
import MLXVLM
import XCTest

/// Test the Gemma 3n audio encoder compilation and basic functionality.
/// These tests verify that the audio encoder components work correctly
/// without requiring model weights (synthetic data only).
class Gemma3nAudioTests: XCTestCase {

    /// Test that the audio configuration decodes correctly from JSON.
    func testAudioConfigDecoding() throws {
        let json = """
            {
                "input_feat_size": 128,
                "hidden_size": 1536,
                "conf_num_attention_heads": 8,
                "conf_num_hidden_layers": 12,
                "conf_attention_chunk_size": 12,
                "conf_attention_context_left": 13,
                "conf_attention_context_right": 0,
                "conf_attention_invalid_logits_value": -1000000000.0,
                "conf_attention_logit_cap": 50.0,
                "conf_conv_kernel_size": 5,
                "conf_reduction_factor": 4,
                "conf_residual_weight": 0.5,
                "sscp_conv_channel_size": [128, 32],
                "sscp_conv_kernel_size": [[3, 3], [3, 3]],
                "sscp_conv_stride_size": [[2, 2], [2, 2]],
                "sscp_conv_eps": 0.001,
                "rms_norm_eps": 0.000001,
                "gradient_clipping": 10000000000.0,
                "vocab_size": 128,
                "vocab_offset": 262272
            }
            """
        let data = json.data(using: .utf8)!
        let config = try JSONDecoder().decode(Gemma3nAudioConfiguration.self, from: data)

        XCTAssertEqual(config.inputFeatSize, 128)
        XCTAssertEqual(config.hiddenSize, 1536)
        XCTAssertEqual(config.confNumAttentionHeads, 8)
        XCTAssertEqual(config.confNumHiddenLayers, 12)
        XCTAssertEqual(config.headDim, 192)  // 1536 / 8
        XCTAssertEqual(config.maxBackward, 12)  // contextLeft - 1
        XCTAssertEqual(config.maxForward, 0)
        XCTAssertEqual(config.contextSize, 24)  // chunk(12) + backward(12) + forward(0)
        XCTAssertEqual(config.maxSpanPlusOne, 13)  // backward(12) + forward(0) + 1
        XCTAssertEqual(config.vocabOffset, 262272)
    }

    /// Test that the full multimodal configuration decodes correctly.
    func testMultimodalConfigDecoding() throws {
        let json = """
            {
                "model_type": "gemma3n",
                "vocab_size": 257152,
                "audio_token_id": 262273,
                "image_token_id": 262145,
                "audio_soft_tokens_per_image": 188,
                "vision_soft_tokens_per_image": 256,
                "hidden_size": 2048,
                "rms_norm_eps": 0.000001,
                "pad_token_id": 0,
                "text_config": {
                    "model_type": "gemma3n_text",
                    "hidden_size": 2048,
                    "num_hidden_layers": 32,
                    "intermediate_size": [8192],
                    "num_attention_heads": 8,
                    "head_dim": 256,
                    "rms_norm_eps": 0.000001,
                    "vocab_size": 262400,
                    "num_key_value_heads": 4,
                    "num_kv_shared_layers": 14,
                    "vocab_size_per_layer_input": 262144,
                    "sliding_window": 1024,
                    "max_position_embeddings": 32768,
                    "rope_local_base_freq": 10000.0,
                    "rope_theta": 1000000.0,
                    "final_logit_softcapping": 30.0,
                    "hidden_size_per_layer_input": 1024,
                    "altup_num_inputs": 4,
                    "altup_correct_scale": true,
                    "altup_active_idx": 0,
                    "laurel_rank": 64
                },
                "audio_config": {
                    "input_feat_size": 128,
                    "hidden_size": 1536,
                    "conf_num_attention_heads": 8,
                    "conf_num_hidden_layers": 12,
                    "conf_attention_chunk_size": 12,
                    "conf_attention_context_left": 13,
                    "conf_attention_context_right": 0,
                    "conf_attention_invalid_logits_value": -1000000000.0,
                    "conf_attention_logit_cap": 50.0,
                    "conf_conv_kernel_size": 5,
                    "conf_reduction_factor": 4,
                    "conf_residual_weight": 0.5,
                    "sscp_conv_channel_size": [128, 32],
                    "sscp_conv_kernel_size": [[3, 3], [3, 3]],
                    "sscp_conv_stride_size": [[2, 2], [2, 2]],
                    "sscp_conv_eps": 0.001,
                    "rms_norm_eps": 0.000001,
                    "gradient_clipping": 10000000000.0,
                    "vocab_size": 128,
                    "vocab_offset": 262272
                }
            }
            """
        let data = json.data(using: .utf8)!
        let config = try JSONDecoder().decode(Gemma3nConfiguration.self, from: data)

        XCTAssertEqual(config.modelType, "gemma3n")
        XCTAssertEqual(config.audioTokenId, 262273)
        XCTAssertEqual(config.audioSoftTokensPerImage, 188)
        XCTAssertEqual(config.textConfig.hiddenSize, 2048)
        XCTAssertEqual(config.textConfig.numHiddenLayers, 32)
        XCTAssertEqual(config.audioConfig.hiddenSize, 1536)
        XCTAssertEqual(config.audioConfig.confNumHiddenLayers, 12)
    }

    /// Test that the audio encoder initializes and produces correct output shapes.
    func testAudioEncoderShapes() throws {
        let json = """
            {
                "input_feat_size": 128,
                "hidden_size": 1536,
                "conf_num_attention_heads": 8,
                "conf_num_hidden_layers": 2,
                "conf_attention_chunk_size": 12,
                "conf_attention_context_left": 13,
                "conf_attention_context_right": 0,
                "conf_attention_invalid_logits_value": -1000000000.0,
                "conf_attention_logit_cap": 50.0,
                "conf_conv_kernel_size": 5,
                "conf_reduction_factor": 4,
                "conf_residual_weight": 0.5,
                "sscp_conv_channel_size": [128, 32],
                "sscp_conv_kernel_size": [[3, 3], [3, 3]],
                "sscp_conv_stride_size": [[2, 2], [2, 2]],
                "sscp_conv_eps": 0.001,
                "rms_norm_eps": 0.000001,
                "gradient_clipping": 10000000000.0,
                "vocab_size": 128,
                "vocab_offset": 262272
            }
            """
        let data = json.data(using: .utf8)!
        let config = try JSONDecoder().decode(Gemma3nAudioConfiguration.self, from: data)

        // Use only 2 conformer layers for fast testing
        let audioModel = Gemma3nAudioModel(config)

        // Synthetic mel input: [B=1, T=48, F=128]
        // Using 48 frames so it's divisible by stride product (4) and reduction factor (4)
        let mel = MLXRandom.normal([1, 48, 128])
        let mask = MLXArray.zeros([1, 48]).asType(.bool)

        let (output, outputMask) = audioModel(mel, mask: mask)
        eval(output)
        eval(outputMask)

        // Expected: T=48 → T/4 (conv stride) = 12 → T/4 (reduction) = 3
        print("Audio encoder output shape: \(output.shape)")
        print("Audio encoder mask shape: \(outputMask.shape)")

        XCTAssertEqual(output.dim(0), 1, "Batch size should be 1")
        XCTAssertEqual(output.dim(2), 1536, "Hidden dim should match config")
        XCTAssertTrue(output.dim(1) > 0, "Time dimension should be > 0")
    }
}
