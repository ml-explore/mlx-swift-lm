// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import XCTest

@testable import MLXLLM
@testable import MLXVLM

final class Qwen38Tests: XCTestCase {

    private func releasedQwen38ConfigJSON() -> String {
        """
        {
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "model_type": "qwen3_5",
            "text_config": {
                "model_type": "qwen3_5_text",
                "hidden_size": 16,
                "num_hidden_layers": 4,
                "intermediate_size": 32,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 8,
                "linear_num_value_heads": 2,
                "linear_num_key_heads": 1,
                "linear_key_head_dim": 8,
                "linear_value_head_dim": 8,
                "linear_conv_kernel_dim": 4,
                "vocab_size": 64,
                "full_attention_interval": 4,
                "layer_types": [
                    "linear_attention",
                    "linear_attention",
                    "linear_attention",
                    "full_attention"
                ]
            },
            "vision_config": {
                "model_type": "qwen3_5",
                "depth": 2,
                "hidden_size": 16,
                "intermediate_size": 32,
                "out_hidden_size": 16,
                "num_heads": 2,
                "patch_size": 16,
                "spatial_merge_size": 2,
                "temporal_patch_size": 2,
                "num_position_embeddings": 64
            }
        }
        """
    }

    private func qwen38TextConfigJSON() -> String {
        """
        {
            "architectures": ["Qwen3_5ForCausalLM"],
            "model_type": "qwen3_8",
            "hidden_size": 16,
            "num_hidden_layers": 4,
            "intermediate_size": 32,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "linear_num_value_heads": 2,
            "linear_num_key_heads": 1,
            "linear_key_head_dim": 8,
            "linear_value_head_dim": 8,
            "linear_conv_kernel_dim": 4,
            "vocab_size": 64,
            "full_attention_interval": 4,
            "layer_types": [
                "linear_attention",
                "full_attention",
                "linear_attention",
                "full_attention"
            ],
            "mtp_num_hidden_layers": 1,
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 10000000.0,
                "partial_rotary_factor": 0.25
            }
        }
        """
    }

    private func qwen38VLMConfigJSON() -> String {
        """
        {
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "model_type": "qwen3_8",
            "image_token_id": 248056,
            "video_token_id": 248057,
            "vision_start_token_id": 248053,
            "vision_end_token_id": 248054,
            "text_config": {
                "model_type": "qwen3_8_text",
                "hidden_size": 16,
                "num_hidden_layers": 4,
                "intermediate_size": 32,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 8,
                "linear_num_value_heads": 2,
                "linear_num_key_heads": 1,
                "linear_key_head_dim": 8,
                "linear_value_head_dim": 8,
                "linear_conv_kernel_dim": 4,
                "vocab_size": 64,
                "full_attention_interval": 2,
                "rope_parameters": {
                    "rope_type": "default",
                    "rope_theta": 10000000.0,
                    "partial_rotary_factor": 0.25
                }
            },
            "vision_config": {
                "model_type": "qwen3_8",
                "depth": 2,
                "hidden_size": 16,
                "intermediate_size": 32,
                "out_hidden_size": 16,
                "num_heads": 2,
                "patch_size": 16,
                "spatial_merge_size": 2,
                "temporal_patch_size": 2,
                "num_position_embeddings": 64
            }
        }
        """
    }

    // MARK: - Configuration & Type Registry Tests

    func testReleasedQwen38MetadataRoutesThroughExistingQwen35Architecture() async throws {
        let data = Data(releasedQwen38ConfigJSON().utf8)
        let base = try JSONDecoder.json5().decode(BaseConfiguration.self, from: data)
        XCTAssertEqual(base.modelType, "qwen3_5")

        let textModel = try await LLMTypeRegistry.shared.createModel(
            configuration: data, modelType: base.modelType)
        let vlmModel = try await VLMTypeRegistry.shared.createModel(
            configuration: data, modelType: base.modelType)

        XCTAssertTrue(textModel is MLXLLM.Qwen35Model)
        XCTAssertTrue(vlmModel is MLXVLM.Qwen35)
        let textBackbone = (textModel as! MLXLLM.Qwen35Model).languageModel.model
        let vlmBackbone = (vlmModel as! MLXVLM.Qwen35).languageModel.model
        XCTAssertEqual(textBackbone.ssmIdx, 0)
        XCTAssertEqual(textBackbone.faIdx, 3)
        XCTAssertEqual(vlmBackbone.ssmIdx, 0)
        XCTAssertEqual(vlmBackbone.faIdx, 3)
    }

    func testQwen38LLMTypeRegistryCreatesModel() async throws {
        let json = qwen38TextConfigJSON()
        let data = Data(json.utf8)

        for type in ["qwen3_8", "qwen3_8_text"] {
            let model = try await LLMTypeRegistry.shared.createModel(
                configuration: data, modelType: type)
            XCTAssertEqual(model.toolCallFormat, .qwen35)
            XCTAssertEqual(model.reasoningConfig, QwenReasoningProtocol.tagged)
        }
    }

    func testQwen38VLMTypeRegistryCreatesModel() async throws {
        let json = qwen38VLMConfigJSON()
        let data = Data(json.utf8)

        let model = try await VLMTypeRegistry.shared.createModel(
            configuration: data, modelType: "qwen3_8")
        XCTAssertEqual(model.toolCallFormat, .qwen35)
        XCTAssertEqual(model.reasoningConfig, QwenReasoningProtocol.tagged)
    }

    func testQwen38ExplicitLayerTypes() throws {
        let json = qwen38TextConfigJSON()
        let config = try JSONDecoder.json5().decode(
            MLXLLM.Qwen35TextConfiguration.self, from: Data(json.utf8))
        XCTAssertEqual(
            config.layerTypes,
            [
                "linear_attention",
                "full_attention",
                "linear_attention",
                "full_attention",
            ])

        let model = MLXLLM.Qwen35TextModel(config)
        XCTAssertEqual(model.model.layers.count, 4)
        XCTAssertTrue(model.model.layers[0].isLinear)
        XCTAssertFalse(model.model.layers[1].isLinear)
        XCTAssertTrue(model.model.layers[2].isLinear)
        XCTAssertFalse(model.model.layers[3].isLinear)
        XCTAssertEqual(model.model.ssmIdx, 0)
        XCTAssertEqual(model.model.faIdx, 1)
    }

    func testQwen38ExplicitLayerTypesMustDescribeEveryLayer() throws {
        let json = """
            {
                "num_hidden_layers": 4,
                "full_attention_interval": 2,
                "layer_types": [
                    "linear_attention",
                    "full_attention",
                    "linear_attention"
                ]
            }
            """

        XCTAssertThrowsError(
            try JSONDecoder.json5().decode(
                MLXLLM.Qwen35TextConfiguration.self, from: Data(json.utf8))
        ) { error in
            XCTAssertTrue(error.localizedDescription.contains("layer_types"))
        }
    }

    func testQwen38RejectsUnknownLayerTypesAndInvalidFallbackInterval() throws {
        let unknownLayerType = """
            {
                "num_hidden_layers": 1,
                "layer_types": ["sliding_attention"]
            }
            """
        XCTAssertThrowsError(
            try JSONDecoder.json5().decode(
                MLXLLM.Qwen35TextConfiguration.self, from: Data(unknownLayerType.utf8))
        ) { error in
            XCTAssertTrue(error.localizedDescription.contains("sliding_attention"))
        }

        let invalidInterval = """
            {
                "num_hidden_layers": 1,
                "full_attention_interval": 0
            }
            """
        XCTAssertThrowsError(
            try JSONDecoder.json5().decode(
                MLXLLM.Qwen35TextConfiguration.self, from: Data(invalidInterval.utf8))
        ) { error in
            XCTAssertTrue(error.localizedDescription.contains("full_attention_interval"))
        }
    }

    // MARK: - Registry Presets

    func testQwen38LLMRegistryConfigurations() {
        let config4bit = LLMRegistry.qwen3_8_27b_4bit
        XCTAssertEqual(config4bit.name, "mlx-community/Qwen3.8-27B-4bit")
        XCTAssertEqual(config4bit.extraEOSTokens, ["<|im_end|>"])

        let config8bit = LLMRegistry.qwen3_8_27b_8bit
        XCTAssertEqual(config8bit.name, "mlx-community/Qwen3.8-27B-8bit")
        XCTAssertEqual(config8bit.extraEOSTokens, ["<|im_end|>"])

        XCTAssertTrue(LLMRegistry.shared.contains(id: "mlx-community/Qwen3.8-27B-4bit"))
        XCTAssertTrue(LLMRegistry.shared.contains(id: "mlx-community/Qwen3.8-27B-8bit"))
    }

    func testQwen38VLMRegistryConfigurations() {
        let config4bit = VLMRegistry.qwen3_8_27B_4bit
        XCTAssertEqual(config4bit.name, "mlx-community/Qwen3.8-27B-4bit")
        XCTAssertEqual(config4bit.extraEOSTokens, ["<|im_end|>"])

        let config8bit = VLMRegistry.qwen3_8_27B_8bit
        XCTAssertEqual(config8bit.name, "mlx-community/Qwen3.8-27B-8bit")
        XCTAssertEqual(config8bit.extraEOSTokens, ["<|im_end|>"])

        XCTAssertTrue(VLMRegistry.shared.contains(id: "mlx-community/Qwen3.8-27B-4bit"))
        XCTAssertTrue(VLMRegistry.shared.contains(id: "mlx-community/Qwen3.8-27B-8bit"))
    }

    func testQwen38MTPRegistryConfiguration() {
        let config = MTPDrafterRegistry.qwen3_8_27b_mtp_4bit
        XCTAssertEqual(config.name, "mlx-community/Qwen3.8-27B-MTP-4bit")
        XCTAssertTrue(MTPDrafterRegistry.shared.contains(id: config.name))
    }

    // MARK: - Weight Sanitization Tests

    func testQwen38TextModelSanitizeStripsVisualPrefixes() throws {
        let config = try JSONDecoder.json5().decode(
            MLXLLM.Qwen35Configuration.self, from: Data(qwen38TextConfigJSON().utf8))
        let model = MLXLLM.Qwen35Model(config)

        let mockWeights: [String: MLXArray] = [
            "model.embed_tokens.weight": MLXArray.zeros([64, 16]),
            "visual.patch_embed.proj.weight": MLXArray.zeros([16, 3, 16, 16]),
            "model.visual.blocks.0.attn.qkv.weight": MLXArray.zeros([16, 16]),
            "vision_tower.blocks.0.norm.weight": MLXArray.zeros([16]),
        ]

        let sanitized = model.sanitize(weights: mockWeights)
        XCTAssertNotNil(sanitized["language_model.model.embed_tokens.weight"])
        XCTAssertNil(sanitized["visual.patch_embed.proj.weight"])
        XCTAssertNil(sanitized["language_model.visual.patch_embed.proj.weight"])
        XCTAssertNil(sanitized["model.visual.blocks.0.attn.qkv.weight"])
        XCTAssertNil(sanitized["vision_tower.blocks.0.norm.weight"])
    }

    func testQwen38VLMModelSanitizeRemapsVisualPrefixes() throws {
        let config = try JSONDecoder.json5().decode(
            MLXVLM.Qwen35Configuration.self, from: Data(qwen38VLMConfigJSON().utf8))
        let model = MLXVLM.Qwen35(config)

        let mockWeights: [String: MLXArray] = [
            "language_model.model.embed_tokens.weight": MLXArray.zeros([64, 16]),
            "visual.merger.mlp.0.weight": MLXArray.zeros([16, 16]),
            "model.visual.merger.mlp.1.weight": MLXArray.zeros([16, 16]),
        ]

        let sanitized = model.sanitize(weights: mockWeights)
        XCTAssertNotNil(sanitized["language_model.model.embed_tokens.weight"])
        XCTAssertNotNil(sanitized["vision_tower.merger.mlp.0.weight"])
        XCTAssertNotNil(sanitized["vision_tower.merger.mlp.1.weight"])
    }

    // MARK: - MTP Speculative Decoding Registration Tests

    func testQwen38MTPRegistrations() async throws {
        await Qwen35TextMTPRegistration.register()
        await Qwen35VLMMTPRegistration.register()

        let textData = Data(qwen38TextConfigJSON().utf8)
        let vlmData = Data(qwen38VLMConfigJSON().utf8)

        // Text drafter matches
        let textDrafter = try await MTPDrafterTypeRegistry.shared.createModel(
            configuration: textData, modelType: "qwen3_8_text")
        XCTAssertNotNil(textDrafter)

        let textMTPDrafter = try await MTPDrafterTypeRegistry.shared.createModel(
            configuration: textData, modelType: "qwen3_8_mtp")
        XCTAssertNotNil(textMTPDrafter)

        // VLM drafter matches
        let vlmMTPDrafter = try await MTPDrafterTypeRegistry.visionLanguage.createModel(
            configuration: vlmData, modelType: "qwen3_8_mtp")
        XCTAssertTrue(vlmMTPDrafter is Qwen35VLMNextNDraftModel)
    }
}
