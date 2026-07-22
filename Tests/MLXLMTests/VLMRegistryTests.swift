// Copyright © 2026 Apple Inc.

import Foundation
import MLXLMCommon
import MLXVLM
import XCTest

final class VLMRegistryTests: XCTestCase {

    func testGemma4VLMRegistryUsesTurnEndToken() {
        for configuration in [
            VLMRegistry.gemma4_E2B_it_4bit,
            VLMRegistry.gemma4_E4B_it_4bit,
            VLMRegistry.gemma4_31B_it_4bit,
            VLMRegistry.gemma4_26BA4B_it_4bit,
        ] {
            XCTAssertEqual(configuration.extraEOSTokens, ["<turn|>"])
        }
    }

    func testDeepseekOCRModelTypeLoadsPinnedHFConfig() async throws {
        let contains = await VLMTypeRegistry.shared.contains("deepseekocr")
        XCTAssertTrue(contains)
        let model = try await VLMTypeRegistry.shared.createModel(
            configuration: Self.deepseekOCRConfig.data(using: .utf8)!,
            modelType: "deepseekocr")
        XCTAssertTrue(model is DeepseekOCR)
        XCTAssertEqual(VLMRegistry.deepseekOCR5bit.name, "mlx-community/DeepSeek-OCR-5bit")
    }

    func testDeepseekOCRProcessorAliasesLoadPinnedHFConfig() async throws {
        for processorType in ["DeepseekOCRProcessor", "DeepseekVLV2Processor"] {
            let processor = try await VLMProcessorTypeRegistry.shared.createModel(
                configuration: Self.deepseekOCRProcessorConfig.data(using: .utf8)!,
                processorType: processorType,
                tokenizer: StubTokenizer())
            XCTAssertTrue(processor is DeepseekOCRProcessor)
        }
    }

    func testDeepseekOCRFactoryFailsGracefullyWhenWeightsAreMissing() async throws {
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        try Self.deepseekOCRConfig.data(using: .utf8)!.write(
            to: directory.appendingPathComponent("config.json"))
        try Self.deepseekOCRProcessorConfig.data(using: .utf8)!.write(
            to: directory.appendingPathComponent("processor_config.json"))

        do {
            _ = try await VLMModelFactory.shared.load(from: directory, using: StubTokenizerLoader())
            XCTFail("expected missing weights to throw")
        } catch {
            XCTAssertFalse((error as NSError).localizedDescription.isEmpty)
        }
    }

    private func makeTemporaryDirectory() throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("VLMRegistryTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return directory
    }
}

private final class StubTokenizerLoader: TokenizerLoader, @unchecked Sendable {
    func load(from directory: URL) async throws -> any Tokenizer {
        StubTokenizer()
    }
}

private struct StubTokenizer: Tokenizer {
    func encode(text: String, addSpecialTokens: Bool) -> [Int] { [] }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { "" }
    func convertTokenToId(_ token: String) -> Int? { nil }
    func convertIdToToken(_ id: Int) -> String? { nil }
    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }
    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        []
    }
}

extension VLMRegistryTests {
    fileprivate static let deepseekOCRConfig = #"""
        {
         "architectures": [
          "DeepseekOCRForCausalLM"
         ],
         "auto_map": {
          "AutoConfig": "modeling_deepseekocr.DeepseekOCRConfig",
          "AutoModel": "modeling_deepseekocr.DeepseekOCRForCausalLM"
         },
         "bos_token_id": 0,
         "candidate_resolutions": [[1024, 1024]],
         "eos_token_id": 1,
         "first_k_dense_replace": 1,
         "global_view_pos": "head",
         "hidden_size": 1280,
         "intermediate_size": 6848,
         "kv_lora_rank": null,
         "language_config": {
          "architectures": ["DeepseekV2ForCausalLM"],
          "bos_token_id": 0,
          "eos_token_id": 1,
          "first_k_dense_replace": 1,
          "hidden_size": 1280,
          "intermediate_size": 6848,
          "kv_lora_rank": null,
          "lm_head": true,
          "max_position_embeddings": 8192,
          "moe_intermediate_size": 896,
          "n_group": 1,
          "n_routed_experts": 64,
          "n_shared_experts": 2,
          "num_attention_heads": 10,
          "num_experts_per_tok": 6,
          "num_hidden_layers": 12,
          "num_key_value_heads": 10,
          "q_lora_rank": null,
          "qk_nope_head_dim": 0,
          "qk_rope_head_dim": 0,
          "rm_head": false,
          "topk_group": 1,
          "topk_method": "greedy",
          "torch_dtype": "bfloat16",
          "use_mla": false,
          "v_head_dim": 0,
          "vocab_size": 129280
         },
         "lm_head": true,
         "max_position_embeddings": 8192,
         "model_type": "deepseekocr",
         "moe_intermediate_size": 896,
         "n_group": 1,
         "n_routed_experts": 64,
         "n_shared_experts": 2,
         "num_attention_heads": 10,
         "num_experts_per_tok": 6,
         "num_hidden_layers": 12,
         "num_key_value_heads": 10,
         "projector_config": {
          "input_dim": 2048,
          "model_type": "mlp_projector",
          "n_embed": 1280,
          "projector_type": "linear"
         },
         "q_lora_rank": null,
         "qk_nope_head_dim": 0,
         "qk_rope_head_dim": 0,
         "quantization": {
          "group_size": 64,
          "bits": 5,
          "mode": "affine"
         },
         "quantization_config": {
          "group_size": 64,
          "bits": 5,
          "mode": "affine"
         },
         "rm_head": false,
         "tile_tag": "2D",
         "topk_group": 1,
         "topk_method": "greedy",
         "transformers_version": "4.46.3",
         "use_mla": false,
         "v_head_dim": 0,
         "vision_config": {
          "image_size": 1024,
          "mlp_ratio": 3.7362,
          "model_name": "deeplip_b_l",
          "model_type": "vision",
          "width": {
           "clip-l-14-224": {
            "heads": 16,
            "image_size": 224,
            "layers": 24,
            "patch_size": 14,
            "width": 1024
           },
           "sam_vit_b": {
            "downsample_channels": [512, 1024],
            "global_attn_indexes": [2, 5, 8, 11],
            "heads": 12,
            "layers": 12,
            "width": 768
           }
          }
         },
         "vocab_size": 129280
        }
        """#

    fileprivate static let deepseekOCRProcessorConfig = #"""
        {
         "add_special_token": false,
         "candidate_resolutions": [[1024, 1024]],
         "downsample_ratio": 4,
         "ignore_id": -100,
         "image_mean": [0.5, 0.5, 0.5],
         "image_std": [0.5, 0.5, 0.5],
         "image_token": " ",
         "mask_prompt": false,
         "normalize": true,
         "pad_token": "<｜▁pad▁｜>",
         "patch_size": 16,
         "processor_class": "DeepseekVLV2Processor",
         "sft_format": "deepseek"
        }
        """#
}
