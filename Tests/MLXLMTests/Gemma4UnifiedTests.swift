// Copyright © 2026 Apple Inc.

import CoreImage
import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import MLXVLM

private struct Gemma4UnifiedTestTokenizer: Tokenizer {
    let vocabularySize: Int = 64
    let bosToken: String? = nil
    let eosToken: String? = nil
    let eosTokenId: Int? = 1
    let unknownToken: String? = nil
    let unknownTokenId: Int? = 0

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        [31, 2]
    }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        tokenIds.map(String.init).joined(separator: " ")
    }

    func convertTokenToId(_ token: String) -> Int? {
        Int(token)
    }

    func convertIdToToken(_ id: Int) -> String? {
        String(id)
    }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        [31, 2]
    }
}

struct Gemma4UnifiedTests {
    private func decodeConfig(_ json: String) throws -> Gemma4UnifiedConfiguration {
        try JSONDecoder.json5().decode(Gemma4UnifiedConfiguration.self, from: Data(json.utf8))
    }

    @Test("Gemma4 Unified config decodes unified defaults and eoa_token_index")
    func configDecoding() throws {
        let config = try decodeConfig(
            """
            {
              "model_type": "gemma4_unified",
              "eoa_token_index": 258883,
              "text_config": { "model_type": "gemma4_unified_text" },
              "vision_config": { "model_type": "gemma4_unified_vision" },
              "audio_config": { "model_type": "gemma4_unified_audio" }
            }
            """)

        #expect(config.modelType == "gemma4_unified")
        #expect(config.eoaTokenId == 258883)
        #expect(config.textConfiguration.modelType == "gemma4_unified_text")
        #expect(config.textConfiguration.hiddenSize == 3840)
        #expect(config.textConfiguration.numKVSharedLayers == 0)
        #expect(config.textConfiguration.hiddenSizePerLayerInput == 0)
        #expect(config.textConfiguration.attentionKEqV)
        #expect(config.textConfiguration.useBidirectionalAttention == "vision")
        #expect(config.visionConfiguration?.modelPatchSize == 48)
    }

    @Test("Gemma4 Unified processor emits model patches and position ids")
    func processorPatchifiesImages() async throws {
        let data = Data(
            """
            {
              "processor_class": "Gemma4UnifiedProcessor",
              "image_token_id": 31,
              "boi_token_id": 28,
              "eoi_token_id": 29,
              "image_processor": {
                "patch_size": 2,
                "pooling_kernel_size": 2,
                "model_patch_size": 4,
                "max_soft_tokens": 4,
                "size": { "height": 8, "width": 8 }
              }
            }
            """.utf8)
        let config = try JSONDecoder.json5().decode(
            Gemma4UnifiedProcessorConfiguration.self, from: data)
        let processor = Gemma4UnifiedProcessor(config, tokenizer: Gemma4UnifiedTestTokenizer())
        let image = CIImage(color: .black).cropped(to: CGRect(x: 0, y: 0, width: 8, height: 8))

        let input = try await processor.prepare(
            input: UserInput(prompt: "describe", images: [.ciImage(image)]))

        #expect(input.image?.pixels.shape == [1, 4, 48])
        #expect(input.image?.positionIds?.shape == [1, 4, 2])
        #expect(input.text.tokens.asArray(Int32.self) == [28, 31, 31, 31, 31, 29, 2])
    }

    @Test("Gemma4 Unified model accepts vision embeddings")
    func modelVisionForward() throws {
        let config = try decodeConfig(
            """
            {
              "model_type": "gemma4_unified",
              "vocab_size": 32,
              "image_token_id": 31,
              "audio_token_id": 30,
              "video_token_id": 29,
              "text_config": {
                "model_type": "gemma4_unified_text",
                "hidden_size": 8,
                "num_hidden_layers": 1,
                "intermediate_size": 16,
                "num_attention_heads": 1,
                "num_key_value_heads": 1,
                "num_global_key_value_heads": 1,
                "head_dim": 8,
                "global_head_dim": 8,
                "vocab_size": 32,
                "vocab_size_per_layer_input": 32,
                "num_kv_shared_layers": 0,
                "hidden_size_per_layer_input": 0,
                "sliding_window": 8,
                "sliding_window_pattern": 1,
                "attention_k_eq_v": true,
                "use_double_wide_mlp": false,
                "layer_types": ["full_attention"],
                "tie_word_embeddings": true
              },
              "vision_config": {
                "model_type": "gemma4_unified_vision",
                "patch_size": 2,
                "pooling_kernel_size": 2,
                "model_patch_size": 4,
                "mm_embed_dim": 8,
                "mm_posemb_size": 4,
                "num_soft_tokens": 4,
                "output_proj_dims": 8
              },
              "audio_config": null
            }
            """)
        let model = Gemma4Unified(config)
        let inputIds = MLXArray([0, 31, 31, 31, 31, 1]).reshaped(1, 6)
        let pixelValues = MLXArray.zeros([1, 4, 48], dtype: .float32)
        let positionIds = MLXArray.zeros([1, 4, 2], dtype: .int32)
        let input = LMInput(
            text: .init(tokens: inputIds),
            image: .init(pixels: pixelValues, positionIds: positionIds)
        )

        let result = try model.prepare(
            input, cache: model.newCache(parameters: nil), windowSize: nil)

        guard case .logits(let output) = result else {
            Issue.record("Expected Gemma4Unified.prepare to return logits")
            return
        }
        #expect(output.logits.shape == [1, 6, 32])
    }
}
