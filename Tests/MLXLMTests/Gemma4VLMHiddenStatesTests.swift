// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
@_spi(GemmaEncoder) import MLXVLM
import Testing

/// `hiddenStates(_:)` on the Gemma 4 VLMs must return the tensor the LM head consumes for the
/// same input `prepare` sees, so a client head can read it from the loaded VLM.
///
/// Imports `MLXVLM` without `@testable`, as a client would, so it fails to compile if the
/// method is not reachable. Tiny random models; no weights are downloaded.
struct Gemma4VLMHiddenStatesTests {

    enum Variant: String, CaseIterable, CustomTestStringConvertible {
        case gemma4, gemma4Unified
        var testDescription: String { rawValue }
    }

    /// The tied LM head plus the logit softcap, applied to `hidden`.
    private static func headLogits(
        _ hidden: MLXArray, parameters: [String: MLXArray], softcap: Float?
    ) throws -> MLXArray {
        #expect(parameters["language_model.lm_head.weight"] == nil)
        let embedding = try #require(parameters["language_model.model.embed_tokens.weight"])
        var logits = matmul(hidden, embedding.T)
        if let softcap, softcap > 0 {
            logits = tanh(logits / softcap) * softcap
        }
        return logits
    }

    /// Text only: the head on `hiddenStates` gives the model's own logits.
    @Test(arguments: Variant.allCases)
    func textHiddenStatesAreTheLMHeadInput(_ variant: Variant) throws {
        let tokens = MLXArray((0 ..< 8).map { Int32(($0 * 7 + 3) % 64) }).reshaped(1, 8)
        let input = LMInput(tokens: tokens)
        let hidden: MLXArray
        let logits: MLXArray
        let parameters: [String: MLXArray]
        let softcap: Float?
        switch variant {
        case .gemma4:
            let config = try JSONDecoder().decode(
                Gemma4Configuration.self, from: Data(Self.textJSON.utf8))
            let model = Gemma4(config)
            hidden = try model.hiddenStates(input)
            logits = model(tokens, cache: nil)
            parameters = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
            softcap = config.textConfiguration.finalLogitSoftcapping
        case .gemma4Unified:
            let config = try JSONDecoder().decode(
                Gemma4UnifiedConfiguration.self, from: Data(Self.unifiedJSON.utf8))
            let model = Gemma4Unified(config)
            hidden = try model.hiddenStates(input)
            logits = model(tokens, cache: nil)
            parameters = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
            softcap = config.textConfiguration.finalLogitSoftcapping
        }
        let expected = try Self.headLogits(hidden, parameters: parameters, softcap: softcap)
        eval(expected, logits)
        #expect(hidden.shape == [1, 8, 64])
        #expect(allClose(expected, logits, atol: 1e-5).item(Bool.self))
    }

    /// Image prompt with bidirectional vision attention: the head on `hiddenStates` gives
    /// `prepare`'s logits. This needs the same soft tokens, per-layer inputs and vision mask.
    @Test func imageHiddenStatesMatchPrepare() throws {
        let config = try JSONDecoder().decode(
            Gemma4Configuration.self, from: Data(Self.visionJSON.utf8))
        let model = Gemma4(config)
        eval(model)

        let tokens = MLXArray([5, 6, 100, 100, 100, 100, 7, 8].map { Int32($0) }).reshaped(1, 8)
        // 8x8 pixels with patch size 4: 4 soft tokens for the 4 placeholders.
        let pixels = (MLXArray(0 ..< 192).reshaped([1, 3, 8, 8]).asType(.float32)) / 192.0
        let input = LMInput(
            text: LMInput.Text(tokens: tokens), image: LMInput.ProcessedImage(pixels: pixels))

        let hidden = try model.hiddenStates(input)
        let result = try model.prepare(
            input, cache: try model.newCache(parameters: nil), state: nil,
            prefill: .init(stepSize: 1024))
        guard case .logits(let output) = result else {
            Issue.record("Expected .logits from Gemma4.prepare")
            return
        }
        let expected = try Self.headLogits(
            hidden, parameters: Dictionary(uniqueKeysWithValues: model.parameters().flattened()),
            softcap: config.textConfiguration.finalLogitSoftcapping)
        eval(expected, output.logits)
        #expect(expected.shape == output.logits.shape)
        #expect(allClose(expected, output.logits, atol: 1e-5).item(Bool.self))
    }

    // MARK: - Configs

    /// Per-layer inputs and KV-shared layers on, as in E2B.
    private static let textJSON = """
        {
          "text_config": {
            "hidden_size": 64, "num_hidden_layers": 4, "intermediate_size": 128,
            "num_attention_heads": 2, "num_key_value_heads": 1,
            "head_dim": 16, "global_head_dim": 32,
            "vocab_size": 64, "vocab_size_per_layer_input": 64,
            "num_kv_shared_layers": 2, "hidden_size_per_layer_input": 8,
            "sliding_window": 8, "sliding_window_pattern": 2,
            "max_position_embeddings": 512
          },
          "vision_config": {
            "num_hidden_layers": 1, "hidden_size": 16, "intermediate_size": 32,
            "num_attention_heads": 2, "head_dim": 8, "position_embedding_size": 64
          }
        }
        """

    private static let unifiedJSON = """
        {
          "model_type": "gemma4_unified", "vocab_size": 64,
          "text_config": {
            "model_type": "gemma4_unified_text",
            "hidden_size": 64, "num_hidden_layers": 2, "intermediate_size": 128,
            "num_attention_heads": 2, "num_key_value_heads": 1,
            "num_global_key_value_heads": 1, "head_dim": 16, "global_head_dim": 32,
            "vocab_size": 64, "num_kv_shared_layers": 0, "hidden_size_per_layer_input": 0,
            "sliding_window": 8, "sliding_window_pattern": 2,
            "max_position_embeddings": 512
          },
          "vision_config": null,
          "audio_config": null
        }
        """

    /// The vision tower emits 4 soft tokens per image; image tokens attend both ways.
    private static let visionJSON = """
        {
          "model_type": "gemma4",
          "image_token_id": 100,
          "text_config": {
            "hidden_size": 32, "num_hidden_layers": 2, "intermediate_size": 64,
            "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 16,
            "global_head_dim": 32, "vocab_size": 200, "vocab_size_per_layer_input": 200,
            "num_kv_shared_layers": 0, "hidden_size_per_layer_input": 8,
            "sliding_window": 16, "sliding_window_pattern": 2, "max_position_embeddings": 512,
            "use_bidirectional_attention": "vision"
          },
          "vision_config": {
            "num_hidden_layers": 1, "hidden_size": 16, "intermediate_size": 32,
            "num_attention_heads": 2, "head_dim": 8, "patch_size": 4,
            "default_output_length": 4, "pooling_kernel_size": 1, "position_embedding_size": 8
          }
        }
        """
}
