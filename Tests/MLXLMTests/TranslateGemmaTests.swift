// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import XCTest

/// Regression coverage for TranslateGemma.
///
/// TranslateGemma is a translation-tuned fine-tune of Gemma 3 (the vision tower is removed
/// in the `mlx-community` checkpoints), so it loads through the existing `gemma3` text path
/// in ``Gemma3TextModel``. These tests lock in the two behaviors that make that work:
///   1. The configuration decoder resolving the text fields from the nested `text_config`
///      block of a `Gemma3ForConditionalGeneration` config.
///   2. `sanitize` flattening the `language_model.` prefix and tying `lm_head`.
/// Plus a guard that the registry presets point at the correct Hugging Face ids.
public class TranslateGemmaTests: XCTestCase {

    // MARK: - Configuration decoding

    /// A TranslateGemma `config.json` is a `Gemma3ForConditionalGeneration` config: a top-level
    /// `model_type: gemma3` with the real text hyper-parameters nested under `text_config` and
    /// no `vision_config`. The decoder must read the text fields from `text_config`.
    func testConfigurationDecodingFromNestedTextConfig() throws {
        // Values mirror mlx-community/translategemma-4b-it-4bit (text_config).
        let json = """
            {
                "model_type": "gemma3",
                "architectures": ["Gemma3ForConditionalGeneration"],
                "image_token_index": 262144,
                "text_config": {
                    "model_type": "gemma3_text",
                    "hidden_size": 2560,
                    "num_hidden_layers": 34,
                    "intermediate_size": 10240,
                    "num_attention_heads": 8,
                    "head_dim": 256,
                    "rms_norm_eps": 1e-6,
                    "vocab_size": 262208,
                    "num_key_value_heads": 4,
                    "rope_theta": 1000000.0,
                    "rope_local_base_freq": 10000.0,
                    "query_pre_attn_scalar": 256,
                    "sliding_window": 1024,
                    "max_position_embeddings": 131072
                }
            }
            """

        let config = try JSONDecoder().decode(
            Gemma3TextConfiguration.self, from: json.data(using: .utf8)!)

        // Resolved from text_config, not the conditional-generation wrapper.
        XCTAssertEqual(config.modelType, "gemma3_text")
        XCTAssertEqual(config.hiddenSize, 2560)
        XCTAssertEqual(config.hiddenLayers, 34)
        XCTAssertEqual(config.intermediateSize, 10240)
        XCTAssertEqual(config.attentionHeads, 8)
        XCTAssertEqual(config.headDim, 256)
        XCTAssertEqual(config.vocabularySize, 262208)
        XCTAssertEqual(config.kvHeads, 4)
        XCTAssertEqual(config.ropeTheta, 1_000_000.0)
        XCTAssertEqual(config.ropeLocalBaseFreq, 10_000.0)
        XCTAssertEqual(config.slidingWindow, 1024)
        XCTAssertEqual(config.maxPositionEmbeddings, 131072)
        // Absent in the 4b text_config -> falls back to the Gemma 3 default pattern.
        XCTAssertEqual(config.slidingWindowPattern, 6)
    }

    // MARK: - Weight sanitization

    /// `mlx_vlm.convert` leaves text weights under a `language_model.` prefix, and TranslateGemma
    /// ties `lm_head` to the token embeddings. `sanitize` must undo both.
    func testSanitizeStripsLanguageModelPrefixAndTiesLmHead() {
        let config = Self.tinyConfig()
        let model = Gemma3TextModel(config)

        let weights: [String: MLXArray] = [
            "language_model.model.embed_tokens.weight":
                MLXArray.zeros([config.vocabularySize, config.hiddenSize]),
            "language_model.model.norm.weight": MLXArray.zeros([config.hiddenSize]),
        ]

        let sanitized = model.sanitize(weights: weights)

        // Prefix removed.
        XCTAssertNil(sanitized["language_model.model.embed_tokens.weight"])
        XCTAssertNotNil(sanitized["model.embed_tokens.weight"])
        XCTAssertNotNil(sanitized["model.norm.weight"])
        // lm_head tied from the embeddings (the checkpoint ships no lm_head.weight).
        XCTAssertNotNil(sanitized["lm_head.weight"])
        XCTAssertEqual(sanitized["lm_head.weight"]?.dim(0), config.vocabularySize)
    }

    // MARK: - Registry presets

    func testPresetsResolveToExpectedIds() {
        let expected: [(ModelConfiguration, String)] = [
            (LLMRegistry.translategemma_4b_it_4bit, "mlx-community/translategemma-4b-it-4bit"),
            (LLMRegistry.translategemma_4b_it_8bit, "mlx-community/translategemma-4b-it-8bit"),
            (LLMRegistry.translategemma_12b_it_4bit, "mlx-community/translategemma-12b-it-4bit"),
            (LLMRegistry.translategemma_12b_it_8bit, "mlx-community/translategemma-12b-it-8bit"),
            (LLMRegistry.translategemma_27b_it_4bit, "mlx-community/translategemma-27b-it-4bit"),
            (LLMRegistry.translategemma_27b_it_8bit, "mlx-community/translategemma-27b-it-8bit"),
        ]

        for (configuration, id) in expected {
            XCTAssertEqual(configuration.name, id)
            // Gemma 3 chat turns end with <end_of_turn>; required to stop generation cleanly.
            XCTAssertTrue(configuration.extraEOSTokens.contains("<end_of_turn>"))
        }
    }

    // MARK: - Message generator

    /// With language codes in `additionalContext`, the generator must emit the structured
    /// single-mapping content the TranslateGemma chat template requires.
    func testMessageGeneratorBuildsStructuredTranslationContent() {
        let generator = Gemma3MessageGenerator()
        let input = UserInput(
            chat: [.user("Hello, how are you?")],
            additionalContext: ["source_lang_code": "en", "target_lang_code": "fr"]
        )

        let messages = generator.generate(from: input)
        XCTAssertEqual(messages.count, 1)
        XCTAssertEqual(messages[0]["role"] as? String, "user")

        guard let content = messages[0]["content"] as? [[String: String]],
            let part = content.first
        else {
            return XCTFail(
                "Expected a structured content list, got: \(String(describing: messages[0]["content"]))"
            )
        }
        XCTAssertEqual(content.count, 1)
        XCTAssertEqual(part["type"], "text")
        XCTAssertEqual(part["source_lang_code"], "en")
        XCTAssertEqual(part["target_lang_code"], "fr")
        XCTAssertEqual(part["text"], "Hello, how are you?")
    }

    /// Without language codes the generator is identical to the default (plain Gemma 3),
    /// so non-translation `gemma3` models are unaffected.
    func testMessageGeneratorFallsBackToPlainContentWithoutCodes() {
        let generator = Gemma3MessageGenerator()
        let input = UserInput(chat: [.user("Why is the sky blue?")])

        let messages = generator.generate(from: input)
        XCTAssertEqual(messages.count, 1)
        XCTAssertEqual(messages[0]["role"] as? String, "user")
        XCTAssertEqual(messages[0]["content"] as? String, "Why is the sky blue?")
    }

    // MARK: - Helpers

    /// Minimal configuration to exercise `sanitize` without allocating a full-size model.
    private static func tinyConfig() -> Gemma3TextConfiguration {
        Gemma3TextConfiguration(
            modelType: "gemma3_text",
            hiddenSize: 8,
            hiddenLayers: 1,
            intermediateSize: 16,
            attentionHeads: 2,
            headDim: 4,
            rmsNormEps: 1e-6,
            vocabularySize: 32,
            kvHeads: 1,
            ropeTheta: 1_000_000.0,
            ropeLocalBaseFreq: 10_000.0,
            ropeTraditional: false,
            queryPreAttnScalar: 256,
            slidingWindow: 16,
            slidingWindowPattern: 6,
            maxPositionEmbeddings: 64
        )
    }
}
