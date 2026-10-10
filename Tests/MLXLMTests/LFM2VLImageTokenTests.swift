// Copyright © 2026 Apple Inc.
//
// Regression coverage for the LFM2-VL image placeholder id.
//
// The processor expands the `<image>` placeholder the chat template emits, and
// the model later scans the prompt for `config.json`'s `image_token_id`. The
// two must agree, and the id is per-checkpoint: 396 on LFM2-VL and
// LFM2.5-VL-1.6B, 124907 on LFM2.5-VL-3B. A hardcoded 396 left the placeholder
// unexpanded on the 3B, and the model then aborted the host process with
// "Image features and image tokens do not match: tokens: 1".

import CoreImage
import Foundation
import MLX
import MLXLMCommon
import XCTest

@testable import MLXVLM

/// A tokenizer with a fixed vocabulary mapping and a fixed chat template output.
private struct StubTokenizer: Tokenizer {
    var ids: [String: Int] = [:]
    var promptTokens: [Int] = []

    func convertTokenToId(_ token: String) -> Int? { ids[token] }
    func convertIdToToken(_ id: Int) -> String? { ids.first { $0.value == id }?.key }
    func encode(text: String, addSpecialTokens: Bool) -> [Int] { promptTokens }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { "" }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] { promptTokens }

    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }
}

final class LFM2VLImageTokenTests: XCTestCase {

    private func processorConfig(_ json: String = "{}") throws -> LFM2VLProcessorConfiguration {
        try JSONDecoder().decode(LFM2VLProcessorConfiguration.self, from: Data(json.utf8))
    }

    private func modelConfig(imageTokenId: Int?) throws -> LFM2VLConfiguration {
        let declared = imageTokenId.map { "\"image_token_id\": \($0)," } ?? ""
        let json = """
            {
              "model_type": "lfm2_vl",
              \(declared)
              "text_config": {
                "model_type": "lfm2",
                "hidden_size": 64,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "vocab_size": 128000
              },
              "vision_config": {
                "model_type": "siglip2_vision_model",
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_hidden_layers": 2,
                "num_attention_heads": 4
              }
            }
            """
        return try JSONDecoder().decode(LFM2VLConfiguration.self, from: Data(json.utf8))
    }

    private func makeImage(width: CGFloat, height: CGFloat) -> CIImage {
        let filter = CIFilter(name: "CIConstantColorGenerator")!
        filter.setValue(CIColor(red: 0.5, green: 0.5, blue: 0.5), forKey: "inputColor")
        return filter.outputImage!.cropped(to: CGRect(x: 0, y: 0, width: width, height: height))
    }

    private func imageInput() -> UserInput {
        UserInput(prompt: "describe", images: [.ciImage(makeImage(width: 512, height: 512))])
    }

    // MARK: - Configuration

    func testProcessorConfigurationDefaultsToTheTemplatePlaceholder() throws {
        XCTAssertEqual(LFM2VLProcessorConfiguration.defaultImageToken, "<image>")
        XCTAssertEqual(try processorConfig().imageToken, "<image>")
    }

    func testProcessorConfigurationHonorsAnImageTokenOverride() throws {
        let config = try processorConfig(#"{"image_token": "<|image|>"}"#)
        XCTAssertEqual(config.imageToken, "<|image|>")
    }

    /// LFM2-VL-1.6B ships no `image_token_id`, so the config default must stay
    /// at that checkpoint's real id.
    func testImageTokenIndexDefaultsToTheLFM2VLId() throws {
        let config = try modelConfig(imageTokenId: nil)
        XCTAssertEqual(LFM2VLConfiguration.defaultImageTokenId, 396)
        XCTAssertEqual(config.imageTokenIndex, LFM2VLConfiguration.defaultImageTokenId)
    }

    func testImageTokenIndexHonorsTheConfiguredId() throws {
        XCTAssertEqual(try modelConfig(imageTokenId: 124_907).imageTokenIndex, 124_907)
    }

    // MARK: - Placeholder expansion

    /// The reported bug. On a checkpoint whose placeholder is not 396 the
    /// expansion must still produce exactly one marker per image feature, using
    /// the id the tokenizer reports.
    func testPrepareExpandsThePlaceholderUsingTheTokenizerId() async throws {
        let imageTokenId = 124_907
        let config = try processorConfig()
        let processor = LFM2VLProcessor(
            config,
            tokenizer: StubTokenizer(
                ids: ["<image>": imageTokenId],
                promptTokens: [1, imageTokenId, 2]))

        let result = try await processor.prepare(input: imageInput())

        let frames = try XCTUnwrap(result.image?.frames)
        let downsample = config.downsampleFactor
        let expected = frames.reduce(0) { $0 + ($1.h / downsample) * ($1.w / downsample) }
        XCTAssertGreaterThan(expected, 1, "the fixture image must need a real patch grid")

        let tokens = result.text.tokens.asArray(Int.self)
        XCTAssertEqual(
            tokens.filter { $0 == imageTokenId }.count, expected,
            "the placeholder must expand to one marker per image feature")
        XCTAssertEqual(
            tokens.filter { $0 == LFM2VLConfiguration.defaultImageTokenId }.count, 0,
            "396 is not this checkpoint's image token and must not be emitted")
        XCTAssertEqual(tokens.first, 1, "surrounding prompt tokens must be preserved")
        XCTAssertEqual(tokens.last, 2, "surrounding prompt tokens must be preserved")
    }

    /// A tokenizer with no placeholder entry must fail closed. Guessing 396 is
    /// unsafe: in the LFM2.5 vocabulary 396 is the ordinary BPE token "ab", so a
    /// fallback would match prose instead of reporting the problem.
    func testPrepareThrowsWhenTheTokenizerHasNoImageToken() async throws {
        let processor = LFM2VLProcessor(
            try processorConfig(),
            tokenizer: StubTokenizer(ids: [:], promptTokens: [1, 2]))

        do {
            _ = try await processor.prepare(input: imageInput())
            XCTFail("prepare must throw when the tokenizer has no image token")
        } catch let error as VLMError {
            guard case .processing(let message) = error else {
                return XCTFail("expected VLMError.processing, got \(error)")
            }
            XCTAssertTrue(
                message.contains("<image>"),
                "the error must name the missing token, got: \(message)")
        }
    }

    /// The override path: a checkpoint that spells the placeholder differently
    /// resolves through its processor config, with no code change.
    func testPrepareHonorsAConfiguredPlaceholderSpelling() async throws {
        let imageTokenId = 777
        let processor = LFM2VLProcessor(
            try processorConfig(#"{"image_token": "<|image|>"}"#),
            tokenizer: StubTokenizer(
                ids: ["<|image|>": imageTokenId],
                promptTokens: [1, imageTokenId, 2]))

        let result = try await processor.prepare(input: imageInput())
        let tokens = result.text.tokens.asArray(Int.self)

        XCTAssertGreaterThan(
            tokens.filter { $0 == imageTokenId }.count, 1,
            "the configured placeholder must be expanded")
    }
}
