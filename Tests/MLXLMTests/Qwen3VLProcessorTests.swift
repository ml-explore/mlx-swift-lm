// Copyright © 2026 Apple Inc.

import CoreImage
import Foundation
import MLX
import MLXLMCommon
import XCTest

@testable import MLXVLM

final class Qwen3VLProcessorTests: XCTestCase {

    private struct ProcessorTokenizer: Tokenizer {
        let promptTokens: [Int]

        var bosToken: String? = nil
        var eosToken: String? = nil
        var unknownToken: String? = nil

        func encode(text: String, addSpecialTokens: Bool) -> [Int] {
            if text.contains("<|vision_start|>") {
                return [90, 91, 92]
            }
            return [1, 2, 3]
        }

        func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { "" }
        func convertTokenToId(_ token: String) -> Int? { nil }
        func convertIdToToken(_ id: Int) -> String? { nil }

        func applyChatTemplate(
            messages: [[String: any Sendable]],
            tools: [[String: any Sendable]]?,
            additionalContext: [String: any Sendable]?
        ) throws -> [Int] {
            promptTokens
        }
    }

    private func makeProcessor(promptTokens: [Int]) throws -> Qwen3VLProcessor {
        let json = """
            {
              "image_mean": [0.5, 0.5, 0.5],
              "image_std": [0.5, 0.5, 0.5],
              "min_pixels": 1024,
              "max_pixels": 1024,
              "merge_size": 2,
              "patch_size": 16,
              "temporal_patch_size": 2,
              "image_processor_type": "Qwen2VLImageProcessor"
            }
            """
        let config = try JSONDecoder().decode(
            Qwen3VLProcessorConfiguration.self, from: Data(json.utf8))
        return Qwen3VLProcessor(
            config, tokenizer: ProcessorTokenizer(promptTokens: promptTokens))
    }

    func testTextOnlyInputDoesNotCarryRedundantAttentionMask() async throws {
        let processor = try makeProcessor(promptTokens: [1, 2, 3])

        let prepared = try await processor.prepare(input: UserInput(prompt: "hello"))

        XCTAssertEqual(prepared.text.tokens.shape, [1, 3])
        XCTAssertNil(prepared.text.mask)
        XCTAssertNil(prepared.image)
        XCTAssertNil(prepared.video)
    }

    /// No Qwen VL model reads the mask, and a mask stops `ChatSession` from reusing the
    /// cache, so a follow-up to an image would prefill the whole conversation again.
    func testImageInputCarriesNoAttentionMaskAndKeepsMediaPayload() async throws {
        let json = Data(
            """
            {
              "image_mean": [0.5, 0.5, 0.5],
              "image_std": [0.5, 0.5, 0.5],
              "min_pixels": 1024,
              "max_pixels": 1024,
              "merge_size": 2,
              "patch_size": 16,
              "temporal_patch_size": 2,
              "image_processor_type": "Qwen2VLImageProcessor"
            }
            """.utf8)
        let tokenizer = ProcessorTokenizer(promptTokens: [90, 91, 92])
        let processors: [(String, any UserInputProcessor)] = [
            (
                "Qwen2VL",
                Qwen2VLProcessor(
                    try JSONDecoder().decode(Qwen2VLProcessorConfiguration.self, from: json),
                    tokenizer: tokenizer)
            ),
            (
                "Qwen25VL",
                Qwen25VLProcessor(
                    try JSONDecoder().decode(Qwen25VLProcessorConfiguration.self, from: json),
                    tokenizer: tokenizer)
            ),
            (
                "Qwen3VL",
                Qwen3VLProcessor(
                    try JSONDecoder().decode(Qwen3VLProcessorConfiguration.self, from: json),
                    tokenizer: tokenizer)
            ),
        ]
        let image = CIImage(color: .black).cropped(
            to: CGRect(x: 0, y: 0, width: 32, height: 32))
        var input = UserInput(prompt: "describe", images: [.ciImage(image)])
        input.processing = .init(minPixels: 1024, maxPixels: 1024)

        for (name, processor) in processors {
            let prepared = try await processor.prepare(input: input)

            XCTAssertNil(prepared.text.mask, name)
            XCTAssertNotNil(prepared.image, name)
            XCTAssertNil(prepared.video, name)
        }
    }
}
