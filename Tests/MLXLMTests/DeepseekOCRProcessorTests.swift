// Copyright © 2026 Apple Inc.

import CoreImage
import Foundation
import MLX
import MLXLMCommon
import XCTest

@_spi(Testing) @testable import MLXVLM

final class DeepseekOCRProcessorTests: XCTestCase {

    func testGundamModeMatchesExpectedCropMetadataAndTokenMask() async throws {
        let processor = try makeProcessor()
        let input = UserInput(
            prompt: "Describe this page.",
            images: [.ciImage(makeSolidImage(width: 800, height: 400, color: .red))])

        let prepared = try await processor.prepareForTesting(input: input)

        XCTAssertEqual(prepared.mode, .gundam)
        XCTAssertEqual(prepared.pixelValues.shape, [1, 3, 1024, 1024])
        XCTAssertEqual(prepared.localCrops.shape, [2, 3, 640, 640])
        XCTAssertEqual(prepared.imagesSpatialCrop.asArray(Int32.self), [2, 1])
        XCTAssertEqual(prepared.imagesSeqMask.asType(.int32).sum().item(Int.self), 483)
        // BOS + image lattice + raw user text (matches Python tokenize_with_images).
        XCTAssertEqual(prepared.inputIds.shape[0], 1)
        XCTAssertEqual(prepared.inputIds[0, 0].item(Int.self), 0)
        XCTAssertEqual(prepared.imagesSeqMask[0, 0].item(Bool.self), false)
        XCTAssertGreaterThanOrEqual(prepared.inputIds.shape[1], 484)

        let globalPixels = prepared.pixelValues.asType(.float32)
        let localPixels = prepared.localCrops.asType(.float32)
        XCTAssertLessThan(globalPixels.sum().item(Float.self), -100_000)
        XCTAssertLessThan(localPixels.sum().item(Float.self), -500_000)
    }

    func testBaseModeIsSelectableAndUsesSingleViewTokenGrid() async throws {
        let processor = try makeProcessor()
        let input = UserInput(
            prompt: "Describe this page.",
            images: [.ciImage(makeSolidImage(width: 800, height: 400, color: .red))],
            additionalContext: DeepseekOCRProcessor.modeContext(.base))

        let prepared = try await processor.prepareForTesting(input: input)

        XCTAssertEqual(prepared.mode, .base)
        XCTAssertEqual(prepared.pixelValues.shape, [1, 3, 640, 640])
        XCTAssertEqual(prepared.imagesSpatialCrop.asArray(Int32.self), [1, 1])
        XCTAssertEqual(prepared.localCrops.shape, [1, 3, 1024, 1024])
        XCTAssertEqual(prepared.imagesSeqMask.asType(.int32).sum().item(Int.self), 111)
        XCTAssertEqual(prepared.inputIds.shape[0], 1)
        XCTAssertEqual(prepared.inputIds[0, 0].item(Int.self), 0)
        XCTAssertGreaterThanOrEqual(prepared.inputIds.shape[1], 112)
    }

    func testPrepareBaseModeOmitsLocalCropsFromLMInput() async throws {
        let processor = try makeProcessor()
        let input = UserInput(
            prompt: "document parsing. ",
            images: [.ciImage(makeSolidImage(width: 800, height: 400, color: .red))],
            additionalContext: DeepseekOCRProcessor.modeContext(.base))

        let lmInput = try await processor.prepare(input: input)

        XCTAssertEqual(lmInput.image?.pixels.shape, [1, 3, 640, 640])
        XCTAssertEqual(lmInput.image?.positionIds?.asArray(Int32.self), [1, 1])
        XCTAssertNil(lmInput.video, "base mode must not pack gundam local crops into video")
    }

    func testPrepareGundamModePacksLocalCropsIntoVideo() async throws {
        let processor = try makeProcessor()
        let input = UserInput(
            prompt: "document parsing. ",
            images: [.ciImage(makeSolidImage(width: 800, height: 400, color: .red))])

        let lmInput = try await processor.prepare(input: input)

        XCTAssertEqual(lmInput.image?.pixels.shape, [1, 3, 1024, 1024])
        XCTAssertEqual(lmInput.image?.positionIds?.asArray(Int32.self), [2, 1])
        XCTAssertEqual(lmInput.video?.pixels.shape, [2, 3, 640, 640])
    }

    func testModeContextHelpers() {
        XCTAssertEqual(
            DeepseekOCRProcessor.mode(from: DeepseekOCRProcessor.modeContext(.base)), .base)
        XCTAssertEqual(
            DeepseekOCRProcessor.mode(from: DeepseekOCRProcessor.modeContext(.gundam)), .gundam)
        XCTAssertEqual(DeepseekOCRProcessor.mode(from: nil), .gundam)
        XCTAssertEqual(
            DeepseekOCRProcessor.mode(from: [DeepseekOCRProcessor.modeContextKey: "nope"]),
            .gundam)
        XCTAssertEqual(
            Set(DeepseekOCRProcessor.Mode.allCases.map(\.rawValue)),
            Set(["gundam", "base"]))
    }

    func testGroundingSpecialTokenStringsAndDefaultIds() {
        XCTAssertEqual(
            Set(DeepseekOCRSpecialTokens.allStrings),
            Set([
                "<|grounding|>", "<|ref|>", "<|/ref|>", "<|det|>", "<|/det|>",
            ]))
        XCTAssertEqual(DeepseekOCRSpecialTokens.refOpen.defaultId, 128_816)
        XCTAssertEqual(DeepseekOCRSpecialTokens.refClose.defaultId, 128_817)
        XCTAssertEqual(DeepseekOCRSpecialTokens.detOpen.defaultId, 128_818)
        XCTAssertEqual(DeepseekOCRSpecialTokens.detClose.defaultId, 128_819)
        XCTAssertEqual(DeepseekOCRSpecialTokens.grounding.defaultId, 128_820)
    }

    func testGroundingTokensResolveViaTokenizerPath() {
        let tokenizer = DeterministicTokenizer()
        let ids = DeepseekOCRSpecialTokens.resolveIds(tokenizer: tokenizer)
        XCTAssertEqual(ids[.grounding], 128_820)
        XCTAssertEqual(ids[.refOpen], 128_816)
        XCTAssertEqual(ids[.refClose], 128_817)
        XCTAssertEqual(ids[.detOpen], 128_818)
        XCTAssertEqual(ids[.detClose], 128_819)
        // Tokenizer lookup wins over defaults when present.
        XCTAssertEqual(
            DeepseekOCRSpecialTokens.id(of: .grounding, tokenizer: tokenizer), 128_820)
    }

    func testGroundingPromptBuildersMatchPythonShapes() {
        XCTAssertEqual(
            DeepseekOCRSpecialTokens.groundingPrompt(),
            "<|grounding|>OCR this image.")
        XCTAssertEqual(
            DeepseekOCRSpecialTokens.groundingPrompt("OCR this image"),
            "<|grounding|>OCR this image.")
        XCTAssertEqual(
            DeepseekOCRSpecialTokens.groundingMarkdownPrompt(),
            "<|grounding|>Convert the document to markdown.")
        XCTAssertEqual(
            DeepseekOCRSpecialTokens.locatePrompt("Total assets"),
            "Locate <|ref|>Total assets<|/ref|> in the image.")
    }

    func testPreparePreservesGroundingPromptTokensInInputIds() async throws {
        let processor = try makeProcessor(tokenizer: DeterministicTokenizer())
        let prompt = DeepseekOCRSpecialTokens.groundingPrompt()
        let input = UserInput(
            prompt: prompt,
            images: [.ciImage(makeSolidImage(width: 200, height: 200, color: .blue))],
            additionalContext: DeepseekOCRProcessor.modeContext(.base))

        let prepared = try await processor.prepareForTesting(input: input)
        let ids = prepared.inputIds.asArray(Int32.self).map(Int.init)
        // DeterministicTokenizer encodes known special tokens as single IDs.
        XCTAssertTrue(
            ids.contains(DeepseekOCRSpecialTokens.grounding.defaultId),
            "grounding token must survive prepare() encode path; ids=\(ids)")
    }

    func testParseDetectionsExtractsNormalizedBoxes() {
        let sample =
            #"<|/ref|><|det|>[[330, 198, 558, 230]]<|/det|>"#
            + "\n"
            + #"<|ref|>body<|/ref|><|det|>[[10,20,30,40]]<|/det|>"#
        let boxes = DeepseekOCRSpecialTokens.parseDetections(from: sample)
        XCTAssertEqual(
            boxes,
            [
                .init(x1: 330, y1: 198, x2: 558, y2: 230),
                .init(x1: 10, y1: 20, x2: 30, y2: 40),
            ])
    }

    private func makeProcessor(tokenizer: any Tokenizer = DeterministicTokenizer()) throws
        -> DeepseekOCRProcessor
    {
        let config = try JSONDecoder().decode(
            DeepseekOCRProcessorConfiguration.self,
            from: Data(Self.processorConfigJSON.utf8))
        return DeepseekOCRProcessor(config, tokenizer: tokenizer)
    }

    private func makeSolidImage(width: CGFloat, height: CGFloat, color: CIColor) -> CIImage {
        CIImage(color: color).cropped(to: CGRect(x: 0, y: 0, width: width, height: height))
    }

    private static let processorConfigJSON = #"""
        {
         "candidate_resolutions": [[1024, 1024]],
         "downsample_ratio": 4,
         "image_mean": [0.5, 0.5, 0.5],
         "image_std": [0.5, 0.5, 0.5],
         "image_token": "<image>",
         "patch_size": 16,
         "size": {
          "shortest_edge": 1024,
          "longest_edge": 1024
         }
        }
        """#
}

private struct DeterministicTokenizer: Tokenizer {
    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        guard !text.isEmpty else { return [] }
        // Split on whitespace but keep DeepSeek special tokens as atomic pieces
        // (mirrors HF tokenizer single-id encoding for <|grounding|> / <|ref|> / …).
        var ids = [Int]()
        var remaining = text[...]
        while !remaining.isEmpty {
            if remaining.first?.isWhitespace == true {
                remaining = remaining.drop(while: \.isWhitespace)
                continue
            }
            if let match = DeepseekOCRSpecialTokens.allCases.first(where: {
                remaining.hasPrefix($0.rawValue)
            }) {
                ids.append(match.defaultId)
                remaining = remaining.dropFirst(match.rawValue.count)
                continue
            }
            if remaining.hasPrefix("<image>") {
                ids.append(999)
                remaining = remaining.dropFirst("<image>".count)
                continue
            }
            // Next whitespace-delimited word → synthetic id.
            let word = remaining.prefix(while: { !$0.isWhitespace && $0 != "<" })
            if word.isEmpty {
                // Unknown '<' fragment — consume one scalar.
                ids.append(20 + ids.count)
                remaining = remaining.dropFirst()
            } else {
                ids.append(20 + ids.count)
                remaining = remaining.dropFirst(word.count)
            }
        }
        return ids
    }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { "" }

    func convertTokenToId(_ token: String) -> Int? {
        if let special = DeepseekOCRSpecialTokens(rawValue: token) {
            return special.defaultId
        }
        switch token {
        case "<image>": return 999
        case "<s>": return 0
        default: return nil
        }
    }

    func convertIdToToken(_ id: Int) -> String? { nil }

    var bosToken: String? { "<s>" }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        [0, 20, 21]
    }
}
