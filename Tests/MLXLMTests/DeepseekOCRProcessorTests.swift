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

    private func makeProcessor() throws -> DeepseekOCRProcessor {
        let config = try JSONDecoder().decode(
            DeepseekOCRProcessorConfiguration.self,
            from: Data(Self.processorConfigJSON.utf8))
        return DeepseekOCRProcessor(config, tokenizer: DeterministicTokenizer())
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
        return text.split(separator: " ").enumerated().map { 20 + $0.offset }
    }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { "" }

    func convertTokenToId(_ token: String) -> Int? {
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
