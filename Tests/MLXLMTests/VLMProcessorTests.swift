import CoreImage
import CoreMedia
import Foundation
import MLXLMCommon
import XCTest

@testable import MLXVLM

private struct FixedTokenizer: Tokenizer {
    var encodedTokens: [Int] = [11, 12, 13]
    var decodedText: String = "User: describe"
    var tokenIds: [String: Int] = [:]
    var returnsUnknownTokenIdForMissingTokens = false
    var bosToken: String?
    var eosToken: String?
    var unknownToken: String?

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        encodedTokens + text.unicodeScalars.map { Int($0.value % 97) + 1 }
    }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        decodedText
    }

    func convertTokenToId(_ token: String) -> Int? {
        if let tokenId = tokenIds[token] {
            return tokenId
        }
        guard returnsUnknownTokenIdForMissingTokens, let unknownToken else {
            return nil
        }
        return tokenIds[unknownToken]
    }

    func convertIdToToken(_ id: Int) -> String? {
        tokenIds.first { $0.value == id }?.key
    }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        encodedTokens
    }
}

private func smolProcessorConfig(
    imageSequenceLength: Int? = 4,
    maxFrames: Int = 3,
    fps: Int = 10,
    imageEdge: Int = 8
) -> SmolVLMProcessorConfiguration {
    SmolVLMProcessorConfiguration(
        imageMean: [0, 0, 0],
        imageStd: [1, 1, 1],
        size: .init(longestEdge: imageEdge),
        maxImageSize: .init(longestEdge: imageEdge),
        videoSampling: .init(fps: fps, maxFrames: maxFrames),
        imageSequenceLength: imageSequenceLength)
}

private func smolProcessorConfigJSON(sequenceKey: String, sequenceLength: Int) -> Data {
    Data(
        """
        {
          "image_mean": [0.5, 0.5, 0.5],
          "image_std": [0.25, 0.25, 0.25],
          "size": { "longest_edge": 16 },
          "max_image_size": { "longest_edge": 8 },
          "video_sampling": { "fps": 2, "max_frames": 5 },
          "\(sequenceKey)": \(sequenceLength)
        }
        """.utf8)
}

private func glmOcrProcessorConfig() throws -> GlmOcrProcessorConfiguration {
    try JSONDecoder().decode(
        GlmOcrProcessorConfiguration.self,
        from: Data(
            """
            {
              "image_mean": [0.48145466, 0.4578275, 0.40821073],
              "image_std": [0.26862954, 0.26130258, 0.27577711],
              "merge_size": 2,
              "patch_size": 14,
              "temporal_patch_size": 2,
              "size": {"shortest_edge": 12544, "longest_edge": 9633792}
            }
            """.utf8))
}

private func testImage(edge: CGFloat = 8) -> CIImage {
    CIImage(color: CIColor(red: 0.5, green: 0.25, blue: 0.75))
        .cropped(to: CGRect(x: 0, y: 0, width: edge, height: edge))
}

private func testVideoFrames(count: Int, edge: CGFloat = 8) -> [VideoFrame] {
    (0 ..< count).map { index in
        VideoFrame(
            frame: testImage(edge: edge),
            timeStamp: CMTime(value: CMTimeValue(index), timescale: 1))
    }
}

final class VLMProcessorTests: XCTestCase {
    func testFastVLMBaseConfigurationExposesCorrectlySpelledModelMaxLength() throws {
        let data = Data(
            """
            {
              "model_type": "fastvlm",
              "image_token_index": -200,
              "eos_token_id": 151645,
              "mm_projector_type": "mlp2x_gelu",
              "mm_hidden_size": 1024,
              "tokenizer_model_max_length": 4096,
              "tokenizer_padding_side": "right"
            }
            """.utf8)

        let config = try JSONDecoder().decode(
            FastVLMConfiguration.BaseConfiguration.self, from: data)

        XCTAssertEqual(config.tokenizerModelMaxLength, 4096)
        XCTAssertEqual(config.imageTokenIndex, -200)
    }

    func testFastVLMTextTokenSplittingTrimsRightPaddingSide() throws {
        let split = try FastVLM.splitTextTokensForMultimodal(
            inputIds: [1, 2, -200, 3, 4, 5],
            imageTokenId: -200,
            imageLength: 3,
            maxLength: 6,
            paddingSide: "right")

        XCTAssertEqual(split.beforeTokens, [1, 2])
        XCTAssertEqual(split.afterTokens, [3])
    }

    func testFastVLMTextTokenSplittingTrimsLeftPaddingSide() throws {
        let split = try FastVLM.splitTextTokensForMultimodal(
            inputIds: [1, 2, -200, 3, 4, 5],
            imageTokenId: -200,
            imageLength: 3,
            maxLength: 6,
            paddingSide: "left")

        XCTAssertEqual(split.beforeTokens, [])
        XCTAssertEqual(split.afterTokens, [3, 4, 5])
    }

    func testFastVLMTextTokenSplittingRequiresImageToken() {
        XCTAssertThrowsError(
            try FastVLM.splitTextTokensForMultimodal(
                inputIds: [1, 2, 3],
                imageTokenId: -200,
                imageLength: 2,
                maxLength: 8,
                paddingSide: "right")
        ) { error in
            guard case VLMError.processing(let message) = error else {
                return XCTFail("Expected VLMError.processing, got \(error)")
            }
            XCTAssertTrue(message.contains("missing image token"))
        }
    }

    func testFastVLMTextTokenSplittingRejectsOversizedImageBlock() {
        XCTAssertThrowsError(
            try FastVLM.splitTextTokensForMultimodal(
                inputIds: [1, -200, 2],
                imageTokenId: -200,
                imageLength: 9,
                maxLength: 8,
                paddingSide: "right")
        ) { error in
            guard case VLMError.processing(let message) = error else {
                return XCTFail("Expected VLMError.processing, got \(error)")
            }
            XCTAssertTrue(message.contains("exceed tokenizer_model_max_length"))
        }
    }

    func testSmolVLMProcessorConfigurationDecodesImageSequenceLengthAliases() throws {
        let decoder = JSONDecoder()

        let imageSeqLen = try decoder.decode(
            SmolVLMProcessorConfiguration.self,
            from: smolProcessorConfigJSON(sequenceKey: "image_seq_len", sequenceLength: 81))
        let imageSequenceLength = try decoder.decode(
            SmolVLMProcessorConfiguration.self,
            from: smolProcessorConfigJSON(sequenceKey: "image_sequence_length", sequenceLength: 82))
        let imageSequenceLen = try decoder.decode(
            SmolVLMProcessorConfiguration.self,
            from: smolProcessorConfigJSON(sequenceKey: "image_sequence_len", sequenceLength: 83))

        XCTAssertEqual(imageSeqLen.imageSequenceLength, 81)
        XCTAssertEqual(imageSequenceLength.imageSequenceLength, 82)
        XCTAssertEqual(imageSequenceLen.imageSequenceLength, 83)
    }

    func testSmolVLMProcessorConfigurationRejectsNonPositiveImageSequenceLength() {
        XCTAssertThrowsError(
            try JSONDecoder().decode(
                SmolVLMProcessorConfiguration.self,
                from: smolProcessorConfigJSON(sequenceKey: "image_seq_len", sequenceLength: 0)))
    }

    func testSmolVLMMessageGeneratorUsesSmolMediaOrdering() throws {
        let input = UserInput(
            prompt: "Describe the media.",
            images: [.ciImage(testImage())],
            videos: [.frames(testVideoFrames(count: 1))])

        let message = try XCTUnwrap(SmolVLM2MessageGenerator().generate(from: input).first)
        XCTAssertEqual(message["role"] as? String, "user")
        let content = try XCTUnwrap(message["content"] as? [[String: any Sendable]])

        XCTAssertEqual(content.count, 3)
        XCTAssertEqual(content[0]["type"] as? String, "image")
        XCTAssertEqual(content[1]["type"] as? String, "video")
        XCTAssertEqual(content[2]["type"] as? String, "text")
        XCTAssertEqual(content[2]["text"] as? String, "Describe the media.")
    }

    func testSmolVLMProcessorUsesTokenizerImageTokenIdWithFallback() {
        let config = smolProcessorConfig()
        let mappedProcessor = SmolVLMProcessor(
            config, tokenizer: FixedTokenizer(tokenIds: ["<image>": 777]))
        let fallbackProcessor = SmolVLMProcessor(config, tokenizer: FixedTokenizer())
        let unknownFallbackProcessor = SmolVLMProcessor(
            config,
            tokenizer: FixedTokenizer(
                tokenIds: ["<unk>": 102],
                returnsUnknownTokenIdForMissingTokens: true,
                unknownToken: "<unk>"))

        XCTAssertEqual(mappedProcessor.imageTokenId, 777)
        XCTAssertEqual(fallbackProcessor.imageTokenId, 49190)
        XCTAssertEqual(unknownFallbackProcessor.imageTokenId, 49190)
    }

    func testSmolVLMProcessorClampsConfiguredVideoFramesToAtLeastOne() {
        let processor = SmolVLMProcessor(
            smolProcessorConfig(maxFrames: 0),
            tokenizer: FixedTokenizer())

        XCTAssertEqual(processor.maxVideoFrames, 1)
    }

    func testMediaProcessingSampledFrameCountHonorsMaxFramesForInMemoryVideoFrames() {
        let duration = CMTime(value: 9, timescale: 1)

        XCTAssertEqual(
            MediaProcessing.sampleTimes(
                duration: duration, processing: .init(), targetFPS: { _ in 30 },
                maxFrames: 3, totalAvailableFrames: 10
            ).count,
            3)
        XCTAssertEqual(
            MediaProcessing.sampleTimes(
                duration: duration, processing: .init(), targetFPS: { _ in 30 },
                maxFrames: 12, totalAvailableFrames: 10
            ).count,
            10)
        XCTAssertEqual(
            MediaProcessing.sampleTimes(
                duration: duration, processing: .init(), targetFPS: { _ in 30 },
                maxFrames: 0, totalAvailableFrames: 10
            ).count,
            1)
    }

    func testMediaProcessingLimitsInMemoryVideoFrames() async throws {
        let result = try await MediaProcessing.asProcessedSequence(
            .frames(testVideoFrames(count: 10)),
            targetFPS: { _ in 30 },
            maxFrames: 3)
        XCTAssertEqual(result.frames.count, 3)
    }

    func testMediaProcessingRejectsEmptyInMemoryVideoFrames() async {
        do {
            _ = try await MediaProcessing.asProcessedSequence(
                .frames([]),
                targetFPS: { _ in 30 },
                maxFrames: 3)
            XCTFail("Expected empty video frames to throw")
        } catch {
            guard case VLMError.processing(let message) = error else {
                return XCTFail("Expected VLMError.processing, got \(error)")
            }
            XCTAssertTrue(message.contains("at least one frame"))
        }
    }

    func testGlmOcrMessageGeneratorPutsImagesBeforeText() throws {
        let input = UserInput(prompt: "Text Recognition:", images: [.ciImage(testImage())])

        let message = try XCTUnwrap(GlmOcrMessageGenerator().generate(from: input).first)
        let content = try XCTUnwrap(message["content"] as? [[String: any Sendable]])

        XCTAssertEqual(content.count, 2)
        XCTAssertEqual(content[0]["type"] as? String, "image")
        XCTAssertEqual(content[1]["type"] as? String, "text")
        XCTAssertEqual(content[1]["text"] as? String, "Text Recognition:")
    }

    func testGlmOcrProcessorScalesUpImagesWithASideUnderFactor() throws {
        let config = try glmOcrProcessorConfig()
        let processor = GlmOcrProcessor(config, tokenizer: FixedTokenizer())
        let image = CIImage(color: CIColor(red: 0.5, green: 0.25, blue: 0.75))
            .cropped(to: CGRect(x: 0, y: 0, width: 400, height: 27))

        let (_, grid) = try processor.preprocess(images: [image], processing: nil)

        let factor = config.patchSize * config.mergeSize
        let height = grid.h * config.patchSize
        let width = grid.w * config.patchSize
        XCTAssertGreaterThanOrEqual(height, factor)
        XCTAssertEqual(height % factor, 0)
        XCTAssertEqual(width % factor, 0)
    }

    func testQwenVLTargetSizeScalesUpSmallSidesWhenAsked() throws {
        let tall = try QwenVL.targetSize(
            height: 27, width: 400, factor: 28,
            minPixels: 112 * 112, maxPixels: 28 * 28 * 15000, scaleUpSmallSides: true)
        XCTAssertEqual(tall.0, 56)
        XCTAssertEqual(tall.1, 448)

        let wide = try QwenVL.targetSize(
            height: 400, width: 27, factor: 28,
            minPixels: 112 * 112, maxPixels: 28 * 28 * 15000, scaleUpSmallSides: true)
        XCTAssertEqual(wide.0, 448)
        XCTAssertEqual(wide.1, 56)
    }

    func testQwenVLTargetSizeStillRejectsSidesUnderFactor() {
        XCTAssertThrowsError(
            try QwenVL.targetSize(
                height: 27, width: 400, factor: 28,
                minPixels: 56 * 56, maxPixels: 28 * 28 * 1280))
        XCTAssertThrowsError(
            try QwenVL.targetSize(
                height: 400, width: 27, factor: 28,
                minPixels: 56 * 56, maxPixels: 28 * 28 * 1280))
    }
}
