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
            MediaProcessing.sampledFrameCount(
                fps: 30, duration: duration, maxFrames: 3, availableFrames: 10),
            3)
        XCTAssertEqual(
            MediaProcessing.sampledFrameCount(
                fps: 30, duration: duration, maxFrames: 12, availableFrames: 10),
            10)
        XCTAssertEqual(
            MediaProcessing.sampledFrameCount(
                fps: 30, duration: duration, maxFrames: 0, availableFrames: 10),
            1)
        XCTAssertEqual(
            MediaProcessing.sampledFrameCount(
                fps: 30, duration: duration, maxFrames: 3, availableFrames: 0),
            0)
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
}
