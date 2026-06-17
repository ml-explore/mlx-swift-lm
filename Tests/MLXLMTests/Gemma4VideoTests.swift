// Copyright © 2026 Apple Inc.
//
// Covers the public surface introduced by the Gemma 4 video-tower work:
// processor configuration decode (video-related fields and their defaults)
// and the message-generator placeholder expansion.

import Foundation
import MLXLMCommon
import MLXVLM
import XCTest

final class Gemma4VideoTests: XCTestCase {

    private static let baseConfigFields = """
        "processor_class": "Gemma4Processor",
        "do_normalize": true,
        "image_mean": [0.5, 0.5, 0.5],
        "image_std": [0.5, 0.5, 0.5],
        "image_seq_length": 280,
        "size": {"width": 800, "height": 800},
        "image_token_id": 258880,
        "boi_token_id": 255999,
        "eoi_token_id": 258882
        """

    func testProcessorConfigVideoFieldsExplicit() throws {
        let json = """
            {
                \(Self.baseConfigFields),
                "video_token_id": 258881,
                "video_seq_length": 32,
                "video_frame_size": {"width": 256, "height": 256},
                "video_max_frames": 8,
                "video_fps": 4.0
            }
            """

        let config = try JSONDecoder().decode(
            Gemma4ProcessorConfiguration.self, from: Data(json.utf8))

        XCTAssertEqual(config.videoTokenId, 258881)
        XCTAssertEqual(config.videoSeqLength, 32)
        XCTAssertEqual(config.videoMaxFrames, 8)
        XCTAssertEqual(config.videoFps, 4.0)
        XCTAssertEqual(config.videoFrameSize?.width, 256)
        XCTAssertEqual(config.videoFrameSize?.height, 256)
    }

    func testProcessorConfigVideoDefaultsApplyWhenMissing() throws {
        // Older `preprocessor_config.json` files predate the video fields. The
        // decoder must apply sensible defaults rather than failing, so legacy
        // configs keep loading and the image-only path stays usable.
        let json = """
            {
                \(Self.baseConfigFields)
            }
            """

        let config = try JSONDecoder().decode(
            Gemma4ProcessorConfiguration.self, from: Data(json.utf8))

        XCTAssertNil(config.videoTokenId)
        XCTAssertNil(config.videoFrameSize)
        XCTAssertEqual(config.videoSeqLength, 64)
        XCTAssertEqual(config.videoMaxFrames, 16)
        XCTAssertEqual(config.videoFps, 2.0)
    }

    func testMessageGeneratorEmitsOnePlaceholderPerVideoBeforeContent() throws {
        // Two `.url` video stubs — the message generator doesn't open them; it
        // only counts them to size the `<|video|>` placeholder run that the
        // processor expands at the text level after `applyChatTemplate`.
        let stub = URL(fileURLWithPath: "/tmp/stub.mov")
        let message = Chat.Message.user(
            "describe both clips",
            videos: [.url(stub), .url(stub)]
        )

        let generated = Gemma4MessageGenerator().generate(message: message)

        XCTAssertEqual(generated["role"] as? String, "user")

        let content = try XCTUnwrap(generated["content"] as? [[String: any Sendable]])
        let textBlock = try XCTUnwrap(
            content.first { ($0["type"] as? String) == "text" })
        let text = try XCTUnwrap(textBlock["text"] as? String)

        let placeholder = Gemma4Processor.videoPlaceholder
        XCTAssertEqual(text, "\(placeholder)\(placeholder)describe both clips")
    }

    func testMessageGeneratorOmitsPlaceholderWhenNoVideos() throws {
        let message = Chat.Message.user("hello, world")

        let generated = Gemma4MessageGenerator().generate(message: message)
        let content = try XCTUnwrap(generated["content"] as? [[String: any Sendable]])
        let textBlock = try XCTUnwrap(
            content.first { ($0["type"] as? String) == "text" })
        let text = try XCTUnwrap(textBlock["text"] as? String)

        XCTAssertEqual(text, "hello, world")
    }
}
