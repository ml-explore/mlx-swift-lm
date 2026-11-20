import CoreImage
import CoreMedia
import Foundation
import MLX
import MLXLMCommon
import XCTest

@testable import MLXVLM

final class LFM3OmniProcessorTests: XCTestCase {
    private struct TestTokenizer: Tokenizer {
        let tokens: [Int]
        var bosToken: String? { nil }
        var eosToken: String? { nil }
        var unknownToken: String? { nil }
        func encode(text: String, addSpecialTokens: Bool) -> [Int] {
            [13] + Array(repeating: 11, count: text.components(separatedBy: "<|video|>").count - 1)
                + [14]
        }
        func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { "" }
        func convertTokenToId(_ token: String) -> Int? {
            [
                "<|image|>": 10, "<|video|>": 11, "<|audio|>": 12, "<|vision_start|>": 13,
                "<|vision_end|>": 14,
            ][token]
        }
        func convertIdToToken(_ id: Int) -> String? { nil }
        func applyChatTemplate(
            messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
            additionalContext: [String: any Sendable]?
        ) throws -> [Int] { tokens }
    }

    private func processor(_ tokens: [Int]) throws -> LFM3OmniProcessor {
        let json = """
            {"image_processor":{"patch_size":16,"temporal_patch_size":2,"merge_size":2,
              "image_mean":[0.5,0.5,0.5],"image_std":[0.5,0.5,0.5],
              "size":{"shortest_edge":1024,"longest_edge":1024}},
             "video_processor":{"fps":2,"min_frames":4,"max_frames":16,
              "size":{"shortest_edge":1024,"longest_edge":4096}},
             "audio_token":"<|audio|>","feature_extractor":{"sampling_rate":16000,
              "feature_size":128,"n_fft":400,"win_length":400,"hop_length":160,
              "encoder_stride":8,"log_floor":-12,"normalization_offset":4,"normalization_divisor":4}}
            """
        return LFM3OmniProcessor(
            try JSONDecoder().decode(
                LFM3OmniProcessorConfiguration.self,
                from: Data(json.utf8)), tokenizer: TestTokenizer(tokens: tokens))
    }

    func testAudioPaddingAndMarkerExpansion() async throws {
        var input = UserInput(
            prompt: "listen",
            audios: [.array(MLXArray.zeros([400])), .array(MLXArray.zeros([1680]))])
        input.processing.audio.sampleRate = 16000
        let result = try await processor([1, 13, 12, 14, 12, 2]).prepare(input: input)
        XCTAssertEqual(result.text.tokens.asArray(Int.self), [1, 13, 12, 14, 13, 12, 12, 14, 2])
        XCTAssertEqual(result.audio?.features.shape, [2, 128, 9])
        XCTAssertEqual(result.audio?.mask?.asType(.int32).sum(axis: 1).asArray(Int.self), [1, 9])
    }

    func testSoundtrackSplitsAtFrameGroupStarts() async throws {
        let image = CIImage(color: CIColor(red: 1, green: 0, blue: 0)).cropped(
            to: CGRect(x: 0, y: 0, width: 32, height: 32))
        let frames = (0 ..< 4).map {
            VideoFrame(
                image: .ciImage(image),
                timeStamp: .init(seconds: Double($0) / 2, preferredTimescale: 600))
        }
        var input = UserInput(
            prompt: "describe", videos: [.frames(frames)],
            audios: [.array(MLXArray.zeros([32000])), .array(MLXArray.zeros([400]))],
            additionalContext: ["use_audio_in_video": true])
        input.processing.audio.sampleRate = 16000
        input.processing.video = .init(targetFrames: 4)
        let result = try await processor([1, 13, 11, 14, 13, 12, 14, 13, 12, 14, 2]).prepare(
            input: input)
        let expected =
            [1, 13, 13, 11, 14] + Array(repeating: 12, count: 12)
            + [13, 11, 14] + Array(repeating: 12, count: 13) + [14, 13, 12, 14, 2]
        XCTAssertEqual(result.text.tokens.asArray(Int.self), expected)
        XCTAssertEqual(result.audio?.mask?.asType(.int32).sum(axis: 1).asArray(Int.self), [199, 1])
    }

    func testThinkingAndMissingAudioAreRejected() async throws {
        for input in [
            UserInput(prompt: "hello", additionalContext: ["enable_thinking": true]),
            UserInput(prompt: "listen"),
        ] {
            do {
                _ = try await processor([12]).prepare(input: input)
                XCTFail("Expected invalid Omni input to fail")
            } catch { XCTAssertTrue(error is VLMError) }
        }
    }
}
