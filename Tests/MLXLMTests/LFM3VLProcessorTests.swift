import CoreImage
import CoreMedia
import Foundation
import MLX
import MLXLMCommon
import XCTest

@testable import MLXVLM

final class LFM3VLProcessorTests: XCTestCase {
    private struct TestTokenizer: Tokenizer {
        let tokens: [Int]
        var bosToken: String? { nil }
        var eosToken: String? { nil }
        var unknownToken: String? { nil }
        func encode(text: String, addSpecialTokens: Bool) -> [Int] {
            Array(repeating: 11, count: text.components(separatedBy: "<|video|>").count - 1)
        }
        func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { "" }
        func convertTokenToId(_ token: String) -> Int? { token == "<|image|>" ? 10 : 11 }
        func convertIdToToken(_ id: Int) -> String? { nil }
        func applyChatTemplate(
            messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
            additionalContext: [String: any Sendable]?
        ) throws -> [Int] { tokens }
    }

    private func processor(_ tokens: [Int]) throws -> LFM3VLProcessor {
        let json = """
            {"image_processor": {"patch_size":16,"temporal_patch_size":2,"merge_size":2,
              "image_mean":[0.5,0.5,0.5],"image_std":[0.5,0.5,0.5],
              "size":{"shortest_edge":1024,"longest_edge":1024}},
             "video_processor":{"fps":2,"min_frames":4,"max_frames":16,
              "size":{"shortest_edge":1024,"longest_edge":4096}}}
            """
        return LFM3VLProcessor(
            try JSONDecoder().decode(
                LFM3VLProcessorConfiguration.self,
                from: Data(json.utf8)), tokenizer: TestTokenizer(tokens: tokens))
    }
    private var image: CIImage {
        CIImage(color: CIColor(red: 1, green: 0, blue: 0)).cropped(
            to: CGRect(x: 0, y: 0, width: 32, height: 32))
    }
    func testImagePatchChannelsAndExpansion() async throws {
        let input = try await processor([1, 10, 2]).prepare(
            input: UserInput(prompt: "describe", images: [.ciImage(image)]))
        let pixels = try XCTUnwrap(input.image?.pixels)
        XCTAssertEqual(pixels.shape, [4, 1536])
        XCTAssertEqual(input.text.tokens.asArray(Int.self), [1, 10, 2])
        let channels = pixels.reshaped(4, 3, 2, 16, 16).mean(axes: [0, 2, 3, 4]).asArray(Float.self)
        XCTAssertEqual(channels[0], 1, accuracy: 1e-5)
        XCTAssertEqual(channels[1], -1, accuracy: 1e-5)
        XCTAssertEqual(channels[2], -1, accuracy: 1e-5)
    }
    func testTextOnlyHasNoMedia() async throws {
        let input = try await processor([1, 2]).prepare(input: UserInput(prompt: "hello"))
        XCTAssertEqual(input.text.tokens.shape, [1, 2])
        XCTAssertNil(input.image)
        XCTAssertNil(input.video)
    }
    func testUnusedImageIsRejected() async throws {
        do {
            _ = try await processor([1]).prepare(
                input: UserInput(prompt: "describe", images: [.ciImage(image)]))
            XCTFail("Expected an unused image error")
        } catch { XCTAssertTrue(error is VLMError) }
    }
    func testVideoTemporalGroups() async throws {
        let frames = (0 ..< 4).map {
            VideoFrame(
                image: .ciImage(image),
                timeStamp: .init(seconds: Double($0), preferredTimescale: 600))
        }
        var input = UserInput(prompt: "describe", videos: [.frames(frames)])
        input.processing.video = .init(targetFrames: 4)
        let prepared = try await processor([1, 11, 2]).prepare(input: input)
        XCTAssertEqual(prepared.video?.frames?.first?.t, 2)
        XCTAssertEqual(prepared.video?.pixels.shape, [8, 1536])
        XCTAssertEqual(prepared.text.tokens.asArray(Int.self), [1, 11, 11, 2])
    }
    func testResizeMatchesReferenceRoundingAndSmallImages() throws {
        let rounded = try LFM3VLProcessor.targetSize(
            height: 80, width: 80, factor: 32,
            minPixels: 1024, maxPixels: 16384)
        XCTAssertEqual(rounded.0, 64)
        XCTAssertEqual(rounded.1, 64)
        let tiny = try LFM3VLProcessor.targetSize(
            height: 1, width: 1, factor: 32,
            minPixels: 1024, maxPixels: 16384)
        XCTAssertEqual(tiny.0, 32)
        XCTAssertEqual(tiny.1, 32)
        let oddVideo = try LFM3VLProcessor.targetSize(
            height: 64, width: 64, factor: 32,
            frames: 3, temporalFactor: 2, minPixels: 14000, maxPixels: 20000)
        XCTAssertEqual(oddVideo.0, 64)
        XCTAssertEqual(oddVideo.1, 64)
        XCTAssertThrowsError(
            try LFM3VLProcessor.targetSize(
                height: 0, width: 32, factor: 32,
                minPixels: 1024, maxPixels: 16384))
    }

}
