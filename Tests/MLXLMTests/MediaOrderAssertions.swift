// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import XCTest

/// Qwen-style VLMs encode every image before every video, while a prompt can place a video
/// first (a video in one turn, an image in a later one). Each placeholder run must still get
/// its own media's features.
///
/// Uses the token ids and vision geometry the tiny Qwen VL fixtures share: image 500, video 501,
/// vision start 502, patch 16, temporal patch 2, merge 2. The model must give video tokens
/// M-RoPE positions; Qwen2-VL and Qwen2.5-VL give them text positions, so they cannot use it.
struct MediaOrderAssertions {
    private let imageTokenId: Int32 = 500
    private let videoTokenId: Int32 = 501
    private let visionStartTokenId: Int32 = 502

    /// A video before an image must match the same prompt with the two runs swapped: the same
    /// pixels sit in the same places, only which run is called the video changes. One grid of
    /// THW (1, 4, 4) has the same M-RoPE positions as an image or as a video.
    func assertVideoBeforeImageMatchesImageBeforeVideo<M: LanguageModel>(
        _ model: M, windowedStepSize: Int = 4, file: StaticString = #filePath, line: UInt = #line
    ) throws {
        try withRandomState(MLXRandom.RandomState(seed: 61)) {
            let first = MLXRandom.normal([16, 3 * 2 * 16 * 16])
            let second = MLXRandom.normal([16, 3 * 2 * 16 * 16])
            let frames = [THW(1, 4, 4)]

            func prompt(_ firstToken: Int32, _ secondToken: Int32) -> MLXArray {
                let text: [Int32] = [7, 20, 33, 46]
                let firstRun = [visionStartTokenId] + [Int32](repeating: firstToken, count: 4)
                let secondRun = [visionStartTokenId] + [Int32](repeating: secondToken, count: 4)
                return MLXArray(text + firstRun + text + secondRun + text)
                    .expandedDimensions(axis: 0)
            }

            func lastLogits(
                _ tokens: MLXArray, image: MLXArray, video: MLXArray, prefill: PrefillParameters
            ) throws -> MLXArray {
                let input = LMInput(
                    text: .init(tokens: tokens), image: .init(pixels: image, frames: frames),
                    video: .init(pixels: video, frames: frames))
                guard
                    case .logits(let output) = try model.prepare(
                        input, cache: try model.newCache(parameters: nil), state: nil,
                        prefill: prefill)
                else { throw XCTSkip("expected .logits from prepare") }
                return output.logits[0..., -1, 0...]
            }

            let imageFirst = try lastLogits(
                prompt(imageTokenId, videoTokenId), image: first, video: second,
                prefill: .init(chunking: .unchunked))
            let videoFirst = try lastLogits(
                prompt(videoTokenId, imageTokenId), image: second, video: first,
                prefill: .init(chunking: .unchunked))
            let videoFirstWindowed = try lastLogits(
                prompt(videoTokenId, imageTokenId), image: second, video: first,
                prefill: .init(stepSize: windowedStepSize))

            XCTAssertLessThanOrEqual(
                abs(videoFirst - imageFirst).max().item(Float.self),
                MatmulPrecision.splitTolerance,
                "a video before an image took the wrong features", file: file, line: line)
            XCTAssertLessThanOrEqual(
                abs(videoFirstWindowed - imageFirst).max().item(Float.self),
                MatmulPrecision.splitTolerance,
                "a windowed prefill with a video before an image took the wrong features",
                file: file, line: line)
        }
    }
}
