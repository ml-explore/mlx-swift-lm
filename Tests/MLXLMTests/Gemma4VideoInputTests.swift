// Copyright © 2026 Apple Inc.

import CoreImage
import CoreMedia
import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import MLXVLM

// MARK: - Video input for the base Gemma 4 (`gemma4`) VLM
//
// Gemma 4 has no separate video encoder: each frame runs through the same vision
// tower as images and is trimmed to the smaller per-frame video budget
// (`vision_soft_tokens_per_video_frame`, 70 for E4B) before scattering onto the
// `<video>` soft-token positions (`video_token_id`, 258884). These tests cover the
// wiring on the non-unified `Gemma4` class and its `Gemma4Processor`.

struct Gemma4VideoInputTests {

    /// `Gemma4Configuration` decodes `video_token_id` when present and leaves it nil
    /// otherwise (older text+vision checkpoints keep working unchanged).
    @Test("Gemma4Configuration decodes video_token_id")
    func decodesVideoTokenId() throws {
        let withVideo = try Self.decodeConfig(videoTokenLine: "\"video_token_id\": 258884,")
        #expect(withVideo.videoTokenId == 258884)
        #expect(withVideo.visionSoftTokensPerVideoFrame == 70)

        let withoutVideo = try Self.decodeConfig(videoTokenLine: "")
        #expect(withoutVideo.videoTokenId == nil)
    }

    /// With the per-frame video budget >= the vision tower's output length there is no
    /// truncation, so the same pixels fed as an image and as a video produce identical
    /// logits — proving video reuses the vision tower and scatters at `video_token_id`.
    @Test("Image and video pixels scatter identically (no truncation)")
    func imageAndVideoAgree() throws {
        let model = Gemma4(try Self.makeTinyConfig(videoSoftTokens: 4))
        eval(model)

        let imageTokenId = 100
        let videoTokenId = 101
        func prompt(_ mm: Int) -> [Int] { [5, 6, mm, mm, mm, mm, 7, 8] }
        // 8x8 with patch_size 4 → a 2x2 patch grid → 4 soft tokens per image
        // under the aspect-preserving tower (which emits patch-count tokens).
        let pixels = (MLXArray(0 ..< 192).reshaped([1, 3, 8, 8]).asType(.float32)) / 192.0

        func lastLogits(mm: Int, asVideo: Bool) throws -> MLXArray {
            let tokens = MLXArray(prompt(mm).map { Int32($0) }).expandedDimensions(axis: 0)
            let text = LMInput.Text(tokens: tokens)
            let input =
                asVideo
                ? LMInput(text: text, video: LMInput.ProcessedVideo(pixels: pixels))
                : LMInput(text: text, image: LMInput.ProcessedImage(pixels: pixels))
            let result = try model.prepare(
                input, cache: model.newCache(parameters: nil), state: nil, windowSize: 1024)
            guard case .logits(let out) = result else {
                Issue.record("Expected .logits from Gemma4.prepare (multimodal branch)")
                return MLXArray(0)
            }
            let logits = out.logits[0..., -1, 0...]
            eval(logits)
            return logits
        }

        let imageLogits = try lastLogits(mm: imageTokenId, asVideo: false)
        let videoLogits = try lastLogits(mm: videoTokenId, asVideo: true)
        #expect(imageLogits.shape == [1, 200])
        #expect(
            allClose(imageLogits, videoLogits, rtol: 1e-5, atol: 1e-5).item(Bool.self),
            "Video pixels must scatter through the vision tower identically to image pixels.")
    }

    /// A multi-frame video is trimmed to `visionSoftTokensPerVideoFrame` tokens per
    /// frame, so a 2-frame clip with budget 2 scatters exactly 4 tokens.
    @Test("Multi-frame video truncates to the per-frame budget")
    func videoTruncatesPerFrame() throws {
        let softPerFrame = 2
        let numFrames = 2
        let videoTokenId = 101
        let model = Gemma4(try Self.makeTinyConfig(videoSoftTokens: softPerFrame))
        eval(model)

        var tokens: [Int32] = [5, 6]
        tokens += Array(repeating: Int32(videoTokenId), count: numFrames * softPerFrame)
        tokens += [7]
        let text = LMInput.Text(tokens: MLXArray(tokens).expandedDimensions(axis: 0))
        // 8x8 frames → 4 tower tokens per frame, truncated to the budget of 2.
        let pixels =
            (MLXArray(0 ..< (numFrames * 3 * 8 * 8)).reshaped([numFrames, 3, 8, 8])
                .asType(.float32)) / 400.0
        let input = LMInput(text: text, video: LMInput.ProcessedVideo(pixels: pixels))

        let result = try model.prepare(
            input, cache: model.newCache(parameters: nil), state: nil, windowSize: 1024)
        guard case .logits(let out) = result else {
            Issue.record("Expected .logits from Gemma4.prepare")
            return
        }
        eval(out.logits)
        #expect(out.logits.shape == [1, tokens.count, 200])
    }

    /// `Gemma4Processor.processVideos` samples/preprocesses frames into a
    /// `[totalFrames, C, H, W]` tensor at the video frame size, with per-video counts
    /// that sum to the frame axis.
    @Test("Processor extracts video frames to [F, C, 432, 432]")
    func processorExtractsFrames() async throws {
        let processor = Gemma4Processor(
            try Self.makeProcessorConfig(), tokenizer: VideoTestTokenizer())

        // Four synthetic 64x64 frames at 1s spacing.
        let frames = (0 ..< 4).map { i in
            UserInput.VideoFrame(
                frame: CIImage(color: CIColor(red: 0.3, green: 0.5, blue: 0.7))
                    .cropped(to: CGRect(x: 0, y: 0, width: 64, height: 64)),
                timeStamp: CMTime(value: Int64(i), timescale: 1))
        }
        let (pixels, frameCounts) = try await processor.processVideos(
            [.frames(frames)], processing: nil)
        eval(pixels)

        #expect(frameCounts.count == 1)
        #expect(pixels.dim(0) == frameCounts.reduce(0, +))
        #expect(pixels.dim(0) >= 1)
        #expect(Array(pixels.shape.dropFirst()) == [3, 432, 432])
    }

    // MARK: - Helpers

    private static func makeTinyConfig(videoSoftTokens: Int) throws -> Gemma4Configuration {
        try decodeConfig(
            videoTokenLine: "\"video_token_id\": 101,",
            extraLines: "\"vision_soft_tokens_per_video_frame\": \(videoSoftTokens),")
    }

    /// Tiny `gemma4` config. `pooling_kernel_size 1` + `default_output_length 4` make the
    /// vision tower emit 4 soft tokens per frame regardless of patch count.
    private static func decodeConfig(videoTokenLine: String, extraLines: String = "") throws
        -> Gemma4Configuration
    {
        let json = """
            {
              "model_type": "gemma4",
              "image_token_id": 100,
              \(videoTokenLine)
              \(extraLines)
              "text_config": {
                "hidden_size": 32, "num_hidden_layers": 2, "intermediate_size": 64,
                "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 16,
                "global_head_dim": 32, "vocab_size": 200, "vocab_size_per_layer_input": 200,
                "num_kv_shared_layers": 0, "hidden_size_per_layer_input": 8,
                "sliding_window": 16, "sliding_window_pattern": 2, "max_position_embeddings": 512
              },
              "vision_config": {
                "num_hidden_layers": 1, "hidden_size": 16, "intermediate_size": 32,
                "num_attention_heads": 2, "head_dim": 8, "patch_size": 4,
                "default_output_length": 4, "pooling_kernel_size": 1, "position_embedding_size": 8
              }
            }
            """
        return try JSONDecoder().decode(Gemma4Configuration.self, from: Data(json.utf8))
    }

    private static func makeProcessorConfig() throws -> Gemma4ProcessorConfiguration {
        let json = """
            {
              "processor_class": "Gemma4Processor",
              "do_normalize": true,
              "image_seq_length": 280,
              "image_token_id": 258880,
              "video_token_id": 258884,
              "video_soft_tokens_per_frame": 70
            }
            """
        return try JSONDecoder().decode(Gemma4ProcessorConfiguration.self, from: Data(json.utf8))
    }
}

/// Minimal tokenizer so `Gemma4Processor` can be constructed; `processVideos` never
/// touches it, and the pixel-shape tests don't exercise the chat template.
private struct VideoTestTokenizer: Tokenizer {
    let vocabularySize: Int = 8
    let bosToken: String? = nil
    let eosToken: String? = nil
    let eosTokenId: Int? = 1
    let unknownToken: String? = nil
    let unknownTokenId: Int? = 0

    func encode(text: String, addSpecialTokens: Bool) -> [Int] { [] }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { "" }
    func convertTokenToId(_ token: String) -> Int? { Int(token) }
    func convertIdToToken(_ id: Int) -> String? { String(id) }
    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] { [] }
}
