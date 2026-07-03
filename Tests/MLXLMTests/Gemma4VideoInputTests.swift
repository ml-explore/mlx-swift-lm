// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import MLXVLM

// MARK: - Video input for the base Gemma 4 (`gemma4`) VLM
//
// Gemma 4 has no separate video encoder: video frames run through the same
// vision tower as images and scatter onto the `<video>` soft-token positions
// (`video_token_id`, 258884 for E4B). These tests cover the wiring added to the
// non-unified `Gemma4` class — decoding `video_token_id`, and routing
// `input.video?.pixels` through the shared vision path.

struct Gemma4VideoInputTests {

    /// `Gemma4Configuration` decodes `video_token_id` when present and leaves it
    /// nil otherwise (older text+vision checkpoints keep working unchanged).
    @Test("Gemma4Configuration decodes video_token_id")
    func decodesVideoTokenId() throws {
        let withVideo = try Self.decodeConfig(videoTokenLine: "\"video_token_id\": 258884,")
        #expect(withVideo.videoTokenId == 258884)

        let withoutVideo = try Self.decodeConfig(videoTokenLine: "")
        #expect(withoutVideo.videoTokenId == nil)
    }

    /// The same pixels fed as an image and as a video must produce identical
    /// logits: video reuses the vision tower and scatters at `video_token_id`
    /// exactly as the image path scatters at `image_token_id`. Also exercises
    /// the per-layer-input mask, which must exclude video soft tokens.
    @Test("Image and video pixels scatter identically")
    func imageAndVideoAgree() throws {
        let model = Gemma4(try Self.makeTinyConfig())
        eval(model)

        // 4 multimodal soft tokens == the tiny vision tower's output length
        // (default_output_length 4, pooling_kernel_size 1).
        let imageTokenId = 100
        let videoTokenId = 101
        func prompt(_ mm: Int) -> [Int] { [5, 6, mm, mm, mm, mm, 7, 8] }
        // Deterministic [1, 3, 4, 4] pixels (B, C, H, W); patch_size 4 → 1 patch.
        let pixels = (MLXArray(0 ..< 48).reshaped([1, 3, 4, 4]).asType(.float32)) / 48.0

        func lastLogits(mm: Int, asVideo: Bool) throws -> MLXArray {
            let tokens = MLXArray(prompt(mm).map { Int32($0) }).expandedDimensions(axis: 0)
            let text = LMInput.Text(tokens: tokens)
            let input =
                asVideo
                ? LMInput(text: text, video: LMInput.ProcessedVideo(pixels: pixels))
                : LMInput(text: text, image: LMInput.ProcessedImage(pixels: pixels))
            let result = try model.prepare(input, cache: model.newCache(parameters: nil),
                windowSize: 1024)
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

    // MARK: - Helpers

    private static func makeTinyConfig() throws -> Gemma4Configuration {
        try decodeConfig(videoTokenLine: "\"video_token_id\": 101,")
    }

    /// Tiny `gemma4` config. `pooling_kernel_size 1` makes the vision tower emit
    /// exactly `default_output_length` (4) soft tokens regardless of patch count.
    private static func decodeConfig(videoTokenLine: String) throws -> Gemma4Configuration {
        let json = """
            {
              "model_type": "gemma4",
              "image_token_id": 100,
              \(videoTokenLine)
              "text_config": {
                "hidden_size": 32,
                "num_hidden_layers": 2,
                "intermediate_size": 64,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 16,
                "global_head_dim": 32,
                "vocab_size": 200,
                "vocab_size_per_layer_input": 200,
                "num_kv_shared_layers": 0,
                "hidden_size_per_layer_input": 8,
                "sliding_window": 16,
                "sliding_window_pattern": 2,
                "max_position_embeddings": 512
              },
              "vision_config": {
                "num_hidden_layers": 1,
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_attention_heads": 2,
                "head_dim": 8,
                "patch_size": 4,
                "default_output_length": 4,
                "pooling_kernel_size": 1,
                "position_embedding_size": 8
              }
            }
            """
        return try JSONDecoder().decode(Gemma4Configuration.self, from: Data(json.utf8))
    }
}
