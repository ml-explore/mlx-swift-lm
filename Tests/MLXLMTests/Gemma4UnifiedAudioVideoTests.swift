// Copyright © 2026 Apple Inc.

import CoreImage
import CoreMedia
import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import MLXVLM

// MARK: - Audio + video for the encoder-free Gemma 4 Unified (12B)

/// Tokenizer for the unified processor tests. `applyChatTemplate` returns a
/// fixed prompt with modality placeholders; `encode` returns a single marker
/// token (9) so tests can assert where video timestamp text was spliced.
private struct UnifiedAVTestTokenizer: Tokenizer {
    var template: [Int] = [3, 30, 2]

    let vocabularySize: Int = 32
    let bosToken: String? = nil
    let eosToken: String? = nil
    let eosTokenId: Int? = 1
    let unknownToken: String? = nil
    let unknownTokenId: Int? = 0

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        [9]
    }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        tokenIds.map(String.init).joined(separator: " ")
    }

    func convertTokenToId(_ token: String) -> Int? {
        Int(token)
    }

    func convertIdToToken(_ id: Int) -> String? {
        String(id)
    }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        template
    }
}

struct Gemma4UnifiedAudioVideoTests {

    private func decodeProcessorConfig(_ json: String) throws
        -> Gemma4UnifiedProcessorConfiguration
    {
        try JSONDecoder.json5().decode(
            Gemma4UnifiedProcessorConfiguration.self, from: Data(json.utf8))
    }

    private func decodeModelConfig(_ json: String) throws -> Gemma4UnifiedConfiguration {
        try JSONDecoder.json5().decode(Gemma4UnifiedConfiguration.self, from: Data(json.utf8))
    }

    /// Processor config with the tiny ids used across these tests:
    /// audio 30, boa 27, eoa 26, video 29, boi 28, eoi 31, image 25.
    private func tinyProcessorConfig(extraLines: String = "") throws
        -> Gemma4UnifiedProcessorConfiguration
    {
        try decodeProcessorConfig(
            """
            {
              "processor_class": "Gemma4UnifiedProcessor",
              "image_token_id": 25,
              "audio_token_id": 30,
              "video_token_id": 29,
              "boi_token_id": 28,
              "eoi_token_id": 31,
              "boa_token_id": 27,
              "eoa_token_id": 26,
              "video_soft_tokens_per_frame": 4,
              \(extraLines)
              "image_processor": {
                "patch_size": 2,
                "pooling_kernel_size": 2,
                "model_patch_size": 4,
                "max_soft_tokens": 4,
                "size": { "height": 8, "width": 8 }
              }
            }
            """)
    }

    // MARK: - Configuration decoding

    @Test("Unified processor config decodes audio and video keys from processor_config.json")
    func configDecodesAudioVideoKeys() throws {
        // Mirrors the real mlx-community/gemma-4-12B-it-4bit processor_config.json:
        // audio params nested under `feature_extractor` (with vestigial mel keys),
        // eoa id spelled `eoa_token_index`, no explicit boa/video keys.
        let config = try decodeProcessorConfig(
            """
            {
              "processor_class": "Gemma4UnifiedProcessor",
              "audio_ms_per_token": 40,
              "eoa_token_index": 258883,
              "feature_extractor": {
                "feature_extractor_type": "Gemma4UnifiedAudioFeatureExtractor",
                "sampling_rate": 16000,
                "num_mel_filters": 128,
                "fft_length": 512,
                "hop_length": 160,
                "chunk_duration": 8.0,
                "overlap_duration": 1.0
              },
              "image_processor": {
                "max_soft_tokens": 280,
                "patch_size": 16,
                "pooling_kernel_size": 3,
                "model_patch_size": 48
              }
            }
            """)

        #expect(config.audioTokenId == 258_881)
        #expect(config.boaTokenId == 256_000)
        #expect(config.eoaTokenId == 258_883)
        #expect(config.audioSampleRate == 16_000)
        #expect(config.audioSamplesPerToken == 640)
        #expect(config.audioSeqLength == 750)
        #expect(config.audioMsPerToken == 40)
        #expect(config.videoTokenId == 258_884)
        #expect(config.videoSoftTokensPerFrame == 70)
        #expect(config.videoMaxFrames == 32)
    }

    // MARK: - Audio

    @Test("Unified audio features are raw waveform frames (reference framing parity)")
    func audioFramingMatchesReference() async throws {
        let config = try tinyProcessorConfig()
        let processor = Gemma4UnifiedProcessor(config, tokenizer: UnifiedAVTestTokenizer())

        // 1600 samples = 2.5 frames of 640 → 3 tokens, tail zero-padded.
        let samples = (0 ..< 1600).map { Float($0) * 1e-3 }
        let input = try await processor.prepare(
            input: UserInput(prompt: "transcribe", audios: [.array(MLXArray(samples))]))

        let audio = try #require(input.audio)
        #expect(audio.features.shape == [1, 3, 640])
        #expect(audio.mask?.shape == [1, 3])
        #expect(audio.mask?.asType(.int32).sum().item(Int.self) == 3)

        // Frame contents must be the exact reshaped waveform (no windowing, no
        // normalization): frame 0 = samples[0..<640], frame 2 tail = zeros.
        let features = audio.features.asArray(Float.self)
        #expect(features[0] == samples[0])
        #expect(features[639] == samples[639])
        #expect(features[640] == samples[640])
        #expect(features[2 * 640 + 319] == samples[1599])
        #expect(features[(2 * 640 + 320) ..< (3 * 640)].allSatisfy { $0 == 0 })

        // Template [3, <audio>, 2] → 3, boa, audio×3, eoa, 2.
        #expect(input.text.tokens.asArray(Int32.self) == [3, 27, 30, 30, 30, 26, 2])
    }

    @Test("Unified audio truncates to audio_seq_length tokens (30s cap)")
    func audioTruncatesToSeqLength() async throws {
        let config = try tinyProcessorConfig(extraLines: "\"audio_seq_length\": 2,")
        let processor = Gemma4UnifiedProcessor(config, tokenizer: UnifiedAVTestTokenizer())

        let samples = [Float](repeating: 0.25, count: 5 * 640)
        let input = try await processor.prepare(
            input: UserInput(prompt: "transcribe", audios: [.array(MLXArray(samples))]))

        let audio = try #require(input.audio)
        #expect(audio.features.shape == [1, 2, 640])
        #expect(input.text.tokens.asArray(Int32.self) == [3, 27, 30, 30, 26, 2])
    }

    @Test("Unified audio batches clips with padding masked out")
    func multipleAudiosArePaddedAndMasked() async throws {
        let config = try tinyProcessorConfig()
        var tokenizer = UnifiedAVTestTokenizer()
        tokenizer.template = [3, 30, 4, 30, 2]
        let processor = Gemma4UnifiedProcessor(config, tokenizer: tokenizer)

        let short = [Float](repeating: 0.5, count: 640)
        let long = [Float](repeating: -0.5, count: 1600)
        let input = try await processor.prepare(
            input: UserInput(
                prompt: "compare",
                audios: [.array(MLXArray(short)), .array(MLXArray(long))]))

        let audio = try #require(input.audio)
        #expect(audio.features.shape == [2, 3, 640])
        let maskCounts = try #require(audio.mask?.asType(.int32).sum(axis: -1))
        #expect(maskCounts.asArray(Int.self) == [1, 3])
        #expect(
            input.text.tokens.asArray(Int32.self)
                == [3, 27, 30, 26, 4, 27, 30, 30, 30, 26, 2])
    }

    @Test("Unified audio processor → model round trip (scatter counts match)")
    func audioProcessorModelRoundTrip() async throws {
        let config = try tinyProcessorConfig()
        let processor = Gemma4UnifiedProcessor(config, tokenizer: UnifiedAVTestTokenizer())
        let model = Gemma4Unified(
            try decodeModelConfig(
                """
                {
                  "model_type": "gemma4_unified",
                  "vocab_size": 32,
                  "image_token_id": 25,
                  "audio_token_id": 30,
                  "video_token_id": 29,
                  "text_config": {
                    "model_type": "gemma4_unified_text",
                    "hidden_size": 8,
                    "num_hidden_layers": 1,
                    "intermediate_size": 16,
                    "num_attention_heads": 1,
                    "num_key_value_heads": 1,
                    "num_global_key_value_heads": 1,
                    "head_dim": 8,
                    "global_head_dim": 8,
                    "vocab_size": 32,
                    "vocab_size_per_layer_input": 32,
                    "num_kv_shared_layers": 0,
                    "hidden_size_per_layer_input": 0,
                    "sliding_window": 8,
                    "sliding_window_pattern": 1,
                    "attention_k_eq_v": true,
                    "use_double_wide_mlp": false,
                    "layer_types": ["full_attention"],
                    "tie_word_embeddings": true
                  },
                  "vision_config": null,
                  "audio_config": {
                    "model_type": "gemma4_unified_audio",
                    "audio_samples_per_token": 640,
                    "audio_embed_dim": 640,
                    "hidden_size": 640,
                    "output_proj_dims": 640
                  }
                }
                """))

        let samples = (0 ..< 1600).map { Float($0) * 1e-3 }
        let input = try await processor.prepare(
            input: UserInput(prompt: "transcribe", audios: [.array(MLXArray(samples))]))

        let result = try model.prepare(
            input, cache: model.newCache(parameters: nil), windowSize: nil)

        guard case .logits(let output) = result else {
            Issue.record("Expected Gemma4Unified.prepare to return logits")
            return
        }
        // Prompt expands to [3, boa, audio×3, eoa, 2] = 7 tokens; a count
        // mismatch between placeholders and feature rows would have thrown.
        #expect(output.logits.shape == [1, 7, 32])
    }

    // MARK: - Video

    @Test("Unified processor expands video placeholders per frame with timestamps")
    func videoExpansionPerFrame() async throws {
        let config = try tinyProcessorConfig()
        var tokenizer = UnifiedAVTestTokenizer()
        tokenizer.template = [3, 29, 2]
        let processor = Gemma4UnifiedProcessor(config, tokenizer: tokenizer)

        let frame = CIImage(color: .gray).cropped(to: CGRect(x: 0, y: 0, width: 8, height: 8))
        let video = UserInput.Video.frames([
            .init(frame: frame, timeStamp: CMTime(seconds: 0, preferredTimescale: 600)),
            .init(frame: frame, timeStamp: CMTime(seconds: 2, preferredTimescale: 600)),
        ])

        let input = try await processor.prepare(
            input: UserInput(prompt: "describe", videos: [video]))

        let videoInput = try #require(input.video)
        // 8×8 frame / 4px model patches = 4 patches per frame, budget 4 → no padding.
        #expect(videoInput.pixels.shape == [2, 4, 48])
        #expect(videoInput.positionIds?.shape == [2, 4, 2])

        // Per frame: timestamp marker (9) + boi + video×4 + eoi.
        #expect(
            input.text.tokens.asArray(Int32.self)
                == [3, 9, 28, 29, 29, 29, 29, 31, 9, 28, 29, 29, 29, 29, 31, 2])
    }

    @Test("Unified video frames use the per-frame soft-token budget, padded with -1 positions")
    func videoFramesUsePerFrameBudget() async throws {
        // Frame budget 4 but a 8×4 frame only yields 2 real patches → padding.
        let config = try tinyProcessorConfig(extraLines: "\"do_resize\": false,")
        let processor = Gemma4UnifiedProcessor(config, tokenizer: UnifiedAVTestTokenizer())

        let frame = CIImage(color: .gray).cropped(to: CGRect(x: 0, y: 0, width: 8, height: 4))
        let videoData = try await processor.processVideos(
            [.frames([.init(frame: frame, timeStamp: .zero)])], processing: nil)

        #expect(videoData.pixels.shape == [1, 4, 48])
        #expect(videoData.positionIds.shape == [1, 4, 2])
        #expect(videoData.tokenCounts == [[2]])
        let positions = videoData.positionIds.asArray(Int32.self)
        #expect(Array(positions[0 ..< 4]) == [0, 0, 1, 0])
        #expect(Array(positions[4 ..< 8]) == [-1, -1, -1, -1])
    }

    @Test("Unified video timestamp text formats mm:ss like the reference")
    func videoTimestampFormatting() {
        #expect(gemma4VideoTimestampText(seconds: 0) == "00:00")
        #expect(gemma4VideoTimestampText(seconds: 2.4) == "00:02")
        #expect(gemma4VideoTimestampText(seconds: 61.9) == "01:01")
        #expect(gemma4VideoTimestampText(seconds: 600) == "10:00")
    }

    @Test("Unified model scatters video features through the vision embedder")
    func videoEndToEndTinyModel() throws {
        let model = Gemma4Unified(
            try decodeModelConfig(
                """
                {
                  "model_type": "gemma4_unified",
                  "vocab_size": 32,
                  "image_token_id": 25,
                  "audio_token_id": 30,
                  "video_token_id": 29,
                  "text_config": {
                    "model_type": "gemma4_unified_text",
                    "hidden_size": 8,
                    "num_hidden_layers": 1,
                    "intermediate_size": 16,
                    "num_attention_heads": 1,
                    "num_key_value_heads": 1,
                    "num_global_key_value_heads": 1,
                    "head_dim": 8,
                    "global_head_dim": 8,
                    "vocab_size": 32,
                    "vocab_size_per_layer_input": 32,
                    "num_kv_shared_layers": 0,
                    "hidden_size_per_layer_input": 0,
                    "sliding_window": 8,
                    "sliding_window_pattern": 1,
                    "attention_k_eq_v": true,
                    "use_double_wide_mlp": false,
                    "layer_types": ["full_attention"],
                    "tie_word_embeddings": true
                  },
                  "vision_config": {
                    "model_type": "gemma4_unified_vision",
                    "patch_size": 2,
                    "pooling_kernel_size": 2,
                    "model_patch_size": 4,
                    "mm_embed_dim": 8,
                    "mm_posemb_size": 4,
                    "num_soft_tokens": 4,
                    "output_proj_dims": 8
                  },
                  "audio_config": null
                }
                """))

        // Two frames × 3 real patches (1 padded with -1 positions) = 6 video tokens.
        let inputIds = MLXArray([0, 29, 29, 29, 29, 29, 29, 1]).reshaped(1, 8)
        let pixels = MLXArray.zeros([2, 4, 48], dtype: .float32)
        var positionValues = [Int32](repeating: 0, count: 2 * 4 * 2)
        positionValues[6] = -1
        positionValues[7] = -1
        positionValues[14] = -1
        positionValues[15] = -1
        let positionIds = MLXArray(positionValues, [2, 4, 2])

        let input = LMInput(
            text: .init(tokens: inputIds),
            video: .init(pixels: pixels, positionIds: positionIds)
        )

        let result = try model.prepare(
            input, cache: model.newCache(parameters: nil), windowSize: nil)

        guard case .logits(let output) = result else {
            Issue.record("Expected Gemma4Unified.prepare to return logits")
            return
        }
        #expect(output.logits.shape == [1, 8, 32])
    }
}
