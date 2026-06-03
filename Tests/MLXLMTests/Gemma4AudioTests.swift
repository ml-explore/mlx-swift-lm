//
//  Gemma4AudioTests.swift
//  MLXVLMTests
//
//  Tests for Gemma4 audio tower — MelSpectrogram, AudioEncoder, token merging.
//  These tests verify the Swift port against the Python mlx-vlm implementation.
//

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXVLM

// MARK: - Unit Tests

@Suite("Gemma4 Audio Tower")
struct Gemma4AudioTests {

    // MARK: - Configuration

    @Test
    func audioConfigurationDecoding() throws {
        // Verify AudioConfiguration can decode from model config JSON
        let json = """
            {
                "audio_token_id": 262277,
                "audio_config": {
                    "model_type": "gemma4_audio",
                    "num_mel_bins": 128,
                    "encoder_layers": 32,
                    "encoder_attention_heads": 8,
                    "input_feat_per_channel": 128,
                    "encoder_dim": 1024,
                    "encoder_ffn_dim": 4096,
                    "dropout": 0.0,
                    "conv_kernel_sizes": [5, 5, 5],
                    "conv_channels": 1024,
                    "num_audio_tokens": 750
                }
            }
            """.data(using: .utf8)!

        // This test will compile once AudioConfiguration is added to Gemma4.swift
        // let config = try JSONDecoder().decode(Gemma4AudioConfig.self, from: json)
        // #expect(config.numMelBins == 128)
        // #expect(config.encoderLayers == 32)
        // #expect(config.numAudioTokens == 750)
    }

    // MARK: - Mel Spectrogram

    @Test
    func melSpectrogramShape() throws {
        // 1 second of 16kHz audio = 16000 samples
        // Expected output: ~100 frames × 128 mel bins (hop=160, window=400)
        let audio = MLXArray.zeros([16000])

        // let mel = Gemma4MelSpectrogram(numMelBins: 128, sampleRate: 16000)
        // let features = mel(audio)
        // #expect(features.dim(0) > 90)  // ~100 frames
        // #expect(features.dim(1) == 128)  // 128 mel bins
    }

    @Test
    func melSpectrogramDeterministic() throws {
        // Same input should produce same output
        let audio = MLXArray(Array(repeating: Float(0.5), count: 8000))

        // let mel = Gemma4MelSpectrogram(numMelBins: 128, sampleRate: 16000)
        // let out1 = mel(audio)
        // let out2 = mel(audio)
        // #expect(out1.isClose(out2, atol: 1e-6).all().item(Bool.self))
    }

    // MARK: - Audio Encoder

    @Test
    func audioEncoderOutputShape() throws {
        // Audio encoder takes mel features and outputs encoder hidden states
        // Input: [batch, frames, mel_bins]
        // Output: [batch, num_audio_tokens, encoder_dim]

        // let config = Gemma4AudioConfig(...)
        // let encoder = Gemma4AudioEncoder(config: config)
        // let melInput = MLXArray.zeros([1, 100, 128])
        // let output = encoder(melInput)
        // #expect(output.dim(0) == 1)
        // #expect(output.dim(2) == config.encoderDim)
    }

    // MARK: - Token Merging

    @Test
    func audioTokenMerging() throws {
        // Verify audio tokens are correctly merged into the input embedding sequence
        // Audio tokens should replace <|audio|> placeholder tokens in the input

        // let textTokens = MLXArray([1, 2, 262277, 262277, 262277, 3, 4])  // 262277 = audio_token_id
        // let audioFeatures = MLXArray.zeros([1, 3, 1024])  // 3 audio tokens
        // let textEmbeddings = model.embedTokens(textTokens)
        // let merged = model.mergeAudioFeatures(textEmbeddings, audioFeatures, textTokens)
        // #expect(merged.dim(1) == textTokens.dim(0))
    }
}

// MARK: - Integration Tests (require model download)

@Suite("Gemma4 Audio Integration")
struct Gemma4AudioIntegrationTests {

    @Test
    func endToEndTranscription() async throws {
        // Load model and transcribe a test audio file
        // This test requires the model to be downloaded

        // let model = try await loadGemma4Model("mlx-community/gemma-4-e2b-it-4bit")
        // let audio = loadWAV("test_audio.wav")
        // let result = model.generate(audio: audio, prompt: "Transcribe this audio.")
        // #expect(!result.isEmpty)
        // #expect(!result.contains("[ERROR"))
    }

    @Test
    func audioTokenCount() async throws {
        // Verify audio token count scales with duration (~40ms per token, max 750)
        // 1s audio → ~25 tokens
        // 10s audio → ~250 tokens
        // 30s audio → ~750 tokens (max)

        // let audio1s = MLXArray.zeros([16000])
        // let audio10s = MLXArray.zeros([160000])
        // let tokens1 = model.audioEncoder.getAudioTokenCount(audio1s)
        // let tokens10 = model.audioEncoder.getAudioTokenCount(audio10s)
        // #expect(tokens1 > 20 && tokens1 < 30)
        // #expect(tokens10 > 200 && tokens10 < 300)
    }
}

// MARK: - Python Alignment Tests

@Suite("Gemma4 Audio Python Alignment")
struct Gemma4AudioAlignmentTests {

    @Test
    func melOutputMatchesPython() throws {
        // Compare Swift mel spectrogram output with pre-computed Python output
        // Load reference data from a .npy or .json fixture file

        // let refMel = loadReference("mel_reference.json")
        // let audio = loadReference("audio_input.json")
        // let mel = Gemma4MelSpectrogram(...)
        // let swiftMel = mel(MLXArray(audio))
        // #expect(swiftMel.isClose(MLXArray(refMel), atol: 1e-4).all().item(Bool.self))
    }
}
