// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXVLM

// MARK: - Gemma 4 audio encoder (`gemma4_audio` USM-Conformer) parity tests
//
// These pin the Swift port of the audio tower to the Python reference
// (Blaizzy/mlx-vlm `mlx_vlm/models/gemma4/audio.py`): the checkpoint weight-key
// names, the SSCP time-subsampling factor (4×), the output shape / projection
// width, and the padding-frame zeroing. Full numerical parity against exported
// reference activations is a follow-up (needs a Python+mlx env and the
// checkpoint); everything here runs from synthetic tensors.
//
// Run via Xcode (`swift test` on the CLI cannot load MLX's default.metallib):
//   xcodebuild test -scheme mlx-swift-lm-Package \
//     -destination 'platform=macOS,arch=arm64' \
//     -only-testing:MLXLMTests/Gemma4AudioTests \
//     -skipPackagePluginValidation -skipMacroValidation

struct Gemma4AudioTests {

    // MARK: Config

    /// A small-but-faithful audio config: real chunk/context/conv/scale values,
    /// a reduced hidden size and layer count for fast tests. Mel bins stay 128
    /// (the SSCP `INPUT_FEAT_SIZE`), so the frequency reduction 128→64→32 holds.
    private static func audioConfig(
        hiddenSize: Int = 64, layers: Int = 2, outputProjDims: Int? = 48
    ) -> Gemma4AudioConfiguration {
        let json = """
            {
              "model_type": "gemma4_audio",
              "hidden_size": \(hiddenSize),
              "num_hidden_layers": \(layers),
              "num_attention_heads": 4,
              "subsampling_conv_channels": [128, 32],
              "conv_kernel_size": 5,
              "residual_weight": 0.5,
              "attention_chunk_size": 12,
              "attention_context_left": 13,
              "attention_context_right": 0,
              "attention_logit_cap": 50.0,
              "attention_invalid_logits_value": -1e9,
              "rms_norm_eps": 1e-6,
              "gradient_clipping": 1e10,
              \(outputProjDims.map { "\"output_proj_dims\": \($0)," } ?? "")
              "use_clipped_linears": true
            }
            """
        return try! JSONDecoder().decode(
            Gemma4AudioConfiguration.self, from: Data(json.utf8))
    }

    /// Expected subsampled time length after the two stride-2 SSCP conv blocks
    /// (kernel 3, symmetric pad 1): `(t - 1) / 2 + 1`, applied twice.
    private static func subsampledLength(_ t: Int) -> Int {
        var n = t
        for _ in 0 ..< 2 { n = (n - 1) / 2 + 1 }
        return n
    }

    @Test("audio_config decodes to the checkpoint's real field values")
    func audioConfigDecodesRealFields() {
        let json = """
            {
              "model_type": "gemma4_audio", "hidden_size": 1024, "num_hidden_layers": 12,
              "num_attention_heads": 8, "subsampling_conv_channels": [128, 32],
              "conv_kernel_size": 5, "residual_weight": 0.5, "attention_chunk_size": 12,
              "attention_context_left": 13, "attention_context_right": 0,
              "attention_logit_cap": 50.0, "rms_norm_eps": 1e-6, "gradient_clipping": 1e10,
              "output_proj_dims": 1536, "use_clipped_linears": true
            }
            """
        let c = try! JSONDecoder().decode(Gemma4AudioConfiguration.self, from: Data(json.utf8))
        #expect(c.hiddenSize == 1024)
        #expect(c.hiddenLayers == 12)
        #expect(c.attentionHeads == 8)
        #expect(c.headDim == 128)
        #expect(c.subsamplingConvChannels == [128, 32])
        #expect(c.convKernelSize == 5)
        #expect(c.residualWeight == 0.5)
        #expect(c.attentionChunkSize == 12)
        #expect(c.attentionContextLeft == 13)
        #expect(c.attentionContextRight == 0)
        #expect(c.attentionLogitCap == 50.0)
        #expect(c.outputProjectionDimensions == 1536)
        #expect(c.useClippedLinears == true)
    }

    @Test("Gemma4Configuration decodes audio_config and the audio/video token ids")
    func gemma4ConfigDecodesAudioSection() {
        // Minimal top-level config: vision_config defaults, a real audio_config,
        // and the E4B token ids. text_config carries only what the text decoder needs.
        let json = """
            {
              "model_type": "gemma4",
              "text_config": {
                "model_type": "gemma4_text", "hidden_size": 8, "num_hidden_layers": 2,
                "intermediate_size": 16, "num_attention_heads": 2, "num_key_value_heads": 1,
                "head_dim": 4, "global_head_dim": 4, "vocab_size": 12,
                "vocab_size_per_layer_input": 12, "num_kv_shared_layers": 0,
                "hidden_size_per_layer_input": 0, "sliding_window": 8,
                "sliding_window_pattern": 1, "layer_types": ["full_attention", "full_attention"],
                "rope_parameters": {}, "tie_word_embeddings": true
              },
              "vision_config": {},
              "audio_config": {
                "model_type": "gemma4_audio", "hidden_size": 1024, "num_hidden_layers": 12,
                "num_attention_heads": 8, "subsampling_conv_channels": [128, 32],
                "output_proj_dims": 1536, "use_clipped_linears": true
              },
              "image_token_id": 258880, "audio_token_id": 258881, "video_token_id": 258884,
              "boi_token_id": 255999, "eoi_token_id": 258882,
              "boa_token_id": 256000, "eoa_token_id": 258883
            }
            """
        let config = try! JSONDecoder().decode(Gemma4Configuration.self, from: Data(json.utf8))
        #expect(config.audioConfiguration != nil)
        #expect(config.audioConfiguration?.hiddenSize == 1024)
        #expect(config.audioConfiguration?.outputProjectionDimensions == 1536)
        #expect(config.audioTokenId == 258881)
        #expect(config.videoTokenId == 258884)
        #expect(config.boaTokenId == 256000)
        #expect(config.eoaTokenId == 258883)
    }

    // MARK: Weight-key parity

    @Test("audio_tower module tree exposes the checkpoint weight keys")
    func audioTowerExposesCheckpointWeightKeys() {
        let model = Gemma4AudioModel(config: Self.audioConfig(layers: 2))
        let keys = Set(model.parameters().flattened().map(\.0))

        // SSCP: two conv blocks (Conv2d weight + LayerNorm weight) and the projection.
        #expect(keys.contains("subsample_conv_projection.layer0.conv.weight"))
        #expect(keys.contains("subsample_conv_projection.layer0.norm.weight"))
        #expect(keys.contains("subsample_conv_projection.layer1.conv.weight"))
        #expect(keys.contains("subsample_conv_projection.input_proj_linear.weight"))

        // Conformer block 0: macaron FFNs, attention (clippable `.linear.` nesting +
        // the plain `relative_k_proj` and the `per_dim_scale` parameter), light-conv, norms.
        #expect(keys.contains("layers.0.feed_forward1.ffw_layer_1.linear.weight"))
        #expect(keys.contains("layers.0.feed_forward2.ffw_layer_2.linear.weight"))
        #expect(keys.contains("layers.0.self_attn.q_proj.linear.weight"))
        #expect(keys.contains("layers.0.self_attn.post.linear.weight"))
        #expect(keys.contains("layers.0.self_attn.relative_k_proj.weight"))
        #expect(keys.contains("layers.0.self_attn.per_dim_scale"))
        #expect(keys.contains("layers.0.lconv1d.depthwise_conv1d.weight"))
        #expect(keys.contains("layers.0.lconv1d.linear_start.linear.weight"))
        #expect(keys.contains("layers.0.norm_out.weight"))

        // Output projection (Linear with bias).
        #expect(keys.contains("output_proj.weight"))
        #expect(keys.contains("output_proj.bias"))

        // The depthwise conv weight is MLX layout `[C, K, 1]`.
        let depthwise = model.parameters().flattened().first {
            $0.0 == "layers.0.lconv1d.depthwise_conv1d.weight"
        }!.1
        #expect(depthwise.shape == [64, 5, 1])
    }

    @Test("A synthetic audio_tower checkpoint round-trips through the strict loader")
    func audioTowerRoundTripsSyntheticCheckpoint() throws {
        let source = Gemma4AudioModel(config: Self.audioConfig(layers: 2))
        eval(source)
        var checkpoint = [String: MLXArray]()
        for (key, value) in source.parameters().flattened() {
            checkpoint[key] = value
        }
        let model = Gemma4AudioModel(config: Self.audioConfig(layers: 2))
        // `[.all]` is what MLXLMCommon.loadWeights applies — throws on any module
        // parameter with no matching weight, or any leftover unmatched weight.
        try model.update(
            parameters: ModuleParameters.unflattened(checkpoint), verify: [.all])
        eval(model)
    }

    // MARK: Forward pass

    @Test("Encoder subsamples time 4× and projects to output_proj_dims")
    func audioTowerForwardShapeAndSubsampling() {
        let config = Self.audioConfig(layers: 2, outputProjDims: 48)
        let model = Gemma4AudioModel(config: config)
        eval(model)

        let t = 50
        let mel = MLXRandom.normal([1, t, 128])
        let mask = MLXArray.zeros([1, t]).asType(.bool)  // all valid
        let (encodings, subMask) = model(mel, audioMelMask: mask)
        eval(encodings, subMask)

        let expectedT = Self.subsampledLength(t)  // 50 -> 25 -> 13
        #expect(encodings.shape == [1, expectedT, 48])
        #expect(subMask.shape == [1, expectedT])
    }

    @Test("Encoder without output_proj keeps hidden_size and stays finite")
    func audioTowerNoOutputProjKeepsHidden() {
        let config = Self.audioConfig(hiddenSize: 64, layers: 2, outputProjDims: nil)
        let model = Gemma4AudioModel(config: config)
        eval(model)

        let mel = MLXRandom.normal([1, 40, 128])
        let mask = MLXArray.zeros([1, 40]).asType(.bool)
        let (encodings, _) = model(mel, audioMelMask: mask)
        eval(encodings)

        #expect(encodings.shape == [1, Self.subsampledLength(40), 64])
        // gradient_clipping (1e10) keeps everything finite; guard against NaN/Inf.
        let maxAbs = encodings.abs().max().item(Float.self)
        #expect(maxAbs.isFinite)
    }

    @Test("Padding frames are zeroed in the encoder output")
    func audioTowerZeroesPaddedFrames() {
        let config = Self.audioConfig(layers: 2, outputProjDims: 48)
        let model = Gemma4AudioModel(config: config)
        eval(model)

        let t = 50
        let mel = MLXRandom.normal([1, t, 128])
        // Mark the last 10 mel frames as padding (mask == true). After the 4×
        // subsample these land on the final output frames, which must be zeroed.
        var maskValues = [Int32](repeating: 0, count: t)
        for i in (t - 10) ..< t { maskValues[i] = 1 }
        let mask = MLXArray(maskValues).reshaped(1, t).asType(.bool)

        let (encodings, subMask) = model(mel, audioMelMask: mask)
        eval(encodings, subMask)

        let lastFrame = encodings[0, encodings.dim(1) - 1]
        let lastFrameEnergy = lastFrame.abs().sum().item(Float.self)
        #expect(lastFrameEnergy == 0.0)

        // And the subsampled mask must flag that final frame as padding.
        let subMaskValues = subMask.asType(.int32).asArray(Int32.self)
        #expect(subMaskValues.last == 1)
    }

    // MARK: Mel feature extractor

    @Test("HTK mel filterbank is [257, 128], non-negative, mostly non-empty")
    func melFilterBankValid() {
        let bins = 257
        let mel = 128
        let bank = Gemma4AudioFeatureExtractor.melFilterBank(
            numFreqBins: bins, numMel: mel, minFrequency: 0, maxFrequency: 8000, sampleRate: 16_000)
        #expect(bank.count == bins * mel)
        #expect(bank.allSatisfy { $0 >= 0 && $0 <= 1.0001 })
        #expect(bank.contains { $0 > 0.5 })  // triangles reach their peak

        func columnMax(_ m: Int) -> Float {
            var v: Float = 0
            for k in 0 ..< bins { v = max(v, bank[k * mel + m]) }
            return v
        }
        // The lowest mel filters can be narrower than the FFT bin spacing
        // (31.25 Hz) and come out empty — this matches the reference
        // `_mel_filter_bank` / librosa. Only a few may be empty; the upper mel
        // range is always well resolved.
        let nonEmpty = (0 ..< mel).filter { columnMax($0) > 0 }.count
        #expect(nonEmpty >= mel - 5)
        for m in (mel / 2) ..< mel { #expect(columnMax(m) > 0) }
    }

    @Test("Extractor yields [1, T, 128] with the reference frame count")
    func extractorShapeAndFrameCount() {
        let extractor = Gemma4AudioFeatureExtractor()
        let samples = 16_000  // 1 s @ 16 kHz (already a multiple of 128)
        let (features, mask) = extractor(MLXRandom.normal([samples]))
        eval(features, mask)

        #expect(features.dim(0) == 1)
        #expect(features.dim(2) == 128)
        #expect(mask.shape == [1, features.dim(1)])

        // paddedLen = frameLength/2 + samples = 160 + 16000 = 16160;
        // numFrames = (16160 - 321) / 160 + 1 = 99.
        #expect(features.dim(1) == 99)
        // A whole-second clip aligned to 128 samples has every frame valid.
        #expect(mask.asType(.int32).sum().item(Int.self) == 99)
        #expect(features.max().item(Float.self).isFinite)
    }

    @Test("Extractor is deterministic")
    func extractorDeterministic() {
        let extractor = Gemma4AudioFeatureExtractor()
        let wave = MLXRandom.normal([12_000])
        let (a, _) = extractor(wave)
        let (b, _) = extractor(wave)
        eval(a, b)
        #expect((a - b).abs().max().item(Float.self) == 0)
    }

    @Test(
        "audioTokenCount = ceil(ceil(V/2)/2)",
        arguments: [1, 2, 4, 13, 25, 50, 99, 100, 400, 3000])
    func audioTokenCountFormula(validFrames: Int) {
        let expected = ((validFrames + 1) / 2 + 1) / 2
        #expect(
            Gemma4AudioFeatureExtractor.audioTokenCount(validFrames: validFrames) == expected)
    }

    /// The crucial end-to-end consistency check: the token count the processor
    /// would emit for a clip (`audioTokenCount` of the valid mel frames) must equal
    /// the number of valid frames the audio tower actually produces after
    /// subsampling — otherwise the prompt's audio-token count and the scattered
    /// encoder frames disagree and the scatter throws.
    @Test("Extractor token count matches the tower's valid subsampled frames")
    func extractorTowerFrameCountConsistency() {
        let extractor = Gemma4AudioFeatureExtractor()
        let model = Gemma4AudioModel(config: Self.audioConfig(layers: 2))
        eval(model)

        for samples in [8_000, 16_000, 24_000] {
            let (features, mask) = extractor(MLXRandom.normal([samples]))
            let validFrames = mask.asType(.int32).sum().item(Int.self)
            let expectedTokens = Gemma4AudioFeatureExtractor.audioTokenCount(
                validFrames: validFrames)

            // Tower wants `true == padding`, so invert the extractor's valid mask.
            let (encodings, subMask) = model(features, audioMelMask: logicalNot(mask))
            eval(encodings, subMask)
            let validSubsampled = logicalNot(subMask).asType(.int32).sum().item(Int.self)

            #expect(
                validSubsampled == expectedTokens,
                "samples=\(samples): tower produced \(validSubsampled) valid frames, processor would emit \(expectedTokens) audio tokens")
        }
    }
}
