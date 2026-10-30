// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXVLM

// Tests of the LFM2-VL model: configuration, weights, projector and image features. Exact
// comparisons run on the CPU; see LFM2BidirectionalTests.swift. Expected values come from
// tools/lfm2_vl_test_vectors.py.

private func tinyLFM2VLJSON(textExtras: String = "", extras: String = "") -> String {
    """
    {
      "model_type": "lfm2_vl",\(extras)
      "image_token_id": 7,
      "downsample_factor": 2,
      "projector_hidden_size": 16,
      "projector_use_layernorm": false,
      "text_config": {
        "model_type": "lfm2", "vocab_size": 32, "hidden_size": 16, "num_hidden_layers": 2,
        "num_attention_heads": 2, "num_key_value_heads": 1, "intermediate_size": 32,
        "block_dim": 16, "block_multiple_of": 8, "block_auto_adjust_ff_dim": false,
        "layer_types": ["conv", "full_attention"]\(textExtras)
      },
      "vision_config": {
        "model_type": "siglip2_vision_model", "hidden_size": 16, "intermediate_size": 32,
        "num_hidden_layers": 2, "num_attention_heads": 2, "num_channels": 3,
        "num_patches": 16, "patch_size": 2, "layer_norm_eps": 1e-6
      }
    }
    """
}

private func decode<T: Decodable>(_ type: T.Type, _ json: String) throws -> T {
    try JSONDecoder().decode(type, from: Data(json.utf8))
}

private func textConfig(_ fields: String) throws -> LFM2VLConfiguration.TextConfiguration {
    try decode(
        LFM2VLConfiguration.TextConfiguration.self,
        """
        {
          "model_type": "lfm2", "vocab_size": 32, "hidden_size": 16, "num_hidden_layers": 2,
          "num_attention_heads": 2, "num_key_value_heads": 1\(fields)
        }
        """)
}

private func maxAbsDiff(_ a: MLXArray, _ b: MLXArray) -> Float {
    abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
}

private func randomPixels(_ shape: [Int], seed: UInt64) -> MLXArray {
    withRandomState(MLXRandom.RandomState(seed: seed)) { MLXRandom.normal(shape) }
}

struct LFM2VLTests {

    // MARK: - Configuration

    @Test("Text config falls back to intermediate_size for the MLP width")
    func feedForwardWidth() throws {
        let fromIntermediate = try textConfig(
            #", "intermediate_size": 48, "block_auto_adjust_ff_dim": false"#)
        #expect(fromIntermediate.blockFFDim == 48)
        #expect(fromIntermediate.feedForwardDimensions == 48)

        let both = try textConfig(
            #", "block_ff_dim": 40, "intermediate_size": 48, "block_auto_adjust_ff_dim": false"#)
        #expect(both.blockFFDim == 40)

        #expect(try textConfig("").blockFFDim == 16)

        // LFM2-VL-450M: 6656 -> int(2*6656/3) = 4437 -> next multiple of 256.
        let adjusted = try textConfig(
            #", "block_ff_dim": 6656, "block_ffn_dim_multiplier": 1.0, "block_multiple_of": 256"#)
        #expect(adjusted.feedForwardDimensions == 4608)

        let scaled = try textConfig(
            #", "intermediate_size": 300, "block_ffn_dim_multiplier": 1.5, "block_multiple_of": 8"#
        )
        #expect(scaled.feedForwardDimensions == 304)

        let noMultiplier = try textConfig(
            #", "block_ff_dim": 96, "block_ffn_dim_multiplier": null, "block_multiple_of": 7"#)
        #expect(noMultiplier.feedForwardDimensions == 70)
    }

    @Test("Text config reads rope_parameters.rope_theta before rope_theta")
    func ropeTheta() throws {
        #expect(try textConfig("").ropeTheta == 1_000_000)
        #expect(try textConfig(#", "rope_theta": 20000.0"#).ropeTheta == 20_000)
        #expect(
            try textConfig(
                #", "rope_parameters": {"rope_theta": 500000.0, "rope_type": "default"}"#
            )
            .ropeTheta == 500_000)
        #expect(
            try textConfig(#", "rope_theta": 20000.0, "rope_parameters": {"rope_theta": 500000.0}"#)
                .ropeTheta == 500_000)
        #expect(
            try textConfig(#", "rope_theta": 20000.0, "rope_parameters": {"rope_type": "default"}"#)
                .ropeTheta == 20_000)
        #expect(try textConfig(#", "rope_parameters": null"#).ropeTheta == 1_000_000)
    }

    @Test("Text config reads tie_word_embeddings and its tie_embedding alias")
    func textTieWordEmbeddings() throws {
        #expect(try textConfig("").tieWordEmbeddings)
        #expect(try textConfig(#", "tie_word_embeddings": null"#).tieWordEmbeddings)
        #expect(try textConfig(#", "tie_word_embeddings": true"#).tieWordEmbeddings)
        #expect(try !textConfig(#", "tie_word_embeddings": false"#).tieWordEmbeddings)
        // LFM2-VL-3B writes only `tie_embedding`. Like `Lfm2Config`, the alias wins.
        #expect(try !textConfig(#", "tie_embedding": false"#).tieWordEmbeddings)
        #expect(
            try textConfig(#", "tie_embedding": true, "tie_word_embeddings": false"#)
                .tieWordEmbeddings)
    }

    @Test("The top-level tie flags win over text_config, as in Lfm2VlConfig")
    func tieWordEmbeddings() throws {
        func tied(_ extras: String, text textExtras: String = "") throws -> Bool {
            try decode(
                LFM2VLConfiguration.self, tinyLFM2VLJSON(textExtras: textExtras, extras: extras)
            ).tieWordEmbeddings
        }
        #expect(try tied(""))
        // transformers 5 writes an untied fine-tune with the text flag left at true.
        #expect(
            try !tied(#" "tie_word_embeddings": false,"#, text: #", "tie_word_embeddings": true"#))
        #expect(
            try tied(#" "tie_word_embeddings": true,"#, text: #", "tie_word_embeddings": false"#))
        #expect(try !tied(#" "tie_embedding": false, "tie_word_embeddings": true,"#))
        #expect(try tied(#" "tie_embedding": true,"#, text: #", "tie_word_embeddings": false"#))
        // Without a top-level flag, text_config decides, for mlx-vlm conversions.
        #expect(try !tied("", text: #", "tie_word_embeddings": false"#))
        #expect(try tied("", text: #", "tie_embedding": true"#))
    }

    @Test("Tower config follows the LFM2-VL vision_config and feature layer")
    func towerConfiguration() throws {
        let tiny = Siglip2NaFlex.Configuration(
            hiddenSize: 16, intermediateSize: 32, numHiddenLayers: 2, numAttentionHeads: 2,
            numChannels: 3, numPatches: 16, patchSize: 2, layerNormEps: 1e-6)
        let lfm2 = try decode(LFM2VLConfiguration.self, tinyLFM2VLJSON())
        #expect(
            Siglip2NaFlex.Configuration(lfm2.visionConfiguration, visionFeatureLayer: -1) == tiny)
        #expect(
            Siglip2NaFlex.Configuration(lfm2.visionConfiguration, visionFeatureLayer: -2)
                .numHiddenLayers == 1)
        #expect(
            Siglip2NaFlex.Configuration(lfm2.visionConfiguration, visionFeatureLayer: 0)
                .numHiddenLayers == 1)
    }

    // MARK: - LFM2VL keys and accessors

    private func tinyModel(textExtras: String = "", extras: String = "") throws -> LFM2VL {
        let config = try decode(
            LFM2VLConfiguration.self, tinyLFM2VLJSON(textExtras: textExtras, extras: extras))
        let model = withRandomState(MLXRandom.RandomState(seed: 3)) { LFM2VL(config) }
        eval(model)
        return model
    }

    @Test("LFM2VL parameter keys are unchanged")
    func parameterKeys() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = try tinyModel()
            let keys = Set(model.parameters().flattened().map(\.0))

            var vision: Set<String> = [
                "vision_tower.embeddings.patch_embedding.weight",
                "vision_tower.embeddings.patch_embedding.bias",
                "vision_tower.embeddings.position_embedding.weight",
                "vision_tower.post_layernorm.weight",
                "vision_tower.post_layernorm.bias",
            ]
            for layer in 0 ..< 2 {
                let prefix = "vision_tower.encoder.layers.\(layer)."
                for name in [
                    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                    "self_attn.out_proj", "layer_norm1", "layer_norm2", "mlp.fc1", "mlp.fc2",
                ] {
                    vision.insert(prefix + name + ".weight")
                    vision.insert(prefix + name + ".bias")
                }
            }
            #expect(keys.filter { $0.hasPrefix("vision_tower.") } == vision)
            #expect(
                keys.filter { $0.hasPrefix("multi_modal_projector.") }
                    == Set([
                        "multi_modal_projector.linear_1.weight",
                        "multi_modal_projector.linear_1.bias",
                        "multi_modal_projector.linear_2.weight",
                        "multi_modal_projector.linear_2.bias",
                    ]))
            #expect(!keys.contains { $0.hasPrefix("pixel_unshuffle") })
            #expect(keys.contains("language_model.model.layers.0.conv.conv.weight"))
            #expect(keys.contains("language_model.model.layers.1.self_attn.q_proj.weight"))
            #expect(!keys.contains("language_model.lm_head.weight"))

            let untied = try tinyModel(textExtras: #", "tie_word_embeddings": false"#)
            let untiedKeys = Set(untied.parameters().flattened().map(\.0))
            #expect(untiedKeys == keys.union(["language_model.lm_head.weight"]))
            let topUntied = try tinyModel(
                textExtras: #", "tie_word_embeddings": true"#,
                extras: #" "tie_word_embeddings": false,"#)
            #expect(Set(topUntied.parameters().flattened().map(\.0)) == untiedKeys)
            let topTied = try tinyModel(
                textExtras: #", "tie_word_embeddings": false"#,
                extras: #" "tie_word_embeddings": true,"#)
            #expect(Set(topTied.parameters().flattened().map(\.0)) == keys)
        }
    }

    @Test("sanitize maps Hugging Face names and drops unused keys")
    func sanitizeKeys() throws {
        try Device.withDefaultDevice(.cpu) {
            let hf: [String: MLXArray] = [
                "model.vision_tower.vision_model.embeddings.patch_embedding.weight": MLXArray.zeros(
                    [1]),
                "model.vision_tower.vision_model.embeddings.position_ids": MLXArray.zeros([1]),
                "model.vision_tower.vision_model.encoder.layers.0.mlp.fc1.weight": MLXArray.zeros([
                    1
                ]),
                "model.vision_tower.vision_model.post_layernorm.weight": MLXArray.zeros([1]),
                "model.multi_modal_projector.linear_1.weight": MLXArray.zeros([1]),
                "model.multi_modal_projector.layer_norm.weight": MLXArray.zeros([1]),
                "model.language_model.embed_tokens.weight": MLXArray.zeros([1]),
                "model.language_model.layers.0.conv.conv.weight": MLXArray.zeros([16, 1, 3]),
                "lm_head.weight": MLXArray.zeros([1]),
            ]
            let tied = try tinyModel().sanitize(weights: hf)
            #expect(
                Set(tied.keys) == [
                    "vision_tower.embeddings.patch_embedding.weight",
                    "vision_tower.encoder.layers.0.mlp.fc1.weight",
                    "vision_tower.post_layernorm.weight",
                    "multi_modal_projector.linear_1.weight",
                    "language_model.model.embed_tokens.weight",
                    "language_model.model.layers.0.conv.conv.weight",
                ])
            #expect(tied["language_model.model.layers.0.conv.conv.weight"]?.shape == [16, 3, 1])

            let untiedModel = try tinyModel(
                textExtras: #", "tie_word_embeddings": true"#,
                extras: #" "tie_word_embeddings": false,"#)
            let untied = untiedModel.sanitize(weights: hf)
            #expect(Set(untied.keys) == Set(tied.keys).union(["language_model.lm_head.weight"]))
            #expect(Set(untiedModel.sanitize(weights: untied).keys) == Set(untied.keys))
            let topTied = try tinyModel(
                textExtras: #", "tie_word_embeddings": false"#,
                extras: #" "tie_word_embeddings": true,"#)
            #expect(Set(topTied.sanitize(weights: hf).keys) == Set(tied.keys))
        }
    }

    @Test("inputEmbeddings fills image tokens across a batch in row-major order")
    func batchedInputEmbeddings() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = try tinyModel()
            let image = model.config.imageTokenIndex
            let rows = [[1, image, image, 2, 3], [image, 4, image, image, 5]]
            let features = randomPixels([5, 16], seed: 6)

            let batched = try model.inputEmbeddings(
                tokens: MLXArray(rows.flatMap { $0 }.map { Int32($0) }, [2, 5]),
                imageFeatures: features)
            #expect(batched.shape == [2, 5, 16])

            for (i, (row, range)) in zip(rows, [0 ..< 2, 2 ..< 5]).enumerated() {
                let single = try model.inputEmbeddings(
                    tokens: MLXArray(row.map { Int32($0) })[.newAxis],
                    imageFeatures: features[range])
                #expect(maxAbsDiff(batched[i ..< (i + 1)], single) == 0)
            }
            #expect(maxAbsDiff(batched[1, 0], features[2]) == 0)
        }
    }

    @Test("An untied model projects with lm_head")
    func untiedHead() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = try tinyModel(textExtras: #", "tie_word_embeddings": false"#)
            let text = MLXArray([Int32(1), 2, 3, 4])[.newAxis]
            #expect(abs(model(text, cache: nil)).max().item(Float.self) > 0)
            // With a zero head the logits are zero; tied embeddings would not give that.
            try model.update(
                parameters: ModuleParameters.unflattened([
                    "language_model.lm_head.weight": MLXArray.zeros([32, 16])
                ]), verify: [.noUnusedKeys])
            #expect(abs(model(text, cache: nil)).max().item(Float.self) == 0)
        }
    }

    // MARK: - Projector and image features

    @Test("PixelUnshuffleBlock pads odd sizes and folds 2x2 blocks into channels")
    func pixelUnshuffle() {
        Device.withDefaultDevice(.cpu) {
            let x = MLXArray((0 ..< 15).map { Float($0) }, [1, 3, 5, 1])
            let y = PixelUnshuffleBlock(factor: 2)(x)
            #expect(y.shape == [1, 2, 3, 4])
            // transformers' `pixel_unshuffle` on the zero-padded input.
            #expect(y[0, 0, 0].asArray(Float.self) == [0, 1, 5, 6])
            #expect(y[0, 1, 2].asArray(Float.self) == [14, 0, 0, 0])
        }
    }

    @Test("The model reads only the first h * w patches of each view")
    func modelIgnoresPaddedPatches() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = try tinyModel()
            // Views of 4x4 and 2x4 patches; the second has 8 rows of padding.
            let shapes = [(4, 4), (2, 4)]
            let pixels = randomPixels([2, 16, 12], seed: 4)
            let ids = [1, 2] + Array(repeating: model.config.imageTokenIndex, count: 6) + [3]
            let tokens = MLXArray(ids.map { Int32($0) })[.newAxis]

            func logits(_ pixels: MLXArray) throws -> MLXArray {
                let input = LMInput(
                    text: .init(tokens: tokens),
                    image: .init(pixels: pixels, frames: shapes.map { THW(1, $0.0, $0.1) }))
                guard
                    case .logits(let output) = try model.prepare(
                        input, cache: model.newCache(parameters: nil), state: nil,
                        prefill: PrefillParameters())
                else { throw VLMError.processing("expected logits") }
                return output.logits
            }

            // The padding values do not matter.
            let padding = MLXArray((0 ..< 32).map { $0 >= 24 }, [2, 16, 1])
            let zeroPadded = which(padding, 0, pixels)
            #expect(try maxAbsDiff(logits(zeroPadded), logits(pixels)) == 0)

            // Each view alone, cut to its patches, gives the same features.
            let features = model.imageFeatures(pixels: pixels, spatialShapes: shapes)
            let alone = concatenated(
                shapes.enumerated().map { i, shape in
                    model.imageFeatures(
                        pixels: pixels[i ..< (i + 1), 0 ..< (shape.0 * shape.1)],
                        spatialShapes: [shape])
                }, axis: 0)
            #expect(features.shape == [6, 16])
            #expect(maxAbsDiff(features, alone) == 0)
        }
    }
}
