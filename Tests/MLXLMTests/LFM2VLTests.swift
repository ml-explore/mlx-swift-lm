// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXVLM

// Tests of the LFM2-VL text configuration and output head. They run on the CPU.

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

    // MARK: - Weights and head

    private func tinyModel(textExtras: String = "", extras: String = "") throws -> LFM2VL {
        let config = try decode(
            LFM2VLConfiguration.self, tinyLFM2VLJSON(textExtras: textExtras, extras: extras))
        let model = withRandomState(MLXRandom.RandomState(seed: 3)) { LFM2VL(config) }
        eval(model)
        return model
    }

    @Test("sanitize maps Hugging Face names and drops unused keys")
    func sanitizeKeys() throws {
        try Device.withDefaultDevice(.cpu) {
            let hf: [String: MLXArray] = [
                "model.language_model.embed_tokens.weight": MLXArray.zeros([1]),
                "model.language_model.layers.0.conv.conv.weight": MLXArray.zeros([16, 1, 3]),
                "lm_head.weight": MLXArray.zeros([1]),
            ]
            let tied = try tinyModel().sanitize(weights: hf)
            #expect(
                Set(tied.keys) == [
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
}
