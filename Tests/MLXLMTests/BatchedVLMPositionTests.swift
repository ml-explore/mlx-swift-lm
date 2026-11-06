// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXVLM
import Testing

@Suite(.serialized)
struct BatchedVLMPositionTests {

    @Test(
        "Stateless VLM text rows keep their positions through ragged decode and admission",
        arguments: BatchedVLMPositionFamily.allCases)
    func raggedTextRows(family: BatchedVLMPositionFamily) throws {
        try Device.withDefaultDevice(.cpu) {
            let model = try withRandomState(MLXRandom.RandomState(seed: 741)) {
                try family.makeModel()
            }
            eval(model)
            try expectVLMPositionParity(model)
        }
    }
}

enum BatchedVLMPositionFamily: Sendable, CaseIterable {
    case pixtral
    case lfm2VL
    case museGlimmer

    func makeModel() throws -> any LanguageModel {
        switch self {
        case .pixtral:
            let configuration = try JSONDecoder().decode(
                PixtralConfiguration.self,
                from: Data(
                    """
                    {
                      "model_type": "pixtral", "vocab_size": 64, "image_token_index": 63,
                      "text_config": {
                        "model_type": "mistral", "hidden_size": 32, "num_hidden_layers": 2,
                        "intermediate_size": 64, "num_attention_heads": 4,
                        "num_key_value_heads": 2, "head_dim": 8, "rms_norm_eps": 1e-5,
                        "vocab_size": 64, "rope_theta": 10000.0, "use_qk_norm": true
                      },
                      "vision_config": {
                        "model_type": "pixtral", "hidden_size": 16, "num_hidden_layers": 1,
                        "num_attention_heads": 2, "intermediate_size": 32,
                        "patch_size": 4, "image_size": 8
                      }
                    }
                    """.utf8))
            return PixtralVLM(configuration)
        case .lfm2VL:
            let configuration = try JSONDecoder().decode(
                LFM2VLConfiguration.self,
                from: Data(
                    """
                    {
                      "model_type": "lfm2_vl", "image_token_id": 63,
                      "downsample_factor": 2, "projector_hidden_size": 16,
                      "projector_use_layernorm": false,
                      "text_config": {
                        "model_type": "lfm2", "vocab_size": 64, "hidden_size": 32,
                        "num_hidden_layers": 3, "num_attention_heads": 4,
                        "num_key_value_heads": 2, "intermediate_size": 64, "block_dim": 32,
                        "block_multiple_of": 8, "block_auto_adjust_ff_dim": false,
                        "layer_types": ["full_attention", "conv", "full_attention"]
                      },
                      "vision_config": {
                        "model_type": "siglip2_vision_model", "hidden_size": 16,
                        "intermediate_size": 32, "num_hidden_layers": 1,
                        "num_attention_heads": 2, "num_channels": 3,
                        "num_patches": 16, "patch_size": 2, "layer_norm_eps": 1e-6
                      }
                    }
                    """.utf8))
            return LFM2VL(configuration)
        case .museGlimmer:
            return try MuseGlimmerForwardTests.model()
        }
    }
}

/// Use the model's single-row prefill, then compare each batched forward with its own cache.
private func expectVLMPositionParity(_ model: any LanguageModel) throws {
    func prefill(_ prompt: [Int32]) throws -> [KVCache] {
        let cache = try model.newCache(parameters: nil)
        let output: LMOutput
        switch try model.prepare(
            LMInput(tokens: MLXArray(prompt)), cache: cache, state: nil,
            prefill: .init(stepSize: 2))
        {
        case .logits(let prepared):
            output = prepared
        case .tokens(let remaining):
            let text = remaining.tokens.ndim == 1 ? remaining[text: .newAxis] : remaining
            output = model.nextTokenLogits(text, cache: cache, state: nil)
        }
        try #require(output.state?.isEmpty != false, "these text rows must be stateless")
        eval(cache)
        return cache
    }

    var singles = try [[Int32(1), 2, 3], [4, 5, 6, 7, 8, 9, 10, 11]].map(prefill)
    let batch = try singles[0].indices.map { layer in
        try mergeCaches(singles.map { $0[layer].copy() })
    }
    func decode(_ tokens: [[Int32]]) {
        let input = MLXArray(tokens.flatMap { $0 }).reshaped([tokens.count, tokens[0].count])
        let actual = model.nextTokenLogits(.init(tokens: input), cache: batch, state: nil)
        for row in tokens.indices {
            let expected = model.nextTokenLogits(
                .init(tokens: MLXArray(tokens[row])[.newAxis]), cache: singles[row], state: nil)
            #expect(
                maxAbsDifference(actual.logits[row ..< row + 1], expected.logits) < 1e-4,
                "row \(row), chunk length \(tokens[row].count)")
        }
    }
    decode([[12], [13]])
    decode([[14, 15], [16, 17]])

    let added = try [[Int32(18)], [19, 20, 21, 22, 23]].map(prefill)
    for layer in batch.indices {
        let incoming = try mergeCaches(added.map { $0[layer].copy() })
        try batch[layer].validateExtension(other: incoming)
        batch[layer].extend(other: incoming)
    }
    singles += added
    decode([[24, 25], [26, 27], [28, 29], [30, 31]])

    batch.forEach { $0.filter(batchIndices: batchRows(3, 0)) }
    singles = [singles[3], singles[0]]
    decode([[32, 33, 34], [35, 36, 37]])
    batch.forEach { $0.filter(batchIndices: batchRows(1)) }
    singles = [singles[1]]
    decode([[38]])
}
