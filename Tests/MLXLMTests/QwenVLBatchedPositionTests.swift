// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN
import MLXVLM
import Testing

@testable import MLXLMCommon

@Suite(.serialized)
struct QwenVLBatchedPositionTests {
    @Test(
        "Stateless Qwen VL text rows use their own positions", arguments: QwenVLTextFamily.allCases)
    func raggedTextLogits(family: QwenVLTextFamily) throws {
        try Device.withDefaultDevice(.cpu) {
            let model = try tinyQwenVLText(family)
            let prompts = [[Int32(1), 2, 3], [4, 5, 6, 7, 8, 9, 10]]
            var singles = try prompts.map { prompt in
                let cache = try model.newCache(parameters: nil)
                let output = model.nextTokenLogits(
                    .init(tokens: MLXArray(prompt)[.newAxis]), cache: cache, state: nil)
                #expect(output.state?.isEmpty != false)
                eval(output.logits, cache)
                return cache
            }
            let batch = try singles[0].indices.map { layer in
                try mergeCaches(singles.map { $0[layer].copy() })
            }
            let first = try #require(batch.first as? BatchKVCache)
            #expect(first.batchOffset.asArray(Int32.self) == [3, 7])

            for length in [1, 2, 1] {
                let tokens = MLXArray((0 ..< 2 * length).map { Int32(11 + $0) })
                    .reshaped([2, length])
                let actual = model.nextTokenLogits(.init(tokens: tokens), cache: batch, state: nil)
                #expect(actual.state?.isEmpty != false)
                for row in singles.indices {
                    let expected = model.nextTokenLogits(
                        .init(tokens: tokens[row ..< row + 1]), cache: singles[row], state: nil)
                    #expect(
                        maxAbsDifference(actual.logits[row ..< row + 1], expected.logits) < 1e-4,
                        "\(family), row \(row), chunk length \(length)")
                }
                let expectedOffsets = singles.map { $0[0].offset }
                #expect(first.batchOffset.asArray(Int.self) == expectedOffsets)
            }
            batch.forEach { $0.filter(batchIndices: batchRows(0)) }
            singles = [singles[0]]
            if case .scalar(let offset) = first.ropeOffset {
                #expect(offset == singles[0][0].offset)
            } else {
                Issue.record("A remaining single row must use the scalar RoPE path")
            }
            let tokens = MLXArray([Int32(21), 22])[.newAxis]
            let actual = model.nextTokenLogits(.init(tokens: tokens), cache: batch, state: nil)
            let expected = model.nextTokenLogits(
                .init(tokens: tokens), cache: singles[0], state: nil)
            #expect(maxAbsDifference(actual.logits, expected.logits) < 1e-4)
        }
    }

}

enum QwenVLTextFamily: String, CaseIterable, Sendable {
    case qwen2 = "qwen2_vl"
    case qwen25 = "qwen2_5_vl"
}

func tinyQwenVLText(_ family: QwenVLTextFamily) throws -> any LanguageModel {
    var configuration: [String: Any] = [
        "model_type": family.rawValue, "hidden_size": 64, "num_hidden_layers": 2,
        "intermediate_size": 128, "num_attention_heads": 4, "num_key_value_heads": 2,
        "rms_norm_eps": 1e-6, "vocab_size": 64, "max_position_embeddings": 4096,
        "rope_theta": 1000000.0, "rope_traditional": false, "tie_word_embeddings": false,
        "image_token_id": 60, "video_token_id": 61, "vision_start_token_id": 62,
        "vision_end_token_id": 63, "vision_token_id": 59,
        "rope_scaling": ["type": "mrope", "mrope_section": [2, 3, 3]],
    ]
    if family == .qwen2 {
        configuration["vision_config"] = [
            "depth": 1, "embed_dim": 32, "hidden_size": 64, "num_heads": 2,
            "patch_size": 16, "mlp_ratio": 2.0, "spatial_patch_size": 16,
            "spatial_merge_size": 2, "temporal_patch_size": 2, "in_channels": 3,
        ]
    } else {
        configuration["sliding_window"] = 32768
        configuration["use_sliding_window"] = false
        configuration["max_window_layers"] = 2
        configuration["vision_config"] = [
            "depth": 1, "hidden_size": 32, "intermediate_size": 64, "out_hidden_size": 64,
            "num_heads": 2, "patch_size": 16, "spatial_patch_size": 16,
            "spatial_merge_size": 2, "temporal_patch_size": 2, "window_size": 64,
            "fullatt_block_indexes": [0], "tokens_per_second": 2, "in_chans": 3,
        ]
    }
    let data = try JSONSerialization.data(withJSONObject: configuration)
    let model: any LanguageModel
    switch family {
    case .qwen2:
        let config = try JSONDecoder().decode(Qwen2VLConfiguration.self, from: data)
        model = withRandomState(MLXRandom.RandomState(seed: 917)) { Qwen2VL(config) }
    case .qwen25:
        let config = try JSONDecoder().decode(Qwen25VLConfiguration.self, from: data)
        model = withRandomState(MLXRandom.RandomState(seed: 918)) { Qwen25VL(config) }
    }
    model.train(false)
    eval(model)
    return model
}
