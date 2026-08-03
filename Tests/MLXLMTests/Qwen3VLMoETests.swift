// Copyright © 2026 Apple Inc.

import Foundation
import MLXVLM
import XCTest

final class Qwen3VLMoETests: XCTestCase {

    private func minimalConfigData(
        tieWordEmbeddings: Bool = false
    ) -> Data {
        let json = """
            {
                "model_type": "qwen3_vl_moe",
                "text_config": {
                    "model_type": "qwen3_vl_moe",
                    "hidden_size": 8,
                    "num_hidden_layers": 1,
                    "intermediate_size": 16,
                    "num_attention_heads": 1,
                    "num_key_value_heads": 1,
                    "num_experts": 2,
                    "num_experts_per_tok": 1,
                    "decoder_sparse_step": 1,
                    "mlp_only_layers": [],
                    "moe_intermediate_size": 12,
                    "rms_norm_eps": 0.000001,
                    "vocab_size": 32,
                    "head_dim": 8,
                    "rope_theta": 1000000,
                    "max_position_embeddings": 32768,
                    "rope_scaling": {
                        "rope_type": "default",
                        "mrope_section": [24, 20, 20]
                    },
                    "tie_word_embeddings": \(tieWordEmbeddings)
                },
                "vision_config": {
                    "model_type": "qwen3_vl_moe",
                    "depth": 1,
                    "hidden_size": 8,
                    "intermediate_size": 16,
                    "out_hidden_size": 8,
                    "num_heads": 1,
                    "patch_size": 16,
                    "spatial_merge_size": 1,
                    "temporal_patch_size": 1,
                    "num_position_embeddings": 8
                }
            }
            """
        return Data(json.utf8)
    }

    private func makeMinimalConfig(
        tieWordEmbeddings: Bool = false
    ) throws -> Qwen3VLMoEConfiguration {
        try JSONDecoder().decode(
            Qwen3VLMoEConfiguration.self,
            from: minimalConfigData(tieWordEmbeddings: tieWordEmbeddings))
    }

    func testConfigurationDecodesMRoPESection() throws {
        let config = try makeMinimalConfig()

        XCTAssertEqual(config.modelType, "qwen3_vl_moe")
        XCTAssertEqual(config.textConfiguration.ropeScaling?.mropeSection, [24, 20, 20])
    }

    func testConfigurationDecodesMoEFieldsAndDefaults() throws {
        let config = try makeMinimalConfig()

        XCTAssertEqual(config.imageTokenIndex, 151_655)
        XCTAssertEqual(config.videoTokenIndex, 151_656)
        XCTAssertEqual(config.textConfiguration.numExperts, 2)
        XCTAssertEqual(config.textConfiguration.numExpertsPerTok, 1)
        XCTAssertEqual(config.textConfiguration.decoderSparseStep, 1)
        XCTAssertEqual(config.textConfiguration.moeIntermediateSize, 12)
        XCTAssertEqual(config.textConfiguration.numKeyValueHeads, 1)
        XCTAssertFalse(config.textConfiguration.tieWordEmbeddings)
    }
}
