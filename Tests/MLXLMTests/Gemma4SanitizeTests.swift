// Copyright © 2026 Apple Inc.

import Foundation
import Testing
@testable import MLXVLM

struct Gemma4SanitizeTests {

    @Test("Gemma4 sanitize strips only text-backbone shared-layer KV weights")
    func sanitizeScopesSharedKVWeightStripToTextBackbone() throws {
        let config = try Self.makeTinyConfig()
        let textConfig = config.textConfiguration

        #expect(
            !Gemma4.isRedundantTextKVSharedWeight(
                "language_model.model.layers.3.self_attn.k_proj.weight",
                textConfig: textConfig))
        #expect(
            Gemma4.isRedundantTextKVSharedWeight(
                "language_model.model.layers.4.self_attn.k_proj.weight",
                textConfig: textConfig))
        #expect(
            Gemma4.isRedundantTextKVSharedWeight(
                "language_model.model.layers.5.self_attn.v_proj.weight",
                textConfig: textConfig))
        #expect(
            Gemma4.isRedundantTextKVSharedWeight(
                "language_model.model.layers.5.self_attn.k_norm.weight",
                textConfig: textConfig))
        #expect(
            Gemma4.isRedundantTextKVSharedWeight(
                "language_model.model.layers.5.self_attn.v_norm.weight",
                textConfig: textConfig))
        #expect(
            !Gemma4.isRedundantTextKVSharedWeight(
                "vision_tower.layers.5.self_attn.k_proj.linear.weight",
                textConfig: textConfig))
    }

    @Test("Gemma4 text backbone marks only shared tail layers as KV-shared-only")
    func sharedTailLayersDoNotRequireLocalKVProjectionWeights() throws {
        let config = try Self.makeTinyConfig()
        let textConfig = config.textConfiguration

        #expect(!Gemma4TextBackbone.isKVSharedOnlyLayer(3, textConfig: textConfig))
        #expect(Gemma4TextBackbone.isKVSharedOnlyLayer(4, textConfig: textConfig))
        #expect(Gemma4TextBackbone.isKVSharedOnlyLayer(5, textConfig: textConfig))
    }

    private static func makeTinyConfig() throws -> Gemma4Configuration {
        let json = """
            {
                "text_config": {
                    "hidden_size": 64,
                    "num_hidden_layers": 6,
                    "intermediate_size": 128,
                    "num_attention_heads": 2,
                    "num_key_value_heads": 1,
                    "head_dim": 16,
                    "global_head_dim": 32,
                    "vocab_size": 200,
                    "vocab_size_per_layer_input": 200,
                    "num_kv_shared_layers": 2,
                    "hidden_size_per_layer_input": 8,
                    "sliding_window": 8,
                    "sliding_window_pattern": 3,
                    "max_position_embeddings": 4096
                },
                "vision_config": {
                    "num_hidden_layers": 1,
                    "hidden_size": 16,
                    "intermediate_size": 32,
                    "num_attention_heads": 2,
                    "head_dim": 8,
                    "position_embedding_size": 64
                }
            }
            """
        return try JSONDecoder().decode(
            Gemma4Configuration.self, from: Data(json.utf8))
    }
}
