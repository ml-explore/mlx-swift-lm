// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXLLM

final class Gemma4LoRATests: XCTestCase {

    private func configuration() throws -> Gemma4TextConfiguration {
        let json = """
            {
                "model_type": "gemma4_text",
                "hidden_size": 64,
                "num_hidden_layers": 4,
                "intermediate_size": 128,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 32,
                "global_head_dim": 32,
                "num_kv_shared_layers": 1,
                "hidden_size_per_layer_input": 8,
                "vocab_size": 32,
                "vocab_size_per_layer_input": 32,
                "sliding_window": 8,
                "sliding_window_pattern": 2,
                "layer_types": ["sliding_attention", "full_attention", "sliding_attention", "full_attention"],
                "use_double_wide_mlp": false
            }
            """
        return try JSONDecoder().decode(Gemma4TextConfiguration.self, from: Data(json.utf8))
    }

    /// Regression for Sightline #873. `loraLayers` used to return only the
    /// attention modules, so a PEFT adapter that also targets `mlp.*` (every
    /// Sightline adapter does) never matched a module and `load` failed
    /// `.noUnusedKeys`. Returning the decoder layers matches Gemma3/Qwen/Llama
    /// (and the Gemma4 VLM wrapper) and lets both key families resolve.
    ///
    /// `selfAttn` / `mlp` on `Gemma4DecoderLayer` and `qProj` / `gateProj` on
    /// the attention/MLP classes are `fileprivate`/`private`, so we assert
    /// through the module tree instead of naming them: `LoRAContainer` wraps the
    /// targeted `Linear`s in `LoRALinear`, one per (layer × key). With four
    /// decoder layers and the two keys below that is eight — before the patch it
    /// is zero, because `load` throws first.
    func testLoRALayersExposeAttentionAndMLP() throws {
        let model = Gemma4TextModel(try configuration())
        let rank = 4
        let adapter = LoRAContainer(
            configuration: .init(
                numLayers: 4,
                loraParameters: .init(
                    rank: rank, scale: 1.0, keys: ["self_attn.q_proj", "mlp.gate_proj"])),
            parameters: ModuleParameters.unflattened([
                "model.layers.0.self_attn.q_proj.lora_a": MLXArray.zeros([64, rank]),
                "model.layers.0.self_attn.q_proj.lora_b": MLXArray.zeros([rank, 64]),
                "model.layers.0.mlp.gate_proj.lora_a": MLXArray.zeros([64, rank]),
                "model.layers.0.mlp.gate_proj.lora_b": MLXArray.zeros([rank, 128]),
            ]))
        try adapter.load(into: model)

        let wrapped = model.modules().compactMap { $0 as? LoRALinear }
        XCTAssertEqual(wrapped.count, 8)
    }
}
