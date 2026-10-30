// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import Testing

@testable import MLXLLM
@testable import MLXLMCommon
@testable import MLXVLM

/// Models do not read the towers and heads of a checkpoint that they drop.
@Suite(.serialized)
struct UnreadCheckpointTowersTests {

    @Test func gemma4TextModelSkipsVisionAndAudio() throws {
        let model = MLXLLM.Gemma4Model(
            try decode(
                MLXLLM.Gemma4Configuration.self,
                #"{"model_type": "gemma4", "#
                    + #""text_config": \#(gemma4TextConfigJSON)}"#))
        try expectUnread(
            [
                "vision_tower.encoder.layers.0.mlp.weight", "embed_vision.projection.weight",
                "model.vision_embedder.patch_dense.weight", "audio_tower.layers.0.weight",
                "model.embed_audio.projection.weight",
            ], loading: model)
    }

    @Test func gemma4VLMSkipsAudio() throws {
        let model = MLXVLM.Gemma4(
            try decode(
                MLXVLM.Gemma4Configuration.self,
                #"{"model_type": "gemma4", "text_config": \#(gemma4TextConfigJSON), "#
                    + #""vision_config": \#(gemma4VisionConfigJSON)}"#))
        try expectUnread(
            ["audio_tower.layers.0.weight", "embed_audio.projection.weight"], loading: model)
    }

    @Test func qwen35TextModelSkipsVisionAndMTP() throws {
        let model = MLXLLM.Qwen35Model(
            try decode(
                MLXLLM.Qwen35Configuration.self,
                #"{"model_type": "qwen3_5", "text_config": \#(qwen35TextConfigJSON)}"#))
        try expectUnread(
            [
                "vision_tower.blocks.0.attn.qkv.weight", "model.visual.merger.weight",
                "mtp.fc.weight",
            ], loading: model)
    }

    @Test func qwen35VLMSkipsMTP() throws {
        let model = MLXVLM.Qwen35(
            try decode(
                MLXVLM.Qwen35Configuration.self,
                #"{"model_type": "qwen3_5", "text_config": \#(qwen35TextConfigJSON), "#
                    + #""vision_config": \#(qwen35VisionConfigJSON)}"#))
        try expectUnread(["mtp.fc.weight", "language_model.mtp.norm.weight"], loading: model)
    }
}

/// Saves the model's own tensors together with `unread`, then expects the model to load and
/// the loader to read only the tensors the loaded model holds.
private func expectUnread(_ unread: [String], loading model: any BaseLanguageModel) throws {
    let directory = FileManager.default.temporaryDirectory
        .appendingPathComponent("UnreadCheckpointTowersTests-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }

    var weights = Dictionary(
        uniqueKeysWithValues: model.parameters().flattened())
    for name in unread {
        weights[name] = MLXArray.zeros([2, 2])
    }
    let url = directory.appendingPathComponent("model.safetensors")
    try save(arrays: weights, url: url)

    try loadWeights(modelDirectory: directory, model: model)
    let skipped =
        try prepareComponents(
            of: model, urls: [url], excluded: [], loading: .onFirstUse, perLayerQuantization: nil)
        + model.excludedCheckpointNamespaces
    let read = try loadModelCheckpoint(urls: [url], excludedNamespaces: skipped).weights.keys
    #expect(Set(read) == Set(model.parameters().flattened().map(\.0)))
}

private func decode<T: Decodable>(_ type: T.Type, _ json: String) throws -> T {
    try JSONDecoder().decode(type, from: Data(json.utf8))
}

private let gemma4TextConfigJSON = """
    {
        "model_type": "gemma4_text", "hidden_size": 32, "num_hidden_layers": 2,
        "intermediate_size": 64, "num_attention_heads": 2, "num_key_value_heads": 1,
        "head_dim": 16, "global_head_dim": 32, "vocab_size": 64,
        "vocab_size_per_layer_input": 64, "num_kv_shared_layers": 0,
        "hidden_size_per_layer_input": 8, "sliding_window": 16, "sliding_window_pattern": 2,
        "max_position_embeddings": 128
    }
    """

private let gemma4VisionConfigJSON = """
    {
        "num_hidden_layers": 1, "hidden_size": 16, "intermediate_size": 32,
        "num_attention_heads": 2, "head_dim": 8, "patch_size": 4, "default_output_length": 4,
        "pooling_kernel_size": 1, "position_embedding_size": 8
    }
    """

private let qwen35TextConfigJSON = """
    {
        "model_type": "qwen3_5_text", "hidden_size": 16, "num_hidden_layers": 2,
        "intermediate_size": 32, "num_attention_heads": 2, "num_key_value_heads": 1,
        "head_dim": 8, "linear_num_value_heads": 2, "linear_num_key_heads": 1,
        "linear_key_head_dim": 8, "linear_value_head_dim": 8, "linear_conv_kernel_dim": 2,
        "rms_norm_eps": 1e-6, "vocab_size": 16, "max_position_embeddings": 64,
        "tie_word_embeddings": true, "full_attention_interval": 2,
        "num_experts": 0, "num_experts_per_tok": 0
    }
    """

private let qwen35VisionConfigJSON = """
    {
        "model_type": "qwen3_5_vit", "depth": 1, "hidden_size": 16, "intermediate_size": 32,
        "out_hidden_size": 16, "num_heads": 2, "patch_size": 2, "spatial_merge_size": 1,
        "temporal_patch_size": 1, "num_position_embeddings": 16
    }
    """
