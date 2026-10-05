// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

struct Gemma3nCheckpointTests {
    @Test
    func wrapperRenamePreservesMixedPrecisionDuringStrictLoading() throws {
        let config = try configuration()
        let source = Gemma3nTextModel(config: config)
        quantize(model: source) { path, _ in
            if path.hasSuffix("self_attn.q_proj") { return (32, 8, .affine) }
            if path.hasSuffix("self_attn.k_proj") { return (32, 4, .affine) }
            return nil
        }
        let weights = Dictionary(
            uniqueKeysWithValues: source.parameters().flattened().map {
                ("model." + $0.0, $0.1)
            })
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
            UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try save(arrays: weights, url: directory.appendingPathComponent("model.safetensors"))
        let loaded = Gemma3nTextModel(config: config)
        let settings = BaseConfiguration.PerLayerQuantization(
            quantization: .init(groupSize: 32, bits: 4),
            perLayerQuantization: Dictionary(
                uniqueKeysWithValues: (0 ..< 2).map {
                    (
                        "model.language_model.layers.\($0).self_attn.q_proj",
                        .quantize(.init(groupSize: 32, bits: 8))
                    )
                }))
        try loadWeights(modelDirectory: directory, model: loaded, perLayerQuantization: settings)
        for layer in loaded.languageModel.layers {
            #expect((layer.selfAttn.qProj as? QuantizedLinear)?.bits == 8)
            #expect((layer.selfAttn.kProj as? QuantizedLinear)?.bits == 4)
        }
        let expected = Dictionary(uniqueKeysWithValues: source.parameters().flattened())
        for (name, value) in loaded.parameters().flattened() {
            #expect(arrayEqual(value, try #require(expected[name])).item(Bool.self))
        }
    }

    @Test
    func wrapperAliasesFailBeforeUpdatingTheModel() throws {
        let model = Gemma3nTextModel(config: try configuration())
        let weights = [
            "model.language_model.norm.weight": MLXArray(1),
            "language_model.norm.weight": MLXArray(2),
        ]
        #expect(throws: ModelCheckpoint.MappingError.self) {
            try model.prepareCheckpoint(.init(weights: weights))
        }
        let baseModel: any BaseLanguageModel = model
        #expect(throws: ModelCheckpoint.MappingError.self) {
            try baseModel.sanitize(weights: weights)
        }
        #expect(throws: ModelCheckpoint.MappingError.self) {
            try baseModel.sanitize(weights: weights, metadata: [:])
        }
    }

    private func configuration() throws -> Gemma3nTextConfiguration {
        let data = try JSONEncoder().encode(Gemma3nTextConfiguration())
        var values = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
        values.merge([
            "hidden_size": 64, "num_hidden_layers": 2, "intermediate_size": [64, 64],
            "num_attention_heads": 2, "head_dim": 32, "vocab_size": 16,
            "num_key_value_heads": 2, "vocab_size_per_layer_input": 16,
            "hidden_size_per_layer_input": 32, "altup_num_inputs": 2, "laurel_rank": 8,
            "layer_types": ["sliding_attention", "full_attention"],
            "activation_sparsity_pattern": [0, 0],
        ]) { _, new in new }
        return try JSONDecoder().decode(
            Gemma3nTextConfiguration.self, from: JSONSerialization.data(withJSONObject: values))
    }
}
