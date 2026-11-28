// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLMCommon

struct ModelCheckpointTests {
    @Test
    func legacySanitizationErrorsReachCheckpointPreparation() {
        final class FailingModel: Module, BaseLanguageModel {
            enum Failure: Error { case sanitization }

            func sanitize(weights: [String: MLXArray]) throws -> [String: MLXArray] {
                throw Failure.sanitization
            }
        }
        let model: any BaseLanguageModel = FailingModel()
        #expect(throws: FailingModel.Failure.self) {
            try model.prepareCheckpoint(.init(weights: [:]))
        }
    }

    @Test
    func nameMappingsPreservePrecisionAndSourceMetadata() throws {
        let checkpoint = ModelCheckpoint(
            weights: ["head.fc.weight": MLXArray.zeros([32, 32])],
            metadata: ["format": "mlx"], weightMetadata: ["head.fc.weight": [:]],
            perLayerQuantization: .init(
                quantization: .init(groupSize: 32, bits: 4),
                perLayerQuantization: [
                    "head.fc": .quantize(.init(groupSize: 32, bits: 8)),
                    "target.fc": .skip,
                ]))
        let mapped = try checkpoint.mapNames { name in
            name.hasPrefix("head.") ? "mtp." + name.dropFirst(5) : nil
        }
        #expect(mapped.weights["mtp.fc.weight"] != nil)
        #expect(mapped.metadata(forWeight: "mtp.fc.weight").isEmpty)
        #expect(mapped.metadata["format"] == "mlx")
        #expect(mapped.perLayerQuantization?.quantization(layer: "mtp.fc")?.bits == 8)
        #expect(mapped.perLayerQuantization?.quantization(layer: "mtp.other")?.bits == 4)
        #expect(mapped.perLayerQuantization?.perLayerQuantization["target.fc"] == nil)
    }

    @Test
    func patternRulesRunInOrderOnTensorsAndLayerSettings() throws {
        let mapping = CheckpointNameMapping([
            .replaceMatches(#"^model\.llm\."#, with: "model.language_model."),
            .replaceMatches(
                #"\.attn_res\.mlp_queries\.(\d+)$"#, with: ".layers.$1.mlp_input.query"),
            .replaceMatches(#"\.(q|k)_norm\."#, with: ".$1_layernorm."),
        ])
        let checkpoint = ModelCheckpoint(
            weights: [
                "model.llm.attn_res.mlp_queries.12": MLXArray(1),
                "model.llm.layers.3.k_norm.weight": MLXArray(2),
                "lm_head.weight": MLXArray(3),
            ],
            perLayerQuantization: .init(perLayerQuantization: [
                "model.llm.layers.3.k_norm": .skip
            ]))
        let mapped = try checkpoint.mapNames(using: mapping)
        #expect(
            Set(mapped.weights.keys) == [
                "model.language_model.layers.12.mlp_input.query",
                "model.language_model.layers.3.k_layernorm.weight", "lm_head.weight",
            ])
        // A layer setting names a module, so the parameter-name rule leaves it.
        #expect(
            mapped.perLayerQuantization?.perLayerQuantization.keys.first
                == "model.language_model.layers.3.k_norm")
    }

    @Test
    func tensorAliasesCannotOverwriteEachOther() {
        let checkpoint = ModelCheckpoint(weights: [
            "a.weight": MLXArray(1), "b.weight": MLXArray(2),
        ])
        #expect(throws: ModelCheckpoint.MappingError.self) {
            try checkpoint.mapNames { _ in "weight" }
        }
    }

    @Test
    func layerSettingAliasesCannotOverwriteEachOther() {
        let checkpoint = ModelCheckpoint(
            weights: [:],
            perLayerQuantization: .init(perLayerQuantization: [
                "a": .skip, "b": .quantize(.init(groupSize: 32, bits: 8)),
            ]))
        #expect(throws: ModelCheckpoint.MappingError.self) {
            try checkpoint.mapNames { _ in "layer" }
        }
    }

    @Test
    func shardMetadataStaysWithTheTensorThatSuppliedIt() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
            UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let first = directory.appendingPathComponent("model.safetensors")
        let second = directory.appendingPathComponent("mtp.safetensors")
        try save(
            arrays: ["target.weight": MLXArray(1), "shared.weight": MLXArray(1)],
            metadata: ["format": "mlx"], url: first)
        try save(arrays: ["mtp.weight": MLXArray(2), "shared.weight": MLXArray(2)], url: second)
        let checkpoint = try loadModelCheckpoint(urls: [first, second])
        #expect(checkpoint.metadata["format"] == "mlx")
        #expect(checkpoint.metadata(forWeight: "target.weight")["format"] == "mlx")
        #expect(checkpoint.metadata(forWeight: "mtp.weight").isEmpty)
        #expect(checkpoint.metadata(forWeight: "shared.weight").isEmpty)
        #expect(checkpoint.weights["shared.weight"]?.item(Int.self) == 2)
    }
}
