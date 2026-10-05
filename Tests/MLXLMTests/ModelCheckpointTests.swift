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
