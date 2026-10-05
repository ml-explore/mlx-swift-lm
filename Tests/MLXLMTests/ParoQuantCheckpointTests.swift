// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN
import MLXScriptedLM
import Testing

@testable import MLXLMCommon

struct ParoQuantCheckpointTests {
    @Test(arguments: [false, true])
    func checkpointPreparationPreservesPrecision(prepared: Bool) async throws {
        let directory = try makeCheckpoint(prepared: prepared)
        defer { try? FileManager.default.removeItem(at: directory) }
        let container = try await loadParoQuantModel(
            from: directory,
            typeRegistry: ModelTypeRegistry(creators: [
                "qwen3_5": { _ in CheckpointModel() }
            ]),
            tokenizerLoader: TestTokenizerLoader())
        try await container.perform { context in
            let model = try #require(context.model as? CheckpointModel)
            let layer = try #require(model.layer as? QuantizedLinear)
            #expect(layer.bits == 8)
            #expect(layer.weight.asArray(UInt32.self).allSatisfy { $0 == 17 })
        }
    }

    @Test(arguments: [false, true])
    func checkpointPreparationPropagatesCollisions(prepared: Bool) async throws {
        let directory = try makeCheckpoint(prepared: prepared, collision: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        await #expect(throws: ModelCheckpoint.MappingError.self) {
            try await loadParoQuantModel(
                from: directory,
                typeRegistry: ModelTypeRegistry(creators: [
                    "qwen3_5": { _ in CheckpointModel() }
                ]),
                tokenizerLoader: TestTokenizerLoader())
        }
    }

    private func makeCheckpoint(prepared: Bool, collision: Bool = false) throws -> URL {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
            UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let config = """
            {
                "model_type": "qwen3_5",
                "architectures": ["Qwen3_5ForConditionalGeneration"],
                "quantization_config": {
                    "quant_method": "paroquant", "bits": 4, "group_size": 32, "krot": 8
                }
            }
            """
        try Data(config.utf8).write(to: directory.appendingPathComponent("config.json"))
        var weights = [
            "source.layer.weight": MLXArray.full([32, 8], values: MLXArray(UInt32(17))),
            "source.layer.scales": MLXArray.ones([32, 1]),
            "source.layer.biases": MLXArray.zeros([32, 1]),
            "mtp.layer.weight": MLXArray.zeros([32, 32]),
        ]
        if collision {
            weights["layer.weight"] = MLXArray.zeros([32, 8], type: UInt32.self)
        }
        try save(arrays: weights, url: directory.appendingPathComponent("model.safetensors"))
        if prepared {
            ParoQuantPreparedCheckpoint.write(
                weights: weights,
                manifest: try ParoQuantPreparedCheckpoint.currentManifest(directory: directory),
                directory: directory)
            #expect(
                FileManager.default.fileExists(
                    atPath: directory.appendingPathComponent(
                        ParoQuantPreparedCheckpoint.fileName
                    ).path))
        }
        return directory
    }

    private enum TestError: Error {
        case legacySanitizerCalled
    }

    private struct TestTokenizerLoader: TokenizerLoader {
        func load(from directory: URL) async throws -> any Tokenizer {
            PseudoWordTokenizer()
        }
    }

    private final class CheckpointModel: Module, LanguageModel, KVCacheDimensionProvider {
        @ModuleInfo var layer: Linear
        let kvHeads: [Int] = []

        override init() {
            _layer.wrappedValue = Linear(32, 32, bias: false)
            super.init()
        }

        func prepareCheckpoint(_ checkpoint: ModelCheckpoint) throws -> ModelCheckpoint {
            var checkpoint = checkpoint
            checkpoint.perLayerQuantization?.perLayerQuantization["source.layer"] =
                .quantize(.init(groupSize: 32, bits: 8))
            return try checkpoint.mapNames(
                using: .init([
                    .excludePrefix("mtp"), .replacePrefix("source", with: ""),
                ]))
        }

        func sanitize(weights: [String: MLXArray]) throws -> [String: MLXArray] {
            throw TestError.legacySanitizerCalled
        }

        func prepare(
            _ input: LMInput, cache: [KVCache], state: LMOutput.State?, prefill: PrefillParameters
        ) throws -> PrepareResult {
            .tokens(input.text)
        }
    }
}
