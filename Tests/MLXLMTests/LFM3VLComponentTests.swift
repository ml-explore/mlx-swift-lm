// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN
import XCTest

@testable import MLXLMCommon
@testable import MLXVLM

final class LFM3VLComponentTests: XCTestCase {
    private func configuration() throws -> LFM3VLConfiguration {
        let json = """
            {"text_config": {"model_type":"lfm3", "vocab_size":32,
              "hidden_size":16, "num_hidden_layers":2, "num_attention_heads":2,
              "num_key_value_heads":1, "head_dim":8, "intermediate_size":24,
              "layer_types":["conv","full_attention"]},
             "vision_config": {"hidden_size":16, "intermediate_size":24,
              "num_attention_heads":2, "num_key_value_heads":1, "patch_size":2,
              "temporal_patch_size":2, "spatial_merge_size":2, "in_channels":3,
              "conv_kernel_size":3, "norm_eps":0.00001, "num_hidden_layers":2,
              "layer_types":["conv","full_attention"],
              "rope_parameters":{"rope_type":"axial","rope_theta":1000000}},
             "image_token_id":10, "video_token_id":11, "downsample_factor":2,
             "projector_hidden_size":64, "projector_bias":true,
             "projector_layer_norm_eps":0.00001, "projector_hidden_act":"gelu"}
            """
        return try JSONDecoder().decode(LFM3VLConfiguration.self, from: Data(json.utf8))
    }

    private func checkpoint(rawNames: Bool = false, quantizedProjector: Bool = false) throws -> URL
    {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("LFM3VLComponents-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let model = LFM3VL(try configuration())
        if quantizedProjector {
            quantize(model: model, groupSize: 64, bits: 8) { path, _ in
                path.hasPrefix("multi_modal_projector.linear_")
            }
        }
        let weights = Dictionary(
            uniqueKeysWithValues: model.parameters().flattened().map {
                name, value in
                let rawName =
                    name.hasPrefix("language_model.model.")
                    ? name.replacingOccurrences(
                        of: "language_model.model.", with: "model.language_model.")
                    : "model.\(name)"
                return (rawNames ? rawName : name, value)
            })
        try MLX.save(arrays: weights, url: directory.appendingPathComponent("model.safetensors"))
        return directory
    }

    private func logits(_ model: LFM3VL, image: Bool) throws -> MLXArray {
        let input = LMInput(
            text: .init(tokens: MLXArray(image ? [1, 10, 2] : [1, 2, 3])),
            image: image ? .init(pixels: MLXArray.ones([4, 24]), frames: [THW(1, 2, 2)]) : nil)
        let cache = try model.newCache(parameters: nil)
        guard
            case .logits(let output) = try model.prepare(
                input, cache: cache, state: nil, prefill: .init())
        else { throw VLMError.processing("Expected prepared logits") }
        eval(output.logits)
        return output.logits
    }

    func testFirstImageMatchesImmediateLoadingWithoutChangingModuleTree() throws {
        for rawNames in [false, true] {
            let directory = try checkpoint(rawNames: rawNames)
            defer { try? FileManager.default.removeItem(at: directory) }
            let lazy = LFM3VL(try configuration())
            let eager = LFM3VL(try configuration())
            try loadWeights(modelDirectory: directory, model: lazy)
            try loadWeights(modelDirectory: directory, model: eager, componentLoading: .immediate)
            let holder = try XCTUnwrap(lazy.onDemandComponents.first)
            let modules = lazy.namedModules().map(\.0)
            let parameters = lazy.parameters().flattened().map(\.0)
            XCTAssertFalse(holder.isLoaded)
            XCTAssertTrue(try lazy.memory().components.isEmpty)
            XCTAssertFalse(parameters.contains { $0.hasPrefix("vision_tower.") })
            XCTAssertFalse(parameters.contains { $0.hasPrefix("multi_modal_projector.") })
            XCTAssertTrue(
                allClose(try logits(lazy, image: false), try logits(eager, image: false)).item(
                    Bool.self))
            XCTAssertFalse(holder.isLoaded)
            XCTAssertTrue(
                allClose(try logits(lazy, image: true), try logits(eager, image: true)).item(
                    Bool.self))
            XCTAssertTrue(holder.isLoaded)
            let memory = try lazy.memory()
            XCTAssertEqual(memory.components.map(\.kind), [.vision])
            XCTAssertGreaterThan(memory.components[0].bytes, 0)
            XCTAssertTrue(
                allClose(try logits(lazy, image: true), try logits(eager, image: true)).item(
                    Bool.self))
            XCTAssertEqual(lazy.namedModules().map(\.0), modules)
            XCTAssertEqual(lazy.parameters().flattened().map(\.0), parameters)
        }
    }

    func testExcludedVisionAllowsTextAndRejectsImages() throws {
        let directory = try checkpoint()
        defer { try? FileManager.default.removeItem(at: directory) }
        let model = LFM3VL(try configuration())
        try loadWeights(modelDirectory: directory, model: model, excludedComponents: [.vision])
        XCTAssertEqual(try logits(model, image: false).shape, [1, 1, 32])
        XCTAssertThrowsError(try logits(model, image: true)) { error in
            guard case OnDemandComponentError.unavailable(.vision) = error else {
                return XCTFail("Unexpected error: \(error)")
            }
        }
    }

    func testTextDoesNotReadMissingVisionTensorsAndFailedLoadCanRetry() throws {
        let directory = try checkpoint()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("model.safetensors")
        let weights = try MLX.loadArrays(url: url)
        eval(Array(weights.values))
        try MLX.save(arrays: weights.filter { $0.key.hasPrefix("language_model.") }, url: url)
        let model = LFM3VL(try configuration())
        try loadWeights(modelDirectory: directory, model: model)
        _ = try logits(model, image: false)
        XCTAssertThrowsError(try logits(model, image: true))
        XCTAssertFalse(model.onDemandComponents[0].isLoaded)
        try MLX.save(arrays: weights, url: url)
        _ = try logits(model, image: true)
        XCTAssertTrue(model.onDemandComponents[0].isLoaded)
    }

    func testLazyProjectorUsesCheckpointPrecisionAndConcurrentLoadsShareOneModule() throws {
        let directory = try checkpoint(rawNames: true, quantizedProjector: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let precision = BaseConfiguration.PerLayerQuantization(
            quantization: .init(groupSize: 64, bits: 4),
            perLayerQuantization: [
                "model.multi_modal_projector.linear_1": .quantize(.init(groupSize: 64, bits: 8)),
                "model.multi_modal_projector.linear_2": .quantize(.init(groupSize: 64, bits: 8)),
            ])
        let lazy = LFM3VL(try configuration())
        let eager = LFM3VL(try configuration())
        try loadWeights(modelDirectory: directory, model: lazy, perLayerQuantization: precision)
        try loadWeights(
            modelDirectory: directory, model: eager, perLayerQuantization: precision,
            componentLoading: .immediate)
        let holder = lazy.onDemandComponents[0]
        final class Loads: @unchecked Sendable {
            let lock = NSLock()
            var identities = Set<ObjectIdentifier>()
            var errors = [String]()
        }
        let loads = Loads()
        DispatchQueue.concurrentPerform(iterations: 8) { _ in
            do {
                let identity = ObjectIdentifier(try holder.loadModule())
                loads.lock.withLock { _ = loads.identities.insert(identity) }
            } catch {
                loads.lock.withLock { loads.errors.append(String(describing: error)) }
            }
        }
        XCTAssertTrue(loads.errors.isEmpty, loads.errors.description)
        XCTAssertEqual(loads.identities.count, 1)
        let modules = Dictionary(
            uniqueKeysWithValues: try holder.loadModule().leafModules().flattened())
        XCTAssertEqual((modules["multi_modal_projector.linear_1"] as? QuantizedLinear)?.bits, 8)
        XCTAssertTrue(
            allClose(try logits(lazy, image: true), try logits(eager, image: true)).item(Bool.self))
    }

    func testCheckpointRenamingPreservesProjectorQuantization() throws {
        let model = LFM3VL(try configuration())
        let checkpoint = ModelCheckpoint(
            weights: ["model.multi_modal_projector.linear_1.weight": MLXArray.ones([64, 64])],
            perLayerQuantization: .init(
                quantization: .init(groupSize: 64, bits: 4),
                perLayerQuantization: ["model.multi_modal_projector.linear_1": .skip]))
        let prepared = try model.prepareCheckpoint(checkpoint)
        XCTAssertNotNil(prepared.weights["multi_modal_projector.linear_1.weight"])
        guard
            case .skip = prepared.perLayerQuantization?.perLayerQuantization[
                "multi_modal_projector.linear_1"]
        else {
            return XCTFail("Projector precision must follow its renamed module")
        }
    }

    func testLMHeadPrecisionNamesAreNormalizedOnce() throws {
        let model = LFM3VL(try configuration())
        for name in ["lm_head", "language_model.lm_head"] {
            let checkpoint = ModelCheckpoint(
                weights: [:],
                perLayerQuantization: .init(
                    quantization: .init(groupSize: 64, bits: 4), perLayerQuantization: [name: .skip]
                ))
            let prepared = try model.prepareCheckpoint(checkpoint)
            XCTAssertEqual(
                Set(prepared.perLayerQuantization?.perLayerQuantization.keys.map { $0 } ?? []),
                ["language_model.lm_head"])
        }
    }
}
