// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN
import XCTest

@testable import MLXLMCommon
@testable import MLXVLM

final class LFM2VLComponentTests: XCTestCase {
    private func configuration() throws -> LFM2VLConfiguration {
        try JSONDecoder().decode(
            LFM2VLConfiguration.self,
            from: Data(
                #"""
                {
                  "model_type": "lfm2_vl", "image_token_id": 7,
                  "downsample_factor": 2, "projector_hidden_size": 64,
                  "text_config": {
                    "model_type": "lfm2", "hidden_size": 64,
                    "num_hidden_layers": 1, "num_attention_heads": 4,
                    "num_key_value_heads": 2, "vocab_size": 128,
                    "block_ff_dim": 64, "block_auto_adjust_ff_dim": false
                  },
                  "vision_config": {
                    "model_type": "siglip2_vision_model", "hidden_size": 16,
                    "intermediate_size": 32, "num_hidden_layers": 1,
                    "num_attention_heads": 4, "image_size": 4,
                    "patch_size": 2, "num_patches": 4
                  }
                }
                """#.utf8))
    }

    private func checkpoint(rawNames: Bool = false, quantizedProjector: Bool = false) throws -> URL
    {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("LFM2VLComponents-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let model = LFM2VL(try configuration())
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

    private func logits(_ model: LFM2VL, image: Bool) throws -> MLXArray {
        let input = LMInput(
            text: .init(tokens: MLXArray(image ? [1, 7, 2] : [1, 2, 3])),
            image: image ? .init(pixels: MLXArray.ones([1, 4, 12]), frames: [THW(1, 2, 2)]) : nil)
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
            let lazy = LFM2VL(try configuration())
            let eager = LFM2VL(try configuration())
            try loadWeights(modelDirectory: directory, model: lazy)
            try loadWeights(modelDirectory: directory, model: eager, componentLoading: .immediate)
            let holder = try XCTUnwrap(lazy.onDemandComponents.first)
            let modules = lazy.namedModules().map(\.0)
            let parameters = lazy.parameters().flattened().map(\.0)
            XCTAssertFalse(holder.isLoaded)
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
        let model = LFM2VL(try configuration())
        try loadWeights(modelDirectory: directory, model: model, excludedComponents: [.vision])
        XCTAssertEqual(try logits(model, image: false).shape, [1, 1, 128])
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
        let model = LFM2VL(try configuration())
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
        let lazy = LFM2VL(try configuration())
        let eager = LFM2VL(try configuration())
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
        let model = LFM2VL(try configuration())
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
}
