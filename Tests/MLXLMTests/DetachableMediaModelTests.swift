// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN
import XCTest

@testable import MLXLMCommon

/// The Muse-Glimmer layout in miniature: a bf16 tower and a quantized top-level projection.
private final class TwoTowerModel: Module, BaseLanguageModel, DetachableMediaModel {
    @ModuleInfo(key: "language_model") var language: Linear
    @ModuleInfo(key: "vision_tower") var tower: Linear?
    @ModuleInfo(key: "vision_projection") var projection: Linear?

    var mediaWeightSource: MediaWeightSource?

    override init() {
        _language.wrappedValue = Linear(64, 64, bias: false)
        _tower.wrappedValue = Linear(64, 64, bias: false)
        _projection.wrappedValue = Linear(64, 64, bias: false)
    }

    var mediaModuleKeys: [String] { ["vision_tower", "vision_projection"] }

    func makeMediaModules() -> [String: Module] {
        [
            "vision_tower": Linear(64, 64, bias: false),
            "vision_projection": Linear(64, 64, bias: false),
        ]
    }

    // Raw checkpoints scope everything under `model.`, as `transformers` exports do.
    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        Dictionary(
            uniqueKeysWithValues: weights.map { key, value in
                (key.hasPrefix("model.") ? String(key.dropFirst("model.".count)) : key, value)
            })
    }
}

final class DetachableMediaModelTests: XCTestCase {

    private let quantization = BaseConfiguration.Quantization(groupSize: 64, bits: 4)

    func testPrefixMatchingCoversRawAndSanitizedNames() {
        let prefixes = ["vision_tower.", "vision_adapter."]

        XCTAssertTrue(matchesWeightPrefixes("vision_tower.ln_pre.weight", prefixes: prefixes))
        XCTAssertTrue(
            matchesWeightPrefixes("model.vision_tower.layers.0.q.weight", prefixes: prefixes))
        XCTAssertFalse(
            matchesWeightPrefixes("language_model.model.layers.0.q.weight", prefixes: prefixes))
        XCTAssertFalse(matchesWeightPrefixes("vision_tower_extra.weight", prefixes: prefixes))
    }

    func testDetachedLoadDoesNotReadTheMediaWeights() throws {
        let directory = try writeCheckpoint()
        defer { try? FileManager.default.removeItem(at: directory) }

        let model = try detachedModel(loadedFrom: directory)

        XCTAssertFalse(model.mediaModulesAreAttached)
        XCTAssertNil(model.tower)
        XCTAssertNil(model.projection)
        XCTAssertEqual(
            Set(model.parameters().flattened().map(\.0)),
            ["language_model.weight", "language_model.scales", "language_model.biases"])

        let prefixes = model.mediaWeightPrefixes
        let (weights, _) = try loadWeightArrays(
            urls: [directory.appendingPathComponent("model.safetensors")]
        ) { !matchesWeightPrefixes($0, prefixes: prefixes) }
        XCTAssertFalse(weights.keys.contains { $0.contains("vision") })
    }

    /// Reloading must end with exactly the model an eager load builds, including a quantized
    /// module at the top of the tree.
    func testLoadAfterDetachMatchesAnEagerLoad() throws {
        let directory = try writeCheckpoint()
        defer { try? FileManager.default.removeItem(at: directory) }

        let eager = TwoTowerModel()
        try loadWeights(modelDirectory: directory, model: eager, quantization: quantization)

        let reloaded = try detachedModel(loadedFrom: directory)
        let language = reloaded.language
        try reloaded.loadMediaModules()

        XCTAssertTrue(reloaded.mediaModulesAreAttached)
        XCTAssertTrue(reloaded.projection is QuantizedLinear)
        XCTAssertFalse(reloaded.tower is QuantizedLinear)
        XCTAssertTrue(reloaded.language === language, "loading leaves the language model alone")

        let expected = Dictionary(uniqueKeysWithValues: eager.parameters().flattened())
        let actual = Dictionary(uniqueKeysWithValues: reloaded.parameters().flattened())
        XCTAssertEqual(Set(actual.keys), Set(expected.keys))
        for (key, value) in expected {
            XCTAssertTrue(arrayEqual(try XCTUnwrap(actual[key]), value).item(Bool.self), key)
        }
    }

    func testLoadIsANoOpOnceAttached() throws {
        let directory = try writeCheckpoint()
        defer { try? FileManager.default.removeItem(at: directory) }

        let model = try detachedModel(loadedFrom: directory)
        try model.loadMediaModules()
        let tower = try XCTUnwrap(model.tower)
        try model.loadMediaModules()

        XCTAssertTrue(model.tower === tower)
    }

    func testDetachReleasesTheModulesAndTheNextLoadRestoresThem() throws {
        let directory = try writeCheckpoint()
        defer { try? FileManager.default.removeItem(at: directory) }

        let model = try detachedModel(loadedFrom: directory)
        try model.loadMediaModules()
        let weight = try XCTUnwrap(model.tower).weight

        try model.detachMediaModules()
        XCTAssertNil(model.tower)

        try model.loadMediaModules()
        XCTAssertTrue(arrayEqual(try XCTUnwrap(model.tower).weight, weight).item(Bool.self))
    }

    func testLoadWithoutASourceLeavesTheModelDetached() throws {
        // Nothing to read from: the model's own media path reports the missing tower.
        let model = TwoTowerModel()
        try model.detachMediaModules()

        try model.loadMediaModules()

        XCTAssertFalse(model.mediaModulesAreAttached)
    }

    func testIncompleteCheckpointFailsAndLeavesTheModelDetached() throws {
        let directory = try writeCheckpoint(omitting: "model.vision_tower.weight")
        defer { try? FileManager.default.removeItem(at: directory) }

        let model = try detachedModel(loadedFrom: directory)

        XCTAssertThrowsError(try model.loadMediaModules()) { error in
            guard case UpdateError.keyNotFound = error else {
                return XCTFail("expected keyNotFound, got \(error)")
            }
        }
        XCTAssertFalse(model.mediaModulesAreAttached, "unloaded modules never reach the model")
    }

    /// Conversion writes every weight, including media modules a factory detached.
    func testConversionWritesDetachedModules() throws {
        let directory = try writeCheckpoint(quantized: false)
        let output = FileManager.default.temporaryDirectory
            .appendingPathComponent("DetachableMediaModelTests-out-\(UUID().uuidString)")
        defer {
            try? FileManager.default.removeItem(at: directory)
            try? FileManager.default.removeItem(at: output)
        }

        let model = TwoTowerModel()
        try model.detachMediaModules(loadingFrom: MediaWeightSource(modelDirectory: directory))
        let result = try convert(modelDirectory: directory, model: model, to: output)

        var saved = Set<String>()
        for url in result.weightsURLs {
            saved.formUnion(try loadArrays(url: url).keys)
        }
        XCTAssertTrue(saved.isSuperset(of: ["vision_tower.weight", "vision_projection.weight"]))
    }

    // MARK: - Fixtures

    /// A model loaded the way a factory loads it.
    private func detachedModel(loadedFrom directory: URL) throws -> TwoTowerModel {
        let model = TwoTowerModel()
        try model.detachMediaModules(
            loadingFrom: MediaWeightSource(modelDirectory: directory, quantization: quantization))
        try loadWeights(modelDirectory: directory, model: model, quantization: quantization)
        return model
    }

    /// Writes a raw checkpoint, by default with a quantized language layer and projection.
    private func writeCheckpoint(omitting omitted: String? = nil, quantized: Bool = true) throws
        -> URL
    {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(
                "DetachableMediaModelTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try Data("{}".utf8).write(to: directory.appendingPathComponent("config.json"))

        let source = TwoTowerModel()
        if quantized {
            quantize(
                model: source, groupSize: 64, bits: 4, filter: { path, _ in path != "vision_tower" }
            )
        }
        var arrays = [String: MLXArray]()
        for (key, value) in source.parameters().flattened() where "model.\(key)" != omitted {
            arrays["model.\(key)"] = value
        }
        try save(arrays: arrays, url: directory.appendingPathComponent("model.safetensors"))
        return directory
    }
}
