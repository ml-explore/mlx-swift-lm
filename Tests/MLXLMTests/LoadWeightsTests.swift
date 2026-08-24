// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN
import XCTest

@testable import MLXLMCommon

/// A model with a head whose weights may live outside the selected weight files.
private class TwoLayerModel: Module, BaseLanguageModel {
    @ModuleInfo(key: "layer") var layer: Linear
    @ModuleInfo(key: "projector") var projector: Linear

    override init() {
        _layer.wrappedValue = Linear(2, 2, bias: false)
        _projector.wrappedValue = Linear(2, 2, bias: false)
    }
}

/// The same model, declaring the sidecar its checkpoint ships the head in — like
/// `JinaRerankerModel` and `projector.safetensors`.
private final class SidecarDeclaringModel: TwoLayerModel, AdditionalWeightFilesProviding {
    var additionalWeightFiles: [String] { ["projector.safetensors"] }
}

final class LoadWeightsTests: XCTestCase {

    // MARK: - Index

    func testIndexSelectsOnlyTheFilesItNames() throws {
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        try writeEmptyFile("model.safetensors", in: directory)
        try writeEmptyFile("model-extra.safetensors", in: directory)
        try writeIndex(["model.norm.weight": "model.safetensors"], in: directory)

        let names = try safetensorWeightURLs(in: directory).map(\.lastPathComponent)

        XCTAssertEqual(names, ["model.safetensors"])
    }

    func testIndexMayNameFilesInSubdirectories() throws {
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        // An index naming a nested file is a deliberate statement about where this model's
        // weights live, unlike a nested file nobody claims.
        try writeEmptyFile("shards/model-00001-of-00001.safetensors", in: directory)
        try writeIndex(
            ["model.norm.weight": "shards/model-00001-of-00001.safetensors"], in: directory)

        let names = try safetensorWeightURLs(in: directory).map(\.lastPathComponent)

        XCTAssertEqual(names, ["model-00001-of-00001.safetensors"])
    }

    // MARK: - Convention fallback

    func testStaleIndexFallsBackToTheConventionalNames() throws {
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        // mlx-community/Qwen3-VL-4B-Instruct-4bit: one `model.safetensors`, but an index
        // carried over from the unquantized source repo that names two shards it never shipped.
        try writeEmptyFile("model.safetensors", in: directory)
        try writeEmptyFile("head.safetensors", in: directory)
        try writeIndex(
            [
                "model.norm.weight": "model-00001-of-00002.safetensors",
                "model.embed_tokens.weight": "model-00002-of-00002.safetensors",
            ], in: directory)

        let names = try safetensorWeightURLs(in: directory).map(\.lastPathComponent)

        // the convention picks the weights back up without dragging in an unrelated file
        XCTAssertEqual(names, ["model.safetensors"])
    }

    func testPartiallyStaleIndexFallsBackToTheConventionalNames() throws {
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        try writeEmptyFile("model-00001-of-00002.safetensors", in: directory)
        try writeIndex(
            [
                "model.norm.weight": "model-00001-of-00002.safetensors",
                "model.embed_tokens.weight": "model-00002-of-00002.safetensors",
            ], in: directory)

        let names = try safetensorWeightURLs(in: directory).map(\.lastPathComponent)

        XCTAssertEqual(names, ["model-00001-of-00002.safetensors"])
    }

    func testNoIndexUsesTheConventionalNames() throws {
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        try writeEmptyFile("model-00001-of-00002.safetensors", in: directory)
        try writeEmptyFile("model-00002-of-00002.safetensors", in: directory)
        try writeEmptyFile("mtp.safetensors", in: directory)

        let names = try safetensorWeightURLs(in: directory).map(\.lastPathComponent)

        XCTAssertEqual(
            names, ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"])
    }

    func testFallsBackToWeightNamesThenToEverythingPresent() throws {
        let weightPrefixed = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: weightPrefixed) }
        try writeEmptyFile("weights.safetensors", in: weightPrefixed)
        try writeEmptyFile("prerotated_cache.safetensors", in: weightPrefixed)

        XCTAssertEqual(
            try safetensorWeightURLs(in: weightPrefixed).map(\.lastPathComponent),
            ["weights.safetensors"])

        let unconventional = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: unconventional) }
        try writeEmptyFile("adapters.safetensors", in: unconventional)

        // nothing conventional to go on: load what is there rather than nothing at all
        XCTAssertEqual(
            try safetensorWeightURLs(in: unconventional).map(\.lastPathComponent),
            ["adapters.safetensors"])
    }

    // MARK: - Subdirectories

    func testNestedWeightFilesAreNeverSelectedOnTheirOwn() throws {
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        // mlx-community/Qwen3.5-4B-OptiQ-4bit ships its auxiliary weights under `optiq/`.
        // Loading those into the model corrupts generation silently (#408), so they must stay
        // out however the weight files are chosen -- including a nested HF snapshot cache under
        // a local checkpoint directory.
        try writeEmptyFile("model.safetensors", in: directory)
        try writeEmptyFile("optiq/mtp.safetensors", in: directory)
        try writeEmptyFile("optiq/optiq_vision.safetensors", in: directory)

        for selection in [WeightFileSelection.automatic, .allFilesPresent] {
            XCTAssertEqual(
                try safetensorWeightURLs(in: directory, selection: selection)
                    .map(\.lastPathComponent),
                ["model.safetensors"],
                "\(selection) must not descend into subdirectories")
        }

        // ... and the same with an index that no longer matches the shipped files
        try writeIndex(
            ["model.norm.weight": "model-00001-of-00002.safetensors"], in: directory)
        XCTAssertEqual(
            try safetensorWeightURLs(in: directory).map(\.lastPathComponent),
            ["model.safetensors"])
    }

    // MARK: - Additional files

    func testAdditionalFilesAreAppendedAndDeduplicated() throws {
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        try writeEmptyFile("model.safetensors", in: directory)
        try writeEmptyFile("projector.safetensors", in: directory)
        try writeIndex(["model.norm.weight": "model.safetensors"], in: directory)

        // the selected file comes first so its metadata wins
        XCTAssertEqual(
            try safetensorWeightURLs(
                in: directory,
                additionalFiles: ["projector.safetensors", "missing.safetensors"]
            ).map(\.lastPathComponent),
            ["model.safetensors", "projector.safetensors"])

        // an already-selected file is not loaded twice
        XCTAssertEqual(
            try safetensorWeightURLs(
                in: directory, additionalFiles: ["model.safetensors"]
            ).map(\.lastPathComponent),
            ["model.safetensors"])
    }

    // MARK: - Caller policy

    func testAllFilesPresentOverridesTheIndex() throws {
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        try writeEmptyFile("model.safetensors", in: directory)
        try writeEmptyFile("head.safetensors", in: directory)
        try writeIndex(["model.norm.weight": "model.safetensors"], in: directory)

        XCTAssertEqual(
            try safetensorWeightURLs(in: directory, selection: .allFilesPresent)
                .map(\.lastPathComponent),
            ["head.safetensors", "model.safetensors"])
    }

    // MARK: - loadWeights end to end

    func testLoadWeightsReadsSidecarWeightsDeclaredByTheModel() throws {
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        try writeSidecarCheckpoint(in: directory)

        let model = SidecarDeclaringModel()
        try loadWeights(modelDirectory: directory, model: model)

        XCTAssertEqual(model.projector.weight.asArray(Float.self), [1, 2, 3, 4])
    }

    func testLoadWeightsFailsWhenTheSidecarIsNotDeclared() throws {
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        try writeSidecarCheckpoint(in: directory)

        // Neither the index nor the `model*` convention covers `projector.safetensors`, so the
        // head is never loaded -- `verify: [.all]` is what turns that into the keyNotFound
        // error from #560 instead of a silently untrained head.
        let model = TwoLayerModel()
        XCTAssertThrowsError(try loadWeights(modelDirectory: directory, model: model))
    }

    func testLoadWeightsHonorsTheCallerSelectionPolicy() throws {
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        try writeSidecarCheckpoint(in: directory)

        // The escape hatch for a checkpoint no model in the registry knows about.
        let model = TwoLayerModel()
        try loadWeights(
            modelDirectory: directory, model: model, weightFileSelection: .allFilesPresent)

        XCTAssertEqual(model.projector.weight.asArray(Float.self), [1, 2, 3, 4])
    }

    /// Writes a checkpoint whose index names only `model.safetensors` while the head lives in
    /// `projector.safetensors`, the `jinaai/jina-reranker-v3-mlx` layout.
    private func writeSidecarCheckpoint(in directory: URL) throws {
        try save(
            arrays: ["layer.weight": MLXArray.zeros([2, 2])],
            url: directory.appendingPathComponent("model.safetensors"))
        try save(
            arrays: [
                "projector.weight": MLXArray(converting: [1.0, 2.0, 3.0, 4.0]).reshaped(2, 2)
            ],
            url: directory.appendingPathComponent("projector.safetensors"))
        try writeIndex(["layer.weight": "model.safetensors"], in: directory)
    }

    private func makeTemporaryDirectory() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("LoadWeightsTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    private func writeEmptyFile(_ name: String, in directory: URL) throws {
        let url = directory.appendingPathComponent(name)
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Data().write(to: url)
    }

    private func writeIndex(_ weightMap: [String: String], in directory: URL) throws {
        let index: [String: Any] = [
            "metadata": ["total_size": 1],
            "weight_map": weightMap,
        ]
        let data = try JSONSerialization.data(withJSONObject: index)
        try data.write(to: directory.appendingPathComponent("model.safetensors.index.json"))
    }
}
