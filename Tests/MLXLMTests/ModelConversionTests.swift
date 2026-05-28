// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import XCTest

@testable import MLXLMCommon

final class ModelConversionTests: XCTestCase {

    func testModelConversionOptionsDefaultsLeaveQuantizationDefaultsUnspecified() {
        let options = ModelConversionOptions()

        XCTAssertNil(options.bits)
        XCTAssertNil(options.groupSize)
        XCTAssertEqual(options.mode, .affine)
        XCTAssertNil(options.quantization.bits)
        XCTAssertNil(options.quantization.groupSize)
    }

    func testResolveForModelConversionDownloadsTokenizerSidecarPatterns() async throws {
        let downloader = ConversionMockDownloader()
        let configuration = ModelConfiguration(
            id: "org/model",
            revision: "abc123",
            tokenizerSource: .id("org/tokenizer", revision: "tok456"))

        let resolved = try await resolveForModelConversion(
            configuration: configuration,
            from: downloader,
            useLatest: false,
            progressHandler: { _ in })

        let calls = downloader.calls.value
        XCTAssertEqual(calls.count, 2)
        XCTAssertEqual(calls[0].id, "org/model")
        XCTAssertEqual(calls[0].revision, "abc123")
        XCTAssertTrue(calls[0].patterns.contains("*.safetensors"))
        XCTAssertTrue(calls[0].patterns.contains("*.model"))
        XCTAssertTrue(calls[0].patterns.contains("*.tiktoken"))

        XCTAssertEqual(calls[1].id, "org/tokenizer")
        XCTAssertEqual(calls[1].revision, "tok456")
        XCTAssertFalse(calls[1].patterns.contains("*.safetensors"))
        XCTAssertTrue(calls[1].patterns.contains("*.txt"))
        XCTAssertTrue(calls[1].patterns.contains("*.model"))
        XCTAssertNotEqual(resolved.modelDirectory, resolved.tokenizerDirectory)
    }

    func testConfigQuantizationUpdatePreservesExistingKeys() throws {
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        let configURL = directory.appendingPathComponent("config.json")
        try """
        {
          "model_type": "llama",
          "hidden_size": 128
        }
        """.data(using: .utf8)!.write(to: configURL)

        try updateModelConfigWithQuantization(
            at: directory, bits: 4, groupSize: 32, mode: .mxfp4)

        let data = try Data(contentsOf: configURL)
        let json = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        XCTAssertEqual(json["model_type"] as? String, "llama")
        XCTAssertEqual(json["hidden_size"] as? Int, 128)

        let quantization = try XCTUnwrap(json["quantization"] as? [String: Any])
        XCTAssertEqual(quantization["bits"] as? Int, 4)
        XCTAssertEqual(quantization["group_size"] as? Int, 32)
        XCTAssertEqual(quantization["mode"] as? String, "mxfp4")
    }

    func testConfigQuantizationUpdateResolvesNilDefaults() throws {
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        let configURL = directory.appendingPathComponent("config.json")
        try """
        {
          "model_type": "llama"
        }
        """.data(using: .utf8)!.write(to: configURL)

        try updateModelConfigWithQuantization(
            at: directory,
            quantization: .init(mode: .affine),
            layerQuantization: [
                "model.layers.0.mlp.down_proj": .quantize(.init(mode: .mxfp8)),
                "model.layers.0.self_attn.q_norm": .skip,
            ])

        let data = try Data(contentsOf: configURL)
        let json = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        let quantization = try XCTUnwrap(json["quantization"] as? [String: Any])
        XCTAssertEqual(quantization["bits"] as? Int, 4)
        XCTAssertEqual(quantization["group_size"] as? Int, 64)
        XCTAssertEqual(quantization["mode"] as? String, "affine")

        let layer = try XCTUnwrap(
            quantization["model.layers.0.mlp.down_proj"] as? [String: Any])
        XCTAssertEqual(layer["bits"] as? Int, 8)
        XCTAssertEqual(layer["group_size"] as? Int, 32)
        XCTAssertEqual(layer["mode"] as? String, "mxfp8")
        XCTAssertEqual(quantization["model.layers.0.self_attn.q_norm"] as? Bool, false)
    }

    func testCopyModelConversionFilesCopiesSidecarsAndSkipsWeights() throws {
        let modelDirectory = try makeTemporaryDirectory()
        let tokenizerDirectory = try makeTemporaryDirectory()
        let outputDirectory = try makeTemporaryDirectory()
        defer {
            try? FileManager.default.removeItem(at: modelDirectory)
            try? FileManager.default.removeItem(at: tokenizerDirectory)
            try? FileManager.default.removeItem(at: outputDirectory)
        }

        try write("{}", to: modelDirectory.appendingPathComponent("config.json"))
        try write("{}", to: modelDirectory.appendingPathComponent("generation_config.json"))
        try write("{}", to: modelDirectory.appendingPathComponent("tokenizer_config.json"))
        try write("{}", to: modelDirectory.appendingPathComponent("tokenizer.jsonl"))
        try write("print('custom')", to: modelDirectory.appendingPathComponent("tokenization.py"))
        try write("weights", to: modelDirectory.appendingPathComponent("model.safetensors"))
        try write(
            "index", to: modelDirectory.appendingPathComponent("model.safetensors.index.json"))
        try write("{}", to: tokenizerDirectory.appendingPathComponent("config.json"))
        try write("{}", to: tokenizerDirectory.appendingPathComponent("tokenizer.json"))
        try write("template", to: tokenizerDirectory.appendingPathComponent("chat_template.jinja"))

        try copyModelConversionFiles(
            modelDirectory: modelDirectory,
            tokenizerDirectory: tokenizerDirectory,
            to: outputDirectory)

        XCTAssertTrue(fileExists("config.json", in: outputDirectory))
        XCTAssertTrue(fileExists("generation_config.json", in: outputDirectory))
        XCTAssertTrue(fileExists("tokenizer_config.json", in: outputDirectory))
        XCTAssertTrue(fileExists("tokenizer.jsonl", in: outputDirectory))
        XCTAssertTrue(fileExists("tokenization.py", in: outputDirectory))
        XCTAssertTrue(fileExists("tokenizer.json", in: outputDirectory))
        XCTAssertTrue(fileExists("chat_template.jinja", in: outputDirectory))
        XCTAssertFalse(fileExists("model.safetensors", in: outputDirectory))
        XCTAssertFalse(fileExists("model.safetensors.index.json", in: outputDirectory))
    }

    func testRemoveExistingModelWeightsClearsStaleOutputArtifacts() throws {
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        try write("old", to: directory.appendingPathComponent("model.safetensors"))
        try write("old", to: directory.appendingPathComponent("model-00001-of-00002.safetensors"))
        try write("old", to: directory.appendingPathComponent("model.safetensors.index.json"))
        try write("{}", to: directory.appendingPathComponent("config.json"))

        try removeExistingModelWeights(in: directory)

        XCTAssertFalse(fileExists("model.safetensors", in: directory))
        XCTAssertFalse(fileExists("model-00001-of-00002.safetensors", in: directory))
        XCTAssertFalse(fileExists("model.safetensors.index.json", in: directory))
        XCTAssertTrue(fileExists("config.json", in: directory))
    }

    func testPrepareOutputDirectoryRejectsExistingPathByDefault() throws {
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        XCTAssertThrowsError(
            try prepareOutputDirectoryForModelConversion(
                directory, overwriteExistingOutput: false)
        ) { error in
            XCTAssertEqual(error as? ModelConversionError, .outputDirectoryExists(directory))
        }
    }

    func testPrepareOutputDirectoryOverwriteRemovesStaleFiles() throws {
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        try write("stale", to: directory.appendingPathComponent("tokenizer.model"))

        try prepareOutputDirectoryForModelConversion(directory, overwriteExistingOutput: true)

        XCTAssertTrue(FileManager.default.fileExists(atPath: directory.path))
        XCTAssertFalse(fileExists("tokenizer.model", in: directory))
    }

    func testValidateOutputDirectoryRejectsModelDirectory() throws {
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        XCTAssertThrowsError(
            try validateModelConversionOutputDirectory(
                directory, modelDirectory: directory, tokenizerDirectory: nil)
        ) { error in
            XCTAssertEqual(error as? ModelConversionError, .outputDirectoryMatchesSource(directory))
        }
    }

    func testValidateOutputDirectoryRejectsModelDirectoryParent() throws {
        let parentDirectory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: parentDirectory) }

        let modelDirectory = parentDirectory.appendingPathComponent("model")
        try FileManager.default.createDirectory(
            at: modelDirectory, withIntermediateDirectories: true)

        XCTAssertThrowsError(
            try validateModelConversionOutputDirectory(
                parentDirectory, modelDirectory: modelDirectory, tokenizerDirectory: nil)
        ) { error in
            XCTAssertEqual(
                error as? ModelConversionError,
                .outputDirectoryMatchesSource(parentDirectory))
        }
    }

    func testValidateOutputDirectoryRejectsModelDirectoryChild() throws {
        let modelDirectory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: modelDirectory) }

        let outputDirectory = modelDirectory.appendingPathComponent("converted")

        XCTAssertThrowsError(
            try validateModelConversionOutputDirectory(
                outputDirectory, modelDirectory: modelDirectory, tokenizerDirectory: nil)
        ) { error in
            XCTAssertEqual(
                error as? ModelConversionError,
                .outputDirectoryMatchesSource(outputDirectory))
        }
    }

    func testValidateOutputDirectoryRejectsTokenizerDirectory() throws {
        let modelDirectory = try makeTemporaryDirectory()
        let tokenizerDirectory = try makeTemporaryDirectory()
        defer {
            try? FileManager.default.removeItem(at: modelDirectory)
            try? FileManager.default.removeItem(at: tokenizerDirectory)
        }

        XCTAssertThrowsError(
            try validateModelConversionOutputDirectory(
                tokenizerDirectory,
                modelDirectory: modelDirectory,
                tokenizerDirectory: tokenizerDirectory)
        ) { error in
            XCTAssertEqual(
                error as? ModelConversionError,
                .outputDirectoryMatchesSource(tokenizerDirectory))
        }
    }

    func testValidateConvertibleWeightsRejectsPyTorchOnlyDirectory() throws {
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        try write("weights", to: directory.appendingPathComponent("pytorch_model.bin"))

        XCTAssertThrowsError(try validateConvertibleWeights(in: directory)) { error in
            XCTAssertEqual(
                error as? ModelConversionError,
                .unsupportedPyTorchWeights(directory))
        }
    }

    func testValidateConvertibleWeightsRejectsQuantizedSafetensors() throws {
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        try writeSafetensorsHeader(
            [
                "model.layers.0.self_attn.q_proj.weight": [
                    "dtype": "F16", "shape": [1], "data_offsets": [0, 2],
                ],
                "model.layers.0.self_attn.q_proj.scales": [
                    "dtype": "F16", "shape": [1], "data_offsets": [2, 4],
                ],
            ],
            to: directory.appendingPathComponent("model.safetensors"))

        XCTAssertThrowsError(try validateConvertibleWeights(in: directory)) { error in
            XCTAssertEqual(error as? ModelConversionError, .sourceAlreadyQuantized(directory))
        }
    }

    func testSourceConfigurationIsQuantizedDetectsBothConfigKeys() throws {
        let quantization = try XCTUnwrap(
            """
            {
              "model_type": "llama",
              "quantization": { "bits": 4, "group_size": 64 }
            }
            """.data(using: .utf8))
        let quantizationConfig = try XCTUnwrap(
            """
            {
              "model_type": "llama",
              "quantization_config": { "bits": 4, "group_size": 64 }
            }
            """.data(using: .utf8))
        let plain = try XCTUnwrap(
            """
            {
              "model_type": "llama"
            }
            """.data(using: .utf8))

        XCTAssertTrue(sourceConfigurationIsQuantized(quantization))
        XCTAssertTrue(sourceConfigurationIsQuantized(quantizationConfig))
        XCTAssertFalse(sourceConfigurationIsQuantized(plain))
    }

    func testConfigQuantizationUpdateAcceptsJSON5Config() throws {
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        let configURL = directory.appendingPathComponent("config.json")
        try """
        {
          // JSON5 comments are accepted by normal model loading.
          "model_type": "llama",
        }
        """.data(using: .utf8)!.write(to: configURL)

        try updateModelConfigWithQuantization(
            at: directory, bits: 4, groupSize: 64, mode: .affine)

        let data = try Data(contentsOf: configURL)
        XCTAssertTrue(sourceConfigurationIsQuantized(data))
        let json = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        XCTAssertEqual(json["model_type"] as? String, "llama")

        let quantization = try XCTUnwrap(json["quantization"] as? [String: Any])
        XCTAssertEqual(quantization["bits"] as? Int, 4)
        XCTAssertEqual(quantization["group_size"] as? Int, 64)
        XCTAssertEqual(quantization["mode"] as? String, "affine")
    }

    func testValidateSourceConfigurationRejectsQuantizedConfig() throws {
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        try """
        {
          "model_type": "llama",
          "quantization_config": { "bits": 4, "group_size": 64 }
        }
        """.data(using: .utf8)!.write(to: directory.appendingPathComponent("config.json"))

        XCTAssertThrowsError(try validateSourceConfigurationForModelConversion(in: directory)) {
            error in
            XCTAssertEqual(error as? ModelConversionError, .sourceAlreadyQuantized(directory))
        }
    }

    func testSaveModelConversionIndexWritesMetadataAndWeightMap() throws {
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        try saveModelConversionIndex(
            weightMap: [
                "b.weight": "model-00001-of-00002.safetensors",
                "a.weight": "model-00002-of-00002.safetensors",
            ],
            totalSize: 16,
            to: directory)

        let indexURL = directory.appendingPathComponent("model.safetensors.index.json")
        let data = try Data(contentsOf: indexURL)
        let json = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        let metadata = try XCTUnwrap(json["metadata"] as? [String: Any])
        XCTAssertEqual(metadata["total_size"] as? Int, 16)

        let weightMap = try XCTUnwrap(json["weight_map"] as? [String: String])
        XCTAssertEqual(weightMap["b.weight"], "model-00001-of-00002.safetensors")
        XCTAssertEqual(weightMap["a.weight"], "model-00002-of-00002.safetensors")
    }

    func testUpdateConfigWritesQuantizationAndQuantizationConfig() throws {
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        let configURL = directory.appendingPathComponent("config.json")
        try """
        {
          "model_type": "llama"
        }
        """.data(using: .utf8)!.write(to: configURL)

        try updateModelConfigWithQuantization(
            at: directory,
            quantization: .init(bits: 4, groupSize: 64, mode: .affine),
            layerQuantization: [
                "model.layers.0.mlp.down_proj": .quantize(
                    .init(bits: 8, groupSize: 32, mode: .mxfp8))
            ])

        let data = try Data(contentsOf: configURL)
        let json = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        let quantization = try XCTUnwrap(json["quantization"] as? [String: Any])
        let quantizationConfig = try XCTUnwrap(json["quantization_config"] as? [String: Any])
        XCTAssertEqual(quantization["bits"] as? Int, 4)
        XCTAssertEqual(quantization["group_size"] as? Int, 64)
        XCTAssertEqual(quantization["mode"] as? String, "affine")
        XCTAssertEqual(quantizationConfig["mode"] as? String, "affine")

        let layer = try XCTUnwrap(
            quantization["model.layers.0.mlp.down_proj"] as? [String: Any])
        XCTAssertEqual(layer["bits"] as? Int, 8)
        XCTAssertEqual(layer["group_size"] as? Int, 32)
        XCTAssertEqual(layer["mode"] as? String, "mxfp8")
    }

    private func makeTemporaryDirectory() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("ModelConversionTests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    private func write(_ contents: String, to url: URL) throws {
        try contents.data(using: .utf8)!.write(to: url)
    }

    private func writeSafetensorsHeader(_ header: [String: Any], to url: URL) throws {
        let headerData = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
        var data = Data()
        var headerSize = UInt64(headerData.count).littleEndian
        withUnsafeBytes(of: &headerSize) { data.append(contentsOf: $0) }
        data.append(headerData)
        try data.write(to: url)
    }

    private func fileExists(_ filename: String, in directory: URL) -> Bool {
        FileManager.default.fileExists(
            atPath: directory.appendingPathComponent(filename).path)
    }
}

private struct ConversionMockDownloader: Downloader {
    struct Call: Equatable, Sendable {
        let id: String
        let revision: String?
        let patterns: [String]
    }

    let calls = ConversionLockIsolated<[Call]>([])

    func download(
        id: String,
        revision: String?,
        matching patterns: [String],
        useLatest: Bool,
        progressHandler: @Sendable @escaping (Progress) -> Void
    ) async throws -> URL {
        calls.withLock { $0.append(Call(id: id, revision: revision, patterns: patterns)) }
        return URL(filePath: "/mock/\(id.replacingOccurrences(of: "/", with: "_"))")
    }
}

private final class ConversionLockIsolated<Value: Sendable>: @unchecked Sendable {
    private var storage: Value
    private let lock = NSLock()

    init(_ value: Value) {
        storage = value
    }

    func withLock<R>(_ body: (inout Value) -> R) -> R {
        lock.lock()
        defer { lock.unlock() }
        return body(&storage)
    }

    var value: Value {
        lock.lock()
        defer { lock.unlock() }
        return storage
    }
}
