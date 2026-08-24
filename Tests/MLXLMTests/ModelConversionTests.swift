// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN
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

    func testSaveModelConversionWeightsWritesModelMetadata() throws {
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        let weightsURL = try saveModelConversionWeights(
            [("model.weight", MLXArray.ones([2, 2], dtype: .float32))],
            to: directory,
            maxShardSize: 1024,
            metadata: ["mlx_swift_lm.test.scaling": "v1"]
        )[0]

        let (_, metadata) = try loadArraysAndMetadata(url: weightsURL)
        XCTAssertEqual(metadata["format"], "mlx")
        XCTAssertEqual(metadata["mlx_swift_lm.test.scaling"], "v1")
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

    // MARK: - q4Zero calibration

    /// Unpacks MLX's 4-bit affine layout: 8 codes per uint32, low nibble first.
    private func unpackCodes(_ packed: MLXArray) -> [Int] {
        let words = packed.asArray(UInt32.self)
        var codes = [Int]()
        codes.reserveCapacity(words.count * 8)
        for word in words {
            for nibble in 0 ..< 8 {
                codes.append(Int((word >> UInt32(nibble * 4)) & 0xF))
            }
        }
        return codes
    }

    /// One 32-element group whose extremum is NEGATIVE, with hand-computable codes.
    ///
    /// extremum = -4, so d = -4 / -8 = 0.5 and code = floor(w / 0.5 + 8.5).
    private func negativeExtremumGroup() -> (weights: [Float], expectedCodes: [Int]) {
        var weights = [Float](repeating: 0, count: 32)
        weights[0] = -4.0  // the signed extremum
        weights[1] = 0.0
        weights[2] = 0.5
        weights[3] = -0.5
        weights[4] = 3.5
        var expected = [Int](repeating: 8, count: 32)  // w == 0 -> floor(8.5) == 8
        expected[0] = 0  // floor(-8 + 8.5)
        expected[1] = 8
        expected[2] = 9  // floor(1 + 8.5)
        expected[3] = 7  // floor(-1 + 8.5)
        expected[4] = 15  // floor(7 + 8.5)
        return (weights, expected)
    }

    func testQ4ZeroCalibrationDefaultsToStandard() {
        XCTAssertEqual(ModelConversionQuantization().calibration, .standard)
        XCTAssertEqual(ModelConversionOptions().calibration, .standard)
    }

    func testQ4ZeroProducesHandComputableCodesScalesAndBiases() {
        let (weights, expected) = negativeExtremumGroup()
        let weight = MLXArray(weights, [1, 32])

        let (packed, scales, biases) = q4ZeroQuantized(weight)

        XCTAssertEqual(unpackCodes(packed), expected)
        XCTAssertEqual(scales.asArray(Float.self), [0.5])
        XCTAssertEqual(biases.asArray(Float.self), [-4.0])
        // First word packs codes 0,8,9,7,15,8,8,8 low-nibble-first.
        XCTAssertEqual(packed.asArray(UInt32.self)[0], 0x888F_7980)
    }

    /// The regression that matters: `-max(abs(w))/8` flips the scale's sign on every group
    /// whose extremum is negative, silently producing a different grid rather than an error.
    func testQ4ZeroUsesSignedExtremumNotAbsMax() {
        let (weights, _) = negativeExtremumGroup()
        let weight = MLXArray(weights, [1, 32])

        let (packed, scales, _) = q4ZeroQuantized(weight)

        // Signed extremum is -4 -> d = +0.5 and the extremum encodes to code 0.
        XCTAssertEqual(scales.asArray(Float.self), [0.5])
        XCTAssertEqual(unpackCodes(packed)[0], 0)

        // -absMax/8 would give d = -0.5, sending the same element to code 15.
        let absMaxScale = -weights.map { abs($0) }.max()! / 8
        XCTAssertEqual(absMaxScale, -0.5)
        let wrongCode = min(15, max(0, Int((weights[0] / absMaxScale + 8.5).rounded(.down))))
        XCTAssertEqual(wrongCode, 15)
    }

    func testQ4ZeroHandlesPositiveExtremumAndClipsAtBothEnds() {
        // extremum = +4 -> d = -0.5. code = floor(w / -0.5 + 8.5).
        var weights = [Float](repeating: 0, count: 32)
        weights[0] = 4.0
        weights[1] = -3.5
        let weight = MLXArray(weights, [1, 32])

        let (packed, scales, biases) = q4ZeroQuantized(weight)
        let codes = unpackCodes(packed)

        XCTAssertEqual(scales.asArray(Float.self), [-0.5])
        XCTAssertEqual(biases.asArray(Float.self), [4.0])
        XCTAssertEqual(codes[0], 0)  // floor(-8 + 8.5)
        XCTAssertEqual(codes[1], 15)  // floor(7 + 8.5)
        XCTAssertTrue(codes.allSatisfy { $0 >= 0 && $0 <= 15 })
    }

    func testQ4ZeroAllZeroGroupDoesNotDivideByZero() {
        let weight = MLXArray([Float](repeating: 0, count: 32), [1, 32])

        let (packed, scales, biases) = q4ZeroQuantized(weight)

        XCTAssertEqual(scales.asArray(Float.self), [0])
        XCTAssertEqual(biases.asArray(Float.self), [0])
        // inverse is forced to 0, so every code is floor(0 + 8.5) == 8.
        XCTAssertEqual(unpackCodes(packed), [Int](repeating: 8, count: 32))
    }

    func testQ4ZeroScalesPerGroupAcrossMultipleGroupsAndRows() {
        // Two rows, two groups each: distinct extrema prove grouping follows the input axis
        // rather than collapsing across rows or groups.
        var values = [Float](repeating: 0, count: 2 * 64)
        values[0] = -4.0  // row 0, group 0 -> d = 0.5
        values[32] = 2.0  // row 0, group 1 -> d = -0.25
        values[64] = 8.0  // row 1, group 0 -> d = -1.0
        values[96] = -1.0  // row 1, group 1 -> d = 0.125
        let weight = MLXArray(values, [2, 64])

        let (_, scales, biases) = q4ZeroQuantized(weight)

        XCTAssertEqual(scales.shape, [2, 2])
        XCTAssertEqual(scales.asArray(Float.self), [0.5, -0.25, -1.0, 0.125])
        XCTAssertEqual(biases.asArray(Float.self), [-4.0, 2.0, 8.0, -1.0])
    }

    func testQ4ZeroDequantizesOntoTheIntendedLattice() {
        let (weights, expected) = negativeExtremumGroup()
        let weight = MLXArray(weights, [1, 32])

        let (packed, scales, biases) = q4ZeroQuantized(weight)
        let restored = dequantized(
            packed, scales: scales, biases: biases, groupSize: 32, bits: 4, mode: .affine)

        let scale = scales.asArray(Float.self)[0]
        let expectedValues = expected.map { (Float($0) - 8) * scale }
        let actual = restored.asArray(Float.self)
        XCTAssertEqual(actual.count, expectedValues.count)
        for (lhs, rhs) in zip(actual, expectedValues) {
            XCTAssertEqual(lhs, rhs, accuracy: 1e-6)
        }
    }

    func testQ4ZeroCalibrationRejectsIncompatibleSettings() {
        for quantization in [
            ModelConversionQuantization(bits: 8, calibration: .q4Zero),
            ModelConversionQuantization(groupSize: 64, calibration: .q4Zero),
        ] {
            XCTAssertThrowsError(try validateModelConversionCalibration(quantization)) { error in
                guard case ModelConversionError.incompatibleCalibration = error else {
                    return XCTFail("expected incompatibleCalibration, got \(error)")
                }
            }
        }
    }

    func testQ4ZeroCalibrationAcceptsResolvedAndOmittedSettings() {
        XCTAssertNoThrow(
            try validateModelConversionCalibration(
                ModelConversionQuantization(calibration: .q4Zero)))
        XCTAssertNoThrow(
            try validateModelConversionCalibration(
                ModelConversionQuantization(bits: 4, groupSize: 32, calibration: .q4Zero)))
    }

    func testStandardCalibrationImposesNoConstraints() {
        XCTAssertNoThrow(
            try validateModelConversionCalibration(
                ModelConversionQuantization(bits: 8, groupSize: 64)))
    }

    // MARK: - q4Zero wiring

    /// Minimal model exposing one Linear and one Embedding for conversion-path tests.
    private final class StubConversionModel: Module, BaseLanguageModel {
        let linear: Linear
        let embedding: Embedding

        init(inputWidth: Int) {
            self.linear = Linear(inputWidth, 8, bias: true)
            self.embedding = Embedding(embeddingCount: 4, dimensions: inputWidth)
        }

        func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] { weights }
    }

    /// Regression: q4Zero must resolve its own geometry. Falling through to the affine mode
    /// defaults yields group size 64, which then skips every linear whose input width is a
    /// multiple of 32 but not 64.
    func testQ4ZeroResolvesToFourBitsGroupThirtyTwo() {
        let resolved = effectiveModelConversionQuantization(
            ModelConversionQuantization(calibration: .q4Zero))

        XCTAssertEqual(resolved.bits, 4)
        XCTAssertEqual(resolved.groupSize, 32)
        XCTAssertEqual(resolved.mode, .affine)

        // Standard calibration keeps the mode-derived defaults.
        let standard = effectiveModelConversionQuantization(ModelConversionQuantization())
        XCTAssertEqual(standard.groupSize, 64)
    }

    /// Regression: an invalid per-layer override must throw, not be silently dropped.
    func testInvalidPerLayerOverrideThrowsBeforeConversion() {
        let model = StubConversionModel(inputWidth: 32)
        let options = ModelConversionOptions(quantizationPredicate: { _, _ in
            .quantize(ModelConversionQuantization(bits: 8, calibration: .q4Zero))
        })

        XCTAssertThrowsError(
            try modelConversionQuantizationDecisions(model: model, options: options)
        ) { error in
            guard case ModelConversionError.incompatibleCalibration = error else {
                return XCTFail("expected incompatibleCalibration, got \(error)")
            }
        }
    }

    func testValidPerLayerOverridesResolveForEveryLeaf() throws {
        let model = StubConversionModel(inputWidth: 32)
        let options = ModelConversionOptions(quantizationPredicate: { _, _ in
            .quantize(ModelConversionQuantization(calibration: .q4Zero))
        })

        let decisions = try modelConversionQuantizationDecisions(model: model, options: options)

        XCTAssertFalse(decisions.isEmpty)
        XCTAssertTrue(
            decisions.allSatisfy {
                if case .quantize(let override) = $0.2 { return override?.calibration == .q4Zero }
                return false
            })
    }

    /// Regression: the predicate must be consulted exactly once per leaf. Resolving twice
    /// would let a stateful predicate pass preflight and then fail after the output
    /// directory has already been created and populated.
    func testQuantizationPredicateIsInvokedOncePerLeaf() throws {
        let model = StubConversionModel(inputWidth: 32)
        let counts = LockedCounts()
        let options = ModelConversionOptions(quantizationPredicate: { path, _ in
            counts.increment(path)
            return .quantize()
        })

        let decisions = try modelConversionQuantizationDecisions(model: model, options: options)

        XCTAssertFalse(decisions.isEmpty)
        XCTAssertEqual(counts.snapshot().count, decisions.count)
        XCTAssertTrue(counts.snapshot().values.allSatisfy { $0 == 1 })
    }

    private final class LockedCounts: @unchecked Sendable {
        private let lock = NSLock()
        private var storage = [String: Int]()

        func increment(_ key: String) {
            lock.lock()
            defer { lock.unlock() }
            storage[key, default: 0] += 1
        }

        func snapshot() -> [String: Int] {
            lock.lock()
            defer { lock.unlock() }
            return storage
        }
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
