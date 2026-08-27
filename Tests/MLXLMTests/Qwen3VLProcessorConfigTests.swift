// Copyright © 2026 Apple Inc.
//
// Verifies `Qwen3VLProcessorConfiguration` decodes the new-style
// `size.{longest_edge, shortest_edge}` pixel-area budget that recent Qwen3-VL
// configs ship, in addition to the legacy `min_pixels`/`max_pixels`. Also
// verifies the narrow Qwen3.5/3.6 fallback used when otherwise valid VLM
// repositories omit processor metadata.

import Foundation
import XCTest

@testable import MLXVLM

final class Qwen3VLProcessorConfigTests: XCTestCase {

    private func decode(_ json: String) throws -> Qwen3VLProcessorConfiguration {
        try JSONDecoder().decode(
            Qwen3VLProcessorConfiguration.self, from: Data(json.utf8))
    }

    /// Common required fields; the budget block is appended per-case.
    private func config(budget: String) -> String {
        """
        {
            "image_mean": [0.5, 0.5, 0.5],
            "image_std": [0.5, 0.5, 0.5],
            "merge_size": 2,
            "patch_size": 16,
            "temporal_patch_size": 2,
            "image_processor_type": "Qwen2VLImageProcessorFast"\(budget.isEmpty ? "" : ",\n    " + budget)
        }
        """
    }

    /// The real Qwen3.6/3.5 PARO shape: only new-style edges, no legacy keys.
    func testNewStyleEdgesDecodeToBudget() throws {
        let cfg = try decode(
            config(
                budget: #""size": { "longest_edge": 16777216, "shortest_edge": 65536 }"#))
        XCTAssertEqual(cfg.minPixels, 65536, "shortest_edge → minPixels")
        XCTAssertEqual(cfg.maxPixels, 16_777_216, "longest_edge → maxPixels")
    }

    /// Legacy top-level keys still win.
    func testLegacyTopLevelPixelsHonored() throws {
        let cfg = try decode(
            config(
                budget: #""min_pixels": 1024, "max_pixels": 200000"#))
        XCTAssertEqual(cfg.minPixels, 1024)
        XCTAssertEqual(cfg.maxPixels, 200_000)
    }

    /// Legacy keys nested inside `size` are honored.
    func testLegacySizePixelsHonored() throws {
        let cfg = try decode(
            config(
                budget: #""size": { "min_pixels": 2048, "max_pixels": 300000 }"#))
        XCTAssertEqual(cfg.minPixels, 2048)
        XCTAssertEqual(cfg.maxPixels, 300_000)
    }

    /// Top-level legacy keys take precedence over a `size` block.
    func testTopLevelPixelsBeatSizeEdges() throws {
        let cfg = try decode(
            config(
                budget:
                    #""min_pixels": 100, "max_pixels": 999, "size": { "longest_edge": 5, "shortest_edge": 5 }"#
            ))
        XCTAssertEqual(cfg.minPixels, 100)
        XCTAssertEqual(cfg.maxPixels, 999)
    }

    /// No budget at all falls back to the historical defaults.
    func testAbsentBudgetFallsBackToDefaults() throws {
        let cfg = try decode(config(budget: ""))
        XCTAssertEqual(cfg.minPixels, 4 * 28 * 28)
        XCTAssertEqual(cfg.maxPixels, 16384 * 28 * 28)
    }
}

final class Qwen35ProcessorFallbackTests: XCTestCase {

    func testDenseFallbackEncodesVisionGeometryAndQwenDefaults() throws {
        let fallback = try XCTUnwrap(
            Qwen35ProcessorLoadingResolver().fallbackProcessorConfiguration(
                for: VLMProcessorLoadingContext(
                    modelId: "example/qwen",
                    modelType: "qwen3_5",
                    configurationData: makeQwen35ConfigurationData(modelType: "qwen3_5"))))
        let config = try JSONDecoder().decode(
            Qwen3VLProcessorConfiguration.self, from: fallback.data)

        XCTAssertEqual(fallback.processorType, "Qwen3VLProcessor")
        XCTAssertEqual(config.imageMean, [0.5, 0.5, 0.5])
        XCTAssertEqual(config.imageStd, [0.5, 0.5, 0.5])
        XCTAssertEqual(config.minPixels, 65_536)
        XCTAssertEqual(config.maxPixels, 16_777_216)
        XCTAssertEqual(config.patchSize, 14)
        XCTAssertEqual(config.mergeSize, 3)
        XCTAssertEqual(config.temporalPatchSize, 4)
        XCTAssertEqual(config.imageProcessorType, "Qwen2VLImageProcessorFast")
    }

    func testMoEFallbackUsesInheritedQwen35Configuration() throws {
        let fallback = try XCTUnwrap(
            Qwen35ProcessorLoadingResolver().fallbackProcessorConfiguration(
                for: VLMProcessorLoadingContext(
                    modelId: "example/qwen-moe",
                    modelType: "qwen3_5_moe",
                    configurationData: makeQwen35ConfigurationData(
                        modelType: "qwen3_5_moe", patchSize: 12, mergeSize: 2,
                        temporalPatchSize: 3))))
        let config = try JSONDecoder().decode(
            Qwen3VLProcessorConfiguration.self, from: fallback.data)

        XCTAssertEqual(fallback.processorType, "Qwen3VLProcessor")
        XCTAssertEqual(config.patchSize, 12)
        XCTAssertEqual(config.mergeSize, 2)
        XCTAssertEqual(config.temporalPatchSize, 3)
    }

    func testFallbackRejectsUnrelatedModelType() throws {
        XCTAssertNil(
            try Qwen35ProcessorLoadingResolver().fallbackProcessorConfiguration(
                for: VLMProcessorLoadingContext(
                    modelId: "example/qwen",
                    modelType: "qwen3_vl",
                    configurationData: makeQwen35ConfigurationData(modelType: "qwen3_5"))))
    }

    func testMissingFilesUseFallback() async throws {
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let fallback = try processorConfiguration(named: "FallbackProcessor")

        let resolved = try await loadProcessorConfig(from: directory) { fallback }

        XCTAssertEqual(resolved.data, fallback.data)
        XCTAssertEqual(resolved.processorType, "FallbackProcessor")
    }

    func testPreprocessorConfigWinsOverProcessorAndFallback() async throws {
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let preprocessor = try processorConfiguration(named: "Preprocessor")
        let processor = try processorConfiguration(named: "Processor")
        let fallback = try processorConfiguration(named: "Fallback")
        try preprocessor.data.write(
            to: directory.appending(component: "preprocessor_config.json"))
        try processor.data.write(to: directory.appending(component: "processor_config.json"))

        let resolved = try await loadProcessorConfig(from: directory) { fallback }

        XCTAssertEqual(resolved.data, preprocessor.data)
        XCTAssertEqual(resolved.processorType, "Preprocessor")
    }

    func testProcessorConfigWinsOverFallback() async throws {
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let processor = try processorConfiguration(named: "Processor")
        let fallback = try processorConfiguration(named: "Fallback")
        try processor.data.write(to: directory.appending(component: "processor_config.json"))

        let resolved = try await loadProcessorConfig(from: directory) { fallback }

        XCTAssertEqual(resolved.data, processor.data)
        XCTAssertEqual(resolved.processorType, "Processor")
    }

    func testMalformedPreprocessorIsNotMasked() async throws {
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let processor = try processorConfiguration(named: "Processor")
        let fallback = try processorConfiguration(named: "Fallback")
        try Data("{".utf8).write(
            to: directory.appending(component: "preprocessor_config.json"))
        try processor.data.write(to: directory.appending(component: "processor_config.json"))

        do {
            _ = try await loadProcessorConfig(from: directory) { fallback }
            XCTFail("Expected malformed preprocessor_config.json to throw")
        } catch let error as ProcessorConfigError {
            XCTAssertEqual(error.filename, "preprocessor_config.json")
        }
    }

    func testMalformedProcessorIsNotMasked() async throws {
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let fallback = try processorConfiguration(named: "Fallback")
        try Data("{".utf8).write(
            to: directory.appending(component: "processor_config.json"))

        do {
            _ = try await loadProcessorConfig(from: directory) { fallback }
            XCTFail("Expected malformed processor_config.json to throw")
        } catch let error as ProcessorConfigError {
            XCTAssertEqual(error.filename, "processor_config.json")
        }
    }

    func testMissingFilesWithoutFallbackRetainProcessorFilename() async throws {
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        do {
            _ = try await loadProcessorConfig(from: directory)
            XCTFail("Expected a missing processor_config.json error")
        } catch let error as ProcessorConfigError {
            XCTAssertEqual(error.filename, "processor_config.json")
        }
    }

    func testQwen35ConfigurationRequiresCompleteVisionConfig() {
        let data = Data(
            """
            {
                "model_type": "qwen3_5",
                "text_config": {},
                "vision_config": {
                    "model_type": "qwen3_vl",
                    "patch_size": 16,
                    "spatial_merge_size": 2,
                    "temporal_patch_size": 2
                }
            }
            """.utf8)

        XCTAssertThrowsError(try JSONDecoder().decode(Qwen35Configuration.self, from: data))
    }

    private func makeQwen35ConfigurationData(
        modelType: String,
        patchSize: Int = 14,
        mergeSize: Int = 3,
        temporalPatchSize: Int = 4
    ) -> Data {
        let json = """
            {
                "model_type": "\(modelType)",
                "text_config": {
                    "model_type": "\(modelType)",
                    "hidden_size": 8,
                    "num_hidden_layers": 1,
                    "intermediate_size": 16,
                    "num_attention_heads": 1,
                    "num_key_value_heads": 1,
                    "linear_num_value_heads": 1,
                    "linear_num_key_heads": 1,
                    "linear_key_head_dim": 8,
                    "linear_value_head_dim": 8,
                    "linear_conv_kernel_dim": 4,
                    "vocab_size": 32,
                    "full_attention_interval": 2,
                    "num_experts": 0,
                    "num_experts_per_tok": 0
                },
                "vision_config": {
                    "model_type": "qwen3_vl",
                    "depth": 1,
                    "hidden_size": 8,
                    "intermediate_size": 16,
                    "out_hidden_size": 8,
                    "num_heads": 1,
                    "patch_size": \(patchSize),
                    "spatial_merge_size": \(mergeSize),
                    "temporal_patch_size": \(temporalPatchSize),
                    "num_position_embeddings": 9
                }
            }
            """
        return Data(json.utf8)
    }

    private func processorConfiguration(
        named processorClass: String
    ) throws -> VLMProcessorConfiguration {
        let config = BaseProcessorConfiguration(processorClass: processorClass)
        return VLMProcessorConfiguration(
            data: try JSONEncoder().encode(config), processorType: processorClass)
    }

    private func makeTemporaryDirectory() throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appending(component: "Qwen35ProcessorFallbackTests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        return directory
    }
}
