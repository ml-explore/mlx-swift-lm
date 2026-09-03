// Copyright © 2026 Apple Inc.

import Foundation
import XCTest

@testable import MLXVLM

final class VLMProcessorLoadingRegistryTests: XCTestCase {

    func testExternalResolversComposeFallbackAndTypeResolution() async throws {
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let fallbackData = Data(#"{"source":"external-package"}"#.utf8)
        let registry = VLMProcessorLoadingRegistry(resolvers: [
            TestResolver(
                configuration: VLMProcessorConfiguration(
                    data: fallbackData, processorType: "DeclaredProcessor")),
            TestResolver(processorType: "ExternalProcessor"),
        ])

        let resolved = try await resolveProcessorConfiguration(
            from: directory, context: context(), registry: registry)

        XCTAssertEqual(resolved.data, fallbackData)
        XCTAssertEqual(resolved.processorType, "ExternalProcessor")
    }

    func testMostRecentlyRegisteredResolverWinsForEachHook() throws {
        let firstData = Data("first".utf8)
        let secondData = Data("second".utf8)
        let registry = VLMProcessorLoadingRegistry(resolvers: [
            TestResolver(
                configuration: VLMProcessorConfiguration(
                    data: firstData, processorType: "FirstDeclared"),
                processorType: "FirstChoice")
        ])
        registry.register(
            TestResolver(
                configuration: VLMProcessorConfiguration(
                    data: secondData, processorType: "SecondDeclared"),
                processorType: "SecondChoice"))

        XCTAssertEqual(
            try registry.fallbackProcessorConfiguration(for: context())?.data,
            secondData)
        XCTAssertEqual(
            try registry.processorType(
                for: context(), declaredProcessorType: "CheckpointProcessor"),
            "SecondChoice")
    }

    func testNilDefersEachHookIndependently() throws {
        let fallbackData = Data("fallback".utf8)
        let registry = VLMProcessorLoadingRegistry(resolvers: [
            TestResolver(
                configuration: VLMProcessorConfiguration(
                    data: fallbackData, processorType: "FallbackProcessor"),
                processorType: "EarlierChoice"),
            TestResolver(processorType: "LaterChoice"),
        ])

        XCTAssertEqual(
            try registry.fallbackProcessorConfiguration(for: context())?.data,
            fallbackData)
        XCTAssertEqual(
            try registry.processorType(
                for: context(), declaredProcessorType: "CheckpointProcessor"),
            "LaterChoice")
    }

    func testCheckpointConfigurationDoesNotInvokeFallbackResolver() async throws {
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let checkpointData = Data(#"{"processor_class":"CheckpointProcessor"}"#.utf8)
        try checkpointData.write(
            to: directory.appending(component: "processor_config.json"))
        let registry = VLMProcessorLoadingRegistry(resolvers: [ThrowingFallbackResolver()])

        let resolved = try await resolveProcessorConfiguration(
            from: directory, context: context(), registry: registry)

        XCTAssertEqual(resolved.data, checkpointData)
        XCTAssertEqual(resolved.processorType, "CheckpointProcessor")
    }

    func testCheckpointProcessorTypeCanBeCorrected() async throws {
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let checkpointData = Data(#"{"processor_class":"IncorrectProcessor"}"#.utf8)
        try checkpointData.write(
            to: directory.appending(component: "processor_config.json"))
        let registry = VLMProcessorLoadingRegistry(resolvers: [
            TestResolver(processorType: "ExternalProcessor")
        ])

        let resolved = try await resolveProcessorConfiguration(
            from: directory, context: context(), registry: registry)

        XCTAssertEqual(resolved.data, checkpointData)
        XCTAssertEqual(resolved.processorType, "ExternalProcessor")
    }

    func testExistingConfigurationWithoutProcessorClassCanBeResolved() async throws {
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let checkpointData = Data(#"{"image_mean":[0.5,0.5,0.5]}"#.utf8)
        try checkpointData.write(
            to: directory.appending(component: "preprocessor_config.json"))
        let registry = VLMProcessorLoadingRegistry(resolvers: [
            MissingProcessorTypeResolver(processorType: "ExternalProcessor")
        ])

        let resolved = try await resolveProcessorConfiguration(
            from: directory, context: context(), registry: registry)

        XCTAssertEqual(resolved.data, checkpointData)
        XCTAssertEqual(resolved.processorType, "ExternalProcessor")
    }

    func testExistingConfigurationWithoutProcessorClassFailsWhenUnresolved() async throws {
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        try Data(#"{"image_mean":[0.5,0.5,0.5]}"#.utf8).write(
            to: directory.appending(component: "preprocessor_config.json"))

        do {
            _ = try await resolveProcessorConfiguration(
                from: directory, context: context(),
                registry: VLMProcessorLoadingRegistry())
            XCTFail("Expected an unresolved processor type to throw")
        } catch let error as ProcessorConfigError {
            XCTAssertEqual(error.filename, "preprocessor_config.json")
            guard case DecodingError.keyNotFound(let key, _) = error.underlying else {
                return XCTFail("Expected a missing processor_class decoding error")
            }
            XCTAssertEqual(key.stringValue, "processor_class")
        }
    }

    func testSharedRegistryAppliesBuiltInModelTypeRules() throws {
        let registry = VLMProcessorLoadingRegistry.shared
        XCTAssertEqual(
            try registry.processorType(
                for: context(modelType: "mistral3"),
                declaredProcessorType: "PixtralProcessor"),
            "Mistral3Processor")
        XCTAssertEqual(
            try registry.processorType(
                for: context(modelType: "gemma4_unified"),
                declaredProcessorType: "AutoProcessor"),
            "Gemma4UnifiedProcessor")
        for modelType in ["unlimited-ocr", "unlimited_ocr"] {
            XCTAssertEqual(
                try registry.processorType(
                    for: context(modelType: modelType),
                    declaredProcessorType: "DeepseekVLV2Processor"),
                "UnlimitedOCRProcessor")
        }
        XCTAssertNil(
            try registry.processorType(
                for: context(modelType: "unrelated"),
                declaredProcessorType: "AutoProcessor"))
    }

    private func context(modelType: String = "external_vlm") -> VLMProcessorLoadingContext {
        VLMProcessorLoadingContext(
            modelId: "example/model",
            modelType: modelType,
            configurationData: Data(#"{"model_type":"external_vlm"}"#.utf8))
    }

    private func makeTemporaryDirectory() throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appending(component: "VLMProcessorLoadingRegistryTests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        return directory
    }
}

private struct TestResolver: VLMProcessorLoadingResolver {
    var configuration: VLMProcessorConfiguration?
    var processorType: String?

    init(
        configuration: VLMProcessorConfiguration? = nil,
        processorType: String? = nil
    ) {
        self.configuration = configuration
        self.processorType = processorType
    }

    func fallbackProcessorConfiguration(
        for context: VLMProcessorLoadingContext
    ) throws -> VLMProcessorConfiguration? {
        configuration
    }

    func processorType(
        for context: VLMProcessorLoadingContext,
        declaredProcessorType: String?
    ) throws -> String? {
        processorType
    }
}

private struct ThrowingFallbackResolver: VLMProcessorLoadingResolver {
    enum Failure: Error {
        case unexpectedlyInvoked
    }

    func fallbackProcessorConfiguration(
        for context: VLMProcessorLoadingContext
    ) throws -> VLMProcessorConfiguration? {
        throw Failure.unexpectedlyInvoked
    }
}

private struct MissingProcessorTypeResolver: VLMProcessorLoadingResolver {
    let processorType: String

    func processorType(
        for context: VLMProcessorLoadingContext,
        declaredProcessorType: String?
    ) throws -> String? {
        declaredProcessorType == nil ? processorType : nil
    }
}
