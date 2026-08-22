// Copyright © 2026 Apple Inc.

import Foundation
import XCTest

@testable import MLXVLM

final class VLMProcessorLoadingRegistryTests: XCTestCase {

    func testExternalResolversComposeFallbackAndTypeOverride() async throws {
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
                processorType: "FirstOverride")
        ])
        registry.register(
            TestResolver(
                configuration: VLMProcessorConfiguration(
                    data: secondData, processorType: "SecondDeclared"),
                processorType: "SecondOverride"))

        XCTAssertEqual(
            try registry.processorConfigurationFallback(for: context())?.data,
            secondData)
        XCTAssertEqual(
            try registry.processorTypeOverride(
                for: context(), declaredProcessorType: "CheckpointProcessor"),
            "SecondOverride")
    }

    func testNilDefersEachHookIndependently() throws {
        let fallbackData = Data("fallback".utf8)
        let registry = VLMProcessorLoadingRegistry(resolvers: [
            TestResolver(
                configuration: VLMProcessorConfiguration(
                    data: fallbackData, processorType: "FallbackProcessor"),
                processorType: "EarlierOverride"),
            TestResolver(processorType: "LaterOverride"),
        ])

        XCTAssertEqual(
            try registry.processorConfigurationFallback(for: context())?.data,
            fallbackData)
        XCTAssertEqual(
            try registry.processorTypeOverride(
                for: context(), declaredProcessorType: "CheckpointProcessor"),
            "LaterOverride")
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

    func testBuiltInTypeRulesUseThePublicResolverPath() throws {
        let resolver = ModelTypeProcessorOverrideResolver(processorTypes: [
            "mistral3": "Mistral3Processor",
            "gemma4_unified": "Gemma4UnifiedProcessor",
        ])

        XCTAssertEqual(
            try resolver.processorTypeOverride(
                for: context(modelType: "mistral3"),
                declaredProcessorType: "PixtralProcessor"),
            "Mistral3Processor")
        XCTAssertEqual(
            try resolver.processorTypeOverride(
                for: context(modelType: "gemma4_unified"),
                declaredProcessorType: "AutoProcessor"),
            "Gemma4UnifiedProcessor")
        XCTAssertNil(
            try resolver.processorTypeOverride(
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

private struct TestResolver: VLMProcessorLoadingResolving {
    var configuration: VLMProcessorConfiguration?
    var processorType: String?

    init(
        configuration: VLMProcessorConfiguration? = nil,
        processorType: String? = nil
    ) {
        self.configuration = configuration
        self.processorType = processorType
    }

    func processorConfigurationFallback(
        for context: VLMProcessorLoadingContext
    ) throws -> VLMProcessorConfiguration? {
        configuration
    }

    func processorTypeOverride(
        for context: VLMProcessorLoadingContext,
        declaredProcessorType: String
    ) throws -> String? {
        processorType
    }
}

private struct ThrowingFallbackResolver: VLMProcessorLoadingResolving {
    enum Failure: Error {
        case unexpectedlyInvoked
    }

    func processorConfigurationFallback(
        for context: VLMProcessorLoadingContext
    ) throws -> VLMProcessorConfiguration? {
        throw Failure.unexpectedlyInvoked
    }
}
