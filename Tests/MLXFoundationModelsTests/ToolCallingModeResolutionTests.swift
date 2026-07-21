// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

import FoundationModels
import Testing
@testable import MLXFoundationModels

@Suite
struct ToolCallingModeResolutionTests {
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private func tool(named name: String) -> Transcript.ToolDefinition {
        Transcript.ToolDefinition(
            name: name,
            description: "Test tool",
            parameters: String.generationSchema)
    }

    @Test func nilDefaultsToAllowed() throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let mode = ToolCallingModeResolution.resolve(nil)
        #expect(mode == GenerationOptions.ToolCallingMode.allowed)
        #expect(ToolCallingModeResolution.usesAllowedBehavior(mode))
        #expect(
            try ToolCallingModeResolution.enabledToolDefinitions(
                for: mode, from: [tool(named: "real")]
            ).count == 1)
    }

    @Test func preservesAllDocumentedModes() {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let modes: [GenerationOptions.ToolCallingMode] = [
            .allowed, .required, .disallowed,
        ]
        for mode in modes {
            #expect(ToolCallingModeResolution.resolve(mode) == mode)
        }
    }

    @Test func requiredRejectsEmptyEnabledToolDefinitions() {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        #expect(throws: ToolCallingModeResolution.Error.requiredToolsMissing) {
            _ = try ToolCallingModeResolution.enabledToolDefinitions(
                for: .required, from: [])
        }
    }

    @Test func disallowedDropsEvenManuallyEnabledDefinitions() throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let definitions = try ToolCallingModeResolution.enabledToolDefinitions(
            for: .disallowed, from: [tool(named: "must_not_render")])
        #expect(definitions.isEmpty)
    }

    @Test func requiredPreservesEnabledToolDefinitions() throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let definitions = try ToolCallingModeResolution.enabledToolDefinitions(
            for: .required,
            from: [tool(named: "first"), tool(named: "second")])
        #expect(definitions.map(\.name) == ["first", "second"])
        #expect(!ToolCallingModeResolution.usesAllowedBehavior(.required))
        #expect(!ToolCallingModeResolution.usesAllowedBehavior(.disallowed))
    }

    @Test func requiredGuidedDefinitionsExcludeTheResponseEscape() throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let definitions = try ToolCallingModeResolution.guidedToolDefinitions(
            for: .required,
            from: [tool(named: "real")],
            responseSchema: nil)
        #expect(definitions.map(\.name) == ["real"])
    }

    @Test func allowedTemporarilyRetainsTheResponseEscape() throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let definitions = try ToolCallingModeResolution.guidedToolDefinitions(
            for: .allowed,
            from: [tool(named: "real")],
            responseSchema: nil)
        #expect(definitions.map(\.name) == ["real", FinalAnswerTool.toolName])
    }
}

#endif
