// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration
#if canImport(FoundationModels, _version: 2)

import FoundationModels

@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
enum ToolCallingModeResolution {
    enum Error: Swift.Error, Equatable {
        case requiredToolsMissing
    }

    static func resolve(
        _ mode: GenerationOptions.ToolCallingMode?
    ) -> GenerationOptions.ToolCallingMode {
        mode ?? .allowed
    }

    static func usesAllowedBehavior(
        _ mode: GenerationOptions.ToolCallingMode
    ) -> Bool {
        switch mode.kind {
        case .allowed:
            return true
        case .required, .disallowed:
            return false
        @unknown default:
            return true
        }
    }

    static func enabledToolDefinitions(
        for mode: GenerationOptions.ToolCallingMode,
        from definitions: [Transcript.ToolDefinition]
    ) throws -> [Transcript.ToolDefinition] {
        if usesAllowedBehavior(mode) {
            return definitions
        }
        if mode.kind == .disallowed {
            return []
        }
        guard !definitions.isEmpty else { throw Error.requiredToolsMissing }
        return definitions
    }
}

#endif
#endif
