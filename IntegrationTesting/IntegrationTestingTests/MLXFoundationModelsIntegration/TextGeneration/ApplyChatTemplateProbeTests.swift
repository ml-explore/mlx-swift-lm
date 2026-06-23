// Copyright © 2026 Apple Inc.

import Foundation
import MLXLMCommon
import Testing

@testable import MLXFoundationModels

/// Empirical probe that `applyChatTemplate` does not crash and produces tokens.
///
/// mlx-swift-lm goes straight through the model's `UserInputProcessor`, which
/// calls `applyChatTemplate` on the underlying tokenizer. These probes
/// exercise that path directly through the MLXLMCommon `Tokenizer` protocol
/// surface, with and without tools.
@Suite(.serialized, .timeLimit(.minutes(3)))
struct ApplyChatTemplateProbeTests {

    @Test
    func applyChatTemplateWithoutToolsDoesNotCrash() async throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let model = makeTestModel(TestFixtures.defaultModelID)
        let container = try await loadTestModelContainer(id: model.modelID)

        try await container.perform { context in
            let messages: [[String: any Sendable]] = [
                ["role": "user", "content": "Say hello in one word."]
            ]
            let tokens = try context.tokenizer.applyChatTemplate(messages: messages)
            #expect(!tokens.isEmpty, "Chat template without tools should produce tokens")
        }
    }

    @Test
    func applyChatTemplateWithToolsDoesNotCrash() async throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let model = makeTestModel(TestFixtures.defaultModelID)
        let container = try await loadTestModelContainer(id: model.modelID)

        try await container.perform { context in
            let messages: [[String: any Sendable]] = [
                ["role": "user", "content": "What's the weather in Tokyo?"]
            ]

            // OpenAI-style tool spec, which swift-transformers expects.
            let weatherTool: [String: any Sendable] = [
                "type": "function",
                "function": [
                    "name": "get_weather",
                    "description": "Get the current weather for a location.",
                    "parameters": [
                        "type": "object",
                        "properties": [
                            "location": [
                                "type": "string",
                                "description": "City and state, e.g. 'San Francisco, CA'.",
                            ] as [String: any Sendable]
                        ] as [String: any Sendable],
                        "required": ["location"],
                    ] as [String: any Sendable],
                ] as [String: any Sendable],
            ]

            let tokens = try context.tokenizer.applyChatTemplate(
                messages: messages,
                tools: [weatherTool]
            )
            #expect(!tokens.isEmpty, "Chat template with tools should produce tokens")
        }
    }
}
