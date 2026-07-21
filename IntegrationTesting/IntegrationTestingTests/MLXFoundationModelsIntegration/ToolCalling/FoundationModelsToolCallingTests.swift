// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration

import Testing
import Foundation
import MLX
import FoundationModels
@testable import MLXFoundationModels

@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
@Generable
private struct WeatherArgs {
    @Guide(description: "City and state, e.g. 'San Francisco, CA'.")
    var location: String
}

/// End-to-end test for tool calling via guided generation.
///
/// This suite validates that when a request has `enabledTools`, the
/// executor (1) formats tools into the prompt via the tokenizer's native
/// tool-aware chat template, (2) constrains the model's output to a
/// union-of-tools JSON envelope via xgrammar, and (3) parses the result
/// into either a `toolCallDelta` (real tool) or `textDelta` (synthetic
/// final-answer tool).
@Suite(.serialized, .timeLimit(.minutes(5)))
struct FoundationModelsToolCallingTests {

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private func developerTool(named name: String) -> Transcript.ToolDefinition {
        Transcript.ToolDefinition(
            name: name,
            description: "Get the current weather in a given location.",
            parameters: WeatherArgs.generationSchema)
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private func weatherTool() -> Transcript.ToolDefinition {
        developerTool(named: "get_weather")
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private func singlePromptTranscript(_ text: String) -> Transcript {
        Transcript(entries: [
            .prompt(
                Transcript.Prompt(
                    segments: [.text(Transcript.TextSegment(content: text))],
                    responseFormat: nil))
        ])
    }

    @Test("Setup: release GPU state from prior suites")
    func clearGPUBeforeToolCalling() async {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let before = GPU.snapshot()
        await releaseAllGPUMemory()
        let after = GPU.snapshot()
        let freed = (before.activeMemory - after.activeMemory) / (1024 * 1024)
        let cache = before.cacheMemory / (1024 * 1024)
        print("[ToolCallingSetup] freed \(freed)MB active, \(cache)MB cache")
    }

    @Test
    func toolsEnabledEmitsToolCallOrText() async throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let model = makeTestModel(TestFixtures.defaultModelID)
        let executor = try makeMLXExecutor(for: model)

        let request = makeExecutorRequest(
            transcript: singlePromptTranscript("What's the weather in Tokyo?"),
            enabledTools: [weatherTool()]
        )

        let stream = try await executeResponse(executor, request: request, model: model)

        var sawWeatherToolCall = false
        var sawText = false
        var textContent = ""

        for try await event in stream {
            if case .toolCall(_, let name, let arguments) = event {
                if name == "get_weather" {
                    sawWeatherToolCall = true
                    let data = Data(arguments.utf8)
                    let parsed = try? JSONSerialization.jsonObject(with: data)
                    #expect(
                        parsed != nil,
                        "Tool call arguments should be valid JSON: \(arguments)")
                }
            } else if case .appendText(let chunk, _, .response) = event {
                sawText = true
                textContent += chunk
            }
        }

        // Exactly one of the two paths should have produced output.
        #expect(
            sawWeatherToolCall || sawText,
            "Executor with enabled tools must emit either a toolCallDelta or a textDelta"
        )

        if sawWeatherToolCall {
            #expect(
                textContent.isEmpty,
                "When a real tool call fires, no text deltas should be emitted"
            )
        } else {
            #expect(
                !textContent.isEmpty,
                "When the synthetic final-answer tool fires, text should be non-empty"
            )
        }
    }

    /// With tool-aware prompt formatting plus the tool-call grammar
    /// that allows `<tool_call>`-wrapped output, the model can both *see* the
    /// available tools in the prompt and emit them in its trained format.
    /// For a weather query, Qwen should pick `get_weather` rather than
    /// hallucinating via the synthetic final-answer path.
    @Test
    func toolAwarePromptRoutesWeatherQueryToGetWeather() async throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let model = makeTestModel(TestFixtures.defaultModelID)
        let executor = try makeMLXExecutor(for: model)

        let request = makeExecutorRequest(
            transcript: singlePromptTranscript("What's the current weather in Tokyo, Japan?"),
            enabledTools: [weatherTool()]
        )

        let stream = try await executeResponse(executor, request: request, model: model)

        var toolCallName: String? = nil
        var toolCallArguments: String? = nil
        var textContent = ""

        for try await event in stream {
            if case .toolCall(_, let name, let arguments) = event {
                toolCallName = name
                toolCallArguments = arguments
            } else if case .appendText(let chunk, _, .response) = event {
                textContent += chunk
            }
        }

        #expect(
            toolCallName == "get_weather",
            "With the tool defined in the prompt, the model should pick get_weather for a weather query. Got toolCall=\(toolCallName ?? "nil"), text=\"\(textContent.prefix(120))\""
        )

        if let args = toolCallArguments {
            let data = Data(args.utf8)
            let parsed = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
            #expect(
                parsed?["location"] is String,
                "get_weather arguments should have a string 'location' field (stricter content checks deferred)"
            )
        }
    }

    @Test func requiredCannotAnswerWithoutCallingARealTool() async throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let model = makeTestModel(TestFixtures.defaultModelID)
        let executor = try makeMLXExecutor(for: model)
        let request = makeExecutorRequest(
            transcript: singlePromptTranscript("Say hello without using a tool."),
            enabledTools: [weatherTool()],
            generationOptions: GenerationOptions(
                maximumResponseTokens: 128,
                toolCallingMode: .required))

        let stream = try await executeResponse(executor, request: request, model: model)
        var names: [String] = []
        var response = ""
        for try await event in stream {
            if case .toolCall(_, let name, _) = event { names.append(name) }
            if case .appendText(let text, _, .response) = event { response += text }
        }

        #expect(names == ["get_weather"])
        #expect(response.isEmpty)
    }

    @Test func disallowedIgnoresManuallyEnabledTools() async throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let model = makeTestModel(TestFixtures.defaultModelID)
        let executor = try makeMLXExecutor(for: model)
        let request = makeExecutorRequest(
            transcript: singlePromptTranscript("Say hello in one sentence."),
            enabledTools: [weatherTool()],
            generationOptions: GenerationOptions(
                maximumResponseTokens: 64,
                toolCallingMode: .disallowed))

        let stream = try await executeResponse(executor, request: request, model: model)
        var sawToolCall = false
        var response = ""
        for try await event in stream {
            if case .toolCall = event { sawToolCall = true }
            if case .appendText(let text, _, .response) = event { response += text }
        }

        #expect(!sawToolCall)
        #expect(!response.isEmpty)
    }

    @Test func requiredTreatsDeveloperFinalAnswerNameAsTool() async throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let model = makeTestModel(TestFixtures.defaultModelID)
        let executor = try makeMLXExecutor(for: model)
        let request = makeExecutorRequest(
            transcript: singlePromptTranscript("Use the available tool."),
            enabledTools: [developerTool(named: FinalAnswerTool.toolName)],
            generationOptions: GenerationOptions(
                maximumResponseTokens: 128,
                toolCallingMode: .required))

        let stream = try await executeResponse(executor, request: request, model: model)
        var names: [String] = []
        var response = ""
        for try await event in stream {
            if case .toolCall(_, let name, _) = event { names.append(name) }
            if case .appendText(let text, _, .response) = event { response += text }
        }

        #expect(names == [FinalAnswerTool.toolName])
        #expect(response.isEmpty)
    }

    @Test func requiredTruncationNeverEmitsResponseText() async throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let model = makeTestModel(TestFixtures.defaultModelID)
        let executor = try makeMLXExecutor(for: model)
        let request = makeExecutorRequest(
            transcript: singlePromptTranscript("What's the weather in Tokyo?"),
            enabledTools: [weatherTool()],
            generationOptions: GenerationOptions(
                maximumResponseTokens: 1,
                toolCallingMode: .required))

        let stream = try await executeResponse(executor, request: request, model: model)
        var response = ""
        var sawIncompleteOutput = false
        for try await event in stream {
            if case .appendText(let text, _, .response) = event { response += text }
            if case .updateMetadata(let metadata, _) = event {
                sawIncompleteOutput = (metadata["incompleteOutput"] as? Bool) == true
            }
        }

        #expect(response.isEmpty)
        #expect(sawIncompleteOutput)
    }
}

#endif
