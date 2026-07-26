// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

import Testing
import Foundation
import MLX
import FoundationModels
@testable import MLXFoundationModels
@testable import MLXGuidedGeneration

@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
@Generable
private struct WeatherArgs {
    @Guide(description: "City and state, e.g. 'San Francisco, CA'.")
    var location: String
}

@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
@Generable
private struct Greeting {
    var message: String
}

@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
@Generable
private enum DeveloperMarkerValue: String {
    case recorded
}

@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
@Generable
private struct DeveloperMarkerArguments {
    var value: DeveloperMarkerValue
}

/// End-to-end tests for Foundation Models tool-calling modes.
///
/// This suite validates that when a request has `enabledTools`, the
/// executor (1) formats tools into the prompt via the tokenizer's native
/// tool-aware chat template, (2) routes native `.allowed` output to either
/// a response or a real tool call, and (3) constrains `.required` output to
/// a developer-tool JSON envelope via xgrammar.
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

    @Test func allowedCanAnswerWithoutCallingATool() async throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let model = makeTestModel(TestFixtures.defaultModelID)
        let executor = try makeMLXExecutor(for: model)
        let request = makeExecutorRequest(
            transcript: singlePromptTranscript("Say hello in one short sentence."),
            enabledTools: [weatherTool()],
            generationOptions: GenerationOptions(
                maximumResponseTokens: 64,
                toolCallingMode: .allowed))

        let stream = try await executeResponse(executor, request: request, model: model)
        var calls: [String] = []
        var response = ""
        for try await event in stream {
            if case .toolCall(_, let name, _) = event { calls.append(name) }
            if case .appendText(let text, _, .response) = event { response += text }
        }
        #expect(calls.isEmpty)
        #expect(!response.isEmpty)
    }

    @Test func allowedCanChooseARealTool() async throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let model = makeTestModel(TestFixtures.defaultModelID)
        let executor = try makeMLXExecutor(for: model)
        let request = makeExecutorRequest(
            transcript: singlePromptTranscript("What is the current weather in Tokyo?"),
            enabledTools: [weatherTool()],
            generationOptions: GenerationOptions(
                maximumResponseTokens: 128,
                toolCallingMode: .allowed))

        let stream = try await executeResponse(executor, request: request, model: model)
        var calls: [String] = []
        var response = ""
        for try await event in stream {
            if case .toolCall(_, let name, _) = event { calls.append(name) }
            if case .appendText(let text, _, .response) = event { response += text }
        }
        #expect(calls == ["get_weather"])
        #expect(response.isEmpty)
    }

    @Test func allowedResponseStillHonorsResponseSchema() async throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let model = makeTestModel(TestFixtures.defaultModelID)
        let executor = try makeMLXExecutor(for: model)
        let request = makeExecutorRequest(
            transcript: singlePromptTranscript("Say hello in one short sentence."),
            enabledTools: [weatherTool()],
            schema: Greeting.generationSchema,
            generationOptions: GenerationOptions(
                maximumResponseTokens: 128,
                toolCallingMode: .allowed))

        let stream = try await executeResponse(executor, request: request, model: model)
        var calls: [String] = []
        var responseJSON = ""
        for try await event in stream {
            if case .toolCall(_, let name, _) = event { calls.append(name) }
            if case .appendText(let text, _, .response) = event { responseJSON += text }
        }
        #expect(calls.isEmpty)
        let content = try GeneratedContent(json: responseJSON)
        _ = try Greeting(content)
    }

    /// With tool-aware prompt formatting plus native allowed generation, the
    /// model can both *see* the available tools in the prompt and emit them in
    /// its trained format.
    /// For a weather query, Qwen should pick `get_weather` rather than
    /// answering without the requested live data.
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

    @Test func requiredGuidedCancellationPropagates() async throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let model = makeTestModel(TestFixtures.defaultModelID)
        let executor = try makeMLXExecutor(for: model)
        let request = makeExecutorRequest(
            transcript: singlePromptTranscript("What is the weather in Tokyo?"),
            enabledTools: [weatherTool()],
            generationOptions: GenerationOptions(
                maximumResponseTokens: 128,
                toolCallingMode: .required))
        let sink = GuidedGenerationDiagnosticSink(cancelAfterEmitCount: 1)
        let stream = try await executeResponse(
            executor,
            request: request,
            model: model,
            guidedGenerationSink: sink)

        var sawCancellation = false
        do {
            for try await _ in stream {}
        } catch is CancellationError {
            sawCancellation = true
        }
        await stream.cancelAndWait()

        #expect(sink.emitCount >= 1, "required generation must reach its guided emit")
        #expect(sawCancellation, "required guided cancellation must not become normal EOS")
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

    @Test func formerSyntheticNameIsADeveloperTool() async throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let model = makeTestModel(TestFixtures.defaultModelID)
        let executor = try makeMLXExecutor(for: model)
        let developerTool = Transcript.ToolDefinition(
            name: "mlx_final_answer",
            description: "Record a developer-owned marker.",
            parameters: DeveloperMarkerArguments.generationSchema)
        let request = makeExecutorRequest(
            transcript: singlePromptTranscript("Record the marker."),
            enabledTools: [developerTool],
            generationOptions: GenerationOptions(
                maximumResponseTokens: 64,
                toolCallingMode: .required))

        let stream = try await executeResponse(executor, request: request, model: model)
        var names: [String] = []
        var response = ""
        for try await event in stream {
            if case .toolCall(_, let name, _) = event { names.append(name) }
            if case .appendText(let text, _, .response) = event { response += text }
        }

        #expect(names == ["mlx_final_answer"])
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
