// Copyright © 2026 Apple Inc.

#if GuidedGenerationSupport

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
    struct ToolCallingIntegrationTests {

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

            let weatherTool = Transcript.ToolDefinition(
                name: "get_weather",
                description: "Get the current weather in a given location.",
                parameters: WeatherArgs.generationSchema
            )

            let transcript = Transcript(entries: [
                .prompt(
                    Transcript.Prompt(
                        segments: [
                            .text(Transcript.TextSegment(content: "What's the weather in Tokyo?"))
                        ], responseFormat: nil))
            ])

            let request = makeExecutorRequest(
                transcript: transcript,
                enabledTools: [weatherTool]
            )

            let stream = try await executeResponse(executor, request: request, model: model)

            var sawWeatherToolCall = false
            var sawText = false
            var textContent = ""

            for try await event in stream {
                if let toolCalls = event as? LanguageModelExecutorGenerationChannel.ToolCalls,
                    case .toolCall(let toolCall) = toolCalls.action,
                    case .appendArguments(let argsDelta) = toolCall.action
                {
                    if toolCall.name == "get_weather" {
                        sawWeatherToolCall = true
                        let data = Data(argsDelta.content.utf8)
                        let parsed = try? JSONSerialization.jsonObject(with: data)
                        #expect(
                            parsed != nil,
                            "Tool call arguments should be valid JSON: \(argsDelta.content)")
                    }
                } else if let response = event as? LanguageModelExecutorGenerationChannel.Response,
                    case .appendText(let delta) = response.action
                {
                    sawText = true
                    textContent += delta.content
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

            let weatherTool = Transcript.ToolDefinition(
                name: "get_weather",
                description:
                    "Get the current weather in a given location. Use this whenever the user asks about weather, temperature, or conditions anywhere.",
                parameters: WeatherArgs.generationSchema
            )

            let transcript = Transcript(entries: [
                .prompt(
                    Transcript.Prompt(
                        segments: [
                            .text(
                                Transcript.TextSegment(
                                    content: "What's the current weather in Tokyo, Japan?"))
                        ], responseFormat: nil))
            ])

            let request = makeExecutorRequest(
                transcript: transcript,
                enabledTools: [weatherTool]
            )

            let stream = try await executeResponse(executor, request: request, model: model)

            var toolCallName: String? = nil
            var toolCallArguments: String? = nil
            var textContent = ""

            for try await event in stream {
                if let toolCalls = event as? LanguageModelExecutorGenerationChannel.ToolCalls,
                    case .toolCall(let toolCall) = toolCalls.action,
                    case .appendArguments(let argsDelta) = toolCall.action
                {
                    toolCallName = toolCall.name
                    toolCallArguments = argsDelta.content
                } else if let response = event as? LanguageModelExecutorGenerationChannel.Response,
                    case .appendText(let delta) = response.action
                {
                    textContent += delta.content
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
    }

#endif
