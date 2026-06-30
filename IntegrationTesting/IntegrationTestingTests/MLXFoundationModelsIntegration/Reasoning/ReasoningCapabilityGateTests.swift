// Copyright © 2025 Apple Inc.

#if FoundationModelsIntegration

    import Foundation
    import FoundationModels
    import Testing

    @testable import MLXFoundationModels
    import MLXLMCommon

    /// The declared-capability reasoning gate.
    ///
    /// On-device characterization (no-leak streaming, real-model behavior) is in
    /// `ReasoningCapabilityGateOnDeviceTests`. Here we keep the
    /// suite focused on the throwing-path that fires before any token is
    /// generated, which can run anywhere the FM trait compiles.
    @Suite(.serialized, .timeLimit(.minutes(15)))
    struct ReasoningCapabilityGateTests {

        enum Models {
            static let qwen3 = "mlx-community/Qwen3-1.7B-4bit"
            static let r1Distill = "mlx-community/DeepSeek-R1-Distill-Qwen-1.5B-4bit"
        }

        @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
        private func promptTranscript(_ text: String) -> Transcript {
            Transcript(entries: [
                .prompt(
                    Transcript.Prompt(
                        segments: [.text(Transcript.TextSegment(content: text))],
                        responseFormat: nil))
            ])
        }

        /// .reasoning omitted on a model whose inferred reasoning strategy is .alwaysOn must
        /// raise `unsupportedCapability` before generation — never silently leak
        /// `<think>` into the response.
        @Test func alwaysOnRefusesWhenReasoningOmitted() async throws {
            guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
            let model = makeTestModel(
                Models.r1Distill,
                capabilities: [])
            let executor = try makeMLXExecutor(for: model)
            let request = makeExecutorRequest(
                transcript: promptTranscript("Hello"),
                generationOptions: GenerationOptions(maximumResponseTokens: 16))
            let stream = try await executeResponse(executor, request: request, model: model)
            await #expect(throws: LanguageModelError.self) {
                for try await _ in stream {}
            }
        }

        /// .reasoning omitted on a toggleable model (Qwen3 .templateFlag) must
        /// succeed — the prompt-level disable kicks in and no <think> appears in
        /// the response.
        @Test func toggleableModelHonorsReasoningOmission() async throws {
            guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
            let model = makeTestModel(
                Models.qwen3,
                capabilities: [])
            let executor = try makeMLXExecutor(for: model)
            let request = makeExecutorRequest(
                transcript: promptTranscript("Reply with exactly the word OK."),
                generationOptions: GenerationOptions(maximumResponseTokens: 64))
            let stream = try await executeResponse(executor, request: request, model: model)
            var response = ""
            var reasoning = ""
            for try await event in stream {
                if let r = event as? LanguageModelExecutorGenerationChannel.Response,
                    case .appendText(let fragment) = r.action
                {
                    response += fragment.content
                } else if let r = event as? LanguageModelExecutorGenerationChannel.Reasoning,
                    case .appendText(let fragment) = r.action
                {
                    reasoning += fragment.content
                }
            }
            // No <think> leak.
            #expect(!response.contains("<think>"))
            #expect(!response.contains("</think>"))
            // Reasoning isn't declared, so no .reasoning events.
            #expect(reasoning.isEmpty)
        }

        // MARK: - Gate must apply to tool-calling and schema paths too

        /// .alwaysOn model + tool-calling + .reasoning OMITTED must throw
        /// `unsupportedCapability` before generation. The gate is path-independent:
        /// the same error fires on the tools path, schema path, and
        /// unconstrained path alike.
        @Test func alwaysOnRefusesWhenReasoningOmittedWithTools() async throws {
            guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
            let model = makeTestModel(
                Models.r1Distill,
                capabilities: [.toolCalling])
            let executor = try makeMLXExecutor(for: model)
            let weatherTool = Transcript.ToolDefinition(
                name: "get_weather",
                description: "Get the current weather in a given location.",
                parameters: Int.generationSchema)
            let request = makeExecutorRequest(
                transcript: promptTranscript("What is the weather in Tokyo?"),
                enabledTools: [weatherTool],
                generationOptions: GenerationOptions(maximumResponseTokens: 16))
            let stream = try await executeResponse(executor, request: request, model: model)
            await #expect(throws: LanguageModelError.self) {
                for try await _ in stream {}
            }
        }

        /// .alwaysOn model + schema + .reasoning OMITTED must throw
        /// `unsupportedCapability` before generation.
        @Test func alwaysOnRefusesWhenReasoningOmittedWithSchema() async throws {
            guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
            let model = makeTestModel(
                Models.r1Distill,
                capabilities: [.guidedGeneration])
            let executor = try makeMLXExecutor(for: model)
            let request = makeExecutorRequest(
                transcript: promptTranscript("Pick a number."),
                schema: Int.generationSchema,
                generationOptions: GenerationOptions(maximumResponseTokens: 16))
            let stream = try await executeResponse(executor, request: request, model: model)
            await #expect(throws: LanguageModelError.self) {
                for try await _ in stream {}
            }
        }
    }

#endif  // FoundationModelsIntegration
