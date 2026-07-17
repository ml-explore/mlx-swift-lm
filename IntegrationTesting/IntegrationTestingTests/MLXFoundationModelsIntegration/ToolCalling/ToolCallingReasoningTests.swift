// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration

import Foundation
import MLX
import FoundationModels
import Testing

@testable import MLXFoundationModels
import MLXLMCommon

@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
@Generable
private struct WeatherArgs {
    @Guide(description: "City and state, e.g. 'San Francisco, CA'.")
    var location: String
}

/// Think-then-call: a reasoning model given tools reasons unconstrained
/// first, then emits a grammar-constrained tool call.
///
/// Device-only (requires a device running iOS 27.0+): loads real models. v1 family scope is
/// Qwen3/QwQ (template renders tools AND honors `enable_thinking`); R1-Distill is
/// de-scoped (tool-blind template) and must fall through to the existing
/// single-phase tool path unchanged.
@Suite(.serialized, .timeLimit(.minutes(15)))
struct ToolCallingReasoningTests {

    @Test("Setup: release GPU state from prior suites")
    func clearGPUBeforeToolCallingReasoning() async {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let before = GPU.snapshot()
        await releaseAllGPUMemory()
        let after = GPU.snapshot()
        let freed = (before.activeMemory - after.activeMemory) / (1024 * 1024)
        let cache = before.cacheMemory / (1024 * 1024)
        print("[ToolCallingReasoningSetup] freed \(freed)MB active, \(cache)MB cache")
    }

    enum Models {
        static let qwen3 = "mlx-community/Qwen3-1.7B-4bit"
        static let r1Distill = "mlx-community/DeepSeek-R1-Distill-Qwen-1.5B-4bit"
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private static func weatherTool() -> Transcript.ToolDefinition {
        Transcript.ToolDefinition(
            name: "get_weather",
            description: "Get the current weather in a given location. "
                + "Use this whenever the user asks about weather, temperature, or conditions.",
            parameters: WeatherArgs.generationSchema)
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private func weatherTranscript() -> Transcript {
        Transcript(entries: [
            .prompt(
                Transcript.Prompt(
                    segments: [
                        .text(Transcript.TextSegment(content: "What's the weather in Tokyo?"))
                    ],
                    responseFormat: nil))
        ])
    }

    /// Streams a tool-calling response, capturing reasoning/response text, the
    /// first tool call, and whether any reasoning arrived before the first tool call.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private struct Collected {
        var reasoning = ""
        var response = ""
        var toolCallName: String?
        var toolArgs = ""
        var reasoningBeforeToolCall = false
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private func collect(_ stream: TestResponseStream) async throws -> Collected {
        var c = Collected()
        for try await event in stream {
            if case .appendText(let chunk, _, .reasoning) = event {
                c.reasoning += chunk
            } else if case .toolCall(_, let name, let arguments) = event {
                if c.toolCallName == nil {
                    c.toolCallName = name
                    c.reasoningBeforeToolCall = !c.reasoning.isEmpty
                }
                c.toolArgs += arguments
            } else if case .appendText(let chunk, _, .response) = event {
                c.response += chunk
            }
        }
        return c
    }

    private func leaks(_ s: String) -> Bool { s.contains("<think>") || s.contains("</think>") }

    // MARK: - Headline: Qwen3 think-then-call

    /// Qwen3 + a weather tool: reasoning streams first (its own `.reasoning`
    /// entry), then a valid tool call — with no `<think>`/`</think>` leaking into
    /// the response or the tool-call arguments.
    @Test func qwen3ReasonsThenCallsTool() async throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let model = makeReasoningTestModel(Models.qwen3)
        let executor = try makeMLXExecutor(for: model)
        let request = makeExecutorRequest(
            transcript: weatherTranscript(),
            enabledTools: [Self.weatherTool()],
            generationOptions: GenerationOptions(maximumResponseTokens: 1024))
        let c = try await collect(
            try await executeResponse(executor, request: request, model: model))

        #expect(!c.reasoning.isEmpty, "expected reasoning before the tool call")
        #expect(c.toolCallName != nil, "expected a tool call after reasoning")
        #expect(c.reasoningBeforeToolCall, "reasoning must precede the tool call (ordered)")
        #expect(!leaks(c.reasoning) || c.reasoning.contains("<think>") == false)  // markers consumed, not echoed
        #expect(!leaks(c.response), "no reasoning markers may leak into the response")
        #expect(!leaks(c.toolArgs), "no reasoning markers may leak into tool arguments")
        if c.toolCallName == "get_weather", !c.toolArgs.isEmpty {
            let parsed =
                try? JSONSerialization.jsonObject(with: Data(c.toolArgs.utf8)) as? [String: Any]
            #expect(
                parsed?["location"] is String,
                "get_weather arguments should carry a string location")
        }
    }

    // MARK: - Gating / no-regression

    /// Thinking disabled on Qwen3 → single-phase tool calling, no reasoning.
    @Test func qwen3ThinkingDisabledStaysSinglePhase() async throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let model = makeTestModel(Models.qwen3)
        let executor = try makeMLXExecutor(for: model)
        var contextOptions = ContextOptions()
        contextOptions.reasoningLevel = .custom("no_think")
        let request = makeExecutorRequest(
            transcript: weatherTranscript(),
            enabledTools: [Self.weatherTool()],
            generationOptions: GenerationOptions(maximumResponseTokens: 256),
            contextOptions: contextOptions)
        let c = try await collect(
            try await executeResponse(executor, request: request, model: model))
        #expect(c.reasoning.isEmpty, "thinking disabled → no reasoning phase")
        #expect(
            c.toolCallName != nil || !c.response.isEmpty, "still produces a tool call or answer"
        )
        #expect(!leaks(c.response))
    }

    /// A non-reasoning model + tools → unchanged single-phase, no reasoning.
    @Test func nonReasoningModelUnchanged() async throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let model = makeTestModel(TestFixtures.gemmaModelID)
        let executor = try makeMLXExecutor(for: model)
        let request = makeExecutorRequest(
            transcript: weatherTranscript(),
            enabledTools: [Self.weatherTool()],
            generationOptions: GenerationOptions(maximumResponseTokens: 256))
        let c = try await collect(
            try await executeResponse(executor, request: request, model: model))
        #expect(c.reasoning.isEmpty)
        #expect(c.toolCallName != nil || !c.response.isEmpty)
    }

    /// R1-Distill's template is tool-blind (cannot honor `tools:`), but the
    /// path-independent capability gate fires before generation:
    /// using an `.alwaysOn` model without declaring `.reasoning` must throw
    /// `unsupportedCapability` on every path: tools, schema, and unconstrained.
    @Test func r1DistillDescopedToSinglePhase() async throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let model = makeTestModel(Models.r1Distill)
        let executor = try makeMLXExecutor(for: model)
        let request = makeExecutorRequest(
            transcript: weatherTranscript(),
            enabledTools: [Self.weatherTool()],
            generationOptions: GenerationOptions(maximumResponseTokens: 256))
        let stream = try await executeResponse(executor, request: request, model: model)
        await #expect(
            throws: LanguageModelError.self,
            "R1-Distill requires .reasoning to be declared; gate fires path-independently"
        ) {
            for try await _ in stream {}
        }
    }

    /// Cancellation during the reasoning phase unwinds cleanly (GPU sync) without
    /// crashing the serialized suite.
    @Test func cancellationDuringReasoningUnwinds() async throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let model = makeTestModel(Models.qwen3)
        let executor = try makeMLXExecutor(for: model)
        let request = makeExecutorRequest(
            transcript: weatherTranscript(),
            enabledTools: [Self.weatherTool()],
            generationOptions: GenerationOptions(maximumResponseTokens: 1024))
        let stream = try await executeResponse(executor, request: request, model: model)
        var events = 0
        for try await _ in stream {
            events += 1
            if events >= 2 { break }  // early break → respond is cancelled mid-flight
        }
        #expect(events >= 1)
    }
}

#endif  // FoundationModelsIntegration
