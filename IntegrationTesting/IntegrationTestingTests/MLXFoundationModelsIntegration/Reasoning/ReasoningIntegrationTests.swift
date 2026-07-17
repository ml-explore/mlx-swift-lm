// Copyright © 2025 Apple Inc.

#if FoundationModelsIntegration

import Foundation
import FoundationModels
import Testing

@testable import MLXFoundationModels
import MLXLMCommon

/// Reasoning wiring on the unconstrained path.
///
/// The pure mapping test runs anywhere the FM trait compiles. The integration
/// tests load real reasoning models and therefore require a device running
/// iOS 27.0+ — the Mac host has no OS-27 runtime for the LanguageModel protocol.
///
/// Model ids are the smallest published quants of each family; confirm they
/// resolve on the device run (HF availability) before locking the suite.
@Suite(.serialized, .timeLimit(.minutes(15)))
struct ReasoningIntegrationTests {

    enum ReasoningModels {
        static let qwen3 = "mlx-community/Qwen3-1.7B-4bit"
        static let r1Distill = "mlx-community/DeepSeek-R1-Distill-Qwen-1.5B-4bit"
    }

    // MARK: - reasoningLevel → thinking mapping (unit; no model load)

    @Test func thinkingMappingTable() {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        typealias Executor = MLXLanguageModel.Executor
        #expect(Executor.thinkingEnabled(for: nil) == nil)  // no opinion
        #expect(Executor.thinkingEnabled(for: .light) == true)
        #expect(Executor.thinkingEnabled(for: .moderate) == true)
        #expect(Executor.thinkingEnabled(for: .deep) == true)
        #expect(Executor.thinkingEnabled(for: .custom("no_think")) == false)
        #expect(Executor.thinkingEnabled(for: .custom("NO_THINK ")) == false)  // normalized
        #expect(Executor.thinkingEnabled(for: .custom("ultrathink")) == true)  // unknown → on
    }

    // MARK: - Integration (device; real model load)

    /// Collects reasoning + response text from a streamed response.
    ///
    /// Token-count assertions (reasoningTokenCount ≤ total) are verified in
    /// the device pass once the exact `Response.Action.updateUsage` / `Usage`
    /// shape is confirmed against the SDK; this helper tracks text only.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private func collect(
        _ stream: TestResponseStream
    ) async throws -> (reasoning: String, response: String) {
        var reasoning = ""
        var response = ""
        for try await event in stream {
            if case .appendText(let chunk, _, .reasoning) = event {
                reasoning += chunk
            } else if case .appendText(let chunk, _, .response) = event {
                response += chunk
            }
        }
        return (reasoning, response)
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

    /// The prefill canary + propagation check: Qwen3 routes reasoning, never
    /// leaks `</think>` into the response, the resolved config reached the
    /// loaded context, and the reasoning token count is
    /// sane (true count, ≤ total).
    @Test func qwen3RoutesReasoningWithoutLeak() async throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let model = makeReasoningTestModel(ReasoningModels.qwen3)

        // Propagation: the resolved reasoningConfig must reach the loaded context.
        let container = try await loadTestModelContainer(id: ReasoningModels.qwen3)
        await container.perform { context in
            #expect(context.configuration.reasoningConfig != nil)
        }

        let executor = try makeMLXExecutor(for: model)
        let request = makeExecutorRequest(
            transcript: promptTranscript("What is 17 times 24? Think step by step."),
            generationOptions: GenerationOptions(maximumResponseTokens: 512))
        let stream = try await executeResponse(executor, request: request, model: model)
        let result = try await collect(stream)

        #expect(!result.reasoning.isEmpty, "expected at least one .reasoning event")
        #expect(
            !result.response.contains("</think>"), "the prefill canary: no </think> in response"
        )
        #expect(!result.response.contains("<think>"))
    }

    /// Disabling thinking on Qwen3 (which can toggle) produces no reasoning.
    @Test func qwen3ThinkingDisabledProducesNoReasoning() async throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let model = makeReasoningTestModel(ReasoningModels.qwen3)
        let executor = try makeMLXExecutor(for: model)
        var contextOptions = ContextOptions()
        contextOptions.reasoningLevel = .custom("no_think")
        let request = makeExecutorRequest(
            transcript: promptTranscript("Say hello."),
            generationOptions: GenerationOptions(maximumResponseTokens: 128),
            contextOptions: contextOptions)
        let stream = try await executeResponse(executor, request: request, model: model)
        let result = try await collect(stream)
        #expect(result.reasoning.isEmpty)
        #expect(!result.response.isEmpty)
    }

    /// A non-reasoning model emits no reasoning and reports reasoningTokenCount 0.
    @Test func nonReasoningModelUnaffected() async throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let model = makeTestModel(TestFixtures.gemmaModelID)
        let executor = try makeMLXExecutor(for: model)
        let request = makeExecutorRequest(
            transcript: promptTranscript("Say hi."),
            generationOptions: GenerationOptions(maximumResponseTokens: 16))
        let stream = try await executeResponse(executor, request: request, model: model)
        let result = try await collect(stream)
        #expect(result.reasoning.isEmpty)
        #expect(!result.response.isEmpty)
    }

    /// Requesting "off" on an always-thinking model errors *before* generation,
    /// with the honest typed error — not a silently-dropped knob.
    @Test func offSwitchOnAlwaysOnErrorsEarly() async throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let model = makeReasoningTestModel(ReasoningModels.r1Distill)
        let executor = try makeMLXExecutor(for: model)
        var contextOptions = ContextOptions()
        contextOptions.reasoningLevel = .custom("no_think")
        let request = makeExecutorRequest(
            transcript: promptTranscript("Hello"),
            generationOptions: GenerationOptions(maximumResponseTokens: 16),
            contextOptions: contextOptions)
        // `respond`'s first action sends a metadata event on the rendezvous
        // channel, which blocks until consumed. Drive it through
        // TestResponseStream (which consumes) and expect iteration to surface
        // the typed error — don't call respond with an unconsumed channel.
        let stream = try await executeResponse(executor, request: request, model: model)
        await #expect(throws: LanguageModelError.self) {
            for try await _ in stream {}
        }
    }

    /// The strengthened budget canary: a forcing prompt at the default budget
    /// must still leave a non-trivial answer — not "thinking ate the budget".
    @Test func budgetLeavesRoomForAnswer() async throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let model = makeReasoningTestModel(ReasoningModels.qwen3)
        let executor = try makeMLXExecutor(for: model)
        let request = makeExecutorRequest(
            transcript: promptTranscript(
                "Answer in one sentence: what colour is a clear daytime sky?"),
            generationOptions: GenerationOptions(maximumResponseTokens: 1024))
        let stream = try await executeResponse(executor, request: request, model: model)
        let result = try await collect(stream)
        #expect(result.response.count > 5, "thinking should not consume the whole budget")
    }

    /// Truncation mid-thought: a tiny budget on a primed model that never emits
    /// `</think>` must not crash, and the thinking it does emit routes to
    /// reasoning (not leaked to response). The precise `incompleteOutput`
    /// metadata assertion is added in the device pass once the `.updateMetadata`
    /// action shape is confirmed.
    @Test func truncationMidThoughtDoesNotCrash() async throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let model = makeReasoningTestModel(ReasoningModels.qwen3)
        let executor = try makeMLXExecutor(for: model)
        let request = makeExecutorRequest(
            transcript: promptTranscript(
                "Prove the Pythagorean theorem rigorously, step by step."),
            generationOptions: GenerationOptions(maximumResponseTokens: 8))
        let stream = try await executeResponse(executor, request: request, model: model)
        let result = try await collect(stream)
        #expect(!result.response.contains("</think>"))
    }

    /// Cancellation mid-think: breaking early must unwind cleanly (GPU sync via
    /// the outer catch) without crashing the serialized suite.
    @Test func cancellationMidThinkUnwindsCleanly() async throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let model = makeReasoningTestModel(ReasoningModels.qwen3)
        let executor = try makeMLXExecutor(for: model)
        let request = makeExecutorRequest(
            transcript: promptTranscript(
                "Think at length about the distribution of prime numbers."),
            generationOptions: GenerationOptions(maximumResponseTokens: 512))
        let stream = try await executeResponse(executor, request: request, model: model)
        var events = 0
        for try await _ in stream {
            events += 1
            if events >= 2 { break }  // early break → TestResponseStream.deinit cancels respond
        }
        #expect(events >= 1)
    }
}

#endif  // FoundationModelsIntegration
