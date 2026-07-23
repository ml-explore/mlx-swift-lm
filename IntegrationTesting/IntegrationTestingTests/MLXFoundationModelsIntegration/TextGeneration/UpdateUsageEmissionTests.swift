// Copyright © 2026 Apple Inc.
//
// Integration tests for `LanguageModelExecutorGenerationChannel.Response.updateUsage`
// emission across the three generation paths: unconstrained, guided
// (schema-constrained), and tool-calling (envelope grammar).
//
// Each test runs the executor against a real model and asserts that at
// least one `.updateUsage` event was emitted with positive prompt and
// completion token counts. We assert on the *last* observed usage rather
// than "exactly one" because SKILL.md treats `updateUsage` as
// last-write-wins -- the framework's `TranscriptWritingAggregator`
// wholesale-replaces prior totals on each event, so the contract is
// "the final emission carries authoritative cumulative totals."
//
// Suite is `.serialized` and gated on both traits because the schema/
// tool-calling tests load `ModelContainer` and require xgrammar.

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

import Testing
import Foundation
import FoundationModels
@testable import MLXFoundationModels

/// Generable type used by the guided-generation usage test. Has to be at
/// file scope for `@Generable` to emit its schema outside a function body.
@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
@Generable
private struct YesOrNoAnswer {
    @Guide(description: "Either 'yes' or 'no'.")
    var answer: String
}

/// Generable type used by the tool-calling usage test. Same pattern.
@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
@Generable
private struct ToolCallTemperatureArgs {
    @Guide(description: "City and state, e.g. 'San Francisco, CA'.")
    var location: String
}

@Suite(.serialized, .timeLimit(.minutes(5)))
struct UpdateUsageEmissionTests {

    @Test
    func usage_emittedOnUnconstrainedPath() async throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let model = makeTestModel(TestFixtures.defaultModelID)
        let executor = try makeMLXExecutor(for: model)

        let transcript = Transcript(entries: [
            .prompt(
                Transcript.Prompt(
                    segments: [
                        .text(Transcript.TextSegment(content: "Say 'hi' briefly."))
                    ], responseFormat: nil))
        ])
        let request = makeExecutorRequest(transcript: transcript)

        let stream = try await executeResponse(executor, request: request, model: model)

        let usage = try await collectFinalUsage(from: stream)
        #expect(usage.input > 0, "Prompt token count should be positive on unconstrained path")
        #expect(
            usage.output > 0, "Completion token count should be positive on unconstrained path")
    }

    @Test
    func usage_emittedOnGuidedPath() async throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let model = makeTestModel(TestFixtures.defaultModelID)
        let executor = try makeMLXExecutor(for: model)

        let transcript = Transcript(entries: [
            .prompt(
                Transcript.Prompt(
                    segments: [
                        .text(
                            Transcript.TextSegment(content: "Is the sky blue? Reply yes or no.")
                        )
                    ], responseFormat: nil))
        ])
        let request = makeExecutorRequest(
            transcript: transcript,
            schema: YesOrNoAnswer.generationSchema
        )

        let stream = try await executeResponse(executor, request: request, model: model)

        let usage = try await collectFinalUsage(from: stream)
        #expect(usage.input > 0, "Prompt token count should be positive on guided path")
        #expect(usage.output > 0, "Completion token count should be positive on guided path")
    }

    @Test
    func usage_emittedOnToolCallingPath() async throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let model = makeTestModel(TestFixtures.defaultModelID)
        let executor = try makeMLXExecutor(for: model)

        let weatherTool = Transcript.ToolDefinition(
            name: "get_weather",
            description: "Get the current weather in a given location.",
            parameters: ToolCallTemperatureArgs.generationSchema
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

        let usage = try await collectFinalUsage(from: stream)
        #expect(usage.input > 0, "Prompt token count should be positive on tool-calling path")
        #expect(
            usage.output > 0, "Completion token count should be positive on tool-calling path")
    }
}

/// Drains the stream and returns the final `(input, output)` token counts
/// observed in any `.updateUsage` event. Throws if no `.updateUsage` event
/// was seen -- the contract is that every successful generation emits at
/// least one cumulative usage event before completion.
@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
private func collectFinalUsage(
    from stream: TestResponseStream
) async throws -> (input: Int, output: Int) {
    var lastUsage: (input: Int, output: Int)?
    for try await event in stream {
        if case .updateUsage(let input, let output, _) = event {
            lastUsage = (input.totalTokenCount, output.totalTokenCount)
        }
    }
    return try #require(
        lastUsage, "Expected at least one .updateUsage event before stream completion")
}

#endif  // FoundationModelsIntegration
