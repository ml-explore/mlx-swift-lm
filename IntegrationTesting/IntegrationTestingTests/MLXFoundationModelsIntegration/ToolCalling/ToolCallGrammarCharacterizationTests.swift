// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration

import Foundation
import FoundationModels
import MLX
import MLXLMCommon
import Testing

@testable import MLXFoundationModels
@testable import MLXGuidedGeneration

/// The on/off choice the model fills in for the flashlight tool. A `@Generable`
/// enum becomes a fixed set of choices in the argument schema.
@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
@Generable
private enum CharacterizationFlashlightState: String {
    case on
}

@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
@Generable
private struct CharacterizationFlashlightArguments {
    @Guide(description: "Whether to turn the flashlight on or off.")
    var state: CharacterizationFlashlightState
}

/// Characterizes the tool-call grammar/tokenizer boundary on a real model.
///
/// This is a diagnostic probe, not a behavioral guarantee. It drives the real
/// executor tool path on each target model on a fresh turn, captures what
/// actually happens at the grammar/tokenizer boundary through
/// `GuidedGenerationDiagnosticSink` (token IDs, whether the grammar terminated,
/// the exact buffer handed to the parser, and whether that buffer parses),
/// records all of it for the token-level trace, and pins one stable invariant
/// per model: whether the buffer parses as a tool call.
///
/// Rows are data-driven. Gemma-4 E2B is the motivating bug: its native
/// tool-call dialect is not the Qwen/bare-JSON shape the grammar enforces and
/// the mask appears inert, so the buffer does not parse. Qwen-3 is the control:
/// the grammar targets its dialect, so the buffer parses. When a similar
/// boundary bug surfaces on another model, add a row; the harness is reused.
@Suite(.serialized, .timeLimit(.minutes(5)))
struct ToolCallGrammarCharacterizationTests {

    struct Case: CustomStringConvertible, Sendable {
        let modelID: String
        /// The pinned invariant: whether the buffer handed to the parser is
        /// expected to parse as a valid tool call today.
        let expectsValidToolCall: Bool
        var description: String { modelID }
    }

    static let cases: [Case] = [
        // Gemma-4 E2B was the motivating bug: its tool-call buffer failed to
        // parse because the structural-tag grammar embedded a single
        // `{name, arguments}` json_schema whose property order xgrammar did
        // not hard-enforce, so greedy decoding could skip the `name` key.
        // Forcing `name` first via per-tool tag prefixes
        // (`SchemaConverter.encodeToolCallingGrammar`) fixed it: gemma now
        // emits a clean `{"name": "set_flashlight", "arguments": {...}}` that
        // parses. Pinned `true` alongside the qwen control.
        Case(modelID: TestFixtures.gemma4ModelID, expectsValidToolCall: true),
        // Control: the grammar targets Qwen's dialect, so the buffer parses.
        Case(modelID: TestFixtures.qwen3ModelID, expectsValidToolCall: true),
    ]

    private static let toolName = "set_flashlight"

    @Test("Setup: release GPU state from prior suites")
    func clearGPUBeforeCharacterization() async {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        await releaseAllGPUMemory()
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private func flashlightTool() -> Transcript.ToolDefinition {
        Transcript.ToolDefinition(
            name: Self.toolName,
            description: "Turn the device flashlight (torch) on or off.",
            parameters: CharacterizationFlashlightArguments.generationSchema)
    }

    /// A fresh turn: instructions plus a single user request that warrants a
    /// tool call. No prior tool call/output, so the model's first action this
    /// turn is expected to be a same-turn tool call, which is the path we trace.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private func freshTurnTranscript() -> Transcript {
        let instructions = Transcript.Instructions(
            segments: [
                .text(
                    Transcript.TextSegment(
                        content:
                            "You control the device flashlight. When the user asks, call the flashlight tool to turn the light on or off."
                    ))
            ],
            toolDefinitions: [flashlightTool()])
        let prompt = Transcript.Prompt(
            segments: [.text(Transcript.TextSegment(content: "Turn on the flashlight."))],
            responseFormat: nil)
        return Transcript(entries: [.instructions(instructions), .prompt(prompt)])
    }

    @Test(arguments: ToolCallGrammarCharacterizationTests.cases)
    func characterizeToolCallBoundary(_ testCase: Case) async throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

        let model = makeTestModel(testCase.modelID)
        let executor = try makeMLXExecutor(for: model)
        let request = makeExecutorRequest(
            transcript: freshTurnTranscript(),
            enabledTools: [flashlightTool()],
            generationOptions: GenerationOptions(
                maximumResponseTokens: 128,
                toolCallingMode: .required))

        let sink = GuidedGenerationDiagnosticSink()

        // Bind the sink so the real executor's generation loop and tool path
        // record into it. The unstructured producer Task created inside
        // executeResponse inherits this task-local binding.
        try await GuidedGenerationDiagnosticSink.$current.withValue(sink) {
            let stream = try await executeResponse(executor, request: request, model: model)
            for try await _ in stream {}  // drain to completion
        }

        // Decode the sampled token IDs back to text for the trace.
        let sampledText = try await model.loadContainer().perform { context in
            context.tokenizer.decode(tokenIds: sink.sampledTokenIDs)
        }

        // Record everything (never fails): this block is the token-level trace
        // used to scope the later fix-plan tasks.
        print(
            """
            [ToolCallCharacterization][\(testCase.modelID)]
              sampledTokenIDs (\(sink.sampledTokenIDs.count)): \(sink.sampledTokenIDs)
              sampledDecoded: \(sampledText)
              fastForwardTokenIDs (\(sink.fastForwardTokenIDs.count)): \(sink.fastForwardTokenIDs)
              grammarTerminated: \(sink.grammarTerminated)
              generatedTokenCount: \(sink.generatedTokenCount)
              incompleteOutput: \(sink.incompleteOutput)
              parsedAsToolCall: \(String(describing: sink.parsedAsToolCall))
              parsedName: \(String(describing: sink.parsedName))
              finalBuffer: \(String(describing: sink.finalBuffer))
            [/ToolCallCharacterization]
            """)

        // Sanity: generation actually ran and the tool path handed off a buffer.
        #expect(
            sink.generatedTokenCount >= 1,
            "No tokens were generated (\(testCase.modelID))")
        #expect(
            sink.finalBuffer != nil,
            "Tool path never handed a buffer to the parser (\(testCase.modelID))")

        // Pinned invariant: whether the buffer parses as a tool call. Robust
        // across both leak signatures (comma-less JSON and pure native syntax),
        // since both fail the parse.
        #expect(
            sink.parsedAsToolCall == testCase.expectsValidToolCall,
            """
            Parse verdict for \(testCase.modelID) was \
            \(String(describing: sink.parsedAsToolCall)), expected \
            \(testCase.expectsValidToolCall). Buffer: \
            \(String(describing: sink.finalBuffer)). A flip here is a finding: \
            for the characterized-bug row it likely means the fix landed (update \
            expectsValidToolCall to true); for the control row it means the \
            grammar path regressed.
            """)
    }
}

#endif  // FoundationModelsIntegration
