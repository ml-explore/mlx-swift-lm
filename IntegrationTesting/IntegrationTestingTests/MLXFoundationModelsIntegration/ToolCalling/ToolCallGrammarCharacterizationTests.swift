// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration

import Foundation
import FoundationModels
import MLX
import MLXLMCommon
import Testing

@testable import MLXFoundationModels
@testable import MLXGuidedGeneration

/// The fixed `on` state the model fills in for this characterization tool.
/// A one-case `@Generable` enum bounds the generated argument.
@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
@Generable
private enum CharacterizationFlashlightState: String {
    case on
}

@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
@Generable
private struct CharacterizationFlashlightArguments {
    @Guide(description: "The fixed on state for this characterization.")
    var state: CharacterizationFlashlightState
}

/// Characterizes the explicit `.required` guided tool-call grammar/tokenizer
/// boundary on real models.
///
/// This drives the real executor's required guided path on each target model on
/// a fresh turn and captures the grammar/tokenizer boundary through
/// `GuidedGenerationDiagnosticSink` (token IDs, whether the grammar terminated,
/// the exact buffer handed to the parser, and whether that buffer parses),
/// recording a token-level trace while pinning valid parsing as the stable
/// invariant.
///
/// Both Gemma-4 E2B and Qwen-3 are expected to terminate the guided grammar and
/// hand the parser a valid developer tool call. Add rows when another model
/// needs this same required-mode boundary coverage.
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
        // Required guided grammar must terminate and parse a developer tool for
        // each supported family.
        Case(modelID: TestFixtures.gemma4ModelID, expectsValidToolCall: true),
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
            description: "Turn the device flashlight (torch) on.",
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
                            "You control the device flashlight. Call the flashlight tool to turn the light on."
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

        // Record the token-level trace without affecting the assertions below.
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

        // Pinned invariant: required guided output must parse as a tool call.
        #expect(
            sink.parsedAsToolCall == testCase.expectsValidToolCall,
            """
            Parse verdict for \(testCase.modelID) was \
            \(String(describing: sink.parsedAsToolCall)), expected \
            \(testCase.expectsValidToolCall). Buffer: \
            \(String(describing: sink.finalBuffer)). A false verdict is a \
            grammar-path regression.
            """)
    }
}

#endif  // FoundationModelsIntegration
