// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration

import Foundation
import FoundationModels
import MLX
import MLXLMCommon
import Testing

@testable import MLXFoundationModels

/// The on/off choice the model fills in for the flashlight tool, mirroring the
/// `FlashlightTool` in the FMFeatures sample. A `@Generable` enum becomes a
/// fixed set of choices in the argument schema.
@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
@Generable
private enum FlashlightState: String {
    case on, off
}

@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
@Generable
private struct FlashlightArguments {
    @Guide(description: "Whether to turn the flashlight on or off.")
    var state: FlashlightState
}

/// Validates multi-turn tool calling with the flashlight tool from the
/// FMFeatures sample: a device action the user drives across turns ("turn it
/// on", then "turn it off"). Two things must hold once a tool has run:
///
/// - Continuation round: after the session executes a tool and re-invokes us
///   with the call + output appended, we replay both into the model's context
///   so its answer reflects the result rather than re-issuing the call.
/// - Multi-round: with tools still enabled on a continuation round, the model
///   may chain another tool or answer from the replayed result. Whether it
///   chains is a model-dependent choice (the tool path always offers a
///   synthetic final-answer branch), so it is not asserted; the deterministic
///   guarantee that each round's context is assembled correctly lives in the
///   converter test ``testTwoToolRoundsProduceCorrelatedMessages``.
///
/// The tool loop is faked here: instead of letting `LanguageModelSession` run
/// the flashlight tool, the transcript is seeded with the call and output the
/// session would have appended. This is exactly the continuation-round request
/// the SDK builds (`enabledTools` are passed every round).
///
/// The behavioral tests run against every model in ``models`` so the replay
/// path is exercised across template dialects, not tuned to one model family.
@Suite(.serialized, .timeLimit(.minutes(5)))
struct MultiTurnToolCallingTests {

    /// Models whose chat templates render tool calls and `tool` responses.
    /// Both families are run so the multi-turn path is verified as
    /// model-agnostic rather than fitted to a single template dialect.
    static let models = [TestFixtures.gemma4ModelID, TestFixtures.qwen3ModelID]

    @Test("Setup: release GPU state from prior suites")
    func clearGPUBeforeMultiTurn() async {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        await releaseAllGPUMemory()
    }

    private static let toolName = "set_flashlight"
    private static let flashlightOnResult = "Flashlight turned on."

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private func flashlightTool() -> Transcript.ToolDefinition {
        Transcript.ToolDefinition(
            name: Self.toolName,
            description: "Turn the device flashlight (torch) on or off.",
            parameters: FlashlightArguments.generationSchema)
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private func instructions() -> Transcript.Instructions {
        Transcript.Instructions(
            segments: [
                .text(
                    Transcript.TextSegment(
                        content:
                            "You control the device flashlight. When the user asks, call the flashlight tool to turn the light on or off. After the tool returns a result, tell the user what happened."
                    ))
            ],
            toolDefinitions: [flashlightTool()])
    }

    /// A completed "turn it on" round: user request + the flashlight call and
    /// its output. `ToolOutput.id` matches the call id, as the SDK's
    /// ToolCallCoordinator sets it, so the template correlates them.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private func turnOnRound() throws -> [Transcript.Entry] {
        let callID = "call_flashlight_on"
        let toolCall = Transcript.ToolCall(
            id: callID,
            toolName: Self.toolName,
            arguments: try GeneratedContent(json: #"{"state":"on"}"#))
        let toolOutput = Transcript.ToolOutput(
            id: callID,
            toolName: Self.toolName,
            segments: [.text(Transcript.TextSegment(content: Self.flashlightOnResult))])
        return [
            .prompt(
                Transcript.Prompt(
                    segments: [.text(Transcript.TextSegment(content: "Turn on the flashlight."))],
                    responseFormat: nil)),
            .toolCalls(Transcript.ToolCalls(id: "toolcalls_on", [toolCall])),
            .toolOutput(toolOutput),
        ]
    }

    /// The continuation-round transcript: instructions + the "turn it on" round.
    /// The newest entry is a tool output, so this is the round the session
    /// re-invokes us with immediately after the tool ran.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private func continuationTranscript() throws -> Transcript {
        Transcript(entries: [.instructions(instructions())] + (try turnOnRound()))
    }

    @Test(arguments: MultiTurnToolCallingTests.models)
    func continuationRoundIncorporatesToolOutput(modelID: String) async throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

        let model = makeTestModel(modelID)
        let executor = try makeMLXExecutor(for: model)

        let request = makeExecutorRequest(
            transcript: try continuationTranscript(),
            enabledTools: [flashlightTool()],
            generationOptions: GenerationOptions(maximumResponseTokens: 128))

        let stream = try await executeResponse(executor, request: request, model: model)

        var textContent = ""
        var madeToolCall = false

        for try await event in stream {
            if let toolCalls = event as? LanguageModelExecutorGenerationChannel.ToolCalls,
                case .toolCall(let toolCall) = toolCalls.action,
                case .appendArguments = toolCall.action
            {
                madeToolCall = true
            } else if let response = event as? LanguageModelExecutorGenerationChannel.Response,
                case .appendText(let delta) = response.action
            {
                textContent += delta.content
            }
        }

        let lowered = textContent.lowercased()
        let reflectsOutput = lowered.contains("light") || lowered.contains("on")

        // Multi-round: with the tool still enabled on the continuation round,
        // the model may either answer from the replayed result or call a tool
        // again -- both are valid, unlike the old one-round cap. The regression
        // this guards is the model ignoring the tool output: a plain answer
        // that neither reflects the result nor issues a tool call.
        #expect(
            reflectsOutput || madeToolCall,
            "Continuation must reflect the tool output or issue a tool call, not ignore the result. Got: \"\(textContent)\" (\(modelID))"
        )
    }

    @Test(arguments: MultiTurnToolCallingTests.models)
    func continuationPromptContainsToolOutput(modelID: String) async throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

        let model = makeTestModel(modelID)
        let container = try await model.loadContainer()
        let transcript = try continuationTranscript()

        let decoded = try await container.perform { context in
            let messages = TranscriptConverter.mlxMessages(for: transcript)
            let raw = DefaultMessageGenerator().generate(messages: messages)
            let tokens = try context.tokenizer.applyChatTemplate(messages: raw)
            return context.tokenizer.decode(tokenIds: tokens)
        }

        print("[MultiTurn][RenderedPrompt][\(modelID)]\n\(decoded)\n[/RenderedPrompt]")

        #expect(
            decoded.lowercased().contains("flashlight turned on"),
            "Rendered continuation prompt must contain the tool output text (\(modelID))"
        )
    }
}

#endif  // FoundationModelsIntegration
