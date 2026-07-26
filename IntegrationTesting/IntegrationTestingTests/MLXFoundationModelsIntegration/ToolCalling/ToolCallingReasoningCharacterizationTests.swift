// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

import Testing
import Foundation
import MLX
import FoundationModels
@testable import MLXFoundationModels
import MLXLMCommon

@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
@Generable
private enum WeatherLocation: String {
    case tokyo = "Tokyo"
}

@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
@Generable
private struct WeatherArgs {
    @Guide(description: "Location to query.")
    var location: WeatherLocation
}

/// Characterizes two empirical facts about today's tool-calling path
/// (device/manual-gated, requires a device running iOS 27.0+). Touches no production code.
///
/// What it answers:
///   1. REQUIRED ENVELOPE (`qwen3RequiredModeEmitsOnlyDeveloperToolCall`): does
///      `.required` mode emit only its enabled developer tool? The structural tag
///      is compiled in the *non-triggered* form (`xg_compile_structural_tag` with
///      `nullopt`, no `token_triggered_tags`; see `XGrammarBridge.swift:409`), so
///      guided generation starts at generated-token zero and must produce a tool
///      envelope rather than response text or reasoning markers.
///   2. TOOL-AWARE THINKING SEED (`toolAwareTemplateHonorsEnableThinking`): does the
///      3-arg `applyChatTemplate(messages:tools:additionalContext:)` produce a
///      *distinct* thinking-primed prompt on the tool path, and what `primedInside`
///      does the tool-aware tail imply per family? Tool blocks can move the
///      assistant-prompt boundary, so `primedInside` must be seeded from the
///      tool-aware tail specifically rather than the no-tools tail.
///
/// NOTE on the budget question (`maximumResponseTokens` semantics under reasoning):
/// deliberately NOT measured here — it's a protocol-contract question better settled
/// against AFM / SKILL.md than a single MLX run. Tracked separately.
@Suite(.serialized, .timeLimit(.minutes(10)))
struct ToolCallingReasoningCharacterizationTests {

    static let qwen3 = "mlx-community/Qwen3-1.7B-4bit"
    static let r1Distill = "mlx-community/DeepSeek-R1-Distill-Qwen-1.5B-4bit"
    static let thinkConfig = ReasoningConfig(
        startDelimiter: "<think>", endDelimiter: "</think>",
        promptStrategy: .templateFlag(key: "enable_thinking", defaultOn: true))

    @Test("Setup: release GPU state from prior suites")
    func clearGPUBeforeCharacterization() async {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        await releaseAllGPUMemory()
    }

    // MARK: - 1. Required mode: does guidance emit only the developer tool?

    /// Drives Qwen3 plus a weather tool through explicit `.required` mode. This
    /// fixture does not declare reasoning, so production disables thinking in the
    /// tool-aware template before token-zero structural guidance begins. The
    /// falsifiable assertion is that guidance emits the enabled developer tool,
    /// its arguments contain no reasoning markers, and no response text appears.
    @Test func qwen3RequiredModeEmitsOnlyDeveloperToolCall() async throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let model = makeTestModel(Self.qwen3)
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
            enabledTools: [weatherTool],
            generationOptions: GenerationOptions(
                maximumResponseTokens: 128,
                toolCallingMode: .required))
        let stream = try await executeResponse(executor, request: request, model: model)

        var responseText = ""
        var toolCallName: String? = nil
        var toolArgs = ""
        for try await event in stream {
            if case .toolCall(_, let name, let arguments) = event {
                toolCallName = name
                toolArgs += arguments
            } else if case .appendText(let chunk, _, .response) = event {
                responseText += chunk
            }
        }

        print(
            "TOOLCALL-CHAR [qwen3+tools] toolCall=\(toolCallName ?? "nil") "
                + "responseText=<<<\(responseText.prefix(200))>>> args=<<<\(toolArgs.prefix(200))>>>"
        )

        // Required guidance must emit the enabled developer tool rather than
        // reasoning markers or response text.
        let leakedInArgs = toolArgs.contains("<think>") || toolArgs.contains("</think>")
        #expect(
            !leakedInArgs,
            "Tool-call arguments must not contain reasoning markers.")
        #expect(toolCallName == "get_weather")
        #expect(responseText.isEmpty)
    }

    // MARK: - 2. Tool-aware thinking seed: does the 3-arg template honor enable_thinking?

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private func toolAwareTail(
        modelId: String, additionalContext: [String: any Sendable]?, label: String
    ) async throws -> String {
        let weatherTool = Transcript.ToolDefinition(
            name: "get_weather",
            description: "Get the current weather in a given location.",
            parameters: WeatherArgs.generationSchema
        )
        let toolSpecs = try ToolCallingConversions.makeToolSpecs(from: [weatherTool])
        let messages: [[String: any Sendable]] = [
            ["role": "user", "content": "What's the weather in Tokyo?"]
        ]
        let container = try await loadTestModelContainer(id: modelId)
        return try await container.perform { context in
            let tokens = try context.tokenizer.applyChatTemplate(
                messages: messages, tools: toolSpecs, additionalContext: additionalContext)
            let tail = context.tokenizer.decode(tokenIds: Array(tokens.suffix(48)))
            print("TOOLCALL-CHAR [\(label)] tail=<<<\(tail)>>>")
            return tail
        }
    }

    /// Confirms the tool-aware prompt mechanism: the 3-arg tool-aware template must
    /// respond to `enable_thinking`, and the tool-aware thinking-on tail's
    /// `primedInside` seed must be computed from THIS tail (not the no-tools tail).
    /// Records the per-family seed so the tool-path reasoning gate uses verified
    /// reality.
    @Test func toolAwareTemplateHonorsEnableThinking() async throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

        // Qwen3: compare the rendered tool-aware tail with thinking on vs off.
        let qOn = try await toolAwareTail(
            modelId: Self.qwen3, additionalContext: ["enable_thinking": true],
            label: "qwen3-tools-thinking-on")
        let qOff = try await toolAwareTail(
            modelId: Self.qwen3, additionalContext: ["enable_thinking": false],
            label: "qwen3-tools-thinking-off")
        #expect(
            qOn != qOff,
            "The tool-aware template must HONOR enable_thinking (distinct prompts); if equal, the tool path cannot toggle thinking via additionalContext."
        )

        let qOnPrimed = ReasoningEventEmitter.promptEndsInsideReasoning(
            renderedPromptTail: qOn, config: Self.thinkConfig)
        print(
            "TOOLCALL-CHAR [qwen3-tools-thinking-on primedInside]=\(qOnPrimed) "
                + "(expected false per the in-stream finding)")

        // R1-Distill: always-on, no knob — record the tool-aware primedInside seed.
        let r1 = try await toolAwareTail(
            modelId: Self.r1Distill, additionalContext: nil, label: "r1-distill-tools")
        let r1Primed = ReasoningEventEmitter.promptEndsInsideReasoning(
            renderedPromptTail: r1, config: Self.thinkConfig)
        print(
            "TOOLCALL-CHAR [r1-distill-tools primedInside]=\(r1Primed) (expected true if it prefills)"
        )
        #expect(!r1.isEmpty)
    }
}

#endif  // FoundationModelsIntegration
