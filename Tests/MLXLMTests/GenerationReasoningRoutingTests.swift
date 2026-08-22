// Copyright © 2026 Apple Inc.

import Foundation
import Testing

@testable import MLXLMCommon

/// Reasoning routing in ``StandardTokenStreamDecoder``: thinking is split out of the
/// response before the tool-call parser sees it, and surfaces as its own event.
///
/// These drive the decoder directly rather than a model, so every case is exact.
@Suite("Generation reasoning routing")
struct GenerationReasoningRoutingTests {

    // MARK: - Harness

    /// What one decoder run produced, in arrival order.
    private struct Collected {
        var reasoning: [String] = []
        var response: [String] = []
        var toolCalls: [ToolCall] = []
        var stopped = false

        var reasoningText: String { reasoning.joined() }
        var responseText: String { response.joined() }
    }

    /// Feeds `fragments` through the decoder one "token" at a time, then finishes.
    private func run(
        _ fragments: [String],
        format: ToolCallFormat = .json,
        tools: [[String: any Sendable]]? = nil,
        stopStrings: Set<String> = [],
        reasoning: (config: ReasoningConfig, primedInside: Bool)? = nil
    ) -> Collected {
        let decoding = Dictionary(
            uniqueKeysWithValues: fragments.enumerated().map { ($0.offset + 1, $0.element) })
        var decoder = StandardTokenStreamDecoder(
            tokenizer: FragmentTokenizer(decoding: decoding),
            format: format,
            tools: tools,
            stopStrings: stopStrings,
            reasoning: reasoning)

        var collected = Collected()
        func consume(_ event: TokenStreamEvent) -> Bool {
            switch event {
            case .reasoning(let text): collected.reasoning.append(text)
            case .response(let text): collected.response.append(text)
            case .toolCall(let call): collected.toolCalls.append(call)
            case .rejectedToolCall, .protocolError: break
            case .stop: collected.stopped = true
            }
            return true
        }

        // Stop pushing once the decoder says to, as `generateLoopTask` does; a `where`
        // clause would keep iterating and feed tokens past a semantic stop.
        for id in 1 ... fragments.count {
            if !decoder.push(id, emit: consume) { break }
        }
        _ = decoder.finish(emit: consume)
        return collected
    }

    private var thinkTags: ReasoningConfig { .thinkTagsWithEnableThinking }

    // MARK: - Delimiter-pair families

    @Test("A <think> block is routed out of the answer")
    func thinkBlockIsRouted() {
        let result = run(
            ["<think>", "weigh ", "the options", "</think>", "The answer."],
            reasoning: (config: thinkTags, primedInside: false))

        #expect(result.reasoningText == "weigh the options")
        #expect(result.responseText == "The answer.")
        #expect(!result.responseText.contains("<think>"))
        #expect(!result.responseText.contains("</think>"))
    }

    @Test("A delimiter split across tokens is still recognized")
    func delimiterSplitAcrossTokensIsRecognized() {
        let result = run(
            ["<th", "ink>", "hidden", "</thi", "nk>", "shown"],
            reasoning: (config: thinkTags, primedInside: false))

        #expect(result.reasoningText == "hidden")
        #expect(result.responseText == "shown")
    }

    @Test("A primed prompt routes the opening thought it never re-emits")
    func primedPromptRoutesLeadingThought() {
        // DeepSeek-R1 prefills `<think>` into the prompt and generates only the close.
        let result = run(
            ["reason ", "about it", "</think>", "Answer."],
            reasoning: (config: .alwaysOnThinking, primedInside: true))

        #expect(result.reasoningText == "reason about it")
        #expect(result.responseText == "Answer.")
    }

    @Test("Thinking cut off before its close is still delivered, not stranded")
    func unterminatedThoughtIsFlushedOnFinish() {
        let result = run(
            ["<think>", "ran out of budget"],
            reasoning: (config: thinkTags, primedInside: false))

        #expect(result.reasoningText == "ran out of budget")
        #expect(result.responseText.isEmpty)
    }

    // MARK: - Gemma 4 labeled channels

    @Test("A Gemma 4 thought channel is routed and its label consumed")
    func gemma4ThoughtChannelIsRouted() {
        let result = run(
            ["<|channel>", "thought\n", "step one", "<channel|>", "Final answer."],
            format: .gemma4,
            reasoning: (config: .gemma4, primedInside: false))

        #expect(result.reasoningText == "step one")
        #expect(result.responseText == "Final answer.")
        // The label is metadata, not thinking, and must not reach either stream.
        #expect(!result.reasoningText.contains("thought"))
        #expect(!result.responseText.contains("<|channel>"))
        #expect(!result.responseText.contains("<channel|>"))
    }

    @Test("An unknown Gemma 4 channel label routes to reasoning rather than leaking")
    func gemma4UnknownLabelRoutesToReasoning() {
        let result = run(
            ["<|channel>", "scratch\n", "private", "<channel|>", "Answer."],
            format: .gemma4,
            reasoning: (config: .gemma4, primedInside: false))

        #expect(result.reasoningText == "private")
        #expect(result.responseText == "Answer.")
    }

    // MARK: - Interaction with the tool-call parser

    @Test("Tool syntax inside a thought channel does not become a tool call")
    func toolSyntaxInsideReasoningIsNotPromoted() {
        let result = run(
            [
                "<|channel>", "thought\n",
                #"maybe <|tool_call>call:search{q:<|"|>x<|"|>}<tool_call|>"#,
                "<channel|>", "I will not call it.",
            ],
            format: .gemma4,
            reasoning: (config: .gemma4, primedInside: false))

        #expect(result.toolCalls.isEmpty)
        #expect(result.responseText == "I will not call it.")
    }

    @Test("Qwen's implicit end delimiter exits reasoning and still opens a tool call")
    func implicitEndDelimiterExitsReasoningAndParses() {
        // `QwenReasoningProtocol` declares `<tool_call>` as an implicit exit: the model
        // opens a call straight out of its thinking block without closing `</think>`.
        // The delimiter is content, not framing, so it must stay in the stream and reach
        // the parser - dropping it would strand the call.
        // Primed with no generated opener: Qwen's template prefills `<think>` into the
        // prompt, so the model's first generated token is already thought content.
        //
        // `.qwen35` deliberately, not `.json`: the JSON format accepts a bare
        // `{"name":...}` object with no tags at all, so the call would still parse even
        // if `<tool_call>` had been swallowed as framing - the assertion would hold
        // vacuously. `.qwen35` requires the tag, so this fails if the delimiter is
        // consumed rather than left in the stream.
        let result = run(
            [
                "call the tool",
                #"<tool_call>{"name":"get_weather","arguments":{"city":"Paris"}}</tool_call>"#,
            ],
            format: .qwen35,
            reasoning: (config: QwenReasoningProtocol.tagged, primedInside: true))

        #expect(result.reasoningText == "call the tool")
        #expect(result.toolCalls.count == 1)
        #expect(result.toolCalls.first?.function.name == "get_weather")
        // The whole tool-call span was consumed by the parser, not leaked as answer text.
        #expect(result.responseText.isEmpty)
    }

    @Test("A tool call after the channel closes is still parsed")
    func toolCallAfterReasoningIsParsed() {
        // The shape Gemma 4's own template writes: the thought closes, then the call.
        let result = run(
            [
                "<|channel>", "thought\n", "need the weather", "<channel|>",
                #"<|tool_call>call:get_weather{city:<|"|>Paris<|"|>}<tool_call|>"#,
            ],
            format: .gemma4,
            reasoning: (config: .gemma4, primedInside: false))

        #expect(result.reasoningText == "need the weather")
        #expect(result.toolCalls.count == 1)
        #expect(result.toolCalls.first?.function.name == "get_weather")
    }

    // MARK: - Models with no reasoning protocol

    @Test("With no reasoning config the stream is unchanged")
    func noReasoningConfigLeavesStreamUnchanged() {
        let fragments = ["<think>", "not treated as thinking", "</think>", " tail"]
        let routed = run(fragments, reasoning: nil)

        #expect(routed.reasoning.isEmpty)
        #expect(routed.responseText == fragments.joined())
    }

    @Test("A tool call is parsed identically with and without reasoning routing")
    func toolCallParsingIsUnaffected() {
        let call = #"<tool_call>{"name":"get_weather","arguments":{"city":"Paris"}}</tool_call>"#

        let plain = run([call], reasoning: nil)
        let routed = run([call], reasoning: (config: thinkTags, primedInside: false))

        #expect(plain.toolCalls.count == 1)
        #expect(routed.toolCalls.count == 1)
        #expect(plain.toolCalls.first?.function.name == routed.toolCalls.first?.function.name)
    }

    // MARK: - Stop strings

    @Test("Stop strings still cut the stream when reasoning routing is on")
    func stopStringsStillApply() {
        let result = run(
            ["<think>", "a", "</think>", "visible", "<stop>", "hidden"],
            stopStrings: ["<stop>"],
            reasoning: (config: thinkTags, primedInside: false))

        #expect(result.stopped)
        #expect(result.responseText == "visible")
        #expect(!result.responseText.contains("hidden"))
    }

    // MARK: - isInsideReasoning

    @Test("isInsideReasoning tracks the span, for token attribution")
    func isInsideReasoningTracksTheSpan() {
        var decoder = StandardTokenStreamDecoder(
            tokenizer: FragmentTokenizer(decoding: [1: "<think>", 2: "x", 3: "</think>", 4: "y"]),
            format: .json,
            tools: nil,
            stopStrings: [],
            reasoning: (config: thinkTags, primedInside: false))

        #expect(!decoder.isInsideReasoning)
        _ = decoder.push(1) { _ in true }
        _ = decoder.push(2) { _ in true }
        #expect(decoder.isInsideReasoning)
        _ = decoder.push(3) { _ in true }
        _ = decoder.push(4) { _ in true }
        #expect(!decoder.isInsideReasoning)
    }

    @Test("A decoder with no reasoning config never reports being inside reasoning")
    func isInsideReasoningIsFalseWithoutConfig() {
        var decoder = StandardTokenStreamDecoder(
            tokenizer: FragmentTokenizer(decoding: [1: "<think>"]),
            format: .json,
            tools: nil,
            stopStrings: [])

        _ = decoder.push(1) { _ in true }
        #expect(!decoder.isInsideReasoning)
    }
}

/// Decodes each token id to a fixed fragment, so a test spells the model's output
/// directly and controls exactly where the token boundaries fall.
private struct FragmentTokenizer: Tokenizer {
    let decoding: [Int: String]

    func encode(text: String, addSpecialTokens: Bool) -> [Int] { [] }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        tokenIds.map { decoding[$0] ?? "" }.joined()
    }

    func convertTokenToId(_ token: String) -> Int? { nil }
    func convertIdToToken(_ id: Int) -> String? { decoding[id] }

    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] { [] }
}
