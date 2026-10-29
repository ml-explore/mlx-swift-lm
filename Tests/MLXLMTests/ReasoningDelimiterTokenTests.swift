// Copyright © 2026 Apple Inc.

import Foundation
import Testing

@testable import MLXLMCommon

/// Reasoning delimiters matched by token id.
///
/// When both delimiters are single tokens, only those ids open and close a span.
/// The same characters arriving as ordinary text tokens are content.
@Suite("Reasoning delimiter tokens")
struct ReasoningDelimiterTokenTests {

    // MARK: - Harness

    /// Maps ids to text. Ids below 10 are control tokens: only they resolve from
    /// their text, as a real tokenizer resolves only its control tokens whole.
    private struct VocabularyTokenizer: Tokenizer {
        let vocabulary: [Int: String]

        func encode(text: String, addSpecialTokens: Bool) -> [Int] { [] }
        func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
            tokenIds.map { vocabulary[$0] ?? "" }.joined()
        }
        func convertTokenToId(_ token: String) -> Int? {
            vocabulary.filter { $0.key < 10 && $0.value == token }.keys.min()
        }
        func convertIdToToken(_ id: Int) -> String? { vocabulary[id] }
        var bosToken: String? { nil }
        var eosToken: String? { nil }
        var unknownToken: String? { nil }
        func applyChatTemplate(
            messages: [[String: any Sendable]],
            tools: [[String: any Sendable]]?,
            additionalContext: [String: any Sendable]?
        ) throws -> [Int] { [] }
    }

    private struct Collected {
        var reasoning: [String] = []
        var response: [String] = []
        var toolCalls: [ToolCall] = []
        var stopped = false

        var reasoningText: String { reasoning.joined() }
        var responseText: String { response.joined() }
    }

    /// Ids 1 and 2 are the config's delimiters as control tokens, and id 3 is
    /// `<tool_call>` as one when `toolCallToken` is set. Every other fragment gets
    /// its own id, even when it spells a delimiter.
    private func run(
        _ fragments: [Fragment],
        config: ReasoningConfig,
        format: ToolCallFormat = .json,
        stopStrings: Set<String> = [],
        primedInside: Bool = false,
        toolCallToken: Bool = false
    ) -> Collected {
        var vocabulary = [1: config.startDelimiter, 2: config.endDelimiter]
        if toolCallToken { vocabulary[3] = "<tool_call>" }
        var ids: [Int] = []
        for (offset, fragment) in fragments.enumerated() {
            switch fragment {
            case .open: ids.append(1)
            case .close: ids.append(2)
            case .toolCall: ids.append(3)
            case .text(let text):
                vocabulary[offset + 10] = text
                ids.append(offset + 10)
            }
        }
        var decoder = StandardTokenStreamDecoder(
            tokenizer: VocabularyTokenizer(vocabulary: vocabulary),
            format: format,
            tools: nil,
            stopStrings: stopStrings,
            reasoning: (config: config, primedInside: primedInside))

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
        for id in ids {
            if !decoder.push(id, emit: consume) { break }
        }
        _ = decoder.finish(emit: consume)
        return collected
    }

    private enum Fragment: ExpressibleByStringLiteral {
        case open
        case close
        case toolCall
        case text(String)

        init(stringLiteral value: String) { self = .text(value) }
    }

    private var qwen: ReasoningConfig { QwenReasoningProtocol.tagged }

    // MARK: - Resolution

    @Test("Delimiters resolve only when each is one token that decodes to itself")
    func resolution() {
        let whole = VocabularyTokenizer(vocabulary: [1: "<think>", 2: "</think>"])
        let tokens = ReasoningDelimiterTokens(config: qwen, tokenizer: whole)
        #expect(tokens?.start == 1)
        #expect(tokens?.end == 2)
        #expect(tokens?.boundary(for: 1) == .start)
        #expect(tokens?.boundary(for: 2) == .end)
        #expect(tokens?.boundary(for: 3) == nil)
        #expect(tokens?.implicitEndDelimiters == [])

        // An implicit end delimiter resolves on its own, when it is a token.
        let withCall = VocabularyTokenizer(
            vocabulary: [1: "<think>", 2: "</think>", 3: "<tool_call>"])
        let callTokens = ReasoningDelimiterTokens(config: qwen, tokenizer: withCall)
        #expect(callTokens?.boundary(for: 3) == .implicitEnd("<tool_call>"))
        #expect(callTokens?.implicitEndDelimiters == ["<tool_call>"])

        // Only one of the two is a token: fall back to text for both.
        let half = VocabularyTokenizer(vocabulary: [1: "<think>", 2: "</thi", 3: "nk>"])
        #expect(ReasoningDelimiterTokens(config: qwen, tokenizer: half) == nil)

        // A tokenizer that answers with an id for anything is not trusted.
        #expect(ReasoningDelimiterTokens(config: qwen, tokenizer: AnyIDTokenizer()) == nil)
    }

    private struct AnyIDTokenizer: Tokenizer {
        func encode(text: String, addSpecialTokens: Bool) -> [Int] { [7] }
        func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { "word" }
        func convertTokenToId(_ token: String) -> Int? { 7 }
        func convertIdToToken(_ id: Int) -> String? { "word" }
        var bosToken: String? { nil }
        var eosToken: String? { nil }
        var unknownToken: String? { nil }
        func applyChatTemplate(
            messages: [[String: any Sendable]],
            tools: [[String: any Sendable]]?,
            additionalContext: [String: any Sendable]?
        ) throws -> [Int] { [] }
    }

    // MARK: - Delimiter pairs

    @Test("Delimiter tokens route a thought out of the answer")
    func tokensRouteReasoning() {
        let result = run([.open, "\nweigh ", "it", .close, "\n\nThe answer."], config: qwen)
        #expect(result.reasoningText == "weigh it")
        #expect(result.responseText == "The answer.")
    }

    @Test("A <think> spelled out in the answer stays in the answer")
    func spelledOpenerInAnswerIsText() {
        let result = run(
            ["Wrap it in ", "<th", "ink>", " and ", "</think>", " tags."], config: qwen)
        #expect(result.reasoning.isEmpty)
        #expect(result.responseText == "Wrap it in <think> and </think> tags.")
    }

    @Test("A </think> spelled out in the middle of a thought does not close it")
    func spelledCloserInThoughtIsText() {
        let result = run(
            [.open, "the tag ", "</think>", " ends it", .close, "Answer."], config: qwen)
        #expect(result.reasoningText == "the tag </think> ends it")
        #expect(result.responseText == "Answer.")
    }

    @Test("A <think> spelled out in the middle of a thought stays in the thought")
    func spelledOpenerInThoughtIsText() {
        let result = run([.open, "why ", "<think>", " here", .close, "Answer."], config: qwen)
        #expect(result.reasoningText == "why <think> here")
        #expect(result.responseText == "Answer.")
    }

    @Test("An opener token inside an open thought that never closes stays reasoning")
    func nestedOpenerTokenNeverClosed() {
        let result = run([.open, "first ", .open, "second"], config: qwen)
        #expect(result.reasoningText == "first <think>second")
        #expect(result.response.isEmpty)
    }

    @Test("A close token with no open thought is shown, as the text scan shows it")
    func strayCloseTokenIsText() {
        let result = run(["Answer ", .close, " more"], config: qwen)
        #expect(result.reasoning.isEmpty)
        #expect(result.responseText == "Answer </think> more")
    }

    @Test("A primed thought closes on the close token")
    func primedThoughtCloses() {
        let result = run(["reason", .close, "Answer."], config: qwen, primedInside: true)
        #expect(result.reasoningText == "reason")
        #expect(result.responseText == "Answer.")
    }

    @Test("A primed channel closes on the close token")
    func primedChannelCloses() {
        let result = run(
            ["carried", .close, "Answer."], config: .gemma4, format: .gemma4, primedInside: true)
        #expect(result.reasoningText == "carried")
        #expect(result.responseText == "Answer.")
    }

    @Test("An opener token in the middle of the answer opens a thought")
    func openerTokenMidAnswerOpensAThought() {
        let result = run(["Answer ", .open, "more", .close, " done"], config: qwen)
        #expect(result.reasoningText == "more")
        #expect(result.responseText == "Answer done")
    }

    @Test("An implicit end delimiter that is not a token is still found as text")
    func implicitEndStillScanned() {
        let result = run(
            [
                .open, "need the file", "<tool_call>",
                "{\"name\": \"read\", \"arguments\": {}}", "</tool_call>",
            ], config: qwen)
        #expect(result.reasoningText == "need the file")
        #expect(result.toolCalls.map(\.function.name) == ["read"])
    }

    private let call = "{\"name\": \"read\", \"arguments\": {}}"

    @Test("A <tool_call> token ends an open thought and starts the call")
    func toolCallTokenEndsThought() {
        let result = run(
            [.open, "need the file", .toolCall, .text(call), "</tool_call>"],
            config: qwen, toolCallToken: true)
        #expect(result.reasoningText == "need the file")
        #expect(result.response.isEmpty)
        #expect(result.toolCalls.map(\.function.name) == ["read"])
    }

    @Test("A <tool_call> spelled out in a thought does not end it when it is a token")
    func spelledToolCallInThoughtIsText() {
        let result = run(
            [.open, "the ", "<tool_", "call>", " tag starts a call", .close, "Answer."],
            config: qwen, toolCallToken: true)
        #expect(result.reasoningText == "the <tool_call> tag starts a call")
        #expect(result.responseText == "Answer.")
        #expect(result.toolCalls.isEmpty)
    }

    @Test("A spelled <tool_call> in a thought, then the token, ends the thought at the token")
    func spelledThenTokenToolCall() {
        let result = run(
            [.open, "after ", "<tool_call>", " I call", .toolCall, .text(call), "</tool_call>"],
            config: qwen, toolCallToken: true)
        #expect(result.reasoningText == "after <tool_call> I call")
        #expect(result.toolCalls.map(\.function.name) == ["read"])
    }

    @Test("A <tool_call> token ends a primed thought")
    func toolCallTokenEndsPrimedThought() {
        let result = run(
            ["need the file", .toolCall, .text(call), "</tool_call>"],
            config: qwen, primedInside: true, toolCallToken: true)
        #expect(result.reasoningText == "need the file")
        #expect(result.toolCalls.map(\.function.name) == ["read"])
    }

    @Test("A <tool_call> token after a closed thought is an ordinary call")
    func toolCallTokenAfterThought() {
        let result = run(
            [.open, "plan", .close, "\n\n", .toolCall, .text(call), "</tool_call>"],
            config: qwen, toolCallToken: true)
        #expect(result.reasoningText == "plan")
        #expect(result.response.isEmpty)
        #expect(result.toolCalls.map(\.function.name) == ["read"])
    }

    @Test("A <tool_call> token with no thought at all is an ordinary call")
    func toolCallTokenWithoutThought() {
        let result = run(
            ["Reading. ", .toolCall, .text(call), "</tool_call>"],
            config: qwen, toolCallToken: true)
        #expect(result.reasoning.isEmpty)
        #expect(result.responseText == "Reading. ")
        #expect(result.toolCalls.map(\.function.name) == ["read"])
    }

    @Test("A stop string equal to <tool_call> stops on its token inside a thought")
    func stopStringOnToolCallToken() {
        let result = run(
            [.open, "thinking", .toolCall, .text(call), "</tool_call>"],
            config: qwen, stopStrings: ["<tool_call>"], toolCallToken: true)
        #expect(result.stopped)
        #expect(result.reasoningText == "thinking")
        #expect(result.toolCalls.isEmpty)
    }

    @Test("A <tool_call> token ends a channel whose label never ended")
    func toolCallTokenDuringLabel() {
        var config = ReasoningConfig.gemma4
        config.implicitEndDelimiters = ["<tool_call>"]
        let result = run(
            [.open, "tho", .toolCall, .text(call), "</tool_call>"],
            config: config, toolCallToken: true)
        #expect(result.reasoningText == "tho")
        #expect(result.toolCalls.map(\.function.name) == ["read"])
    }

    @Test("Text held for a scanned implicit end precedes a <tool_call> token")
    func heldTextPrecedesToolCallToken() {
        var config = qwen
        config.implicitEndDelimiters.append("<function=")
        let result = run(
            [.open, "about ", "<func", .toolCall, .text(call), "</tool_call>"],
            config: config, toolCallToken: true)
        #expect(result.reasoningText == "about <func")
        #expect(result.toolCalls.map(\.function.name) == ["read"])
    }

    @Test("Text held before a boundary token is delivered ahead of it")
    func heldTextPrecedesBoundary() {
        // `<tool_c` is held as a possible implicit end until the close token arrives.
        let result = run([.open, "about ", "<tool_c", .close, "Answer."], config: qwen)
        #expect(result.reasoningText == "about <tool_c")
        #expect(result.responseText == "Answer.")
    }

    @Test("A stop string equal to a delimiter still stops on its token")
    func stopStringOnDelimiterToken() {
        let result = run(
            [.open, "thinking", .close, "never shown"], config: qwen, stopStrings: ["</think>"])
        #expect(result.stopped)
        #expect(result.reasoningText == "thinking")
        #expect(result.response.isEmpty)
    }

    // MARK: - Labeled channels

    @Test("Channel tokens route a thought channel and consume its label")
    func channelTokensRoute() {
        let result = run(
            [.open, "thought\n", "step one", .close, "Final answer."],
            config: .gemma4, format: .gemma4)
        #expect(result.reasoningText == "step one")
        #expect(result.responseText == "Final answer.")
    }

    @Test("Channel markers spelled out in the answer stay in the answer")
    func spelledChannelMarkersInAnswerAreText() {
        let result = run(
            [
                .open, "thought\n", "plan", .close,
                "A channel opens with ", "<|channel>", "thought", " and ends with ", "<channel|>",
                ".",
            ],
            config: .gemma4, format: .gemma4)
        #expect(result.reasoningText == "plan")
        #expect(
            result.responseText
                == "A channel opens with <|channel>thought and ends with <channel|>.")
    }

    @Test("Channel markers spelled out in a thought do not close or reopen it")
    func spelledChannelMarkersInThoughtAreText() {
        let result = run(
            [
                .open, "thought\n", "the ", "<channel|>", " then ", "<|channel>", " marker", .close,
                "Done.",
            ],
            config: .gemma4, format: .gemma4)
        #expect(result.reasoningText == "the <channel|> then <|channel> marker")
        #expect(result.responseText == "Done.")
    }

    @Test("An opener token inside an open channel starts the next channel")
    func nestedChannelTokenStartsNext() {
        let result = run(
            [.open, "thought\n", "a", .open, "thought\n", "b", .close, "c"],
            config: .gemma4, format: .gemma4)
        #expect(result.reasoningText == "ab")
        #expect(result.responseText == "c")
    }

    @Test("A nested channel token that never closes flushes as reasoning")
    func nestedChannelTokenNeverClosed() {
        let result = run(
            [.open, "thought\n", "a", .open, "thought\n", "b"],
            config: .gemma4, format: .gemma4)
        #expect(result.reasoningText == "ab")
        #expect(result.response.isEmpty)
    }

    @Test("An opener token right after an opener token abandons the first label")
    func openerTokenDuringLabel() {
        let result = run(
            [.open, "tho", .open, "thought\n", "b", .close, "c"],
            config: .gemma4, format: .gemma4)
        #expect(result.reasoningText == "thob")
        #expect(result.responseText == "c")
    }

    @Test("A channel closed before its label ends is consumed whole")
    func emptyChannelTokens() {
        let result = run([.open, "thought", .close, "Answer."], config: .gemma4, format: .gemma4)
        #expect(result.reasoning.isEmpty)
        #expect(result.responseText == "Answer.")
    }

    @Test("A stray close token outside any channel is swallowed")
    func strayChannelCloseToken() {
        let result = run(["Answer ", .close, "more"], config: .gemma4, format: .gemma4)
        #expect(result.reasoning.isEmpty)
        #expect(result.responseText == "Answer more")
    }

    @Test("A response channel routes to the answer")
    func responseChannelTokens() {
        let result = run(
            [.open, "thought\n", "plan", .close, .open, "content\n", "shown", .close],
            config: .gemma4, format: .gemma4)
        #expect(result.reasoningText == "plan")
        #expect(result.responseText == "shown")
    }

    // MARK: - Token collector

    @Test("The collector stops on the close token, not on its spelling")
    func collectorIgnoresSpelledClose() {
        let tokenizer = VocabularyTokenizer(vocabulary: [
            1: "<think>", 2: "</think>", 3: "a ", 4: "</thi", 5: "nk>", 6: " b",
        ])
        var collector = ReasoningTokenCollector(
            config: qwen, primedInside: false, tokenizer: tokenizer)
        var segments: [ReasoningEventEmitter.Segment] = []
        for token in [1, 3, 4, 5, 6] {
            segments += collector.ingest(token)
            #expect(!collector.shouldStopAfterReasoning)
        }
        segments += collector.ingest(2)
        #expect(collector.shouldStopAfterReasoning)
        segments += collector.finalize()

        let reasoning = segments.compactMap { segment -> String? in
            if case .reasoning(let text) = segment { return text }
            return nil
        }
        #expect(reasoning.joined() == "a </think> b")
        #expect(collector.reasoningTokenIDs == [1, 3, 4, 5, 6, 2])
    }

    @Test("The collector does not stop on an opener token inside an open thought")
    func collectorNestedOpenerTokenDoesNotStop() {
        let tokenizer = VocabularyTokenizer(vocabulary: [
            1: "<think>", 2: "</think>", 3: "first ", 4: "second",
        ])
        var collector = ReasoningTokenCollector(
            config: qwen, primedInside: false, tokenizer: tokenizer)
        var segments: [ReasoningEventEmitter.Segment] = []
        for token in [1, 3, 1, 4] {
            segments += collector.ingest(token)
            #expect(!collector.shouldStopAfterReasoning)
        }
        segments += collector.finalize()

        let reasoning = segments.compactMap { segment -> String? in
            if case .reasoning(let text) = segment { return text }
            return nil
        }
        #expect(reasoning.joined() == "first <think>second")
        #expect(collector.reasoningTokenIDs == [1, 3, 1, 4])
    }

    @Test("The collector stops on the <tool_call> token, not on its spelling")
    func collectorStopsOnToolCallToken() {
        let tokenizer = VocabularyTokenizer(vocabulary: [
            1: "<think>", 2: "</think>", 3: "<tool_call>", 10: "a ", 11: "<tool_", 12: "call>",
            13: " b",
        ])
        var collector = ReasoningTokenCollector(
            config: qwen, primedInside: false, tokenizer: tokenizer)
        var segments: [ReasoningEventEmitter.Segment] = []
        for token in [1, 10, 11, 12, 13] {
            segments += collector.ingest(token)
            #expect(!collector.shouldStopAfterReasoning)
        }
        segments += collector.ingest(3)
        #expect(collector.shouldStopAfterReasoning)
        segments += collector.finalize()

        #expect(reasoningText(segments) == "a <tool_call> b")
        // The tag stays in the stream for the tool-call parser.
        #expect(segments.last == .response("<tool_call>"))
        #expect(collector.reasoningTokenIDs == [1, 10, 11, 12, 13, 3])
    }

    private func reasoningText(_ segments: [ReasoningEventEmitter.Segment]) -> String {
        segments.compactMap { segment -> String? in
            if case .reasoning(let text) = segment { return text }
            return nil
        }.joined()
    }

    // MARK: - Prompt priming by token

    /// The template's prefill and a `<think>` the user typed share one token id.
    /// Only position tells them apart: a prefill is the last thing in the prompt.
    @Test("An opener token primes only when it ends the prompt")
    func openerTokenPrimesOnlyAtTheEnd() {
        let tokenizer = VocabularyTokenizer(vocabulary: [
            1: "<think>", 2: "</think>", 10: "user\n", 11: "What does ", 12: " mean?",
            13: "assistant\n", 14: "\n",
        ])
        func primed(_ tokens: [Int]) -> Bool {
            ReasoningEventEmitter.promptEndsInsideReasoning(
                promptTokens: tokens, config: qwen, tokenizer: tokenizer)
        }
        #expect(!primed([10, 11, 1, 12, 13]))
        #expect(primed([10, 11, 1, 12, 13, 1, 14]))
        #expect(!primed([10, 11, 1, 12, 13, 1, 14, 2, 14]))
        #expect(!primed([]))
    }

    @Test("A prefill buried under more whitespace tokens than the tail holds still primes")
    func prefillBeyondTheDecodedTailPrimes() {
        let tokenizer = VocabularyTokenizer(vocabulary: [
            1: "<think>", 2: "</think>", 10: "word ", 14: "\n",
        ])
        let padding = Array(
            repeating: 14, count: ReasoningEventEmitter.promptTailTokenCount + 10)
        #expect(
            ReasoningEventEmitter.promptEndsInsideReasoning(
                promptTokens: [10, 1] + padding, config: qwen, tokenizer: tokenizer))
        #expect(
            !ReasoningEventEmitter.promptEndsInsideReasoning(
                promptTokens: [10, 1, 10] + padding, config: qwen, tokenizer: tokenizer))
    }

    @Test("Channel tokens prime only when an open channel ends the prompt")
    func channelTokensPrime() {
        let tokenizer = VocabularyTokenizer(vocabulary: [
            1: "<|channel>", 2: "<channel|>", 10: "model\n", 11: "thought\n", 12: "user\n",
            13: "What does ", 14: " mean?",
        ])
        func primed(_ tokens: [Int]) -> Bool {
            ReasoningEventEmitter.promptEndsInsideReasoning(
                promptTokens: tokens, config: .gemma4, tokenizer: tokenizer)
        }
        #expect(primed([10, 1, 11]))
        #expect(!primed([10, 1, 11, 2]))
        #expect(!primed([12, 13, 1, 14, 10]))
    }
}
