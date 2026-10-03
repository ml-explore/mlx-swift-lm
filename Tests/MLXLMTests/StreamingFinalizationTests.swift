// Copyright © 2026 Apple Inc.

import Foundation
import Testing

@testable import MLXLMCommon

@Suite struct StreamingFinalizationTests {
    @Test(arguments: ["visible\u{fffd}", "visible<stop>hidden\u{fffd}", "visible<sto\u{fffd}"])
    func standardDecoderFiltersFinalText(_ text: String) {
        var decoder = StandardTokenStreamDecoder(
            tokenizer: FinalizationTokenizer(pieces: [text]), format: .json,
            tools: nil, stopStrings: ["<stop>"])
        _ = decoder.push(0) { _ in
            Issue.record("Text ending in a replacement character must wait for finish")
            return true
        }
        var output = ""
        var stopped = false
        _ = decoder.finish { event in
            if case .response(let chunk) = event { output += chunk }
            if case .stop = event { stopped = true }
            return true
        }
        #expect(output == (text.contains("<stop>") ? "visible" : text))
        #expect(stopped == text.contains("<stop>"))
    }

    @Test func standardDecoderFlushesBeforeToolParsingFinishes() {
        let text =
            #"<tool_call>{"name":"search","arguments":{"query":"test"}}</tool_call>"#
            + "\u{fffd}"
        var decoder = StandardTokenStreamDecoder(
            tokenizer: FinalizationTokenizer(pieces: [text]), format: .json,
            tools: nil, stopStrings: [])
        _ = decoder.push(0) { _ in true }
        var calls: [ToolCall] = []
        var output = ""
        #expect(
            decoder.finish { event in
                if case .toolCall(let call) = event { calls.append(call) }
                if case .response(let text) = event { output += text }
                return true
            })
        #expect(calls.count == 1)
        #expect(calls.first?.function.name == "search")
        #expect(output == "\u{fffd}")
    }

    @Test func finalOutputHonorsConsumerTermination() {
        var decoder = StandardTokenStreamDecoder(
            tokenizer: FinalizationTokenizer(pieces: ["visible<stop>hidden\u{fffd}"]),
            format: .json, tools: nil, stopStrings: ["<stop>"])
        _ = decoder.push(0) { _ in true }
        var count = 0
        #expect(
            !decoder.finish { _ in
                count += 1
                return false
            })
        #expect(count == 1)
    }

    @Test func reasoningCollectorRoutesFinalTextBeforeClosingItsEmitter() {
        var collector = ReasoningTokenCollector(
            config: .init(
                startDelimiter: "<think>", endDelimiter: "</think>", promptStrategy: .alwaysOn),
            primedInside: false,
            tokenizer: FinalizationTokenizer(pieces: ["<think>private</think>public\u{fffd}"]))
        #expect(collector.ingest(0).isEmpty)
        let segments = collector.finalize()
        #expect(
            segments.compactMap { if case .reasoning(let text) = $0 { text } else { nil } }.joined()
                == "private")
        #expect(
            segments.compactMap { if case .response(let text) = $0 { text } else { nil } }.joined()
                == "public\u{fffd}")
        #expect(collector.finalize().isEmpty)
    }

    @Test(arguments: [false, true])
    func harmonyFlushesClosedAndOpenFrames(_ incomplete: Bool) throws {
        let pieces = [
            "<|start|>", "<|channel|>", "<|message|>", "<|end|>", "<|call|>", "<|return|>",
            "<|constrain|>",
            "analysis", "private\u{fffd}", "final", "first\u{fffd}", "second\u{fffd}",
        ]
        let tokenizer = FinalizationTokenizer(pieces: pieces)
        var parser = try #require(HarmonyFrameParser(tokenizer: tokenizer))
        var router = HarmonyOutputRouter(tokenizer: tokenizer, allowedToolNames: nil)
        var events: [HarmonyOutputRouter.Event] = []
        let tokens = [1, 7, 2, 8, 3, 1, 9, 2, 10, 3, 1, 9, 2, 11] + (incomplete ? [] : [5])
        for token in tokens {
            for step in parser.push(token) { events += router.route(step) }
        }
        for step in parser.finish() { events += router.route(step) }
        events += router.finish()
        #expect(
            events.compactMap { if case .reasoning(let text) = $0 { text } else { nil } } == [
                "private\u{fffd}"
            ])
        #expect(
            events.compactMap { if case .response(let text) = $0 { text } else { nil } } == [
                "first\u{fffd}", "second\u{fffd}",
            ])
        #expect(router.finish().isEmpty)
    }

    @Test func onyxFlushesClosedFramesAndDiscardsInterruptedText() throws {
        let tokenizer = FinalizationTokenizer(pieces: [
            "<|start|>", "<|message|>", "<|eom|>", "<|eot|>", " to=self", "private\u{fffd}",
            "assistant to=user", "public\u{fffd}", "discard\u{fffd}",
        ])
        var decoder = try #require(
            OnyxStreamAdapter(tokenizer: tokenizer, tools: nil, stopStrings: []))
        var events: [TokenStreamEvent] = []
        for token in [4, 1, 5, 2, 0, 6, 1, 8, 0, 6, 1, 7, 3] {
            _ = decoder.push(token) {
                events.append($0)
                return true
            }
        }
        _ = decoder.finish {
            events.append($0)
            return true
        }
        #expect(
            events.compactMap { if case .reasoning(let text) = $0 { text } else { nil } } == [
                "private\u{fffd}"
            ])
        #expect(
            events.compactMap { if case .response(let text) = $0 { text } else { nil } } == [
                "public\u{fffd}"
            ])
        #expect(events.filter { if case .protocolError = $0 { true } else { false } }.count == 1)
    }

    @Test(arguments: [false, true])
    func generationFlushesBeforeCompletionInfoAndReportsFinalStop(_ stopped: Bool) async {
        let tokenizer = FinalizationTokenizer(pieces: [
            stopped ? "visible<stop>hidden\u{fffd}" : "visible\u{fffd}"
        ])
        let (stream, task) = generateTask(
            promptTokenCount: 1, modelConfiguration: .init(id: "fixture", stopStrings: ["<stop>"]),
            tokenizer: tokenizer, iterator: FinalizationTokenIterator())
        var output = ""
        var info: GenerateCompletionInfo?
        for await event in stream {
            switch event {
            case .chunk(let text):
                #expect(info == nil)
                output += text
            case .info(let value): info = value
            default: Issue.record("Unexpected generation event")
            }
        }
        await task.value
        #expect(output == (stopped ? "visible" : "visible\u{fffd}"))
        #expect(info?.stopReason == (stopped ? .stop : .length))
        #expect(info?.generationTokenCount == 1)
    }
}

private struct FinalizationTokenizer: Tokenizer {
    let pieces: [String]
    func encode(text: String, addSpecialTokens: Bool) -> [Int] { [] }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        tokenIds.map { pieces[$0] }.joined()
    }
    func convertTokenToId(_ token: String) -> Int? { pieces.firstIndex(of: token) }
    func convertIdToToken(_ id: Int) -> String? { pieces.indices.contains(id) ? pieces[id] : nil }
    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }
    func applyChatTemplate(
        messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] { [] }
}

private struct FinalizationTokenIterator: TokenIteratorProtocol {
    var tokenCount = 0
    let maxTokens: Int? = 1
    let promptPrefillTime: TimeInterval = 0
    mutating func next() -> Int? {
        guard tokenCount == 0 else { return nil }
        tokenCount += 1
        return 0
    }
}
