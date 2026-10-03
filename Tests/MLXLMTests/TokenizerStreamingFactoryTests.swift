// Copyright © 2026 Apple Inc.

import Foundation
import Testing

@testable import MLXLMCommon

@Suite struct TokenizerStreamingFactoryTests {
    @Test func factoryDispatchesThroughExistentialAndCreatesIndependentStreams() {
        let tokenizer: any Tokenizer = BufferedStreamTokenizer(pieces: ["é", "🙂", "<special>"])
        var first = tokenizer.makeStreamingDetokenizer()
        var second = tokenizer.makeStreamingDetokenizer()
        first.append(token: 0)
        first.append(token: 1)
        second.append(token: 2)
        #expect(first.next() == nil)
        #expect(second.next() == nil)
        #expect(first.finish() == "é🙂")
        #expect(second.finish() == "<special>")
        #expect(first.finish() == nil)
    }

    @Test func standardDecoderFlushesBeforeToolParsingFinishes() {
        let tokenizer = BufferedStreamTokenizer(pieces: [
            #"<tool_call>{"name":"get_weather","arguments":{"city":"Paris"}}</tool_call>"#
        ])
        var decoder = StandardTokenStreamDecoder(
            tokenizer: tokenizer, format: .json, tools: nil, stopStrings: [])
        #expect(
            decoder.push(0) { _ in
                Issue.record("Output must wait for finish")
                return true
            })
        var calls: [ToolCall] = []
        #expect(
            decoder.finish { event in
                if case .toolCall(let call) = event { calls.append(call) }
                return true
            })
        #expect(calls.count == 1)
        #expect(calls.first?.function.name == "get_weather")
        #expect(calls.first?.function.arguments["city"] == .string("Paris"))
    }

    @Test(arguments: ["visible<stop>hidden", "visible<sto"])
    func standardDecoderFiltersFinalText(_ text: String) {
        var decoder = StandardTokenStreamDecoder(
            tokenizer: BufferedStreamTokenizer(pieces: [text]), format: .json, tools: nil,
            stopStrings: ["<stop>"])
        _ = decoder.push(0) { _ in true }
        var output = ""
        var stopped = false
        _ = decoder.finish { event in
            if case .response(let chunk) = event { output += chunk }
            if case .stop = event { stopped = true }
            return true
        }
        #expect(output == (text.hasSuffix("hidden") ? "visible" : text))
        #expect(stopped == text.hasSuffix("hidden"))
    }

    @Test func finalOutputHonorsConsumerTermination() {
        var decoder = StandardTokenStreamDecoder(
            tokenizer: BufferedStreamTokenizer(pieces: ["visible<stop>hidden"]),
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

    @Test func reasoningCollectorRoutesFinalTextBeforeFinalizingDelimiterState() {
        var collector = ReasoningTokenCollector(
            config: .init(
                startDelimiter: "<think>", endDelimiter: "</think>", promptStrategy: .alwaysOn),
            primedInside: false,
            tokenizer: BufferedStreamTokenizer(pieces: ["<think>private</think>public"]))
        #expect(collector.ingest(0).isEmpty)
        let segments = collector.finalize()
        #expect(
            segments.compactMap { if case .reasoning(let text) = $0 { text } else { nil } }.joined()
                == "private")
        #expect(
            segments.compactMap { if case .response(let text) = $0 { text } else { nil } }.joined()
                == "public")
        #expect(collector.reasoningTokenIDs == [0])
        #expect(collector.finalize().isEmpty)
    }

    @Test(arguments: [false, true])
    func harmonyFlushesAndResetsEachFrame(_ incomplete: Bool) throws {
        let pieces = [
            "<|start|>", "<|channel|>", "<|message|>", "<|end|>", "<|call|>", "<|return|>",
            "<|constrain|>",
            "analysis", "private", "final", "first", "second",
        ]
        let tokenizer = BufferedStreamTokenizer(pieces: pieces)
        var parser = try #require(HarmonyFrameParser(tokenizer: tokenizer))
        var router = HarmonyOutputRouter(tokenizer: tokenizer, allowedToolNames: nil)
        var events: [HarmonyOutputRouter.Event] = []
        let input = [1, 7, 2, 8, 3, 1, 9, 2, 10, 3, 1, 9, 2, 11] + (incomplete ? [] : [5])
        for token in input {
            for step in parser.push(token) { events += router.route(step) }
        }
        for step in parser.finish() { events += router.route(step) }
        events += router.finish()
        #expect(
            events.compactMap { if case .reasoning(let text) = $0 { text } else { nil } } == [
                "private"
            ])
        #expect(
            events.compactMap { if case .response(let text) = $0 { text } else { nil } } == [
                "first", "second",
            ])
        #expect(router.finish().isEmpty)
    }

    @Test(arguments: [false, true])
    func generationFlushesBeforeCompletionInfoAndReportsFinalStop(_ stopped: Bool) async {
        let tokenizer = BufferedStreamTokenizer(pieces: [
            stopped ? "visible<stop>hidden" : "visible"
        ])
        let (stream, task) = generateTask(
            promptTokenCount: 1,
            modelConfiguration: .init(id: "fixture", stopStrings: ["<stop>"]),
            tokenizer: tokenizer,
            iterator: SingleTokenIterator())
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
        #expect(output == "visible")
        #expect(info?.stopReason == (stopped ? .stop : .length))
        #expect(info?.generationTokenCount == 1)
    }

    @Test func onyxFlushesClosedFramesAndDiscardsInterruptedState() throws {
        let tokenizer = BufferedStreamTokenizer(pieces: [
            "<|start|>", "<|message|>", "<|eom|>", "<|eot|>", " to=self", "private",
            "assistant to=user", "public", "discard",
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
                "private"
            ])
        #expect(
            events.compactMap { if case .response(let text) = $0 { text } else { nil } } == [
                "public"
            ])
        #expect(events.filter { if case .protocolError = $0 { true } else { false } }.count == 1)
    }
}

private struct BufferedStreamTokenizer: Tokenizer {
    let pieces: [String]
    func makeStreamingDetokenizer() -> any StreamingDetokenizer {
        BufferedDetokenizer(pieces: pieces)
    }
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

private struct BufferedDetokenizer: StreamingDetokenizer {
    let pieces: [String]
    var pending = ""
    mutating func append(token: Int) { pending += pieces[token] }
    mutating func next() -> String? { nil }
    mutating func finish() -> String? {
        defer { pending = "" }
        return pending.isEmpty ? nil : pending
    }
}

private struct SingleTokenIterator: TokenIteratorProtocol {
    var tokenCount = 0
    let maxTokens: Int? = 1
    let promptPrefillTime: TimeInterval = 0
    mutating func next() -> Int? {
        guard tokenCount == 0 else { return nil }
        tokenCount += 1
        return 0
    }
}
