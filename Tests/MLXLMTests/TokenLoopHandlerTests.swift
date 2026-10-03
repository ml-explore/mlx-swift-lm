// Copyright © 2026 Apple Inc.

import MLXLMCommon
import XCTest

final class TokenLoopHandlerTests: XCTestCase {
    private struct Iterator: TokenIteratorProtocol {
        let tokens: [Int]
        var reportsLogProbabilities = true
        var tokenCount = 0
        var maxTokens: Int? { tokens.count }
        var promptPrefillTime: Double { 0 }

        var lastLogProbabilities: GenerateTokenLogProbabilities? {
            guard reportsLogProbabilities, tokenCount > 0 else { return nil }
            return .init(chosen: .init(token: tokens[tokenCount - 1], logProbability: -0.5))
        }

        mutating func next() -> Int? {
            guard tokenCount < tokens.count else { return nil }
            defer { tokenCount += 1 }
            return tokens[tokenCount]
        }
    }

    private struct ByteTokenizer: Tokenizer {
        var bosToken: String? { nil }
        var eosToken: String? { nil }
        var unknownToken: String? { nil }

        func encode(text: String, addSpecialTokens: Bool) -> [Int] {
            text.utf8.map(Int.init)
        }

        func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
            String(decoding: tokenIds.compactMap(UInt8.init(exactly:)), as: UTF8.self)
        }

        func convertTokenToId(_ token: String) -> Int? {
            token.utf8.count == 1 ? token.utf8.first.map(Int.init) : nil
        }

        func convertIdToToken(_ id: Int) -> String? {
            UInt8(exactly: id).map { String(decoding: [$0], as: UTF8.self) }
        }

        func applyChatTemplate(
            messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
            additionalContext: [String: any Sendable]?
        ) throws -> [Int] {
            throw TokenizerError.missingChatTemplate
        }
    }

    private struct BoundaryHandler: TokenLoopHandler {
        var additionalStopTokenIDs: Set<Int> { [33] }
        var receivesStopTokens: Bool { true }

        mutating func onToken(
            _ token: Int, logProbabilities: GenerateTokenLogProbabilities?,
            emit: (sending String) -> Bool
        ) -> TokenLoopDisposition {
            emit("token:\(token)") ? .more : .cancelled
        }

        mutating func onStopToken(
            _ token: Int, logProbabilities: GenerateTokenLogProbabilities?,
            emit: (sending String) -> Bool
        ) -> TokenLoopDisposition {
            emit("stop:\(token)") ? .stop : .cancelled
        }

        mutating func onGenerationEnd(emit: (sending String) -> Bool) {
            _ = emit("end")
        }

        func infoEvent(_ info: GenerateCompletionInfo) -> String {
            "count:\(info.generationTokenCount)"
        }
    }

    func testRawCompositionKeepsTokenAndProbabilityEventsAligned() async {
        for reportsLogProbabilities in [false, true] {
            let (stream, task) = generateLoopTask(
                promptTokenCount: 1, modelConfiguration: .init(id: "test"),
                tokenizer: ByteTokenizer(),
                iterator: Iterator(
                    tokens: [65, 66, 67], reportsLogProbabilities: reportsLogProbabilities),
                handler: LogProbabilityTokenLoopHandler(RawTokenLoopHandler()))

            var events: [String] = []
            for await event in stream {
                switch event {
                case .probability(let value):
                    events.append("probability:\(value.chosen.token)")
                case .generation(.token(let token)):
                    events.append("token:\(token)")
                case .generation(.info(let info)):
                    XCTAssertEqual(info.generationTokenCount, 3)
                    XCTAssertEqual(info.stopReason, .length)
                    events.append("info")
                }
            }
            await task.value

            let expected = [65, 66, 67].flatMap { token in
                reportsLogProbabilities
                    ? ["probability:\(token)", "token:\(token)"] : ["token:\(token)"]
            }
            XCTAssertEqual(events, expected + ["info"])
        }
    }

    func testCompositionForwardsStopPolicyAndFinalEvents() async {
        for includeStopToken in [false, true] {
            let (stream, task) = generateLoopTask(
                promptTokenCount: 1, modelConfiguration: .init(id: "test"),
                tokenizer: ByteTokenizer(), iterator: Iterator(tokens: [65, 33, 66]),
                includeStopToken: includeStopToken,
                handler: LogProbabilityTokenLoopHandler(BoundaryHandler()))

            var events: [String] = []
            for await event in stream {
                switch event {
                case .probability(let value): events.append("probability:\(value.chosen.token)")
                case .generation(let value): events.append(value)
                }
            }
            await task.value

            XCTAssertEqual(
                events,
                ["probability:65", "token:65"]
                    + (includeStopToken ? ["probability:33"] : [])
                    + ["stop:33", "end", "count:\(includeStopToken ? 2 : 1)"])
        }
    }

    func testCompositionPreservesTextAndToolCalls() async {
        let text =
            #"start<tool_call>{"name":"weather","arguments":{"city":"Paris"}}</tool_call>end"#
        let tokenizer = ByteTokenizer()
        let tokens = tokenizer.encode(text: text)
        let (stream, task) = generateLoopTask(
            promptTokenCount: 1, modelConfiguration: .init(id: "test"), tokenizer: tokenizer,
            iterator: Iterator(tokens: tokens),
            handler: LogProbabilityTokenLoopHandler(
                TextToolTokenLoopHandler(tokenizer: tokenizer, format: .json)))

        var response = ""
        var calls: [ToolCall.Function] = []
        var observedTokens: [Int] = []
        var completion: GenerateCompletionInfo?
        for await event in stream {
            switch event {
            case .probability(let value): observedTokens.append(value.chosen.token)
            case .generation(let generation):
                switch generation {
                case .chunk(let text): response += text
                case .toolCall(let call): calls.append(call.function)
                case .rejectedToolCall: XCTFail("valid tool call was rejected")
                case .info(let info): completion = info
                }
            }
        }
        await task.value

        XCTAssertEqual(response, "startend")
        XCTAssertEqual(calls, [.init(name: "weather", arguments: ["city": .string("Paris")])])
        XCTAssertEqual(observedTokens, tokens)
        XCTAssertEqual(completion?.generationTokenCount, tokens.count)
        XCTAssertEqual(completion?.rejectedToolCallCount, 0)
    }

    func testTerminatedProbabilityEmissionStopsTheWrappedHandler() {
        var handler = LogProbabilityTokenLoopHandler(BoundaryHandler())
        let values = GenerateTokenLogProbabilities(chosen: .init(token: 65, logProbability: -0.5))
        var emissions = 0
        let emit: (sending LogProbabilityGeneration<String>) -> Bool = { _ in
            emissions += 1
            return false
        }

        guard case .cancelled = handler.onToken(65, logProbabilities: values, emit: emit) else {
            XCTFail("terminated stream must cancel generation")
            return
        }
        XCTAssertEqual(emissions, 1)

        guard case .cancelled = handler.onStopToken(65, logProbabilities: values, emit: emit) else {
            XCTFail("terminated stream must cancel stop-token handling")
            return
        }
        XCTAssertEqual(emissions, 2)
    }
}
