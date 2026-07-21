// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

import MLXLMCommon
import Testing
@testable import MLXFoundationModels

@Suite
struct AllowedToolOutputRouterTests {
    private let tools: [[String: any Sendable]] = [[
        "type": "function",
        "function": [
            "name": "get_weather",
            "description": "Get weather",
            "parameters": ["type": "object"] as [String: any Sendable],
        ] as [String: any Sendable],
    ]]

    @Test func plainTextRemainsAResponse() {
        var router = AllowedToolOutputRouter(format: .json, tools: tools)
        #expect(router.process("Hello") == [.response("Hello")])
        #expect(router.finish().isEmpty)
    }

    @Test func taggedToolCallNeverLeaksAsResponseText() {
        var router = AllowedToolOutputRouter(format: .json, tools: tools)
        let events = router.process(
            #"<tool_call>{"name":"get_weather","arguments":{"location":"Tokyo"}}</tool_call>"#)
        #expect(events.count == 1)
        guard case .toolCall(let call) = events[0] else {
            Issue.record("Expected one tool call")
            return
        }
        #expect(call.function.name == "get_weather")
        #expect(call.function.arguments["location"] == .string("Tokyo"))
    }

    @Test func splitToolCallIsBufferedUntilComplete() {
        var router = AllowedToolOutputRouter(format: .json, tools: tools)
        #expect(router.process(#"<tool_call>{"name":"get_weather","arguments":{"#).isEmpty)
        let events = router.process(#""location":"Paris"}}</tool_call>"#)
        #expect(events.count == 1)
        guard case .toolCall(let call) = events[0] else {
            Issue.record("Expected one tool call")
            return
        }
        #expect(call.function.arguments["location"] == .string("Paris"))
    }

    @Test func reasoningIsSeparatedBeforeToolParsing() {
        let config = ReasoningConfig(
            startDelimiter: "<think>",
            endDelimiter: "</think>",
            promptStrategy: .templateFlag(key: "enable_thinking", defaultOn: true))
        var router = AllowedToolOutputRouter(
            format: .json, tools: tools,
            reasoning: (config: config, primedInside: false))

        let events = router.process(
            #"<think>check live data</think><tool_call>{"name":"get_weather","arguments":{"location":"Rome"}}</tool_call>"#)
        #expect(events.first == .reasoning("check live data"))
        #expect(events.contains { if case .toolCall = $0 { true } else { false } })
        #expect(!router.isInsideReasoning)
    }

    @Test func eosFlushesNonToolJSONAsResponse() {
        var router = AllowedToolOutputRouter(format: .json, tools: tools)
        #expect(router.process(#"{"answer":"hello"}"#) == [.response(#"{"answer":"hello"}"#)])
        #expect(router.finish().isEmpty)
    }
}

#endif
