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

    @Test func llamaProtocolMarkerNeverLeaksAsResponseText() {
        var router = AllowedToolOutputRouter(format: .llama3, tools: tools)
        let events = router.process(
            #"<|python_tag|>{"name":"get_weather","arguments":{"location":"Tokyo"}}"#)

        #expect(events.count == 1)
        guard case .toolCall(let call) = events[0] else {
            Issue.record("Expected one tool call without protocol text")
            return
        }
        #expect(call.function.name == "get_weather")
    }

    @Test func malformedTaggedToolSyntaxNeverLeaksAsResponseText() {
        var router = AllowedToolOutputRouter(format: .json, tools: tools)
        #expect(router.process(#"<tool_call>{"name":}</tool_call>"#).isEmpty)
        #expect(router.finish().isEmpty)
    }

    @Test func mixedCallTextCallEventsPreserveSourceOrder() {
        var router = AllowedToolOutputRouter(format: .json, tools: tools)
        let events = router.process(
            #"<tool_call>{"name":"get_weather","arguments":{}}</tool_call>between<tool_call>{"name":"get_weather","arguments":{}}</tool_call>"#)

        #expect(events.count == 3)
        guard case .toolCall = events[0] else {
            Issue.record("Expected the first event to be a tool call")
            return
        }
        #expect(events[1] == .response("between"))
        guard case .toolCall = events[2] else {
            Issue.record("Expected the final event to be a tool call")
            return
        }
    }

    @Test func eosPreservesTextAfterRecoveredMistralCall() {
        var router = AllowedToolOutputRouter(format: .mistral, tools: tools)
        #expect(router.process(#"[TOOL_CALLS]get_weather[ARGS]{"location":"Rome"}done"#).isEmpty)

        let events = router.finish()
        #expect(events.count == 2)
        guard case .toolCall(let call) = events[0] else {
            Issue.record("Expected the recovered Mistral call first")
            return
        }
        #expect(call.function.name == "get_weather")
        #expect(events[1] == .response("done"))
    }

    @Test func eosPreservesTextAfterRecoveredLFM2Call() {
        var router = AllowedToolOutputRouter(format: .lfm2, tools: tools)
        #expect(router.process("<|tool_call_start|>[get_weather()]after").isEmpty)

        let events = router.finish()
        #expect(events.count == 2)
        guard case .toolCall(let call) = events[0] else {
            Issue.record("Expected the recovered LFM2 call first")
            return
        }
        #expect(call.function.name == "get_weather")
        #expect(events[1] == .response("after"))
    }

    @Test func partialOrMismatchedProtocolAtEOSNeverLeaksAsResponseText() {
        var partialRouter = AllowedToolOutputRouter(format: .json, tools: tools)
        #expect(partialRouter.process("<tool_cal").isEmpty)
        #expect(partialRouter.finish().isEmpty)

        var mismatchedRouter = AllowedToolOutputRouter(format: .json, tools: tools)
        #expect(mismatchedRouter.process("<tool_callx>").isEmpty)
        #expect(mismatchedRouter.finish().isEmpty)
    }

    @Test func callsOutsideEnabledToolDefinitionsAreSuppressed() {
        var router = AllowedToolOutputRouter(format: .llama3, tools: tools)
        #expect(router.process(#"<|python_tag|>{"name":"delete_everything","arguments":{}}"#).isEmpty)
        #expect(router.finish().isEmpty)
    }
}

#endif
