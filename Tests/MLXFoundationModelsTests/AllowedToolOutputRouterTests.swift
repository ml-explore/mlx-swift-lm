// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

import MLXLMCommon
import Testing
@testable import MLXFoundationModels

@Suite
struct AllowedToolOutputRouterTests {
    private let tools: [[String: any Sendable]] = [
        [
            "type": "function",
            "function": [
                "name": "get_weather",
                "description": "Get weather",
                "parameters": ["type": "object"] as [String: any Sendable],
            ] as [String: any Sendable],
        ]
    ]

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

    @Test func eosRecoversQwenJSONCallWithRedundantOuterBraces() {
        var router = AllowedToolOutputRouter(format: .json, tools: tools)
        #expect(
            router.process(
                #"<tool_call>{{"name":"get_weather","arguments":{"location":"Tokyo"}}}}"#
            ).isEmpty)

        let events = router.finish()
        #expect(events.count == 1)
        guard events.count == 1 else { return }
        guard case .toolCall(let call) = events[0] else {
            Issue.record("Expected the EOS-delimited Qwen JSON tool call")
            return
        }
        #expect(call.function.name == "get_weather")
        #expect(call.function.arguments["location"] == .string("Tokyo"))
    }

    @Test func eosDoesNotRecoverRedundantJSONBracesWithArbitrarySuffix() {
        var router = AllowedToolOutputRouter(format: .json, tools: tools)
        #expect(
            router.process(
                #"<tool_call>{{"name":"get_weather","arguments":{"location":"Tokyo"}}}}oops"#
            ).isEmpty)
        #expect(router.finish().isEmpty)
    }

    @Test func eosDoesNotRecoverQwenJSONCallWithOnlyOneRedundantTrailingBrace() {
        var router = AllowedToolOutputRouter(format: .json, tools: tools)
        #expect(
            router.process(
                #"<tool_call>{{"name":"get_weather","arguments":{"location":"Tokyo"}}}"#
            ).isEmpty)
        #expect(router.finish().isEmpty)
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
            #"<think>check live data</think><tool_call>{"name":"get_weather","arguments":{"location":"Rome"}}</tool_call>"#
        )
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

    @Test func splitLlamaCallIgnoresBracesInsideJSONString() {
        var router = AllowedToolOutputRouter(format: .llama3, tools: tools)
        #expect(
            router.process(
                #"<|python_tag|>{"name":"get_weather","arguments":{"query":"}}""#
            ).isEmpty)

        let events = router.process("}}")
        #expect(events.count == 1)
        guard events.count == 1 else { return }
        guard case .toolCall(let call) = events[0] else {
            Issue.record("Expected one Llama call without response or protocol text")
            return
        }
        #expect(call.function.name == "get_weather")
        #expect(call.function.arguments["query"] == .string("}}"))
        #expect(router.finish().isEmpty)
    }

    @Test func malformedTaggedToolSyntaxNeverLeaksAsResponseText() {
        var router = AllowedToolOutputRouter(format: .json, tools: tools)
        #expect(router.process(#"<tool_call>{"name":}</tool_call>"#).isEmpty)
        #expect(router.finish().isEmpty)
    }

    @Test func mixedCallTextCallEventsPreserveSourceOrder() {
        var router = AllowedToolOutputRouter(format: .json, tools: tools)
        let events = router.process(
            #"<tool_call>{"name":"get_weather","arguments":{}}</tool_call>between<tool_call>{"name":"get_weather","arguments":{}}</tool_call>"#
        )

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
        #expect(
            router.process(#"<|python_tag|>{"name":"delete_everything","arguments":{}}"#).isEmpty)
        #expect(router.finish().isEmpty)
    }

    @Test func splitCallTextCallEventsPreserveSourceOrder() {
        var router = AllowedToolOutputRouter(format: .json, tools: tools)
        #expect(router.process(#"<tool_call>{"name":"get_weather","arguments":{"#).isEmpty)

        let events = router.process(
            #"}}</tool_call>between<tool_call>{"name":"get_weather","arguments":{}}</tool_call>"#)
        #expect(events.count == 3)
        guard case .toolCall = events[0] else {
            Issue.record("Expected the completed split call first")
            return
        }
        #expect(events[1] == .response("between"))
        guard case .toolCall = events[2] else {
            Issue.record("Expected the second call after intervening text")
            return
        }
    }

    @Test func strayClosingMarkerNeverLeaksAsResponseText() {
        var router = AllowedToolOutputRouter(format: .json, tools: tools)
        let events = router.process(
            #"<tool_call>{"name":"get_weather","arguments":{}}</tool_call></tool_call>"#)
        #expect(events.count == 1)
        guard case .toolCall = events[0] else {
            Issue.record("Expected only the valid call")
            return
        }
    }

    @Test func ordinaryTagLikeTextRemainsAResponse() {
        var jsonRouter = AllowedToolOutputRouter(format: .json, tools: tools)
        #expect(
            jsonRouter.process("Use <table> for layout") == [.response("Use <table> for layout")])

        var mistralRouter = AllowedToolOutputRouter(format: .mistral, tools: tools)
        #expect(mistralRouter.process("[Today] is sunny") == [.response("[Today] is sunny")])
    }

    @Test func malformedProtocolSuppressesOnlyTheMarkerSpan() {
        var router = AllowedToolOutputRouter(format: .json, tools: tools)
        #expect(router.process("before <tool_callx> after") == [.response("before  after")])
    }

    @Test func eosRecoversLFM2NestedArrayAndStringBracketArguments() {
        var router = AllowedToolOutputRouter(format: .lfm2, tools: tools)
        #expect(
            router.process(#"<|tool_call_start|>[get_weather(location=["Paris", "]"])]after"#)
                .isEmpty)

        let events = router.finish()
        #expect(events.count == 2)
        guard case .toolCall(let call) = events[0] else {
            Issue.record("Expected the balanced LFM2 call first")
            return
        }
        #expect(call.function.name == "get_weather")
        #expect(events[1] == .response("after"))
    }

    @Test func llamaMarkerSplitFromJSONNeverLeaksAsResponse() {
        var router = AllowedToolOutputRouter(format: .llama3, tools: tools)
        #expect(router.process("<|python_tag|>").isEmpty)

        let events = router.process(#"{"name":"get_weather","arguments":{}}"#)
        #expect(events.count == 1)
        guard case .toolCall(let call) = events[0] else {
            Issue.record("Expected the split Llama call without marker text")
            return
        }
        #expect(call.function.name == "get_weather")
    }

    @Test func leadingResponsePrecedesAnIncompleteTaggedCall() {
        var router = AllowedToolOutputRouter(format: .json, tools: tools)
        #expect(
            router.process(#"before <tool_call>{"name":"get_weather","arguments":{"#)
                == [.response("before ")])

        let events = router.process(#"}}</tool_call>"#)
        #expect(events.count == 1)
        guard case .toolCall = events[0] else {
            Issue.record("Expected the completed call after leading response")
            return
        }
    }

    @Test func leadingResponsePrecedesAnIncompleteSecondTaggedCall() {
        var router = AllowedToolOutputRouter(format: .json, tools: tools)
        let first = router.process(
            #"<tool_call>{"name":"get_weather","arguments":{}}</tool_call>between <tool_call>{"name":"get_weather","arguments":{"#
        )
        #expect(first.count == 2)
        guard first.count == 2 else { return }
        guard case .toolCall = first[0] else {
            Issue.record("Expected the first call")
            return
        }
        #expect(first[1] == .response("between "))

        let second = router.process(#"}}</tool_call>"#)
        #expect(second.count == 1)
        guard case .toolCall = second[0] else {
            Issue.record("Expected the second completed call")
            return
        }
    }

    @Test func leadingResponsePrecedesAStrayClosingMarker() {
        var router = AllowedToolOutputRouter(format: .json, tools: tools)
        let events = router.process(
            #"<tool_call>{"name":"get_weather","arguments":{}}</tool_call>before </tool_call>"#)
        #expect(events.count == 2)
        guard events.count == 2 else { return }
        guard case .toolCall = events[0] else {
            Issue.record("Expected the valid call")
            return
        }
        #expect(events[1] == .response("before "))
    }

    @Test func nearProtocolMarkersSuppressWithoutStrippingOrdinaryTags() {
        var ordinaryRouter = AllowedToolOutputRouter(format: .json, tools: tools)
        #expect(
            ordinaryRouter.process("Use <toolbar> and <tool> labels")
                == [.response("Use <toolbar> and <tool> labels")])

        var partialRouter = AllowedToolOutputRouter(format: .json, tools: tools)
        #expect(partialRouter.process("before <tool_cal") == [.response("before ")])
        #expect(partialRouter.finish().isEmpty)

        var mismatchRouter = AllowedToolOutputRouter(format: .json, tools: tools)
        #expect(mismatchRouter.process("before <tool_callx> after") == [.response("before  after")])

        var mistralRouter = AllowedToolOutputRouter(format: .mistral, tools: tools)
        #expect(mistralRouter.process("before [TOOL_CALLX] after") == [.response("before  after")])
    }

    @Test func specializedEOSPathsSanitizeOnlyMalformedResidualMarkers() {
        var mistralRouter = AllowedToolOutputRouter(format: .mistral, tools: tools)
        #expect(
            mistralRouter.process(
                #"[TOOL_CALLS]get_weather[ARGS]{"location":"Rome"}before [TOOL_CALLX] after"#
            ).isEmpty)
        let mistralEvents = mistralRouter.finish()
        #expect(mistralEvents.count == 2)
        guard mistralEvents.count == 2 else { return }
        guard case .toolCall = mistralEvents[0] else {
            Issue.record("Expected the recovered Mistral call")
            return
        }
        #expect(mistralEvents[1] == .response("before  after"))

        var lfmRouter = AllowedToolOutputRouter(format: .lfm2, tools: tools)
        #expect(
            lfmRouter.process(
                "<|tool_call_start|>[get_weather()]before <|tool_call_startX> after"
            ).isEmpty)
        let lfmEvents = lfmRouter.finish()
        #expect(lfmEvents.count == 2)
        guard lfmEvents.count == 2 else { return }
        guard case .toolCall = lfmEvents[0] else {
            Issue.record("Expected the recovered LFM2 call")
            return
        }
        #expect(lfmEvents[1] == .response("before  after"))
    }

    @Test func eosRecoversLFM2SingleQuotedEscapedBracketArguments() {
        var router = AllowedToolOutputRouter(format: .lfm2, tools: tools)
        #expect(
            router.process(
                #"<|tool_call_start|>[get_weather(location='it\']s ] sunny')]after"#
            ).isEmpty)

        let events = router.finish()
        #expect(events.count == 2)
        guard events.count == 2 else { return }
        guard case .toolCall(let call) = events[0] else {
            Issue.record("Expected the single-quoted LFM2 call")
            return
        }
        #expect(call.function.name == "get_weather")
        #expect(events[1] == .response("after"))
    }

    @Test func eosSuppressesUnclosedExactProtocolTails() {
        var jsonRouter = AllowedToolOutputRouter(format: .json, tools: tools)
        #expect(jsonRouter.process(#"before <tool_call>{"name":"#) == [.response("before ")])
        #expect(jsonRouter.finish().isEmpty)

        var lfmRouter = AllowedToolOutputRouter(format: .lfm2, tools: tools)
        #expect(
            lfmRouter.process("before <|tool_call_start|>[get_weather(")
                == [.response("before ")])
        #expect(lfmRouter.finish().isEmpty)

        var mistralRouter = AllowedToolOutputRouter(format: .mistral, tools: tools)
        #expect(
            mistralRouter.process("before [TOOL_CALLS]get_weather[ARGS]{")
                == [.response("before ")])
        #expect(mistralRouter.finish().isEmpty)
    }

    @Test func responseBeforeIncompleteBareJSONSecondCallIsOrdered() {
        var router = AllowedToolOutputRouter(format: .json, tools: tools)
        let initial = router.process(
            #"<tool_call>{"name":"get_weather","arguments":{}}</tool_call>between {"name":"get_weather","arguments":{"#
        )
        #expect(initial.count == 2)
        guard initial.count == 2 else { return }
        guard case .toolCall = initial[0] else {
            Issue.record("Expected the complete tagged call first")
            return
        }
        #expect(initial[1] == .response("between "))

        let completed = router.process(#"}}"#)
        #expect(completed.count == 1)
        guard case .toolCall = completed[0] else {
            Issue.record("Expected the buffered bare JSON call")
            return
        }
    }

    @Test func eosEmitsLFM2PrefixBeforeUnfinishedSecondCallExactlyOnce() {
        var router = AllowedToolOutputRouter(format: .lfm2, tools: tools)
        #expect(
            router.process(
                "<|tool_call_start|>[get_weather()]between <|tool_call_start|>[get_weather("
            ).isEmpty)

        let events = router.finish()
        #expect(events.count == 2)
        guard events.count == 2 else { return }
        guard case .toolCall = events[0] else {
            Issue.record("Expected the completed first LFM2 call")
            return
        }
        #expect(events[1] == .response("between "))
    }

    @Test func eosSanitizesLFM2InterCallResponsePrefix() {
        var router = AllowedToolOutputRouter(format: .lfm2, tools: tools)
        #expect(
            router.process(
                "<|tool_call_start|>[get_weather()]before <|tool_call_startX> after <|tool_call_start|>[get_weather()]"
            ).isEmpty)

        let events = router.finish()
        #expect(events.count == 3)
        guard events.count == 3 else { return }
        guard case .toolCall = events[0] else {
            Issue.record("Expected the first LFM2 call")
            return
        }
        #expect(events[1] == .response("before  after "))
        guard case .toolCall = events[2] else {
            Issue.record("Expected the second LFM2 call")
            return
        }
    }
}

#endif
