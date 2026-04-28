//
//  File.swift
//  mlx-swift-lm
//
//  Created by Ronald Mannak on 4/28/26.
//

import Foundation
import MLXLMCommon
import Testing

struct ToolCallIdTests {
    @Test("Tool message round-trips tool_call_id through DefaultMessageGenerator")
    func testToolMessageRoundTripsId() throws {
        let generator = DefaultMessageGenerator()
        let message = Chat.Message.tool("ok", id: "call_1")

        let dict = generator.generate(message: message)

        #expect(dict["role"] as? String == "tool")
        #expect(dict["content"] as? String == "ok")
        #expect(dict["tool_call_id"] as? String == "call_1")
    }

    @Test("Assistant message round-trips tool_calls through DefaultMessageGenerator")
    func testAssistantMessageRoundTripsToolCalls() throws {
        let tc1 = ToolCall(
            function: .init(
                name: "get_weather",
                arguments: ["location": .string("Paris")]
            ),
            id: "call_a"
        )
        let tc2 = ToolCall(
            function: .init(
                name: "get_time",
                arguments: ["timezone": .string("UTC")]
            ),
            id: "call_b"
        )

        let generator = DefaultMessageGenerator()
        let message = Chat.Message.assistant("", toolCalls: [tc1, tc2])

        let dict = generator.generate(message: message)

        #expect(dict["role"] as? String == "assistant")

        let calls = try #require(dict["tool_calls"] as? [[String: any Sendable]])
        #expect(calls.count == 2)

        let first = calls[0]
        #expect(first["id"] as? String == "call_a")
        #expect(first["type"] as? String == "function")
        let firstFn = try #require(first["function"] as? [String: any Sendable])
        #expect(firstFn["name"] as? String == "get_weather")
        let firstArgs = try #require(firstFn["arguments"] as? [String: any Sendable])
        #expect(firstArgs["location"] as? String == "Paris")

        let second = calls[1]
        #expect(second["id"] as? String == "call_b")
        #expect(second["type"] as? String == "function")
        let secondFn = try #require(second["function"] as? [String: any Sendable])
        #expect(secondFn["name"] as? String == "get_time")
        let secondArgs = try #require(secondFn["arguments"] as? [String: any Sendable])
        #expect(secondArgs["timezone"] as? String == "UTC")
    }

    @Test("Plain user message does not emit tool_call_id or tool_calls keys")
    func testPlainMessageDoesNotEmitToolKeys() throws {
        let generator = DefaultMessageGenerator()
        let message = Chat.Message.user("hi")

        let dict = generator.generate(message: message)

        #expect(dict["tool_call_id"] == nil)
        #expect(dict["tool_calls"] == nil)
    }

    @Test("ToolCallProcessor leaves id nil when the parser does not provide one")
    func testToolCallProcessorLeavesIdNil() throws {
        let processor = ToolCallProcessor(format: .json)
        let content = "<tool_call>{\"name\":\"x\",\"arguments\":{}}</tool_call>"

        _ = processor.processChunk(content)
        processor.processEOS()

        #expect(processor.toolCalls.count == 1)
        let toolCall = try #require(processor.toolCalls.first)
        #expect(toolCall.function.name == "x")
        #expect(toolCall.id == nil)
    }

    @Test("ToolCall initialized without id stays nil (source compatibility)")
    func testToolCallDefaultIdIsNil() throws {
        let tc = ToolCall(function: .init(name: "noop", arguments: [:]))
        #expect(tc.id == nil)
    }
}
