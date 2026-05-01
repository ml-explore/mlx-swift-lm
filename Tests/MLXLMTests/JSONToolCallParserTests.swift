// Copyright © 2026 Apple Inc.
//
// Verifies the #169 fix: JSONToolCallParser must accept the
// `{"arguments":"{\"k\":\"v\"}"}` shape that Granite 4 (and others) emit,
// in addition to the standard nested-object form.

import Foundation
import MLXLMCommon
import Testing

struct JSONToolCallParserTests {

    private let parser = JSONToolCallParser(
        startTag: "<tool_call>",
        endTag: "</tool_call>")

    private func stringValue(_ v: JSONValue?) -> String? {
        if case let .string(s) = v { return s }
        return nil
    }

    @Test("Parses nested object arguments (standard shape)")
    func nestedObjectArguments() throws {
        let raw = """
            <tool_call>{"name": "get_weather", "arguments": {"city": "Tokyo", "unit": "c"}}</tool_call>
            """
        let call = try #require(parser.parse(content: raw, tools: nil))
        #expect(call.function.name == "get_weather")
        #expect(stringValue(call.function.arguments["city"]) == "Tokyo")
        #expect(stringValue(call.function.arguments["unit"]) == "c")
    }

    @Test("Parses JSON-encoded string arguments (Granite 4 shape, #169)")
    func stringifiedArguments() throws {
        // The arguments field is a JSON-encoded string of the inner object.
        // Pre-fix this returned nil (parse failed) and the tool call was
        // silently dropped.
        let raw = """
            <tool_call>{"name": "get_weather", "arguments": "{\\"city\\": \\"Tokyo\\", \\"unit\\": \\"c\\"}"}</tool_call>
            """
        let call = try #require(parser.parse(content: raw, tools: nil))
        #expect(call.function.name == "get_weather")
        #expect(stringValue(call.function.arguments["city"]) == "Tokyo")
        #expect(stringValue(call.function.arguments["unit"]) == "c")
    }

    @Test("Empty object arguments are preserved")
    func emptyArguments() throws {
        let raw = """
            <tool_call>{"name": "ping", "arguments": {}}</tool_call>
            """
        let call = try #require(parser.parse(content: raw, tools: nil))
        #expect(call.function.name == "ping")
        #expect(call.function.arguments.isEmpty)
    }

    @Test("Empty stringified arguments are also preserved")
    func emptyStringifiedArguments() throws {
        let raw = """
            <tool_call>{"name": "ping", "arguments": "{}"}</tool_call>
            """
        let call = try #require(parser.parse(content: raw, tools: nil))
        #expect(call.function.name == "ping")
        #expect(call.function.arguments.isEmpty)
    }

    @Test("Malformed stringified arguments returns nil instead of crashing")
    func malformedStringifiedArguments() {
        let raw = """
            <tool_call>{"name": "ping", "arguments": "not-json"}</tool_call>
            """
        // The parser should fail gracefully on a malformed stringified blob;
        // the wire function decode throws and parse returns nil.
        #expect(parser.parse(content: raw, tools: nil) == nil)
    }
}
