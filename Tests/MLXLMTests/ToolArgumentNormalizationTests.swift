// Copyright © 2026 Apple Inc.

import Foundation
import Testing

@testable import MLXLMCommon

struct ToolArgumentNormalizationTests {
    private func tools(_ schema: [String: any Sendable]) -> [[String: any Sendable]] {
        let parameters: [String: any Sendable] = [
            "type": "object", "properties": ["value": schema], "required": ["value"],
        ]
        return [["function": ["name": "read", "parameters": parameters] as [String: any Sendable]]]
    }

    @Test("JSON, native XML and recovered XML read the same declared values")
    func dialectParity() throws {
        let cases: [(String, String, JSONValue)] = [
            ("integer", "6", .int(6)), ("integer", "1.0", .int(1)),
            ("int64", "1.0", .int(1)), ("float64", "0.25", .double(0.25)),
            ("integer", "1.20e2", .int(120)), ("integer", "-0.0", .int(0)),
            ("integer", "9007199254740993.0", .int(9_007_199_254_740_993)),
            ("integer", "9223372036854775807.0", .int(Int.max)),
            ("integer", "-9223372036854775808.0", .int(Int.min)),
            ("number", "1e100", .double(1e100)), ("number", "0.25", .double(0.25)),
            ("boolean", "false", .bool(false)), ("boolean", "yes", .bool(true)),
            ("array", "['SF', 'LA']", .array([.string("SF"), .string("LA")])),
            ("array", "\"SF\", \"LA\"", .array([.string("SF"), .string("LA")])),
            ("array", "('SF', 'LA')", .array([.string("SF"), .string("LA")])),
            ("object", "{'enabled': False}", .object(["enabled": .bool(false)])),
        ]
        for (type, text, expected) in cases {
            let json = String(
                decoding: try JSONSerialization.data(withJSONObject: [
                    "name": "read", "arguments": ["value": text],
                ]), as: UTF8.self)
            let xml = "<function=read><parameter=value>\(text)</parameter></function>"
            for (format, payload): (ToolCallFormat, String) in [
                (.json, "<tool_call>\(json)</tool_call>"),
                (.xmlFunction, "<tool_call>\(xml)</tool_call>"),
                (.lfm2, xml),
            ] {
                let processor = ToolCallProcessor(format: format, tools: tools(["type": type]))
                _ = processor.processChunk(payload)
                processor.processEOS()
                #expect(
                    processor.toolCalls.first?.function.arguments["value"] == expected,
                    "\(type): \(text), \(format)")
                #expect(processor.rejectedToolCalls.isEmpty)
            }
        }
    }

    @Test("Unconvertible values survive normalization and strict validation rejects them")
    func invalidValues() {
        let cases = [
            ("integer", "1.5"), ("integer", "1.00000000000000000000000000000000000000001"),
            ("integer", "9223372036854775808"), ("number", "nan"), ("number", "inf"),
            ("boolean", "maybe"), ("array", "SF, LA"), ("array", "{'a': 1}"),
            ("object", "[1, 2]"), ("array", "__import__('os').system('x')"),
        ]
        for (type, text) in cases {
            #expect(
                ToolArgumentNormalization.normalize(.string(text), schema: ["type": type])
                    == .string(text))
            for policy in ToolCallValidationPolicy.allCases {
                let processor = ToolCallProcessor(
                    format: .lfm2, tools: tools(["type": type]),
                    toolCallPolicy: .init(validation: policy))
                _ = processor.processChunk(
                    "<function=read><parameter=value>\(text)</parameter></function>")
                processor.processEOS()
                if policy == .strict {
                    #expect(processor.toolCalls.isEmpty)
                    #expect(processor.rejectedToolCalls.first?.reason == .invalidArguments)
                    #expect(processor.recoveredToolCallCount == 0)
                } else {
                    #expect(processor.toolCalls.first?.function.arguments["value"] == .string(text))
                    #expect(processor.recoveredToolCallCount == 1)
                }
            }
        }
    }

    @Test("Strings and ambiguous unions retain their spelling")
    func ambiguity() {
        let schemas: [[String: any Sendable]] = [
            ["type": "string"], ["type": ["string", "integer"]],
            ["type": ["boolean", "integer"]],
            ["anyOf": [["type": "integer"], ["$ref": "#/$defs/text"]]],
            ["type": "integer", "$ref": "#/$defs/other"],
            ["description": "opaque"],
        ]
        for schema in schemas {
            #expect(
                ToolArgumentNormalization.normalize(.string("001"), schema: schema)
                    == .string("001"))
        }
        #expect(
            ToolArgumentNormalization.normalize(
                .string("null"), schema: ["type": ["string", "null"]]) == .string("null"))
    }

    @Test("Numeric unions and intersections respect integer subtyping")
    func numericCombinators() {
        let intersection: [String: any Sendable] = [
            "allOf": [["type": "number"], ["type": "integer"]]
        ]
        #expect(
            ToolArgumentNormalization.normalize(.string("6.0"), schema: intersection) == .int(6))
        let union: [String: any Sendable] = ["type": ["integer", "number"]]
        #expect(ToolArgumentNormalization.normalize(.string("1.5"), schema: union) == .double(1.5))
    }

    @Test("Normalization descends into declared objects and arrays and preserves IDs")
    func nestedArguments() {
        let item: [String: any Sendable] = [
            "type": "object", "properties": ["count": ["type": "integer"]],
        ]
        let schema: [String: any Sendable] = ["type": "array", "items": item]
        let call = ToolCall(
            function: .init(name: "read", arguments: ["value": .string("[{'count': '2'}]")]),
            id: "keep-me")
        let normalized = ToolArgumentNormalization.normalize(call, tools: tools(schema))
        #expect(normalized.id == call.id)
        #expect(normalized.function.arguments["value"] == .array([.object(["count": .int(2)])]))
        #expect(ToolArgumentNormalization.normalize(normalized, tools: tools(schema)) == normalized)
    }

    @Test("Permissive schema validation never bypasses name authorization or syntax")
    func authorization() {
        for recovery in ToolCallRecoveryPolicy.allCases {
            let processor = ToolCallProcessor(
                format: .json, tools: tools(["type": "integer"]),
                toolCallPolicy: .init(recovery: recovery, validation: .permissive))
            _ = processor.processChunk(
                #"<tool_call>{"name":"other","arguments":{"value":"6"}}</tool_call>"#)
            processor.processEOS()
            #expect(processor.toolCalls.isEmpty)
            #expect(processor.rejectedToolCalls.first?.reason == .undeclaredTool)
        }
    }

    @Test("Extreme exponents and non-integral decimals cannot overflow or round into integers")
    func extremeIntegers() {
        for text in [
            "1e9223372036854775807", "1e-9223372036854775808",
            "1.0e-9223372036854775808", "1e-999999999999999999999999",
            "-9223372036854775809.0", "99999999999999999999.1", "0.00000000000000000001",
        ] {
            #expect(
                ToolArgumentNormalization.normalize(.string(text), schema: ["type": "integer"])
                    == .string(text))
        }
        for (text, value) in [("0e10000", 0), ("1200e-2", 12), (".000e-300", 0)] {
            #expect(
                ToolArgumentNormalization.normalize(.string(text), schema: ["type": "integer"])
                    == .int(value))
        }
    }

    @Test("Python container parsing is bounded and rejects incomplete structure")
    func boundedLiterals() {
        #expect(
            tryParsePythonLiteral(
                String(repeating: "[", count: 64) + "0" + String(repeating: "]", count: 64)) == nil)
        #expect(tryParsePythonLiteral("['SF',") == nil)
        #expect(tryParsePythonLiteral("[f()]") == nil)
    }

    @Test("Python string escapes decode as Python does")
    func pythonStringEscapes() {
        for (literal, value) in [
            (#"'\x41\101\0'"#, "AA\0"),
            (#"'\a\v'"#, "\u{7}\u{b}"),
            ("'line\\\nnext'", "linenext"),
            (#"'\ud83d\ude00'"#, "\u{1F600}"),
            (#"'\d'"#, #"\d"#),
        ] {
            #expect(tryParsePythonLiteral(literal) as? String == value)
        }
        #expect(tryParsePythonLiteral(#"'\ud83d'"#) == nil)
        #expect(tryParsePythonLiteral(#"'\x4'"#) == nil)
    }

    @Test("Python triple-quoted strings span lines and end at three quotes")
    func pythonTripleQuotedStrings() {
        for (literal, value) in [
            ("'''line 1\nline 2'''", "line 1\nline 2"),
            ("\"\"\"It's \"quoted\" text\"\"\"", "It's \"quoted\" text"),
            ("''''a'''", "'a"),
            ("''''''", ""),
        ] {
            #expect(tryParsePythonLiteral(literal) as? String == value)
        }
        #expect(tryParsePythonLiteral("'''a''''") == nil)
        #expect(tryParsePythonLiteral("'''a''") == nil)
    }
}
