// Copyright © 2026 Apple Inc.

import Foundation
import MLXLMCommon
import Testing

/// Structural strictness of the native `.xmlFunction` protocol.
///
/// `XMLFunctionParser` previously accepted a structurally degraded payload and
/// returned a call with silently missing arguments. The only thing preventing
/// execution was schema-level required-argument validation, so a tool whose
/// schema omitted `required` could be invoked with no arguments at all from a
/// truncated frame. These tests pin the parser to the shared
/// `QwenXMLPayloadScanner` grammar and deliberately declare **no** `required`
/// list, so structural validity alone must carry the rejection.
struct XMLFunctionParserStrictnessTests {

    private let parser = XMLFunctionParser(startTag: "<tool_call>", endTag: "</tool_call>")

    /// Schemas without a `required` list: required-argument validation cannot
    /// mask a structural parser flaw here.
    private func toolSchemas(_ names: String...) -> [[String: any Sendable]] {
        names.map { name in
            ["function": ["name": name] as [String: any Sendable]]
        }
    }

    // MARK: - Structural rejection

    @Test("An unclosed <parameter> never becomes a call")
    func unclosedParameterIsRejected() {
        let payloads = [
            "<function=weather><parameter=city>Paris</function>",
            "<tool_call><function=weather><parameter=city>Paris</function></tool_call>",
            """
            <tool_call>
            <function=weather>
            <parameter=city>Paris
            </function>
            </tool_call>
            """,
            // First parameter closes, second does not: partial success is still
            // a structurally invalid payload.
            "<function=weather><parameter=city>Paris</parameter><parameter=unit>c</function>",
        ]

        for payload in payloads {
            #expect(
                parser.parse(content: payload, tools: nil) == nil,
                "expected structural rejection of: \(payload)")
        }
    }

    @Test("A missing </function> never becomes a call")
    func missingFunctionCloseIsRejected() {
        let payloads = [
            "<function=weather>",
            "<function=weather><parameter=city>Paris</parameter>",
            "<tool_call><function=weather><parameter=city>Paris</parameter></tool_call>",
        ]

        for payload in payloads {
            #expect(
                parser.parse(content: payload, tools: nil) == nil,
                "expected structural rejection of: \(payload)")
        }
    }

    @Test("Malformed function and parameter names never become a call")
    func malformedNamesAreRejected() {
        let payloads = [
            "<function=><parameter=city>Paris</parameter></function>",
            "<function=get weather><parameter=city>Paris</parameter></function>",
            "<function=weather><parameter=>Paris</parameter></function>",
        ]

        for payload in payloads {
            #expect(
                parser.parse(content: payload, tools: nil) == nil,
                "expected structural rejection of: \(payload)")
        }
    }

    @Test("Content beyond the payload is never silently discarded")
    func trailingContentIsRejected() {
        let payloads = [
            // A second call in one frame must not silently collapse to the first.
            "<function=a><parameter=x>1</parameter></function><function=b></function>",
            "<function=weather></function>trailing text",
            // A complete frame followed by text: the streaming processor splits
            // this, so a direct parse must not quietly drop the remainder.
            "<tool_call><function=weather></function></tool_call>trailing",
        ]

        for payload in payloads {
            #expect(
                parser.parse(content: payload, tools: nil) == nil,
                "expected rejection of payload with trailing content: \(payload)")
        }
    }

    // MARK: - Literal markers inside values are data

    @Test("A literal </function> inside a parameter value is preserved")
    func literalFunctionCloseIsPreserved() throws {
        let content = """
            <tool_call>
            <function=weather>
            <parameter=city>
            literal </function> text
            </parameter>
            </function>
            </tool_call>
            """

        let call = try #require(parser.parse(content: content, tools: nil))

        #expect(call.function.name == "weather")
        #expect(call.function.arguments["city"] == .string("literal </function> text"))
    }

    @Test("A literal </tool_call> inside a parameter value is preserved")
    func literalFrameCloseIsPreserved() throws {
        let content =
            "<tool_call><function=weather><parameter=city>a</tool_call>b</parameter>"
            + "</function></tool_call>"

        let call = try #require(parser.parse(content: content, tools: nil))

        #expect(call.function.arguments["city"] == .string("a</tool_call>b"))
    }

    // MARK: - Canonical payloads keep working

    @Test("Canonical payloads still parse unchanged")
    func canonicalPayloadsStillParse() throws {
        let unframed = try #require(
            parser.parse(
                content:
                    "<function=get_weather><parameter=location>Tokyo</parameter>"
                    + "<parameter=unit>celsius</parameter></function>",
                tools: nil))
        #expect(unframed.function.name == "get_weather")
        #expect(unframed.function.arguments["location"] == .string("Tokyo"))
        #expect(unframed.function.arguments["unit"] == .string("celsius"))

        let framedMultiline = try #require(
            parser.parse(
                content: """
                    <tool_call>
                    <function=get_current_datetime>
                    </function>
                    </tool_call>
                    """,
                tools: nil))
        #expect(framedMultiline.function.name == "get_current_datetime")
        #expect(framedMultiline.function.arguments.isEmpty)

        let multilineValue = try #require(
            parser.parse(
                content: """
                    <function=get_weather>
                    <parameter=location>
                    Tokyo
                    </parameter>
                    </function>
                    """,
                tools: nil))
        #expect(multilineValue.function.arguments["location"] == .string("Tokyo"))
    }

    @Test("Schema-driven type conversion is unchanged")
    func typeConversionIsUnchanged() throws {
        let tools: [[String: any Sendable]] = [
            [
                "function": [
                    "name": "search",
                    "parameters": [
                        "properties": [
                            "page": ["type": "integer"],
                            "filters": ["type": "object"],
                        ]
                    ],
                ] as [String: any Sendable]
            ]
        ]
        let content =
            "<tool_call><function=search><parameter=page>1</parameter>"
            + #"<parameter=filters>{"archived": false, "limit": 1}</parameter></function></tool_call>"#

        let call = try #require(parser.parse(content: content, tools: tools))

        #expect(call.function.arguments["page"] == .int(1))
        #expect(
            call.function.arguments["filters"]
                == .object(["archived": .bool(false), "limit": .int(1)]))
    }

    // MARK: - End-to-end through ToolCallProcessor

    @Test("Native XML rejects an unclosed parameter at EOS")
    func nativeXMLRejectsUnclosedParameterAtEOS() {
        let processor = ToolCallProcessor(
            format: .xmlFunction,
            tools: toolSchemas("weather"))

        _ = processor.processChunk(
            "<tool_call><function=weather><parameter=city>Paris</function></tool_call>")
        processor.processEOS()

        #expect(processor.toolCalls.isEmpty)
        #expect(processor.rejectedToolCalls.count == 1)
    }

    @Test("A truncated frame is rejected at every chunk boundary")
    func truncatedFrameRejectedAtEveryChunkBoundary() {
        let payload = """
            <tool_call>
            <function=weather>
            <parameter=city>Paris
            </function>
            </tool_call>
            """

        for splitOffset in 0 ... payload.count {
            let split = payload.index(payload.startIndex, offsetBy: splitOffset)
            let processor = ToolCallProcessor(
                format: .xmlFunction,
                tools: toolSchemas("weather"))

            _ = processor.processChunk(String(payload[..<split]))
            _ = processor.processChunk(String(payload[split...]))
            processor.processEOS()

            #expect(
                processor.toolCalls.isEmpty,
                "executed a truncated frame when split at \(splitOffset)")
            #expect(
                processor.rejectedToolCalls.count == 1,
                "expected exactly one rejection when split at \(splitOffset)")
        }
    }

    @Test("A canonical call still executes at every chunk boundary")
    func canonicalCallExecutesAtEveryChunkBoundary() throws {
        let payload = """
            <tool_call>
            <function=weather>
            <parameter=city>
            Paris
            </parameter>
            </function>
            </tool_call>
            """

        for splitOffset in 0 ... payload.count {
            let split = payload.index(payload.startIndex, offsetBy: splitOffset)
            let processor = ToolCallProcessor(
                format: .xmlFunction,
                tools: toolSchemas("weather"))

            _ = processor.processChunk(String(payload[..<split]))
            _ = processor.processChunk(String(payload[split...]))
            processor.processEOS()

            #expect(
                processor.rejectedToolCalls.isEmpty,
                "rejected a canonical call when split at \(splitOffset)")
            #expect(
                processor.toolCalls.count == 1,
                "lost a canonical call when split at \(splitOffset)")
            let call = try #require(processor.toolCalls.first)
            #expect(call.function.name == "weather")
            #expect(
                call.function.arguments["city"] == .string("Paris"),
                "dropped arguments when split at \(splitOffset)")
        }
    }

    @Test("A literal </function> argument survives streaming at every chunk boundary")
    func literalFunctionCloseSurvivesStreaming() throws {
        let payload = """
            <tool_call>
            <function=weather>
            <parameter=city>
            literal </function> text
            </parameter>
            </function>
            </tool_call>
            """

        for splitOffset in 0 ... payload.count {
            let split = payload.index(payload.startIndex, offsetBy: splitOffset)
            let processor = ToolCallProcessor(
                format: .xmlFunction,
                tools: toolSchemas("weather"))

            _ = processor.processChunk(String(payload[..<split]))
            _ = processor.processChunk(String(payload[split...]))
            processor.processEOS()

            #expect(
                processor.toolCalls.count == 1,
                "lost the call when split at \(splitOffset)")
            let call = try #require(processor.toolCalls.first)
            #expect(
                call.function.arguments["city"] == .string("literal </function> text"),
                "corrupted the argument when split at \(splitOffset)")
        }
    }

    @Test("Text following a canonical call is still emitted")
    func trailingTextAfterCallIsPreserved() {
        let processor = ToolCallProcessor(
            format: .xmlFunction,
            tools: toolSchemas("weather"))

        var response = ""
        response +=
            processor.processChunk(
                "<tool_call><function=weather><parameter=city>Paris</parameter>"
                    + "</function></tool_call>done") ?? ""
        response += processor.processEOS(returnBufferedText: true) ?? ""

        #expect(processor.toolCalls.count == 1)
        #expect(processor.rejectedToolCalls.isEmpty)
        #expect(response.contains("done"))
    }

    @Test("Consecutive canonical calls are all executed")
    func consecutiveCallsAreExecuted() {
        let processor = ToolCallProcessor(
            format: .xmlFunction,
            tools: toolSchemas("weather", "clock"))

        _ = processor.processChunk(
            "<tool_call><function=weather><parameter=city>Paris</parameter></function></tool_call>"
                + "<tool_call><function=clock></function></tool_call>")
        processor.processEOS()

        #expect(processor.rejectedToolCalls.isEmpty)
        #expect(processor.toolCalls.map(\.function.name) == ["weather", "clock"])
    }

    // MARK: - Cross-parser parity

    @Test("Both Qwen XML dialect parsers agree on structural validity")
    func dialectParsersAgree() {
        let qwen35 = Qwen35ToolCallParser(startTag: "<tool_call>", endTag: "</tool_call>")
        let tools = toolSchemas("weather")

        let malformed = [
            "<tool_call><function=weather><parameter=city>Paris</function></tool_call>",
            "<tool_call><function=weather><parameter=city>Paris</parameter></tool_call>",
            "<tool_call><function=></function></tool_call>",
        ]
        for payload in malformed {
            #expect(parser.parse(content: payload, tools: tools) == nil, "xmlFunction: \(payload)")
            #expect(qwen35.parse(content: payload, tools: tools) == nil, "qwen35: \(payload)")
        }

        let canonical =
            "<tool_call><function=weather><parameter=city>Paris</parameter></function></tool_call>"
        #expect(parser.parse(content: canonical, tools: tools) != nil)
        #expect(qwen35.parse(content: canonical, tools: tools) != nil)
    }

    /// Before this fix the schema's `required` list was the only defense: the
    /// same truncated frame was executed as `weather()` without one and
    /// rejected with one. Both must now be rejected structurally.
    @Test("Rejection no longer depends on a declared required list")
    func rejectionIsIndependentOfRequiredList() {
        let withoutRequired: [[String: any Sendable]] = toolSchemas("weather")
        let withRequired: [[String: any Sendable]] = [
            [
                "function": [
                    "name": "weather",
                    "parameters": [
                        "type": "object",
                        "required": ["city"],
                    ] as [String: any Sendable],
                ] as [String: any Sendable]
            ]
        ]

        for tools in [withoutRequired, withRequired] {
            let processor = ToolCallProcessor(format: .xmlFunction, tools: tools)
            _ = processor.processChunk(
                "<tool_call><function=weather><parameter=city>Paris</function></tool_call>")
            processor.processEOS()

            #expect(processor.toolCalls.isEmpty)
            #expect(processor.rejectedToolCalls.count == 1)
        }
    }
}
