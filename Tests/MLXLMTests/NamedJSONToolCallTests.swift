// Copyright © 2026 Apple Inc.

import Foundation
import Testing

@testable import MLXLMCommon

/// GLM-4-0414's markerless `name\n{json}` dialect.
@Suite("Markerless named-JSON tool calls")
struct NamedJSONToolCallTests {
    private static let tools: [[String: any Sendable]] = [
        [
            "type": "function",
            "function": [
                "name": "get_weather",
                "parameters": [
                    "type": "object",
                    "properties": [
                        "location": ["type": "string"] as [String: any Sendable],
                        "unit": ["type": "string"] as [String: any Sendable],
                        "days": ["type": "integer"] as [String: any Sendable],
                    ] as [String: any Sendable],
                ] as [String: any Sendable],
            ] as [String: any Sendable],
        ],
        [
            "type": "function",
            "function": ["name": "get_time"] as [String: any Sendable],
        ],
        [
            "type": "function",
            "function": [
                "name": "write_file",
                "parameters": [
                    "type": "object",
                    "properties": [
                        "contents": ["type": "string"] as [String: any Sendable]
                    ] as [String: any Sendable],
                ] as [String: any Sendable],
            ] as [String: any Sendable],
        ],
    ]

    private static let call = "get_weather\n{\"location\": \"Paris\", \"unit\": \"celsius\"}"

    private struct Result: Equatable {
        var text = ""
        var calls: [ToolCall.Function] = []
        var rejections: [RejectedToolCall.Reason] = []
        var recovered = 0
    }

    private func process(
        _ chunks: [String], format: ToolCallFormat = .glm4,
        tools: [[String: any Sendable]]? = Self.tools, ordered: Bool
    ) -> Result {
        let processor = ToolCallProcessor(format: format, tools: tools)
        var result = Result()
        if ordered {
            let outputs = chunks.flatMap { processor.processChunkOutputs($0) }
            for output in outputs + processor.processEOSOutputs() {
                switch output {
                case .response(let text): result.text += text
                case .toolCall(let call): result.calls.append(call.function)
                case .rejectedToolCall(let rejection): result.rejections.append(rejection.reason)
                }
            }
        } else {
            for chunk in chunks {
                result.text += processor.processChunk(chunk) ?? ""
            }
            result.text += processor.processEOS(returnBufferedText: true) ?? ""
            result.calls = processor.drainToolCalls().map(\.function)
            result.rejections = processor.drainRejectedToolCalls().map(\.reason)
        }
        result.recovered = processor.recoveredToolCallCount
        return result
    }

    /// Whole, per character, every two-way split, and a trailing empty chunk.
    private func chunkings(_ text: String) -> [[String]] {
        [[text], text.map(String.init)]
            + text.indices.map { [String(text[..<$0]), String(text[$0...])] }
            + [[text, ""]]
    }

    private static let weatherInParis = ToolCall.Function(
        name: "get_weather",
        arguments: ["location": .string("Paris"), "unit": .string("celsius")])

    // MARK: - Parser

    @Test("The GLM4 parser reads a markerless call")
    func parserReadsMarkerlessCall() throws {
        let call = try #require(GLM4ToolCallParser().parse(content: Self.call, tools: Self.tools))
        #expect(call.function == Self.weatherInParis)
    }

    @Test("The GLM4 parser still reads the arg_key dialect")
    func parserReadsArgKeyDialect() throws {
        let content =
            "<tool_call>get_weather<arg_key>location</arg_key><arg_value>Paris</arg_value></tool_call>"
        let call = try #require(GLM4ToolCallParser().parse(content: content, tools: Self.tools))
        #expect(call.function.name == "get_weather")
        #expect(call.function.arguments["location"] == .string("Paris"))
    }

    @Test("A framed name-then-JSON payload parses without a schema")
    func parserReadsFramedNamedJSON() throws {
        let content = "<tool_call>\n" + Self.call + "\n</tool_call>"
        let call = try #require(GLM4ToolCallParser().parse(content: content, tools: nil))
        #expect(call.function == Self.weatherInParis)
    }

    @Test("Protocol markers inside JSON strings remain unchanged")
    func parserPreservesProtocolMarkersInStringArguments() throws {
        let callText =
            "write_file\n{\"contents\": \"<tool_call>hello</tool_call> world\"}"
        let inputs = [callText, "<tool_call>\(callText)</tool_call>"]

        for content in inputs {
            let call = try #require(GLM4ToolCallParser().parse(content: content, tools: nil))
            #expect(
                call.function.arguments["contents"]
                    == .string("<tool_call>hello</tool_call> world"),
                "\(content)")
        }
    }

    @Test("The parser declines payloads that are not exactly one call")
    func parserDeclinesNonCalls() {
        let parser = GLM4ToolCallParser()
        let rejected = [
            "unknown\n{\"location\": \"Paris\"}",
            "get_weather\n[\"Paris\"]",
            "get_weather\n{\"location\": \"Paris\"} and more",
            "get_weather\n{\"location\": ",
            "get_weather",
            "{\"location\": \"Paris\"}",
        ]
        for content in rejected {
            #expect(parser.parse(content: content, tools: Self.tools) == nil, "\(content)")
        }
    }

    // MARK: - Scanner

    @Test("The scanner classifies line-leading text")
    func scannerClassification() throws {
        let scanner = try #require(NamedJSONCallScanner(toolNames: ["get_weather", "get_time"]))
        #expect(scanner.scan("") == .prefix)
        #expect(scanner.scan("get_") == .prefix)
        #expect(scanner.scan("get_weather") == .prefix)
        #expect(scanner.scan("get_weather \n") == .prefix)
        #expect(scanner.scan("get_weather\n{\"a\"") == .openArguments(name: "get_weather"))
        #expect(scanner.scan("Get_weather") == .mismatch)
        #expect(scanner.scan("get_weather_v2\n{}") == .mismatch)
        #expect(scanner.scan("get_weather is a tool") == .mismatch)
        #expect(scanner.scan("get_weather\n[1]") == .mismatch)
        #expect(scanner.scan("get_weather\n{oops}") == .mismatch)
        #expect(scanner.scan(" get_weather\n{}") == .mismatch)
        let overlongSeparator = "get_weather" + String(repeating: "\n", count: 9) + "{}"
        #expect(scanner.scan(overlongSeparator[...]) == .mismatch)

        let complete = scanner.scan("get_time {}\ntail")
        #expect(complete == .complete(name: "get_time", arguments: "{}"))
    }

    @Test("Names that share a prefix, or extend one, resolve on their separator")
    func scannerOverlappingNames() throws {
        let scanner = try #require(
            NamedJSONCallScanner(toolNames: ["get", "get_weather", "météo"]))
        #expect(scanner.scan("get") == .prefix)
        #expect(scanner.scan("get\n{}") == .complete(name: "get", arguments: "{}"))
        #expect(scanner.scan("get_weather\n{}") == .complete(name: "get_weather", arguments: "{}"))
        #expect(scanner.scan("get_wea") == .prefix)
        #expect(scanner.scan("get_wea\n{}") == .mismatch)
        #expect(scanner.scan("getx\n{}") == .mismatch)
        #expect(scanner.scan("mét") == .prefix)
        #expect(scanner.scan("météo {}") == .complete(name: "météo", arguments: "{}"))
        #expect(scanner.maximumPrefixLength == "get_weather".utf8.count + 8)
        #expect(NamedJSONCallScanner(toolNames: [""]) == nil)
    }

    // MARK: - Streaming

    @Test("A markerless call is detected at every chunk boundary", arguments: [false, true])
    func detectedAtEveryBoundary(ordered: Bool) {
        for chunks in chunkings(Self.call) {
            let result = process(chunks, ordered: ordered)
            #expect(result.text.isEmpty, "\(chunks)")
            #expect(result.calls == [Self.weatherInParis], "\(chunks)")
            #expect(result.rejections.isEmpty, "\(chunks)")
            #expect(result.recovered == 0, "\(chunks)")
        }
    }

    @Test("Detection holds with a large tool list and CRLF line breaks")
    func largeToolListAndCRLF() {
        let tools =
            Self.tools
            + (0 ..< 500).map { index -> [String: any Sendable] in
                [
                    "type": "function",
                    "function": ["name": "tool_\(index)"] as [String: any Sendable],
                ]
            }
        let text = "Checking.\r\n" + Self.call.replacingOccurrences(of: "\n", with: "\r\n")
        for chunks in [[text], text.map(String.init)] {
            let result = process(chunks, tools: tools, ordered: true)
            #expect(result.text == "Checking.\r\n", "\(chunks)")
            #expect(result.calls == [Self.weatherInParis], "\(chunks)")
        }
    }

    @Test("Prose and reasoning before the call stay text", arguments: [false, true])
    func precedingTextIsPreserved(ordered: Bool) {
        let prefixes = ["Let me check.\n", "<think>get_weather\n{}</think>\n", "\n"]
        for prefix in prefixes {
            for chunks in chunkings(prefix + Self.call) {
                let result = process(chunks, ordered: ordered)
                #expect(result.text == prefix, "\(chunks)")
                #expect(result.calls == [Self.weatherInParis], "\(chunks)")
            }
        }
    }

    @Test("Text after the call and a second call are both handled", arguments: [false, true])
    func trailingTextAndSecondCall(ordered: Bool) {
        let text = Self.call + "\nget_time {}\nDone."
        for chunks in chunkings(text) {
            let result = process(chunks, ordered: ordered)
            #expect(result.text == "\n\nDone.", "\(chunks)")
            #expect(
                result.calls == [
                    Self.weatherInParis, ToolCall.Function(name: "get_time", arguments: [:]),
                ], "\(chunks)")
        }
    }

    @Test("Schema types are applied to markerless arguments")
    func argumentsAreNormalized() throws {
        let result = process(
            ["get_weather\n{\"location\": \"Paris\", \"days\": \"3\"}"], ordered: true)
        let call = try #require(result.calls.first)
        #expect(call.arguments["days"] == .int(3))
    }

    @Test("Text that only resembles a call remains text", arguments: [false, true])
    func lookalikesRemainText(ordered: Bool) {
        let lookalikes = [
            "get_weather is the tool I would use.",
            "Use get_weather\n{\"location\": \"Paris\"} for this.",
            "unknown_tool\n{\"location\": \"Paris\"}",
            "get_weather\n{not json}",
            "get_weather\n[\"Paris\"]",
            "```\nget_weather\n{\"location\": \"Paris\"}\n```",
            "`get_weather\n{\"location\": \"Paris\"}`",
            "get_weather\n{\"location\": \"Paris\"",
        ]
        for text in lookalikes {
            for chunks in [[text], text.map(String.init)] {
                let result = process(chunks, ordered: ordered)
                #expect(result.text == text, "\(chunks)")
                #expect(result.calls.isEmpty, "\(chunks)")
                #expect(result.rejections.isEmpty, "\(chunks)")
            }
        }
    }

    @Test("Without declared tools nothing anchors the dialect", arguments: [false, true])
    func requiresDeclaredTools(ordered: Bool) {
        for tools in [nil, []] as [[[String: any Sendable]]?] {
            let result = process([Self.call], tools: tools, ordered: ordered)
            #expect(result.text == Self.call)
            #expect(result.calls.isEmpty)
        }
    }

    @Test("Other formats do not adopt the dialect", arguments: [false, true])
    func otherFormatsUnaffected(ordered: Bool) {
        for format: ToolCallFormat in [.json, .lfm2, .mistral, .xmlFunction, .qwen35, .llama3] {
            let result = process([Self.call], format: format, ordered: ordered)
            #expect(result.text == Self.call, "\(format)")
            #expect(result.calls.isEmpty, "\(format)")
        }
    }

    @Test("The framed arg_key dialect is untouched")
    func argKeyDialectStillStreams() {
        let text =
            "<tool_call>get_weather<arg_key>location</arg_key><arg_value>Paris</arg_value></tool_call>"
        for chunks in chunkings(text) {
            let result = process(chunks, ordered: true)
            #expect(result.text.isEmpty, "\(chunks)")
            #expect(result.calls.map(\.name) == ["get_weather"], "\(chunks)")
        }
    }

    @Test("A processor is reusable after a markerless call")
    func reusableAfterEOS() {
        let processor = ToolCallProcessor(format: .glm4, tools: Self.tools)
        for _ in 0 ..< 2 {
            _ = processor.processChunk(Self.call)
            processor.processEOS()
            #expect(processor.drainToolCalls().map(\.function) == [Self.weatherInParis])
        }
    }
}
