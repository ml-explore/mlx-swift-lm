// Copyright © 2026 Apple Inc.

import Foundation
import Testing

@testable import MLXLMCommon

struct ToolCallRecoveryRegressionTests {
    private let tools: [[String: any Sendable]] = [
        ["function": ["name": "weather"] as [String: any Sendable]]
    ]
    private let call = #"<tool_call>{"name":"weather","arguments":{"city":"Paris"}}</tool_call>"#

    @Test("Native syntax survives recovery under every policy and chunk boundary")
    func nativeSyntax() {
        let cases: [(ToolCallFormat, String, String)] = [
            (
                .glm4,
                "<tool_call>weather<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call>",
                ""
            ),
            (.glm4, "weather\n{\"city\":\"Paris\"}", ""),
            (.glm4, "Checking.\nweather \n{\"city\":\"Paris\"}", "Checking.\n"),
            (.llama3, #"{"name":"weather","parameters":{"city":"Paris"}}"#, ""),
            (.json, "The 12\" model. " + call, "The 12\" model. "),
            (.json, "„Ready\". " + call, "„Ready\". "),
            (.json, "[for example, see below. " + call, "[for example, see below. "),
            (.json, "[not an array. " + call, "[not an array. "),
        ]
        for (format, text, prefix) in cases {
            let characters = Array(text)
            for policy in ToolCallRecoveryPolicy.allCases {
                for split in 0 ... characters.count {
                    let chunks = [String(characters[..<split]), String(characters[split...])]
                    let processor = ToolCallProcessor(
                        format: format, tools: tools, toolCallPolicy: .init(recovery: policy))
                    let visible =
                        chunks.compactMap { processor.processChunk($0) }.joined()
                        + (processor.processEOS(returnBufferedText: true) ?? "")
                    #expect(visible == prefix, "\(format), \(policy), split \(split)")
                    #expect(processor.toolCalls.count == 1)
                    #expect(
                        processor.toolCalls.first?.function.arguments["city"] == .string("Paris"))
                    #expect(processor.rejectedToolCalls.isEmpty)
                    #expect(processor.recoveredToolCallCount == 0)

                    let ordered = ToolCallProcessor(
                        format: format, tools: tools, toolCallPolicy: .init(recovery: policy))
                    let outputs =
                        chunks.flatMap { ordered.processChunkOutputs($0) }
                        + ordered.processEOSOutputs()
                    #expect(
                        outputs.compactMap {
                            if case .response(let text) = $0 { text } else { nil }
                        }.joined() == prefix)
                    #expect(
                        outputs.filter { if case .toolCall = $0 { true } else { false } }.count == 1
                    )
                    #expect(
                        outputs.last.map { if case .toolCall = $0 { true } else { false } } == true)
                }
            }
        }
    }

    @Test("Structured data remains inert, with an executable call only after the value")
    func structuredData() throws {
        let payloads: [Any] = [
            call, ["example": call], [true, false, NSNull(), 1.25e-8, call],
            ["nested": [["quote": "\\\"🦋", "example": call]]],
        ]
        for payload in payloads {
            let data = try JSONSerialization.data(
                withJSONObject: payload, options: [.fragmentsAllowed, .sortedKeys])
            let text = String(decoding: data, as: UTF8.self)
            for format: ToolCallFormat in [.json, .llama3, .lfm2] {
                let processor = ToolCallProcessor(format: format, tools: tools)
                var outputs: [ToolCallProcessor.Output] = []
                for character in text + " " + call {
                    outputs += processor.processChunkOutputs(String(character))
                }
                outputs += processor.processEOSOutputs()
                #expect(
                    outputs.compactMap { if case .response(let text) = $0 { text } else { nil } }
                        .joined() == text + " ")
                #expect(
                    outputs.filter { if case .toolCall = $0 { true } else { false } }.count == 1)
            }
        }
    }

    @Test("JSON syntax tracking rejects impossible prefixes and bounds nesting")
    func jsonPrefixes() {
        for text in ["[for", "[nullx", "[01", "[1e+z", "{\"a\" 1", "[true,]", "{\"a\":1]", "\"\\q"]
        {
            var scanner = JSONPrefixScanner()
            if case .invalid = scanner.scan(text) {
            } else {
                Issue.record("Accepted impossible prefix: \(text)")
            }
        }
        var scanner = JSONPrefixScanner()
        if case .depthLimit = scanner.scan(String(repeating: "[", count: 257)) {
        } else {
            Issue.record("Missing nesting bound")
        }
    }

    @Test("EOS clears prose context before a new generation")
    func reuseAfterEOS() throws {
        let processor = ToolCallProcessor(format: .lfm2, tools: tools)
        _ = processor.processChunk("word")
        processor.processEOS()
        let data = try JSONSerialization.data(withJSONObject: call, options: .fragmentsAllowed)
        let quoted = String(decoding: data, as: UTF8.self)
        let visible =
            (processor.processChunk(quoted) ?? "")
            + (processor.processEOS(returnBufferedText: true) ?? "")
        #expect(visible == quoted)
        #expect(processor.toolCalls.isEmpty)
    }
}
