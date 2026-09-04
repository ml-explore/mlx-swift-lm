import Foundation
import MLXLMCommon
import Testing

struct ToolCallProcessorStreamingTests {
    private static let tools: [[String: any Sendable]] = [
        ["function": ["name": "weather"] as [String: any Sendable]]
    ]
    private static let function = ToolCall.Function(
        name: "weather", arguments: ["city": JSONValue.string("Paris")])
    private static let xml =
        "<tool_call><function=weather><parameter=city>Paris</parameter></function></tool_call>"
    private static let json =
        #"<tool_call>{"name":"weather","arguments":{"city":"Paris"}}</tool_call>"#

    private enum Event: Equatable {
        case response(String)
        case call(ToolCall.Function)
        case rejected(RejectedToolCall.Reason)
    }

    private struct Result: Equatable {
        var text = ""
        var calls: [ToolCall.Function] = []
        var rejections: [RejectedToolCall.Reason] = []
        var events: [Event] = []
    }

    // Use nil tools to exercise the native scanner without recovery preprocessing.
    private func process(
        _ chunks: [String], format: ToolCallFormat = .xmlFunction,
        tools: [[String: any Sendable]]? = nil, ordered: Bool
    ) -> Result {
        let processor = ToolCallProcessor(format: format, tools: tools)
        var result = Result()
        if ordered {
            var outputs = chunks.flatMap { processor.processChunkOutputs($0) }
            outputs += processor.processEOSOutputs()
            for output in outputs {
                switch output {
                case .response(let text):
                    result.text += text
                    if case .response(let previous) = result.events.last {
                        result.events[result.events.count - 1] = .response(previous + text)
                    } else {
                        result.events.append(.response(text))
                    }
                case .toolCall(let call):
                    result.calls.append(call.function)
                    result.events.append(.call(call.function))
                case .rejectedToolCall(let rejection):
                    result.rejections.append(rejection.reason)
                    result.events.append(.rejected(rejection.reason))
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
        return result
    }

    private func chunkings(_ text: String) -> [[String]] {
        [[text], text.map(String.init)]
            + text.indices.map { [String(text[..<$0]), String(text[$0...])] }
            + [[text, ""]]
    }

    @Test("A non-tool marker must not hide a later native call", arguments: [false, true])
    func ordinaryMarkerBeforeCall(ordered: Bool) {
        let result = process(["</think>\n" + Self.xml + "done"], ordered: ordered)
        #expect(result.text == "</think>\ndone")
        #expect(result.calls == [Self.function])
        #expect(result.rejections.isEmpty)
        if ordered {
            #expect(
                result.events == [.response("</think>\n"), .call(Self.function), .response("done")])
        }
    }

    @Test("False openers preserve text and calls at every split", arguments: [false, true])
    func falseOpenersAtEverySplit(ordered: Bool) {
        let dialects: [(ToolCallFormat, String)] = [
            (.xmlFunction, Self.xml), (.json, Self.json), (.qwen35, Self.xml),
        ]
        for (format, call) in dialects {
            for tools in [nil, Self.tools] {
                for prefix in [
                    "</think>\n", "a < b < c\n", "<note>hi</note>", "<", "<t", "🧪 e\u{301} < b\n",
                ] {
                    let text = prefix + call + "after"
                    for chunks in chunkings(text) {
                        let result = process(chunks, format: format, tools: tools, ordered: ordered)
                        #expect(result.text == prefix + "after", "\(format): \(chunks)")
                        #expect(result.calls == [Self.function], "\(format): \(chunks)")
                        #expect(result.rejections.isEmpty, "\(format): \(chunks)")
                        if ordered {
                            #expect(
                                result.events == [
                                    .response(prefix), .call(Self.function), .response("after"),
                                ])
                        }
                    }
                }
            }
        }
    }

    @Test("Text before a partial marker is emitted immediately", arguments: [false, true])
    func partialMarkerRetainsOnlyTheCandidate(ordered: Bool) {
        let processor = ToolCallProcessor(format: .xmlFunction)
        let prefix = "before < b "
        if ordered {
            let leading = processor.processChunkOutputs(prefix + "<tool_")
            #expect(leading.allSatisfy { if case .response = $0 { true } else { false } })
            #expect(
                leading.compactMap { if case .response(let text) = $0 { text } else { nil } }
                    .joined() == prefix)
            let outputs = processor.processChunkOutputs(String(Self.xml.dropFirst(6)))
            #expect(outputs.count == 1)
            if case .toolCall(let call) = outputs.first {
                #expect(call.function == Self.function)
            } else {
                Issue.record("Expected the completed call")
            }
            #expect(processor.processEOSOutputs().isEmpty)
        } else {
            #expect(processor.processChunk(prefix + "<tool_") == prefix)
            #expect(processor.processChunk(String(Self.xml.dropFirst(6))) == nil)
            #expect(processor.toolCalls.map(\.function) == [Self.function])
            #expect(processor.processEOS(returnBufferedText: true) == nil)
        }
    }

    @Test("False openers preserve ordinary text and EOS rejection rules", arguments: [false, true])
    func residualTextAndRejections(ordered: Bool) {
        for suffix in ["", "<", "<t", "<tool_call>"] {
            let prefix = "a < b "
            for chunks in chunkings(prefix + suffix) {
                let result = process(chunks, ordered: ordered)
                #expect(result.calls.isEmpty)
                if suffix == "<tool_call>" {
                    #expect(result.text == prefix)
                    #expect(result.rejections == [.incompleteOutput])
                } else {
                    #expect(result.text == prefix + suffix)
                    #expect(result.rejections.isEmpty)
                }
            }
        }
    }

    @Test(
        "Malformed markers and undeclared names still reject before later calls",
        arguments: [false, true])
    func rejectionsBeforeLaterCalls(ordered: Bool) {
        let malformed = process(["<tool_calx>after</tool_call>"], ordered: ordered)
        #expect(malformed.text == "after")
        #expect(malformed.calls.isEmpty)
        #expect(malformed.rejections == [.malformedSyntax])

        for tools in [nil, Self.tools] {
            let result = process(
                ["a < b <tool_calx>between" + Self.xml + "after"], tools: tools, ordered: ordered)
            #expect(result.text == "a < b betweenafter")
            #expect(result.calls == [Self.function])
            #expect(result.rejections == [.malformedSyntax])
            if ordered {
                #expect(
                    result.events == [
                        .response("a < b "), .rejected(.malformedSyntax), .response("between"),
                        .call(Self.function), .response("after"),
                    ])
            }
        }
        for tools in [Self.tools, []] {
            let unknown = Self.xml.replacingOccurrences(of: "weather", with: "unknown")
            for chunks in chunkings("<" + unknown + "between < b " + Self.xml) {
                let result = process(chunks, tools: tools, ordered: ordered)
                #expect(result.text == "<between < b ")
                #expect(result.calls == (tools.isEmpty ? [] : [Self.function]))
                #expect(
                    result.rejections
                        == (tools.isEmpty ? [.undeclaredTool, .undeclaredTool] : [.undeclaredTool]))
            }
        }
    }

    @Test("Many false candidates do not require recursive parsing")
    func manyFalseCandidates() {
        let prefix = String(repeating: "<x ", count: 10_000)
        let result = process([prefix + Self.xml], ordered: true)
        #expect(result.text == prefix)
        #expect(result.calls == [Self.function])
        #expect(result.rejections.isEmpty)
    }
}
