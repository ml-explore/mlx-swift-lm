// Copyright © 2026 Apple Inc.

import Foundation
import Testing

@testable import MLXLMCommon

@Suite(.serialized)
struct ToolCallPolicyTests {
    @Test(
        "Generation policies have independent value semantics and permissive validation by default")
    func defaultsAndCopies() {
        let defaults = GenerateParameters()
        #expect(defaults.toolCallPolicy == .init(recovery: .conservative, validation: .permissive))
        var custom = defaults
        custom.toolCallPolicy.recovery = .disabled
        custom.toolCallPolicy.validation = .strict
        #expect(defaults.toolCallPolicy == ToolCallPolicy())
        #expect(custom.toolCallPolicy == .init(recovery: .disabled, validation: .strict))
        #expect(
            GenerateParameters(toolCallPolicy: custom.toolCallPolicy).toolCallPolicy
                == custom.toolCallPolicy)
    }

    @Test("Default processing forwards schema violations but still rejects undeclared tools")
    func defaultProcessing() {
        let tools: [[String: any Sendable]] = [
            [
                "function": [
                    "name": "read",
                    "parameters": [
                        "type": "object", "properties": ["value": ["type": "integer"]],
                        "required": ["value"],
                    ] as [String: any Sendable],
                ] as [String: any Sendable]
            ]
        ]
        for arguments: [String: JSONValue] in [["value": .string("six")], [:]] {
            let call = ToolCall(function: .init(name: "read", arguments: arguments))
            let json = arguments.isEmpty ? "{}" : #"{"value":"six"}"#
            for format: ToolCallFormat in [.json, .lfm2] {
                let processor = ToolCallProcessor(format: format, tools: tools)
                _ = processor.processChunk(
                    "<tool_call>{\"name\":\"read\",\"arguments\":\(json)}</tool_call>")
                processor.processEOS()
                #expect(processor.toolCalls.first?.function == call.function)
                #expect(processor.rejectedToolCalls.isEmpty)
            }
        }
        let processor = ToolCallProcessor(format: .json, tools: tools)
        _ = processor.processChunk(#"<tool_call>{"name":"other","arguments":{}}</tool_call>"#)
        processor.processEOS()
        #expect(processor.toolCalls.isEmpty)
        #expect(processor.rejectedToolCalls.first?.reason == .undeclaredTool)
    }

    @Test(
        "Generation and token recording preserve both policies and rejection telemetry",
        arguments: ToolCallRecoveryPolicy.allCases, ToolCallValidationPolicy.allCases)
    func generation(recovery: ToolCallRecoveryPolicy, validation: ToolCallValidationPolicy)
        async throws
    {
        let policy = ToolCallPolicy(recovery: recovery, validation: validation)
        let parameters: [String: any Sendable] = [
            "type": "object", "properties": ["value": ["type": "integer"]],
            "required": ["value"],
        ]
        let tools: [[String: any Sendable]] = [
            ["function": ["name": "read", "parameters": parameters] as [String: any Sendable]]
        ]
        for native in [true, false] {
            for value in ["6", "six"] {
                for recordTokens in [false, true] {
                    let text =
                        native
                        ? "<tool_call>{\"name\":\"read\",\"arguments\":{\"value\":\"\(value)\"}}</tool_call>"
                        : "<function=read><parameter=value>\(value)</parameter></function>"
                    let tokenizer = PolicyTokenizer()
                    let tokens = tokenizer.encode(text: text)
                    let iterator = PolicyIterator(tokens: tokens)
                    let configuration = ModelConfiguration(id: "policy-test", toolCallFormat: .json)
                    var events: [Generation] = []
                    if recordTokens {
                        let (stream, task) = generateTaskRecordingTokens(
                            promptTokenCount: 0, modelConfiguration: configuration,
                            tokenizer: tokenizer, iterator: iterator, tools: tools,
                            toolCallPolicy: policy)
                        for await event in stream { events.append(event) }
                        let recordedTokens = (await task.value).consume().generatedTokens
                        #expect(recordedTokens == tokens)
                    } else {
                        let (stream, task) = generateTask(
                            promptTokenCount: 0, modelConfiguration: configuration,
                            tokenizer: tokenizer, iterator: iterator, tools: tools,
                            toolCallPolicy: policy)
                        for await event in stream { events.append(event) }
                        await task.value
                    }
                    let parsed = native || recovery != .disabled
                    let accepted = parsed && (value == "6" || validation == .permissive)
                    let calls = events.compactMap(\.toolCall)
                    let rejected = events.compactMap(\.rejectedToolCall)
                    #expect(calls.count == (accepted ? 1 : 0))
                    if accepted {
                        #expect(
                            calls.first?.function.arguments["value"]
                                == (value == "6" ? .int(6) : .string(value)))
                    }
                    #expect(rejected.count == (parsed && !accepted ? 1 : 0))
                    #expect(rejected.allSatisfy { $0.reason == .invalidArguments })
                    let info = try #require(events.compactMap(\.info).last)
                    #expect(info.rejectedToolCallCount == rejected.count)
                    #expect(info.recoveredToolCallCount == (!native && accepted ? 1 : 0))
                }
            }
        }
    }
}

/// One Unicode scalar per token makes every syntax boundary a streaming boundary.
private struct PolicyTokenizer: Tokenizer {
    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        text.unicodeScalars.map { Int($0.value) }
    }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        String(String.UnicodeScalarView(tokenIds.compactMap(Unicode.Scalar.init)))
    }
    func convertTokenToId(_ token: String) -> Int? { nil }
    func convertIdToToken(_ id: Int) -> String? { Unicode.Scalar(id).map(String.init) }
    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }
    func applyChatTemplate(
        messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] { [] }
}

private struct PolicyIterator: TokenIteratorProtocol {
    let tokens: [Int]
    var tokenCount = 0
    var maxTokens: Int? { nil }
    var promptPrefillTime: TimeInterval { 0 }
    mutating func next() -> Int? {
        guard tokenCount < tokens.count else { return nil }
        defer { tokenCount += 1 }
        return tokens[tokenCount]
    }
}
