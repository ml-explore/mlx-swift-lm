// Copyright © 2026 Apple Inc.

import Foundation
import Testing

@testable import MLXLMCommon

@Suite("Text tool-call recovery")
struct TextToolCallRecoveryTests {
    private static let tools: [[String: any Sendable]] = [
        [
            "function": [
                "name": "weather"
            ] as [String: any Sendable]
        ]
    ]

    private struct Dialect {
        let format: ToolCallFormat
        let text: String
    }

    private static let alternateDialects = [
        Dialect(
            format: .lfm2,
            text: #"<tool_call>{"name":"weather","arguments":{"city":"Paris"}}</tool_call>"#),
        Dialect(
            format: .json,
            text: #"[TOOL_CALLS]weather{"city":"Paris"}"#),
        Dialect(
            format: .json,
            text: #"weather[ARGS]{"city":"Paris"}"#),
        Dialect(
            format: .json,
            text: "<function=weather><parameter=city>Paris</parameter></function>"),
        Dialect(
            format: .json,
            text: #"<|tool_call>call:weather{city:<|"|>Paris<|"|>}<tool_call|>"#),
    ]

    private func responseText(_ outputs: [ToolCallProcessor.Output]) -> String {
        outputs.compactMap { output in
            if case .response(let text) = output { return text }
            return nil
        }.joined()
    }

    @Test("Every supported alternate dialect promotes a declared call")
    func promotesAlternateDialects() throws {
        for dialect in Self.alternateDialects {
            let processor = ToolCallProcessor(format: dialect.format, tools: Self.tools)
            _ = processor.processChunk(dialect.text)

            let call = try #require(processor.toolCalls.first, "dialect: \(dialect.text)")
            #expect(processor.toolCalls.count == 1, "dialect: \(dialect.text)")
            #expect(call.function.name == "weather")
            #expect(call.function.arguments["city"] == .string("Paris"))
            #expect(processor.recoveredToolCallCount == 1)
            #expect(processor.recoveryEvents.count == 1)
        }
    }

    @Test("Recovery is invariant at every streaming boundary")
    func everyStreamingBoundary() {
        for dialect in Self.alternateDialects {
            let characters = Array(dialect.text)
            for split in 1 ..< characters.count {
                let processor = ToolCallProcessor(format: dialect.format, tools: Self.tools)
                _ = processor.processChunk(String(characters[..<split]))
                _ = processor.processChunk(String(characters[split...]))

                #expect(
                    processor.toolCalls.count == 1,
                    "dialect: \(dialect.text), split: \(split)")
                #expect(processor.toolCalls.first?.function.name == "weather")
            }
        }
    }

    @Test("Mistral heals the no-ARGS variant at EOS")
    func mistralNoArgsAtEOS() throws {
        let processor = ToolCallProcessor(format: .mistral, tools: Self.tools)
        #expect(processor.processChunk(#"[TOOL_CALLS]weather{"city":"Paris"}"#) == nil)
        processor.processEOS()

        let call = try #require(processor.toolCalls.first)
        #expect(call.function.name == "weather")
        #expect(call.function.arguments["city"] == .string("Paris"))
    }

    @Test("Native frames are opaque to alternate markers in argument strings")
    func nativeFrameIsOpaque() throws {
        let processor = ToolCallProcessor(format: .qwen35, tools: Self.tools)
        let text =
            #"<tool_call>{"name":"weather","arguments":{"city":"literal <function=fake> and weather[ARGS]{}"}}</tool_call>"#

        for character in text {
            _ = processor.processChunk(String(character))
        }

        let call = try #require(processor.toolCalls.first)
        #expect(processor.toolCalls.count == 1)
        #expect(
            call.function.arguments["city"]
                == .string("literal <function=fake> and weather[ARGS]{}"))
    }

    @Test("Llama's native inline marker remains attached to its JSON payload")
    func llamaInlineMarkerIsOpaque() {
        let processor = ToolCallProcessor(format: .llama3, tools: Self.tools)
        let text =
            #"<|python_tag|>{"name":"weather","arguments":{"city":"Paris"}}"#

        #expect(processor.processChunk(text) == nil)
        #expect(processor.toolCalls.count == 1)
        #expect(processor.toolCalls.first?.function.name == "weather")
    }

    @Test("Protocol examples inside an ordinary JSON string remain response text")
    func ordinaryJSONIsOpaque() {
        let processor = ToolCallProcessor(format: .lfm2, tools: Self.tools)
        let text =
            #"{"example":"weather[ARGS]{\"city\":\"Paris\"} and <function=weather><parameter=city>Paris</parameter></function>"}"#

        #expect(processor.processChunk(text) == text)
        #expect(processor.toolCalls.isEmpty)

        let nativeJSON = ToolCallProcessor(format: .json, tools: Self.tools)
        let nativeMarkerInData =
            #"{"example":"<tool_call>{\"name\":\"weather\",\"arguments\":{\"city\":\"Paris\"}}</tool_call>"}"#
        #expect(nativeJSON.processChunk(nativeMarkerInData) == nativeMarkerInData)
        #expect(nativeJSON.toolCalls.isEmpty)
    }

    @Test("Native tool syntax inside reasoning is inert response text")
    func reasoningCannotDispatchNativeCall() {
        let processor = ToolCallProcessor(format: .json, tools: Self.tools)
        let text =
            #"<think>consider <tool_call>{"name":"weather","arguments":{"city":"Paris"}}</tool_call> carefully</think>"#
        let outputs = processor.processChunkOutputs(
            text)

        #expect(responseText(outputs) == text)
        #expect(processor.toolCalls.isEmpty)
        #expect(processor.rejectedToolCalls.isEmpty)
    }

    @Test("Malformed explicit candidates are rejected and never executable")
    func malformedCandidatesAreRejected() {
        let processor = ToolCallProcessor(format: .json, tools: Self.tools)
        let undeclared = #"other[ARGS]{"city":"Paris"}"#
        let malformed = "<function=weather>not closed"

        #expect(processor.processChunk(undeclared) == undeclared)
        let visible =
            (processor.processChunk(malformed) ?? "")
            + (processor.processEOS(returnBufferedText: true) ?? "")
        #expect(visible == "not closed")
        #expect(processor.toolCalls.isEmpty)
        #expect(processor.rejectedToolCalls.count == 1)
        #expect(processor.rejectedToolCalls.first?.reason == .malformedSyntax)
    }

    @Test("Recovery is disabled without a nonempty declared-tool allowlist")
    func requiresDeclaredTools() {
        let text = #"weather[ARGS]{"city":"Paris"}"#
        let unconstrained = ToolCallProcessor(format: .json)
        let empty = ToolCallProcessor(format: .json, tools: [])

        #expect(unconstrained.processChunk(text) == text)
        #expect(empty.processChunk(text) == text)
        #expect(unconstrained.toolCalls.isEmpty)
        #expect(empty.toolCalls.isEmpty)
    }

    @Test("Markerless names require an exact lexical boundary across chunks")
    func markerlessLexicalBoundary() {
        let processor = ToolCallProcessor(format: .json, tools: Self.tools)
        let first = processor.processChunk("not")
        let second = processor.processChunk(#"weather[ARGS]{"city":"Paris"}"#)

        #expect((first ?? "") + (second ?? "") == #"notweather[ARGS]{"city":"Paris"}"#)
        #expect(processor.toolCalls.isEmpty)
    }

    @Test("Incomplete recovery buffers are bounded")
    func boundedBuffer() {
        let processor = ToolCallProcessor(format: .json, tools: Self.tools)
        // Combining scalars form very few extended grapheme clusters, so this
        // verifies the limit is a byte limit rather than `String.count`.
        let text =
            "<function=weather><parameter=city>"
            + String(repeating: "\u{0301}", count: 40_000)

        #expect(processor.processChunk(text) == nil)
        #expect(processor.toolCalls.isEmpty)
        #expect(processor.rejectedToolCalls.first?.reason == .resourceLimitExceeded)

        // Do not resume execution from the middle of the oversized attempt.
        _ = processor.processChunk(#"weather[ARGS]{"city":"Paris"}"#)
        #expect(processor.toolCalls.isEmpty)

        processor.processEOS()
        _ = processor.processChunk(#"weather[ARGS]{"city":"Paris"}"#)
        #expect(processor.toolCalls.count == 1)
    }

    @Test("Complete oversized calls cannot bypass the byte limit")
    func completeOversizedCallsAreRejected() {
        let alternate = ToolCallProcessor(format: .json, tools: Self.tools)
        let hugeValue = String(repeating: "x", count: 66_000)
        let alternateText =
            "<function=weather><parameter=city>" + hugeValue
            + "</parameter></function>"

        _ = alternate.processChunk(alternateText)
        #expect(alternate.toolCalls.isEmpty)
        #expect(alternate.rejectedToolCalls.first?.reason == .resourceLimitExceeded)

        let native = ToolCallProcessor(format: .json, tools: Self.tools)
        let nativeText =
            #"<tool_call>{"name":"weather","arguments":{"city":""# + hugeValue
            + #""}}</tool_call>"#
        _ = native.processChunk(nativeText)
        #expect(native.toolCalls.isEmpty)
        #expect(native.rejectedToolCalls.first?.reason == .resourceLimitExceeded)
    }

    @Test("EOS clears an unfinished native-frame shield before processor reuse")
    func eosClearsNativeFrameShield() {
        let processor = ToolCallProcessor(format: .json, tools: Self.tools)
        #expect(processor.processChunk("<tool_call>") == nil)
        processor.processEOS()

        _ = processor.processChunk(
            "<function=weather><parameter=city>Paris</parameter></function>")
        #expect(processor.toolCalls.count == 1)
        #expect(processor.toolCalls.first?.function.name == "weather")
    }

    @Test("Native tool syntax inside Markdown code is inert")
    func markdownCodeCannotDispatchNativeCall() {
        let processor = ToolCallProcessor(format: .json, tools: Self.tools)
        let text =
            "```json\n"
            + #"<tool_call>{"name":"weather","arguments":{"city":"Paris"}}</tool_call>"#
            + "\n```\n"

        let outputs = processor.processChunkOutputs(text)
        #expect(responseText(outputs) == text)
        #expect(processor.toolCalls.isEmpty)
    }

    @Test("Native markers inside JSON arrays and strings are inert")
    func jsonDataCannotDispatchNativeCall() {
        let processor = ToolCallProcessor(format: .mistral, tools: Self.tools)
        let text = #"["[TOOL_CALLS]weather[ARGS]{\"city\":\"Paris\"}"]"#

        #expect(processor.processChunk(text) == text)
        processor.processEOS()
        #expect(processor.toolCalls.isEmpty)
    }

    @Test("Oversized JSON remains fail-closed until EOS")
    func oversizedJSONCannotReenterExecutableContext() {
        let processor = ToolCallProcessor(format: .lfm2, tools: Self.tools)
        let prefix = "[\"" + String(repeating: "a", count: 65_536)

        #expect(processor.processChunk(prefix) == prefix)
        let candidate = #"weather[ARGS]{"city":"Paris"}"#
        #expect(processor.processChunk(candidate) == candidate)
        #expect(processor.toolCalls.isEmpty)

        processor.processEOS()
        _ = processor.processChunk(candidate)
        #expect(processor.toolCalls.count == 1)
    }

    @Test("Qwen JSON preserves literal outer close markers in arguments")
    func qwenJSONLiteralCloseMarker() throws {
        let processor = ToolCallProcessor(format: .qwen35, tools: Self.tools)
        let text =
            #"<tool_call>{"name":"weather","arguments":{"city":"literal </tool_call> remains data"}}</tool_call>"#

        for character in text {
            _ = processor.processChunk(String(character))
        }

        let call = try #require(processor.toolCalls.first)
        #expect(call.function.arguments["city"] == .string("literal </tool_call> remains data"))
        #expect(processor.rejectedToolCalls.isEmpty)
    }

    @Test("Recovery policy is honored by the public processor path")
    func disabledPolicyDoesNotRecover() {
        let processor = ToolCallProcessor(
            format: .json, tools: Self.tools, recoveryPolicy: .disabled)
        let text = "<function=weather><parameter=city>Paris</parameter></function>"

        #expect(processor.processChunk(text) == text)
        #expect(processor.toolCalls.isEmpty)
        #expect(processor.recoveredToolCallCount == 0)

        let nativeInReasoning =
            #"<think><tool_call>{"name":"weather","arguments":{"city":"Paris"}}</tool_call></think>"#
        #expect(processor.processChunk(nativeInReasoning) == nativeInReasoning)
        #expect(processor.toolCalls.isEmpty)

        _ = processor.processChunk(
            #"<tool_call>{"name":"weather","arguments":{"city":"Paris"}}</tool_call>"#)
        #expect(processor.toolCalls.count == 1)
    }

    @Test("Recovered calls must satisfy required arguments")
    func requiredArgumentsAreEnforced() {
        let tools: [[String: any Sendable]] = [
            [
                "function": [
                    "name": "weather",
                    "parameters": [
                        "type": "object",
                        "properties": ["city": ["type": "string"]],
                        "required": ["city"],
                    ] as [String: any Sendable],
                ] as [String: any Sendable]
            ]
        ]
        let processor = ToolCallProcessor(format: .json, tools: tools)

        _ = processor.processChunk("<function=weather></function>")
        #expect(processor.toolCalls.isEmpty)
        #expect(processor.rejectedToolCalls.count == 1)
        #expect(processor.rejectedToolCalls.first?.reason == .invalidArguments)
        #expect(processor.rejectedToolCalls.first?.detail == "arguments.city is required")
        #expect(processor.recoveredToolCallCount == 0)
        #expect(processor.recoveryEvents.isEmpty)

        let native = ToolCallProcessor(format: .json, tools: tools)
        _ = native.processChunk(
            #"<tool_call>{"name":"weather","arguments":{}}</tool_call>"#)
        #expect(native.toolCalls.isEmpty)
        #expect(native.rejectedToolCalls.first?.reason == .invalidArguments)
        #expect(native.rejectedToolCalls.first?.detail == "arguments.city is required")
    }
}
