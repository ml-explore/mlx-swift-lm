// Copyright © 2026 Apple Inc.

import Foundation
import Testing

@testable import MLXLMCommon

/// Schema-driven conversion of textual XML parameter values must agree with
/// what `ToolSchemaValidator` later judges. A value that the declared schema
/// can only accept as a non-string type has to be converted before
/// validation, or a healable call is wrongly rejected.
@Suite("XML parameter value coercion")
struct ParameterCoercionTests {
    private static func envelopeTools(
        parameterSchema: [String: any Sendable]
    ) -> [[String: any Sendable]] {
        [
            [
                "function": [
                    "name": "weather",
                    "parameters": [
                        "type": "object",
                        "properties": ["hour": parameterSchema],
                        "required": ["hour"],
                    ] as [String: any Sendable],
                ] as [String: any Sendable]
            ]
        ]
    }

    private static func flatTools(
        parameterSchema: [String: any Sendable]
    ) -> [[String: any Sendable]] {
        [
            [
                "name": "weather",
                "parameters": [
                    "type": "object",
                    "properties": ["hour": parameterSchema],
                    "required": ["hour"],
                ] as [String: any Sendable],
            ]
        ]
    }

    @Test("Union types without string coerce the textual wire value")
    func unionTypeCoercion() {
        let tools = Self.envelopeTools(parameterSchema: ["type": ["integer", "null"]])

        let converted = convertParameterValue(
            "5", paramName: "hour", funcName: "weather", tools: tools)
        #expect(converted as? Int == 5)

        let null = convertParameterValue(
            "null", paramName: "hour", funcName: "weather", tools: tools)
        #expect(null is NSNull)

        // Unconvertible values stay textual so validation reports the model's
        // actual violation instead of a fabricated value.
        let unconvertible = convertParameterValue(
            "Paris", paramName: "hour", funcName: "weather", tools: tools)
        #expect(unconvertible as? String == "Paris")
    }

    @Test("Unions that admit a string preserve the wire value")
    func unionWithStringStaysString() {
        let tools = Self.envelopeTools(parameterSchema: ["type": ["string", "integer"]])

        let converted = convertParameterValue(
            "5", paramName: "hour", funcName: "weather", tools: tools)
        #expect(converted as? String == "5")
    }

    @Test("anyOf branch types drive conversion")
    func anyOfCoercion() {
        let tools = Self.envelopeTools(
            parameterSchema: [
                "anyOf": [["type": "integer"], ["type": "null"]] as [[String: any Sendable]]
            ])

        let converted = convertParameterValue(
            "7", paramName: "hour", funcName: "weather", tools: tools)
        #expect(converted as? Int == 7)
    }

    @Test("Flat tool declarations coerce like the OpenAI envelope")
    func flatShapeCoercion() {
        let tools = Self.flatTools(parameterSchema: ["type": "integer"])

        let converted = convertParameterValue(
            "5", paramName: "hour", funcName: "weather", tools: tools)
        #expect(converted as? Int == 5)
    }

    @Test("Schemas without recognizable type information stay textual")
    func schemalessValueIsUnchanged() {
        let tools = Self.envelopeTools(parameterSchema: ["description": "an hour"])

        let converted = convertParameterValue(
            "5", paramName: "hour", funcName: "weather", tools: tools)
        #expect(converted as? String == "5")
    }

    @Test("Recovered XML calls with union-typed parameters become executable")
    func recoveredUnionTypedCallIsExecutable() throws {
        let tools = Self.envelopeTools(parameterSchema: ["type": ["integer", "null"]])
        let processor = ToolCallProcessor(format: .json, tools: tools)

        _ = processor.processChunk("<function=weather><parameter=hour>5</parameter></function>")

        let call = try #require(processor.toolCalls.first)
        #expect(processor.rejectedToolCalls.isEmpty)
        #expect(call.function.name == "weather")
        #expect(call.function.arguments["hour"] == .int(5))
    }

    @Test("Native Qwen XML calls with union-typed parameters become executable")
    func nativeUnionTypedCallIsExecutable() throws {
        let tools = Self.envelopeTools(parameterSchema: ["type": ["integer", "null"]])
        let processor = ToolCallProcessor(format: .qwen35, tools: tools)

        let text =
            "<tool_call><function=weather><parameter=hour>7</parameter></function></tool_call>"
        for character in text {
            _ = processor.processChunk(String(character))
        }

        let call = try #require(processor.toolCalls.first)
        #expect(processor.rejectedToolCalls.isEmpty)
        #expect(call.function.arguments["hour"] == .int(7))
    }

    @Test("Definite violations still reject after coercion")
    func definiteViolationsStillReject() {
        let tools = Self.envelopeTools(parameterSchema: ["type": ["integer", "null"]])
        let processor = ToolCallProcessor(format: .json, tools: tools)

        _ = processor.processChunk(
            "<function=weather><parameter=hour>Paris</parameter></function>")

        #expect(processor.toolCalls.isEmpty)
        #expect(processor.rejectedToolCalls.first?.reason == .invalidArguments)
    }
}
