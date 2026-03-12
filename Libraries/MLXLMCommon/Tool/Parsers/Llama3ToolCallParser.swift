// Copyright © 2025 Apple Inc.

import Foundation

/// Parser for Llama 3 tool calls.
/// Llama 3 often outputs inline JSON without standard start/end tags, or preceded by `<|python_tag|>`.
/// It may also output native python function calls like `get_weather(location="San Francisco")`.
public struct Llama3ToolCallParser: ToolCallParser, Sendable {
    public let startTag: String? = nil
    public let endTag: String? = nil

    public init() {}

    private struct LlamaFunction: Codable {
        let name: String
        let parameters: [String: JSONValue]?
        let arguments: [String: JSONValue]?
    }

    public func parse(content: String, tools: [[String: any Sendable]]?) -> ToolCall? {
        var text = content

        // If it outputs python tag, strip it
        if let range = text.range(of: "<|python_tag|>") {
            text = String(text[range.upperBound...])
        }

        let jsonStr = text.trimmingCharacters(in: .whitespacesAndNewlines)

        // Try JSON format first
        if let data = jsonStr.data(using: .utf8),
            let llamaFunc = try? JSONDecoder().decode(LlamaFunction.self, from: data)
        {
            let args = llamaFunc.parameters ?? llamaFunc.arguments ?? [:]

            let function = ToolCall.Function(
                name: llamaFunc.name,
                arguments: args.mapValues { $0.anyValue }
            )
            return ToolCall(function: function)
        }

        // Fallback to Pythonic format
        let pythonicParser = PythonicToolCallParser()
        return pythonicParser.parse(content: jsonStr, tools: tools)
    }

    public func parseEOS(_ toolCallBuffer: String, tools: [[String: any Sendable]]?) -> [ToolCall] {
        var text = toolCallBuffer

        // If it outputs python tag, strip it
        if let range = text.range(of: "<|python_tag|>") {
            text = String(text[range.upperBound...])
        }

        let jsonStr = text.trimmingCharacters(in: .whitespacesAndNewlines)

        guard let data = jsonStr.data(using: .utf8) else {
            return []
        }

        // Try JSON list format
        if let list = try? JSONDecoder().decode([LlamaFunction].self, from: data) {
            return list.map { llamaFunc in
                let args = llamaFunc.parameters ?? llamaFunc.arguments ?? [:]
                let function = ToolCall.Function(
                    name: llamaFunc.name,
                    arguments: args.mapValues { $0.anyValue }
                )
                return ToolCall(function: function)
            }
        }

        // Try single JSON format
        if let llamaFunc = try? JSONDecoder().decode(LlamaFunction.self, from: data) {
            let args = llamaFunc.parameters ?? llamaFunc.arguments ?? [:]
            let function = ToolCall.Function(
                name: llamaFunc.name,
                arguments: args.mapValues { $0.anyValue }
            )
            return [ToolCall(function: function)]
        }

        // Try Pythonic list like [func1(args), func2(args)] or single func1(args)
        let pythonicParser = PythonicToolCallParser()
        return pythonicParser.parseEOS(jsonStr, tools: tools)
    }
}
