// Copyright © 2025 Apple Inc.

import Foundation

/// Parser for JSON format: <tag>{"name": "...", "arguments": {...}}</tag>
/// Reference: https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/tool_parsers/default.py
public struct JSONToolCallParser: ToolCallParser, Sendable {
    public let startTag: String?
    public let endTag: String?
    private let jsonObjectScanner = JSONLeadingObjectScanner(startCharacter: "{")

    public init(startTag: String, endTag: String) {
        self.startTag = startTag
        self.endTag = endTag
    }

    public func parse(content: String, tools: [[String: any Sendable]]?) -> ToolCall? {
        guard let start = startTag, let end = endTag else { return nil }

        // Find the JSON content between tags
        var text = content

        // Strip tags if present
        if let startRange = text.range(of: start) {
            text = String(text[startRange.upperBound...])
        }
        if let endRange = text.range(of: end) {
            text = String(text[..<endRange.lowerBound])
        }

        let jsonStr = text.trimmingCharacters(in: .whitespacesAndNewlines)

        let toolCall = parseToolCall(from: jsonStr) ?? parseRedundantOuterBraces(from: jsonStr)
        guard let toolCall else { return nil }

        // If tool schemas are provided, only accept calls to declared tools.
        if let tools, !tools.isEmpty {
            var isDeclaredTool = false
            for tool in tools {
                let functionSpec = tool["function"] as? [String: any Sendable]
                if functionSpec?["name"] as? String == toolCall.function.name {
                    isDeclaredTool = true
                    break
                }
            }

            guard isDeclaredTool else {
                return nil
            }
        }

        return toolCall
    }

    /// Some Qwen chat templates emit an EOS-delimited JSON call with a
    /// redundant leading brace and one or two redundant closing braces.
    /// Recover only that exact shape and only when the enclosed prefix is one
    /// complete, valid tool-call object followed solely by those braces.
    private func parseRedundantOuterBraces(from text: String) -> ToolCall? {
        guard text.hasPrefix("{{") else { return nil }
        let withoutLeadingBrace = String(text.dropFirst())
        guard let split = jsonObjectScanner.splitLeadingObject(from: withoutLeadingBrace) else {
            return nil
        }
        let trailing = split.trailing.trimmingCharacters(in: .whitespacesAndNewlines)
        guard (1 ... 2).contains(trailing.count), trailing.allSatisfy({ $0 == "}" }) else {
            return nil
        }
        return parseToolCall(from: split.object)
    }

    private func parseToolCall(from text: String) -> ToolCall? {
        guard let data = text.data(using: .utf8) else { return nil }
        return parseToolCall(from: data)
    }

    private func parseToolCall(from data: Data) -> ToolCall? {
        guard var jsonObject = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else {
            return nil
        }

        var id = jsonObject["id"] as? String
        if let functionObject = jsonObject["function"] as? [String: Any] {
            id = id ?? functionObject["id"] as? String
            jsonObject = functionObject
        }

        if let stringifiedArguments = jsonObject["arguments"] as? String {
            guard
                let argumentsData = stringifiedArguments.data(using: .utf8),
                let argumentsObject = try? JSONSerialization.jsonObject(with: argumentsData)
                    as? [String: Any]
            else { return nil }
            jsonObject["arguments"] = argumentsObject
        }

        guard
            let normalizedData = try? JSONSerialization.data(withJSONObject: jsonObject),
            let function = try? JSONDecoder().decode(ToolCall.Function.self, from: normalizedData)
        else { return nil }

        return ToolCall(function: function, id: id)
    }
}
