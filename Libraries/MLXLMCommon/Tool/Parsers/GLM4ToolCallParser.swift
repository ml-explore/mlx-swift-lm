// Copyright © 2025 Apple Inc.

import Foundation

/// Parser for the GLM4 tool-call dialects.
///
/// GLM-4.5 and later frame a call as
/// `<tool_call>func<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>`.
/// GLM-4-0414 writes the function name on its own line followed by a JSON
/// arguments object, with no delimiters at all. `ToolCallProcessor` anchors
/// that markerless dialect on the declared tools before it reaches this parser.
///
/// Reference: https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/tool_parsers/glm47.py
public struct GLM4ToolCallParser: ToolCallParser, Sendable {
    public let startTag: String? = "<tool_call>"
    public let endTag: String? = "</tool_call>"
    public var supportsMarkerlessNamedJSON: Bool { true }

    public init() {}

    public func parse(content: String, tools: [[String: any Sendable]]?) -> ToolCall? {
        // Strip wrapper tags only at the boundaries so literal tag strings in
        // JSON arguments remain unchanged.
        var text = content.trimmingCharacters(in: .whitespacesAndNewlines)
        if let start = startTag, text.hasPrefix(start) {
            text = String(text.dropFirst(start.count))
                .trimmingCharacters(in: .whitespacesAndNewlines)
        }
        if let end = endTag, text.hasSuffix(end) {
            text = String(text.dropLast(end.count))
                .trimmingCharacters(in: .whitespacesAndNewlines)
        }

        // Extract function name (everything before first <arg_key>)
        guard let argKeyStart = text.range(of: "<arg_key>") else {
            return parseNamedJSON(text, tools: tools)
        }
        let funcName = String(text[..<argKeyStart.lowerBound]).trimmingCharacters(
            in: .whitespacesAndNewlines)

        guard !funcName.isEmpty else { return nil }

        var arguments: [String: any Sendable] = [:]

        // Find all arg_key/arg_value pairs
        var searchRange = text.startIndex ..< text.endIndex
        while let keyStart = text.range(of: "<arg_key>", range: searchRange) {
            // Find </arg_key>
            guard
                let keyEnd = text.range(
                    of: "</arg_key>", range: keyStart.upperBound ..< text.endIndex)
            else { break }

            let key = String(text[keyStart.upperBound ..< keyEnd.lowerBound])
                .trimmingCharacters(in: .whitespacesAndNewlines)

            // Find <arg_value> after </arg_key>
            guard
                let valueStart = text.range(
                    of: "<arg_value>", range: keyEnd.upperBound ..< text.endIndex)
            else { break }

            // Find </arg_value>
            guard
                let valueEnd = text.range(
                    of: "</arg_value>", range: valueStart.upperBound ..< text.endIndex)
            else { break }

            let value = String(text[valueStart.upperBound ..< valueEnd.lowerBound])
                .trimmingCharacters(in: .whitespacesAndNewlines)

            // GLM4: deserialize if NOT a string type in schema
            if !isStringType(funcName: funcName, argName: key, tools: tools) {
                arguments[key] = tryParseJSON(value) ?? value
            } else {
                arguments[key] = value
            }

            searchRange = valueEnd.upperBound ..< text.endIndex
        }

        return ToolCall(function: .init(name: funcName, arguments: arguments))
    }

    /// Parses `name\n{json}`. The payload must be exactly one call and, when a
    /// schema is supplied, name one of its tools.
    private func parseNamedJSON(_ text: String, tools: [[String: any Sendable]]?) -> ToolCall? {
        guard let call = NamedJSONCallScanner.split(text[...]),
            isDeclaredTool(String(call.name), tools: tools),
            let arguments = tryParseJSON(String(call.arguments)) as? [String: any Sendable]
        else { return nil }
        return ToolCall(function: .init(name: String(call.name), arguments: arguments))
    }
}
