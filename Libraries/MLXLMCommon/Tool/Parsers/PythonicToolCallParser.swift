// Copyright © 2025 Apple Inc.

import Foundation

/// Parser for Pythonic tool call format: [function_name(arg1='value1', arg2='value2')]
/// Used by LFM2.5 and similar models that output tool calls in Python function call syntax.
/// Reference: LiquidAI LFM2.5 chat template format
public struct PythonicToolCallParser: ToolCallParser, Sendable {
    public let startTag: String?
    public let endTag: String?

    public init(startTag: String? = nil, endTag: String? = nil) {
        self.startTag = startTag
        self.endTag = endTag
    }

    public func parse(content: String, tools: [[String: any Sendable]]?) -> ToolCall? {
        var text = content

        // Strip tags if present
        if let start = startTag, let startRange = text.range(of: start) {
            text = String(text[startRange.upperBound...])
        }
        if let end = endTag, let endRange = text.range(of: end) {
            text = String(text[..<endRange.lowerBound])
        }

        text = text.trimmingCharacters(in: .whitespacesAndNewlines)

        // Pattern: [function_name(args...)] or function_name(args...)
        // Also handle multiple calls: [func1(args), func2(args)]
        let pattern = #"\[?(\w+)\((.*?)\)\]?"#

        guard
            let regex = try? NSRegularExpression(
                pattern: pattern, options: [.dotMatchesLineSeparators]),
            let match = regex.firstMatch(
                in: text, options: [], range: NSRange(text.startIndex..., in: text))
        else { return nil }

        // Extract function name
        guard let nameRange = Range(match.range(at: 1), in: text) else { return nil }
        let funcName = String(text[nameRange])

        // Extract arguments string
        guard let argsRange = Range(match.range(at: 2), in: text) else { return nil }
        let argsString = String(text[argsRange])

        // Parse arguments
        let arguments = parseArguments(argsString, funcName: funcName, tools: tools)

        return ToolCall(function: .init(name: funcName, arguments: arguments))
    }

    /// Parse Pythonic keyword arguments: arg1='value1', arg2="value2", arg3=123
    private func parseArguments(
        _ argsString: String,
        funcName: String,
        tools: [[String: any Sendable]]?
    ) -> [String: any Sendable] {
        var arguments: [String: any Sendable] = [:]

        // Pattern for key=value pairs, handling quoted strings with possible commas inside
        // This handles: key='value', key="value", key=123, key=True, key=None
        let argPattern = #"(\w+)\s*=\s*('(?:[^'\\]|\\.)*'|"(?:[^"\\]|\\.)*"|[^,\)]+)"#

        guard let regex = try? NSRegularExpression(pattern: argPattern, options: []) else {
            return arguments
        }

        let matches = regex.matches(
            in: argsString, options: [], range: NSRange(argsString.startIndex..., in: argsString))

        for match in matches {
            guard let keyRange = Range(match.range(at: 1), in: argsString),
                let valueRange = Range(match.range(at: 2), in: argsString)
            else { continue }

            let key = String(argsString[keyRange])
            var value = String(argsString[valueRange]).trimmingCharacters(in: .whitespaces)

            // Remove surrounding quotes if present
            if (value.hasPrefix("'") && value.hasSuffix("'"))
                || (value.hasPrefix("\"") && value.hasSuffix("\""))
            {
                value = String(value.dropFirst().dropLast())
                // Unescape escaped quotes
                value = value.replacingOccurrences(of: "\\'", with: "'")
                value = value.replacingOccurrences(of: "\\\"", with: "\"")
                value = value.replacingOccurrences(of: "\\\\", with: "\\")
            }

            // Convert value based on schema type if available
            arguments[key] = convertParameterValue(
                value, paramName: key, funcName: funcName, tools: tools)
        }

        return arguments
    }
}
