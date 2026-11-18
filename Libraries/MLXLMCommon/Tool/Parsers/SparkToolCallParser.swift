// Copyright © 2026 Apple Inc.

import Foundation

/// Parses Spark's framed calls and its `arg_key` / `arg_value` arguments.
public struct SparkToolCallParser: ToolCallParser {
    public let startTag: String? = "<tool_call>"
    public let endTag: String? = "</tool_call>"

    public init() {}

    public func parse(content: String, tools: [[String: any Sendable]]?) -> ToolCall? {
        var text = content.trimmingCharacters(in: .whitespacesAndNewlines)[...]
        if text.hasPrefix("<tool_call>") {
            text = text.dropFirst("<tool_call>".count)
        }
        if text.hasSuffix("</tool_call>") {
            text = text.dropLast("</tool_call>".count)
        }
        let firstArgument = text.range(of: "<arg_key>")
        let name = String(text[..<(firstArgument?.lowerBound ?? text.endIndex)])
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard !name.isEmpty, !name.contains(where: { $0.isWhitespace || $0 == "<" || $0 == ">" })
        else { return nil }

        var arguments: [String: JSONValue] = [:]
        var remaining = text[(firstArgument?.lowerBound ?? text.endIndex)...]
        while !remaining.isEmpty {
            guard remaining.hasPrefix("<arg_key>") else { return nil }
            remaining = remaining.dropFirst("<arg_key>".count)
            guard let keyEnd = remaining.range(of: "</arg_key>") else { return nil }
            let key = String(remaining[..<keyEnd.lowerBound])
            guard !key.isEmpty, arguments[key] == nil else { return nil }
            remaining = remaining[keyEnd.upperBound...]
            guard remaining.hasPrefix("<arg_value>") else { return nil }
            remaining = remaining.dropFirst("<arg_value>".count)
            guard let valueEnd = remaining.range(of: "</arg_value>") else { return nil }
            let value = String(remaining[..<valueEnd.lowerBound])
            if isStringType(funcName: name, argName: key, tools: tools) {
                arguments[key] = .string(value)
            } else {
                arguments[key] =
                    (try? JSONDecoder().decode(JSONValue.self, from: Data(value.utf8)))
                    ?? .string(value)
            }
            remaining = remaining[valueEnd.upperBound...]
        }
        return ToolCall(function: .init(name: name, arguments: arguments))
    }
}
