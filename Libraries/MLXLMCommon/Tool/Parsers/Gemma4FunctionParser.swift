// Copyright © 2025 Apple Inc.

import Foundation

/// Parser for Gemma 4 format: `<|tool_call>call:name{key:value,k:<|"|>str<|"|>}<tool_call|>`
///
/// Differs from Gemma 1–3 (`GemmaFunctionParser`) in two ways:
/// - Tool-call tokens are `<|tool_call>` / `<tool_call|>` (asymmetric `|` placement),
///   matching `stc_token` / `etc_token` in Gemma 4's tokenizer config.
/// - The escape marker around string values is `<|"|>` (Gemma 4's `escape_token`)
///   rather than `<escape>`.
public struct Gemma4FunctionParser: ToolCallParser, Sendable {
    public let startTag: String? = "<|tool_call>"
    public let endTag: String? = "<tool_call|>"

    private let escapeMarker = "<|\"|>"

    public init() {}

    public func parse(content: String, tools: [[String: any Sendable]]?) -> ToolCall? {
        var text = content
        if let start = startTag {
            text = text.replacingOccurrences(of: start, with: "")
        }
        if let end = endTag {
            text = text.replacingOccurrences(of: end, with: "")
        }

        guard let callRange = text.range(of: "call:") else { return nil }
        let remaining = String(text[callRange.upperBound...])

        guard let braceStart = remaining.firstIndex(of: "{") else { return nil }
        let funcName = String(remaining[..<braceStart])
        guard !funcName.isEmpty else { return nil }

        guard let braceEnd = remaining.lastIndex(of: "}") else { return nil }
        var argsStr = String(remaining[remaining.index(after: braceStart) ..< braceEnd])

        var arguments: [String: any Sendable] = [:]

        while !argsStr.isEmpty {
            guard let colonIdx = argsStr.firstIndex(of: ":") else { break }
            let key = String(argsStr[..<colonIdx])
            argsStr = String(argsStr[argsStr.index(after: colonIdx)...])

            if argsStr.hasPrefix(escapeMarker) {
                argsStr = String(argsStr.dropFirst(escapeMarker.count))
                guard let endEscape = argsStr.range(of: escapeMarker) else { break }
                let value = String(argsStr[..<endEscape.lowerBound])
                arguments[key] = value
                argsStr = String(argsStr[endEscape.upperBound...])
                if argsStr.hasPrefix(",") {
                    argsStr = String(argsStr.dropFirst())
                }
                continue
            }

            let commaIdx = argsStr.firstIndex(of: ",") ?? argsStr.endIndex
            let value = String(argsStr[..<commaIdx])
            argsStr =
                commaIdx < argsStr.endIndex
                ? String(argsStr[argsStr.index(after: commaIdx)...]) : ""

            if let data = value.data(using: .utf8),
                let json = deserializeJSON(data)
            {
                arguments[key] = json
            } else {
                arguments[key] = value
            }
        }

        return ToolCall(function: .init(name: funcName, arguments: arguments))
    }
}
