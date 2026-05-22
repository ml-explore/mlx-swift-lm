// Copyright © 2025 Apple Inc.

import Foundation

/// Parser for Gemma format: call:name{key:value,k:<escape>str<escape>}
/// Supports both Gemma 3 (<start_function_call>/<end_function_call>) and
/// Gemma 4 (<|tool_call>/<tool_call|>) formats.
/// Reference: https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/tool_parsers/function_gemma.py
public struct GemmaFunctionParser: ToolCallParser, Sendable {
    // Gemma 4 format tags (primary, as Gemma 4 is the current model)
    public let startTag: String? = "<|tool_call>"
    public let endTag: String? = "<tool_call|>"

    // Gemma 3 format tags (fallback detection)
    private static let gemma3StartTag = "<start_function_call>"
    private static let gemma3EndTag = "<end_function_call>"

    // Gemma 3 escape marker
    private let escapeMarker = "<escape>"
    // Gemma 4 escape marker
    private static let gemma4EscapeMarker = "<|\"|>"

    public init() {}

    /// Detects which Gemma format is being used based on content
    private func detectFormat(in text: String) -> (isGemma4: Bool, stripped: String) {
        // Try Gemma 4 tags first (startTag/endTag)
        if let st = startTag, let et = endTag, text.contains(st) || text.contains(et) {
            var stripped = text
            stripped = stripped.replacingOccurrences(of: st, with: "")
            stripped = stripped.replacingOccurrences(of: et, with: "")
            return (true, stripped)
        }
        // Fallback to Gemma 3 tags
        var stripped = text
        stripped = stripped.replacingOccurrences(of: Self.gemma3StartTag, with: "")
        stripped = stripped.replacingOccurrences(of: Self.gemma3EndTag, with: "")
        return (false, stripped)
    }

    public func parse(content: String, tools: [[String: any Sendable]]?) -> ToolCall? {
        // Detect format and strip tags
        let (isGemma4, text) = detectFormat(in: content)

        // Pattern: call:(\w+)\{(.*?)\}
        // Find "call:" followed by function name and arguments in braces
        guard let callRange = text.range(of: "call:") else { return nil }

        let remaining = String(text[callRange.upperBound...])

        // Extract function name (word characters until {)
        guard let braceStart = remaining.firstIndex(of: "{") else { return nil }
        let funcName = String(remaining[..<braceStart])

        guard !funcName.isEmpty else { return nil }

        // Extract arguments string (everything between { and })
        guard let braceEnd = remaining.lastIndex(of: "}") else { return nil }
        var argsStr = String(remaining[remaining.index(after: braceStart) ..< braceEnd])

        var arguments: [String: any Sendable] = [:]

        // Use correct escape marker based on detected format
        let escMarker = isGemma4 ? Self.gemma4EscapeMarker : escapeMarker

        // Parse key:value pairs
        while !argsStr.isEmpty {
            // Find the key (everything before :)
            guard let colonIdx = argsStr.firstIndex(of: ":") else { break }
            let key = String(argsStr[..<colonIdx])
            argsStr = String(argsStr[argsStr.index(after: colonIdx)...])

            // Handle escaped strings
            var parsedValue: String?
            if argsStr.hasPrefix(escMarker) {
                argsStr = String(argsStr.dropFirst(escMarker.count))
                guard let endEscape = argsStr.range(of: escMarker) else { break }
                parsedValue = String(argsStr[..<endEscape.lowerBound])
                argsStr = String(argsStr[endEscape.upperBound...])
            }

            if let pv = parsedValue {
                arguments[key] = pv
                // Skip comma if present
                if argsStr.hasPrefix(",") {
                    argsStr = String(argsStr.dropFirst())
                }
                continue
            }

            // Handle regular values (until comma or end)
            let commaIdx = argsStr.firstIndex(of: ",") ?? argsStr.endIndex
            let rawValue = String(argsStr[..<commaIdx])
            argsStr =
                commaIdx < argsStr.endIndex
                ? String(argsStr[argsStr.index(after: commaIdx)...]) : ""

            // Try JSON decode, fallback to string
            if let data = rawValue.data(using: .utf8),
                let json = deserializeJSON(data)
            {
                arguments[key] = json
            } else {
                arguments[key] = rawValue
            }
        }

        return ToolCall(function: .init(name: funcName, arguments: arguments))
    }
}
