// Copyright © 2025 Apple Inc.

import Foundation

/// Parser for OpenAI Harmony format.
///
/// Format: `<|start|>assistant<|channel|>commentary to=functions.name <|constrain|>json<|message|>{JSON}<|call|>`
///
/// The function name follows `to=` with an optional namespace prefix (e.g. `functions.`),
/// and arguments are JSON-encoded between `<|message|>` and `<|call|>`.
///
/// Reference: https://developers.openai.com/cookbook/articles/openai-harmony
public struct HarmonyToolCallParser: ToolCallParser, Sendable {
    public let startTag: String? = "<|start|>assistant<|channel|>commentary to="
    public let endTag: String? = "<|call|>"

    public init() {}

    public func parse(content: String, tools: [[String: any Sendable]]?) -> ToolCall? {
        var text = content

        // Strip start tag if present
        if let start = startTag, let startRange = text.range(of: start) {
            text = String(text[startRange.upperBound...])
        }

        // Strip end tag if present
        if let end = endTag, let endRange = text.range(of: end) {
            text = String(text[..<endRange.lowerBound])
        }

        // text is now: "functions.func_name <|constrain|>json<|message|>{...}"
        // Extract function name — ends at the first space or '<'
        let funcNameEnd = text.firstIndex(where: { $0 == " " || $0 == "<" }) ?? text.endIndex
        var funcName = String(text[..<funcNameEnd])

        // Strip namespace prefix (e.g. "functions.")
        if let dotIndex = funcName.firstIndex(of: ".") {
            funcName = String(funcName[funcName.index(after: dotIndex)...])
        }

        guard !funcName.isEmpty else { return nil }

        // Extract JSON arguments between <|message|> and end of text
        guard let messageRange = text.range(of: "<|message|>") else { return nil }
        let argsStr = String(text[messageRange.upperBound...])
            .trimmingCharacters(in: .whitespacesAndNewlines)

        guard let arguments = deserialize(argsStr) as? [String: any Sendable] else {
            return nil
        }

        return ToolCall(function: .init(name: funcName, arguments: arguments))
    }
}
