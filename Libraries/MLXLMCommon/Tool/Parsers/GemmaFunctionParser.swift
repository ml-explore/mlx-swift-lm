// Copyright © 2025 Apple Inc.

import Foundation

/// Parser for Gemma format: call:name{key:value,k:<escape>str<escape>}
/// Reference: https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/tool_parsers/function_gemma.py
public struct GemmaFunctionParser: ToolCallParser, Sendable {
    public let startTag: String?
    public let endTag: String?
    public let escapeMarker: String?

    /// Values are either wrapped in the escape marker or written as bare JSON,
    /// so both spans have to be opaque while the argument list is split.
    private let scanner: StructuredTextScanner

    /// Nested objects and arrays are written in the dialect's brace form, whose keys are unquoted.
    private let structuredValues = BareKeyJSONParser()

    public init(startTag: String, endTag: String, escapeMarker: String) {
        self.startTag = startTag
        self.endTag = endTag
        self.escapeMarker = escapeMarker
        self.scanner = StructuredTextScanner(quotes: ["\""], escapeMarker: escapeMarker)
    }

    public func parse(content: String, tools: [[String: any Sendable]]?) -> ToolCall? {
        guard let startTag, let endTag, let marker = escapeMarker else { return nil }

        // Strip tags if present
        var text = content[...]
        if let startRange = text.range(of: startTag) {
            text = text[startRange.upperBound...]
        }
        if let endRange = text.range(of: endTag) {
            text = text[..<endRange.lowerBound]
        }

        // Pattern: call:(\w+)\{(.*)\}, where the closing brace is the one that
        // balances the opening brace rather than the first or last in the text.
        guard let callRange = text.range(of: "call:") else { return nil }
        let remaining = text[callRange.upperBound...]

        guard let braceStart = scanner.firstTopLevelIndex(of: "{", in: remaining),
            let braceEnd = scanner.endOfGroup(in: remaining, openedAt: braceStart)
        else { return nil }

        let funcName = String(remaining[..<braceStart])
        guard !funcName.isEmpty else { return nil }

        let body = remaining[remaining.index(after: braceStart) ..< braceEnd]
        var arguments: [String: any Sendable] = [:]

        // Each field is `key:value`. Splitting at top level keeps a comma that
        // belongs to an escaped string or a nested object out of the split, so a
        // value is never truncated and its remainder never becomes a stray key.
        for field in scanner.splitTopLevel(body, separator: ",") {
            guard let colon = scanner.firstTopLevelIndex(of: ":", in: field) else { continue }
            let key = String(field[..<colon])
            guard !key.isEmpty else { continue }

            let rawValue = field[field.index(after: colon)...].trimmingWhitespace()
            arguments[key] = value(
                of: rawValue, key: key, funcName: funcName, marker: marker, tools: tools)
        }

        return ToolCall(function: .init(name: funcName, arguments: arguments))
    }

    /// Decodes one field value.
    ///
    /// A marker-wrapped value is a string the model escaped precisely because it
    /// may contain protocol punctuation, so it is taken verbatim. Everything else
    /// is typed from the schema when the parameter is declared, and otherwise
    /// decoded as JSON, falling back to the literal text.
    ///
    /// A parameter the schema declares structured is read as a brace-form literal first: the
    /// dialect writes those without quoting their keys, which strict JSON refuses.
    private func value(
        of rawValue: Substring,
        key: String,
        funcName: String,
        marker: String,
        tools: [[String: any Sendable]]?
    ) -> any Sendable {
        if rawValue.hasPrefix(marker) {
            let contentStart = rawValue.index(rawValue.startIndex, offsetBy: marker.count)
            let unescaped =
                rawValue[contentStart...].range(of: marker)
                .map { String(rawValue[contentStart ..< $0.lowerBound]) }
                ?? String(rawValue[contentStart...])
            return convertParameterValue(
                unescaped, paramName: key, funcName: funcName, tools: tools)
        }

        let literal = String(rawValue)
        guard let declaredType = getParameterType(funcName: funcName, paramName: key, tools: tools)
        else {
            return structuredValues.parse(literal) ?? literal
        }

        if Self.isStructured(declaredType), let value = structuredValues.parse(literal) {
            return value
        }
        return convertParameterValue(literal, paramName: key, funcName: funcName, tools: tools)
    }

    /// Whether a declared schema type is one the dialect writes in brace form.
    private static func isStructured(_ type: String) -> Bool {
        let type = type.lowercased()
        return type == "object" || type == "array" || type.hasPrefix("dict")
            || type.hasPrefix("list")
    }
}
