// Copyright © 2026 Apple Inc.

import Foundation

/// Parser for MiniCPM5's XML function format:
/// `<function name="f"><param name="k">v</param></function>`
///
/// A param value containing `<`, `&`, or a newline is wrapped in CDATA by the
/// chat template, and the model follows the same convention when generating.
/// Reference: openbmb/MiniCPM5-2B-MLX chat template
public struct MiniCPM5ToolCallParser: ToolCallParser, Sendable {
    public let startTag: String? = "<function name=\""
    public let endTag: String? = "</function>"

    private static let cdataOpen = "<![CDATA["
    private static let cdataClose = "]]>"

    public init() {}

    public func parse(content: String, tools: [[String: any Sendable]]?) -> ToolCall? {
        // Strip the opening `<function name="` tag.
        guard let startTag, content.hasPrefix(startTag) else { return nil }
        let afterStart = content.index(content.startIndex, offsetBy: startTag.count)

        // Read the function name up to the closing quote of the attribute.
        guard
            let nameEnd = content.range(of: "\">", range: afterStart ..< content.endIndex)
        else { return nil }
        let funcName = String(content[afterStart ..< nameEnd.lowerBound])
        guard !funcName.isEmpty else { return nil }

        // Isolate everything between the opening tag and `</function>`.
        guard
            let functionEnd = Self.range(
                ofTag: "</function>", in: content, range: nameEnd.upperBound ..< content.endIndex)
        else { return nil }
        let paramSection = String(content[nameEnd.upperBound ..< functionEnd.lowerBound])

        let paramConfig = getParameterConfig(funcName: funcName, tools: tools)
        var arguments: [String: any Sendable] = [:]

        // Walk each `<param name="...">value</param>` pair in order.
        var searchRange = paramSection.startIndex ..< paramSection.endIndex
        while let paramStart = Self.range(
            ofTag: "<param name=\"", in: paramSection, range: searchRange)
        {
            // A malformed param tag invalidates the whole call rather than silently dropping just that argument.
            guard
                let paramNameEnd = paramSection.range(
                    of: "\">", range: paramStart.upperBound ..< paramSection.endIndex)
            else { return nil }
            let paramName = String(paramSection[paramStart.upperBound ..< paramNameEnd.lowerBound])

            guard
                let paramEnd = Self.range(
                    ofTag: "</param>", in: paramSection,
                    range: paramNameEnd.upperBound ..< paramSection.endIndex)
            else { return nil }

            // Keep a CDATA value exactly as written, minus its markers.
            // Trim a plain value, in case the model adds spaces or newlines around it.
            let valueSlice = paramSection[paramNameEnd.upperBound ..< paramEnd.lowerBound]
            let rawValue: String
            if valueSlice.hasPrefix(Self.cdataOpen), valueSlice.hasSuffix(Self.cdataClose) {
                rawValue = String(
                    valueSlice.dropFirst(Self.cdataOpen.count).dropLast(Self.cdataClose.count))
            } else {
                rawValue = String(valueSlice.trimmingWhitespace())
            }

            // Coerce the value using the tool schema (e.g. string -> number/bool).
            let paramSchema = paramConfig[paramName] as? [String: any Sendable]
            arguments[paramName] =
                ToolArgumentNormalization.normalize(
                    .string(rawValue), schema: paramSchema
                ).sendableValue

            searchRange = paramEnd.upperBound ..< paramSection.endIndex
        }

        return ToolCall(function: .init(name: funcName, arguments: arguments))
    }

    /// Finds the first `tag` in `range` that is outside CDATA, since CDATA content holds no tags.
    /// Returns nil when an unclosed CDATA hides the rest of the text.
    private static func range(
        ofTag tag: String, in text: String, range: Range<String.Index>
    ) -> Range<String.Index>? {
        var searchStart = range.lowerBound
        while let match = text.range(of: tag, range: searchStart ..< range.upperBound) {
            guard let cdata = text.range(of: cdataOpen, range: searchStart ..< match.lowerBound)
            else { return match }
            guard
                let cdataEnd = text.range(
                    of: cdataClose, range: cdata.upperBound ..< range.upperBound)
            else { return nil }
            searchStart = cdataEnd.upperBound
        }
        return nil
    }
}
