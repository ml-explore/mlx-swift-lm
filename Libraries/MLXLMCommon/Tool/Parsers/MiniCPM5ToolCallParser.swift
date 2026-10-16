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
            let functionEnd = content.range(
                of: "</function>", range: nameEnd.upperBound ..< content.endIndex)
        else { return nil }
        let paramSection = String(content[nameEnd.upperBound ..< functionEnd.lowerBound])

        let paramConfig = getParameterConfig(funcName: funcName, tools: tools)
        var arguments: [String: any Sendable] = [:]

        // Walk each `<param name="...">value</param>` pair in order.
        var searchRange = paramSection.startIndex ..< paramSection.endIndex
        while let paramStart = paramSection.range(of: "<param name=\"", range: searchRange) {
            // A malformed param tag invalidates the whole call rather than silently dropping just that argument.
            guard
                let paramNameEnd = paramSection.range(
                    of: "\">", range: paramStart.upperBound ..< paramSection.endIndex)
            else { return nil }
            let paramName = String(paramSection[paramStart.upperBound ..< paramNameEnd.lowerBound])

            guard
                let paramEnd = paramSection.range(
                    of: "</param>", range: paramNameEnd.upperBound ..< paramSection.endIndex)
            else { return nil }

            // Unwrap CDATA; otherwise trim the whitespace the template pads plain values with.
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
}
