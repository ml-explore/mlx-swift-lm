// Copyright © 2025 Apple Inc.

import Foundation

/// Parser for XML function format: <function=name><parameter=key>value</parameter></function>
/// Reference: https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/tool_parsers/qwen3_coder.py
///
/// Structure is validated with the shared `QwenXMLPayloadScanner`, the same
/// grammar used by ``Qwen35ToolCallParser`` and by cross-dialect recovery. A
/// payload becomes a call only when it is canonical in full:
///
/// - the function close is the *structural* close rather than the first textual
///   `</function>`, so a parameter value may contain that literal verbatim;
/// - every `<parameter=` must close with `</parameter>`;
/// - nothing but whitespace may follow the payload inside its frame.
///
/// A malformed payload yields `nil` instead of a partially populated call. This
/// matters most at end of stream, where a truncated frame must be rejected
/// rather than executed with silently missing arguments — structural validity
/// must never depend on whether the caller's schema declares `required`.
public struct XMLFunctionParser: ToolCallParser, Sendable {
    public let startTag: String?
    public let endTag: String?

    public init(startTag: String, endTag: String) {
        self.startTag = startTag
        self.endTag = endTag
    }

    public func parse(content: String, tools: [[String: any Sendable]]?) -> ToolCall? {
        guard
            let payload = QwenXMLPayloadScanner.framedPayload(
                in: content, startTag: startTag, endTag: endTag),
            payload.hasPrefix(QwenXMLPayloadScanner.functionOpen)
        else { return nil }

        return QwenXMLPayloadScanner.parseCanonical(payload[...], tools: tools)
    }
}
