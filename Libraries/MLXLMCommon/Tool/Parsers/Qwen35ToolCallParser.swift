// Copyright © 2026 Apple Inc.

import Foundation

/// Parser for Qwen 3.5's two observed tool-call payload dialects.
///
/// Qwen 3.5's chat template requests the XML function dialect, but the model can
/// occasionally emit the Qwen/Hermes JSON dialect used by earlier Qwen releases.
/// Both payloads are framed by `<tool_call>` tags. Dialect selection is structural
/// and deterministic: XML must begin with `<function=`, while JSON must begin with
/// `{`. Malformed or ambiguous payloads are never repaired into executable calls.
public struct Qwen35ToolCallParser: ToolCallParser, Sendable {
    public let startTag: String?
    public let endTag: String?

    private let jsonParser: JSONToolCallParser

    public init(startTag: String, endTag: String) {
        self.startTag = startTag
        self.endTag = endTag
        self.jsonParser = JSONToolCallParser(startTag: startTag, endTag: endTag)
    }

    public func parse(content: String, tools: [[String: any Sendable]]?) -> ToolCall? {
        guard let payload = payloadBody(in: content) else { return nil }

        let call: ToolCall?
        if payload.hasPrefix("<function=") {
            call = parseCanonicalXML(payload, tools: tools)
        } else if payload.hasPrefix("{") {
            call = jsonParser.parsePayload(payload)
        } else {
            return nil
        }

        guard let call, isDeclaredTool(call.function.name, tools: tools) else {
            return nil
        }
        return call
    }

    /// Returns the body used only to select a parser. The concrete parser remains
    /// responsible for validating the entire payload.
    private func payloadBody(in content: String) -> String? {
        QwenXMLPayloadScanner.framedPayload(
            in: content, startTag: startTag, endTag: endTag)
    }

    /// Validate and extract a canonical XML payload with the shared structural
    /// scanner. Validation and value extraction use the same grammar, so a
    /// parameter value containing a literal `</function>` cannot truncate the
    /// parse into silently incomplete arguments, and a recognizable prefix plus
    /// unrelated or mixed-dialect text never becomes an executable call.
    private func parseCanonicalXML(
        _ payload: String,
        tools: [[String: any Sendable]]?
    ) -> ToolCall? {
        QwenXMLPayloadScanner.parseCanonical(payload[...], tools: tools)
    }
}
