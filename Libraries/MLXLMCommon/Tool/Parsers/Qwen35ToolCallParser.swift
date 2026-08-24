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

    private let xmlParser: XMLFunctionParser
    private let jsonParser: JSONToolCallParser

    public init(startTag: String, endTag: String) {
        self.startTag = startTag
        self.endTag = endTag
        self.xmlParser = XMLFunctionParser(startTag: startTag, endTag: endTag)
        self.jsonParser = JSONToolCallParser(startTag: startTag, endTag: endTag)
    }

    public func parse(content: String, tools: [[String: any Sendable]]?) -> ToolCall? {
        guard let payload = payloadBody(in: content) else { return nil }

        let call: ToolCall?
        if payload.hasPrefix("<function=") {
            guard isCanonicalXMLPayload(payload) else { return nil }
            call = xmlParser.parse(content: payload, tools: tools)
        } else if payload.hasPrefix("{") {
            call = jsonParser.parse(content: payload, tools: tools)
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
        var payload = content.trimmingCharacters(in: .whitespacesAndNewlines)

        if let startTag, payload.hasPrefix(startTag) {
            payload.removeFirst(startTag.count)
            payload = payload.trimmingCharacters(in: .whitespacesAndNewlines)
        }
        if let endTag {
            if payload.hasSuffix(endTag) {
                payload.removeLast(endTag.count)
                payload = payload.trimmingCharacters(in: .whitespacesAndNewlines)
            } else if payload.contains(endTag) {
                // The streaming processor normally separates trailing content.
                // Direct parser callers must not have it silently discarded.
                return nil
            }
        }

        return payload
    }

    /// Validate the whole XML body before delegating value conversion to the
    /// existing parser. This prevents a recognizable prefix plus unrelated or
    /// mixed-dialect text from becoming an executable call.
    private func isCanonicalXMLPayload(_ payload: String) -> Bool {
        var remainder = payload[...]
        guard consumeNamedOpeningTag("<function=", from: &remainder) else { return false }

        while true {
            remainder = remainder.drop(while: { $0.isWhitespace })
            if remainder.hasPrefix("</function>") {
                remainder = remainder.dropFirst("</function>".count)
                return remainder.allSatisfy(\.isWhitespace)
            }

            guard consumeNamedOpeningTag("<parameter=", from: &remainder),
                let parameterEnd = remainder.range(of: "</parameter>")
            else { return false }
            remainder = remainder[parameterEnd.upperBound...]
        }
    }

    private func consumeNamedOpeningTag(
        _ prefix: String, from text: inout Substring
    ) -> Bool {
        guard text.hasPrefix(prefix) else { return false }
        text = text.dropFirst(prefix.count)
        guard let closingAngle = text.firstIndex(of: ">") else { return false }

        let name = text[..<closingAngle]
        guard !name.isEmpty, !name.contains(where: \.isWhitespace) else { return false }
        text = text[text.index(after: closingAngle)...]
        return true
    }
}
