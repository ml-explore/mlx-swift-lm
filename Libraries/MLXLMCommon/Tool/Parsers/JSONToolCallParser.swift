// Copyright © 2025 Apple Inc.

import Foundation

/// Wire-side decoder that accepts both shapes the upstream models emit:
///   * arguments as a nested object: `{"name":"f","arguments":{"k":"v"}}`
///   * arguments as a JSON-encoded string (Granite 4 etc.):
///     `{"name":"f","arguments":"{\"k\":\"v\"}"}`
private struct WireFunction: Decodable {
    let name: String
    let arguments: [String: JSONValue]

    private enum CodingKeys: String, CodingKey {
        case name
        case arguments
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.name = try c.decode(String.self, forKey: .name)

        if c.contains(.arguments) {
            if let nested = try? c.decode([String: JSONValue].self, forKey: .arguments) {
                self.arguments = nested
            } else if let stringified = try? c.decode(String.self, forKey: .arguments) {
                guard let data = stringified.data(using: .utf8),
                    let nested = try? JSONDecoder().decode(
                        [String: JSONValue].self, from: data)
                else {
                    throw DecodingError.dataCorruptedError(
                        forKey: .arguments, in: c,
                        debugDescription:
                            "arguments is a String but does not contain a JSON object")
                }
                self.arguments = nested
            } else {
                throw DecodingError.dataCorruptedError(
                    forKey: .arguments, in: c,
                    debugDescription: "arguments must be an object or a JSON-encoded string")
            }
        } else {
            self.arguments = [:]
        }
    }
}

/// Parser for JSON format: <tag>{"name": "...", "arguments": {...}}</tag>
/// Reference: https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/tool_parsers/default.py
public struct JSONToolCallParser: ToolCallParser, Sendable {
    public let startTag: String?
    public let endTag: String?

    public init(startTag: String, endTag: String) {
        self.startTag = startTag
        self.endTag = endTag
    }

    public func parse(content: String, tools: [[String: any Sendable]]?) -> ToolCall? {
        guard let start = startTag, let end = endTag else { return nil }

        // Find the JSON content between tags
        var text = content

        // Strip tags if present
        if let startRange = text.range(of: start) {
            text = String(text[startRange.upperBound...])
        }
        if let endRange = text.range(of: end) {
            text = String(text[..<endRange.lowerBound])
        }

        let jsonStr = text.trimmingCharacters(in: .whitespacesAndNewlines)

        guard let data = jsonStr.data(using: .utf8),
            let wire = try? JSONDecoder().decode(WireFunction.self, from: data)
        else { return nil }

        return ToolCall(
            function: ToolCall.Function(name: wire.name, arguments: wire.arguments))
    }
}
