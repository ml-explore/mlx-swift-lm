// Copyright © 2025 Apple Inc.

import Foundation

/// Estimates the minimum token reserve needed to force-complete a valid JSON
/// instance of a given schema.
public enum CompletionReserve {

    // MARK: - Public API

    /// Synthesizes the shortest valid JSON for the schema, tokenizes it,
    /// and returns the token count.
    ///
    /// Falls back to `defaultReserve` if the schema cannot be parsed
    /// or contains unsupported constructs.
    ///
    /// - Parameters:
    ///   - schemaJSON: Raw JSON schema string (e.g., `{"type":"string"}`)
    ///   - tokenizer: Tokenizer to count tokens of the minimal JSON
    ///   - defaultReserve: Fallback value on parse failure (default 64)
    /// - Returns: Estimated token count for forced completion
    public static func estimate(
        schemaJSON: String, tokenizer: any Tokenizer, defaultReserve: Int = 64
    ) -> Int {
        guard let data = schemaJSON.data(using: .utf8),
            let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            let minimal = synthesizeMinimalJSON(json, defs: json["$defs"] as? [String: Any] ?? [:])
        else {
            return defaultReserve
        }
        let tokens = tokenizer.encode(text: minimal)
        return tokens.count
    }

    // MARK: - Private

    private static func synthesizeMinimalJSON(
        _ schema: [String: Any],
        defs: [String: Any],
        visited: Set<String> = []
    ) -> String? {
        // $ref resolution: resolve from the root $defs dictionary
        if let ref = schema["$ref"] as? String {
            guard let defName = refName(ref),
                !visited.contains(defName),
                let defSchema = defs[defName] as? [String: Any]
            else {
                return nil
            }
            return synthesizeMinimalJSON(defSchema, defs: defs, visited: visited.union([defName]))
        }

        // Enum takes priority over type-based synthesis
        if let enumValues = schema["enum"] as? [Any], let first = enumValues.first {
            return jsonEncode(first)
        }

        // anyOf / oneOf: use first alternative
        if let alternatives = (schema["anyOf"] ?? schema["oneOf"]) as? [[String: Any]],
            let first = alternatives.first
        {
            return synthesizeMinimalJSON(first, defs: defs, visited: visited)
        }

        guard let type = schema["type"] as? String else {
            return nil
        }

        switch type {
        case "string":
            return "\"\""
        case "integer", "number":
            return "0"
        case "boolean":
            return "false"
        case "null":
            return "null"
        case "object":
            guard let required = schema["required"] as? [String],
                let properties = schema["properties"] as? [String: Any],
                !required.isEmpty
            else {
                return "{}"
            }
            var parts: [String] = []
            for key in required {
                guard let propSchema = properties[key] as? [String: Any],
                    let value = synthesizeMinimalJSON(propSchema, defs: defs, visited: visited)
                else {
                    return nil
                }
                parts.append("\"\(key)\":\(value)")
            }
            return "{\(parts.joined(separator: ","))}"
        case "array":
            let minItems = schema["minItems"] as? Int ?? 0
            guard minItems > 0,
                let itemSchema = schema["items"] as? [String: Any],
                let itemJSON = synthesizeMinimalJSON(itemSchema, defs: defs, visited: visited)
            else {
                return "[]"
            }
            let elements = Array(repeating: itemJSON, count: minItems)
            return "[\(elements.joined(separator: ","))]"
        default:
            return nil
        }
    }

    /// Extract the definition name from a `#/$defs/Name` reference string.
    private static func refName(_ ref: String) -> String? {
        let prefix = "#/$defs/"
        guard ref.hasPrefix(prefix) else { return nil }
        return String(ref.dropFirst(prefix.count))
    }

    /// JSON-encode a single value from a parsed JSON schema enum.
    private static func jsonEncode(_ value: Any) -> String? {
        guard
            let data = try? JSONSerialization.data(
                withJSONObject: value, options: .fragmentsAllowed),
            let str = String(data: data, encoding: .utf8)
        else {
            return nil
        }
        return str
    }
}
