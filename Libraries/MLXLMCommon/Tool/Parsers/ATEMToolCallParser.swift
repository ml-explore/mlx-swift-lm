// Copyright © 2026 Apple Inc.

import Foundation

/// Parser for Muse Glimmer's Onyx ATEM tool-call format.
///
/// ATEM deliberately is not XML: parameter bodies may contain unescaped,
/// multi-line text. Parse only the protocol's structural tags and never run a
/// global namespace replacement over the payload—the latter silently mutates
/// legitimate string arguments containing `<atem:`.
public struct ATEMToolCallParser: ToolCallParser, Sendable {
    public let startTag: String? = "<atem:function_calls>"
    public let endTag: String? = "</atem:function_calls>"

    private static let invokePrefix = "<atem:invoke"
    private static let invokeEnd = "</atem:invoke>"
    private static let parameterPrefix = "<atem:parameter"
    private static let parameterEnd = "</atem:parameter>"

    public init() {}

    public func parse(content: String, tools: [[String: any Sendable]]?) -> ToolCall? {
        guard let startTag, let endTag,
            let outerStart = content.range(of: startTag),
            let outerEnd = content.range(
                of: endTag, range: outerStart.upperBound ..< content.endIndex),
            content[..<outerStart.lowerBound].allSatisfy(\.isWhitespace),
            content[outerEnd.upperBound...].allSatisfy(\.isWhitespace)
        else { return nil }
        let outerBody = String(content[outerStart.upperBound ..< outerEnd.lowerBound])
        guard
            let invoke = element(
                in: outerBody, prefix: Self.invokePrefix, close: Self.invokeEnd),
            invoke.leading.allSatisfy(\.isWhitespace),
            invoke.trailing.allSatisfy(\.isWhitespace),
            let functionName = quotedAttribute("name", in: invoke.openTag),
            !functionName.isEmpty,
            let schema = functionSchema(named: functionName, tools: tools)
        else { return nil }

        var arguments: [String: any Sendable] = [:]
        var remaining = invoke.body[...]
        while let start = structuralTag(Self.parameterPrefix, in: remaining) {
            // Non-whitespace outside parameter elements is malformed protocol.
            guard remaining[..<start.lowerBound].allSatisfy(\.isWhitespace),
                let tagEnd = remaining[start.lowerBound...].firstIndex(of: ">"),
                let close = remaining.range(
                    of: Self.parameterEnd,
                    range: remaining.index(after: tagEnd) ..< remaining.endIndex)
            else { return nil }

            let openTag = String(remaining[start.lowerBound ... tagEnd])
            guard let name = quotedAttribute("name", in: openTag),
                !name.isEmpty,
                arguments[name] == nil
            else { return nil }

            let rawValue = String(remaining[remaining.index(after: tagEnd) ..< close.lowerBound])
            let parameterSchema = schema.properties[name]
            if schema.rejectsUnknownParameters, parameterSchema == nil { return nil }
            guard let value = parsedValue(rawValue, schema: parameterSchema) else { return nil }
            arguments[name] = value
            remaining = remaining[close.upperBound...]
        }

        guard remaining.allSatisfy(\.isWhitespace),
            schema.required.isSubset(of: Set(arguments.keys))
        else { return nil }

        return ToolCall(function: .init(name: functionName, arguments: arguments))
    }

    public func parseEOS(
        _ toolCallBuffer: String, tools: [[String: any Sendable]]?
    ) -> [ToolCall] {
        guard let startTag, let endTag else { return [] }
        var calls: [ToolCall] = []
        var searchStart = toolCallBuffer.startIndex
        while let start = toolCallBuffer.range(
            of: startTag, range: searchStart ..< toolCallBuffer.endIndex),
            let end = toolCallBuffer.range(
                of: endTag, range: start.upperBound ..< toolCallBuffer.endIndex)
        {
            let framed = String(toolCallBuffer[start.lowerBound ..< end.upperBound])
            if let call = parse(content: framed, tools: tools) {
                calls.append(call)
            }
            searchStart = end.upperBound
        }
        return calls
    }

    // MARK: - Structural scanning

    private func element(
        in text: String, prefix: String, close: String
    ) -> (openTag: String, body: String, leading: Substring, trailing: Substring)? {
        guard let start = structuralTag(prefix, in: text[...]),
            let tagEnd = text[start.lowerBound...].firstIndex(of: ">"),
            let end = text.range(
                of: close, range: text.index(after: tagEnd) ..< text.endIndex)
        else { return nil }
        return (
            String(text[start.lowerBound ... tagEnd]),
            String(text[text.index(after: tagEnd) ..< end.lowerBound]),
            text[..<start.lowerBound],
            text[end.upperBound...]
        )
    }

    private func structuralTag(_ prefix: String, in text: Substring) -> Range<String.Index>? {
        var searchStart = text.startIndex
        while let candidate = text.range(of: prefix, range: searchStart ..< text.endIndex) {
            let boundary = candidate.upperBound
            if boundary == text.endIndex || text[boundary].isWhitespace || text[boundary] == ">" {
                return candidate
            }
            searchStart = candidate.upperBound
        }
        return nil
    }

    private func quotedAttribute(_ name: String, in tag: String) -> String? {
        let needle = name + "="
        var searchStart = tag.startIndex
        while let attribute = tag.range(
            of: needle, range: searchStart ..< tag.endIndex)
        {
            let hasBoundary =
                attribute.lowerBound == tag.startIndex
                || tag[tag.index(before: attribute.lowerBound)].isWhitespace
            guard hasBoundary else {
                searchStart = attribute.upperBound
                continue
            }
            let valueStart = attribute.upperBound
            guard valueStart < tag.endIndex else { return nil }
            let quote = tag[valueStart]
            guard quote == "\"" || quote == "'" else { return nil }
            let contentStart = tag.index(after: valueStart)
            guard let contentEnd = tag[contentStart...].firstIndex(of: quote) else { return nil }
            return String(tag[contentStart ..< contentEnd])
        }
        return nil
    }

    // MARK: - Schema-aware values

    private struct FunctionSchema {
        var properties: [String: [String: any Sendable]]
        var required: Set<String>
        var rejectsUnknownParameters: Bool
    }

    private func functionSchema(
        named name: String, tools: [[String: any Sendable]]?
    ) -> FunctionSchema? {
        guard let tools, !tools.isEmpty else { return nil }
        for tool in tools {
            let function =
                (tool["function"] as? [String: any Sendable]) ?? tool
            guard function["name"] as? String == name else { continue }
            let parameters = function["parameters"] as? [String: any Sendable]
            let rawProperties = parameters?["properties"] as? [String: any Sendable] ?? [:]
            let properties = rawProperties.reduce(into: [String: [String: any Sendable]]()) {
                result, entry in
                if let value = entry.value as? [String: any Sendable] {
                    result[entry.key] = value
                }
            }
            return FunctionSchema(
                properties: properties,
                required: Set(parameters?["required"] as? [String] ?? []),
                rejectsUnknownParameters: parameters?["additionalProperties"] as? Bool == false)
        }
        // A generated name which was not offered must never reach dispatch.
        return nil
    }

    private func parsedValue(
        _ rawValue: String, schema: [String: any Sendable]?
    ) -> (any Sendable)? {
        guard let schema else {
            return tryParseJSON(rawValue) ?? rawValue
        }
        let types = extractTypesFromSchema(schema)
        let normalizedTypes = Set(types.map { $0.lowercased() })
        let nonNullTypes = normalizedTypes.subtracting(["null"])
        if !nonNullTypes.isEmpty,
            nonNullTypes.isSubset(of: ["string", "str", "text"])
        {
            let trimmed = rawValue.trimmingCharacters(in: .whitespacesAndNewlines)
            if normalizedTypes.contains("null"),
                ["null", "none", "nil"].contains(trimmed.lowercased())
            {
                return NSNull()
            }
            // The official Onyx contract explicitly preserves string spaces.
            return rawValue
        }

        let trimmed = rawValue.trimmingCharacters(in: .whitespacesAndNewlines)
        let value = convertValueWithTypes(trimmed, types: types)
        let json = JSONValue.from(value)
        guard matches(json, types: normalizedTypes) else { return nil }
        return value
    }

    private func matches(_ value: JSONValue, types: Set<String>) -> Bool {
        switch value {
        case .null: types.contains("null")
        case .bool: types.contains("boolean") || types.contains("bool")
        case .int: types.contains("integer") || types.contains("int") || types.contains("number")
        case .double: types.contains("number") || types.contains("float")
        case .string: types.contains("string") || types.contains("str") || types.contains("text")
        case .array: types.contains("array")
        case .object: types.contains("object")
        }
    }
}
