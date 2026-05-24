// Copyright © 2026 Apple Inc.

import Foundation
import MLXLMCommon

enum OpenAIResponseFormatSupport {
    static func preparedRequest(_ request: OpenAIChatCompletionRequest) throws -> OpenAIChatCompletionRequest {
        guard let responseFormat = request.responseFormat,
            responseFormat.requiresJSONOutput
        else {
            return request
        }

        var prepared = request
        prepared.messages = request.messages.insertingResponseFormatInstruction(
            try instruction(for: responseFormat)
        )
        return prepared
    }

    static func normalizedContent(
        _ content: String,
        for responseFormat: OpenAIResponseFormat?
    ) throws -> String {
        guard let responseFormat,
            responseFormat.requiresJSONOutput
        else {
            return content
        }

        let decoded = try decodeJSONCandidate(from: content, rootMustBeObject: responseFormat.type == .jsonObject)
        switch responseFormat.type {
        case .text:
            return content
        case .jsonObject:
            guard case .object = decoded.value else {
                throw MLXOpenAIServiceError.invalidResponseFormatOutput(
                    "response_format json_object requires a JSON object"
                )
            }
        case .jsonSchema:
            guard let schema = responseFormat.jsonSchema else {
                throw MLXOpenAIServiceError.invalidResponseFormatOutput(
                    "response_format json_schema requires a json_schema payload"
                )
            }
            try JSONSchemaSubsetValidator.validate(decoded.value, schema: schema.schema)
        }

        return decoded.text
    }

    private static func instruction(for responseFormat: OpenAIResponseFormat) throws -> String {
        switch responseFormat.type {
        case .text:
            return ""
        case .jsonObject:
            return """
                You must respond with a single valid JSON object. Do not include markdown, code fences, comments, \
                explanations, or extra text. The first non-whitespace character must be "{" and the final \
                non-whitespace character must be "}".
                """
        case .jsonSchema:
            guard let schema = responseFormat.jsonSchema else {
                throw MLXOpenAIServiceError.invalidResponseFormatOutput(
                    "response_format json_schema requires a json_schema payload"
                )
            }
            let schemaText = try schema.schema.encodedJSONString()
            let strictText = schema.strict == true
                ? " The output must strictly conform to the schema."
                : ""
            let description = schema.description.map { " Description: \($0)" } ?? ""
            let name = schema.name.map { " Schema name: \($0)." } ?? ""
            return """
                You must respond with a single valid JSON value that conforms to the supplied JSON Schema. Do not \
                include markdown, code fences, comments, explanations, or extra text.\(name)\(description)\(strictText)
                JSON Schema:
                \(schemaText)
                """
        }
    }

    private static func decodeJSONCandidate(
        from content: String,
        rootMustBeObject: Bool
    ) throws -> (value: JSONValue, text: String) {
        let candidates = jsonCandidates(from: content, rootMustBeObject: rootMustBeObject)
        for candidate in candidates {
            if let value = try? JSONDecoder().decode(JSONValue.self, from: Data(candidate.utf8)) {
                return (value, candidate)
            }
        }

        throw MLXOpenAIServiceError.invalidResponseFormatOutput(
            "model output was not valid JSON"
        )
    }

    private static func jsonCandidates(from content: String, rootMustBeObject: Bool) -> [String] {
        let trimmed = content.trimmingCharacters(in: .whitespacesAndNewlines)
        var candidates: [String] = []
        if !trimmed.isEmpty {
            candidates.append(trimmed)
        }
        if let unfenced = trimmed.unfencedMarkdownJSON(), !unfenced.isEmpty {
            candidates.append(unfenced)
        }
        if let balanced = trimmed.firstBalancedJSONValue(rootMustBeObject: rootMustBeObject),
            !balanced.isEmpty
        {
            candidates.append(balanced)
        }
        return candidates.uniqued()
    }
}

private struct JSONSchemaSubsetValidator {
    static func validate(_ value: JSONValue, schema: JSONValue) throws {
        guard case .object(let object) = schema else {
            return
        }

        if let enumValues = object["enum"], case .array(let allowedValues) = enumValues,
            !allowedValues.contains(value)
        {
            throw invalid("value is not one of the schema enum values")
        }

        if let constValue = object["const"], constValue != value {
            throw invalid("value does not match schema const")
        }

        if let typeValue = object["type"] {
            try validateType(value, typeValue: typeValue)
        }

        switch value {
        case .object(let valueObject):
            try validateObject(valueObject, schemaObject: object)
        case .array(let values):
            if let itemsSchema = object["items"] {
                for item in values {
                    try validate(item, schema: itemsSchema)
                }
            }
        case .int(let value):
            try validateNumber(Double(value), schemaObject: object)
        case .double(let value):
            try validateNumber(value, schemaObject: object)
        case .string(let value):
            try validateString(value, schemaObject: object)
        case .null, .bool:
            break
        }
    }

    private static func validateObject(
        _ value: [String: JSONValue],
        schemaObject: [String: JSONValue]
    ) throws {
        let properties = schemaObject["properties"]?.objectValue ?? [:]
        let required = schemaObject["required"]?.stringArrayValue ?? []

        for key in required where value[key] == nil {
            throw invalid("required property '\(key)' is missing")
        }

        for (key, propertySchema) in properties {
            if let propertyValue = value[key] {
                try validate(propertyValue, schema: propertySchema)
            }
        }

        if schemaObject["additionalProperties"] == .bool(false) {
            let extraKeys = Set(value.keys).subtracting(properties.keys)
            if let extraKey = extraKeys.sorted().first {
                throw invalid("additional property '\(extraKey)' is not allowed")
            }
        } else if let additionalSchema = schemaObject["additionalProperties"],
            case .object = additionalSchema
        {
            for (key, propertyValue) in value where properties[key] == nil {
                try validate(propertyValue, schema: additionalSchema)
            }
        }
    }

    private static func validateType(_ value: JSONValue, typeValue: JSONValue) throws {
        switch typeValue {
        case .string(let type):
            guard matches(value, type: type) else {
                throw invalid("value does not match schema type '\(type)'")
            }
        case .array(let types):
            let rawTypes = types.compactMap(\.stringValue)
            guard rawTypes.contains(where: { matches(value, type: $0) }) else {
                throw invalid("value does not match any schema type")
            }
        default:
            break
        }
    }

    private static func validateNumber(_ value: Double, schemaObject: [String: JSONValue]) throws {
        if let minimum = schemaObject["minimum"]?.doubleCompatibleValue,
            value < minimum
        {
            throw invalid("number is below schema minimum")
        }
        if let maximum = schemaObject["maximum"]?.doubleCompatibleValue,
            value > maximum
        {
            throw invalid("number is above schema maximum")
        }
    }

    private static func validateString(_ value: String, schemaObject: [String: JSONValue]) throws {
        if let minLength = schemaObject["minLength"]?.intValue,
            value.count < minLength
        {
            throw invalid("string is shorter than schema minLength")
        }
        if let maxLength = schemaObject["maxLength"]?.intValue,
            value.count > maxLength
        {
            throw invalid("string is longer than schema maxLength")
        }
    }

    private static func matches(_ value: JSONValue, type: String) -> Bool {
        switch (value, type) {
        case (.object, "object"), (.array, "array"), (.string, "string"),
            (.bool, "boolean"), (.null, "null"), (.int, "integer"):
            return true
        case (.int, "number"), (.double, "number"):
            return true
        case (.double(let value), "integer"):
            return value.rounded() == value
        default:
            return false
        }
    }

    private static func invalid(_ message: String) -> MLXOpenAIServiceError {
        .invalidResponseFormatOutput(message)
    }
}

private extension JSONValue {
    var objectValue: [String: JSONValue]? {
        if case .object(let value) = self { return value }
        return nil
    }

    var stringValue: String? {
        if case .string(let value) = self { return value }
        return nil
    }

    var stringArrayValue: [String]? {
        guard case .array(let values) = self else { return nil }
        return values.compactMap(\.stringValue)
    }

    var intValue: Int? {
        if case .int(let value) = self { return value }
        return nil
    }

    var doubleCompatibleValue: Double? {
        switch self {
        case .int(let value):
            return Double(value)
        case .double(let value):
            return value
        default:
            return nil
        }
    }

    func encodedJSONString() throws -> String {
        let data = try JSONEncoder.openAIServer.encode(self)
        return String(decoding: data, as: UTF8.self)
    }
}

private extension Array where Element == OpenAIChatMessage {
    func insertingResponseFormatInstruction(_ instruction: String) -> [OpenAIChatMessage] {
        let message = OpenAIChatMessage(
            role: .system,
            content: .text(instruction)
        )
        let insertionIndex = firstIndex { $0.role != .system } ?? endIndex
        var messages = self
        messages.insert(message, at: insertionIndex)
        return messages
    }
}

private extension Array where Element: Hashable {
    func uniqued() -> [Element] {
        var seen = Set<Element>()
        return filter { seen.insert($0).inserted }
    }
}

private extension String {
    func unfencedMarkdownJSON() -> String? {
        guard hasPrefix("```") else {
            return nil
        }

        var lines = components(separatedBy: .newlines)
        guard !lines.isEmpty else {
            return nil
        }
        lines.removeFirst()
        if lines.last?.trimmingCharacters(in: .whitespacesAndNewlines) == "```" {
            lines.removeLast()
        }
        return lines.joined(separator: "\n").trimmingCharacters(in: .whitespacesAndNewlines)
    }

    func firstBalancedJSONValue(rootMustBeObject: Bool) -> String? {
        let scalars = Array(unicodeScalars)
        guard let start = scalars.firstIndex(where: { scalar in
            rootMustBeObject ? scalar == "{" : (scalar == "{" || scalar == "[")
        }) else {
            return nil
        }

        var stack: [UnicodeScalar] = []
        var inString = false
        var escaped = false

        for index in start..<scalars.endIndex {
            let scalar = scalars[index]
            if inString {
                if escaped {
                    escaped = false
                } else if scalar == "\\" {
                    escaped = true
                } else if scalar == "\"" {
                    inString = false
                }
                continue
            }

            switch scalar {
            case "\"":
                inString = true
            case "{":
                stack.append("}")
            case "[":
                stack.append("]")
            case "}", "]":
                guard stack.last == scalar else {
                    return nil
                }
                stack.removeLast()
                if stack.isEmpty {
                    let substring = String(String.UnicodeScalarView(scalars[start...index]))
                    return substring.trimmingCharacters(in: .whitespacesAndNewlines)
                }
            default:
                break
            }
        }

        return nil
    }
}
