// Copyright © 2026 Apple Inc.

import Foundation
import MLXLMCommon

public enum OpenAIResponseFormatType: String, Codable, Sendable {
    case text
    case jsonObject = "json_object"
    case jsonSchema = "json_schema"
}

public struct OpenAIJSONSchemaResponseFormat: Codable, Sendable, Equatable {
    public var name: String?
    public var description: String?
    public var strict: Bool?
    public var schema: JSONValue

    public init(
        name: String? = nil,
        description: String? = nil,
        strict: Bool? = nil,
        schema: JSONValue
    ) {
        self.name = name
        self.description = description
        self.strict = strict
        self.schema = schema
    }
}

public struct OpenAIResponseFormat: Codable, Sendable, Equatable {
    public var type: OpenAIResponseFormatType
    public var jsonSchema: OpenAIJSONSchemaResponseFormat?

    private enum CodingKeys: String, CodingKey {
        case type
        case jsonSchema = "json_schema"
    }

    public init(type: OpenAIResponseFormatType, jsonSchema: OpenAIJSONSchemaResponseFormat? = nil) {
        self.type = type
        self.jsonSchema = jsonSchema
    }

    public static func text() -> OpenAIResponseFormat {
        .init(type: .text)
    }

    public static func jsonObject() -> OpenAIResponseFormat {
        .init(type: .jsonObject)
    }

    public static func jsonSchema(_ schema: OpenAIJSONSchemaResponseFormat) -> OpenAIResponseFormat {
        .init(type: .jsonSchema, jsonSchema: schema)
    }

    public var requiresJSONOutput: Bool {
        type == .jsonObject || type == .jsonSchema
    }
}
