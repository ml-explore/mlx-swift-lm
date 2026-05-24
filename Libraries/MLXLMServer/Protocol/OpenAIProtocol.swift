// Copyright © 2026 Apple Inc.

import Foundation
import MLXLMCommon

public enum OpenAIRole: String, Codable, Sendable {
    case system
    case user
    case assistant
    case tool
}

public enum OpenAIContentPart: Codable, Sendable, Equatable {
    case text(String)
    case imageURL(String)
    case unsupported(type: String)

    private enum CodingKeys: String, CodingKey {
        case type
        case text
        case imageURL = "image_url"
        case url
    }

    private enum PartType: String, Codable {
        case text
        case imageURL = "image_url"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let type = try container.decode(String.self, forKey: .type)
        switch type {
        case PartType.text.rawValue:
            self = .text(try container.decode(String.self, forKey: .text))
        case PartType.imageURL.rawValue:
            let image = try container.nestedContainer(keyedBy: CodingKeys.self, forKey: .imageURL)
            self = .imageURL(try image.decode(String.self, forKey: .url))
        default:
            self = .unsupported(type: type)
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        switch self {
        case .text(let text):
            try container.encode(PartType.text.rawValue, forKey: .type)
            try container.encode(text, forKey: .text)
        case .imageURL(let url):
            try container.encode(PartType.imageURL.rawValue, forKey: .type)
            var image = container.nestedContainer(keyedBy: CodingKeys.self, forKey: .imageURL)
            try image.encode(url, forKey: .url)
        case .unsupported(let type):
            try container.encode(type, forKey: .type)
        }
    }
}

public enum OpenAIMessageContent: Codable, Sendable, Equatable {
    case text(String)
    case parts([OpenAIContentPart])
    case null

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if container.decodeNil() {
            self = .null
        } else if let text = try? container.decode(String.self) {
            self = .text(text)
        } else {
            self = .parts(try container.decode([OpenAIContentPart].self))
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .text(let text):
            try container.encode(text)
        case .parts(let parts):
            try container.encode(parts)
        case .null:
            try container.encodeNil()
        }
    }

    public var text: String {
        switch self {
        case .text(let text):
            return text
        case .parts(let parts):
            return parts.compactMap {
                if case .text(let text) = $0 { return text }
                return nil
            }.joined()
        case .null:
            return ""
        }
    }
}

public struct OpenAIChatMessage: Codable, Sendable, Equatable {
    public var role: OpenAIRole
    public var content: OpenAIMessageContent
    public var name: String?
    public var toolCallID: String?
    public var toolCalls: [OpenAIToolCall]?
    public var reasoningContent: String?

    private enum CodingKeys: String, CodingKey {
        case role
        case content
        case name
        case toolCallID = "tool_call_id"
        case toolCalls = "tool_calls"
        case reasoningContent = "reasoning_content"
    }

    public init(
        role: OpenAIRole,
        content: OpenAIMessageContent,
        name: String? = nil,
        toolCallID: String? = nil,
        toolCalls: [OpenAIToolCall]? = nil,
        reasoningContent: String? = nil
    ) {
        self.role = role
        self.content = content
        self.name = name
        self.toolCallID = toolCallID
        self.toolCalls = toolCalls
        self.reasoningContent = reasoningContent
    }

    public var textContent: String {
        content.text
    }

    public func chatMessage() -> Chat.Message {
        switch role {
        case .system:
            return .system(textContent)
        case .user:
            return .user(textContent)
        case .assistant:
            return .assistant(textContent)
        case .tool:
            return .tool(textContent)
        }
    }
}

public struct OpenAIFunctionDefinition: Codable, Sendable, Equatable {
    public var name: String
    public var description: String?
    public var parameters: JSONValue?

    public init(name: String, description: String? = nil, parameters: JSONValue? = nil) {
        self.name = name
        self.description = description
        self.parameters = parameters
    }
}

public struct OpenAITool: Codable, Sendable, Equatable {
    public var type: String
    public var function: OpenAIFunctionDefinition

    public init(type: String = "function", function: OpenAIFunctionDefinition) {
        self.type = type
        self.function = function
    }

    public func toolSpec() -> ToolSpec {
        var functionObject: [String: any Sendable] = ["name": function.name]
        if let description = function.description {
            functionObject["description"] = description
        }
        if let parameters = function.parameters {
            functionObject["parameters"] = parameters.sendableValue
        }
        return [
            "type": type,
            "function": functionObject,
        ]
    }
}

extension JSONValue {
    fileprivate var sendableValue: any Sendable {
        switch self {
        case .null:
            return JSONNull()
        case .bool(let value):
            return value
        case .int(let value):
            return value
        case .double(let value):
            return value
        case .string(let value):
            return value
        case .array(let values):
            return values.map(\.sendableValue)
        case .object(let object):
            return object.mapValues(\.sendableValue)
        }
    }
}

private struct JSONNull: Sendable, Hashable {}

public struct OpenAIToolCall: Codable, Sendable, Equatable {
    public struct Function: Codable, Sendable, Equatable {
        public var name: String
        public var arguments: String

        public init(name: String, arguments: String) {
            self.name = name
            self.arguments = arguments
        }
    }

    public var id: String
    public var type: String
    public var function: Function

    public init(id: String, type: String = "function", function: Function) {
        self.id = id
        self.type = type
        self.function = function
    }
}

public enum OpenAIToolChoice: Codable, Sendable, Equatable {
    public enum Mode: String, Codable, Sendable {
        case none
        case auto
        case required
    }

    case mode(Mode)
    case function(name: String)

    private enum CodingKeys: String, CodingKey {
        case type
        case function
        case name
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let raw = try? container.decode(String.self),
            let mode = Mode(rawValue: raw)
        {
            self = .mode(mode)
            return
        }

        let object = try decoder.container(keyedBy: CodingKeys.self)
        let function = try object.nestedContainer(keyedBy: CodingKeys.self, forKey: .function)
        self = .function(name: try function.decode(String.self, forKey: .name))
    }

    public func encode(to encoder: Encoder) throws {
        switch self {
        case .mode(let mode):
            var container = encoder.singleValueContainer()
            try container.encode(mode.rawValue)
        case .function(let name):
            var container = encoder.container(keyedBy: CodingKeys.self)
            try container.encode("function", forKey: .type)
            var function = container.nestedContainer(keyedBy: CodingKeys.self, forKey: .function)
            try function.encode(name, forKey: .name)
        }
    }
}

public struct OpenAIChatCompletionRequest: Codable, Sendable, Equatable {
    public var model: String
    public var messages: [OpenAIChatMessage]
    public var tools: [OpenAITool]?
    public var toolChoice: OpenAIToolChoice?
    public var toolCallParser: String?
    public var reasoningParser: ReasoningParserFormat?
    public var responseFormat: OpenAIResponseFormat?
    public var stream: Bool?
    public var temperature: Float?
    public var topP: Float?
    public var topK: Int?
    public var minP: Float?
    public var maxTokens: Int?
    public var presencePenalty: Float?
    public var frequencyPenalty: Float?
    public var repetitionPenalty: Float?
    public var stop: [String]?
    public var streamOptions: OpenAIStreamOptions?

    private enum CodingKeys: String, CodingKey {
        case model
        case messages
        case tools
        case toolChoice = "tool_choice"
        case toolCallParser = "tool_call_parser"
        case reasoningParser = "reasoning_parser"
        case responseFormat = "response_format"
        case stream
        case temperature
        case topP = "top_p"
        case topK = "top_k"
        case minP = "min_p"
        case maxTokens = "max_tokens"
        case presencePenalty = "presence_penalty"
        case frequencyPenalty = "frequency_penalty"
        case repetitionPenalty = "repetition_penalty"
        case stop
        case streamOptions = "stream_options"
    }

    public init(
        model: String,
        messages: [OpenAIChatMessage],
        tools: [OpenAITool]? = nil,
        toolChoice: OpenAIToolChoice? = nil,
        toolCallParser: String? = nil,
        reasoningParser: ReasoningParserFormat? = nil,
        responseFormat: OpenAIResponseFormat? = nil,
        stream: Bool? = nil,
        temperature: Float? = nil,
        topP: Float? = nil,
        topK: Int? = nil,
        minP: Float? = nil,
        maxTokens: Int? = nil,
        presencePenalty: Float? = nil,
        frequencyPenalty: Float? = nil,
        repetitionPenalty: Float? = nil,
        stop: [String]? = nil,
        streamOptions: OpenAIStreamOptions? = nil
    ) {
        self.model = model
        self.messages = messages
        self.tools = tools
        self.toolChoice = toolChoice
        self.toolCallParser = toolCallParser
        self.reasoningParser = reasoningParser
        self.responseFormat = responseFormat
        self.stream = stream
        self.temperature = temperature
        self.topP = topP
        self.topK = topK
        self.minP = minP
        self.maxTokens = maxTokens
        self.presencePenalty = presencePenalty
        self.frequencyPenalty = frequencyPenalty
        self.repetitionPenalty = repetitionPenalty
        self.stop = stop
        self.streamOptions = streamOptions
    }

    public var generationParameters: GenerateParameters {
        GenerateParameters(
            maxTokens: maxTokens,
            temperature: temperature ?? 0.6,
            topP: topP ?? 1.0,
            topK: topK ?? 0,
            minP: minP ?? 0,
            repetitionPenalty: repetitionPenalty,
            presencePenalty: presencePenalty,
            frequencyPenalty: frequencyPenalty
        )
    }
}

public struct OpenAIStreamOptions: Codable, Sendable, Equatable {
    public var includeUsage: Bool?
    public var continuousUsageStats: Bool?

    private enum CodingKeys: String, CodingKey {
        case includeUsage = "include_usage"
        case continuousUsageStats = "continuous_usage_stats"
    }

    public init(includeUsage: Bool? = nil, continuousUsageStats: Bool? = nil) {
        self.includeUsage = includeUsage
        self.continuousUsageStats = continuousUsageStats
    }
}

public struct OpenAIUsage: Codable, Sendable, Equatable {
    public var promptTokens: Int
    public var completionTokens: Int
    public var totalTokens: Int

    private enum CodingKeys: String, CodingKey {
        case promptTokens = "prompt_tokens"
        case completionTokens = "completion_tokens"
        case totalTokens = "total_tokens"
    }

    public init(promptTokens: Int, completionTokens: Int) {
        self.promptTokens = promptTokens
        self.completionTokens = completionTokens
        self.totalTokens = promptTokens + completionTokens
    }
}

public struct OpenAIChatCompletionChunk: Codable, Sendable, Equatable {
    public struct Choice: Codable, Sendable, Equatable {
        public var index: Int
        public var delta: Delta
        public var finishReason: String?

        private enum CodingKeys: String, CodingKey {
            case index
            case delta
            case finishReason = "finish_reason"
        }

        public init(index: Int, delta: Delta, finishReason: String?) {
            self.index = index
            self.delta = delta
            self.finishReason = finishReason
        }
    }

    public struct Delta: Codable, Sendable, Equatable {
        public var role: String?
        public var content: String?
        public var reasoningContent: String?
        public var toolCalls: [OpenAIToolCall]?

        private enum CodingKeys: String, CodingKey {
            case role
            case content
            case reasoningContent = "reasoning_content"
            case toolCalls = "tool_calls"
        }

        public init(
            role: String?,
            content: String?,
            reasoningContent: String?,
            toolCalls: [OpenAIToolCall]?
        ) {
            self.role = role
            self.content = content
            self.reasoningContent = reasoningContent
            self.toolCalls = toolCalls
        }
    }

    public var id: String
    public var object: String
    public var created: Int
    public var model: String
    public var choices: [Choice]
    public var usage: OpenAIUsage?

    public init(
        id: String,
        model: String,
        choices: [Choice],
        usage: OpenAIUsage?,
        created: Int = Int(Date().timeIntervalSince1970)
    ) {
        self.id = id
        self.object = "chat.completion.chunk"
        self.created = created
        self.model = model
        self.choices = choices
        self.usage = usage
    }
}
