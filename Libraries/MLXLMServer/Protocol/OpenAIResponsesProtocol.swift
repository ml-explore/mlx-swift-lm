// Copyright © 2026 Apple Inc.

import Foundation
import MLXLMCommon

public enum OpenAIResponseInput: Codable, Sendable, Equatable {
    case text(String)
    case messages([OpenAIChatMessage])

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let text = try? container.decode(String.self) {
            self = .text(text)
            return
        }
        self = .messages(try container.decode([OpenAIChatMessage].self))
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .text(let text):
            try container.encode(text)
        case .messages(let messages):
            try container.encode(messages)
        }
    }

    public func messages(instructions: String?) -> [OpenAIChatMessage] {
        var result: [OpenAIChatMessage] = []
        if let instructions, !instructions.isEmpty {
            result.append(.init(role: .system, content: .text(instructions)))
        }
        switch self {
        case .text(let text):
            result.append(.init(role: .user, content: .text(text)))
        case .messages(let messages):
            result.append(contentsOf: messages)
        }
        return result
    }
}

public struct OpenAIResponseReasoningOptions: Codable, Sendable, Equatable {
    public var parser: ReasoningParserFormat?
    public var effort: String?

    public init(parser: ReasoningParserFormat? = nil, effort: String? = nil) {
        self.parser = parser
        self.effort = effort
    }
}

public struct OpenAIResponseRequest: Codable, Sendable, Equatable {
    public var model: String
    public var input: OpenAIResponseInput
    public var instructions: String?
    public var tools: [OpenAITool]?
    public var toolChoice: OpenAIToolChoice?
    public var reasoning: OpenAIResponseReasoningOptions?
    public var stream: Bool?
    public var store: Bool?
    public var temperature: Float?
    public var topP: Float?
    public var maxOutputTokens: Int?
    public var metadata: [String: JSONValue]?

    private enum CodingKeys: String, CodingKey {
        case model
        case input
        case instructions
        case tools
        case toolChoice = "tool_choice"
        case reasoning
        case stream
        case store
        case temperature
        case topP = "top_p"
        case maxOutputTokens = "max_output_tokens"
        case metadata
    }

    public init(
        model: String,
        input: OpenAIResponseInput,
        instructions: String? = nil,
        tools: [OpenAITool]? = nil,
        toolChoice: OpenAIToolChoice? = nil,
        reasoning: OpenAIResponseReasoningOptions? = nil,
        stream: Bool? = nil,
        store: Bool? = nil,
        temperature: Float? = nil,
        topP: Float? = nil,
        maxOutputTokens: Int? = nil,
        metadata: [String: JSONValue]? = nil
    ) {
        self.model = model
        self.input = input
        self.instructions = instructions
        self.tools = tools
        self.toolChoice = toolChoice
        self.reasoning = reasoning
        self.stream = stream
        self.store = store
        self.temperature = temperature
        self.topP = topP
        self.maxOutputTokens = maxOutputTokens
        self.metadata = metadata
    }

    public var chatCompletionRequest: OpenAIChatCompletionRequest {
        .init(
            model: model,
            messages: input.messages(instructions: instructions),
            tools: tools,
            toolChoice: toolChoice,
            reasoningParser: reasoning?.parser,
            stream: stream,
            temperature: temperature,
            topP: topP,
            maxTokens: maxOutputTokens
        )
    }
}

public enum OpenAIResponseStatus: String, Codable, Sendable, Equatable {
    case inProgress = "in_progress"
    case completed
    case cancelled
    case failed
}

public struct OpenAIResponseOutputContent: Codable, Sendable, Equatable {
    public var type: String
    public var text: String

    public init(type: String = "output_text", text: String) {
        self.type = type
        self.text = text
    }
}

public struct OpenAIResponseOutputItem: Codable, Sendable, Equatable {
    public var id: String
    public var type: String
    public var status: OpenAIResponseStatus?
    public var role: OpenAIRole?
    public var content: [OpenAIResponseOutputContent]?
    public var callID: String?
    public var name: String?
    public var arguments: String?

    private enum CodingKeys: String, CodingKey {
        case id
        case type
        case status
        case role
        case content
        case callID = "call_id"
        case name
        case arguments
    }

    public static func message(id: String, text: String) -> Self {
        .init(
            id: id,
            type: "message",
            status: .completed,
            role: .assistant,
            content: [.init(text: text)],
            callID: nil,
            name: nil,
            arguments: nil
        )
    }

    public static func functionCall(id: String, toolCall: OpenAIToolCall) -> Self {
        .init(
            id: id,
            type: "function_call",
            status: .completed,
            role: nil,
            content: nil,
            callID: toolCall.id,
            name: toolCall.function.name,
            arguments: toolCall.function.arguments
        )
    }
}

public struct OpenAIResponse: Codable, Sendable, Equatable {
    public var id: String
    public var object: String
    public var createdAt: Int
    public var status: OpenAIResponseStatus
    public var model: String
    public var output: [OpenAIResponseOutputItem]
    public var outputText: String
    public var usage: OpenAIUsage?
    public var metadata: [String: JSONValue]?

    private enum CodingKeys: String, CodingKey {
        case id
        case object
        case createdAt = "created_at"
        case status
        case model
        case output
        case outputText = "output_text"
        case usage
        case metadata
    }

    public init(
        id: String,
        status: OpenAIResponseStatus,
        model: String,
        output: [OpenAIResponseOutputItem],
        outputText: String,
        usage: OpenAIUsage?,
        metadata: [String: JSONValue]? = nil,
        createdAt: Int = Int(Date().timeIntervalSince1970)
    ) {
        self.id = id
        self.object = "response"
        self.createdAt = createdAt
        self.status = status
        self.model = model
        self.output = output
        self.outputText = outputText
        self.usage = usage
        self.metadata = metadata
    }
}
