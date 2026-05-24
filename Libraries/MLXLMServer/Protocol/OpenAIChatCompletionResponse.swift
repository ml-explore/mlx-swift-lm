// Copyright © 2026 Apple Inc.

import Foundation
import MLXLMCommon

public struct OpenAIChatCompletionResponse: Codable, Sendable, Equatable {
    public struct Choice: Codable, Sendable, Equatable {
        public var index: Int
        public var message: OpenAIChatMessage
        public var finishReason: String?

        private enum CodingKeys: String, CodingKey {
            case index
            case message
            case finishReason = "finish_reason"
        }

        public init(index: Int, message: OpenAIChatMessage, finishReason: String?) {
            self.index = index
            self.message = message
            self.finishReason = finishReason
        }
    }

    public var id: String
    public var object: String
    public var created: Int
    public var model: String
    public var choices: [Choice]
    public var usage: OpenAIUsage

    public init(
        id: String,
        model: String,
        choices: [Choice],
        usage: OpenAIUsage,
        created: Int = Int(Date().timeIntervalSince1970)
    ) {
        self.id = id
        self.object = "chat.completion"
        self.created = created
        self.model = model
        self.choices = choices
        self.usage = usage
    }
}

public struct OpenAIModelListResponse: Codable, Sendable, Equatable {
    public var object: String
    public var data: [MLXServerModel]

    public init(data: [MLXServerModel]) {
        self.object = "list"
        self.data = data
    }
}

public struct OpenAIErrorResponse: Codable, Sendable, Equatable {
    public struct Payload: Codable, Sendable, Equatable {
        public var message: String
        public var type: String
        public var param: String?
        public var code: String?
    }

    public var error: Payload

    public init(message: String, type: String = "server_error", param: String? = nil, code: String? = nil) {
        self.error = .init(message: message, type: type, param: param, code: code)
    }
}

extension OpenAIToolCall {
    init(toolCall: ToolCall, id: String) throws {
        let data = try JSONEncoder.openAIServer.encode(toolCall.function.arguments)
        let arguments = String(decoding: data, as: UTF8.self)
        self.init(
            id: id,
            function: .init(name: toolCall.function.name, arguments: arguments)
        )
    }
}
