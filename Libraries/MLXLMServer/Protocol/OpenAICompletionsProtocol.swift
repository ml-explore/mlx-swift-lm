// Copyright © 2026 Apple Inc.

import Foundation

public enum OpenAICompletionPrompt: Codable, Sendable, Equatable {
    case text(String)
    case texts([String])

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let text = try? container.decode(String.self) {
            self = .text(text)
            return
        }
        self = .texts(try container.decode([String].self))
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .text(let text):
            try container.encode(text)
        case .texts(let texts):
            try container.encode(texts)
        }
    }

    public var firstText: String {
        switch self {
        case .text(let text):
            return text
        case .texts(let texts):
            return texts.first ?? ""
        }
    }
}

public struct OpenAICompletionRequest: Codable, Sendable, Equatable {
    public var model: String
    public var prompt: OpenAICompletionPrompt
    public var stream: Bool?
    public var temperature: Float?
    public var topP: Float?
    public var maxTokens: Int?

    private enum CodingKeys: String, CodingKey {
        case model
        case prompt
        case stream
        case temperature
        case topP = "top_p"
        case maxTokens = "max_tokens"
    }

    public var chatCompletionRequest: OpenAIChatCompletionRequest {
        .init(
            model: model,
            messages: [.init(role: .user, content: .text(prompt.firstText))],
            stream: stream,
            temperature: temperature,
            topP: topP,
            maxTokens: maxTokens
        )
    }
}

public struct OpenAICompletionResponse: Codable, Sendable, Equatable {
    public struct Choice: Codable, Sendable, Equatable {
        public var text: String
        public var index: Int
        public var finishReason: String?

        private enum CodingKeys: String, CodingKey {
            case text
            case index
            case finishReason = "finish_reason"
        }
    }

    public var id: String
    public var object: String
    public var created: Int
    public var model: String
    public var choices: [Choice]
    public var usage: OpenAIUsage

    public init(from chatResponse: OpenAIChatCompletionResponse) {
        self.id = chatResponse.id.replacingOccurrences(of: "chatcmpl", with: "cmpl")
        self.object = "text_completion"
        self.created = chatResponse.created
        self.model = chatResponse.model
        self.choices = chatResponse.choices.map {
            .init(text: $0.message.textContent, index: $0.index, finishReason: $0.finishReason)
        }
        self.usage = chatResponse.usage
    }
}
