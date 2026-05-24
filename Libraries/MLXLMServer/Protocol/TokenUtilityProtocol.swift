// Copyright © 2026 Apple Inc.

public struct TokenizeRequest: Codable, Sendable, Equatable {
    public var model: String?
    public var prompt: String
    public var addSpecialTokens: Bool?

    private enum CodingKeys: String, CodingKey {
        case model
        case prompt
        case addSpecialTokens = "add_special_tokens"
    }

    public init(model: String? = nil, prompt: String, addSpecialTokens: Bool? = nil) {
        self.model = model
        self.prompt = prompt
        self.addSpecialTokens = addSpecialTokens
    }
}

public struct TokenizeResponse: Codable, Sendable, Equatable {
    public var tokens: [Int]

    public init(tokens: [Int]) {
        self.tokens = tokens
    }
}

public struct DetokenizeRequest: Codable, Sendable, Equatable {
    public var model: String?
    public var tokens: [Int]
    public var skipSpecialTokens: Bool?

    private enum CodingKeys: String, CodingKey {
        case model
        case tokens
        case skipSpecialTokens = "skip_special_tokens"
    }

    public init(model: String? = nil, tokens: [Int], skipSpecialTokens: Bool? = nil) {
        self.model = model
        self.tokens = tokens
        self.skipSpecialTokens = skipSpecialTokens
    }
}

public struct DetokenizeResponse: Codable, Sendable, Equatable {
    public var text: String

    public init(text: String) {
        self.text = text
    }
}

public struct ApplyTemplateRequest: Codable, Sendable, Equatable {
    public var model: String?
    public var messages: [OpenAIChatMessage]
    public var tools: [OpenAITool]?

    public init(model: String? = nil, messages: [OpenAIChatMessage], tools: [OpenAITool]? = nil) {
        self.model = model
        self.messages = messages
        self.tools = tools
    }
}
