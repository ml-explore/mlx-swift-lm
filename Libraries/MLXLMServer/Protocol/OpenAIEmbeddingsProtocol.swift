// Copyright © 2026 Apple Inc.

import Foundation

public enum OpenAIEmbeddingInput: Codable, Sendable, Equatable {
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

    public var texts: [String] {
        switch self {
        case .text(let text):
            return [text]
        case .texts(let texts):
            return texts
        }
    }
}

public struct OpenAIEmbeddingRequest: Codable, Sendable, Equatable {
    public var model: String
    public var input: OpenAIEmbeddingInput
    public var encodingFormat: String?
    public var normalize: Bool?

    private enum CodingKeys: String, CodingKey {
        case model
        case input
        case encodingFormat = "encoding_format"
        case normalize
    }

    public init(
        model: String,
        input: OpenAIEmbeddingInput,
        encodingFormat: String? = nil,
        normalize: Bool? = nil
    ) {
        self.model = model
        self.input = input
        self.encodingFormat = encodingFormat
        self.normalize = normalize
    }
}

public struct OpenAIEmbeddingResponse: Codable, Sendable, Equatable {
    public struct Item: Codable, Sendable, Equatable {
        public var object: String
        public var embedding: [Float]
        public var index: Int

        public init(embedding: [Float], index: Int) {
            self.object = "embedding"
            self.embedding = embedding
            self.index = index
        }
    }

    public var object: String
    public var data: [Item]
    public var model: String
    public var usage: OpenAIUsage

    public init(data: [Item], model: String, usage: OpenAIUsage) {
        self.object = "list"
        self.data = data
        self.model = model
        self.usage = usage
    }
}
