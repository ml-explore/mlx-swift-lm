// Copyright © 2024 Apple Inc.

import Foundation

public enum StringOrNumber: Codable, Equatable, Sendable {
    case string(String)
    case float(Float)

    public init(from decoder: Decoder) throws {
        let values = try decoder.singleValueContainer()

        if let v = try? values.decode(Float.self) {
            self = .float(v)
        } else {
            let v = try values.decode(String.self)
            self = .string(v)
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .string(let v): try container.encode(v)
        case .float(let v): try container.encode(v)
        }
    }
}

private class ModelTypeRegistry: @unchecked Sendable {

    // Note: Using NSLock as we have very small (just dictionary get/set)
    // critical sections and expect no contention. This allows the methods
    // to remain synchronous.
    private let lock = NSLock()

    private var creators: [String: @Sendable (Data) throws -> EmbeddingModel] = [
        "bert": { data in
            let configuration = try JSONDecoder().decode(BertConfiguration.self, from: data)
            return BertModel(configuration)
        },
        "roberta": { data in
            let configuration = try JSONDecoder().decode(BertConfiguration.self, from: data)
            return BertModel(configuration)
        },
        "xlm-roberta": { data in
            let configuration = try JSONDecoder().decode(BertConfiguration.self, from: data)
            return BertModel(configuration)
        },
        "distilbert": { data in
            let configuration = try JSONDecoder().decode(BertConfiguration.self, from: data)
            return BertModel(configuration)
        },
        "nomic_bert": { data in
            let configuration = try JSONDecoder().decode(NomicBertConfiguration.self, from: data)
            return NomicBertModel(configuration, pooler: false)
        },
        "qwen3": { data in
            let configuration = try JSONDecoder().decode(Qwen3Configuration.self, from: data)
            return Qwen3Model(configuration)
        },
    ]

    public func registerModelType(
        _ type: String, creator: @Sendable @escaping (Data) throws -> EmbeddingModel
    ) {
        lock.withLock {
            creators[type] = creator
        }
    }

    public func createModel(configuration: Data, rawValue: String) throws -> EmbeddingModel {
        let creator = lock.withLock {
            creators[rawValue]
        }
        guard let creator else {
            throw EmbedderError.unsupportedModelType(rawValue)
        }
        return try creator(configuration)
    }

}

private let modelTypeRegistry = ModelTypeRegistry()

public struct ModelType: RawRepresentable, Codable, Sendable {
    public let rawValue: String

    public init(rawValue: String) {
        self.rawValue = rawValue
    }

    public static func registerModelType(
        _ type: String, creator: @Sendable @escaping (Data) throws -> EmbeddingModel
    ) {
        modelTypeRegistry.registerModelType(type, creator: creator)
    }

    public func createModel(configuration: Data) throws -> EmbeddingModel {
        try modelTypeRegistry.createModel(configuration: configuration, rawValue: rawValue)
    }
}
