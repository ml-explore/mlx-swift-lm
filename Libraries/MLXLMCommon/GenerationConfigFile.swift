// Copyright © 2024 Apple Inc.

import Foundation

/// JSON wrapper for `generation_config.json` file.
///
/// This file can override values from `config.json`, particularly `eos_token_id`.
/// Following mlx-lm Python behavior, if `generation_config.json` exists and contains
/// `eos_token_id`, it takes precedence over the value in `config.json`.
public struct GenerationConfigFile: Codable, Sendable, Equatable {
    public var eosTokenIds: IntOrIntArray?
    public var stopStrings: Set<String>
    public var temperature: Float?
    public var topP: Float?
    public var topK: Int?
    public var minP: Float?
    public var repetitionPenalty: Float?

    enum CodingKeys: String, CodingKey {
        case eosTokenIds = "eos_token_id"
        case stopStrings = "stop_strings"
        case stop
        case temperature
        case topP = "top_p"
        case topK = "top_k"
        case minP = "min_p"
        case repetitionPenalty = "repetition_penalty"
    }

    public init(
        eosTokenIds: IntOrIntArray? = nil,
        stopStrings: Set<String> = [],
        temperature: Float? = nil,
        topP: Float? = nil,
        topK: Int? = nil,
        minP: Float? = nil,
        repetitionPenalty: Float? = nil
    ) {
        self.eosTokenIds = eosTokenIds
        self.stopStrings = stopStrings
        self.temperature = temperature
        self.topP = topP
        self.topK = topK
        self.minP = minP
        self.repetitionPenalty = repetitionPenalty
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        eosTokenIds = try container.decodeIfPresent(IntOrIntArray.self, forKey: .eosTokenIds)

        stopStrings = []
        stopStrings.formUnion(Self.decodeStringSet(from: container, forKey: .stopStrings))
        stopStrings.formUnion(Self.decodeStringSet(from: container, forKey: .stop))
        temperature = try container.decodeIfPresent(Float.self, forKey: .temperature)
        topP = try container.decodeIfPresent(Float.self, forKey: .topP)
        topK = try container.decodeIfPresent(Int.self, forKey: .topK)
        minP = try container.decodeIfPresent(Float.self, forKey: .minP)
        repetitionPenalty = try container.decodeIfPresent(Float.self, forKey: .repetitionPenalty)
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encodeIfPresent(eosTokenIds, forKey: .eosTokenIds)
        if !stopStrings.isEmpty {
            try container.encode(stopStrings.sorted(), forKey: .stopStrings)
        }
        try container.encodeIfPresent(temperature, forKey: .temperature)
        try container.encodeIfPresent(topP, forKey: .topP)
        try container.encodeIfPresent(topK, forKey: .topK)
        try container.encodeIfPresent(minP, forKey: .minP)
        try container.encodeIfPresent(repetitionPenalty, forKey: .repetitionPenalty)
    }

    /// Applies only values explicitly supplied by the checkpoint, preserving
    /// caller-selected hardware, cache, prefill, and token-limit settings.
    public func applyingSamplingDefaults(
        to parameters: GenerateParameters = .init()
    ) -> GenerateParameters {
        var parameters = parameters
        if let temperature { parameters.temperature = temperature }
        if let topP { parameters.topP = topP }
        if let topK { parameters.topK = topK }
        if let minP { parameters.minP = minP }
        if let repetitionPenalty { parameters.repetitionPenalty = repetitionPenalty }
        return parameters
    }

    private static func decodeStringSet(
        from container: KeyedDecodingContainer<CodingKeys>,
        forKey key: CodingKeys
    ) -> Set<String> {
        if let values = try? container.decode([String].self, forKey: key) {
            return Set(values)
        }
        if let value = try? container.decode(String.self, forKey: key) {
            return [value]
        }
        return []
    }
}
