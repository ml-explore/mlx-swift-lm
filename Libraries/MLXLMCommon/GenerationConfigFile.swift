// Copyright © 2024 Apple Inc.

import Foundation

/// JSON wrapper for `generation_config.json` file.
///
/// This file can override values from `config.json`, particularly `eos_token_id`.
/// Following mlx-lm Python behavior, if `generation_config.json` exists and contains
/// `eos_token_id`, it takes precedence over the value in `config.json`.
///
/// It can also declare `suppress_tokens`: token IDs that must never be sampled
/// during generation (see `SuppressTokensLogitsProcessor` in Hugging Face
/// transformers). Model factories merge these into the model's
/// ``SuppressedTokensProviding/suppressedTokenIds``.
public struct GenerationConfigFile: Codable, Sendable {
    public var eosTokenIds: IntOrIntArray?
    public var stopStrings: Set<String>
    public var suppressTokens: IntOrIntArray?

    enum CodingKeys: String, CodingKey {
        case eosTokenIds = "eos_token_id"
        case stopStrings = "stop_strings"
        case stop
        case suppressTokens = "suppress_tokens"
    }

    public init(
        eosTokenIds: IntOrIntArray? = nil, stopStrings: Set<String> = [],
        suppressTokens: IntOrIntArray? = nil
    ) {
        self.eosTokenIds = eosTokenIds
        self.stopStrings = stopStrings
        self.suppressTokens = suppressTokens
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        eosTokenIds = try container.decodeIfPresent(IntOrIntArray.self, forKey: .eosTokenIds)
        suppressTokens = try container.decodeIfPresent(IntOrIntArray.self, forKey: .suppressTokens)

        stopStrings = []
        stopStrings.formUnion(Self.decodeStringSet(from: container, forKey: .stopStrings))
        stopStrings.formUnion(Self.decodeStringSet(from: container, forKey: .stop))
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encodeIfPresent(eosTokenIds, forKey: .eosTokenIds)
        try container.encodeIfPresent(suppressTokens, forKey: .suppressTokens)
        if !stopStrings.isEmpty {
            try container.encode(stopStrings.sorted(), forKey: .stopStrings)
        }
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
