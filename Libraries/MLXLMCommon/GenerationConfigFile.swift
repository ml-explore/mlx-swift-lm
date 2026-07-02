// Copyright © 2024 Apple Inc.

import Foundation

/// JSON wrapper for `generation_config.json` file.
///
/// This file can override values from `config.json`, particularly `eos_token_id`.
/// Following mlx-lm Python behavior, if `generation_config.json` exists and contains
/// `eos_token_id`, it takes precedence over the value in `config.json`.
public struct GenerationConfigFile: Codable, Sendable {
    public var eosTokenIds: IntOrIntArray?
    public var stopStrings: Set<String>

    enum CodingKeys: String, CodingKey {
        case eosTokenIds = "eos_token_id"
        case stopStrings = "stop_strings"
        case stop
    }

    public init(eosTokenIds: IntOrIntArray? = nil, stopStrings: Set<String> = []) {
        self.eosTokenIds = eosTokenIds
        self.stopStrings = stopStrings
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        eosTokenIds = try container.decodeIfPresent(IntOrIntArray.self, forKey: .eosTokenIds)

        stopStrings = []
        stopStrings.formUnion(Self.decodeStringSet(from: container, forKey: .stopStrings))
        stopStrings.formUnion(Self.decodeStringSet(from: container, forKey: .stop))
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encodeIfPresent(eosTokenIds, forKey: .eosTokenIds)
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
