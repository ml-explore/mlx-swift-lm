// Copyright © 2024 Apple Inc.

import Foundation

/// A protocol for tokenizing text into token IDs and decoding token IDs into text.
public protocol Tokenizer: Sendable {
    func encode(text: String, addSpecialTokens: Bool) throws -> [Int]
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) throws -> String
    func convertTokenToId(_ token: String) -> Int?
    func convertIdToToken(_ id: Int) -> String?

    var bosToken: String? { get }
    var eosToken: String? { get }
    var unknownToken: String? { get }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int]
}

extension Tokenizer {
    public func encode(text: String) throws -> [Int] {
        try encode(text: text, addSpecialTokens: true)
    }

    public func decode(tokenIds: [Int]) throws -> String {
        try decode(tokenIds: tokenIds, skipSpecialTokens: false)
    }

    public var eosTokenId: Int? {
        guard let eosToken else { return nil }
        return convertTokenToId(eosToken)
    }

    public var unknownTokenId: Int? {
        guard let unknownToken else { return nil }
        return convertTokenToId(unknownToken)
    }

    public func applyChatTemplate(
        messages: [[String: any Sendable]]
    ) throws -> [Int] {
        try applyChatTemplate(messages: messages, tools: nil, additionalContext: nil)
    }

    public func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?
    ) throws -> [Int] {
        try applyChatTemplate(messages: messages, tools: tools, additionalContext: nil)
    }
}

/// A `Tokenizer` that exposes a raw decode path for streaming detokenization.
///
/// Conforming tokenizers promise that ``rawDecode(tokenIds:skipSpecialTokens:)``
/// returns text without any post-decode cleanup — no whitespace fixups, no
/// retroactive contraction rewrites, nothing that would break the byte-prefix
/// monotonicity that ``StreamingDetokenizer`` depends on. Cleanup, when the
/// tokenizer applies it, belongs in `decode(tokenIds:skipSpecialTokens:)`.
///
/// ``StreamingDetokenizer`` uses `rawDecode` automatically when the tokenizer
/// conforms; tokenizers that don't conform fall back to `decode` and rely on
/// the reset-and-retry recovery in the generation loop when cleanup violates
/// the prefix invariant.
public protocol StreamingDecodeTokenizer: Tokenizer {
    func rawDecode(tokenIds: [Int], skipSpecialTokens: Bool) throws -> String
}

public enum TokenizerError: LocalizedError, Equatable {
    case missingChatTemplate
    case invalidStreamingPrefix(tokenId: Int, expectedPrefix: String, actualString: String)

    public var errorDescription: String? {
        switch self {
        case .missingChatTemplate:
            "This tokenizer does not have a chat template."
        case .invalidStreamingPrefix(let tokenId, let expectedPrefix, let actualString):
            "Streaming detokenizer prefix invariant violated for token \(tokenId): expected prefix \(expectedPrefix.debugDescription), got \(actualString.debugDescription)."
        }
    }
}
