// Copyright © 2026 Apple Inc.

import Foundation
import MLXLMCommon

/// A deterministic tokenizer for random-weight models.
///
/// Most tests should use ``ScriptedTokenizer``. Use this one when the model's output is
/// arbitrary ids, as with random weights, and the text does not matter.
///
/// Every id is in `0 ..< vocabularySize`, so any id fits a tiny model's embedding and any
/// id the model emits decodes. Each id is a fixed pseudo-word (`ka`, `lomi`, ...), and
/// decoding joins words with spaces, so random output is readable text.
///
/// There is no EOS or unknown token: random output never stops early, and generation
/// ends at `maxTokens`.
///
/// `encode` splits on whitespace. A vocabulary word maps to its own id, so model output
/// re-encodes to the same ids. Any other word maps to a stable hash of itself. Not
/// lossless, and it does not need to be.
public struct PseudoWordTokenizer: MLXLMCommon.Tokenizer {

    public let vocabularySize: Int
    public let template: any ScriptedChatTemplate
    private let words: [String]
    private let ids: [String: Int]

    public var bosToken: String? { nil }
    public var eosToken: String? { nil }
    public var unknownToken: String? { nil }

    public init(
        vocabularySize: Int = 100, template: any ScriptedChatTemplate = MinimalChatTemplate()
    ) {
        precondition(vocabularySize >= 2, "vocabularySize must be at least 2")
        self.vocabularySize = vocabularySize
        self.template = template
        let words = (0 ..< vocabularySize).map(Self.word)
        self.words = words
        self.ids = Dictionary(uniqueKeysWithValues: words.enumerated().map { ($1, $0) })
    }

    /// The word for `id`: its base-10 digits as syllables, so every id is distinct.
    public static func word(_ id: Int) -> String {
        let syllables = ["ka", "lo", "mi", "nu", "pe", "ri", "so", "ta", "ve", "zu"]
        return String(id).map { syllables[Int(String($0))!] }.joined()
    }

    public func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        text.split(whereSeparator: \.isWhitespace).map { id(for: String($0)) }
    }

    public func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        tokenIds.compactMap { words.indices.contains($0) ? words[$0] : nil }
            .joined(separator: " ")
    }

    public func convertTokenToId(_ token: String) -> Int? {
        ids[token]
    }

    public func convertIdToToken(_ id: Int) -> String? {
        words.indices.contains(id) ? words[id] : nil
    }

    public func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        try template.render(messages: messages, tools: tools, additionalContext: additionalContext)
            .flatMap { encode(text: $0, addSpecialTokens: false) }
    }

    private func id(for word: String) -> Int {
        if let id = ids[word] {
            return id
        }
        // FNV-1a: stable across runs and platforms, unlike `hashValue`.
        var hash: UInt64 = 0xcbf2_9ce4_8422_2325
        for byte in word.utf8 {
            hash = (hash ^ UInt64(byte)) &* 0x100_0000_01b3
        }
        return Int(hash % UInt64(vocabularySize))
    }
}
