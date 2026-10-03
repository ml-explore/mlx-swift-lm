// Copyright © 2024 Apple Inc.

import Foundation

/// A protocol for tokenizing text into token IDs and decoding token IDs into text.
public protocol Tokenizer: Sendable {
    func encode(text: String, addSpecialTokens: Bool) -> [Int]
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String
    func convertTokenToId(_ token: String) -> Int?
    func convertIdToToken(_ id: Int) -> String?

    /// Creates independent decoding state for one generated sequence.
    /// Override to supply a tokenizer-specific incremental implementation.
    func makeStreamingDetokenizer() -> any StreamingDetokenizer

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
    public func makeStreamingDetokenizer() -> any StreamingDetokenizer {
        NaiveStreamingDetokenizer(tokenizer: self)
    }

    public func encode(text: String) -> [Int] {
        encode(text: text, addSpecialTokens: true)
    }

    public func decode(tokenIds: [Int]) -> String {
        decode(tokenIds: tokenIds, skipSpecialTokens: false)
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

public enum TokenizerError: LocalizedError {
    case missingChatTemplate

    public var errorDescription: String? {
        switch self {
        case .missingChatTemplate:
            "This tokenizer does not have a chat template."
        }
    }
}

/// Decodes one generated sequence. Preserve special tokens for downstream protocol parsing.
/// `next()` returns available text, or nil until more tokens arrive. Custom implementations
/// should emit stable text, holding incomplete bytes or context-dependent suffixes as needed.
public protocol StreamingDetokenizer: IteratorProtocol<String> {
    mutating func append(token: Int)

    /// Ends the sequence and returns all remaining text, including any unread output.
    /// Call after the last token and before finishing downstream parsers. Repeated calls
    /// return nil. Create a new decoder before appending another sequence.
    mutating func finish() -> String?
}

extension StreamingDetokenizer {
    /// Preserves the legacy behavior for decoders without a final flush operation.
    public mutating func finish() -> String? { nil }
}

public struct NaiveStreamingDetokenizer: StreamingDetokenizer {
    let tokenizer: any Tokenizer

    var segmentTokens = [Int]()
    var segment = ""

    public init(tokenizer: any Tokenizer) {
        self.tokenizer = tokenizer
    }

    public mutating func append(token: Int) {
        segmentTokens.append(token)
    }

    mutating func startNewSegment() {
        let lastToken = segmentTokens.last
        segmentTokens.removeAll()
        if let lastToken {
            segmentTokens.append(lastToken)
            segment = tokenizer.decode(tokenIds: segmentTokens)
        } else {
            segment = ""
        }
    }

    public mutating func next() -> String? {
        let newSegment = tokenizer.decode(tokenIds: segmentTokens)

        // Emit the part of `newSegment` beyond its common prefix with the text
        // already emitted for this segment. A character-count suffix
        // (`newSegment.suffix(newSegment.count - segment.count)`) assumes
        // `decode` is append-only, but SentencePiece decoding is not: appending
        // a token can rewrite earlier whitespace. For example, with Gemma's
        // tokenizer `decode([\n, "  "]) == "\n  "` while
        // `decode([\n, "  ", ","]) == "\n ,"` — a space collapses. The
        // count-based suffix then computes `suffix(0)` and silently drops the
        // newly decoded character, including structural JSON commas, which
        // corrupts guided-generation output into unparseable JSON.
        //
        // This recovers correctly as long as the already-emitted prefix is
        // stable — i.e. the rewrite happens at or after the divergence point.
        // Real SentencePiece decoding only collapses whitespace at the append
        // boundary, so that holds; a decode that rewrote an already-emitted
        // interior character would re-emit the tail (no worse than the old
        // behavior, and not observed in practice).
        //
        // The prefix is measured in Unicode scalars, not `Character`s. A
        // `Character` is a grapheme cluster, and a token routinely appends a
        // combining scalar to the cluster the previous token ended: a
        // variation selector (`🏳` then U+FE0F), a zero-width joiner, a
        // combining accent (`e` then U+0301). Compared as `Character`s the
        // old text's last cluster no longer equals the merged cluster, so a
        // `Character`-level prefix (`commonPrefix(with:)`) stops one cluster
        // early and the whole merged cluster is emitted a second time — the
        // consumer receives `🏳🏳️`, `''️`, `eé`. Scalars compare exactly (no
        // canonical equivalence), and the emitted tail is precisely the
        // scalars the new token added.
        let common = zip(newSegment.unicodeScalars, segment.unicodeScalars)
            .prefix { $0 == $1 }.count
        let new = String(newSegment.unicodeScalars.dropFirst(common))

        // if the new segment ends with REPLACEMENT CHARACTER this means
        // that the token didn't produce a complete unicode character
        if new.last == "\u{fffd}" {
            return nil
        }

        if new.hasSuffix("\n") {
            startNewSegment()
        } else {
            self.segment = newSegment
        }

        return new
    }
}
