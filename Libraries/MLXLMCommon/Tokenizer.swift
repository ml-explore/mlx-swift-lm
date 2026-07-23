// Copyright © 2024 Apple Inc.

import Foundation

/// A protocol for tokenizing text into token IDs and decoding token IDs into text.
public protocol Tokenizer: Sendable {
    func encode(text: String, addSpecialTokens: Bool) -> [Int]
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String
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

/// Optional tokenizer capability for bounded-memory streaming decode.
///
/// Conform only when the decoded suffix after appending token IDs depends on
/// at most a known number of preceding token IDs. Existing ``Tokenizer``
/// conformers do not need to adopt this protocol and retain the historical
/// unbounded streaming buffer.
public protocol BoundedStreamingDecodeTokenizer: Tokenizer {
    /// The preceding-token overlap retained between streaming decode steps.
    /// Return `nil` to disable compaction for a particular instance.
    var streamingDecodeContextSize: Int? { get }
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

public protocol StreamingDetokenizer: IteratorProtocol<String> {
    mutating func append(token: Int)
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

    mutating func startNewSegment(retaining tokenCount: Int = 1) {
        guard tokenCount > 0, !segmentTokens.isEmpty else {
            segmentTokens.removeAll(keepingCapacity: true)
            segment = ""
            return
        }

        let retainedTokens = Array(segmentTokens.suffix(tokenCount))
        segmentTokens.removeAll(keepingCapacity: true)
        segmentTokens.append(contentsOf: retainedTokens)
        segment = tokenizer.decode(tokenIds: segmentTokens)
    }

    public mutating func next() -> String? {
        let newSegment = tokenizer.decode(tokenIds: segmentTokens)
        let new = newSegment.suffix(newSegment.count - segment.count)

        // if the new segment ends with REPLACEMENT CHARACTER this means
        // that the token didn't produce a complete unicode character
        if new.last == "\u{fffd}" {
            return nil
        }

        if new.hasSuffix("\n") {
            startNewSegment()
        } else {
            self.segment = newSegment

            if let contextSize =
                (tokenizer as? any BoundedStreamingDecodeTokenizer)?.streamingDecodeContextSize,
                contextSize > 0,
                segmentTokens.count > contextSize
            {
                startNewSegment(retaining: contextSize)
            }
        }

        return String(new)
    }
}
