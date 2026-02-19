// Copyright © 2024 Apple Inc.

import Foundation
import Tokenizers

@available(
    *, unavailable, message: "Use AutoTokenizer.from(directory:) from swift-tokenizers instead"
)
public func loadTokenizerConfig(configuration: ModelConfiguration, hub: Any) async throws -> (
    Any, Any
) {
    fatalError()
}

@available(
    *, unavailable,
    message: "Use AutoTokenizer.register(_:for:) from swift-tokenizers instead"
)
public class TokenizerReplacementRegistry: @unchecked Sendable {
    public subscript(key: String) -> String? {
        get { fatalError() }
        set { fatalError() }
    }
}

@available(
    *, unavailable,
    message: "Use AutoTokenizer.register(_:for:) from swift-tokenizers instead"
)
public let replacementTokenizers = TokenizerReplacementRegistry()

public protocol StreamingDetokenizer: IteratorProtocol<String> {

    mutating func append(token: Int)

}

public struct NaiveStreamingDetokenizer: StreamingDetokenizer {
    let tokenizer: Tokenizer

    var segmentTokens = [Int]()
    var segment = ""

    public init(tokenizer: Tokenizer) {
        self.tokenizer = tokenizer
    }

    mutating public func append(token: Int) {
        segmentTokens.append(token)
    }

    mutating func startNewSegment() {
        let lastToken = segmentTokens.last
        segmentTokens.removeAll()
        if let lastToken {
            segmentTokens.append(lastToken)
            segment = tokenizer.decode(tokens: segmentTokens)
        } else {
            segment = ""
        }
    }

    public mutating func next() -> String? {
        let newSegment = tokenizer.decode(tokens: segmentTokens)
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
        }

        return String(new)
    }

}
