// Copyright © 2024 Apple Inc.

import Foundation
import Hub
import Tokenizers

struct TokenizerError: Error {
    let message: String
}

public func loadTokenizer(configuration: ModelConfiguration, hub: HubApi) async throws -> Tokenizer
{
    switch configuration.id {
    case .id(let id, let revision):
        return try await AutoTokenizer.from(
            pretrained: configuration.tokenizerId ?? id,
            hubApi: hub,
            revision: revision
        )
    case .directory(let directory):
        return try await AutoTokenizer.from(modelFolder: directory, hubApi: hub)
    }
}

@available(
    *, deprecated, message: "Use LanguageModelConfigurationFromHub from swift-transformers directly"
)
public func loadTokenizerConfig(configuration: ModelConfiguration, hub: HubApi) async throws -> (
    Config, Config
) {
    // from AutoTokenizer.from() -- this lets us override parts of the configuration
    let config: LanguageModelConfigurationFromHub

    switch configuration.id {
    case .id(let id, let revision):
        do {
            // the load can fail (async when we try to use it)
            let loaded = LanguageModelConfigurationFromHub(
                modelName: configuration.tokenizerId ?? id, revision: revision, hubApi: hub)
            _ = try await loaded.tokenizerConfig
            config = loaded
        } catch {
            let nserror = error as NSError
            if nserror.domain == NSURLErrorDomain
                && nserror.code == NSURLErrorNotConnectedToInternet
            {
                // Internet connection appears to be offline -- fall back to loading from
                // the local directory
                config = LanguageModelConfigurationFromHub(
                    modelFolder: configuration.modelDirectory(hub: hub), hubApi: hub)
            } else {
                throw error
            }
        }
    case .directory(let directory):
        config = LanguageModelConfigurationFromHub(modelFolder: directory, hubApi: hub)
    }

    guard let tokenizerConfig = try await config.tokenizerConfig else {
        throw TokenizerError(message: "missing config")
    }
    let tokenizerData = try await config.tokenizerData

    return (tokenizerConfig, tokenizerData)
}

@available(
    *, unavailable,
    message: "Use AutoTokenizer.register(_:for:) from swift-transformers instead"
)
public class TokenizerReplacementRegistry: @unchecked Sendable {}

@available(
    *, unavailable,
    message: "Use AutoTokenizer.register(_:for:) from swift-transformers instead"
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
