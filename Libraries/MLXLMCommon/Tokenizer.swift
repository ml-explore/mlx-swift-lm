// Copyright © 2024 Apple Inc.

import Foundation
import Hub
import Tokenizers

struct TokenizerError: Error {
    let message: String
}

private enum StreamingDetokenizerKind {
    case naive
    case bpe
}

private final class StreamingDetokenizerRegistry: @unchecked Sendable {
    static let shared = StreamingDetokenizerRegistry()
    private let lock = NSLock()
    private var kinds: [ObjectIdentifier: StreamingDetokenizerKind] = [:]

    func set(kind: StreamingDetokenizerKind, for tokenizer: Tokenizer) {
        let key = ObjectIdentifier(tokenizer as AnyObject)
        lock.withLock {
            kinds[key] = kind
        }
    }

    func kind(for tokenizer: Tokenizer) -> StreamingDetokenizerKind? {
        let key = ObjectIdentifier(tokenizer as AnyObject)
        return lock.withLock { kinds[key] }
    }
}

public func loadTokenizer(configuration: ModelConfiguration, hub: HubApi) async throws -> Tokenizer
{
    var (tokenizerConfig, tokenizerData) = try await loadTokenizerConfig(
        configuration: configuration, hub: hub)

    if let overrideTokenizer = configuration.overrideTokenizer {
        if var dictionary = tokenizerConfig.dictionary() {
            dictionary["tokenizer_class"] = .init(overrideTokenizer)
            tokenizerConfig = Config(dictionary)
        }
    }

    let tokenizer = try AutoTokenizer.from(
        tokenizerConfig: tokenizerConfig,
        tokenizerData: tokenizerData
    )
    registerStreamingDetokenizerKind(tokenizer: tokenizer, tokenizerData: tokenizerData)
    return tokenizer
}

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

    guard var tokenizerConfig = try await config.tokenizerConfig else {
        throw TokenizerError(message: "missing config")
    }
    let tokenizerData = try await config.tokenizerData

    tokenizerConfig = updateTokenizerConfig(tokenizerConfig)

    return (tokenizerConfig, tokenizerData)
}

private func updateTokenizerConfig(_ tokenizerConfig: Config) -> Config {
    // Workaround: replacement tokenizers for unhandled values in swift-transformers
    if let tokenizerClass = tokenizerConfig.tokenizerClass?.string(),
        let replacement = replacementTokenizers[tokenizerClass]
    {
        if var dictionary = tokenizerConfig.dictionary() {
            dictionary["tokenizer_class"] = .init(replacement)
            return Config(dictionary)
        }
    }
    return tokenizerConfig
}

public class TokenizerReplacementRegistry: @unchecked Sendable {

    // Note: using NSLock as we have very small (just dictionary get/set)
    // critical sections and expect no contention. this allows the methods
    // to remain synchronous.
    private let lock = NSLock()

    /// overrides for TokenizerModel/knownTokenizers
    private var replacementTokenizers = [
        "InternLM2Tokenizer": "PreTrainedTokenizer",
        "Qwen2Tokenizer": "PreTrainedTokenizer",
        "Qwen3Tokenizer": "PreTrainedTokenizer",
        "CohereTokenizer": "PreTrainedTokenizer",
        "GPTNeoXTokenizer": "PreTrainedTokenizer",
        "TokenizersBackend": "PreTrainedTokenizer",
    ]

    public subscript(key: String) -> String? {
        get {
            lock.withLock {
                replacementTokenizers[key]
            }
        }
        set {
            lock.withLock {
                replacementTokenizers[key] = newValue
            }
        }
    }
}

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

public struct BPEStreamingDetokenizer: StreamingDetokenizer {
    private static let byteDecoder: [UnicodeScalar: UInt8] = {
        var byteSet = Set<UInt8>()
        var mapping: [UnicodeScalar: UInt8] = [:]

        func addRange(_ range: ClosedRange<Int>) {
            for value in range {
                byteSet.insert(UInt8(value))
            }
        }

        addRange(33 ... 126)
        addRange(161 ... 172)
        addRange(174 ... 255)

        var n = 0
        for value in 0 ... 255 {
            let byte = UInt8(value)
            if byteSet.contains(byte) {
                if let scalar = UnicodeScalar(value) {
                    mapping[scalar] = byte
                }
            } else {
                if let scalar = UnicodeScalar(256 + n) {
                    mapping[scalar] = byte
                }
                n += 1
            }
        }
        return mapping
    }()

    private let tokenizer: Tokenizer
    private var tokenCache: [Int: String] = [:]
    private var unflushed: String = ""
    private var hasEmitted: Bool = false

    public init(tokenizer: Tokenizer) {
        self.tokenizer = tokenizer
    }

    public mutating func append(token: Int) {
        let value: String
        if let cached = tokenCache[token] {
            value = cached
        } else {
            let resolved = tokenizer.convertIdToToken(token) ?? "!"
            tokenCache[token] = resolved
            value = resolved
        }
        unflushed.append(contentsOf: value)
    }

    public mutating func next() -> String? {
        guard !unflushed.isEmpty else { return nil }
        let decoded = decodeBytes(unflushed)
        if decoded.hasSuffix("\u{fffd}") {
            return nil
        }
        var output = decoded
        if !hasEmitted, output.first == " " {
            output.removeFirst()
        }
        unflushed = ""
        if !output.isEmpty {
            hasEmitted = true
        }
        return output
    }

    private func decodeBytes(_ text: String) -> String {
        var bytes: [UInt8] = []
        bytes.reserveCapacity(text.utf8.count)
        for scalar in text.unicodeScalars {
            if let byte = Self.byteDecoder[scalar] {
                bytes.append(byte)
            } else {
                bytes.append(contentsOf: String(scalar).utf8)
            }
        }
        return String(decoding: bytes, as: UTF8.self)
    }
}

private func registerStreamingDetokenizerKind(tokenizer: Tokenizer, tokenizerData: Config) {
    let kind = inferStreamingDetokenizerKind(tokenizerData)
    StreamingDetokenizerRegistry.shared.set(kind: kind, for: tokenizer)
}

private func inferStreamingDetokenizerKind(_ tokenizerData: Config) -> StreamingDetokenizerKind {
    let decoderType = tokenizerData["decoder"]["type"].string()
    if decoderType == "ByteLevel" {
        return .bpe
    }
    return .naive
}

func makeStreamingDetokenizer(tokenizer: Tokenizer) -> any StreamingDetokenizer {
    let kind = StreamingDetokenizerRegistry.shared.kind(for: tokenizer) ?? .naive
    switch kind {
    case .bpe:
        return BPEStreamingDetokenizer(tokenizer: tokenizer)
    case .naive:
        return NaiveStreamingDetokenizer(tokenizer: tokenizer)
    }
}
