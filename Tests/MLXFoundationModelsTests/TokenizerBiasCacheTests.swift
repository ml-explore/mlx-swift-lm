// Copyright © 2025 Apple Inc.

import Foundation
import FoundationModels
import MLXLMCommon
import Testing

@testable import MLXFoundationModels

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

    // Extends the serialized parent declared in ModelCacheEvictionTests.swift so these
    // cache-touching tests never run concurrently with the other cache suites (the
    // process-global `static let cache` + key-agnostic `evictAll()` would otherwise race).
    extension FoundationModelsCacheTests {

        @Suite("MLXLanguageModel tokenizer-bias cache")
        struct TokenizerBiasCaching {

            @Test("makeTokenizerBias scans the vocab once, then serves from cache")
            func cachesPerModel() async {
                guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

                let tok = CountingTokenizer(tokens: ["a", "b", "}", "\n"])
                let id = "org/bias-\(UUID().uuidString)"

                _ = await MLXLanguageModel.makeTokenizerBias(modelID: id, tokenizer: tok)
                let afterFirst = tok.idLookupCount
                #expect(afterFirst > 0, "first call must scan the vocab")

                _ = await MLXLanguageModel.makeTokenizerBias(modelID: id, tokenizer: tok)
                #expect(
                    tok.idLookupCount == afterFirst,
                    "second call for the same model must hit the cache, not rescan")
            }

            @Test("a different modelID computes a fresh bias")
            func isPerModel() async {
                guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

                let tok = CountingTokenizer(tokens: ["a", "b", "}", "\n"])
                let idA = "org/bias-a-\(UUID().uuidString)"
                let idB = "org/bias-b-\(UUID().uuidString)"

                _ = await MLXLanguageModel.makeTokenizerBias(modelID: idA, tokenizer: tok)
                let afterA = tok.idLookupCount
                _ = await MLXLanguageModel.makeTokenizerBias(modelID: idB, tokenizer: tok)
                #expect(
                    tok.idLookupCount > afterA,
                    "a new modelID must trigger a fresh vocab scan")
            }
        }
    }

    // MARK: - Fixtures

    /// Tokenizer with a fixed vocab that counts `convertIdToToken` calls, so a test can
    /// assert whether a bias computation re-scanned the vocab (cache miss) or not (hit).
    /// `@unchecked Sendable`: the counter is mutated only from serialized test calls.
    private final class CountingTokenizer: Tokenizer, @unchecked Sendable {
        let tokens: [String]
        private(set) var idLookupCount = 0

        init(tokens: [String]) { self.tokens = tokens }

        func encode(text: String, addSpecialTokens: Bool) -> [Int] { [] }
        func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { "" }
        func convertTokenToId(_ token: String) -> Int? { tokens.firstIndex(of: token) }
        func convertIdToToken(_ id: Int) -> String? {
            idLookupCount += 1
            guard id >= 0, id < tokens.count else { return nil }
            return tokens[id]
        }
        var bosToken: String? { nil }
        var eosToken: String? { nil }
        var unknownToken: String? { nil }
        func applyChatTemplate(
            messages: [[String: any Sendable]],
            tools: [[String: any Sendable]]?,
            additionalContext: [String: any Sendable]?
        ) throws -> [Int] { [] }
    }

#endif  // FoundationModelsIntegration && canImport(FoundationModels)
