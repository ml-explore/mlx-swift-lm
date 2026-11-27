// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

import Foundation
import FoundationModels
import MLXLMCommon
import Testing

@testable import MLXFoundationModels
@testable import MLXGuidedGeneration

// Nested under the serialized cache parent (ModelCacheEvictionTests.swift), like
// the other suites that touch the process-global model cache.
extension FoundationModelsCacheTests {

    @Suite("MLXLanguageModel grammar tokenizer cache")
    struct GrammarTokenizerCaching {

        /// The adapter's grammar tokenizer knows every id the loop stops on.
        @Test("makeGrammarTokenizer declares the loop's stop tokens")
        func declaresLoopStopTokens() async throws {
            guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

            let tokenizer = StopTokenizer(
                tokens: ["a", "b", "<eos>", "<end_of_turn>", "<unk>", "<end>"],
                eosToken: "<eos>", unknownToken: "<unk>")
            let modelID = "org/grammar-stops-\(UUID().uuidString)"
            var configuration = ModelConfiguration(id: modelID, extraEOSTokens: ["<end>"])
            configuration.eosTokenIds = [3, 99]

            let grammarTokenizer = try await MLXLanguageModel.makeGrammarTokenizer(
                modelID: modelID, tokenizer: tokenizer, configuration: configuration)

            let loopStops = GuidedGenerationLoop.stopTokenIDs(
                tokenizer: tokenizer, configuration: configuration)
            #expect(loopStops == [2, 3, 4, 5, 99])
            // 99 is outside the vocabulary, so the grammar cannot know it.
            #expect(grammarTokenizer.stopTokenIDs == [2, 3, 4, 5])
        }

        @Test("the same stop tokens reuse the cached grammar tokenizer")
        func sameStopsReuseTheEntry() async throws {
            guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

            let modelID = "org/grammar-reuse-\(UUID().uuidString)"
            var configuration = ModelConfiguration(id: modelID)
            configuration.eosTokenIds = [3]

            let first = try await MLXLanguageModel.makeGrammarTokenizer(
                modelID: modelID, tokenizer: Self.tokenizer, configuration: configuration)
            let second = try await MLXLanguageModel.makeGrammarTokenizer(
                modelID: modelID, tokenizer: Self.tokenizer, configuration: configuration)
            #expect(first === second)
        }

        /// One stop more, then one stop less: each time the entry and the
        /// model's compiled grammars are rebuilt with the new stop tokens.
        @Test("changed stop tokens rebuild the entry and drop compiled grammars")
        func changedStopsRebuildTheEntry() async throws {
            guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

            let modelID = "org/grammar-rebuild-\(UUID().uuidString)"
            var configuration = ModelConfiguration(id: modelID)
            var previous: GrammarTokenizer?
            for (eosTokenIds, expected) in [
                ([3], [2, 3, 4]), ([3, 5], [2, 3, 4, 5]), ([3], [2, 3, 4]),
            ] {
                configuration.eosTokenIds = Set(eosTokenIds)
                let grammarTokenizer = try await MLXLanguageModel.makeGrammarTokenizer(
                    modelID: modelID, tokenizer: Self.tokenizer, configuration: configuration)
                #expect(grammarTokenizer.stopTokenIDs == expected)
                if let previous {
                    #expect(grammarTokenizer !== previous)
                }
                previous = grammarTokenizer

                let constraint = try await MLXLanguageModel.makeConstraint(
                    modelID: modelID, kind: .json, source: #"{"type":"string"}"#,
                    tokenizer: grammarTokenizer, hostTokenizer: Self.tokenizer, fastForward: false)
                #expect(constraint.stopTokenIDs == expected)
            }
        }

        private static let tokenizer = StopTokenizer(
            tokens: ["a", "b", "<eos>", "<end_of_turn>", "<unk>", "<end>"],
            eosToken: "<eos>", unknownToken: "<unk>")
    }
}

/// Tokenizer with a fixed vocab, an EOS token and an unknown token.
private struct StopTokenizer: Tokenizer {
    let tokens: [String]
    let eosToken: String?
    let unknownToken: String?

    func encode(text: String, addSpecialTokens: Bool) -> [Int] { [] }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { "" }
    func convertTokenToId(_ token: String) -> Int? { tokens.firstIndex(of: token) }
    func convertIdToToken(_ id: Int) -> String? {
        guard id >= 0, id < tokens.count else { return nil }
        return tokens[id]
    }
    var bosToken: String? { nil }
    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] { [] }
}

#endif
