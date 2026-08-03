// Copyright © 2026 Apple Inc.

import MLXLMCommon
import Testing

@testable import MLXGuidedGeneration

/// Model-free regression tests pinning the contract that `GuidedGenerationLoop`
/// sources its stop-token set entirely from the `ModelConfiguration` it is
/// handed — `extraEOSTokens` (string → id via the tokenizer) and `eosTokenIds`
/// (inserted directly) — plus the tokenizer's primary EOS. These lock the
/// behavior preserved when the per-call stop-token threading was removed:
/// the configuration is the single place stop tokens are carried.
@Suite
struct StopTokenSourceTests {

    /// Configurable tokenizer: token at index `i` has id `i`; `eosID` (if set)
    /// is surfaced as the tokenizer's primary EOS. Mirrors the existing
    /// `SmallTokenizer`/`ByteTokenizer` stubs in this target.
    private struct ListTokenizer: MLXLMCommon.Tokenizer {
        let tokens: [String]
        let eosID: Int?

        func encode(text: String, addSpecialTokens: Bool) -> [Int] { [] }
        func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { "" }
        func convertTokenToId(_ token: String) -> Int? { tokens.firstIndex(of: token) }
        func convertIdToToken(_ id: Int) -> String? {
            guard id >= 0, id < tokens.count else { return nil }
            return tokens[id]
        }
        var bosToken: String? { nil }
        var eosToken: String? { eosID.flatMap { convertIdToToken($0) } }
        var unknownToken: String? { nil }
        func applyChatTemplate(
            messages: [[String: any Sendable]],
            tools: [[String: any Sendable]]?,
            additionalContext: [String: any Sendable]?
        ) throws -> [Int] { [] }
    }

    @Test("extraEOSTokens in the configuration are mapped to ids and included")
    func extraEOSTokensFromConfigurationAreIncluded() {
        let tokenizer = ListTokenizer(tokens: ["a", "<end_of_turn>", "b"], eosID: nil)
        let config = ModelConfiguration(id: "test", extraEOSTokens: ["<end_of_turn>"])

        let stop = GuidedGenerationLoop.buildStopTokenIDs(
            tokenizer: tokenizer, configuration: config)

        #expect(stop == [1])  // only the configured extra token; ids 0 ("a") and 2 ("b") excluded
    }

    @Test("eosTokenIds in the configuration are included directly")
    func eosTokenIdsFromConfigurationAreIncluded() {
        let tokenizer = ListTokenizer(tokens: ["a", "b"], eosID: nil)
        var config = ModelConfiguration(id: "test")
        config.eosTokenIds = [99, 100]

        let stop = GuidedGenerationLoop.buildStopTokenIDs(
            tokenizer: tokenizer, configuration: config)

        #expect(stop.contains(99))
        #expect(stop.contains(100))
    }

    @Test("the tokenizer's primary EOS is included")
    func tokenizerPrimaryEOSIsIncluded() {
        let tokenizer = ListTokenizer(tokens: ["a", "</s>"], eosID: 1)
        let config = ModelConfiguration(id: "test")

        let stop = GuidedGenerationLoop.buildStopTokenIDs(
            tokenizer: tokenizer, configuration: config)

        #expect(stop == [1])  // empty config ⇒ tokenizer EOS is the only stop id
    }

    @Test("all three configuration-borne sources union together")
    func allSourcesUnion() {
        let tokenizer = ListTokenizer(
            tokens: ["a", "<end_of_turn>", "</s>"], eosID: 2)
        var config = ModelConfiguration(id: "test", extraEOSTokens: ["<end_of_turn>"])
        config.eosTokenIds = [99]

        let stop = GuidedGenerationLoop.buildStopTokenIDs(
            tokenizer: tokenizer, configuration: config)

        #expect(stop.contains(1))  // extraEOSTokens → "<end_of_turn>"
        #expect(stop.contains(2))  // tokenizer EOS → "</s>"
        #expect(stop.contains(99))  // eosTokenIds direct
    }
}
