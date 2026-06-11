// Copyright © 2025 Apple Inc.

import MLX
import MLXLMCommon
import Testing

// MARK: - Stub Tokenizer

/// Tokenizer with a fixed vocabulary list. Token at index `i` has ID `i`.
private struct ListTokenizer: MLXLMCommon.Tokenizer {
    let tokens: [String]

    func encode(text: String, addSpecialTokens: Bool) -> [Int] { [] }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { "" }

    func convertTokenToId(_ token: String) -> Int? {
        self.tokens.firstIndex(of: token)
    }

    func convertIdToToken(_ id: Int) -> String? {
        guard id >= 0, id < self.tokens.count else { return nil }
        return self.tokens[id]
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

// MARK: - Tests

@Suite
struct ClosingTokenBiasTests {

    @Test("Tier-2 closing characters get +100 bias")
    func tier2CharactersGetHundredBias() {
        let tok = ListTokenizer(tokens: [
            "\"",  // 0
            "}",  // 1
            "]",  // 2
            "0",  // 3
            "5",  // 4
            "9",  // 5
            "abc",  // 6 (not closing)
        ])
        let bias = ClosingTokenBias.compute(tokenizer: tok, eosTokenId: nil)
        let values = bias.asArray(Float.self)

        #expect(values[0] == 100.0)  // "
        #expect(values[1] == 100.0)  // }
        #expect(values[2] == 100.0)  // ]
        #expect(values[3] == 100.0)  // 0
        #expect(values[4] == 100.0)  // 5
        #expect(values[5] == 100.0)  // 9
        #expect(values[6] == 0.0)  // abc
    }

    @Test("EOS token gets +200 bias overriding any tier-2 setting")
    func eosTokenGetsTwoHundredBiasOverridingTier2() {
        let tok = ListTokenizer(tokens: [
            "}",  // 0 - tier 2
            "<EOS>",  // 1 - EOS
            "abc",  // 2 - none
        ])
        let bias = ClosingTokenBias.compute(tokenizer: tok, eosTokenId: 1)
        let values = bias.asArray(Float.self)

        #expect(values[0] == 100.0)  // tier 2 only
        #expect(values[1] == 200.0)  // EOS
        #expect(values[2] == 0.0)
    }

    @Test("EOS that overlaps with a tier-2 character takes the +200 bias")
    func eosOverlapsTier2() {
        let tok = ListTokenizer(tokens: [
            "\"",  // 0 - tier 2 AND EOS
            "abc",  // 1
        ])
        let bias = ClosingTokenBias.compute(tokenizer: tok, eosTokenId: 0)
        let values = bias.asArray(Float.self)

        // EOS bias overrides tier-2
        #expect(values[0] == 200.0)
        #expect(values[1] == 0.0)
    }

    @Test("Unknown / non-closing tokens receive 0.0 bias")
    func unknownTokensGetZeroBias() {
        let tok = ListTokenizer(tokens: [
            "hello",
            "world",
            "abc",
            "{",  // opening - not in tier 2
            "[",  // opening - not in tier 2
        ])
        let bias = ClosingTokenBias.compute(tokenizer: tok, eosTokenId: nil)
        let values = bias.asArray(Float.self)

        #expect(values == [0.0, 0.0, 0.0, 0.0, 0.0])
    }

    @Test("Vocab size discovery scans until convertIdToToken returns nil")
    func vocabSizeDiscoveryWorks() {
        let tok = ListTokenizer(tokens: ["a", "b", "}", "]", "\""])
        let bias = ClosingTokenBias.compute(tokenizer: tok, eosTokenId: nil)

        // Discovered vocab size should be 5
        #expect(bias.shape == [5])
    }

    @Test("Out-of-range EOS id is ignored")
    func outOfRangeEOSIgnored() {
        let tok = ListTokenizer(tokens: ["a", "}"])
        let bias = ClosingTokenBias.compute(tokenizer: tok, eosTokenId: 999)
        let values = bias.asArray(Float.self)

        #expect(values[0] == 0.0)
        #expect(values[1] == 100.0)  // tier 2 still applies
    }
}
