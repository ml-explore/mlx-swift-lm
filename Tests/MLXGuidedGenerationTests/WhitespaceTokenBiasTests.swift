// Copyright © 2025 Apple Inc.

import MLX
import MLXGuidedGeneration
import MLXLMCommon
import Testing

// MARK: - Test Tokenizers

/// Minimal 256 single-byte tokenizer for tests.
/// Each byte is its own token ID, enabling exact character-to-ID mapping.
private struct ByteTokenizer: MLXLMCommon.Tokenizer {
    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        Array(text.utf8).map { Int($0) }
    }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        String(bytes: tokenIds.map { UInt8($0 & 0xFF) }, encoding: .utf8) ?? ""
    }

    func convertTokenToId(_ token: String) -> Int? {
        guard let byte = token.utf8.first, token.utf8.count == 1 else { return nil }
        return Int(byte)
    }

    func convertIdToToken(_ id: Int) -> String? {
        guard id >= 0 && id < 256 else { return nil }
        return String(UnicodeScalar(UInt8(id)))
    }

    var bosToken: String? { nil }
    var eosToken: String? { String(UnicodeScalar(UInt8(255))) }
    var unknownToken: String? { nil }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] { [] }
}

/// Configurable tokenizer with an arbitrary token list.
/// Token at index i has ID i. No EOS token.
private struct SmallTokenizer: MLXLMCommon.Tokenizer {
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
struct WhitespaceTokenBiasTests {

    private let tokenizer = ByteTokenizer()

    // MARK: - Single-Byte Whitespace

    @Test(
        "Single-byte JSON whitespace tokens (tab, newline, carriage return, space) receive -200.0 bias and appear in token ID set"
    )
    func singleByteWhitespaceTokensGetNegativeBias() {
        let result = WhitespaceTokenBias.compute(tokenizer: tokenizer)

        let values = result.bias.asArray(Float.self)

        // 0x09 = tab (9), 0x0A = newline (10), 0x0D = carriage return (13), 0x20 = space (32)
        #expect(values[9] == -200.0)  // tab
        #expect(values[10] == -200.0)  // newline
        #expect(values[13] == -200.0)  // carriage return
        #expect(values[32] == -200.0)  // space

        #expect(result.tokenIDs.contains(9))
        #expect(result.tokenIDs.contains(10))
        #expect(result.tokenIDs.contains(13))
        #expect(result.tokenIDs.contains(32))
    }

    // MARK: - Multi-Byte Whitespace

    @Test("Multi-byte all-whitespace token (e.g. newline+spaces) receives -200.0 bias")
    func multiBytePureWhitespaceGetsBias() {
        // Token 0 = "\n  " (newline + two spaces), token 1 = "hello"
        let tok = SmallTokenizer(tokens: ["\n  ", "hello"])
        let result = WhitespaceTokenBias.compute(tokenizer: tok)
        let values = result.bias.asArray(Float.self)

        #expect(values[0] == -200.0)  // all-whitespace multi-byte
        #expect(values[1] == 0.0)  // non-whitespace
        #expect(result.tokenIDs.contains(0))
        #expect(!result.tokenIDs.contains(1))
    }

    // MARK: - Raw Byte Tokens

    @Test("Raw byte tokens <0xHH> for whitespace bytes receive -200.0 bias")
    func rawByteWhitespaceTokensGetBias() {
        let tok = SmallTokenizer(tokens: [
            "<0x09>",  // 0: tab
            "<0x0A>",  // 1: newline
            "<0x0D>",  // 2: carriage return
            "<0x20>",  // 3: space
            "<0x41>",  // 4: 'A' (not whitespace)
            "<0x00>",  // 5: NUL (not whitespace)
            "hello",  // 6: normal token
        ])
        let result = WhitespaceTokenBias.compute(tokenizer: tok)
        let values = result.bias.asArray(Float.self)

        #expect(values[0] == -200.0)  // tab
        #expect(values[1] == -200.0)  // newline
        #expect(values[2] == -200.0)  // carriage return
        #expect(values[3] == -200.0)  // space
        #expect(values[4] == 0.0)  // 'A'
        #expect(values[5] == 0.0)  // NUL
        #expect(values[6] == 0.0)  // normal
        #expect(result.tokenIDs == Set([0, 1, 2, 3]))
    }

    // MARK: - SentencePiece Whitespace

    @Test("SentencePiece space marker and combinations with JSON whitespace receive -200.0 bias")
    func sentencePieceSpaceMarkerGetsBias() {
        // \u{2581} is the SentencePiece lower one-eighth block (space marker)
        let tok = SmallTokenizer(tokens: [
            "\u{2581}",  // 0: lone SentencePiece marker
            "\u{2581}\u{2581}",  // 1: two markers
            "\u{2581} ",  // 2: marker + space
            "\u{2581}a",  // 3: marker + non-whitespace (should NOT be biased)
        ])
        let result = WhitespaceTokenBias.compute(tokenizer: tok)
        let values = result.bias.asArray(Float.self)

        #expect(values[0] == -200.0)  // lone marker
        #expect(values[1] == -200.0)  // two markers
        #expect(values[2] == -200.0)  // marker + space
        #expect(values[3] == 0.0)  // marker + 'a'
        #expect(result.tokenIDs == Set([0, 1, 2]))
    }

    // MARK: - Mixed Content

    @Test("Token with any non-whitespace byte receives 0.0 bias and is not in token ID set")
    func mixedContentTokensGetZeroBias() {
        let tok = SmallTokenizer(tokens: [
            " a",  // 0: space + letter
            "\thello\n",  // 1: tab + text + newline
            "\u{2581}x",  // 2: SentencePiece marker + letter
            "abc",  // 3: pure non-whitespace
        ])
        let result = WhitespaceTokenBias.compute(tokenizer: tok)
        let values = result.bias.asArray(Float.self)

        for id in 0 ..< 4 {
            #expect(values[id] == 0.0, "Token \(id) should have 0.0 bias")
            #expect(!result.tokenIDs.contains(id), "Token \(id) should not be in whitespace set")
        }
    }

    // MARK: - Empty String

    @Test("Empty string token receives 0.0 bias, not classified as whitespace")
    func emptyStringTokenGetsZeroBias() {
        let tok = SmallTokenizer(tokens: ["", " "])
        let result = WhitespaceTokenBias.compute(tokenizer: tok)
        let values = result.bias.asArray(Float.self)

        #expect(values[0] == 0.0)  // empty string
        #expect(values[1] == -200.0)  // space is whitespace
        #expect(!result.tokenIDs.contains(0))
        #expect(result.tokenIDs.contains(1))
    }

    // MARK: - Non-Whitespace

    @Test("Non-whitespace tokens have 0.0 bias in ByteTokenizer")
    func nonWhitespaceTokensHaveZeroBias() {
        let result = WhitespaceTokenBias.compute(tokenizer: tokenizer)
        let values = result.bias.asArray(Float.self)

        // 'A' = 65, '{' = 123, '0' = 48, '"' = 34
        #expect(values[65] == 0.0)  // A
        #expect(values[123] == 0.0)  // {
        #expect(values[48] == 0.0)  // 0
        #expect(values[34] == 0.0)  // "
        #expect(values[0] == 0.0)  // NUL
    }

    // MARK: - Output Shape

    @Test("Output bias shape equals discovered vocab size")
    func outputShapeMatchesVocabSize() {
        // ByteTokenizer has 256 tokens
        let result = WhitespaceTokenBias.compute(tokenizer: tokenizer)
        #expect(result.bias.shape == [256])

        // SmallTokenizer with 5 tokens
        let small = SmallTokenizer(tokens: ["a", " ", "\t", "hi", "\u{2581}"])
        let smallResult = WhitespaceTokenBias.compute(tokenizer: small)
        #expect(smallResult.bias.shape == [5])
    }

    // MARK: - No Whitespace Tokens

    @Test("Tokenizer with no whitespace tokens produces all-zero bias and empty ID set")
    func noWhitespaceTokensProducesAllZeros() {
        let tok = SmallTokenizer(tokens: ["hello", "world", "123"])
        let result = WhitespaceTokenBias.compute(tokenizer: tok)
        let values = result.bias.asArray(Float.self)

        #expect(values == [0.0, 0.0, 0.0])
        #expect(result.tokenIDs.isEmpty)
    }
}
