// Copyright © 2025 Apple Inc.

import MLXGuidedGeneration
import MLXLMCommon
import Testing

// MARK: - Stub Tokenizer

/// Minimal tokenizer stub: each input character maps to one token.
/// Token count therefore equals string length.
private struct StubTokenizer: MLXLMCommon.Tokenizer {
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
        guard id >= 0, id < 256 else { return nil }
        return String(UnicodeScalar(UInt8(id)))
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
struct CompletionReserveTests {

    private let tokenizer = StubTokenizer()

    @Test("Empty object schema returns token count of '{}'")
    func emptyObjectSchemaTokenCount() {
        let schema = #"{"type":"object"}"#
        let reserve = CompletionReserve.estimate(schemaJSON: schema, tokenizer: tokenizer)
        // Minimal JSON for {} object with no required fields => "{}" (2 chars => 2 tokens)
        #expect(reserve == 2)
    }

    @Test("Malformed JSON returns the default reserve")
    func malformedJSONReturnsDefault() {
        let reserve = CompletionReserve.estimate(
            schemaJSON: "not a schema",
            tokenizer: tokenizer,
            defaultReserve: 99
        )
        #expect(reserve == 99)
    }

    @Test("Object with required string property returns expected count")
    func objectWithRequiredStringProperty() {
        let schema =
            #"{"type":"object","required":["name"],"properties":{"name":{"type":"string"}}}"#
        // Minimal JSON: {"name":""} (11 chars => 11 tokens)
        let expected = #"{"name":""}"#.utf8.count
        let reserve = CompletionReserve.estimate(schemaJSON: schema, tokenizer: tokenizer)
        #expect(reserve == expected)
    }

    @Test("Default reserve falls back to 64 when not provided")
    func defaultReserveDefault() {
        let reserve = CompletionReserve.estimate(schemaJSON: "garbage", tokenizer: tokenizer)
        #expect(reserve == 64)
    }
}
