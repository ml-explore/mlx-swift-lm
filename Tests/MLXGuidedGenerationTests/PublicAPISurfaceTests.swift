// Copyright © 2026 Apple Inc.
//
// Pins the deliberate public API of MLXGuidedGeneration. Uses a NON-@testable
// import so it fails to compile if any of these declarations is not `public`.

import MLXGuidedGeneration
import Testing

@Suite
struct PublicAPISurfaceTests {

    @Test
    func grammarConstraintCompilesAndMasksThroughPublicAPI() throws {
        // GrammarTokenizer built from a 256-entry byte-fallback vocab via the
        // public VocabType enum (no CXGrammar import required by the caller).
        let vocab: [String] = (0 ..< 256).map { String(format: "<0x%02X>", $0) }
        let tokenizer = try GrammarTokenizer(
            vocab: vocab,
            vocabType: .byteFallback,
            eosTokenId: 255
        )

        let constraint = try GrammarConstraint(
            tokenizer: tokenizer,
            jsonSchema: #"{ "type": "integer" }"#
        )

        let mask: MaskResult = try constraint.computeMask()
        #expect(!mask.mask.isEmpty)
        #expect(mask.isTerminated == false)
    }

    @Test
    func grammarErrorIsPublicAndTyped() {
        let error: GrammarError = .invalidJSONSchema("bad schema")
        guard case .invalidJSONSchema(let message) = error else {
            Issue.record("expected invalidJSONSchema")
            return
        }
        #expect(message == "bad schema")
    }

    @Test
    func commitResultIsPublic() throws {
        let vocab: [String] = (0 ..< 256).map { String(format: "<0x%02X>", $0) }
        let tokenizer = try GrammarTokenizer(
            vocab: vocab, vocabType: .byteFallback, eosTokenId: 255)
        let constraint = try GrammarConstraint(
            tokenizer: tokenizer, jsonSchema: #"{ "type": "integer" }"#)
        _ = try constraint.computeMask()
        // "0" is ASCII 0x30 = token 48 in the byte-fallback vocab; a valid
        // first token for an integer. commitToken returns a public CommitResult.
        let result: CommitResult = try constraint.commitToken(48)
        #expect(result.isTerminated == true || result.isTerminated == false)
    }
}
