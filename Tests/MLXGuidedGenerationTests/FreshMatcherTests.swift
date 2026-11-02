// Copyright © 2026 Apple Inc.

import Foundation
import Testing

@testable import MLXGuidedGeneration

/// `GrammarConstraint.freshMatcher()` reuses a compiled grammar and does not
/// need `GrammarMatcher::Fork()`, so these run on the vendored xgrammar.
@Suite
struct FreshMatcherTests {

    private func makeByteFallbackTokenizer() throws -> GrammarTokenizer {
        let vocab: [String] = (0 ..< 256).map { String(format: "<0x%02X>", $0) }
        return try GrammarTokenizer(vocab: vocab, vocabType: .byteFallback, eosTokenId: 255)
    }

    private func allows(_ mask: MaskResult, _ token: Int) -> Bool {
        (UInt32(bitPattern: mask.mask[token / 32]) >> UInt32(token % 32)) & 1 == 1
    }

    @Test
    func freshMatcherStartsAtInitialStateAfterSourceAdvances() throws {
        let source = try GrammarConstraint(
            tokenizer: try makeByteFallbackTokenizer(), grammar: #"root ::= "ab""#)
        _ = try source.commitToken(97)  // "a"

        let fresh = try source.freshMatcher()
        let mask = try fresh.computeMask()
        #expect(allows(mask, 97))
        #expect(!allows(mask, 98))
        #expect(allows(try source.computeMask(), 98))
    }

    @Test
    func freshMatchersAreIndependent() throws {
        let template = try GrammarConstraint(
            tokenizer: try makeByteFallbackTokenizer(), grammar: #"root ::= "ab""#)
        let first = try template.freshMatcher()
        _ = try first.commitToken(97)  // "a"
        let second = try first.freshMatcher()

        #expect(allows(try first.computeMask(), 98))
        #expect(!allows(try first.computeMask(), 97))
        for constraint in [template, second] {
            let mask = try constraint.computeMask()
            #expect(allows(mask, 97))
            #expect(!allows(mask, 98))
        }
    }

    @Test
    func freshMatcherKeepsTheCompiledGrammarAlive() throws {
        var source: GrammarConstraint? = try GrammarConstraint(
            tokenizer: try makeByteFallbackTokenizer(), grammar: #"root ::= "ab""#)
        weak var owner = source
        var first: GrammarConstraint? = try source!.freshMatcher()
        source = nil
        let second = try first!.freshMatcher()
        first = nil
        #expect(owner != nil)

        _ = try second.commitToken(97)  // "a"
        let mask = try second.computeMask()
        #expect(allows(mask, 98))
        #expect(!allows(mask, 97))
    }
}
