// Copyright © 2026 Apple Inc.

import MLX
import Testing

@testable import MLXGuidedGeneration

/// A `max_tokens` rule with an end marker. Once its budget runs out, xgrammar
/// closes it without the marker, and the raw mask allows the stop.
@Suite(.serialized)
struct BudgetForceCloseTests {

    // Rule ids follow definition order: root 0, a 1, body 2, marker 3.
    static let grammar = #"""
        root ::= a
        a[max_tokens=2, capture_hidden_body_rule_id=2, capture_hidden_marker_rule_id=3] ::= body marker
        body ::= [x]*
        marker ::= "Z"
        """#

    static let x = 0x78
    static let marker = 0x5A
    static let stop = HarnessTokenizer.eosTokenId

    static func allowed(_ mask: MaskResult) -> [Int] {
        (0 ..< 256).filter { (UInt32(bitPattern: mask.mask[$0 / 32]) >> UInt32($0 % 32)) & 1 == 1 }
    }

    /// Commits "x" while the raw mask allows it and compares the masks at each step.
    @Test func computeMaskMatchesTheRawMask() throws {
        let tokenizer = try HarnessTokenizer().makeGrammarTokenizer()
        let probed = try GrammarConstraint(tokenizer: tokenizer, grammar: Self.grammar)
        let reference = try GrammarConstraint(tokenizer: tokenizer, grammar: Self.grammar)

        var stopBeforeMarker = false
        for step in 0 ..< 8 {
            let probedAllowed = Self.allowed(try probed.computeMask())
            let rawAllowed = Self.allowed(try reference.computeRawMask())
            #expect(probedAllowed == rawAllowed, "step \(step)")
            stopBeforeMarker = stopBeforeMarker || rawAllowed.contains(Self.stop)
            guard rawAllowed.contains(Self.x) else { break }
            _ = try probed.commitToken(Int32(Self.x))
            _ = try reference.commitToken(Int32(Self.x))
        }
        #expect(stopBeforeMarker, "the raw mask never allowed the stop before the marker")
    }

    /// The loop prefers "x", then the marker. Its text must match a walk that
    /// picks from the raw mask with the same preference.
    @Test func loopEndsWithTheRawText() throws {
        try Stream.withNewDefaultStream(device: .cpu) {
            let tokenizer = try HarnessTokenizer().makeGrammarTokenizer()
            let reference = try GrammarConstraint(tokenizer: tokenizer, grammar: Self.grammar)
            var expected = ""
            for _ in 0 ..< 10 {
                let rawAllowed = Self.allowed(try reference.computeRawMask())
                guard let next = [Self.x, Self.marker].first(where: rawAllowed.contains) else {
                    break
                }
                _ = try reference.commitToken(Int32(next))
                expected.append(Character(UnicodeScalar(UInt8(next))))
            }

            let run = try runLoop(
                source: .grammar(Self.grammar),
                policy: .prefer([UInt8(Self.x), UInt8(Self.marker)]),
                maxTokens: 10, fastForward: false, completionReserve: 0)
            #expect(run.error == nil)
            #expect(run.text == expected)
        }
    }
}
