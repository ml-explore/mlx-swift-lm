// Copyright © 2026 Apple Inc.

import Testing

@testable import MLXGuidedGeneration

/// With a `max_tokens` rule, xgrammar's mask decides whether the next commit
/// enforces the budget. `computeMask()` must keep that decision.
@Suite
struct BudgetRuleStopProbeTests {

    private static let grammar = #"""
        root ::= a ("x" "y")?
        a[max_tokens=2] ::= "x"+
        """#

    private static let x: Int32 = 0x78

    private static func allows(_ mask: MaskResult, _ id: Int) -> Bool {
        (UInt32(bitPattern: mask.mask[id / 32]) >> UInt32(id % 32)) & 1 == 1
    }

    /// One constraint calls `computeMask()` `masksPerStep` times per step, the
    /// other only fills the raw mask. Both commit "x" while the raw mask allows it.
    @Test(arguments: [1, 2])
    func computeMaskKeepsTheBudgetDecision(masksPerStep: Int) throws {
        let tokenizer = try HarnessTokenizer().makeGrammarTokenizer(stopTokenIds: [])
        let probed = try GrammarConstraint(tokenizer: tokenizer, grammar: Self.grammar)
        let reference = try GrammarConstraint(tokenizer: tokenizer, grammar: Self.grammar)

        var committed = 0
        var sawStopBit = false
        while committed < 6 {
            var probedMask = try probed.computeMask()
            for _ in 1 ..< masksPerStep {
                probedMask = try probed.computeMask()
            }
            let referenceMask = try reference.computeRawMask()
            #expect(probedMask.mask == referenceMask.mask, "after \(committed) x")
            sawStopBit = sawStopBit || Self.allows(referenceMask, HarnessTokenizer.eosTokenId)
            guard Self.allows(referenceMask, Int(Self.x)) else { break }
            _ = try probed.commitToken(Self.x)
            _ = try reference.commitToken(Self.x)
            committed += 1
        }
        #expect(sawStopBit)
        #expect(committed < 6, "the budget never ended a")
    }
}
