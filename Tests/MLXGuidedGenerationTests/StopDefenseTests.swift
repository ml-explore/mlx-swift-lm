// Copyright © 2026 Apple Inc.

import MLX
import Testing

@testable import MLXGuidedGeneration

/// A raw-mask hook that sets the bits of `ids`, as xgrammar's cached bitset
/// masks can do for stop tokens.
private func settingBits(_ ids: [Int]) -> @Sendable (MaskResult) -> MaskResult {
    { mask in
        var words = mask.mask
        for id in ids {
            words[id / 32] |= Int32(bitPattern: UInt32(1) << UInt32(id % 32))
        }
        return MaskResult(mask: words, isTerminated: mask.isTerminated, needsApply: mask.needsApply)
    }
}

/// `computeMask()` allows stop tokens only where the grammar can end, even
/// when xgrammar's own mask sets their bits elsewhere.
@Suite(.serialized)
struct StopDefenseTests {

    private static let stringSchema =
        #"{"type":"object","properties":{"s":{"type":"string"}},"required":["s"],"additionalProperties":false}"#
    private static let stops = [0x78, HarnessTokenizer.eosTokenId]

    private static func constraint(hook: @escaping @Sendable (MaskResult) -> MaskResult) throws
        -> GrammarConstraint
    {
        let constraint = try GrammarConstraint(
            tokenizer: HarnessTokenizer().makeGrammarTokenizer(stopTokenIds: [0x78]),
            jsonSchema: stringSchema)
        constraint.rawMaskHookForTesting = hook
        return constraint
    }

    @Test func setStopBitsAreClearedInsideAString() throws {
        let constraint = try Self.constraint(hook: settingBits(Self.stops))
        for byte in #"{"s":""#.utf8 {
            _ = try constraint.commitToken(Int32(byte))
        }
        #expect(
            stopBits(try constraint.computeRawMask(), stops: Self.stops, vocabSize: 256) == true)
        let mask = try constraint.computeMask()
        #expect(stopBits(mask, stops: Self.stops, vocabSize: 256) == false)
        #expect(mask.needsApply)
        #expect(GuidedGenerationLoop.maskAllows(mask, tokenId: 0x61, vocabSize: 256))

        for byte in #""}"#.utf8 {
            _ = try constraint.commitToken(Int32(byte))
        }
        #expect(stopBits(try constraint.computeMask(), stops: Self.stops, vocabSize: 256) == true)
    }

    /// A mask with every bit set has no exclusions; after the defense it has.
    @Test func allOnesMaskGetsExclusions() throws {
        let constraint = try Self.constraint(hook: { mask in
            MaskResult(
                mask: [Int32](repeating: -1, count: mask.mask.count),
                isTerminated: mask.isTerminated, needsApply: false)
        })
        _ = try constraint.commitToken(Int32(UInt8(ascii: "{")))
        let mask = try constraint.computeMask()
        #expect(mask.needsApply)
        #expect(stopBits(mask, stops: Self.stops, vocabSize: 256) == false)
        #expect(GuidedGenerationLoop.maskAllows(mask, tokenId: 0x22, vocabSize: 256))
    }

    /// The model prefers the stop `x` inside the string, and its bit is set
    /// there. The run still closes the JSON instead of stopping on `x`.
    @Test func setStopBitDoesNotEndTheRunInsideAString() throws {
        try Stream.withNewDefaultStream(device: .cpu) {
            let run = try runLoop(
                source: .jsonSchema(Self.stringSchema),
                policy: .then(prefix: #"{"s":""#, .prefer([0x78, 0x22, 0x7D, 0xFF])),
                maxTokens: 64, eosTokenIds: [0x78], rawMaskHook: settingBits(Self.stops))
            #expect(run.error == nil)
            #expect(run.text == #"{"s":""}"#)
        }
    }

    /// The cap lands inside the string, where the raw EOS bit is set. The cap
    /// check asks the matcher, not the mask, so the output is incomplete.
    @Test func setStopBitAtTheCapIsStillIncomplete() throws {
        try Stream.withNewDefaultStream(device: .cpu) {
            let run = try runLoop(
                source: .grammar(#"root ::= "{\"a\":\"" [a-z]* "\"}""#),
                policy: .script(#"{"a":"bc"}"#), maxTokens: 7, fastForward: false,
                rawMaskHook: settingBits([HarnessTokenizer.eosTokenId]))
            #expect((run.error as? GuidedGenerationError) == .incompleteOutput)
            #expect(run.text == #"{"a":"b"#)
        }
    }
}
