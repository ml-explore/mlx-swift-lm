// Copyright © 2026 Apple Inc.

import MLX
import Testing

@testable import MLXGuidedGeneration

/// A stop id ends the run only where the grammar can end: the grammar knows
/// the loop's stop ids and never allows them as text.
@Suite(.serialized)
struct DeclaredStopLoopTests {

    private static let stringSchema =
        #"{"type":"object","properties":{"s":{"type":"string"}},"required":["s"],"additionalProperties":false}"#

    /// `x` is a stop id here. Inside the string the grammar masks it, so the
    /// model's next choices close the string and the object.
    @Test func secondaryStopInsideStringIsMasked() throws {
        try Stream.withNewDefaultStream(device: .cpu) {
            let run = try runLoop(
                source: .jsonSchema(Self.stringSchema),
                policy: .then(prefix: #"{"s":""#, .prefer([0x78, 0x22, 0x7D, 0xFF])),
                maxTokens: 64, eosTokenIds: [0x78])
            #expect(run.error == nil)
            #expect(run.text == #"{"s":""}"#)
        }
    }

    /// After `a` the grammar can end, so it allows the stop id `x`, and `x`
    /// stops the run.
    @Test func secondaryStopAtCompleteStateStillStops() throws {
        try Stream.withNewDefaultStream(device: .cpu) {
            let run = try runLoop(
                source: .grammar(#"root ::= "a" "x"?"#),
                policy: .then(prefix: "a", .prefer([0x78])),
                maxTokens: 64, eosTokenIds: [0x78])
            #expect(run.error == nil)
            #expect(run.text == "a")
        }
    }

    /// `x` is never a letter here: the first mask excludes it, and after `a` the
    /// grammar can stop and `x` ends the run.
    @Test func stopIsNeverTextInTheFirstMask() throws {
        try Stream.withNewDefaultStream(device: .cpu) {
            let run = try runLoop(
                source: .grammar(#"root ::= [a-z]+"#), policy: .prefer([0x78]),
                maxTokens: 64, eosTokenIds: [0x78])
            #expect(run.error == nil)
            #expect(run.text == "a")
        }
    }

    /// After `a`, fast-forward emits `b`, and the grammar then needs `x` as text.
    /// `x` is a stop id, so nothing is allowed: the run fails before sampling
    /// instead of returning `ab` as complete.
    @Test func stopNeededAsTextAfterFastForwardThrows() throws {
        try Stream.withNewDefaultStream(device: .cpu) {
            let run = try runLoop(
                source: .grammar(#"root ::= "abx" [a-z]*"#), policy: .script("abxy"),
                maxTokens: 64, eosTokenIds: [0x78])
            #expect(
                (run.error as? GuidedGenerationError)
                    == .noAllowedToken(stopTokenIds: [0x78, HarnessTokenizer.eosTokenId]))
            #expect(run.text == "ab")
        }
    }
}
