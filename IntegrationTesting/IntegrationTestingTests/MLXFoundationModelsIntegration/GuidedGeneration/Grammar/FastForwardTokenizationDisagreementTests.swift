// Copyright © 2026 Apple Inc.
//
// Jump-forward tokenization disagreement graceful fallback.
//
// ## The failure mode
//
// When `fastForward: true`, `GrammarConstraint.commitToken` walks xgrammar's
// `FindJumpForwardString` suffix, asks the host tokenizer to re-encode
// those bytes, and accepts the resulting ids against the matcher one
// at a time. The host tokenizer's encoding decision is a function of
// *its* merge table; xgrammar's FF byte boundary is a function of
// *the grammar's* production rules. The two can disagree: the host
// tokenizer can produce a token whose bytes extend past the FF-forced
// region into grammar-free territory, and the matcher then refuses
// that id on `AcceptToken`. The fallback in `emitFastForwardLocked`
// breaks out of the accept loop without crashing and records the
// disagreement via a counter, preserving the "no crash, generation
// continues" contract.
//
// ## Fixture choice: real-tokenizer cross-wire, not a mock
//
// This uses a misaligned vocab fixture, not a tokenizer mock. A mock
// that synthesizes ids would prove nothing — the
// interesting property is that the disagreement arises from genuine
// tokenizer divergence, not from Swift-side test scaffolding. The
// cross-tokenizer setup here is the minimal such fixture:
//
//   - `GrammarTokenizer` is built from Gemma-3's vocab (byte-fallback
//     SentencePiece, ~262k tokens).
//   - `hostTokenizer` passed to `GrammarConstraint` is Qwen2.5-3B's live
//     tokenizer (GPT-2 byte-level BPE, ~152k tokens, different merges).
//
// Every id Qwen produces for the FF string bytes is reinterpreted by
// xgrammar against Gemma's vocab table. For any realistic FF string
// (JSON punctuation + keys), at least one Qwen id lands on a Gemma
// token whose bytes don't match the FF-forced bytes, and xgrammar's
// mask rejects it. That single rejection is all we need to observe.
//
// ## Grammar choice: EBNF with a strictly forced byte sequence
//
// JSON Schema compiles into an xgrammar automaton that permits
// whitespace around structural tokens. Any permitted whitespace means
// xgrammar's `FindJumpForwardString` returns an empty suffix —
// nothing is *strictly* forced to come next, because the grammar
// accepts whitespace as an alternative. On-device diagnostic probes
// confirmed `ff_length == 0` on every commit for both open-object
// and required-const JSON schemas, so those shapes cannot exercise
// the FF path at all.
//
// An EBNF grammar with a literal string production
// (`root ::= "payload"`) has no whitespace alternative. Every byte
// after the first commit is forced, so xgrammar emits the remainder
// as its jump-forward suffix. The payload below is 32 bytes of
// ASCII chosen to guarantee Qwen's BPE breaks it into multiple
// tokens (mixed case + digits defeats merge-table shortcuts that
// would produce a single whole-string token).
//
// ## The committed-token seed: Gemma's `p`
//
// To enter a state with a non-empty FF suffix, we commit the first
// byte of the payload. `GrammarConstraint` is bound to Gemma's vocab, so
// the seed must be a Gemma id. Gemma encodes literal `p` as a
// specific token id; we look it up via
// `tokenizer.convertTokenToId("p")` so this test survives vocab
// rebuilds without hand-rolled constants. If that lookup ever
// returns nil the test surfaces the broken assumption rather than
// silently skipping.
//
// ## What this test asserts
//
// 1. `constraint.fastForwardDisagreementCount == 0` at construction.
// 2. After one `commitToken(gemmaSeed)` call, the counter is
//    strictly greater than zero — at least one FF accept step saw
//    a Qwen-encoded id that the Gemma-bound matcher rejected.
// 3. The commit itself returned a `CommitResult` — the test did
//    not crash or throw.
//
// Assertion (2) holds because `emitFastForwardLocked` increments the
// counter on the `acceptStatus != XG_OK` branch.
//
// ## What this test does NOT assert
//
// - The exact number of disagreements. xgrammar's FF suffix length
//   and Qwen's tokenization of it are implementation-dependent; pinning
//   an exact count would make the test brittle to upstream tokenizer
//   or grammar changes that don't affect the correctness of the
//   fallback itself.
// - The specific tokens that disagreed. Same rationale.
// - Full generation continuation. The "generation continues"
//   guarantee is covered by the Loop-level integration tests; here we only
//   validate the bridge-level contract that the FF accept loop
//   survives a rejection and the constraint remains usable.
//
// Gated on FoundationModelsIntegration: tokenizer paths go through
// `loadTestModelContainer`. The GrammarConstraint type lives in the
// MLXGuidedGeneration library and is always available alongside the adapter.

#if FoundationModelsIntegration

import Testing
import Foundation
import MLXLMCommon
@testable import MLXFoundationModels
@testable import MLXGuidedGeneration

@Suite(.serialized)
struct FastForwardTokenizationDisagreementTests {

    private enum MissingSeedError: Error {
        /// Raised when Gemma's tokenizer has no id for the seed character.
        /// Surfacing this as an error rather than just an `Issue.record`
        /// lets the outer `perform` unwind cleanly instead of continuing
        /// into a test-body that depends on the seed id being present.
        case seedIdUnavailable
    }

    /// Sendable bundle of everything we need from Gemma's container so
    /// the second `perform` (on Qwen) can build `GrammarTokenizer` and issue
    /// the seed commit without capturing Gemma's non-Sendable
    /// `ModelContext`. Every field is already Sendable: `[String]`,
    /// the C enum, and `Int` primitives.
    private struct GemmaSeeds: Sendable {
        let vocab: [String]
        let vocabType: VocabType
        let eosTokenId: Int32
        let seedTokenId: Int32
    }

    /// Payload string for the forced-byte EBNF grammar. First byte is
    /// `p` — used as the seed token (encoded on Gemma). The remaining
    /// 31 bytes become xgrammar's FF suffix after the seed commit. The
    /// mixed case + digits shape defeats single-token BPE shortcuts on
    /// both Gemma and Qwen, ensuring Qwen's re-encoding produces
    /// multiple tokens for the boundary-safety trim to leave some
    /// in-bounds for the accept loop.
    private static let forcedPayload = "payLoadABC123payLoadDEF456payLoad"

    @Test("mid-FF tokenization disagreement ticks the counter without crashing")
    func testJumpForwardTokenizationDisagreementFallsBackCleanly() async throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let gemmaContainer = try await loadTestModelContainer(id: TestFixtures.gemmaModelID)
        let qwenContainer = try await loadTestModelContainer(id: TestFixtures.defaultModelID)

        let seeds: GemmaSeeds = try await gemmaContainer.perform { gemmaContext in
            let gemmaVocab = TokenizerVocabExtractor.extractForGrammar(
                from: gemmaContext.tokenizer
            )
            let encoded = gemmaContext.tokenizer.encode(
                text: String(Self.forcedPayload.prefix(1)),
                addSpecialTokens: false
            )
            guard let firstId = encoded.first else {
                Issue.record("Gemma tokenizer produced no id for seed byte `p`")
                throw MissingSeedError.seedIdUnavailable
            }
            return GemmaSeeds(
                vocab: gemmaVocab.vocab,
                vocabType: gemmaVocab.vocabType,
                eosTokenId: Int32(gemmaContext.tokenizer.eosTokenId ?? 0),
                seedTokenId: Int32(firstId)
            )
        }

        try await qwenContainer.perform { qwenContext in
            let xgTokenizer = try GrammarTokenizer(
                vocab: seeds.vocab,
                vocabType: seeds.vocabType,
                eosTokenId: seeds.eosTokenId
            )

            // Cross-wire: GrammarTokenizer is Gemma, hostTokenizer is Qwen.
            // Qwen's re-encoding of the FF bytes will land on ids the
            // Gemma-bound matcher does not have in its current mask.
            let grammar = "root ::= \"\(Self.forcedPayload)\"\n"
            let constraint = try GrammarConstraint(
                tokenizer: xgTokenizer,
                grammar: grammar,
                fastForward: true,
                hostTokenizer: qwenContext.tokenizer
            )

            #expect(
                constraint.fastForwardDisagreementCount == 0,
                "fresh constraint must report zero FF disagreements"
            )

            // Commit the seed byte. xgrammar's FF pass then surfaces
            // the remaining 31 bytes of the forced payload, which Qwen
            // re-encodes into ids the Gemma-bound matcher rejects —
            // the disagreement path we want to observe.
            _ = try constraint.commitToken(seeds.seedTokenId)

            #expect(
                constraint.fastForwardDisagreementCount > 0,
                "cross-tokenizer FF must produce at least one rejection — counter stayed at \(constraint.fastForwardDisagreementCount)"
            )
        }
    }
}

#endif
