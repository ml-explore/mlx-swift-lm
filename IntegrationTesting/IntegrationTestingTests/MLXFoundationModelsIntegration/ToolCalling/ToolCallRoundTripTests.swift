// Copyright © 2026 Apple Inc.
//
// Qwen tool-calling structural-tag round-trip — runtime self-consistency.
//
// Loads the live Qwen2.5-3B tokenizer, compiles the structural-tag JSON
// emitted by `SchemaConverter.encodeToolCallingGrammar` into an
// `GrammarConstraint`, and asserts the integration is wired up end-to-end:
//
//   1. The structural tag compiles without throwing (xgrammar accepts
//      the JSON we synthesize).
//   2. The freshly constructed matcher is live: not terminated, and the
//      initial mask carries at least one accepted token.
//   3. Qwen's `<tool_call>` special token is reachable in the initial
//      mask. This is the integration claim the test exists to defend —
//      the structural_tag's `begin: "<tool_call>\n"` field has to land on
//      Qwen's trained special token, not on a byte-fallback decomposition.
//   4. Committing the `<tool_call>` token does not throw and leaves the
//      matcher live (envelope content still pending). A regression that
//      excludes `<tool_call>` from the wrapped arm surfaces as either a
//      reachability miss or a `commitToken` rejection.
//
// This is a **self-consistency** test, not a cross-backend parity test.
// The runtime checks here cover the integration claim without depending
// on a frozen reference fixture.
//
// Suite is `.serialized`: the test loads `ModelContainer`, and we don't
// want to race on `ModelContainer.perform` isolation with concurrently
// running suites.
//
// Gated on both traits because the tokenizer path routes through
// `loadTestModelContainer` and the schema path requires `@Generable`,
// which is behind `FoundationModelsIntegration`.

#if FoundationModelsIntegration

import Testing
import Foundation
import MLXLMCommon
import FoundationModels
@testable import MLXFoundationModels
@testable import MLXGuidedGeneration

/// Must live at file scope so `@Generable` can emit the schema outside
/// a function body.
@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
@Generable
private struct WeatherArgs {
    @Guide(description: "City and state, e.g. 'San Francisco, CA'.")
    var location: String
}

@Suite(.serialized)
struct ToolCallRoundTripTests {

    @Test(
        "Qwen tool-call structural-tag compiles, exposes <tool_call>, and accepts a <tool_call> commit"
    )
    func testQwenToolCallStructuralTagReachabilityAndCommit() async throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let weather = Transcript.ToolDefinition(
            name: "get_weather",
            description: "Get current weather",
            parameters: WeatherArgs.generationSchema
        )
        let structuralTag = try SchemaConverter.encodeToolCallingGrammar(tools: [weather])

        let container = try await loadTestModelContainer(id: TestFixtures.defaultModelID)
        try await container.perform { context in
            let vocab = TokenizerVocabExtractor.extractForGrammar(from: context.tokenizer)
            let tokenizer = try GrammarTokenizer(
                vocab: vocab.vocab,
                vocabType: vocab.vocabType,
                eosTokenId: Int32(context.tokenizer.eosTokenId ?? 0)
            )

            // fastForward: false so commitToken advances exactly one
            // token without auto-emitting jump-forward ids. Compile-time
            // error on malformed structural tag would surface here as a
            // thrown GrammarError.
            let constraint = try GrammarConstraint(
                tokenizer: tokenizer,
                structuralTag: structuralTag,
                fastForward: false,
                hostTokenizer: context.tokenizer
            )

            // 1. Compile + initial mask: matcher is live and not empty.
            let initial = try constraint.computeMask()
            #expect(!initial.isTerminated, "freshly constructed matcher must not be terminated")
            #expect(
                initial.mask.contains(where: { $0 != 0 }),
                "initial mask must have at least one accepted token for the tool-call structural tag"
            )

            // 2. Qwen's `<tool_call>` special token resolves through the
            //    live tokenizer. Use convertTokenToId rather than
            //    tokenizer.encode(text:...): on Qwen2.5,
            //    encode(text:"<tool_call>...", addSpecialTokens:false)
            //    BPE-decomposes the literal into raw bytes (e.g., '<',
            //    'tool_call', '>') instead of returning the trained
            //    special-token id.
            guard let toolCallId = context.tokenizer.convertTokenToId("<tool_call>") else {
                Issue.record(
                    "Qwen tokenizer (\(TestFixtures.defaultModelID)) did not resolve '<tool_call>' as a special token; structural-tag begin field cannot dispatch through the trained pathway"
                )
                return
            }

            // 3. Reachability: the structural-tag's `begin: "<tool_call>\n"`
            //    must expose the trained `<tool_call>` token in the mask.
            //    A regression that drops the wrapped arm or mistypes the
            //    begin field surfaces here.
            #expect(
                Self.isBitSet(in: initial.mask, at: Int32(toolCallId)),
                "<tool_call> token id \(toolCallId) must be reachable in the initial structural-tag mask on \(TestFixtures.defaultModelID)"
            )

            // 4. Drive forward through `<tool_call>`. The matcher must
            //    accept the token (commitToken throws on rejection) and
            //    remain live afterwards (still expecting `\n` + the
            //    embedded envelope, then `\n</tool_call>`).
            let commit = try constraint.commitToken(Int32(toolCallId))
            #expect(
                !commit.isTerminated,
                "matcher must remain live after committing <tool_call>; envelope content still pending"
            )
        }
    }

    /// Returns true iff bit `tokenId` is set in an xgrammar bitmask.
    /// Words are LSB-first: bit `i` of word `w` is token `w * 32 + i`.
    private static func isBitSet(in mask: [Int32], at tokenId: Int32) -> Bool {
        let wordIndex = Int(tokenId) / 32
        let bit = Int(tokenId) % 32
        guard wordIndex >= 0, wordIndex < mask.count else { return false }
        let uword = UInt32(bitPattern: mask[wordIndex])
        return (uword >> bit) & 1 == 1
    }
}

#endif
