// Copyright © 2026 Apple Inc.
//
// Loop invariants on the xgrammar-backed bridge.
//
// Verifies the Loop's constraint contract: the sequence of operations
// the Loop performs on a constraint each decode step. The Loop accepts
// `GrammarConstraint` and reads `mask.sampleMask`
// (`UnsafePointer<UInt32>?`) before handing it to
// `applyMaskAndSample`. `MaskResult.mask` is a Swift `[Int32]`
// array — same wire shape (LSB-first int32 bitmask words) but a
// different Swift surface. The rebind from `[Int32]` to
// `UnsafePointer<UInt32>` is the moving part this test exercises.
//
// The test here composes that rebind end-to-end on live gemma-3
// infrastructure:
//   1. Build an GrammarConstraint bound to the gemma-3 tokenizer with a
//      permissive `{"type":"object"}` schema.
//   2. Compute the initial mask and walk its words to find a valid
//      token (non-empty bitmask precondition — already a property
//      asserted by `testXGConstraintSchemaRoundTrip`, re-asserted
//      here to fail loudly in this context if it ever regresses).
//   3. Synthesize uniform logits, rebind the mask's int32 buffer to
//      `UInt32` (the pointer type `applyMaskAndSample` requires),
//      and call into the Loop helper.
//   4. Assert that the sampled token is actually in the grammar's
//      allow-set (i.e. applyMaskAndSample correctly honored the
//      xgrammar-sourced mask after the rebind — the rebind is a bit
//      cast, not a conversion, so any mismatch would surface as a
//      disallowed token winning argmax).
//   5. Commit the sampled token via the constraint and confirm the
//      matcher advanced without terminating, demonstrating the
//      constraint's `commitToken` return value shape (`CommitResult`)
//      is consumable in the same position the Loop's commit-handling
//      code reads it.
//
// Gated on FoundationModelsIntegration: the tokenizer path goes through
// `loadTestModelContainer`. `GrammarConstraint` lives in the
// MLXGuidedGeneration library and is always available alongside the adapter.

#if FoundationModelsIntegration

    import Testing
    import Foundation
    import MLX
    import MLXLMCommon
    @testable import MLXFoundationModels
    @testable import MLXGuidedGeneration

    @Suite(.serialized)
    struct LoopInvariantsOnXGrammarTests {

        @Test("GrammarConstraint satisfies GuidedGenerationLoop's constraint contract end-to-end")
        func testLoopConstraintContractComposesWithXGConstraint() async throws {
            guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
            let container = try await loadTestModelContainer(id: TestFixtures.gemmaModelID)

            try await container.perform { context in
                let vocab = TokenizerVocabExtractor.extractForGrammar(from: context.tokenizer)
                let tokenizer = try GrammarTokenizer(
                    vocab: vocab.vocab,
                    vocabType: vocab.vocabType,
                    eosTokenId: Int32(context.tokenizer.eosTokenId ?? 0)
                )
                let constraint = try GrammarConstraint(
                    tokenizer: tokenizer,
                    jsonSchema: #"{"type":"object"}"#
                )

                // Step 1: Loop's first move each iteration — computeMask.
                // The Loop reads `mask.sampleMask` (UnsafePointer<UInt32>?)
                // and `mask.isStop`. `MaskResult` exposes `mask: [Int32]`
                // and `isTerminated: Bool` for the same semantic roles.
                let xgMask = try constraint.computeMask()
                #expect(
                    !xgMask.isTerminated,
                    "fresh matcher must not be terminated — Loop reads this as `mask.isStop`")
                #expect(
                    xgMask.mask.contains(where: { $0 != 0 }),
                    "open-object schema must have at least one valid next token")

                // Step 2: the Loop synthesizes activeBias / closingBias and
                // hands mask+logits to `applyMaskAndSample`. Here the bias
                // is nil (normal zone), so the helper reduces to
                // "argmax over grammar-allowed tokens". Uniform logits make
                // the winner unambiguous — whichever token has the lowest
                // id among allowed tokens wins argmax on ties.
                let vocabSize = Int(tokenizer.vocabSize)
                let uniformLogits = MLXArray(Array(repeating: Float(1.0), count: vocabSize))

                // Rebind [Int32] → UnsafePointer<UInt32>. The xgrammar
                // bitmask is documented as "LSB-first int32 bitmask words",
                // which matches the UInt32 bitmask layout the Loop
                // consumes — only the Swift surface type differs. This is a
                // bit cast, not a conversion.
                let sampledToken: UInt32 = xgMask.mask.withUnsafeBufferPointer { buffer in
                    guard let base = buffer.baseAddress else {
                        Issue.record("empty xgrammar mask buffer")
                        return UInt32.max
                    }
                    let maskArray = base.withMemoryRebound(
                        to: UInt32.self, capacity: buffer.count
                    ) { rebound in
                        GuidedGenerationLoop.bitmaskToMLXArray(
                            rebound, maskBitCount: vocabSize, totalCount: vocabSize)
                    }
                    return GuidedGenerationLoop.applyMaskAndSample(
                        logits: uniformLogits[.newAxis, .newAxis, 0...], maskArray: maskArray)
                }
                #expect(sampledToken != UInt32.max, "applyMaskAndSample failed to produce a token")

                // Step 3: the sampled token must be in the grammar's
                // allow-set. If the rebind introduced any bit-interpretation
                // bug, an out-of-grammar token would win argmax (its logit
                // would read as finite rather than -inf). Core assertion:
                // mask semantics survive the
                // [Int32] → UInt32 pointer rebind unchanged.
                let tokenId = Int(sampledToken)
                let word = Int(tokenId / 32)
                let bit = UInt32(tokenId % 32)
                #expect(
                    word < xgMask.mask.count,
                    "sampled token id \(tokenId) outside mask buffer (\(xgMask.mask.count) words)")
                let isAllowed = (UInt32(bitPattern: xgMask.mask[word]) >> bit) & 1 == 1
                #expect(
                    isAllowed,
                    "sampled token id \(tokenId) is not in the grammar allow-set — mask rebind broke semantics"
                )

                // Step 4: the Loop commits the sampled token through
                // `commitToken`, reads `result.tokens` for fast-forward
                // advancement, and checks `result.isStop` (here,
                // `isTerminated`). `CommitResult` matches that shape.
                let commit = try constraint.commitToken(Int32(sampledToken))
                #expect(
                    !commit.isTerminated,
                    "single-token commit on open-object schema must not terminate the matcher")

                // Step 5: the Loop recomputes the mask after each commit.
                // Verify the constraint is still live and responsive — this
                // is the same invariant as `testXGConstraintSchemaRoundTrip`,
                // checked again here to confirm the contract composes back-
                // to-back without requiring a second constraint.
                let nextMask = try constraint.computeMask()
                #expect(
                    !nextMask.isTerminated,
                    "matcher must remain active after one-token commit+recompute")
                #expect(
                    nextMask.mask.contains(where: { $0 != 0 }),
                    "post-commit mask must still admit some next token")
            }
        }
    }

#endif
