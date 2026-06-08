// Copyright © 2026 Apple Inc.
//
// Rollback determinism.
//
// Asserts xgrammar's `GrammarMatcher::Rollback(n)` restores the
// matcher state so the next mask is bit-identical to the mask
// observed before the rolled-back commits. This is an
// intra-backend self-consistency check — no cross-library
// comparison, so bit-exact mask equality is the appropriate bar
// (the mid-string mask-drift sources documented in GoldenReplayTests
// apply between xgrammar and the recorded backend, not within xgrammar).
//
// The rollback is driven from the tier1 replay fixture: it already
// carries a 3-property flat-object schema and a verified commit
// sequence that advances xgrammar through non-terminal steps. The
// test snapshots the mask after K initial commits, commits M
// additional ones, rolls back M, and compares.
//
// Gated on both traits because the tokenizer path routes through
// the same `loadTestModelContainer` as the other tests.

#if GuidedGenerationSupport && FoundationModelsIntegration

    import Testing
    import Foundation
    import MLXLMCommon
    @testable import MLXFoundationModels

    @Suite(.serialized)
    struct RollbackDeterminismTests {

        @Test("rolling back N commits restores the pre-commit mask bit-for-bit")
        func testRollbackProducesBitIdenticalMask() async throws {
            guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
            // Reuse tier1: smallest fixture (11 steps), known good commit
            // sequence that xgrammar accepts end-to-end.
            let fixture = try loadReplayFixture(named: "schema_tier1_steps.json")

            let container = try await loadTestModelContainer(id: fixture.modelId)
            try await container.perform { context in
                let vocab = TokenizerVocabExtractor.extractForXGrammar(from: context.tokenizer)
                let tokenizer = try XGTokenizer(
                    vocab: vocab.vocab,
                    vocabType: vocab.vocabType,
                    eosTokenId: Int32(context.tokenizer.eosTokenId ?? 0)
                )
                let constraint = try XGConstraint(
                    tokenizer: tokenizer,
                    jsonSchema: fixture.schema,
                    fastForward: true,
                    hostTokenizer: context.tokenizer
                )

                // Walk K initial commits to reach a non-trivial mid-document
                // state; snapshot the mask; commit M more; roll back the
                // total number of tokens xgrammar accepted during those M
                // commits (including fast-forward tokens). Both K and M
                // stay in the non-terminal region.
                let committableSteps = fixture.steps.filter {
                    !$0.terminal && $0.committedTokenId != nil
                }
                guard committableSteps.count >= 5 else {
                    Issue.record(
                        "tier1 fixture has \(committableSteps.count) committable steps; need ≥ 5")
                    return
                }
                let k = 3
                let m = 2
                #expect(k + m <= committableSteps.count)

                for step in committableSteps.prefix(k) {
                    _ = try constraint.commitToken(Int32(step.committedTokenId!))
                }

                let pre = try constraint.computeMask()

                // Count every token xgrammar accepted during the M commits —
                // 1 for the sampled token itself + whatever FF tokens the
                // matcher emitted. Rollback operates on xgrammar's actual
                // acceptance count, not Swift commit calls.
                var acceptedDuringM = 0
                for step in committableSteps.dropFirst(k).prefix(m) {
                    let result = try constraint.commitToken(Int32(step.committedTokenId!))
                    acceptedDuringM += 1 + result.tokens.count
                }

                try constraint.rollback(Int32(acceptedDuringM))

                let post = try constraint.computeMask()

                // Bit-identical mask equality on the raw Int32 words: this
                // is the strongest possible intra-backend check and the
                // point of the test.
                #expect(
                    post.mask == pre.mask,
                    "rollback(\(acceptedDuringM)) must restore the mask bit-for-bit; pre-commit and post-rollback masks diverged"
                )
                #expect(
                    post.isTerminated == pre.isTerminated,
                    "rollback(\(acceptedDuringM)) must restore isTerminated; expected \(pre.isTerminated), got \(post.isTerminated)"
                )
            }
        }
    }

    // MARK: - Shared fixture loader
    //
    // Mirrors GoldenReplayTests' private loader. Kept in this file rather
    // than elevated to a common helper because only this suite consumes it
    // today, and a premature helper extraction would obscure the per-test
    // intent. Promote to a shared helper if a third caller shows up.

    private struct ReplayFixture {
        let modelId: String
        let schema: String
        let steps: [ReplayFixtureStep]
    }

    private struct ReplayFixtureStep {
        let stepIndex: Int
        let committedTokenId: Int?
        let terminal: Bool
    }

    private func loadReplayFixture(named filename: String) throws -> ReplayFixture {
        let base = (filename as NSString).deletingPathExtension
        let ext = (filename as NSString).pathExtension
        guard let url = fixturesBundle.url(forResource: base, withExtension: ext) else {
            throw NSError(
                domain: "RollbackDeterminismTests", code: 1,
                userInfo: [NSLocalizedDescriptionKey: "\(filename) missing from bundle"])
        }
        let data = try Data(contentsOf: url)
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
            let modelId = json["modelId"] as? String,
            let schema = json["schema"] as? String,
            let stepsRaw = json["steps"] as? [[String: Any]]
        else {
            throw NSError(
                domain: "RollbackDeterminismTests", code: 2,
                userInfo: [NSLocalizedDescriptionKey: "\(filename) malformed"])
        }
        let steps: [ReplayFixtureStep] = stepsRaw.compactMap { raw in
            guard let idx = raw["stepIndex"] as? Int else { return nil }
            let terminal = (raw["terminal"] as? Bool) ?? false
            let tokenId = raw["committedTokenId"] as? Int
            return ReplayFixtureStep(stepIndex: idx, committedTokenId: tokenId, terminal: terminal)
        }
        return ReplayFixture(modelId: modelId, schema: schema, steps: steps)
    }

#endif
