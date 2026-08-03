// Copyright © 2026 Apple Inc.
//
// Fork independence.
//
// Asserts `GrammarConstraint.clone()` returns an independent matcher:
// commits on the fork must not advance the parent's state, and
// commits on the parent must not advance the fork's state. Mirrors
// xgrammar's `GrammarMatcher::Fork()` contract — deep copy of
// per-session state, shared immutable compiled grammar and
// tokenizer — at the Swift wrapper level.
//
// Scenario source. Uses the tier1 replay fixture: smallest viable
// fixture with a known good commit sequence that xgrammar accepts
// end-to-end. The test commits K initial tokens on the parent,
// snapshots, forks, commits one more token on the fork, and checks:
//   - the parent's post-fork mask still equals the pre-fork snapshot
//     (parent untouched by fork's commit)
//   - the fork's post-commit mask differs from the snapshot
//     (fork actually advanced)
//
// Gated on both traits because the tokenizer path routes through
// `loadTestModelContainer`.

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

import Testing
import Foundation
import MLXLMCommon
@testable import MLXFoundationModels
@testable import MLXGuidedGeneration

@Suite(.serialized)
struct ForkIndependenceTests {

    @Test(
        "fork of a matcher diverges from parent on independent commits",
        .disabled(
            """
            xgrammar matcher Fork()/clone() requires xgrammar >= v0.1.34; the vendored \
            version (v0.1.30) does not provide it. Production handles its absence \
            gracefully — makeConstraint() catches forkFailed and recompiles a fresh \
            constraint — so this is a perf-only optimization, not a correctness gap. \
            Re-enable if the vendored xgrammar is bumped to a version with Fork().
            """))
    func testForkDiverges() async throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let fixture = try loadReplayFixture(named: "schema_tier1_steps.json")

        let container = try await loadTestModelContainer(id: fixture.modelId)
        try await container.perform { context in
            let vocab = TokenizerVocabExtractor.extractForGrammar(from: context.tokenizer)
            let tokenizer = try GrammarTokenizer(
                vocab: vocab.vocab,
                vocabType: vocab.vocabType,
                eosTokenId: Int32(context.tokenizer.eosTokenId ?? 0)
            )
            let parent = try GrammarConstraint(
                tokenizer: tokenizer,
                jsonSchema: fixture.schema,
                fastForward: true,
                hostTokenizer: context.tokenizer
            )

            // Drive the parent through a few committable steps so the
            // fork happens at a non-trivial mid-document state.
            let committableSteps = fixture.steps.filter {
                !$0.terminal && $0.committedTokenId != nil
            }
            guard committableSteps.count >= 4 else {
                Issue.record(
                    "tier1 fixture has \(committableSteps.count) committable steps; need ≥ 4")
                return
            }
            let k = 3
            #expect(k + 1 <= committableSteps.count)

            for step in committableSteps.prefix(k) {
                _ = try parent.commitToken(Int32(step.committedTokenId!))
            }

            let preFork = try parent.computeMask()

            // Fork. The two constraints must share compiled grammar
            // (xgrammar's PIMPL + shared_ptr semantics guarantee this),
            // but carry independent matcher state from here on.
            let fork = try parent.clone()

            // Sanity: at fork-time both masks must agree. If they
            // don't, the clone copied nothing (or the wrong thing).
            let forkAtBirth = try fork.computeMask()
            #expect(
                forkAtBirth.mask == preFork.mask,
                "fork-at-birth mask must equal parent's mask at fork time")

            // Commit one more token on the fork only. Use step K+1's
            // committed token, which the fixture already verified
            // xgrammar accepts at this state.
            let nextStep = committableSteps[k]
            guard let nextToken = nextStep.committedTokenId else {
                Issue.record("tier1 step \(nextStep.stepIndex) missing committedTokenId")
                return
            }
            _ = try fork.commitToken(Int32(nextToken))

            // The parent must be unchanged by the fork's commit.
            // Masks are the strongest observable signal: bit-identical
            // equality on the Int32 array.
            let parentAfter = try parent.computeMask()
            #expect(
                parentAfter.mask == preFork.mask,
                "parent's mask must be unchanged by a commit on the fork")
            #expect(
                parentAfter.isTerminated == preFork.isTerminated,
                "parent's isTerminated must be unchanged by a commit on the fork")

            // The fork must have advanced — its post-commit mask
            // differs from the pre-fork snapshot. (Strict inequality,
            // not isTerminated-flip: the next mask is just the
            // grammar's legal-next-token set at a different state.)
            let forkAfter = try fork.computeMask()
            #expect(
                forkAfter.mask != preFork.mask,
                "fork's mask must differ from the pre-fork snapshot after committing a new token"
            )
        }
    }
}

// MARK: - Shared fixture loader
//
// Local copy of RollbackDeterminismTests' loader; promote to a shared
// helper if a third caller appears.

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
            domain: "ForkIndependenceTests", code: 1,
            userInfo: [NSLocalizedDescriptionKey: "\(filename) missing from bundle"])
    }
    let data = try Data(contentsOf: url)
    guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
        let modelId = json["modelId"] as? String,
        let schema = json["schema"] as? String,
        let stepsRaw = json["steps"] as? [[String: Any]]
    else {
        throw NSError(
            domain: "ForkIndependenceTests", code: 2,
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
