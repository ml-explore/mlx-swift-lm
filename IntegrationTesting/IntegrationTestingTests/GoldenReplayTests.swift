// Copyright © 2026 Apple Inc.
//
// Functional-parity replay against recorded goldens.
//
// Drives each of the four tier fixtures through the xgrammar bridge
// step-by-step and asserts that xgrammar's behavior matches the
// captured goldens at the level the two backends actually agree on:
//
//   - Termination lifecycle: isTerminated must match the fixture on
//     every non-terminal step, and on commit on every non-terminal
//     step. The terminal step's post-final-commit maskIsStop is NOT
//     asserted — see the structural-divergence note below.
//   - Functional token-mask superset: every token the reference
//     committed, xgrammar must also accept. Enforced implicitly —
//     commitToken throws XGError.invalidArgument if xgrammar's mask
//     rejected a token the reference accepted.
//   - Non-empty mask on live matcher: non-terminal steps must offer
//     at least one valid token (an empty mask on a live matcher is
//     an xgrammar-side bug).
//   - Fast-forward emission: commit.tokens must equal the fixture's
//     ffTokenIds byte-for-byte. Pins the jump-forward plumbing and
//     the tokenization-boundary logic that converts xgrammar's raw
//     forced byte-suffix into a safe token prefix.
//   - commitIsStop: whether a commit terminated the matcher must match.
//
// What this test intentionally does NOT assert: byte-exact equality
// of the raw mask bits (sha256, allowedCount, allowedSample), nor the
// post-final-commit terminal-step maskIsStop. xgrammar's special-token
// handling and adaptive mask legitimately diverge from the recorded
// reference. Three structural sources of drift:
//   1. xgrammar correctly excludes empty-decoded / stop tokens
//      mid-grammar via TokenizerInfo's IsSpecialToken check;
//      the reference sample_mask includes them.
//   2. xgrammar uses a precomputed AdaptiveTokenMask that over-permits
//      tokens whose first byte is locally legal but which wedge the
//      parser downstream; the reference rejected those via deeper
//      prefix-aware analysis.
//   3. Post-final-commit terminal state: the reference flipped
//      maskIsStop to true when the next-token mask contained only
//      EOS/stop tokens (an "about-to-stop" signal computed from the
//      mask). xgrammar's IsTerminated() stays false until an explicit
//      EOS commit; the matcher is still "live, accepting EOS." Both
//      agree the document is complete — they disagree only on when the
//      terminated flag flips relative to the unsampled EOS. The
//      fixture's last step captures the reference's eager flip;
//      xgrammar would need an additional EOS commit the fixture did
//      not record.
// Neither difference changes the set of JSON documents either backend
// will ultimately accept, and neither is configurable. xgrammar's
// public FillNextTokenBitmask does not expose allow_special_token,
// and the adaptive-vs-prefix distinction is a design axiom of the
// two libraries. The residual functional checks above are strong
// enough to catch real regressions: a narrowing of xgrammar's mask
// below what the reference committed surfaces as a commit-failure
// throw, not as silent drift.
//
// The fixture schema carries sha256, allowedCount, and allowedSample
// as required fields. They are simply not asserted against; they
// remain available for future diagnostic work or for a stricter check
// once xgrammar gains a prefix-aware mask mode.
//
// Suite is `.serialized`: the tier runs all load the same model
// container and we do not want to race on `ModelContainer.perform`
// isolation or on the xgrammar compiler cache.
//
// Gated on both traits because the tokenizer path routes through the
// same `loadTestModelContainer` as the bridge tests.

#if GuidedGenerationSupport && FoundationModelsIntegration

    import Testing
    import Foundation
    import MLXLMCommon
    @testable import MLXFoundationModels

    @Suite(.serialized)
    struct GoldenReplayTests {

        @Test(
            "tier1 (~11 steps, 3-property flat object) replays with functional parity against goldens"
        )
        func testTier1() async throws {
            guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
            try await replayTier(fixture: "schema_tier1_steps.json")
        }

        @Test(
            "tier2 (~28 steps, nested optional object) replays with functional parity against goldens"
        )
        func testTier2() async throws {
            guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
            try await replayTier(fixture: "schema_tier2_steps.json")
        }

        @Test(
            "tier3 (~54 steps, array of keyed groups) replays with functional parity against goldens"
        )
        func testTier3() async throws {
            guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
            try await replayTier(fixture: "schema_tier3_steps.json")
        }

        @Test(
            "tier4 (~132 steps, multi-section travel doc) replays with functional parity against goldens"
        )
        func testTier4() async throws {
            guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
            try await replayTier(fixture: "schema_tier4_steps.json")
        }

        // MARK: - Replay

        /// Load the named fixture, construct an XGConstraint against its
        /// recorded schema on the live tokenizer, and walk the fixture's
        /// steps asserting per-step functional parity. Each commit
        /// implicitly verifies the token the recorded backend accepted at
        /// this step is also in xgrammar's mask; the explicit checks cover
        /// termination, fast-forward emission, and commit-stop lifecycle.
        /// A passing run means xgrammar matched the recorded behavior on
        /// every externally-observable property for the full document.
        private func replayTier(fixture filename: String) async throws {
            guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
            let fixture = try Self.loadFixture(named: filename)
            // All four tier fixtures were recorded against gemma-3;
            // the recorder embeds the modelId for portability across future
            // multi-tokenizer fixtures. Verify before we load the wrong
            // container and silently compare against mismatched vocab.
            #expect(
                fixture.modelId == TestFixtures.gemmaModelID,
                "golden fixture \(filename) has modelId \(fixture.modelId); expected \(TestFixtures.gemmaModelID). This replay assumes gemma-3 for all four tiers."
            )
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

                for step in fixture.steps {
                    let observed = try constraint.computeMask()

                    if step.terminal {
                        // Terminal step has no commit; the fixture's last
                        // record captures the post-final-commit state and
                        // ends. maskIsStop is NOT asserted here because
                        // xgrammar's IsTerminated() only flips on an explicit
                        // EOS commit. See the header note for the lifecycle
                        // divergence.
                        return
                    }

                    // Termination parity on non-terminal steps: before each
                    // commit, the matcher's live/stopped lifecycle state
                    // must match the fixture. Divergence here would mean
                    // xgrammar prematurely stopped, or the recorded backend
                    // stopped on input xgrammar considers live — a real
                    // bug either side.
                    guard observed.isTerminated == step.maskIsStop else {
                        Issue.record(
                            "fixture \(filename) step \(step.stepIndex): maskIsStop divergence — expected \(step.maskIsStop), got \(observed.isTerminated)"
                        )
                        return
                    }

                    // Non-terminal steps must offer at least one valid
                    // token. An empty mask on a live matcher is an
                    // xgrammar-side bug — surfacing it here gives a
                    // clearer diagnostic than the commit-failure throw
                    // that would follow.
                    guard observed.mask.contains(where: { $0 != 0 }) else {
                        Issue.record(
                            "fixture \(filename) step \(step.stepIndex): observed mask is empty on a non-terminal step"
                        )
                        return
                    }

                    guard let committedId = step.committedTokenId else {
                        Issue.record(
                            "fixture \(filename) step \(step.stepIndex): non-terminal step must carry committedTokenId"
                        )
                        return
                    }

                    // Functional superset check: if xgrammar's mask
                    // rejected a token the recorded backend committed,
                    // commitToken throws XGError.invalidArgument and the
                    // test fails with a clear cause, not a silent drift.
                    let commit = try constraint.commitToken(Int32(committedId))

                    // Fast-forward parity: byte-for-byte equality. The
                    // recorder already dropped the committed token
                    // itself, so commit.tokens maps 1:1 to the fixture's
                    // ffTokenIds. Agreement here pins the jump-forward
                    // plumbing and the tokenization-boundary logic that
                    // converts xgrammar's raw forced byte-suffix into a
                    // safe token prefix.
                    let observedFF = commit.tokens.map { Int($0) }
                    guard observedFF == step.ffTokenIds else {
                        Issue.record(
                            "fixture \(filename) step \(step.stepIndex): ffTokenIds divergence — expected \(step.ffTokenIds), got \(observedFF)"
                        )
                        return
                    }

                    let expectedCommitIsStop = step.commitIsStop ?? false
                    guard commit.isTerminated == expectedCommitIsStop else {
                        Issue.record(
                            "fixture \(filename) step \(step.stepIndex): commitIsStop divergence — expected \(expectedCommitIsStop), got \(commit.isTerminated)"
                        )
                        return
                    }
                }
            }
        }

        // MARK: - Fixture loading

        private struct Fixture {
            let modelId: String
            let schema: String
            let document: String
            let steps: [FixtureStep]
        }

        private struct FixtureStep {
            let stepIndex: Int
            let maskSha256: String
            let maskAllowedCount: Int
            let maskAllowedSample: [Int]
            let maskIsStop: Bool
            /// nil on the terminal step (the recorder writes
            /// `"committedTokenId": null`).
            let committedTokenId: Int?
            let ffTokenIds: [Int]
            /// nil on the terminal step.
            let commitIsStop: Bool?
            let terminal: Bool
        }

        private static func loadFixture(named filename: String) throws -> Fixture {
            // Goldens are bundled as processed resources (see Package.swift
            // `resources: [.process("Fixtures")]`). `#filePath` does not resolve on
            // on-device runs — the test process lives in the iOS sandbox.
            let base = (filename as NSString).deletingPathExtension
            let ext = (filename as NSString).pathExtension
            guard let url = fixturesBundle.url(forResource: base, withExtension: ext) else {
                throw FixtureError.malformed("\(filename): missing from test bundle resources")
            }
            let data = try Data(contentsOf: url)
            guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
                throw FixtureError.malformed("\(filename): top-level not an object")
            }
            guard let modelId = json["modelId"] as? String,
                let schema = json["schema"] as? String,
                let document = json["document"] as? String,
                let stepsRaw = json["steps"] as? [[String: Any]]
            else {
                throw FixtureError.malformed("\(filename): missing modelId/schema/document/steps")
            }

            var steps: [FixtureStep] = []
            steps.reserveCapacity(stepsRaw.count)
            for (i, raw) in stepsRaw.enumerated() {
                guard let stepIndex = raw["stepIndex"] as? Int,
                    let maskSha256 = raw["maskSha256"] as? String,
                    let maskAllowedCount = raw["maskAllowedCount"] as? Int,
                    let maskAllowedSample = raw["maskAllowedSample"] as? [Int],
                    let maskIsStop = raw["maskIsStop"] as? Bool,
                    let ffTokenIds = raw["ffTokenIds"] as? [Int]
                else {
                    throw FixtureError.malformed("\(filename): step \(i) missing required fields")
                }
                let terminal = (raw["terminal"] as? Bool) ?? false
                // committedTokenId / commitIsStop arrive as NSNull on the
                // terminal step; JSONSerialization surfaces NSNull, not
                // absent key, so test `is NSNull` explicitly.
                let committedTokenId: Int? = (raw["committedTokenId"] as? Int)
                let commitIsStop: Bool? = (raw["commitIsStop"] as? Bool)

                steps.append(
                    FixtureStep(
                        stepIndex: stepIndex,
                        maskSha256: maskSha256,
                        maskAllowedCount: maskAllowedCount,
                        maskAllowedSample: maskAllowedSample,
                        maskIsStop: maskIsStop,
                        committedTokenId: committedTokenId,
                        ffTokenIds: ffTokenIds,
                        commitIsStop: commitIsStop,
                        terminal: terminal
                    ))
            }

            return Fixture(modelId: modelId, schema: schema, document: document, steps: steps)
        }

        private enum FixtureError: Error {
            case malformed(String)
        }
    }

#endif
