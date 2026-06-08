// Copyright © 2026 Apple Inc.
//
// Tests for XGrammarBridge Swift wrappers over the CXGrammar C shim.
//
// XGTokenizer construction against live production vocabularies.
// Each test loads a HuggingFace tokenizer via the shared test loader,
// feeds its vocab through `TokenizerVocabExtractor.extractForXGrammar`,
// and constructs an `XGTokenizer` bound to xgrammar's TokenizerInfo.
// The contract under test is:
//   - construction succeeds on a real vocab containing byte-fallback
//     / byte-level-encoded tokens
//   - `XGTokenizer.vocabSize` matches the recorded fixture metadata,
//     which pins the downloader / loader pair to a known snapshot so
//     silent vocab drift surfaces here and not deep inside mask tests.
//
// XGConstraint end-to-end round-trip. Builds on the tokenizer by
// compiling a minimal JSON schema, computing a mask, committing a
// grammar-accepted token, and recomputing. Asserts the matcher is not
// terminated and the mask is non-empty at both steps.
//
// Single-matcher concurrent-access contract. Spawns two detached
// tasks hammering `computeMask`/`commitToken` on one `XGConstraint`;
// asserts the bridge serializes the C-level matcher state so neither
// task crashes and the constraint remains operational afterward. The
// safety is provided by a Swift-side NSLock — xgrammar's matcher is
// not thread-safe, and without serialization concurrent AcceptToken
// calls race on internal PIMPL state.
//
// Exception-unwinding smoke test. Triggers an
// `InvalidJSONSchemaError` deep inside xgrammar's `GrammarCompiler`
// from within a `Task.detached` closure and asserts the shim catches
// it, maps it to the discriminated `XGError.invalidJSONSchema(_)`
// case, and neither crashes the process nor corrupts the detached
// task's stack. C++ exceptions that traverse a Swift -> C -> C++ frame
// chain must not escape the shim; this pins that xgrammar's throwing
// paths survive on-device unwinding.
//
// Gated on `FoundationModelsIntegration` because the live-tokenizer
// path routes through `loadTestModelContainer`; gated on
// `GuidedGenerationSupport` because `XGTokenizer` lives under that
// trait.
//
// Note on coverage: this exercises gemma-3 and qwen2.5; qwen2.5 stands
// in for qwen3 since both are byte-level BPE and the recorded qwen3
// fixture is not yet available. Llama-3 coverage is pending its
// `tokenizer_llama3.json` fixture.

#if GuidedGenerationSupport && FoundationModelsIntegration

    import Testing
    import Foundation
    import MLXLMCommon
    @testable import MLXFoundationModels

    @Suite(.serialized)
    struct XGrammarBridgeTests {

        // MARK: - XGTokenizer construction

        /// Construct XGTokenizer from the live gemma-3 vocab.
        ///
        /// Gemma uses SentencePiece with `<0xNN>` byte-fallback tokens for
        /// bytes that the base vocab doesn't cover. The extractor must hand
        /// xgrammar a representation where those tokens survive the Swift →
        /// C string transport; construction must not throw.
        @Test("XGTokenizer: gemma-3 live vocab constructs; size matches fixture")
        func testXGTokenizerGemma3() async throws {
            guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
            let fixture = try Self.loadTokenizerFixture(named: "tokenizer_gemma3.json")
            let container = try await loadTestModelContainer(id: TestFixtures.gemmaModelID)

            try await container.perform { context in
                let vocab = TokenizerVocabExtractor.extractForXGrammar(from: context.tokenizer)

                let tokenizer = try XGTokenizer(
                    vocab: vocab.vocab,
                    vocabType: vocab.vocabType,
                    eosTokenId: Int32(fixture.eosTokenId)
                )

                #expect(
                    tokenizer.vocabSize == fixture.vocabSize,
                    "XGTokenizer reports vocabSize \(tokenizer.vocabSize); fixture expects \(fixture.vocabSize) for \(TestFixtures.gemmaModelID)"
                )
            }
        }

        /// Construct XGTokenizer from the live qwen2.5 vocab.
        ///
        /// Qwen uses GPT-2 byte-level BPE (via the `bytes_to_unicode` map).
        /// The extractor normalizes those back to raw bytes before handing
        /// them to xgrammar; construction must not throw.
        ///
        /// Stands in for a dedicated qwen3 case until a
        /// `tokenizer_qwen3.json` fixture exists. Same tokenizer family;
        /// mechanically equivalent for byte-level BPE coverage.
        @Test("XGTokenizer: qwen2.5 live vocab constructs; size matches fixture")
        func testXGTokenizerQwen25() async throws {
            guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
            let fixture = try Self.loadTokenizerFixture(named: "tokenizer_qwen25.json")
            let container = try await loadTestModelContainer(id: TestFixtures.defaultModelID)

            try await container.perform { context in
                let vocab = TokenizerVocabExtractor.extractForXGrammar(from: context.tokenizer)

                let tokenizer = try XGTokenizer(
                    vocab: vocab.vocab,
                    vocabType: vocab.vocabType,
                    eosTokenId: Int32(fixture.eosTokenId)
                )

                #expect(
                    tokenizer.vocabSize == fixture.vocabSize,
                    "XGTokenizer reports vocabSize \(tokenizer.vocabSize); fixture expects \(fixture.vocabSize) for \(TestFixtures.defaultModelID)"
                )
            }
        }

        // TODO: add `testXGTokenizerLlama3()` once `tokenizer_llama3.json`
        // lands, for three-tokenizer coverage (gemma-3, qwen3, llama-3).

        // MARK: - XGConstraint schema round-trip

        /// XGConstraint round-trips a JSON schema.
        ///
        /// Compiles `{"type":"object"}` against a live gemma-3 vocab, computes
        /// the initial mask, picks the first grammar-accepted token ID, commits
        /// it, and recomputes. At both steps asserts:
        ///   - matcher is not terminated (open object schema does not accept
        ///     EOS before a single `{` has landed, and does not accept it
        ///     immediately after either)
        ///   - bitmask contains at least one set bit
        ///
        /// The test does not care *which* token is accepted — only that the
        /// round-trip (compile → mask → commit → mask) completes without any
        /// error propagating from the C shim or xgrammar. Golden replay and
        /// exact-state assertions are deferred to a later cycle.
        ///
        /// `flushLogs()` is validated separately as a placeholder returning
        /// `nil`; xgrammar has no log-accumulation stream, so this method
        /// is a typed no-op.
        @Test("XGConstraint: JSON schema round-trips; mask non-empty at both steps")
        func testXGConstraintSchemaRoundTrip() async throws {
            guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
            let fixture = try Self.loadTokenizerFixture(named: "tokenizer_gemma3.json")
            let container = try await loadTestModelContainer(id: TestFixtures.gemmaModelID)

            try await container.perform { context in
                let vocab = TokenizerVocabExtractor.extractForXGrammar(from: context.tokenizer)
                let tokenizer = try XGTokenizer(
                    vocab: vocab.vocab,
                    vocabType: vocab.vocabType,
                    eosTokenId: Int32(fixture.eosTokenId)
                )
                let constraint = try XGConstraint(
                    tokenizer: tokenizer,
                    jsonSchema: #"{"type":"object"}"#
                )

                let initial = try constraint.computeMask()
                #expect(!initial.isTerminated, "freshly constructed matcher must not be terminated")
                #expect(
                    initial.mask.contains(where: { $0 != 0 }),
                    "initial mask must have at least one accepted token for an open object schema"
                )

                guard let validToken = Self.firstSetBit(in: initial.mask) else {
                    Issue.record("no valid token in initial mask for {\"type\":\"object\"}")
                    return
                }

                let commit = try constraint.commitToken(validToken)
                #expect(
                    !commit.isTerminated,
                    "matcher must remain active after a single open-object commit")
                #expect(
                    commit.tokens.isEmpty,
                    "fast-forward is a later cycle; commit must return no FF tokens")

                let next = try constraint.computeMask()
                #expect(!next.isTerminated, "matcher must remain active after recompute")
                #expect(
                    next.mask.contains(where: { $0 != 0 }),
                    "post-commit mask must still have at least one accepted token"
                )

                #expect(
                    constraint.flushLogs() == nil, "flushLogs is a placeholder and must return nil")
            }
        }

        /// Find the first token ID whose corresponding bit is set in an
        /// xgrammar bitmask. Words are LSB-first: bit `i` of word `w` is
        /// token `w * 32 + i`. Returns `nil` if every word is zero.
        private static func firstSetBit(in mask: [Int32]) -> Int32? {
            for (wordIndex, word) in mask.enumerated() where word != 0 {
                let uword = UInt32(bitPattern: word)
                for bit in 0 ..< 32 where (uword >> bit) & 1 == 1 {
                    return Int32(wordIndex * 32 + bit)
                }
            }
            return nil
        }

        // MARK: - Concurrent matcher access

        /// Concurrent access on a single matcher must be serialized.
        ///
        /// `xgrammar::GrammarMatcher` is not thread-safe: `FillNextTokenBitmask`
        /// and `AcceptToken` mutate PIMPL state without synchronization.
        /// Production callers route each session through its own constraint,
        /// so the race does not show up in normal use — but the bridge still
        /// has to fail safely if two callers ever reach a single constraint
        /// concurrently (e.g. through a bug in session routing, or under a
        /// future multi-threaded sampling loop).
        ///
        /// Test shape: spin up two `Task.detached` workers that each run a
        /// compute-then-commit loop for many iterations against the same
        /// `XGConstraint`. `Task.detached` escapes the surrounding actor
        /// isolation so the two workers run on the global executor in
        /// parallel. Assertions:
        ///   - both workers complete without throwing from crashes
        ///   - the constraint responds to a final `computeMask()` call
        ///     without throwing, demonstrating its internal state was not
        ///     corrupted by the concurrent storm
        ///
        /// The stress loop uses `{"type":"array"}` so the grammar accepts
        /// arbitrarily long token streams without terminating, giving both
        /// workers continuous forward progress. A successful commit in
        /// either worker may be rejected on the other side if the grammar
        /// state moved underneath — that is acceptable; the contract is
        /// "no crash", not "every commit succeeds".
        ///
        /// Linearizability is not asserted numerically (xgrammar exposes no
        /// step counter); TSan runs on CI / simulator catch the race
        /// directly if the lock is removed. This test's role on a real
        /// device is the smoke signal: survive the concurrent storm without
        /// UB-induced crashes.
        @Test("XGConstraint: concurrent tasks do not crash or corrupt the matcher")
        func testConcurrentAccessToSingleMatcherIsSerialized() async throws {
            guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
            let fixture = try Self.loadTokenizerFixture(named: "tokenizer_gemma3.json")
            let container = try await loadTestModelContainer(id: TestFixtures.gemmaModelID)

            let constraint: XGConstraint = try await container.perform { context in
                let vocab = TokenizerVocabExtractor.extractForXGrammar(from: context.tokenizer)
                let tokenizer = try XGTokenizer(
                    vocab: vocab.vocab,
                    vocabType: vocab.vocabType,
                    eosTokenId: Int32(fixture.eosTokenId)
                )
                return try XGConstraint(
                    tokenizer: tokenizer,
                    jsonSchema: #"{"type":"array"}"#
                )
            }

            let iterationsPerTask = 200
            async let workerA = Task.detached { [constraint] in
                try Self.stressWorker(on: constraint, iterations: iterationsPerTask)
            }.value
            async let workerB = Task.detached { [constraint] in
                try Self.stressWorker(on: constraint, iterations: iterationsPerTask)
            }.value

            let (stepsA, stepsB) = try await (workerA, workerB)
            #expect(stepsA >= 0)
            #expect(stepsB >= 0)

            // Post-storm liveness: if the matcher were corrupted this call
            // would either crash or throw. A clean return proves the bridge
            // kept state consistent across the concurrent access window.
            _ = try constraint.computeMask()
        }

        /// Run a compute-then-commit loop against `constraint`, stopping
        /// early if the matcher terminates, the mask becomes empty, or any
        /// call throws. Returns the number of successful commits. Commits
        /// that the grammar rejects (because a peer task advanced state)
        /// are treated as a graceful stop condition for this worker — not
        /// a test failure.
        private static func stressWorker(on constraint: XGConstraint, iterations: Int) throws -> Int
        {
            var steps = 0
            for _ in 0 ..< iterations {
                let mask: XGMaskResult
                do {
                    mask = try constraint.computeMask()
                } catch {
                    break
                }
                if mask.isTerminated { break }
                guard let token = firstSetBit(in: mask.mask) else { break }
                do {
                    let commit = try constraint.commitToken(token)
                    steps += 1
                    if commit.isTerminated { break }
                } catch {
                    break
                }
            }
            return steps
        }

        // MARK: - Exception unwinding

        /// xgrammar exceptions unwind cleanly across the Swift -> C -> C++
        /// frame chain.
        ///
        /// Deliberately submits a JSON document that parses as JSON but is
        /// not a valid JSON Schema (`{"type": 42}` — `type` must be a
        /// string or array of strings). `xgrammar::GrammarCompiler::
        /// CompileJSONSchema` throws `InvalidJSONSchemaError` for this
        /// input. The shim's `WithExceptionBoundary` catches it inside the
        /// C++ translation unit and returns `XG_ERR_INVALID_JSON_SCHEMA`;
        /// Swift maps the status to `XGError.invalidJSONSchema(_)`.
        ///
        /// The test runs the construction inside a `Task.detached` closure
        /// to force the throwing call to land on a non-main executor
        /// thread, exercising the unwinding path off the main thread. If
        /// xgrammar's throw were to escape the shim and reach the Swift
        /// runtime, the process would fault here. A clean `throw`/`catch`
        /// round-trip proves the outermost shim `catch(...)` handler is
        /// reachable through the full frame chain and that no exception
        /// unwinds through Swift.
        ///
        /// If the cross-boundary unwinding story is broken, every throwing
        /// entry point in the shim (schema compile, EBNF compile,
        /// accept-token edge cases) is at risk.
        @Test(
            "xgrammar exceptions surface as XGError.invalidJSONSchema across the C++/Swift boundary"
        )
        func testShimCatchesXGrammarExceptionAcrossSwiftBoundary() async throws {
            guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
            let fixture = try Self.loadTokenizerFixture(named: "tokenizer_gemma3.json")
            let container = try await loadTestModelContainer(id: TestFixtures.gemmaModelID)

            let tokenizer: XGTokenizer = try await container.perform { context in
                let vocab = TokenizerVocabExtractor.extractForXGrammar(from: context.tokenizer)
                return try XGTokenizer(
                    vocab: vocab.vocab,
                    vocabType: vocab.vocabType,
                    eosTokenId: Int32(fixture.eosTokenId)
                )
            }

            let result = await Task.detached { [tokenizer] () -> Result<XGConstraint, Error> in
                do {
                    let constraint = try XGConstraint(
                        tokenizer: tokenizer,
                        jsonSchema: #"{"type": 42}"#
                    )
                    return .success(constraint)
                } catch {
                    return .failure(error)
                }
            }.value

            switch result {
            case .success:
                Issue.record("constructing XGConstraint from an invalid JSON Schema must throw")
            case .failure(let error):
                guard case XGError.invalidJSONSchema(let message) = error else {
                    Issue.record(
                        "expected XGError.invalidJSONSchema, got \(type(of: error)): \(error)")
                    return
                }
                #expect(
                    !message.isEmpty,
                    "xg_last_error_message() should carry xgrammar's what() text across the Swift boundary"
                )
            }
        }

        // MARK: - Fixture loading

        private struct TokenizerFixture {
            let vocabSize: Int
            let eosTokenId: Int
            let eosTokenString: String
        }

        private static func loadTokenizerFixture(named filename: String) throws -> TokenizerFixture
        {
            let base = (filename as NSString).deletingPathExtension
            let ext = (filename as NSString).pathExtension
            guard let url = fixturesBundle.url(forResource: base, withExtension: ext) else {
                throw FixtureError.malformed("\(filename): missing from test bundle resources")
            }
            let data = try Data(contentsOf: url)
            let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
            guard let json else {
                throw FixtureError.malformed("\(filename): top-level not an object")
            }
            guard let vocabSize = json["vocabSize"] as? Int else {
                throw FixtureError.malformed("\(filename): missing vocabSize")
            }
            guard let eosTokenId = json["eosTokenId"] as? Int else {
                throw FixtureError.malformed("\(filename): missing eosTokenId")
            }
            guard let eosTokenString = json["eosTokenString"] as? String else {
                throw FixtureError.malformed("\(filename): missing eosTokenString")
            }
            return TokenizerFixture(
                vocabSize: vocabSize,
                eosTokenId: eosTokenId,
                eosTokenString: eosTokenString
            )
        }

        private enum FixtureError: Error {
            case malformed(String)
        }
    }

#endif
