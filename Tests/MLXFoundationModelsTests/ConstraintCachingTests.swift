// Copyright © 2025 Apple Inc.

#if GuidedGenerationSupport

    import Testing
    import Foundation
    import CXGrammar
    @testable import MLXFoundationModels

    /// Tests for grammar compilation caching via constraint cloning.
    ///
    /// The ModelCache stores compiled "template" constraints and clones them
    /// for each request, avoiding repeated grammar compilation (~5-20ms savings).
    @Suite(
        .disabled(
            """
            XGConstraint.clone() requires xgrammar's GrammarMatcher::Fork() (xgrammar >= \
            v0.1.34); the vendored version (v0.1.30) does not provide it, so every clone() \
            in this suite throws. Production handles the absence gracefully — makeConstraint() \
            catches forkFailed and recompiles a fresh constraint — so constraint caching is a \
            perf-only optimization, not a correctness gap. Re-enable when the vendored \
            xgrammar is bumped to a version with Fork().
            """))
    struct ConstraintCachingTests {

        // MARK: - XGConstraint.clone() Tests

        private func makeByteFallbackTokenizer() throws -> XGTokenizer {
            let vocabSize = 256
            let vocab: [String] = (0 ..< vocabSize).map { byte in
                String(format: "<0x%02X>", byte)
            }
            return try XGTokenizer(
                vocab: vocab,
                vocabType: XG_VOCAB_TYPE_BYTE_FALLBACK,
                eosTokenId: Int32(vocabSize - 1)
            )
        }

        @Test
        func clonedConstraintIsIndependent() throws {
            let tokenizer = try makeByteFallbackTokenizer()

            let schema = """
                { "type": "integer" }
                """

            let original = try XGConstraint(
                tokenizer: tokenizer,
                jsonSchema: schema
            )

            // Clone creates a fresh constraint at the same grammar state
            let cloned = try original.clone()

            // Both should compute masks without error
            let originalMask = try original.computeMask()
            let clonedMask = try cloned.computeMask()

            // Neither should be stopped initially
            #expect(!originalMask.isTerminated, "Original should not be stopped")
            #expect(!clonedMask.isTerminated, "Clone should not be stopped")
        }

        @Test
        func clonedConstraintDoesNotAffectOriginal() throws {
            let tokenizer = try makeByteFallbackTokenizer()

            let schema = """
                { "type": "integer" }
                """

            let original = try XGConstraint(
                tokenizer: tokenizer,
                jsonSchema: schema
            )

            let cloned = try original.clone()

            // Advance the clone by computing mask and committing a token
            let mask = try cloned.computeMask()
            #expect(!mask.isTerminated)

            // Commit '4' (ASCII 52) to the clone -- valid for integer
            let _ = try cloned.commitToken(52)

            // Original should still be in its initial state
            let originalMask = try original.computeMask()
            #expect(
                !originalMask.isTerminated, "Original should be unaffected by clone's state changes"
            )
        }

        @Test
        func multipleClonesSupportConcurrentGeneration() throws {
            let tokenizer = try makeByteFallbackTokenizer()

            let schema = """
                { "type": "integer" }
                """

            let template = try XGConstraint(
                tokenizer: tokenizer,
                jsonSchema: schema
            )

            // Create multiple clones -- simulates concurrent requests
            let clone1 = try template.clone()
            let clone2 = try template.clone()
            let clone3 = try template.clone()

            // Each clone should work independently
            for clone in [clone1, clone2, clone3] {
                let mask = try clone.computeMask()
                #expect(!mask.isTerminated)
            }
        }
    }

#endif
