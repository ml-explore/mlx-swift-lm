// Copyright © 2026 Apple Inc.
//
// Error-type parity (category-level).
//
// Asserts that every malformed-schema input surfaces as xgrammar's
// `.invalidJSONSchema` case — i.e. the "bad-schema-or-JSON" category. Exact
// message text is intentionally out of scope; xgrammar's `what()` strings are
// expected to vary across xgrammar upstream revisions. Category membership is
// what matters: every malformed input must be rejected at compile time by
// xgrammar, with a Swift error case that's *distinguishable* from a generic
// shim failure (`.constraintCompilationFailed`).
//
// Why the same case for all of them: xgrammar discriminates only two
// flavors of bad input at compile time — `InvalidJSONError` (bytes
// don't parse as JSON) and `InvalidJSONSchemaError` (parses as JSON
// but rejected as a schema). Both map through the shim's
// discriminated-status path to `GrammarError.invalidJSONSchema`, so the
// "bad JSON" and "bad schema" categories collapse onto a single Swift
// case. The inputs below span both:
//   - `not_json`, `empty_string`      → InvalidJSONError path
//   - `unknown_type`, `enum_not_array`,
//     `dangling_ref`, `top_level_array` → InvalidJSONSchemaError path
// A failing assertion here means a category collapsed: either a
// bad-schema input surfaces as `.constraintCompilationFailed` (the
// shim's catch-all), or — worse — the schema compiled without
// throwing at all.
//
// The malformed inputs are a static inline table rather than a recorded
// "golden": the only thing that matters is the schema string and that it must
// throw `.invalidJSONSchema`. The previous golden file recorded `errorCase` /
// `messagePrefix` / `outcome` fields that were never asserted, so they are
// dropped here.
//
// Gated on both traits because the tokenizer path routes through
// `loadTestModelContainer` the same as the other integration tests.

#if FoundationModelsIntegration

    import Testing
    import Foundation
    import MLXLMCommon
    @testable import MLXFoundationModels
    @testable import MLXGuidedGeneration

    @Suite(.serialized)
    struct MalformedSchemaErrorParityTests {

        /// Malformed schema inputs that must each be rejected at grammar-compile
        /// time as `GrammarError.invalidJSONSchema`.
        private static let malformedSchemas: [(label: String, schema: String)] = [
            ("not_json", "not a schema at all"),
            ("empty_string", ""),
            ("unknown_type", #"{"type":"flibbertigibbet"}"#),
            ("enum_not_array", #"{"type":"string","enum":"not-an-array"}"#),
            ("dangling_ref", ##"{"$ref":"#/$defs/does-not-exist"}"##),
            ("top_level_array", "[]"),
        ]

        @Test("every malformed-schema input surfaces as GrammarError.invalidJSONSchema")
        func testMalformedSchemaErrorsMatchGolden() async throws {
            guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

            let container = try await loadTestModelContainer(id: TestFixtures.defaultModelID)
            try await container.perform { context in
                let vocab = TokenizerVocabExtractor.extractForGrammar(from: context.tokenizer)
                let tokenizer = try GrammarTokenizer(
                    vocab: vocab.vocab,
                    vocabType: vocab.vocabType,
                    eosTokenId: Int32(context.tokenizer.eosTokenId ?? 0)
                )

                for (index, entry) in Self.malformedSchemas.enumerated() {
                    // Each malformed schema must throw. Anything else — a
                    // successful compile or a non-throwing error — is a
                    // category collapse.
                    do {
                        _ = try GrammarConstraint(
                            tokenizer: tokenizer,
                            jsonSchema: entry.schema
                        )
                        Issue.record(
                            "malformed schema #\(index) (\(entry.label)): GrammarConstraint compiled without throwing. Category collapse — xgrammar accepts what should be rejected."
                        )
                    } catch let error as GrammarError {
                        // Category-level parity: every malformed input must
                        // surface as xgrammar's `.invalidJSONSchema`. Any other
                        // case means the shim-level exception-to-status
                        // mapping dropped the input into a different bucket.
                        switch error {
                        case .invalidJSONSchema:
                            // OK — bad-JSON or bad-schema, both categories
                            // legitimately collapse onto this single case
                            // in the current discriminated-status design.
                            break
                        default:
                            Issue.record(
                                "malformed schema #\(index) (\(entry.label)): expected GrammarError.invalidJSONSchema, got \(error). Category collapse."
                            )
                        }
                    } catch {
                        Issue.record(
                            "malformed schema #\(index) (\(entry.label)): expected GrammarError, got \(type(of: error)) — \(error)"
                        )
                    }
                }
            }
        }
    }

#endif
