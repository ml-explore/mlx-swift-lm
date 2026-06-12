// Copyright © 2026 Apple Inc.
//
// Error-type parity (category-level).
//
// Asserts that every malformed-schema input in `malformed_schema_errors.json`
// surfaces as xgrammar's `.invalidJSONSchema` case — i.e. the
// "bad-schema-or-JSON" category. Exact message text is intentionally
// out of scope; xgrammar's `what()` strings are expected to vary across
// xgrammar upstream revisions. Category membership is what matters: every
// entry the fixture captured as rejected at compile time must also be
// rejected at compile time by xgrammar, with a Swift error case that's
// *distinguishable* from a generic shim failure
// (`.constraintCompilationFailed`).
//
// Why the same case for all 6: xgrammar discriminates only two
// flavors of bad input at compile time — `InvalidJSONError` (bytes
// don't parse as JSON) and `InvalidJSONSchemaError` (parses as JSON
// but rejected as a schema). Both map through the shim's
// discriminated-status path to `GrammarError.invalidJSONSchema`, so the
// "bad JSON" and "bad schema" categories collapse onto a single Swift
// case. The fixture's 6 inputs span both:
//   - `not_json`, `empty_string`      → InvalidJSONError path
//   - `unknown_type`, `enum_not_array`,
//     `dangling_ref`, `top_level_array` → InvalidJSONSchemaError path
// A failing assertion here means a category collapsed: either a
// bad-schema input surfaces as `.constraintCompilationFailed` (the
// shim's catch-all), or — worse — the schema compiled without
// throwing at all.
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

        @Test("every malformed-schema input surfaces as GrammarError.invalidJSONSchema")
        func testMalformedSchemaErrorsMatchGolden() async throws {
            guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
            let fixture = try loadMalformedSchemaFixture()
            #expect(
                fixture.modelId == TestFixtures.defaultModelID,
                "golden fixture modelId \(fixture.modelId); expected \(TestFixtures.defaultModelID)"
            )
            #expect(
                fixture.errors.count >= 1,
                "fixture must carry at least one malformed schema")

            let container = try await loadTestModelContainer(id: fixture.modelId)
            try await container.perform { context in
                let vocab = TokenizerVocabExtractor.extractForGrammar(from: context.tokenizer)
                let tokenizer = try GrammarTokenizer(
                    vocab: vocab.vocab,
                    vocabType: vocab.vocabType,
                    eosTokenId: Int32(context.tokenizer.eosTokenId ?? 0)
                )

                for entry in fixture.errors {
                    // Each malformed schema must throw. Anything else — a
                    // successful compile or a non-throwing error — is a
                    // category collapse.
                    do {
                        _ = try GrammarConstraint(
                            tokenizer: tokenizer,
                            jsonSchema: entry.schema
                        )
                        Issue.record(
                            "fixture entry #\(entry.index) (\(entry.label)): GrammarConstraint compiled without throwing; the recorded goldens rejected this as \(entry.errorCase). Category collapse — xgrammar accepts what the prior backend rejected."
                        )
                    } catch let error as GrammarError {
                        // Category-level parity: every recorded
                        // compile-time rejection must surface as
                        // xgrammar's `.invalidJSONSchema`. Any other
                        // case means the shim-level exception-to-status
                        // mapping dropped the input into a different
                        // bucket.
                        switch error {
                        case .invalidJSONSchema:
                            // OK — bad-JSON or bad-schema, both categories
                            // legitimately collapse onto this single case
                            // in the current discriminated-status design.
                            break
                        default:
                            Issue.record(
                                "fixture entry #\(entry.index) (\(entry.label)): expected GrammarError.invalidJSONSchema, got \(error). Category collapse."
                            )
                        }
                    } catch {
                        Issue.record(
                            "fixture entry #\(entry.index) (\(entry.label)): expected GrammarError, got \(type(of: error)) — \(error)"
                        )
                    }
                }
            }
        }
    }

    // MARK: - Fixture loader

    private struct MalformedSchemaFixture {
        let modelId: String
        let errors: [MalformedSchemaEntry]
    }

    private struct MalformedSchemaEntry {
        let index: Int
        let label: String
        let errorCase: String
        let messagePrefix: String
        let outcome: String
        let schema: String
    }

    private func loadMalformedSchemaFixture() throws -> MalformedSchemaFixture {
        guard
            let url = fixturesBundle.url(
                forResource: "malformed_schema_errors", withExtension: "json")
        else {
            throw NSError(
                domain: "MalformedSchemaErrorParityTests", code: 1,
                userInfo: [
                    NSLocalizedDescriptionKey: "malformed_schema_errors.json missing from bundle"
                ])
        }
        let data = try Data(contentsOf: url)
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
            let modelId = json["modelId"] as? String,
            let rawErrors = json["errors"] as? [[String: Any]]
        else {
            throw NSError(
                domain: "MalformedSchemaErrorParityTests", code: 2,
                userInfo: [NSLocalizedDescriptionKey: "malformed_schema_errors.json malformed"])
        }
        let entries: [MalformedSchemaEntry] = rawErrors.compactMap { raw in
            guard let index = raw["index"] as? Int,
                let label = raw["label"] as? String,
                let errorCase = raw["errorCase"] as? String,
                let messagePrefix = raw["messagePrefix"] as? String,
                let outcome = raw["outcome"] as? String,
                let schema = raw["schema"] as? String
            else { return nil }
            return MalformedSchemaEntry(
                index: index,
                label: label,
                errorCase: errorCase,
                messagePrefix: messagePrefix,
                outcome: outcome,
                schema: schema
            )
        }
        return MalformedSchemaFixture(modelId: modelId, errors: entries)
    }

#endif
