// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration

import Testing
import Foundation
import FoundationModels
import MLX
import MLXLMCommon
@testable import MLXFoundationModels
@testable import MLXGuidedGeneration

/// Executable proof of the two usage paths documented in
/// `Libraries/MLXGuidedGeneration/README.md`.
///
/// Both paths drive `TestFixtures.defaultModelID` (the id tests use when
/// they do not care which specific MLX model runs) and constrain output to
/// the same `City` shape:
///
/// 1. **Built-in (FoundationModels).** `LanguageModelSession.respond` with a
///    `@Generable` type, the path `MLXFoundationModels` adds on top of the
///    engine.
/// 2. **Standalone (MLXGuidedGeneration).** The lower-level
///    `extractForGrammar` -> `GrammarTokenizer` -> `GrammarConstraint` ->
///    `GuidedGenerationLoop.run` path, using only `MLXGuidedGeneration` +
///    `MLXLMCommon`. This is the snippet a non-FoundationModels consumer
///    (older OS, any MLX model) copies from the README.
///
/// Semantic correctness is not asserted: the grammar constraint guarantees
/// structural validity, which is what these tests check.
@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
@Generable
private struct City: Codable {
    @Guide(description: "The city name")
    let name: String
    @Guide(description: "The country the city is in")
    let country: String
}

/// The JSON Schema mirror of `City`, used by the standalone path. Matches
/// the `name` + `country` shape of the `@Generable` type above so both
/// README snippets constrain to the same structure.
private let citySchema = #"""
    {"type":"object","properties":{"name":{"type":"string"},"country":{"type":"string"}},"required":["name","country"],"additionalProperties":false}
    """#

@Suite(.serialized, .timeLimit(.minutes(5)))
struct GuidedGenerationUsageTests {

    // MARK: - Built-in FoundationModels path

    @Test("README built-in path: LanguageModelSession respond(generating:) yields a City")
    func builtInGenerablePath() async throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

        let model = makeTestModel(TestFixtures.defaultModelID)
        let session = LanguageModelSession(model: model, tools: [], instructions: nil)

        let response = try await session.respond(
            to: "Name a city to visit in Japan.",
            generating: City.self
        )

        // The @Generable decode succeeding is itself the structural
        // guarantee; just confirm the fields are populated.
        #expect(!response.content.name.isEmpty, "City.name should be populated")
        #expect(!response.content.country.isEmpty, "City.country should be populated")
    }

    // MARK: - Standalone MLXGuidedGeneration path

    @Test("README standalone path: GuidedGenerationLoop yields schema-valid JSON")
    func standaloneGuidedGenerationPath() async throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

        let container = try await loadTestModelContainer(id: TestFixtures.defaultModelID)
        let raw = try await standaloneRawJSON(container: container)

        // The grammar guarantees structural validity: parse it back and
        // confirm the required keys are present and correctly typed.
        let dict = try parseObject(raw, label: "(standalone)")
        #expect(dict["name"] is String, "'name' should be a string")
        #expect(dict["country"] is String, "'country' should be a string")
    }

    // MARK: - Cross-path parity

    @Test("README parity: built-in and standalone paths produce the same JSON shape")
    func bothPathsProduceSameShape() async throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

        // Built-in (FoundationModels) path, collected as raw JSON via the
        // Executor so it is comparable to the standalone output.
        let model = makeTestModel(TestFixtures.defaultModelID)
        let fmRaw = try await builtInRawJSON(model: model)

        // Standalone (MLXGuidedGeneration) path on the same model.
        let container = try await loadTestModelContainer(id: TestFixtures.defaultModelID)
        let standaloneRaw = try await standaloneRawJSON(container: container)

        let fmDict = try parseObject(fmRaw, label: "(built-in)")
        let standaloneDict = try parseObject(standaloneRaw, label: "(standalone)")

        // Parity is structural, not byte-for-byte: the two paths assemble
        // different prompts and apply different bias/reserve policies, so
        // their VALUES may differ. What must match is the schema-constrained
        // SHAPE: the same key set, the same field types, and decodability
        // into the same `City` type.
        #expect(
            Set(fmDict.keys) == Set(standaloneDict.keys),
            "Key sets should match: built-in=\(Set(fmDict.keys)) standalone=\(Set(standaloneDict.keys))"
        )
        #expect(
            Set(fmDict.keys) == ["name", "country"], "Keys should be exactly name + country")
        #expect(fmDict["name"] is String && standaloneDict["name"] is String)
        #expect(fmDict["country"] is String && standaloneDict["country"] is String)

        // Strongest structural check: both decode into the same Swift type.
        _ = try JSONDecoder().decode(City.self, from: Data(fmRaw.utf8))
        _ = try JSONDecoder().decode(City.self, from: Data(standaloneRaw.utf8))
    }

    // MARK: - Helpers

    /// Runs the README standalone path and returns the collected raw JSON.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private func standaloneRawJSON(container: ModelContainer) async throws -> String {
        try await container.perform { context in
            let tokenizer = context.tokenizer

            // 1. Extract the vocab in the shape xgrammar expects.
            let grammarVocab = TokenizerVocabExtractor.extractForGrammar(from: tokenizer)

            // 2. Build a grammar tokenizer.
            let grammarTokenizer = try GrammarTokenizer(
                vocab: grammarVocab.vocab,
                vocabType: grammarVocab.vocabType,
                eosTokenId: Int32(tokenizer.eosTokenId ?? 0)
            )

            // 3. Compile a JSON Schema into a constraint.
            let constraint = try GrammarConstraint(
                tokenizer: grammarTokenizer,
                jsonSchema: citySchema,
                fastForward: true,
                hostTokenizer: tokenizer
            )

            // 4. Run the guided loop, collecting the constrained output.
            let input = try await context.processor.prepare(
                input: UserInput(prompt: "Name a city to visit in Japan, as JSON.")
            )
            var output = ""
            try GuidedGenerationLoop.run(
                input: input,
                context: context,
                constraint: constraint,
                maxTokens: 256,
                vocabSize: grammarTokenizer.vocabSize
            ) { delta in
                output += delta
                return true
            }
            return output
        }
    }

    /// Runs the built-in FoundationModels path through the Executor with
    /// `City.generationSchema` and returns the collected raw JSON. Uses the
    /// Executor (rather than `LanguageModelSession.respond(generating:)`) so
    /// the output is raw JSON text comparable to the standalone path.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private func builtInRawJSON(model: MLXLanguageModel) async throws -> String {
        let executor = try makeMLXExecutor(for: model)
        let transcript = Transcript(entries: [
            .prompt(
                Transcript.Prompt(
                    segments: [
                        .text(Transcript.TextSegment(content: "Name a city to visit in Japan."))
                    ], responseFormat: nil))
        ])
        let request = makeExecutorRequest(
            transcript: transcript, schema: City.generationSchema)

        let stream = try await executeResponse(executor, request: request, model: model)
        var text = ""
        for try await event in stream {
            if case .appendText(let chunk, _, .response) = event {
                text += chunk
            }
        }
        return text
    }

    /// Parses raw text as a non-empty JSON object.
    private func parseObject(_ raw: String, label: String) throws -> [String: Any] {
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        #expect(!trimmed.isEmpty, "Output should be non-empty \(label)")
        let data = try #require(trimmed.data(using: .utf8), "UTF-8 encoding failed \(label)")
        let obj = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        return try #require(obj, "Output should decode as a JSON object \(label): \(trimmed)")
    }
}

#endif
