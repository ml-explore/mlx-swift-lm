// Copyright © 2025 Apple Inc.

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

import Testing
import Foundation
import MLX
import MLXLMCommon
import FoundationModels
@testable import MLXFoundationModels
@testable import MLXGuidedGeneration

/// End-to-end round-trip tests proving guided generation produces valid,
/// decodable JSON for a variety of schema types.
///
/// Each test constrains generation with a schema, collects all text deltas,
/// and verifies the output is structurally valid JSON that decodes to the
/// expected Swift type. Semantic correctness is not asserted -- the 0.5B
/// model may produce surprising values, but the grammar constraint must
/// guarantee structural validity.
@Suite(.serialized, .timeLimit(.minutes(10)))
struct GenerableRoundTripTests {

    // MARK: - Helpers

    /// Collects all text deltas from a guided generation request into a single string.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private func collectText(
        from executor: MLXLanguageModel.Executor,
        request: LanguageModelExecutorGenerationRequest,
        model: MLXLanguageModel
    ) async throws -> String {
        let stream = try await executeResponse(executor, request: request, model: model)
        var text = ""
        for try await event in stream {
            if case .appendText(let chunk, _, .response) = event {
                text += chunk
            }
        }
        return text
    }

    /// Builds a transcript with a single user prompt.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private func transcript(_ prompt: String) -> Transcript {
        Transcript(entries: [
            .prompt(
                Transcript.Prompt(
                    segments: [
                        .text(Transcript.TextSegment(content: prompt))
                    ], responseFormat: nil))
        ])
    }

    /// Asserts the string is valid JSON (fragments allowed), returning the trimmed form.
    @discardableResult
    private func assertValidJSON(_ raw: String, label: String = "") throws -> String {
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        #expect(!trimmed.isEmpty, "Output should be non-empty \(label)")

        let data = try #require(trimmed.data(using: .utf8), "UTF-8 encoding failed \(label)")
        let parsed = try? JSONSerialization.jsonObject(with: data, options: .fragmentsAllowed)
        #expect(parsed != nil, "Output should be valid JSON \(label): \(trimmed)")
        return trimmed
    }

    // MARK: - Primitive Round-Trip Tests

    @Test("Int schema produces decodable integer")
    func intRoundTrip() async throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let model = makeTestModel(TestFixtures.defaultModelID)
        let executor = try makeMLXExecutor(for: model)

        let request = makeExecutorRequest(
            transcript: transcript("What is 2+2? Reply with just the number."),
            schema: Int.generationSchema
        )

        let raw = try await collectText(from: executor, request: request, model: model)
        let trimmed = try assertValidJSON(raw, label: "(Int)")

        let decoded = try JSONDecoder().decode(Int.self, from: Data(trimmed.utf8))
        // No semantic check -- the grammar guarantees it parses as Int.
        _ = decoded
    }

    @Test("String schema produces decodable string")
    func stringRoundTrip() async throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let model = makeTestModel(TestFixtures.defaultModelID)
        let executor = try makeMLXExecutor(for: model)

        let request = makeExecutorRequest(
            transcript: transcript(
                "What is the capital of France? Reply with just the city name."),
            schema: String.generationSchema
        )

        let raw = try await collectText(from: executor, request: request, model: model)
        let trimmed = try assertValidJSON(raw, label: "(String)")

        let decoded = try JSONDecoder().decode(String.self, from: Data(trimmed.utf8))
        #expect(!decoded.isEmpty, "Decoded string should not be empty")
    }

    @Test("Bool schema produces decodable boolean")
    func boolRoundTrip() async throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let model = makeTestModel(TestFixtures.defaultModelID)
        let executor = try makeMLXExecutor(for: model)

        let request = makeExecutorRequest(
            transcript: transcript("Is 2+2 equal to 4? Reply true or false."),
            schema: Bool.generationSchema
        )

        let raw = try await collectText(from: executor, request: request, model: model)
        let trimmed = try assertValidJSON(raw, label: "(Bool)")

        let decoded = try JSONDecoder().decode(Bool.self, from: Data(trimmed.utf8))
        _ = decoded
    }

    // MARK: - Array Round-Trip

    @Test("Array<Int> schema produces decodable integer array")
    func intArrayRoundTrip() async throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let model = makeTestModel(TestFixtures.defaultModelID)
        let executor = try makeMLXExecutor(for: model)

        let request = makeExecutorRequest(
            transcript: transcript(
                "List the first three prime numbers as a JSON array of integers."),
            schema: [Int].generationSchema
        )

        let raw = try await collectText(from: executor, request: request, model: model)
        let trimmed = try assertValidJSON(raw, label: "([Int])")

        let decoded = try JSONDecoder().decode([Int].self, from: Data(trimmed.utf8))
        #expect(!decoded.isEmpty, "Decoded array should not be empty")
    }

    // MARK: - JSON Structural Validity

    @Test("Schema-constrained output passes JSONSerialization with fragmentsAllowed")
    func jsonSerializationRoundTrip() async throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let model = makeTestModel(TestFixtures.defaultModelID)
        let executor = try makeMLXExecutor(for: model)

        // Use Int schema as the baseline structural test
        let request = makeExecutorRequest(
            transcript: transcript("Pick any integer between 1 and 100."),
            schema: Int.generationSchema
        )

        let raw = try await collectText(from: executor, request: request, model: model)
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        let data = try #require(trimmed.data(using: .utf8))

        let obj = try JSONSerialization.jsonObject(with: data, options: .fragmentsAllowed)
        #expect(
            obj is NSNumber,
            "Int schema output should deserialize as NSNumber, got: \(type(of: obj))")
    }

    // MARK: - Sequential Multi-Schema Requests

    @Test("Sequential requests with different schemas both produce valid output")
    func sequentialSchemas() async throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let model = makeTestModel(TestFixtures.defaultModelID)
        let executor = try makeMLXExecutor(for: model)

        // First: Int schema
        let intRequest = makeExecutorRequest(
            transcript: transcript("What is 3+3? Reply with the number."),
            schema: Int.generationSchema
        )
        let intRaw = try await collectText(from: executor, request: intRequest, model: model)
        let intTrimmed = try assertValidJSON(intRaw, label: "(sequential Int)")
        let intValue = try JSONDecoder().decode(Int.self, from: Data(intTrimmed.utf8))
        _ = intValue

        // Second: String schema on the same executor
        let stringRequest = makeExecutorRequest(
            transcript: transcript("Name a color."),
            schema: String.generationSchema
        )
        let stringRaw = try await collectText(
            from: executor, request: stringRequest, model: model)
        let stringTrimmed = try assertValidJSON(stringRaw, label: "(sequential String)")
        let stringValue = try JSONDecoder().decode(String.self, from: Data(stringTrimmed.utf8))
        #expect(!stringValue.isEmpty)
    }

    // MARK: - Schema Converter Fidelity

    @Test("SchemaConverter produces valid JSON Schema from Int.generationSchema")
    func schemaConverterInt() throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let json = try SchemaConverter.encodeToJSON(Int.generationSchema)
        let data = try #require(json.data(using: .utf8))
        let obj = try JSONSerialization.jsonObject(with: data, options: [])

        // The JSON Schema for Int should include "type": "integer"
        if let dict = obj as? [String: Any], let type = dict["type"] as? String {
            #expect(type == "integer", "Int schema should have type 'integer', got '\(type)'")
        }
    }

    @Test("SchemaConverter produces valid JSON Schema from Bool.generationSchema")
    func schemaConverterBool() throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let json = try SchemaConverter.encodeToJSON(Bool.generationSchema)
        let data = try #require(json.data(using: .utf8))
        let obj = try JSONSerialization.jsonObject(with: data, options: [])

        if let dict = obj as? [String: Any], let type = dict["type"] as? String {
            #expect(type == "boolean", "Bool schema should have type 'boolean', got '\(type)'")
        }
    }

    @Test("SchemaConverter produces valid JSON Schema from String.generationSchema")
    func schemaConverterString() throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let json = try SchemaConverter.encodeToJSON(String.generationSchema)
        let data = try #require(json.data(using: .utf8))
        let obj = try JSONSerialization.jsonObject(with: data, options: [])

        if let dict = obj as? [String: Any], let type = dict["type"] as? String {
            #expect(type == "string", "String schema should have type 'string', got '\(type)'")
        }
    }

    // MARK: - Repeated Generation Stability

    @Test("Repeated Int generation is consistently valid JSON")
    func repeatedIntGeneration() async throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let model = makeTestModel(TestFixtures.defaultModelID)
        let executor = try makeMLXExecutor(for: model)

        for i in 0 ..< 3 {
            let request = makeExecutorRequest(
                transcript: transcript("Pick a number between \(i * 10) and \((i + 1) * 10)."),
                schema: Int.generationSchema
            )
            let raw = try await collectText(from: executor, request: request, model: model)
            let trimmed = try assertValidJSON(raw, label: "(iteration \(i))")
            let decoded = try JSONDecoder().decode(Int.self, from: Data(trimmed.utf8))
            _ = decoded
        }
    }

    // MARK: - Structured Object Round-Trip Tests
    //
    // These tests bypass the Executor and drive GuidedGenerationLoop directly
    // with hand-written JSON Schema strings.

    /// Runs guided generation with a raw JSON schema and returns the collected text.
    ///
    /// Mirrors the production `MLXLanguageModel.Executor.respond` call path:
    /// computes the same closing bias, whitespace bias, and zoned completion
    /// reserve that production uses, and passes them to `GuidedGenerationLoop.run`.
    /// Without these, complex schemas (deep nesting + count constraints + `maxLength`
    /// strings) can push the model into no-op whitespace-accepting loops that the
    /// grammar permits but that never terminate — the defaults on `run` (reserve=64,
    /// biases=nil) do not reflect any real call site in the shipped code.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private func generateWithSchema(
        _ jsonSchema: String,
        prompt: String,
        modelID: String = TestFixtures.defaultModelID,
        container: ModelContainer,
        maxTokens: Int = 512,
        kvBits: Int? = nil
    ) async throws -> String {
        try await container.perform { context in
            let xgTokenizer = try await MLXLanguageModel.makeXGTokenizer(
                modelID: modelID,
                tokenizer: context.tokenizer
            )

            let constraint = try GrammarConstraint(
                tokenizer: xgTokenizer,
                jsonSchema: jsonSchema,
                fastForward: true,
                hostTokenizer: context.tokenizer
            )

            let userInput = UserInput(
                chat: [.user(prompt)],
                processing: .init()
            )
            let input = try await context.processor.prepare(input: userInput)

            // Mirror the production bias / reserve computation so the test
            // exercises the same sampling path real callers hit.
            let closingBias = ClosingTokenBias.compute(
                tokenizer: context.tokenizer,
                eosTokenId: context.tokenizer.eosTokenId
            )
            let structuralReserve = CompletionReserve.estimate(
                schemaJSON: jsonSchema,
                tokenizer: context.tokenizer
            )
            let completionReserve = Swift.max(structuralReserve * 3, maxTokens / 4)
            let hardReserve = structuralReserve * 8
            let (whitespaceBias, whitespaceTokenIDs) = WhitespaceTokenBias.compute(
                tokenizer: context.tokenizer
            )

            var collected = ""
            try GuidedGenerationLoop.run(
                input: input,
                context: context,
                constraint: constraint,
                maxTokens: maxTokens,
                vocabSize: Int(xgTokenizer.vocabSize),
                kvBits: kvBits,
                completionReserve: completionReserve,
                hardReserve: hardReserve,
                closingBias: closingBias,
                whitespaceBias: whitespaceBias,
                whitespaceTokenIDs: whitespaceTokenIDs
            ) { text in
                collected += text
                return true
            }
            return collected
        }
    }

    @Test("Flat object schema produces decodable JSON with required keys")
    func flatObjectRoundTrip() async throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let container = try await loadTestModelContainer(id: TestFixtures.defaultModelID)

        let schema = """
            {
                "type": "object",
                "properties": {
                    "name": { "type": "string" },
                    "age": { "type": "integer" }
                },
                "required": ["name", "age"],
                "additionalProperties": false
            }
            """

        let raw = try await generateWithSchema(
            schema,
            prompt: "Describe a person named Alice who is 30 years old. Respond as JSON.",
            container: container
        )

        let trimmed = try assertValidJSON(raw, label: "(flat object)")
        let data = Data(trimmed.utf8)
        let obj = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        let dict = try #require(obj, "Should decode as dictionary")
        #expect(dict["name"] != nil, "Should have 'name' key")
        #expect(dict["age"] != nil, "Should have 'age' key")
        #expect(dict["name"] is String, "'name' should be a string")
    }

    @Test("Nested object schema produces decodable JSON with inner object")
    func nestedObjectRoundTrip() async throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let container = try await loadTestModelContainer(id: TestFixtures.defaultModelID)

        let schema = """
            {
                "type": "object",
                "properties": {
                    "city": { "type": "string" },
                    "population": { "type": "integer" },
                    "coordinates": {
                        "type": "object",
                        "properties": {
                            "lat": { "type": "number" },
                            "lon": { "type": "number" }
                        },
                        "required": ["lat", "lon"],
                        "additionalProperties": false
                    }
                },
                "required": ["city", "population", "coordinates"],
                "additionalProperties": false
            }
            """

        let raw = try await generateWithSchema(
            schema,
            prompt: "Describe Paris with its coordinates. Respond as JSON.",
            container: container
        )

        let trimmed = try assertValidJSON(raw, label: "(nested object)")
        let data = Data(trimmed.utf8)
        let obj = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        let dict = try #require(obj, "Should decode as dictionary")
        #expect(dict["city"] is String, "'city' should be a string")
        #expect(dict["population"] != nil, "Should have 'population' key")

        let coords = try #require(
            dict["coordinates"] as? [String: Any], "Should have nested 'coordinates' object")
        #expect(coords["lat"] is NSNumber, "'lat' should be a number")
        #expect(coords["lon"] is NSNumber, "'lon' should be a number")
    }

    @Test("Array of objects schema produces decodable JSON array")
    func arrayOfObjectsRoundTrip() async throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let container = try await loadTestModelContainer(id: TestFixtures.defaultModelID)

        let schema = """
            {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "item": { "type": "string", "maxLength": 20 },
                        "category": {
                            "type": "string",
                            "enum": ["fruit", "vegetable", "dairy"]
                        }
                    },
                    "required": ["item", "category"],
                    "additionalProperties": false
                },
                "minItems": 1,
                "maxItems": 2
            }
            """

        let raw = try await generateWithSchema(
            schema,
            prompt: "List two grocery items with categories. Respond as a JSON array.",
            container: container
        )

        let trimmed = try assertValidJSON(raw, label: "(array of objects)")
        let data = Data(trimmed.utf8)
        let arr = try JSONSerialization.jsonObject(with: data) as? [[String: Any]]
        let items = try #require(arr, "Should decode as array of dictionaries")
        #expect(!items.isEmpty, "Array should have at least one element")

        for (i, element) in items.enumerated() {
            #expect(element["item"] is String, "Element \(i) 'item' should be a string")
            let category = try #require(
                element["category"] as? String, "Element \(i) should have 'category'")
            #expect(
                ["fruit", "vegetable", "dairy"].contains(category),
                "Element \(i) category '\(category)' should be a valid enum value"
            )
        }
    }

    @Test("Deeply nested object with count-constrained arrays produces valid JSON (Qwen)")
    func deeplyNestedCountConstrainedRoundTrip() async throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        try await runDeeplyNestedCountConstrained(
            modelID: TestFixtures.defaultModelID, label: "Qwen")
    }

    @Test("Deeply nested object with count-constrained arrays produces valid JSON (Gemma)")
    func deeplyNestedCountConstrainedRoundTripGemma() async throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        try await runDeeplyNestedCountConstrained(
            modelID: TestFixtures.gemmaModelID, label: "Gemma")
    }

    private func runDeeplyNestedCountConstrained(modelID: String, label: String) async throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let container = try await loadTestModelContainer(id: modelID)

        let schema = """
            {
                "type": "object",
                "properties": {
                    "title": { "type": "string", "maxLength": 50 },
                    "summary": { "type": "string", "maxLength": 100 },
                    "sections": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "heading": { "type": "string", "maxLength": 30 },
                                "items": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "category": { "type": "string", "enum": ["info", "action", "note"] },
                                            "label": { "type": "string", "maxLength": 30 },
                                            "detail": { "type": "string", "maxLength": 60 }
                                        },
                                        "required": ["category", "label", "detail"],
                                        "additionalProperties": false
                                    },
                                    "minItems": 2,
                                    "maxItems": 2
                                }
                            },
                            "required": ["heading", "items"],
                            "additionalProperties": false
                        },
                        "minItems": 2,
                        "maxItems": 2
                    }
                },
                "required": ["title", "summary", "sections"],
                "additionalProperties": false
            }
            """

        let raw = try await generateWithSchema(
            schema,
            prompt: "Create a two-section itinerary with two items each. Respond as JSON.",
            modelID: modelID,
            container: container,
            maxTokens: 1024
        )

        let trimmed = try assertValidJSON(raw, label: "(deeply nested, \(label))")
        let data = Data(trimmed.utf8)
        let obj = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        let root = try #require(obj, "(\(label)) Should decode as dictionary")

        #expect(root["title"] is String, "(\(label)) Should have 'title' string")
        #expect(root["summary"] is String, "(\(label)) Should have 'summary' string")

        let sections = try #require(
            root["sections"] as? [[String: Any]], "(\(label)) Should have 'sections' array")
        #expect(
            sections.count == 2,
            "(\(label)) sections should have exactly 2 elements, got \(sections.count)")

        for (si, section) in sections.enumerated() {
            #expect(
                section["heading"] is String,
                "(\(label)) Section \(si) should have 'heading' string")

            let items = try #require(
                section["items"] as? [[String: Any]],
                "(\(label)) Section \(si) should have 'items' array"
            )
            #expect(
                items.count == 2,
                "(\(label)) Section \(si) items should have exactly 2 elements, got \(items.count)"
            )

            for (ii, item) in items.enumerated() {
                let category = try #require(
                    item["category"] as? String,
                    "(\(label)) Section \(si) item \(ii) should have 'category' string"
                )
                #expect(
                    ["info", "action", "note"].contains(category),
                    "(\(label)) Section \(si) item \(ii) category '\(category)' should be a valid enum value"
                )
                #expect(
                    item["label"] is String,
                    "(\(label)) Section \(si) item \(ii) should have 'label' string")
                #expect(
                    item["detail"] is String,
                    "(\(label)) Section \(si) item \(ii) should have 'detail' string")
            }
        }
    }

    @Test("Quantized KV cache (kvBits=8) still produces valid structured JSON")
    func quantizedKVCacheProducesValidJSON() async throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let container = try await loadTestModelContainer(id: TestFixtures.defaultModelID)

        let schema = """
            {
                "type": "object",
                "properties": {
                    "name": { "type": "string" },
                    "age": { "type": "integer" }
                },
                "required": ["name", "age"],
                "additionalProperties": false
            }
            """

        let raw = try await generateWithSchema(
            schema,
            prompt: "Describe a person named Bob who is 42 years old. Respond as JSON.",
            container: container,
            kvBits: 8
        )

        let trimmed = try assertValidJSON(raw, label: "(quantized KV cache)")
        let data = Data(trimmed.utf8)
        let obj = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        let dict = try #require(obj, "Should decode as dictionary")
        #expect(dict["name"] != nil, "Should have 'name' key")
        #expect(dict["age"] != nil, "Should have 'age' key")
        #expect(dict["name"] is String, "'name' should be a string")
    }

    @Test("String enum schema constrains output to allowed values")
    func stringEnumRoundTrip() async throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let container = try await loadTestModelContainer(id: TestFixtures.defaultModelID)

        let schema = """
            {
                "type": "object",
                "properties": {
                    "color": {
                        "type": "string",
                        "enum": ["red", "green", "blue"]
                    }
                },
                "required": ["color"],
                "additionalProperties": false
            }
            """

        let raw = try await generateWithSchema(
            schema,
            prompt: "Pick a primary color. Respond as JSON.",
            container: container
        )

        let trimmed = try assertValidJSON(raw, label: "(string enum)")
        let data = Data(trimmed.utf8)
        let obj = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        let dict = try #require(obj, "Should decode as dictionary")
        let color = try #require(dict["color"] as? String, "'color' should be a string")
        #expect(
            ["red", "green", "blue"].contains(color),
            "Color '\(color)' should be one of the enum values"
        )
    }
}

#endif
