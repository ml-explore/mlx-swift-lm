// Copyright (c) 2025 Apple Inc.

#if FoundationModelsIntegration

import Testing
import Foundation
import MLXLMCommon
import MLX
import FoundationModels
@testable import MLXFoundationModels
@testable import MLXGuidedGeneration

/// Incremental guided generation tests with increasing schema complexity.
///
/// Each test builds on the prior's schema, providing diagnostic waypoints:
/// if level N passes but N+1 fails, we know where the budget or grammar
/// breaks down. All schemas use `$ref`/`$defs` to match real `@Generable`
/// output. All string fields have `maxLength` to keep generation bounded.
@Suite(.serialized, .timeLimit(.minutes(5)))
struct GuidedGenerationTests {

    static let modelID = TestFixtures.gemmaModelID

    // MARK: - Activity Enum Values

    private static let validActivityTypes: Set<String> = [
        "sightseeing", "foodAndDining", "shopping", "hotelAndLodging",
    ]

    // MARK: - Test 1: Single Activity

    @Test("Single Activity schema produces valid JSON with enum type and non-empty strings")
    func testSingleActivitySchema() async throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let schema = """
            {
                "$defs": {
                    "Activity": {
                        "type": "object",
                        "properties": {
                            "type": {
                                "type": "string",
                                "enum": ["sightseeing", "foodAndDining", "shopping", "hotelAndLodging"]
                            },
                            "title": { "type": "string", "maxLength": 40 },
                            "description": { "type": "string", "maxLength": 40 }
                        },
                        "required": ["type", "title", "description"],
                        "additionalProperties": false
                    }
                },
                "$ref": "#/$defs/Activity"
            }
            """

        let raw = try await generateConstrainedJSON(
            schema: schema,
            prompt: "Describe a sightseeing activity. Respond as JSON.",
            maxTokens: 512
        )

        let sanitized = sanitize(raw)
        print("[testSingleActivitySchema] Output: \(sanitized)")

        let obj = try #require(
            try JSONSerialization.jsonObject(with: Data(sanitized.utf8)) as? [String: Any],
            "Should produce valid JSON object, got: \(sanitized.prefix(200))"
        )

        let actType = try #require(
            obj["type"] as? String,
            "Should have 'type' string field"
        )
        #expect(
            Self.validActivityTypes.contains(actType),
            "Activity type '\(actType)' should be a valid enum value"
        )

        let title = try #require(obj["title"] as? String, "Should have 'title' string")
        #expect(!title.isEmpty, "Activity title should not be empty")

        let desc = try #require(
            obj["description"] as? String, "Should have 'description' string")
        #expect(!desc.isEmpty, "Activity description should not be empty")
    }

    // MARK: - Test 2: Three Activities

    @Test("Array of 3 Activities produces valid JSON with exactly 3 objects")
    func testThreeActivitiesSchema() async throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let schema = """
            {
                "$defs": {
                    "Activity": {
                        "type": "object",
                        "properties": {
                            "type": {
                                "type": "string",
                                "enum": ["sightseeing", "foodAndDining", "shopping", "hotelAndLodging"]
                            },
                            "title": { "type": "string", "maxLength": 40 },
                            "description": { "type": "string", "maxLength": 40 }
                        },
                        "required": ["type", "title", "description"],
                        "additionalProperties": false
                    }
                },
                "type": "array",
                "items": { "$ref": "#/$defs/Activity" },
                "minItems": 3,
                "maxItems": 3
            }
            """

        let raw = try await generateConstrainedJSON(
            schema: schema,
            prompt: "List 3 travel activities. Respond as JSON.",
            maxTokens: 1024
        )

        let sanitized = sanitize(raw)
        print("[testThreeActivitiesSchema] Output: \(sanitized)")

        let arr = try #require(
            try JSONSerialization.jsonObject(with: Data(sanitized.utf8)) as? [[String: Any]],
            "Should produce valid JSON array, got: \(sanitized.prefix(200))"
        )

        #expect(arr.count == 3, "Should have exactly 3 activities, got \(arr.count)")

        for (i, activity) in arr.enumerated() {
            let actType = try #require(
                activity["type"] as? String,
                "Activity \(i) should have 'type'"
            )
            #expect(
                Self.validActivityTypes.contains(actType),
                "Activity \(i) type '\(actType)' should be valid enum"
            )
            #expect(activity["title"] is String, "Activity \(i) should have 'title'")
            #expect(
                activity["description"] is String, "Activity \(i) should have 'description'")
        }
    }

    // MARK: - Test 3: Single DayPlan

    @Test("Single DayPlan with 3 Activities produces valid JSON with all required fields")
    func testSingleDayPlanSchema() async throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let schema = """
            {
                "$defs": {
                    "Activity": {
                        "type": "object",
                        "properties": {
                            "type": {
                                "type": "string",
                                "enum": ["sightseeing", "foodAndDining", "shopping", "hotelAndLodging"]
                            },
                            "title": { "type": "string", "maxLength": 40 },
                            "description": { "type": "string", "maxLength": 40 }
                        },
                        "required": ["type", "title", "description"],
                        "additionalProperties": false
                    },
                    "DayPlan": {
                        "type": "object",
                        "properties": {
                            "title": { "type": "string", "maxLength": 60 },
                            "subtitle": { "type": "string", "maxLength": 60 },
                            "destination": { "type": "string", "maxLength": 60 },
                            "activities": {
                                "type": "array",
                                "items": { "$ref": "#/$defs/Activity" },
                                "minItems": 3,
                                "maxItems": 3
                            }
                        },
                        "required": ["title", "subtitle", "destination", "activities"],
                        "additionalProperties": false
                    }
                },
                "$ref": "#/$defs/DayPlan"
            }
            """

        let raw = try await generateConstrainedJSON(
            schema: schema,
            prompt: "Plan a day in Tokyo with 3 activities. Respond as JSON.",
            maxTokens: 1536
        )

        let sanitized = sanitize(raw)
        print("[testSingleDayPlanSchema] Output: \(sanitized)")

        let obj = try #require(
            try JSONSerialization.jsonObject(with: Data(sanitized.utf8)) as? [String: Any],
            "Should produce valid JSON object, got: \(sanitized.prefix(200))"
        )

        #expect(obj["title"] is String, "DayPlan should have 'title'")
        #expect(obj["subtitle"] is String, "DayPlan should have 'subtitle'")
        #expect(obj["destination"] is String, "DayPlan should have 'destination'")

        let activities = try #require(
            obj["activities"] as? [[String: Any]],
            "DayPlan should have 'activities' array"
        )
        #expect(
            activities.count == 3,
            "DayPlan should have exactly 3 activities, got \(activities.count)")

        for (i, activity) in activities.enumerated() {
            let actType = try #require(
                activity["type"] as? String,
                "Activity \(i) should have 'type'"
            )
            #expect(
                Self.validActivityTypes.contains(actType),
                "Activity \(i) type '\(actType)' should be valid enum"
            )
            #expect(activity["title"] is String, "Activity \(i) should have 'title'")
            #expect(
                activity["description"] is String, "Activity \(i) should have 'description'")
        }
    }

    // MARK: - Test 4: Full Itinerary (3 days x 3 activities)

    @Test(
        "Full Itinerary schema (3 days x 3 activities) produces valid JSON matching @Generable structure"
    )
    func testItineraryProducesThreeDays() async throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let raw = try await generateConstrainedJSON(
            schema: TestFixtures.itinerarySchemaConstrained,
            prompt: TestFixtures.itineraryPrompt,
            maxTokens: 4096
        )

        let sanitized = sanitize(raw)
        print(
            "[testItineraryProducesThreeDays] Output (\(sanitized.count) chars): \(sanitized.prefix(500))"
        )

        let data = Data(sanitized.utf8)
        let obj = try #require(
            try JSONSerialization.jsonObject(with: data) as? [String: Any],
            "Should produce valid JSON dict, got: \(sanitized.prefix(300))"
        )

        #expect(obj["title"] is String, "Should have 'title' string")
        #expect(obj["destinationName"] is String, "Should have 'destinationName' string")
        #expect(obj["description"] is String, "Should have 'description' string")
        #expect(obj["rationale"] is String, "Should have 'rationale' string")

        let days = try #require(
            obj["days"] as? [[String: Any]],
            "Should have 'days' array"
        )
        #expect(days.count == 3, "Should have exactly 3 days, got \(days.count)")

        for (di, day) in days.enumerated() {
            #expect(day["title"] is String, "Day \(di) should have 'title'")
            #expect(day["subtitle"] is String, "Day \(di) should have 'subtitle'")
            #expect(day["destination"] is String, "Day \(di) should have 'destination'")

            let activities = try #require(
                day["activities"] as? [[String: Any]],
                "Day \(di) should have 'activities' array"
            )
            #expect(
                activities.count == 3,
                "Day \(di) should have exactly 3 activities, got \(activities.count)"
            )

            for (ai, activity) in activities.enumerated() {
                let actType = try #require(
                    activity["type"] as? String,
                    "Day \(di) Activity \(ai) should have 'type'"
                )
                #expect(
                    Self.validActivityTypes.contains(actType),
                    "Day \(di) Activity \(ai) type '\(actType)' should be valid enum"
                )
                #expect(
                    activity["title"] is String, "Day \(di) Activity \(ai) should have 'title'")
                #expect(
                    activity["description"] is String,
                    "Day \(di) Activity \(ai) should have 'description'")
            }
        }
    }

    // MARK: - Helpers

    /// Schema with unbounded strings that a small model will fill verbosely.
    private static let unboundedSchema = """
        {
            "type": "object",
            "properties": {
                "title": { "type": "string" },
                "summary": { "type": "string" },
                "conclusion": { "type": "string" }
            },
            "required": ["title", "summary", "conclusion"],
            "additionalProperties": false
        }
        """

    /// Runs guided generation with configurable hardReserve, rendering the
    /// prompt via the tokenizer's chat template directly.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private func generateConstrainedJSON(
        schema: String,
        prompt: String,
        maxTokens: Int,
        hardReserve: Int = 0,
        diagnosticLog: Bool = false
    ) async throws -> String {
        let modelID = Self.modelID
        let container = try await loadTestModelContainer(id: modelID)

        let raw: String = try await container.perform { context in
            let xgTokenizer = try await MLXLanguageModel.makeXGTokenizer(
                modelID: modelID,
                tokenizer: context.tokenizer
            )
            let constraint = try GrammarConstraint(
                tokenizer: xgTokenizer,
                jsonSchema: schema,
                fastForward: true,
                hostTokenizer: context.tokenizer
            )

            let messages: [[String: any Sendable]] = [
                ["role": "user", "content": prompt]
            ]
            let tokens = try context.tokenizer.applyChatTemplate(messages: messages)
            let input = LMInput(tokens: MLXArray(tokens))

            let closingBias = ClosingTokenBias.compute(
                tokenizer: context.tokenizer,
                eosTokenId: context.tokenizer.eosTokenId
            )
            let (whitespaceBias, whitespaceTokenIDs) = WhitespaceTokenBias.compute(
                tokenizer: context.tokenizer
            )
            let reserve = CompletionReserve.estimate(
                schemaJSON: schema,
                tokenizer: context.tokenizer
            )

            print(
                "[GuidedGenerationTests] CompletionReserve: \(reserve) tokens for maxTokens: \(maxTokens), hardReserve: \(hardReserve)"
            )

            var collected = ""
            var tokenCount = 0
            try GuidedGenerationLoop.run(
                input: input,
                context: context,
                constraint: constraint,
                maxTokens: maxTokens,
                vocabSize: Int(xgTokenizer.vocabSize),
                completionReserve: reserve,
                hardReserve: hardReserve,
                closingBias: closingBias,
                whitespaceBias: whitespaceBias,
                whitespaceTokenIDs: whitespaceTokenIDs,
                diagnosticLog: diagnosticLog
            ) { text in
                collected += text
                tokenCount += 1
                return true
            }
            print(
                "[GuidedGenerationTests] Generated \(tokenCount) token callbacks, \(collected.count) chars"
            )
            return collected
        }

        return raw
    }

    /// Strips control characters below 0x20 (except standard whitespace) and trims.
    private func sanitize(_ raw: String) -> String {
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        return String(trimmed.unicodeScalars.filter { $0.value >= 0x20 })
    }

    // MARK: - Hard Reserve Tests

    @Test(
        "Without hardReserve, tight token budget on unbounded strings produces incomplete structure"
    )
    func testTightBudgetWithoutHardReserveIsIncomplete() async throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let raw: String
        do {
            raw = try await generateConstrainedJSON(
                schema: Self.unboundedSchema,
                prompt:
                    "Write a very detailed and thorough essay about the history of Rome. Be extremely verbose and comprehensive.",
                maxTokens: 128,
                hardReserve: 0
            )
        } catch is GuidedGenerationError {
            // incompleteOutput is one valid way to fail -- test passes
            return
        }

        let sanitized = sanitize(raw)
        print("[testTightBudgetWithoutHardReserveIsIncomplete] Output: \(sanitized)")

        guard
            let obj = try? JSONSerialization.jsonObject(with: Data(sanitized.utf8))
                as? [String: Any]
        else {
            // Not valid JSON at all -- confirms incomplete output
            return
        }

        let hasAllKeys =
            obj["title"] is String
            && obj["summary"] is String
            && obj["conclusion"] is String

        #expect(
            !hasAllKeys,
            "Without hardReserve, tight budget should NOT produce all required keys")
    }

    @Test("With hardReserve, tight token budget still produces structurally complete JSON")
    func testHardReserveForceStructuralCompletion() async throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let raw = try await generateConstrainedJSON(
            schema: Self.unboundedSchema,
            prompt:
                "Write a very detailed and thorough essay about the history of Rome. Be extremely verbose and comprehensive.",
            maxTokens: 256,
            hardReserve: 80
        )

        let sanitized = sanitize(raw)
        print("[testHardReserveForceStructuralCompletion] Output: \(sanitized)")

        let obj = try #require(
            try JSONSerialization.jsonObject(with: Data(sanitized.utf8)) as? [String: Any],
            "hardReserve should produce valid JSON, got: \(sanitized.prefix(200))"
        )

        #expect(obj["title"] is String, "Should have 'title' key")
        #expect(obj["summary"] is String, "Should have 'summary' key")
        #expect(obj["conclusion"] is String, "Should have 'conclusion' key")
    }

    @Test("hardReserve does not degrade output when token budget is generous")
    func testHardReserveWithGenerousBudget() async throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let raw = try await generateConstrainedJSON(
            schema: Self.unboundedSchema,
            prompt: "Give a short travel tip.",
            maxTokens: 512,
            hardReserve: 20
        )

        let sanitized = sanitize(raw)
        let obj = try #require(
            try JSONSerialization.jsonObject(with: Data(sanitized.utf8)) as? [String: Any],
            "Should produce valid JSON"
        )

        let title = try #require(obj["title"] as? String)
        let summary = try #require(obj["summary"] as? String)
        let conclusion = try #require(obj["conclusion"] as? String)

        #expect(!title.isEmpty, "title should have content with generous budget")
        #expect(!summary.isEmpty, "summary should have content with generous budget")
        #expect(!conclusion.isEmpty, "conclusion should have content with generous budget")
    }

    @Test("Production hardReserve multiplier (8x estimate) forces structural completion")
    func testProductionHardReserveMultiplier() async throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let modelID = Self.modelID
        let container = try await loadTestModelContainer(id: modelID)
        let schema = Self.unboundedSchema

        let raw: String = try await container.perform { context in
            let xgTokenizer = try await MLXLanguageModel.makeXGTokenizer(
                modelID: modelID,
                tokenizer: context.tokenizer
            )
            let constraint = try GrammarConstraint(
                tokenizer: xgTokenizer,
                jsonSchema: schema,
                fastForward: true,
                hostTokenizer: context.tokenizer
            )
            let messages: [[String: any Sendable]] = [
                [
                    "role": "user",
                    "content":
                        "Write a very detailed and thorough essay about the history of Rome. Be extremely verbose.",
                ]
            ]
            let tokens = try context.tokenizer.applyChatTemplate(messages: messages)
            let input = LMInput(tokens: MLXArray(tokens))

            let closingBias = ClosingTokenBias.compute(
                tokenizer: context.tokenizer,
                eosTokenId: context.tokenizer.eosTokenId
            )
            let (whitespaceBias, whitespaceTokenIDs) = WhitespaceTokenBias.compute(
                tokenizer: context.tokenizer
            )

            // Mirror the production calculation from MLXLanguageModel
            let structuralReserve = CompletionReserve.estimate(
                schemaJSON: schema,
                tokenizer: context.tokenizer
            )
            let reserve = Swift.max(structuralReserve * 3, 256 / 4)
            let hardReserve = structuralReserve * 8

            print(
                "[testProductionMultiplier] structuralReserve=\(structuralReserve), softReserve=\(reserve), hardReserve=\(hardReserve)"
            )

            var collected = ""
            try GuidedGenerationLoop.run(
                input: input,
                context: context,
                constraint: constraint,
                maxTokens: 256,
                vocabSize: Int(xgTokenizer.vocabSize),
                completionReserve: reserve,
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

        let sanitized = sanitize(raw)
        print("[testProductionMultiplier] Output: \(sanitized.prefix(300))")

        let obj = try #require(
            try JSONSerialization.jsonObject(with: Data(sanitized.utf8)) as? [String: Any],
            "Production multiplier should produce valid JSON, got: \(sanitized.prefix(200))"
        )

        #expect(obj["title"] is String, "Should have 'title' key")
        #expect(obj["summary"] is String, "Should have 'summary' key")
        #expect(obj["conclusion"] is String, "Should have 'conclusion' key")
    }
}

#endif
