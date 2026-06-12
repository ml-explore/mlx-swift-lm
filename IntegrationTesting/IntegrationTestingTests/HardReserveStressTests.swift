// Copyright (c) 2025 Apple Inc.

#if FoundationModelsIntegration

    import Testing
    import Foundation
    import MLXLMCommon
    import MLX
    import FoundationModels
    @testable import MLXFoundationModels
    @testable import MLXGuidedGeneration

    /// Stress tests for the hardReserve multiplier across increasing schema complexity.
    ///
    /// Each tier uses unbounded string fields (no `maxLength`) to maximize adversarial
    /// pressure. The token budget is set to `hardReserve + 128`, forcing the model into
    /// the hard reserve zone after generating just one or two verbose string values.
    @Suite(.serialized, .timeLimit(.minutes(8)))
    struct HardReserveStressTests {

        static let modelID = TestFixtures.gemmaModelID
        static let multiplier = 8

        // MARK: - Tier Schemas

        private static let tier1Schema = """
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

        private static let tier2Schema = """
            {
                "type": "object",
                "properties": {
                    "topic": { "type": "string" },
                    "overview": { "type": "string" },
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": { "type": "string" },
                                "description": { "type": "string" }
                            },
                            "required": ["name", "description"],
                            "additionalProperties": false
                        },
                        "minItems": 3,
                        "maxItems": 3
                    }
                },
                "required": ["topic", "overview", "items"],
                "additionalProperties": false
            }
            """

        private static let tier3Schema = """
            {
                "type": "object",
                "properties": {
                    "title": { "type": "string" },
                    "groups": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": { "type": "string" },
                                "entries": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "label": { "type": "string" },
                                            "detail": { "type": "string" }
                                        },
                                        "required": ["label", "detail"],
                                        "additionalProperties": false
                                    },
                                    "minItems": 3,
                                    "maxItems": 3
                                }
                            },
                            "required": ["name", "entries"],
                            "additionalProperties": false
                        },
                        "minItems": 2,
                        "maxItems": 2
                    }
                },
                "required": ["title", "groups"],
                "additionalProperties": false
            }
            """

        private static let tier4Schema = """
            {
                "type": "object",
                "properties": {
                    "title": { "type": "string" },
                    "destination": { "type": "string" },
                    "description": { "type": "string" },
                    "rationale": { "type": "string" },
                    "days": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "title": { "type": "string" },
                                "subtitle": { "type": "string" },
                                "destination": { "type": "string" },
                                "activities": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "type": { "type": "string" },
                                            "title": { "type": "string" },
                                            "description": { "type": "string" }
                                        },
                                        "required": ["type", "title", "description"],
                                        "additionalProperties": false
                                    },
                                    "minItems": 3,
                                    "maxItems": 3
                                }
                            },
                            "required": ["title", "subtitle", "destination", "activities"],
                            "additionalProperties": false
                        },
                        "minItems": 3,
                        "maxItems": 3
                    }
                },
                "required": ["title", "destination", "description", "rationale", "days"],
                "additionalProperties": false
            }
            """

        // MARK: - Helpers

        /// Runs guided generation with a specified hardReserve, computing
        /// structuralReserve internally and logging diagnostic info.
        @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
        private func generateWithReserve(
            schema: String,
            maxTokens: Int,
            hardReserve: Int
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

                let structuralReserve = CompletionReserve.estimate(
                    schemaJSON: schema,
                    tokenizer: context.tokenizer
                )
                let softReserve = Swift.max(structuralReserve * 3, maxTokens / 4)

                print(
                    "[HardReserveStressTests] structuralReserve=\(structuralReserve), hardReserve=\(hardReserve), maxTokens=\(maxTokens)"
                )

                // Format the prompt via the tokenizer's chat template directly —
                // the same path the production code exercises (the model's
                // UserInputProcessor + the upstream tokenizer handle prompt
                // rendering).
                let messages: [[String: any Sendable]] = [
                    [
                        "role": "user",
                        "content":
                            "Write a very detailed and thorough essay about travel and exploration. Be extremely verbose and comprehensive.",
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

                var collected = ""
                var tokenCount = 0
                try GuidedGenerationLoop.run(
                    input: input,
                    context: context,
                    constraint: constraint,
                    maxTokens: maxTokens,
                    vocabSize: Int(xgTokenizer.vocabSize),
                    completionReserve: softReserve,
                    hardReserve: hardReserve,
                    closingBias: closingBias,
                    whitespaceBias: whitespaceBias,
                    whitespaceTokenIDs: whitespaceTokenIDs,
                    diagnosticLog: false
                ) { text in
                    collected += text
                    tokenCount += 1
                    return true
                }
                print(
                    "[HardReserveStressTests] Generated \(tokenCount) token callbacks, \(collected.count) chars"
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

        // MARK: - Behavior 1: Diagnostic Estimates

        @Test("CompletionReserve estimates increase monotonically across tier schemas")
        func testCompletionReserveEstimatesAreMonotonic() async throws {
            guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
            let container = try await loadTestModelContainer(id: Self.modelID)

            try await container.perform { context in
                let schemas = [
                    ("tier1", Self.tier1Schema),
                    ("tier2", Self.tier2Schema),
                    ("tier3", Self.tier3Schema),
                    ("tier4", Self.tier4Schema),
                ]

                var estimates: [(String, Int)] = []
                for (name, schema) in schemas {
                    let estimate = CompletionReserve.estimate(
                        schemaJSON: schema,
                        tokenizer: context.tokenizer
                    )
                    estimates.append((name, estimate))
                    print(
                        "[HardReserveStressTests] \(name): structuralReserve=\(estimate), hardReserve(\(Self.multiplier)x)=\(estimate * Self.multiplier)"
                    )
                }

                // All estimates must be positive
                for (name, estimate) in estimates {
                    #expect(estimate > 0, "\(name) estimate should be positive, got \(estimate)")
                }

                // Estimates must increase monotonically
                for i in 1 ..< estimates.count {
                    let (prevName, prevEst) = estimates[i - 1]
                    let (currName, currEst) = estimates[i]
                    #expect(
                        currEst > prevEst,
                        "\(currName) estimate (\(currEst)) should exceed \(prevName) estimate (\(prevEst))"
                    )
                }
            }
        }

        // MARK: - Behavior 2: Tier 1

        @Test("Tier 1 (3 fields) with 8x hardReserve produces valid JSON with all keys")
        func testTier1HardReserve() async throws {
            guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
            let container = try await loadTestModelContainer(id: Self.modelID)
            let structuralReserve = try await container.perform { context in
                CompletionReserve.estimate(
                    schemaJSON: Self.tier1Schema, tokenizer: context.tokenizer)
            }
            let hardReserve = structuralReserve * Self.multiplier
            let maxTokens = hardReserve * 2

            let raw = try await generateWithReserve(
                schema: Self.tier1Schema,
                maxTokens: maxTokens,
                hardReserve: hardReserve
            )

            let sanitized = sanitize(raw)
            print("[testTier1HardReserve] Output: \(sanitized.prefix(300))")

            let obj = try #require(
                try JSONSerialization.jsonObject(with: Data(sanitized.utf8)) as? [String: Any],
                "Tier 1 should produce valid JSON, got: \(sanitized.prefix(200))"
            )

            #expect(obj["title"] is String, "Should have 'title' key")
            #expect(obj["summary"] is String, "Should have 'summary' key")
            #expect(obj["conclusion"] is String, "Should have 'conclusion' key")
        }

        // MARK: - Behavior 3: Tier 2

        @Test(
            "Tier 2 (array of 3 items) with 8x hardReserve produces valid JSON with all keys and 3 items"
        )
        func testTier2HardReserve() async throws {
            guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
            let container = try await loadTestModelContainer(id: Self.modelID)
            let structuralReserve = try await container.perform { context in
                CompletionReserve.estimate(
                    schemaJSON: Self.tier2Schema, tokenizer: context.tokenizer)
            }
            let hardReserve = structuralReserve * Self.multiplier
            let maxTokens = hardReserve * 2

            let raw = try await generateWithReserve(
                schema: Self.tier2Schema,
                maxTokens: maxTokens,
                hardReserve: hardReserve
            )

            let sanitized = sanitize(raw)
            print("[testTier2HardReserve] Output: \(sanitized.prefix(500))")

            let obj = try #require(
                try JSONSerialization.jsonObject(with: Data(sanitized.utf8)) as? [String: Any],
                "Tier 2 should produce valid JSON, got: \(sanitized.prefix(200))"
            )

            #expect(obj["topic"] is String, "Should have 'topic' key")
            #expect(obj["overview"] is String, "Should have 'overview' key")

            let items = try #require(
                obj["items"] as? [[String: Any]],
                "Should have 'items' array"
            )
            #expect(items.count == 3, "Should have exactly 3 items, got \(items.count)")

            for (i, item) in items.enumerated() {
                #expect(item["name"] is String, "items[\(i)] should have 'name' key")
                #expect(item["description"] is String, "items[\(i)] should have 'description' key")
            }
        }

        // MARK: - Behavior 4: Tier 3

        @Test(
            "Tier 3 (2 groups x 3 entries) with 8x hardReserve produces valid JSON with correct nesting"
        )
        func testTier3HardReserve() async throws {
            guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
            let container = try await loadTestModelContainer(id: Self.modelID)
            let structuralReserve = try await container.perform { context in
                CompletionReserve.estimate(
                    schemaJSON: Self.tier3Schema, tokenizer: context.tokenizer)
            }
            let hardReserve = structuralReserve * Self.multiplier
            let maxTokens = hardReserve * 2

            let raw = try await generateWithReserve(
                schema: Self.tier3Schema,
                maxTokens: maxTokens,
                hardReserve: hardReserve
            )

            let sanitized = sanitize(raw)
            print("[testTier3HardReserve] Output: \(sanitized.prefix(500))")

            let obj = try #require(
                try JSONSerialization.jsonObject(with: Data(sanitized.utf8)) as? [String: Any],
                "Tier 3 should produce valid JSON, got: \(sanitized.prefix(200))"
            )

            #expect(obj["title"] is String, "Should have 'title' key")

            let groups = try #require(
                obj["groups"] as? [[String: Any]],
                "Should have 'groups' array"
            )
            #expect(groups.count == 2, "Should have exactly 2 groups, got \(groups.count)")

            for (gi, group) in groups.enumerated() {
                #expect(group["name"] is String, "groups[\(gi)] should have 'name' key")

                let entries = try #require(
                    group["entries"] as? [[String: Any]],
                    "groups[\(gi)] should have 'entries' array"
                )
                #expect(
                    entries.count == 3, "groups[\(gi)] should have 3 entries, got \(entries.count)")

                for (ei, entry) in entries.enumerated() {
                    #expect(
                        entry["label"] is String, "groups[\(gi)].entries[\(ei)] should have 'label'"
                    )
                    #expect(
                        entry["detail"] is String,
                        "groups[\(gi)].entries[\(ei)] should have 'detail'")
                }
            }
        }

        // MARK: - Behavior 5: Tier 4

        @Test("Tier 4 (3 days x 3 activities, ~40 fields) with 8x hardReserve produces valid JSON")
        func testTier4HardReserve() async throws {
            guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
            let container = try await loadTestModelContainer(id: Self.modelID)
            let structuralReserve = try await container.perform { context in
                CompletionReserve.estimate(
                    schemaJSON: Self.tier4Schema, tokenizer: context.tokenizer)
            }
            let hardReserve = structuralReserve * Self.multiplier
            let maxTokens = hardReserve * 2

            let raw = try await generateWithReserve(
                schema: Self.tier4Schema,
                maxTokens: maxTokens,
                hardReserve: hardReserve
            )

            let sanitized = sanitize(raw)
            print("[testTier4HardReserve] Output: \(sanitized.prefix(800))")

            let obj = try #require(
                try JSONSerialization.jsonObject(with: Data(sanitized.utf8)) as? [String: Any],
                "Tier 4 should produce valid JSON, got: \(sanitized.prefix(200))"
            )

            #expect(obj["title"] is String, "Should have 'title' key")
            #expect(obj["destination"] is String, "Should have 'destination' key")
            #expect(obj["description"] is String, "Should have 'description' key")
            #expect(obj["rationale"] is String, "Should have 'rationale' key")

            let days = try #require(
                obj["days"] as? [[String: Any]],
                "Should have 'days' array"
            )
            #expect(days.count == 3, "Should have exactly 3 days, got \(days.count)")

            for (di, day) in days.enumerated() {
                #expect(day["title"] is String, "days[\(di)] should have 'title'")
                #expect(day["subtitle"] is String, "days[\(di)] should have 'subtitle'")
                #expect(day["destination"] is String, "days[\(di)] should have 'destination'")

                let activities = try #require(
                    day["activities"] as? [[String: Any]],
                    "days[\(di)] should have 'activities' array"
                )
                #expect(
                    activities.count == 3,
                    "days[\(di)] should have 3 activities, got \(activities.count)")

                for (ai, activity) in activities.enumerated() {
                    #expect(
                        activity["type"] is String,
                        "days[\(di)].activities[\(ai)] should have 'type'")
                    #expect(
                        activity["title"] is String,
                        "days[\(di)].activities[\(ai)] should have 'title'")
                    #expect(
                        activity["description"] is String,
                        "days[\(di)].activities[\(ai)] should have 'description'")
                }
            }
        }

        // MARK: - GPU Memory Cleanup

        @Test("Cleanup: release GPU resources after stress tests")
        func releaseGPUResources() async {
            guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
            let before = GPU.snapshot()
            await releaseAllGPUMemory()
            let after = GPU.snapshot()
            let freed = before.activeMemory - after.activeMemory
            print(
                "[HardReserveCleanup] freed \(freed / (1024 * 1024))MB active, "
                    + "\(before.cacheMemory / (1024 * 1024))MB cache")
        }
    }

#endif
