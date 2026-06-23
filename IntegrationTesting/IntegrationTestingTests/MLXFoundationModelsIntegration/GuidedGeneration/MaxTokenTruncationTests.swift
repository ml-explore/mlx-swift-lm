// Copyright © 2025 Apple Inc.

#if FoundationModelsIntegration

    import Testing
    import Foundation
    import MLXLMCommon
    import MLX
    import FoundationModels
    @testable import MLXFoundationModels
    @testable import MLXGuidedGeneration

    /// Tests that guided generation surfaces typed errors when maxTokens is
    /// exhausted before the grammar reaches an accepting state.
    @Suite(.serialized, .timeLimit(.minutes(5)))
    struct MaxTokenTruncationTests {

        // MARK: - Incomplete Output Detection

        @Test(
            "GuidedGenerationLoop throws incompleteOutput when maxTokens exhausted before grammar stops"
        )
        func lowMaxTokensThrowsIncompleteOutput() async throws {
            guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
            let container = try await loadTestModelContainer(id: TestFixtures.defaultModelID)

            try await container.perform { context in
                // Build a schema requiring a JSON object with many required string
                // properties. Even the opening `{"` consumes multiple tokens, so
                // maxTokens=5 will never let the grammar reach a stop state.
                let complexSchema = """
                    {
                      "type": "object",
                      "properties": {
                        "firstName": { "type": "string" },
                        "lastName": { "type": "string" },
                        "email": { "type": "string" },
                        "phone": { "type": "string" },
                        "address": { "type": "string" }
                      },
                      "required": ["firstName", "lastName", "email", "phone", "address"],
                      "additionalProperties": false
                    }
                    """

                let xgTokenizer = try await MLXLanguageModel.makeXGTokenizer(
                    modelID: TestFixtures.defaultModelID,
                    tokenizer: context.tokenizer
                )

                let constraint = try GrammarConstraint(
                    tokenizer: xgTokenizer,
                    jsonSchema: complexSchema,
                    fastForward: true,
                    hostTokenizer: context.tokenizer
                )

                let userInput = UserInput(
                    chat: [.user("Fill in the contact form.")],
                    processing: .init()
                )
                let input = try await context.processor.prepare(input: userInput)

                // 5 tokens is far too few to complete a multi-property JSON object.
                #expect(throws: GuidedGenerationError.incompleteOutput) {
                    try GuidedGenerationLoop.run(
                        input: input,
                        context: context,
                        constraint: constraint,
                        maxTokens: 5,
                        vocabSize: Int(xgTokenizer.vocabSize)
                    ) { _ in true }
                }
            }
        }

        // MARK: - Normal Generation Succeeds

        @Test("Guided generation with sufficient tokens does not throw")
        func sufficientTokensDoesNotThrow() async throws {
            guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
            let model = makeTestModel(TestFixtures.defaultModelID)
            let executor = try makeMLXExecutor(for: model)

            let transcript = Transcript(entries: [
                .prompt(
                    Transcript.Prompt(
                        segments: [
                            .text(Transcript.TextSegment(content: "Return the number 7 as JSON."))
                        ], responseFormat: nil))
            ])

            // Int schema is tiny -- a single digit completes the grammar well
            // within the default maxTokens budget.
            let request = makeExecutorRequest(
                transcript: transcript,
                schema: Int.generationSchema
            )

            let stream = try await executeResponse(executor, request: request, model: model)

            var fullText = ""
            for try await event in stream {
                if let response = event as? LanguageModelExecutorGenerationChannel.Response,
                    case .appendText(let delta) = response.action
                {
                    fullText += delta.content
                }
            }

            let trimmed = fullText.trimmingCharacters(in: .whitespacesAndNewlines)
            #expect(!trimmed.isEmpty, "Should produce non-empty output")

            let data = trimmed.data(using: .utf8)!
            let parsed = try? JSONSerialization.jsonObject(with: data, options: .fragmentsAllowed)
            #expect(parsed != nil, "Output should be valid JSON: \(trimmed)")
        }

        // MARK: - Error Propagation Through Stream

        @Test("incompleteOutput error propagates through the ResponseStream")
        func errorPropagatesThroughStream() async throws {
            guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
            let container = try await loadTestModelContainer(id: TestFixtures.defaultModelID)

            try await container.perform { context in
                let xgTokenizer = try await MLXLanguageModel.makeXGTokenizer(
                    modelID: TestFixtures.defaultModelID,
                    tokenizer: context.tokenizer
                )

                // Array of strings schema -- needs at least an opening bracket,
                // a quoted string, and a closing bracket.
                let arraySchema = """
                    {
                      "type": "array",
                      "items": { "type": "string" },
                      "minItems": 3
                    }
                    """

                let constraint = try GrammarConstraint(
                    tokenizer: xgTokenizer,
                    jsonSchema: arraySchema,
                    fastForward: true,
                    hostTokenizer: context.tokenizer
                )

                let userInput = UserInput(
                    chat: [.user("List three colors.")],
                    processing: .init()
                )
                let input = try await context.processor.prepare(input: userInput)

                // 3 tokens cannot possibly produce ["x","y","z"]
                #expect(throws: GuidedGenerationError.incompleteOutput) {
                    try GuidedGenerationLoop.run(
                        input: input,
                        context: context,
                        constraint: constraint,
                        maxTokens: 3,
                        vocabSize: Int(xgTokenizer.vocabSize)
                    ) { _ in true }
                }
            }
        }
    }

#endif
