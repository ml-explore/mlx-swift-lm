// Copyright © 2025 Apple Inc.

#if FoundationModelsIntegration

    import Testing
    import Foundation
    import FoundationModels
    @testable import MLXFoundationModels
    @testable import MLXGuidedGeneration

    /// Schema used by `incompleteOutputYieldsMetadata`. Five required string
    /// properties guarantee the grammar cannot reach a stop state within a
    /// small token budget.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Generable
    private struct ContactForm {
        @Guide(description: "The person's first name")
        let firstName: String
        @Guide(description: "The person's last name")
        let lastName: String
        @Guide(description: "Email address")
        let email: String
        @Guide(description: "Phone number")
        let phone: String
        @Guide(description: "Mailing address")
        let address: String
    }

    /// Tests for guided generation wiring in the Executor.
    ///
    /// These tests verify that schemas are properly threaded through the
    /// Executor -> ResponseStream -> GuidedGenerationLoop pipeline.
    @Suite(.serialized, .timeLimit(.minutes(5)))
    struct GuidedGenerationIntegrationTests {

        // MARK: - Schema Presence Tests

        @Test
        func schemaRequestUsesGuidedPath() async throws {
            guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
            let model = makeTestModel(TestFixtures.defaultModelID)
            let executor = try makeMLXExecutor(for: model)

            let transcript = Transcript(entries: [
                .prompt(
                    Transcript.Prompt(
                        segments: [
                            .text(Transcript.TextSegment(content: "What is 2+2? Reply as JSON."))
                        ], responseFormat: nil))
            ])

            let request = makeExecutorRequest(transcript: transcript, schema: Int.generationSchema)

            let stream = try await executeResponse(executor, request: request, model: model)

            var events: [LanguageModelExecutorGenerationChannel.Event] = []
            for try await event in stream {
                events.append(event)
            }

            #expect(events.count >= 2, "Should produce metadata and text events")

            guard
                let firstResponse = events.first
                    as? LanguageModelExecutorGenerationChannel.Response,
                case .updateMetadata = firstResponse.action
            else {
                Issue.record("First event should be metadataUpdate")
                return
            }

            let hasText = events.contains { event in
                if let response = event as? LanguageModelExecutorGenerationChannel.Response,
                    case .appendText = response.action
                {
                    return true
                }
                return false
            }
            #expect(hasText, "Should produce text deltas")
        }

        @Test
        func noSchemaUsesUnconstrainedPath() async throws {
            guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
            let model = makeTestModel(TestFixtures.defaultModelID)
            let executor = try makeMLXExecutor(for: model)

            let transcript = Transcript(entries: [
                .prompt(
                    Transcript.Prompt(
                        segments: [
                            .text(Transcript.TextSegment(content: "Hello"))
                        ], responseFormat: nil))
            ])

            let request = makeExecutorRequest(transcript: transcript)

            let stream = try await executeResponse(executor, request: request, model: model)

            var hasText = false
            for try await event in stream {
                if let response = event as? LanguageModelExecutorGenerationChannel.Response,
                    case .appendText = response.action
                {
                    hasText = true
                    break
                }
            }

            #expect(hasText, "Unconstrained path should still produce text")
        }

        // MARK: - Capability Flag Test

        @Test
        func supportsGuidedGenerationIsTrue() {
            guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
            let model = makeTestModel(TestFixtures.defaultModelID)
            #expect(model.capabilities.contains(.guidedGeneration))
        }

        // MARK: - Multi-Turn Schema Toggling

        @Test
        func multiTurnSchemaToggling() async throws {
            guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
            let model = makeTestModel(TestFixtures.defaultModelID)
            let executor = try makeMLXExecutor(for: model)

            let transcript1 = Transcript(entries: [
                .prompt(
                    Transcript.Prompt(
                        segments: [
                            .text(Transcript.TextSegment(content: "Say hello."))
                        ], responseFormat: nil))
            ])
            let request1 = makeExecutorRequest(transcript: transcript1)
            let stream1 = try await executeResponse(executor, request: request1, model: model)
            var text1 = ""
            for try await event in stream1 {
                if let response = event as? LanguageModelExecutorGenerationChannel.Response,
                    case .appendText(let delta) = response.action
                {
                    text1 += delta.content
                }
            }
            #expect(!text1.isEmpty, "Turn 1 (unconstrained) should produce text")

            let transcript2 = Transcript(entries: [
                .prompt(
                    Transcript.Prompt(
                        segments: [
                            .text(Transcript.TextSegment(content: "What is 1+1?"))
                        ], responseFormat: nil))
            ])
            let request2 = makeExecutorRequest(
                transcript: transcript2, schema: Int.generationSchema)
            let stream2 = try await executeResponse(executor, request: request2, model: model)
            var text2 = ""
            for try await event in stream2 {
                if let response = event as? LanguageModelExecutorGenerationChannel.Response,
                    case .appendText(let delta) = response.action
                {
                    text2 += delta.content
                }
            }
            let trimmed2 = text2.trimmingCharacters(in: .whitespacesAndNewlines)
            // Validate as a JSON integer. Don't decode via JSONSerialization or
            // JSONDecoder -- unbounded grammar + greedy decoding can produce
            // numbers exceeding both Int.max and NSDecimalNumber's 38-digit limit.
            #expect(!trimmed2.isEmpty, "Turn 2 should produce output")
            let isJSONInt =
                trimmed2.first == "-"
                ? trimmed2.dropFirst().allSatisfy(\.isWholeNumber)
                : trimmed2.allSatisfy(\.isWholeNumber)
            #expect(isJSONInt, "Turn 2 should be a valid JSON integer: \(trimmed2.prefix(50))")

            let transcript3 = Transcript(entries: [
                .prompt(
                    Transcript.Prompt(
                        segments: [
                            .text(Transcript.TextSegment(content: "Say goodbye."))
                        ], responseFormat: nil))
            ])
            let request3 = makeExecutorRequest(transcript: transcript3)
            let stream3 = try await executeResponse(executor, request: request3, model: model)
            var text3 = ""
            for try await event in stream3 {
                if let response = event as? LanguageModelExecutorGenerationChannel.Response,
                    case .appendText(let delta) = response.action
                {
                    text3 += delta.content
                }
            }
            #expect(!text3.isEmpty, "Turn 3 (unconstrained) should produce text")
        }

        // MARK: - Concurrent Executor Sessions

        @Test
        func concurrentGuidedSessions() async throws {
            guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
            let model = makeTestModel(TestFixtures.defaultModelID)

            try await withThrowingTaskGroup(of: String.self) { group in
                group.addTask {
                    let executor = try makeMLXExecutor(for: model)
                    let transcript = Transcript(entries: [
                        .prompt(
                            Transcript.Prompt(
                                segments: [
                                    .text(Transcript.TextSegment(content: "What is 2+2?"))
                                ], responseFormat: nil))
                    ])
                    let request = makeExecutorRequest(
                        transcript: transcript,
                        schema: Int.generationSchema
                    )
                    let stream = try await executeResponse(executor, request: request, model: model)
                    var text = ""
                    for try await event in stream {
                        if let response = event as? LanguageModelExecutorGenerationChannel.Response,
                            case .appendText(let delta) = response.action
                        {
                            text += delta.content
                        }
                    }
                    return text
                }
                group.addTask {
                    let executor = try makeMLXExecutor(for: model)
                    let transcript = Transcript(entries: [
                        .prompt(
                            Transcript.Prompt(
                                segments: [
                                    .text(Transcript.TextSegment(content: "Is the sky blue?"))
                                ], responseFormat: nil))
                    ])
                    let request = makeExecutorRequest(
                        transcript: transcript,
                        schema: Bool.generationSchema
                    )
                    let stream = try await executeResponse(executor, request: request, model: model)
                    var text = ""
                    for try await event in stream {
                        if let response = event as? LanguageModelExecutorGenerationChannel.Response,
                            case .appendText(let delta) = response.action
                        {
                            text += delta.content
                        }
                    }
                    return text
                }

                for try await text in group {
                    let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
                    #expect(!trimmed.isEmpty, "Concurrent session should produce output")
                }
            }
        }

        // MARK: - Incomplete Output Metadata Warning

        @Test("incompleteOutput yields metadata warning when maxTokens exhausted")
        func incompleteOutputYieldsMetadata() async throws {
            guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
            let model = makeTestModel(TestFixtures.defaultModelID)
            let executor = try makeMLXExecutor(for: model)

            let transcript = Transcript(entries: [
                .prompt(
                    Transcript.Prompt(
                        segments: [
                            .text(Transcript.TextSegment(content: "Fill in the contact form."))
                        ], responseFormat: nil))
            ])

            // ContactForm has 5 required string properties; 8 tokens is provably
            // insufficient for the grammar to reach a stop state.
            let request = makeExecutorRequest(
                transcript: transcript,
                schema: ContactForm.generationSchema,
                generationOptions: GenerationOptions(maximumResponseTokens: 8)
            )

            let stream = try await executeResponse(executor, request: request, model: model)

            var events: [LanguageModelExecutorGenerationChannel.Event] = []
            for try await event in stream {
                events.append(event)
            }

            let incompleteIdx = events.firstIndex { event in
                guard let response = event as? LanguageModelExecutorGenerationChannel.Response,
                    case .updateMetadata(let metadata) = response.action
                else { return false }
                return (metadata.values["incompleteOutput"] as? Bool) == true
            }
            #expect(
                incompleteIdx != nil,
                "Executor should emit metadataUpdate with incompleteOutput=true when the budget is exhausted before the grammar can complete"
            )

            if let incompleteIdx,
                let lastTextIdx = events.lastIndex(where: {
                    if let response = $0 as? LanguageModelExecutorGenerationChannel.Response,
                        case .appendText = response.action
                    {
                        return true
                    } else {
                        return false
                    }
                })
            {
                #expect(
                    incompleteIdx > lastTextIdx,
                    "incompleteOutput metadata must follow all text deltas, not precede them")
            }
        }
    }

#endif
