// Copyright © 2025 Apple Inc.

#if FoundationModelsIntegration

    import Testing
    import Foundation
    import FoundationModels
    @testable import MLXFoundationModels

    /// Verifies that guided generation streams multiple text delta events
    /// rather than buffering the entire output into a single emission.
    @Suite(.serialized, .timeLimit(.minutes(15)))
    struct StreamingDeltaTests {

        static let modelID = "mlx-community/Qwen2.5-3B-Instruct-4bit"

        // MARK: - Behavior 1: Multiple text deltas

        @Test
        func stringSchemaYieldsMultipleTextDeltas() async throws {
            guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
            let model = makeTestModel(Self.modelID)
            let executor = try makeMLXExecutor(for: model)

            let request = makeExecutorRequest(
                transcript: transcript("Name a color."),
                schema: String.generationSchema
            )

            let stream = try await executeResponse(executor, request: request, model: model)
            var deltaCount = 0
            for try await event in stream {
                if let response = event as? LanguageModelExecutorGenerationChannel.Response,
                    case .appendText = response.action
                {
                    deltaCount += 1
                }
            }

            #expect(deltaCount > 1, "Expected multiple text delta events, got \(deltaCount)")
        }

        // MARK: - Behavior 2: Concatenated deltas form valid JSON

        @Test
        func concatenatedDeltasAreValidJSON() async throws {
            guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
            let model = makeTestModel(Self.modelID)
            let executor = try makeMLXExecutor(for: model)

            let request = makeExecutorRequest(
                transcript: transcript("Name a color."),
                schema: String.generationSchema
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

            let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
            #expect(!trimmed.isEmpty, "Output should be non-empty")

            let data = try #require(trimmed.data(using: .utf8), "UTF-8 encoding failed")
            let parsed = try? JSONSerialization.jsonObject(with: data, options: .fragmentsAllowed)
            #expect(parsed != nil, "Concatenated deltas should be valid JSON: \(trimmed)")

            let decoded = try JSONDecoder().decode(String.self, from: Data(trimmed.utf8))
            #expect(!decoded.isEmpty, "Decoded string should be non-empty")
        }

        // MARK: - Helpers

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
    }

#endif
