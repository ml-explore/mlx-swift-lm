// Copyright © 2025 Apple Inc.

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

import Testing
import Foundation
import FoundationModels
@testable import MLXFoundationModels
@testable import MLXGuidedGeneration

/// Tests that `warmUp()` pre-creates the GrammarTokenizer for guided generation.
@Suite(.serialized, .timeLimit(.minutes(5)))
struct PrewarmGrammarTests {

    @Test
    func prewarmCreatesXGTokenizer() async throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let model = makeTestModel(TestFixtures.defaultModelID)
        let executor = try makeMLXExecutor(for: model)

        // warmUp loads weights, compiles shaders, and pre-creates the
        // model-keyed GrammarTokenizer (the expensive vocab-extraction step
        // a guided consumer would otherwise pay on first respond()).
        try await model.warmUp()

        // Assert the genuine cache hit, not merely that a later respond works
        // (a guided respond succeeds with or without warmup — only the seam
        // proves warmUp did the pre-creation).
        let cached = await MLXLanguageModel.hasCachedXGTokenizer(modelID: model.modelID)
        #expect(cached, "warmUp should pre-create the GrammarTokenizer")

        // And a guided generation still succeeds end-to-end after warmUp.
        let transcript = Transcript(entries: [
            .prompt(
                Transcript.Prompt(
                    segments: [
                        .text(Transcript.TextSegment(content: "Return 42"))
                    ], responseFormat: nil))
        ])
        let request = makeExecutorRequest(
            transcript: transcript,
            schema: Int.generationSchema
        )
        let stream = try await executeResponse(executor, request: request, model: model)

        var hasText = false
        for try await event in stream {
            if case .appendText(_, _, .response) = event {
                hasText = true
                break
            }
        }
        #expect(hasText, "Guided generation after warmUp should produce text")
    }

    @Test
    func prewarmWithoutSchemaStillWorks() async throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let model = makeTestModel(TestFixtures.defaultModelID)
        let executor = try makeMLXExecutor(for: model)

        // warmUp warms weights + shaders (+ the GrammarTokenizer); an unconstrained
        // respond afterward must still work — the GrammarTokenizer pre-creation must
        // not interfere with the no-schema path.
        try await model.warmUp()

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
            if case .appendText(_, _, .response) = event {
                hasText = true
                break
            }
        }
        #expect(hasText, "Unconstrained generation after warmUp should produce text")
    }
}

#endif
