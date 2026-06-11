// Copyright © 2025 Apple Inc.

import Foundation
import FoundationModels
import Testing

@testable import MLXFoundationModels

/// Integration tests for real MLX inference.
///
/// These tests require model download on first run (~300MB from Hugging Face).
/// Subsequent runs use the cached model.
///
/// Note: These tests have a 5-minute timeout to allow for model download
/// and first-run shader compilation.
@Suite(.serialized, .timeLimit(.minutes(5)))
struct IntegrationTests {

    // MARK: - Real Inference Tests

    @Test
    func testRealInferenceProducesOutput() async throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let model = makeTestModel(TestFixtures.defaultModelID)
        let session = LanguageModelSession(
            model: model,
            tools: [],
            instructions: nil
        )

        let response = try await session.respond(to: "What is 2 plus 2?")

        // Should get a non-empty response
        #expect(!response.content.isEmpty, "Response should not be empty")

        // Response should be real inference output
        #expect(
            response.content != "Hello! This is a test response from MLX.",
            "Response should be real inference, not canned"
        )
    }

    @Test
    func testStreamingRealInference() async throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let model = makeTestModel(TestFixtures.defaultModelID)
        let session = LanguageModelSession(
            model: model,
            tools: [],
            instructions: nil
        )

        let stream = session.streamResponse(to: "Say hello in three words.")

        var chunks: [String] = []
        for try await partial in stream {
            chunks.append(partial.content)
        }

        // Should have received multiple streaming updates
        #expect(chunks.count > 1, "Should receive multiple streaming chunks")

        // Final content should not be empty
        #expect(!chunks.last!.isEmpty, "Final chunk should not be empty")
    }

    @Test
    func testModelIdentifierInMetadata() async throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let model = makeTestModel(TestFixtures.defaultModelID)
        let executor = try MLXLanguageModel.Executor(
            configuration: MLXLanguageModel.Executor.Configuration(
                modelIdentifier: model.modelIdentifier)
        )

        let transcript = Transcript(entries: [
            .prompt(
                Transcript.Prompt(
                    segments: [
                        .text(Transcript.TextSegment(content: "Hello"))
                    ], responseFormat: nil))
        ])

        let request = LanguageModelExecutorGenerationRequest(
            id: UUID(),
            transcript: transcript,
            enabledTools: [],
            generationOptions: GenerationOptions(),
            contextOptions: ContextOptions(),
            metadata: [:]
        )
        let channel = LanguageModelExecutorGenerationChannel()
        let respondTask = Task {
            try await executor.respond(to: request, model: model, streamingInto: channel)
        }

        var events: [LanguageModelExecutorGenerationChannel.Event] = []
        for try await event in channel {
            events.append(event)
            if events.count >= 3 {  // Get a few events
                break
            }
        }
        respondTask.cancel()
        try? await respondTask.value

        // First event should be metadata
        guard let response = events.first as? LanguageModelExecutorGenerationChannel.Response,
            case .updateMetadata(let metadata) = response.action
        else {
            Issue.record("First event should be metadataUpdate")
            return
        }

        #expect(
            metadata.values["modelIdentifier"] != nil,
            "Metadata should contain model identifier"
        )
    }

    @Test
    func testMultiTurnConversation() async throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let model = makeTestModel(TestFixtures.defaultModelID)
        let session = LanguageModelSession(
            model: model,
            tools: [],
            instructions: nil
        )

        // First turn
        let response1 = try await session.respond(to: "My name is Alice.")

        #expect(!response1.content.isEmpty, "First response should not be empty")

        // Second turn - model should have context from first turn
        let response2 = try await session.respond(to: "What is my name?")

        #expect(!response2.content.isEmpty, "Second response should not be empty")
    }

    // MARK: - Prewarm / WarmUp Tests

    /// Builds a one-prompt transcript for the warmup/respond tests below.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private func singlePromptTranscript(_ text: String) -> Transcript {
        Transcript(entries: [
            .prompt(
                Transcript.Prompt(
                    segments: [
                        .text(Transcript.TextSegment(content: text))
                    ], responseFormat: nil))
        ])
    }

    /// R3 (weights) + R2 (shaders, structural proxy): `warmUp()` loads the
    /// model and runs a real forward pass. `.available` proves only that the
    /// weights are on disk (it derives from `config.json`, independent of
    /// shader compilation); the fact that `warmUp()` returned without throwing
    /// proves the 1-token generate seam ran to completion — the closest we can
    /// assert to "shaders compiled" without a stopwatch (timing is off-CI).
    @Test
    func testWarmUpLoadsWeightsAndRunsForwardPass() async throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let model = makeTestModel(TestFixtures.defaultModelID)

        try await model.warmUp()

        let available = await model.availability
        #expect(available == .available, "Model should be available after warmUp")
    }

    /// R2/R3: a real `respond()` after `warmUp()` produces output and completes
    /// without a Metal command-buffer crash. Asserts completion-without-throw,
    /// not timing.
    @Test
    func testRespondSucceedsAfterWarmUp() async throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let model = makeTestModel(TestFixtures.defaultModelID)
        let executor = try makeMLXExecutor(for: model)

        try await model.warmUp()

        let request = makeExecutorRequest(transcript: singlePromptTranscript("Hello"))
        let stream = try await executeResponse(executor, request: request, model: model)

        var hasOutput = false
        for try await _ in stream {
            hasOutput = true
            break
        }
        #expect(hasOutput, "respond after warmUp should produce output")
    }

    /// The executor's `prewarm(model:transcript:)` witness does a
    /// fire-and-forget warmup. It must not crash, and a subsequent
    /// `respond` must succeed. The background warmup Task isn't
    /// deterministically observable — deterministic warmup assertions
    /// live in the `warmUp()` tests above.
    @Test
    func testPrewarmDoesNotCrash() async throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let model = makeTestModel(TestFixtures.defaultModelID)
        let executor = try makeMLXExecutor(for: model)

        // Edge (R4): an empty transcript must still be safe — warmUp ignores
        // the transcript and uses a fixed dummy prompt.
        executor.prewarm(model: model, transcript: Transcript(entries: []))

        let request = makeExecutorRequest(transcript: singlePromptTranscript("Hello"))
        let stream = try await executeResponse(executor, request: request, model: model)

        var hasOutput = false
        for try await _ in stream {
            hasOutput = true
            break
        }
        #expect(hasOutput, "Should produce output after prewarm")
    }

    /// R11 / Risks: `warmUp()` is safe to call repeatedly and concurrently. The
    /// second (cache-deduped) call returns fast; the cold concurrent section
    /// exercises the `ModelCache` load-dedup path and the warmup-overlapping-
    /// respond serialization the warmup routes through `container.perform`.
    @Test
    func testWarmUpIsIdempotentAndConcurrencySafe() async throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let model = makeTestModel(TestFixtures.defaultModelID)

        // Idempotence: twice is safe; the second call returns fast from cache.
        try await model.warmUp()
        try await model.warmUp()

        // Evict so the concurrent section starts cold — otherwise the cached
        // container short-circuits ModelCache.load before `container.perform`,
        // and neither the load-dedup nor the GPU/serialization path runs.
        await releaseAllGPUMemory()

        // Concurrent warmups from cold: the second coalesces onto the first's
        // in-flight load task (ModelCache dedup), so they share one forward
        // pass rather than racing two — exercises the dedup path without crash.
        async let w1: Void = model.warmUp()
        async let w2: Void = model.warmUp()
        _ = try await (w1, w2)

        // The real serialization case: a warmup overlapping a respond — two
        // independent entry points each taking the SerialAccessContainer lock
        // for their GPU work, which must not race on the global Stream.gpu.
        await releaseAllGPUMemory()
        let executor = try makeMLXExecutor(for: model)
        let request = makeExecutorRequest(transcript: singlePromptTranscript("Hello"))
        async let warm: Void = model.warmUp()
        let stream = try await executeResponse(executor, request: request, model: model)
        var hasOutput = false
        for try await _ in stream {
            hasOutput = true
            break
        }
        try await warm
        #expect(hasOutput, "respond overlapping a warmUp should still produce output")
    }

    /// R4 (error path): `warmUp()` on a bogus model id throws, but the
    /// executor's fire-and-forget `prewarm` swallows it and never crashes the
    /// caller.
    @Test
    func testWarmUpErrorIsNonFatalThroughPrewarm() async throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let bogus = makeTestModel("definitely/not-a-real-model-zzz")

        // warmUp surfaces the failure to a direct caller...
        await #expect(throws: (any Error).self) {
            try await bogus.warmUp()
        }

        // ...but prewarm's fire-and-forget Task swallows it. This call returns
        // immediately and must not crash the caller.
        let executor = try makeMLXExecutor(for: bogus)
        executor.prewarm(model: bogus, transcript: Transcript(entries: []))
    }

    // MARK: - Stream Cancellation Tests

    @Test
    func testStreamCancellationDoesNotCrash() async throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let model = makeTestModel(TestFixtures.defaultModelID)
        let executor = try MLXLanguageModel.Executor(
            configuration: MLXLanguageModel.Executor.Configuration(
                modelIdentifier: model.modelIdentifier)
        )

        let transcript = Transcript(entries: [
            .prompt(
                Transcript.Prompt(
                    segments: [
                        .text(Transcript.TextSegment(content: "Write a long story about a dragon."))
                    ], responseFormat: nil))
        ])

        let request = LanguageModelExecutorGenerationRequest(
            id: UUID(),
            transcript: transcript,
            enabledTools: [],
            generationOptions: GenerationOptions(),
            contextOptions: ContextOptions(),
            metadata: [:]
        )
        let channel = LanguageModelExecutorGenerationChannel()
        let respondTask = Task {
            try await executor.respond(to: request, model: model, streamingInto: channel)
        }

        var tokenCount = 0
        for try await event in channel {
            if let response = event as? LanguageModelExecutorGenerationChannel.Response,
                case .appendText = response.action
            {
                tokenCount += 1
            }
            // Cancel early after a few tokens
            if tokenCount >= 5 {
                break
            }
        }
        // Cancel the respond task since we broke out early
        respondTask.cancel()

        #expect(tokenCount >= 5, "Should have received at least 5 tokens before cancellation")
    }
}
