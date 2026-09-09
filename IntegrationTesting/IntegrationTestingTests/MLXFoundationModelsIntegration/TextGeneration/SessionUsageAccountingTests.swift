// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

import Testing
import Foundation
import FoundationModels
@testable import MLXFoundationModels

/// Assert on the public `LanguageModelSession.usage`. Check input tokens, not
/// only a total. The executor notifies `generationObserver` before it sends to
/// the channel, so an observer-based test passes when the consumer gets
/// nothing. The framework counts output tokens from text fragments, so a
/// total-only check passes when input tokens are zero.
@Suite(.serialized, .timeLimit(.minutes(10)))
struct SessionUsageAccountingTests {

    @Test
    func freshSessionReportsZeroUsage() async throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

        let model = makeTestModel(TestFixtures.defaultModelID)
        let session = LanguageModelSession(model: model, tools: [], instructions: nil)

        #expect(session.usage.input.totalTokenCount == 0)
        #expect(session.usage.input.cachedTokenCount == 0)
        #expect(session.usage.output.totalTokenCount == 0)
        #expect(session.usage.output.reasoningTokenCount == 0)
        #expect(session.usage.totalTokenCount == 0)
    }

    @Test
    func singleResponseReportsInputAndOutputTokens() async throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

        let model = makeTestModel(TestFixtures.defaultModelID)
        let session = LanguageModelSession(model: model, tools: [], instructions: nil)

        let response = try await session.respond(to: "Say 'hi' briefly.")
        let usage = response.usage

        #expect(
            usage.input.totalTokenCount > 0,
            "Zero input tokens means the adapter did not send updateUsage")
        #expect(usage.output.totalTokenCount > 0)
        #expect(
            usage.totalTokenCount == usage.input.totalTokenCount + usage.output.totalTokenCount)
        #expect(usage.input.cachedTokenCount <= usage.input.totalTokenCount)
        #expect(usage.output.reasoningTokenCount <= usage.output.totalTokenCount)

        #expect(session.usage.input.totalTokenCount == usage.input.totalTokenCount)
        #expect(session.usage.output.totalTokenCount == usage.output.totalTokenCount)
    }

    @Test
    func sessionSumsUsageAcrossTurns() async throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

        let model = makeTestModel(TestFixtures.defaultModelID)
        let session = LanguageModelSession(model: model, tools: [], instructions: nil)

        var expectedInput = 0
        var expectedOutput = 0
        var previousTotal = 0

        for prompt in ["Say 'hi' briefly.", "Name one color.", "Name one animal."] {
            let response = try await session.respond(to: prompt)

            #expect(response.usage.input.totalTokenCount > 0)
            #expect(response.usage.output.totalTokenCount > 0)

            expectedInput += response.usage.input.totalTokenCount
            expectedOutput += response.usage.output.totalTokenCount

            #expect(session.usage.input.totalTokenCount == expectedInput)
            #expect(session.usage.output.totalTokenCount == expectedOutput)
            #expect(
                session.usage.totalTokenCount > previousTotal,
                "Session usage must increase after every response")

            previousTotal = session.usage.totalTokenCount
        }
    }

    @Test
    func separateSessionsDoNotShareUsage() async throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

        let model = makeTestModel(TestFixtures.defaultModelID)
        let first = LanguageModelSession(model: model, tools: [], instructions: nil)
        _ = try await first.respond(to: "Say 'hi' briefly.")
        #expect(first.usage.totalTokenCount > 0)

        let second = LanguageModelSession(model: model, tools: [], instructions: nil)
        #expect(second.usage.totalTokenCount == 0)
    }

    /// The streaming path adds a usage delta for each snapshot. The `respond()`
    /// path adds one total. The two paths can disagree, so test streaming too.
    @Test
    func streamingReportsUsageMatchingFinalSnapshot() async throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

        let model = makeTestModel(TestFixtures.defaultModelID)
        let session = LanguageModelSession(model: model, tools: [], instructions: nil)

        var finalSnapshotUsage: LanguageModelSession.Usage?
        for try await snapshot in session.streamResponse(to: "Say hello in three words.") {
            finalSnapshotUsage = snapshot.usage
        }

        let usage = try #require(finalSnapshotUsage)
        #expect(usage.input.totalTokenCount > 0)
        #expect(usage.output.totalTokenCount > 0)
        #expect(session.usage.input.totalTokenCount == usage.input.totalTokenCount)
        #expect(session.usage.output.totalTokenCount == usage.output.totalTokenCount)
    }
}

#endif  // FoundationModelsIntegration
