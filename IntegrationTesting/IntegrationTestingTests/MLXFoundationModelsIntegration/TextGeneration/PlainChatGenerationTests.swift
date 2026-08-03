// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

import Testing
import Foundation
import FoundationModels
import MLXLMCommon
@testable import MLXFoundationModels

/// Plain-chat generation smoke: a request with no schema and no tools falls
/// through to unconstrained generation and emits text deltas.
///
/// Loads a real model, so it lives in the IntegrationTesting xcodeproj. This
/// behavior is independent of guided generation (which only engages for
/// schema/tool requests); guided generation lives in the MLXGuidedGeneration
/// library and is always available alongside the adapter.
@Suite(.serialized)
struct PlainChatGenerationTests {

    @Test("Plain chat request completes (falls through to unconstrained generation)")
    func chatRequestFallsThrough() async throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let model = makeTestModel(TestFixtures.gemmaModelID)
        let executor = try makeMLXExecutor(for: model)
        let transcript = Transcript(entries: [
            .prompt(
                Transcript.Prompt(
                    segments: [
                        .text(Transcript.TextSegment(content: "Say hi."))
                    ], responseFormat: nil))
        ])
        let request = makeExecutorRequest(
            transcript: transcript,
            generationOptions: GenerationOptions(maximumResponseTokens: 8)
        )
        let stream = try await executeResponse(executor, request: request, model: model)
        var sawTextDelta = false
        for try await event in stream {
            if case .appendText(_, _, .response) = event {
                sawTextDelta = true
            }
        }
        #expect(sawTextDelta, "Plain chat without schema/tools should emit text deltas")
        await releaseAllGPUMemory()
    }
}

#endif  // FoundationModelsIntegration
