// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

import Foundation
import FoundationModels
import Testing

@testable import MLXFoundationModels
import MLXLMCommon

/// On-device behavioral checks: that wired sampling actually changes output.
///
/// Loads real models, so it lives in the IntegrationTesting xcodeproj (runs on a
/// 27 host). The shim *translation* is unit-tested in `SamplingModeShimTests`
/// (package target). The distributional assertion here is *ordinal* (greedy is
/// more deterministic than high-top-k), never an absolute variance band or
/// token-for-token reproducibility, because GPU reduction-order nondeterminism
/// can flip even an argmax decision.
///
/// DEVICE-TUNING NOTE: `sampleCount`, the prompt, and the top-k value below are
/// starting points; confirm on the first run that high-top-k genuinely produces
/// more distinct completions than greedy on the chosen model, and adjust if the
/// prompt is too constrained for sampling to diverge.
@Suite(.serialized, .timeLimit(.minutes(15)))
struct SamplingModeBehaviorTests {

    private static let sampleCount = 12
    private static let creativePrompt =
        "Write one short, imaginative sentence about the sea. Be unpredictable."

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private func promptTranscript(_ text: String) -> Transcript {
        Transcript(entries: [
            .prompt(
                Transcript.Prompt(
                    segments: [.text(Transcript.TextSegment(content: text))],
                    responseFormat: nil))
        ])
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private func responseText(_ stream: TestResponseStream) async throws -> String {
        var response = ""
        for try await event in stream {
            if case .appendText(let chunk, _, .response) = event {
                response += chunk
            }
        }
        return response
    }

    /// Number of distinct completions across `sampleCount` runs of the same prompt.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private func distinctCompletions(
        executor: MLXLanguageModel.Executor,
        model: MLXLanguageModel,
        options: GenerationOptions
    ) async throws -> Int {
        var seen = Set<String>()
        for _ in 0 ..< Self.sampleCount {
            let request = makeExecutorRequest(
                transcript: promptTranscript(Self.creativePrompt),
                generationOptions: options)
            let text = try await responseText(
                try await executeResponse(executor, request: request, model: model))
            seen.insert(text)
        }
        return seen.count
    }

    /// Greedy produces fewer distinct completions than high-top-k sampling —
    /// proving `samplingMode` actually reaches the sampler end-to-end, not just
    /// that the shim compiles. Ordinal, not absolute.
    @Test func greedyIsMoreDeterministicThanHighTopK() async throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let model = makeTestModel(TestFixtures.defaultModelID)
        let executor = try makeMLXExecutor(for: model)

        let greedyDistinct = try await distinctCompletions(
            executor: executor, model: model,
            options: GenerationOptions(samplingMode: .greedy, maximumResponseTokens: 24))
        let topKDistinct = try await distinctCompletions(
            executor: executor, model: model,
            options: GenerationOptions(
                samplingMode: .random(top: 200), maximumResponseTokens: 24))

        #expect(
            greedyDistinct < topKDistinct,
            "greedy distinct=\(greedyDistinct) should be < high-top-k distinct=\(topKDistinct)")
        await releaseAllGPUMemory()
    }
}

#endif  // FoundationModelsIntegration
