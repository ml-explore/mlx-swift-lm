// Copyright © 2025 Apple Inc.

#if FoundationModelsIntegration

    import Foundation
    import FoundationModels
    import Testing

    @testable import MLXFoundationModels
    import MLXLMCommon

    /// Verifies the customizer-vended profile actually drives reasoning
    /// routing in `Executor.respond`. Pairs the unit-level behavior assertions
    /// (override-the-baseline; capability-gate suppression) here; on-device
    /// characterization lives in `ReasoningCapabilityGateTests`.
    @Suite(.serialized, .timeLimit(.minutes(15)))
    struct CustomizerProfileRoutingTests {

        enum Models {
            static let qwen3 = "mlx-community/Qwen3-1.7B-4bit"
            static let r1Distill = "mlx-community/DeepSeek-R1-Distill-Qwen-1.5B-4bit"
        }

        /// A customizer that swaps the reasoning delimiter pair on a per-instance
        /// basis. Verifies that two instances with the same model id but different
        /// customizers get different reasoning behavior, and that the per-call
        /// profile does not pollute the shared container.
        struct DelimiterCustomizer: ModelCustomizer {
            let start: String
            let end: String
            func profile(for context: LoadedModelContext) -> ModelProfile {
                var profile = context.inferred
                if profile.reasoningConfig != nil {
                    profile.reasoningConfig?.startDelimiter = start
                    profile.reasoningConfig?.endDelimiter = end
                }
                return profile
            }
        }

        @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
        private func collect(
            _ stream: TestResponseStream
        ) async throws -> (reasoning: String, response: String) {
            var reasoning = ""
            var response = ""
            for try await event in stream {
                if let r = event as? LanguageModelExecutorGenerationChannel.Reasoning,
                    case .appendText(let fragment) = r.action
                {
                    reasoning += fragment.content
                } else if let r = event as? LanguageModelExecutorGenerationChannel.Response,
                    case .appendText(let fragment) = r.action
                {
                    response += fragment.content
                }
            }
            return (reasoning, response)
        }

        @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
        private func promptTranscript(_ text: String) -> Transcript {
            Transcript(entries: [
                .prompt(
                    Transcript.Prompt(
                        segments: [.text(Transcript.TextSegment(content: text))],
                        responseFormat: nil))
            ])
        }

        // MARK: - Override path: customizer-supplied delimiters reach generation

        @Test func customizerDelimitersDriveRouting() async throws {
            guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
            let model = makeReasoningTestModel(
                Models.qwen3,
                customizer: DelimiterCustomizer(start: "<reason>", end: "</reason>"))
            let executor = try makeMLXExecutor(for: model)
            let request = makeExecutorRequest(
                transcript: promptTranscript("What is 17 times 24? Think step by step."),
                generationOptions: GenerationOptions(maximumResponseTokens: 256))
            let stream = try await executeResponse(executor, request: request, model: model)
            let result = try await collect(stream)
            // Qwen3 emits "<think>" in-stream; with the customizer rewriting
            // delimiters to "<reason>" / "</reason>", the scanner no longer
            // recognizes "<think>", so reasoning routing degrades and the
            // raw <think> text leaks into .response. This proves the profile
            // overrode the inferred delimiters at the routing layer.
            #expect(result.response.contains("<think>"))
            #expect(result.reasoning.isEmpty || !result.reasoning.contains("</think>"))
        }

        // MARK: - Two instances, same id, different customizers, no cross-contamination

        /// Sequential same-id instances must observe their own customizer's
        /// behavior; the shared container is reused but the profile is never
        /// written to it.
        @Test func sequentialInstancesGetIsolatedProfiles() async throws {
            guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
            let inferring = makeReasoningTestModel(Models.qwen3)
            let inferringExecutor = try makeMLXExecutor(for: inferring)
            let request = makeExecutorRequest(
                transcript: promptTranscript("Reply with the single word OK."),
                generationOptions: GenerationOptions(maximumResponseTokens: 64))
            let baselineStream = try await executeResponse(
                inferringExecutor, request: request, model: inferring)
            let baseline = try await collect(baselineStream)
            // Default Qwen3 routing: <think> is the recognized delimiter, so
            // reasoning routes (non-empty) and never leaks into response.
            #expect(!baseline.response.contains("<think>"))

            let overriding = makeReasoningTestModel(
                Models.qwen3,
                customizer: DelimiterCustomizer(start: "<reason>", end: "</reason>"))
            let overridingExecutor = try makeMLXExecutor(for: overriding)
            let overrideStream = try await executeResponse(
                overridingExecutor, request: request, model: overriding)
            let override = try await collect(overrideStream)
            // With the override, Qwen3's literal <think> tokens are not consumed
            // by the routing scanner, so they pass through to .response. This
            // proves the override took effect on this instance.
            #expect(override.response.contains("<think>"))

            // Now repeat the baseline call: the customizer override must not have
            // contaminated the shared container; default routing must still work.
            let baselineAgainStream = try await executeResponse(
                inferringExecutor, request: request, model: inferring)
            let baselineAgain = try await collect(baselineAgainStream)
            #expect(!baselineAgain.response.contains("<think>"))
        }

        /// Both sequential AND concurrent variants of the
        /// same-id/different-customizer isolation check are covered. The concurrent
        /// version interleaves two `respond` calls on the shared `ModelContainer`
        /// actor and verifies each instance saw only its own customizer's profile.
        /// If the profile ever leaked into the cached `ModelContext` or
        /// `Executor.Configuration`, this test would observe one instance's
        /// behavior on the other's output.
        @Test func concurrentInstancesGetIsolatedProfiles() async throws {
            guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

            // Pre-warm the container so neither concurrent task pays for the
            // download on its critical path. This isolates the test to the
            // profile-resolution race rather than the load race.
            _ = try await loadTestModelContainer(id: Models.qwen3)

            let inferring = makeReasoningTestModel(Models.qwen3)
            let inferringExecutor = try makeMLXExecutor(for: inferring)
            let overriding = makeReasoningTestModel(
                Models.qwen3,
                customizer: DelimiterCustomizer(start: "<reason>", end: "</reason>"))
            let overridingExecutor = try makeMLXExecutor(for: overriding)

            let request = makeExecutorRequest(
                transcript: promptTranscript("Reply with the single word OK."),
                generationOptions: GenerationOptions(maximumResponseTokens: 64))

            async let baselineCollected: (reasoning: String, response: String) = {
                let stream = try await executeResponse(
                    inferringExecutor, request: request, model: inferring)
                return try await collect(stream)
            }()
            async let overrideCollected: (reasoning: String, response: String) = {
                let stream = try await executeResponse(
                    overridingExecutor, request: request, model: overriding)
                return try await collect(stream)
            }()
            let baseline = try await baselineCollected
            let override = try await overrideCollected

            // Each instance must reflect its own customizer's view of the world,
            // even though they ran concurrently against the shared container.
            // Inferring instance: <think> consumed by the routing scanner.
            #expect(!baseline.response.contains("<think>"))
            // Overriding instance: customizer rewrote delimiters, scanner doesn't
            // recognize <think>, raw text leaks to .response — proof the override
            // reached this instance and not the other.
            #expect(override.response.contains("<think>"))
        }
    }

#endif  // FoundationModelsIntegration
