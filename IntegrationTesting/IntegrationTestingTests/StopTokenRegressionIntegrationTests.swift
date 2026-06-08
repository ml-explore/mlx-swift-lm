// Copyright © 2026 Apple Inc.

#if GuidedGenerationSupport

    import Testing
    import Foundation
    import MLXLMCommon
    @testable import MLXFoundationModels

    /// Model-loading regression tests for the stop-token set that
    /// `GuidedGenerationLoop` uses to detect end-of-generation. These load real
    /// Gemma / Qwen models, so they live in the IntegrationTesting xcodeproj. The
    /// model-free supply-path check lives in the package target
    /// (`StopTokenRegressionTests`).
    ///
    /// The stop set must union `tokenizer.eosTokenId`,
    /// `configuration.extraEOSTokens`, AND `configuration.eosTokenIds` — the
    /// field populated from `generation_config.json`'s `eos_token_id` at
    /// model-load time. Chat models like Gemma 3 ship
    /// `eos_token_id: [1, 106]` (`<eos>` + `<end_of_turn>`), and that array is
    /// the only source that includes the chat turn-ender. Without it,
    /// Gemma-family models spew tokens past `<end_of_turn>` and never trigger
    /// the stop check.
    @Suite(.serialized)
    struct StopTokenRegressionIntegrationTests {

        /// Gemma 3 270M's tokenizer resolves `eosTokenId` to `<eos>` (id 1), but
        /// the chat turn ender is `<end_of_turn>` (id 106). Only
        /// `configuration.eosTokenIds` (from `generation_config.json`) surfaces
        /// 106. The stop set must include both, or generation never terminates
        /// at the turn boundary.
        @Test("Gemma 3 270M: stop set includes <end_of_turn>")
        func gemmaStopSetIncludesEndOfTurn() async throws {
            guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
            try await withContext(modelID: TestFixtures.gemmaModelID) { tokenizer, configuration in
                let stopSet = GuidedGenerationLoop.buildStopTokenIDs(
                    tokenizer: tokenizer,
                    configuration: configuration
                )

                // <eos> (primary EOS) must remain.
                #expect(
                    stopSet.contains(1),
                    "Gemma stop set must include id 1 (<eos>). Got \(stopSet.sorted())"
                )
                // <end_of_turn> (chat turn ender) must be present — this is the
                // token the chat-tuned model actually emits at turn boundaries.
                #expect(
                    stopSet.contains(106),
                    "Gemma stop set must include id 106 (<end_of_turn>). Got \(stopSet.sorted())"
                )
            }
        }

        /// Qwen 2.5 3B's tokenizer resolves `eosTokenId` directly to
        /// `<|im_end|>` (id 151645). This asserts that source lands in
        /// the stop set.
        @Test("Qwen 2.5 3B: stop set includes <|im_end|>")
        func qwenStopSetIncludesImEnd() async throws {
            guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
            try await withContext(modelID: TestFixtures.defaultModelID) {
                tokenizer, configuration in
                let stopSet = GuidedGenerationLoop.buildStopTokenIDs(
                    tokenizer: tokenizer,
                    configuration: configuration
                )

                #expect(
                    stopSet.contains(151645),
                    "Qwen stop set must include id 151645 (<|im_end|>). Got \(stopSet.sorted())"
                )
            }
        }

        /// A customizer-supplied stop token unions into the stop set
        /// without mutating the cached `ModelConfiguration`. Uses Qwen because its
        /// `<|endoftext|>` token id is well-known and absent from the default
        /// chat-stop set.
        @Test("additionalStopTokens unions into stop set without mutating cached config")
        func additionalStopTokensUnioned() async throws {
            guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
            try await withContext(modelID: TestFixtures.defaultModelID) {
                tokenizer, configuration in
                let extraTokenID = tokenizer.convertTokenToId("<|endoftext|>")
                guard let extraTokenID else {
                    Issue.record(
                        "Test fixture tokenizer is missing <|endoftext|>; cannot verify union")
                    return
                }
                // Baseline: <|endoftext|> isn't in the default stop set for the
                // unconstrained-chat path (Qwen's chat turn-ender is <|im_end|>).
                let baseline = GuidedGenerationLoop.buildStopTokenIDs(
                    tokenizer: tokenizer, configuration: configuration)
                // The cached configuration's extraEOSTokens is untouched by this
                // call site — assert that the union happens at the boundary.
                let extended = GuidedGenerationLoop.buildStopTokenIDs(
                    tokenizer: tokenizer, configuration: configuration,
                    additionalStopTokens: ["<|endoftext|>"])
                #expect(extended.contains(extraTokenID))
                #expect(
                    extended == baseline.union([extraTokenID]),
                    "extended set must be exactly baseline ∪ {<|endoftext|>}; got \(extended.subtracting(baseline.union([extraTokenID])))"
                )
                // The cached configuration must not have been mutated.
                #expect(!configuration.extraEOSTokens.contains("<|endoftext|>"))
            }
        }

        /// An empty `additionalStopTokens` argument is a no-op — the stop set
        /// matches the baseline.
        @Test("empty additionalStopTokens preserves the baseline stop set")
        func additionalStopTokensEmptyIsNoop() async throws {
            guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
            try await withContext(modelID: TestFixtures.gemmaModelID) { tokenizer, configuration in
                let baseline = GuidedGenerationLoop.buildStopTokenIDs(
                    tokenizer: tokenizer, configuration: configuration)
                let withEmpty = GuidedGenerationLoop.buildStopTokenIDs(
                    tokenizer: tokenizer, configuration: configuration,
                    additionalStopTokens: [])
                #expect(baseline == withEmpty)
            }
        }

        // MARK: - Helpers

        @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
        private func withContext(
            modelID: String,
            _ body: @Sendable (any Tokenizer, ModelConfiguration) async throws -> Void
        ) async throws {
            let container = try await loadTestModelContainer(id: modelID)
            try await container.perform { context in
                try await body(context.tokenizer, context.configuration)
            }
        }
    }

#endif
