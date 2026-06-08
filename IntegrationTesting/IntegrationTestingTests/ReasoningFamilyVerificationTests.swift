// Copyright © 2025 Apple Inc.

#if FoundationModelsIntegration

    import Foundation
    import FoundationModels
    import MLX
    import Testing

    @testable import MLXFoundationModels
    import MLXLMCommon

    /// On-device family characterization. Empirically confirms the facts that
    /// cannot be known offline: do Qwen3/R1 rendered prompts prefill the opening
    /// `<think>`? It dumps the rendered prompt tails (grep `REASONING-DUMP`) for
    /// human judgment and asserts the `primedInside` seeding the production path
    /// relies on.
    ///
    /// Requires a device running iOS 27.0+. The Kimi K2 mechanism (delimiter- vs
    /// field-based) is a separate manual investigation, not automated here.
    @Suite(.serialized, .timeLimit(.minutes(15)))
    struct ReasoningFamilyVerificationTests {

        static let qwen3 = "mlx-community/Qwen3-1.7B-4bit"
        static let r1Distill = "mlx-community/DeepSeek-R1-Distill-Qwen-1.5B-4bit"
        static let thinkConfig = ReasoningConfig(
            startDelimiter: "<think>", endDelimiter: "</think>",
            promptStrategy: .templateFlag(key: "enable_thinking", defaultOn: true))

        @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
        private func renderedTail(
            modelId: String, additionalContext: [String: any Sendable]?, label: String
        ) async throws -> String {
            let container = try await loadTestModelContainer(id: modelId)
            return try await container.perform { context in
                let input = try await context.processor.prepare(
                    input: UserInput(
                        chat: [.user("What is 17 times 24?")], additionalContext: additionalContext)
                )
                let tokens = input.text.tokens.asArray(Int.self)
                let tail = context.tokenizer.decode(tokenIds: Array(tokens.suffix(48)))
                print("REASONING-DUMP [\(label)] tail=<<<\(tail)>>>")
                return tail
            }
        }

        /// EMPIRICAL (on-device 2026-06-01): Qwen3-1.7B does NOT prefill `<think>`.
        /// - thinking-on  → prompt ends `<|im_start|>assistant\n` (no marker); the
        ///   model emits `<think>` itself in the stream, so `primedInside` is
        ///   correctly false and the non-primed emitter opens on the stream marker.
        /// - thinking-off → the template injects an empty *closed* `<think>\n\n</think>`
        ///   as the "don't think" signal; `primedInside` must be false (the detection
        ///   must not false-positive on the closed empty block).
        /// (Contrast R1-Distill, which DOES prefill an open `<think>` — see
        /// `r1DistillPromptTail`. The production emitter handles both, which is why
        /// `qwen3RoutesReasoningWithoutLeak` passes despite no prefill.)
        @Test func qwen3DoesNotPrefillThinkBlock() async throws {
            guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
            let onTail = try await renderedTail(
                modelId: Self.qwen3, additionalContext: ["enable_thinking": true],
                label: "qwen3-thinking-on")
            let offTail = try await renderedTail(
                modelId: Self.qwen3, additionalContext: ["enable_thinking": false],
                label: "qwen3-thinking-off")
            #expect(
                !ReasoningEventEmitter.promptEndsInsideReasoning(
                    renderedPromptTail: onTail, config: Self.thinkConfig),
                "Qwen3 thinking-on does not prefill; the model emits <think> in-stream")
            #expect(
                !ReasoningEventEmitter.promptEndsInsideReasoning(
                    renderedPromptTail: offTail, config: Self.thinkConfig),
                "Qwen3 thinking-off injects a CLOSED empty block; must not be mis-primed")
        }

        /// Prefill check for R1-Distill (always-on, no enable_thinking knob). The dump informs
        /// the registry/infer decision; we assert only that the path is exercised.
        @Test func r1DistillPromptTail() async throws {
            guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
            let tail = try await renderedTail(
                modelId: Self.r1Distill, additionalContext: nil, label: "r1-distill")
            let primed = ReasoningEventEmitter.promptEndsInsideReasoning(
                renderedPromptTail: tail, config: Self.thinkConfig)
            print("REASONING-DUMP [r1-distill primedInside]=\(primed)")
            #expect(!tail.isEmpty)
        }

        // MARK: - Default-customizer parity (convenience init == inferred path)

        /// The convenience init wires in `InferringCustomizer`, which returns
        /// `ModelProfile.inferred(for:)` unchanged. That factory calls the same
        /// `ReasoningConfig.infer(from:modelId:configData:)` the registry resolves
        /// through, so behavioral parity is structural — but we pin it explicitly here.
        @Test func qwen3DefaultProfileMatchesInferredReasoning() async throws {
            guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
            let container = try await loadTestModelContainer(id: Self.qwen3)
            await container.perform { context in
                let configData = try? Data(
                    contentsOf:
                        testWeightsLocation(modelIdentifier: Self.qwen3).appendingPathComponent(
                            "config.json"))
                let modelType =
                    configData.flatMap {
                        try? JSONDecoder.json5().decode(BaseConfiguration.self, from: $0).modelType
                    } ?? ""
                let loaded = LoadedModelContext(
                    modelType: modelType, modelId: Self.qwen3,
                    configData: configData, tokenizer: context.tokenizer)
                let profile = InferringCustomizer().profile(for: loaded)
                let inferred = ReasoningConfig.infer(
                    from: modelType, modelId: Self.qwen3, configData: configData)
                #expect(profile.reasoningConfig == inferred)
                #expect(profile.reasoningConfig?.startDelimiter == "<think>")
                #expect(
                    profile.reasoningConfig?.promptStrategy
                        == .templateFlag(key: "enable_thinking", defaultOn: true))
            }
        }

        @Test func r1DistillDefaultProfileMatchesInferredReasoning() async throws {
            guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
            let container = try await loadTestModelContainer(id: Self.r1Distill)
            await container.perform { context in
                let configData = try? Data(
                    contentsOf:
                        testWeightsLocation(modelIdentifier: Self.r1Distill).appendingPathComponent(
                            "config.json"))
                let modelType =
                    configData.flatMap {
                        try? JSONDecoder.json5().decode(BaseConfiguration.self, from: $0).modelType
                    } ?? ""
                let loaded = LoadedModelContext(
                    modelType: modelType, modelId: Self.r1Distill,
                    configData: configData, tokenizer: context.tokenizer)
                let profile = InferringCustomizer().profile(for: loaded)
                let inferred = ReasoningConfig.infer(
                    from: modelType, modelId: Self.r1Distill, configData: configData)
                #expect(profile.reasoningConfig == inferred)
                #expect(profile.reasoningConfig?.promptStrategy == .alwaysOn)
            }
        }
    }

#endif  // FoundationModelsIntegration
