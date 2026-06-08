// Copyright © 2025 Apple Inc.

#if FoundationModelsIntegration

    import Foundation
    import Testing

    @testable import MLXFoundationModels
    import MLXLMCommon

    @Suite
    struct ModelProfileTests {

        private func context(
            modelType: String,
            modelId: String = "",
            configData: Data? = nil
        ) -> LoadedModelContext {
            LoadedModelContext(
                modelType: modelType, modelId: modelId,
                configData: configData, tokenizer: ByteTokenizer())
        }

        // MARK: - Default init

        @Test func defaultInitIsEmpty() {
            let profile = ModelProfile()
            #expect(profile.reasoningConfig == nil)
            #expect(profile.toolCallFormat == nil)
            #expect(profile.extraEOSTokens.isEmpty)
        }

        @Test func customInitRoundTrips() {
            let reasoning = ReasoningConfig(
                startDelimiter: "<r>", endDelimiter: "</r>", promptStrategy: .none)
            let profile = ModelProfile(
                reasoningConfig: reasoning, toolCallFormat: .json,
                extraEOSTokens: ["<|end|>"])
            #expect(profile.reasoningConfig == reasoning)
            #expect(profile.toolCallFormat == .json)
            #expect(profile.extraEOSTokens == ["<|end|>"])
        }

        // MARK: - inferred(for:)

        @Test func inferredQwen3RoutesReasoning() {
            let profile = ModelProfile.inferred(
                for: context(modelType: "qwen3", modelId: "mlx-community/Qwen3-4B-4bit"))
            #expect(profile.reasoningConfig?.startDelimiter == "<think>")
            #expect(profile.reasoningConfig?.endDelimiter == "</think>")
            #expect(
                profile.reasoningConfig?.promptStrategy
                    == .templateFlag(key: "enable_thinking", defaultOn: true))
            #expect(profile.extraEOSTokens.isEmpty)
        }

        @Test func inferredR1DistillIsAlwaysOn() {
            let profile = ModelProfile.inferred(
                for: context(
                    modelType: "qwen2",
                    modelId: "mlx-community/DeepSeek-R1-Distill-Qwen-7B-4bit"))
            #expect(profile.reasoningConfig?.promptStrategy == .alwaysOn)
            #expect(profile.reasoningConfig?.startDelimiter == "<think>")
        }

        @Test func inferredPlainLlamaHasNoReasoning() {
            let profile = ModelProfile.inferred(
                for: context(
                    modelType: "llama", modelId: "mlx-community/Llama-3.2-3B-Instruct-4bit"))
            #expect(profile.reasoningConfig == nil)
        }

        /// `configData` must be threaded into `ToolCallFormat.infer` — Llama 3
        /// detection keys on `vocab_size`/`rope_scaling` from config.json, not
        /// `model_type` alone.
        @Test func inferredLlama3DetectsToolCallFormatFromConfig() throws {
            let configJSON = #"""
                {"model_type": "llama", "vocab_size": 128256}
                """#
            let configData = Data(configJSON.utf8)
            let profile = ModelProfile.inferred(
                for: context(
                    modelType: "llama",
                    modelId: "mlx-community/Llama-3.2-3B-Instruct-4bit",
                    configData: configData))
            #expect(profile.toolCallFormat == .llama3)
        }

        @Test func inferredAlwaysReturnsEmptyEOS() {
            // Across families, the inferred baseline is the empty set — supply
            // tokens via a customizer.
            let qwen3 = ModelProfile.inferred(
                for: context(modelType: "qwen3", modelId: "qwen3-test"))
            let r1 = ModelProfile.inferred(
                for: context(modelType: "deepseek_r1", modelId: "r1-test"))
            let llama = ModelProfile.inferred(
                for: context(modelType: "llama", modelId: "llama-test"))
            #expect(qwen3.extraEOSTokens.isEmpty)
            #expect(r1.extraEOSTokens.isEmpty)
            #expect(llama.extraEOSTokens.isEmpty)
        }

        // MARK: - Equatable

        @Test func equatableHonorsAllFields() {
            let base = ModelProfile(
                reasoningConfig: ReasoningConfig(
                    startDelimiter: "<think>", endDelimiter: "</think>",
                    promptStrategy: .alwaysOn),
                toolCallFormat: .json,
                extraEOSTokens: ["<|end|>"])
            let same = base
            var diffStop = base
            diffStop.extraEOSTokens = ["<|done|>"]
            #expect(base == same)
            #expect(base != diffStop)
        }
    }

#endif  // FoundationModelsIntegration
