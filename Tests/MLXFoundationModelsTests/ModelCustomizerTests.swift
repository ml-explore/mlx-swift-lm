// Copyright © 2025 Apple Inc.

#if FoundationModelsIntegration

    import Foundation
    import Testing

    @testable import MLXFoundationModels
    import MLXLMCommon

    @Suite
    struct ModelCustomizerTests {

        private func qwen3Context() -> LoadedModelContext {
            LoadedModelContext(
                modelType: "qwen3", modelId: "mlx-community/Qwen3-4B-4bit",
                configData: nil, tokenizer: ByteTokenizer())
        }

        private func r1Context() -> LoadedModelContext {
            LoadedModelContext(
                modelType: "qwen2",
                modelId: "mlx-community/DeepSeek-R1-Distill-Qwen-7B-4bit",
                configData: nil, tokenizer: ByteTokenizer())
        }

        private func llamaContext() -> LoadedModelContext {
            LoadedModelContext(
                modelType: "llama", modelId: "mlx-community/Llama-3.2-3B-Instruct-4bit",
                configData: nil, tokenizer: ByteTokenizer())
        }

        // MARK: - InferringCustomizer parity

        @Test func inferringCustomizerMatchesInferredForReasoningModels() {
            let customizer = InferringCustomizer()
            for ctx in [qwen3Context(), r1Context(), llamaContext()] {
                #expect(customizer.profile(for: ctx) == ModelProfile.inferred(for: ctx))
            }
        }

        @Test func contextInferredMatchesProfileFactory() {
            let ctx = qwen3Context()
            #expect(ctx.inferred == ModelProfile.inferred(for: ctx))
        }

        // MARK: - Override path: infer then patch one field

        @Test func customCustomizerPatchesReasoningDelimiterOnly() {
            struct DelimiterOverrideCustomizer: ModelCustomizer {
                func profile(for context: LoadedModelContext) -> ModelProfile {
                    var profile = context.inferred
                    profile.reasoningConfig?.startDelimiter = "<reason>"
                    return profile
                }
            }

            let ctx = qwen3Context()
            let baseline = ModelProfile.inferred(for: ctx)
            let patched = DelimiterOverrideCustomizer().profile(for: ctx)

            #expect(patched.reasoningConfig?.startDelimiter == "<reason>")
            // The other reasoning fields and the rest of the profile stay at the baseline.
            #expect(patched.reasoningConfig?.endDelimiter == baseline.reasoningConfig?.endDelimiter)
            #expect(
                patched.reasoningConfig?.promptStrategy == baseline.reasoningConfig?.promptStrategy)
            #expect(patched.toolCallFormat == baseline.toolCallFormat)
            #expect(patched.extraEOSTokens == baseline.extraEOSTokens)
        }

        // MARK: - .inferring static-member sugar

        /// Verifies `.inferring` resolves at a call site where the parameter type
        /// is `any ModelCustomizer` — proving the `where Self == InferringCustomizer`
        /// extension is wired correctly.
        @Test func dotInferringResolvesAsExistential() {
            func accept(_ customizer: any ModelCustomizer) -> ModelProfile {
                customizer.profile(for: qwen3Context())
            }
            let viaSugar = accept(.inferring)
            let viaDirect = accept(InferringCustomizer())
            #expect(viaSugar == viaDirect)
        }
    }

#endif  // FoundationModelsIntegration
