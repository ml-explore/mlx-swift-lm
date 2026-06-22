// Copyright © 2025 Apple Inc.

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

    import Foundation
    import Testing

    @testable import MLXFoundationModels
    import MLXLMCommon

    @Suite
    struct ModelConfigurationResolverTests {

        // MARK: - ModelDescriptor

        @Test func descriptorStoresInspectionInputs() {
            let configData = Data(#"{"model_type":"qwen3"}"#.utf8)
            let descriptor = ModelDescriptor(
                modelType: "qwen3",
                modelId: "mlx-community/Qwen3-4B-4bit",
                configData: configData,
                tokenizer: ByteTokenizer())
            #expect(descriptor.modelType == "qwen3")
            #expect(descriptor.modelId == "mlx-community/Qwen3-4B-4bit")
            #expect(descriptor.configData == configData)
        }

        // MARK: - DefaultConfigurationResolver pass-through

        private func descriptor(
            modelType: String = "qwen3",
            modelId: String = "mlx-community/Qwen3-4B-4bit"
        ) -> ModelDescriptor {
            ModelDescriptor(
                modelType: modelType, modelId: modelId,
                configData: nil, tokenizer: ByteTokenizer())
        }

        private func sampleConfiguration() -> ModelConfiguration {
            ModelConfiguration(
                directory: URL(fileURLWithPath: "/tmp/model"),
                extraEOSTokens: ["<|im_end|>"],
                eosTokenIds: [151645],
                toolCallFormat: .json,
                reasoningConfig: ReasoningConfig(
                    startDelimiter: "<think>", endDelimiter: "</think>",
                    promptStrategy: .alwaysOn))
        }

        @Test func defaultResolverReturnsConfigurationUnchanged() {
            let config = sampleConfiguration()
            let resolved = DefaultConfigurationResolver().resolve(config, for: descriptor())
            #expect(resolved.reasoningConfig == config.reasoningConfig)
            #expect(resolved.toolCallFormat == config.toolCallFormat)
            #expect(resolved.extraEOSTokens == config.extraEOSTokens)
            #expect(resolved.eosTokenIds == config.eosTokenIds)
        }

        /// `.default` must resolve where the parameter type is `any
        /// ModelConfigurationResolver`, proving the `where Self ==
        /// DefaultConfigurationResolver` extension is wired correctly.
        @Test func dotDefaultResolvesAsExistential() {
            func accept(_ resolver: any ModelConfigurationResolver) -> ModelConfiguration {
                resolver.resolve(sampleConfiguration(), for: descriptor())
            }
            let viaSugar = accept(.default)
            let viaDirect = accept(DefaultConfigurationResolver())
            #expect(viaSugar.reasoningConfig == viaDirect.reasoningConfig)
        }

        /// A custom resolver patches one field and leaves the rest at baseline.
        @Test func customResolverPatchesReasoningDelimiterOnly() {
            struct DelimiterResolver: ModelConfigurationResolver {
                func resolve(
                    _ configuration: ModelConfiguration, for descriptor: ModelDescriptor
                ) -> ModelConfiguration {
                    var c = configuration
                    c.reasoningConfig?.startDelimiter = "<reason>"
                    return c
                }
            }
            let baseline = sampleConfiguration()
            let patched = DelimiterResolver().resolve(baseline, for: descriptor())
            #expect(patched.reasoningConfig?.startDelimiter == "<reason>")
            #expect(patched.reasoningConfig?.endDelimiter == baseline.reasoningConfig?.endDelimiter)
            #expect(patched.toolCallFormat == baseline.toolCallFormat)
            #expect(patched.extraEOSTokens == baseline.extraEOSTokens)
        }

        // MARK: - DevelopmentConfigurationResolver

        @Test func developmentResolverUnionsGemma3StopToken() {
            let base = ModelConfiguration(
                directory: URL(fileURLWithPath: "/tmp/model"),
                extraEOSTokens: [],
                eosTokenIds: [1],
                toolCallFormat: nil,
                reasoningConfig: nil)
            let resolved = DevelopmentConfigurationResolver().resolve(
                base,
                for: descriptor(modelType: "gemma3", modelId: "mlx-community/gemma-3-270m-it-4bit"))
            #expect(resolved.extraEOSTokens.contains("<end_of_turn>"))
            // Existing fields are preserved (additive union, not replace).
            #expect(resolved.eosTokenIds == [1])
        }

        @Test func developmentResolverNoOpForUnknownModelType() {
            let base = ModelConfiguration(
                directory: URL(fileURLWithPath: "/tmp/model"),
                extraEOSTokens: ["<existing>"],
                eosTokenIds: [], toolCallFormat: nil, reasoningConfig: nil)
            let resolved = DevelopmentConfigurationResolver().resolve(
                base, for: descriptor(modelType: "llama", modelId: "llama-test"))
            #expect(resolved.extraEOSTokens == ["<existing>"])
        }

        @Test func developmentResolverUnionsPhi3StopToken() {
            let base = ModelConfiguration(
                directory: URL(fileURLWithPath: "/tmp/model"),
                extraEOSTokens: [],
                eosTokenIds: [2],
                toolCallFormat: nil,
                reasoningConfig: nil)
            let resolved = DevelopmentConfigurationResolver().resolve(
                base,
                for: descriptor(
                    modelType: "phi3", modelId: "mlx-community/Phi-3-mini-4k-instruct-4bit"))
            #expect(resolved.extraEOSTokens.contains("<|end|>"))
            #expect(resolved.eosTokenIds == [2])
        }

        @Test func developmentResolverUnionsOntoExistingTokens() {
            let base = ModelConfiguration(
                directory: URL(fileURLWithPath: "/tmp/model"),
                extraEOSTokens: ["<|im_end|>"],
                eosTokenIds: [],
                toolCallFormat: nil,
                reasoningConfig: nil)
            let resolved = DevelopmentConfigurationResolver().resolve(
                base,
                for: descriptor(
                    modelType: "gemma3", modelId: "mlx-community/gemma-3-270m-it-4bit"))
            // Additive union: the resolver adds its token without clobbering tokens
            // already present in the loaded configuration.
            #expect(resolved.extraEOSTokens == ["<|im_end|>", "<end_of_turn>"])
        }
    }

#endif  // FoundationModelsIntegration && canImport(FoundationModels)
