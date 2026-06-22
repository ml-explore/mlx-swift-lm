// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

    import Testing
    import Foundation
    import MLXLMCommon
    @testable import MLXFoundationModels

    /// Model-free regression test for the stop-token supply path.
    ///
    /// The model-loading regression tests (which load Gemma/Qwen and assert the
    /// stop set `GuidedGenerationLoop` builds) live in the IntegrationTesting
    /// xcodeproj (`StopTokenRegressionIntegrationTests`). This one verifies the
    /// `DevelopmentConfigurationResolver` supply path using a fake tokenizer — no model.
    @Suite(.serialized)
    struct StopTokenRegressionTests {

        /// `DevelopmentConfigurationResolver` carries Gemma 3's `<end_of_turn>` for the
        /// package's examples. Verifies the supply path without exposing a public
        /// token table.
        @Test("DevelopmentConfigurationResolver adds gemma3 <end_of_turn>")
        func developmentResolverCarriesGemmaToken() {
            guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
            let base = ModelConfiguration(
                directory: URL(fileURLWithPath: "/tmp/gemma"),
                extraEOSTokens: [], eosTokenIds: [],
                toolCallFormat: nil, reasoningConfig: nil)
            let descriptor = ModelDescriptor(
                modelType: "gemma3", modelId: "mlx-community/gemma-3-270m-it-4bit",
                configData: nil, tokenizer: ByteTokenizer())
            let resolved = DevelopmentConfigurationResolver().resolve(base, for: descriptor)
            #expect(resolved.extraEOSTokens.contains("<end_of_turn>"))
        }
    }

#endif  // FoundationModelsIntegration && canImport(FoundationModels)
