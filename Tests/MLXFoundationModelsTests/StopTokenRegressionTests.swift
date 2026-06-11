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
    /// `DevelopmentCustomizer` supply path using a fake tokenizer — no model.
    @Suite(.serialized)
    struct StopTokenRegressionTests {

        /// `DevelopmentCustomizer` carries Gemma 3's `<end_of_turn>` for the
        /// package's examples. Verifies the supply path without exposing a public
        /// token table.
        @Test("DevelopmentCustomizer adds gemma3 <end_of_turn>")
        func developmentCustomizerCarriesGemmaToken() {
            guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
            let ctx = LoadedModelContext(
                modelType: "gemma3", modelId: "mlx-community/gemma-3-270m-it-4bit",
                configData: nil, tokenizer: ByteTokenizer())
            let profile = DevelopmentCustomizer().profile(for: ctx)
            #expect(profile.extraEOSTokens.contains("<end_of_turn>"))
        }
    }

#endif  // FoundationModelsIntegration && canImport(FoundationModels)
