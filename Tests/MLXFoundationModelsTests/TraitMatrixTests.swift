// Copyright © 2026 Apple Inc.
//
// TraitMatrixTests: symbol-surface + behavioral checks across the
// `FoundationModelsIntegration` trait, the package's only trait.
//
// Each `#if` block below is active for exactly one trait state. Successfully
// compiling this file under a given trait set is the primary structural
// assertion: the test bodies reference the symbols that must be present.
//
// The `FoundationModelsIntegration`-on arm additionally requires
// `canImport(FoundationModels, _version: 2)`: the adapter surface
// (`MLXLanguageModel` et al.) only exists on the 27 SDK, so on the 26 SDK that
// arm compiles to nothing even when the trait is on. Guided generation is no
// longer trait-gated: whenever the adapter exists, the engine is present.

import Testing

#if FoundationModelsIntegration
    @testable import MLXFoundationModels
    import FoundationModels
    import MLXGuidedGeneration
#else
    @testable import MLXFoundationModels
#endif

@Suite("Trait matrix: FoundationModelsIntegration")
struct TraitMatrixTests {

    // MARK: - FoundationModelsIntegration on (default)

    #if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)
        @Test("FM on: MLXLanguageModel + guided-generation primitives compile")
        func fmOnSurface() {
            guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
            _ = MLXLanguageModel.self
            _ = MLXLanguageModel.Executor.self
            _ = GuidedGenerationLoop.self
            _ = GrammarConstraint.self
            _ = MLXDownloadProgress.self
        }

        @Test("FM on: capabilities stored verbatim from init")
        func capabilitiesStoredVerbatim() {
            guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
            // Capabilities are authoritative: the adapter stores what the caller
            // passes, never inferring from the model id.
            let reasoning = makeStubModel(
                "mlx-community/Qwen3-4B-4bit",
                capabilities: [
                    .reasoning, .guidedGeneration, .toolCalling,
                ]
            ).capabilities
            #expect(reasoning.contains(.reasoning))
            #expect(reasoning.contains(.guidedGeneration))
            #expect(reasoning.contains(.toolCalling))

            let nonReasoning = makeStubModel(
                TestFixtures.gemmaModelID,
                capabilities: [
                    .guidedGeneration, .toolCalling,
                ]
            ).capabilities
            #expect(!nonReasoning.contains(.reasoning))
            #expect(nonReasoning.contains(.guidedGeneration))
        }
    #endif

    // MARK: - FoundationModelsIntegration off

    #if !FoundationModelsIntegration
        @Test("FM off: MLXFoundationModels exposes only MLXDownloadProgress")
        func fmOffSurface() {
            _ = MLXDownloadProgress.self
            // No MLXLanguageModel in this configuration; the fact that this file
            // compiles without referencing it is the assertion.
        }
    #endif
}
