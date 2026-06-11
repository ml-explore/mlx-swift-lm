// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

    import Foundation
    import FoundationModels
    import Testing

    @testable import MLXFoundationModels
    import MLXLMCommon

    /// SDK → bridge-local translation of `GenerationOptions.SamplingMode`.
    ///
    /// This is the one piece the host suite (`MLXLMTests`) cannot cover: it needs
    /// `import FoundationModels` to construct `SamplingMode` values, and `.kind` is
    /// `@available 27`. The mapping *policy* is host-tested in
    /// `SamplingModeMapperTests`; this suite only checks the case translation and
    /// the `seed` drop. It loads no model, so it stays in the package test target;
    /// the on-device behavioral check (`SamplingModeBehaviorTests`) lives in the
    /// IntegrationTesting xcodeproj. Bodies are `guard #available`-gated (Swift
    /// Testing rejects `@available` on `@Suite`/`@Test`), so they no-op below OS 27.
    @Suite("SamplingMode shim translation")
    struct SamplingModeShimTests {

        @Test func nilMapsToNil() {
            guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
            #expect(MLXLanguageModel.Executor.samplingMode(from: nil) == nil)
        }

        @Test func greedyTranslates() {
            guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
            #expect(MLXLanguageModel.Executor.samplingMode(from: .greedy) == .greedy)
        }

        @Test func topKTranslates() {
            guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
            #expect(MLXLanguageModel.Executor.samplingMode(from: .random(top: 40)) == .topK(40))
        }

        @Test func nucleusTranslates() {
            guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
            #expect(
                MLXLanguageModel.Executor.samplingMode(from: .random(probabilityThreshold: 0.9))
                    == .nucleus(0.9))
        }

        /// `seed` is dropped at the shim (MLX exposes no seed-injection hook):
        /// a seeded mode must translate identically to its unseeded form.
        @Test func seedIsDropped() {
            guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
            #expect(
                MLXLanguageModel.Executor.samplingMode(from: .random(top: 40, seed: 7)) == .topK(40)
            )
            #expect(
                MLXLanguageModel.Executor.samplingMode(
                    from: .random(probabilityThreshold: 0.9, seed: 7)) == .nucleus(0.9))
        }

        // A future/unknown `SamplingMode.Kind` cannot be constructed today, so the
        // `@unknown default -> nil` arm is covered by construction, not asserted.
    }

#endif  // FoundationModelsIntegration && canImport(FoundationModels)
