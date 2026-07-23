// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

import FoundationModels
import Testing

@testable import MLXFoundationModels

/// SDK → bridge-local translation of `GenerationOptions.SamplingMode`.
///
/// This suite loads no model. It verifies that the adapter preserves the SDK's
/// mode and optional `UInt64` seed as one internal value.
@Suite("SamplingMode shim translation")
struct SamplingModeShimTests {
    @Test func nilMapsToNil() {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        #expect(MLXLanguageModel.Executor.samplingConfiguration(from: nil) == nil)
    }

    @Test func greedyTranslatesWithoutSeed() {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        #expect(
            MLXLanguageModel.Executor.samplingConfiguration(from: .greedy)
                == MLXSamplingConfiguration(mode: .greedy, seed: nil))
    }

    @Test func unseededTopKTranslates() {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        #expect(
            MLXLanguageModel.Executor.samplingConfiguration(from: .random(top: 40))
                == MLXSamplingConfiguration(mode: .topK(40), seed: nil))
    }

    @Test func unseededNucleusTranslates() {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        #expect(
            MLXLanguageModel.Executor.samplingConfiguration(
                from: .random(probabilityThreshold: 0.9))
                == MLXSamplingConfiguration(mode: .nucleus(0.9), seed: nil))
    }

    @Test func topKPreservesUInt64MaxSeed() {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        #expect(
            MLXLanguageModel.Executor.samplingConfiguration(
                from: .random(top: 40, seed: UInt64.max))
                == MLXSamplingConfiguration(mode: .topK(40), seed: UInt64.max))
    }

    @Test func nucleusPreservesSeed() {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        #expect(
            MLXLanguageModel.Executor.samplingConfiguration(
                from: .random(probabilityThreshold: 0.9, seed: 7))
                == MLXSamplingConfiguration(mode: .nucleus(0.9), seed: 7))
    }

    // A future/unknown `SamplingMode.Kind` cannot be constructed today, so the
    // `@unknown default -> nil` arm is covered by construction, not asserted.
}

#endif
