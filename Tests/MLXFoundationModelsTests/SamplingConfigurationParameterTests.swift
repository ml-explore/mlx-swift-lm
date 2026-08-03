// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

import FoundationModels
import Testing

@testable import MLXFoundationModels
import MLXLMCommon

@Suite("Sampling configuration parameter construction")
struct SamplingConfigurationParameterTests {
    @Test func seededTopKPreservesSeedAndFilter() {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

        let parameters = MLXLanguageModel.Executor.makeParameters(
            maxTokens: 32,
            requestedTemperature: nil,
            samplingConfiguration: MLXSamplingConfiguration(
                mode: .topK(40), seed: UInt64.max))

        #expect(parameters.seed == UInt64.max)
        #expect(parameters.topK == 40)
    }

    @Test func seededNucleusPreservesSeedAndFilter() {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

        let parameters = MLXLanguageModel.Executor.makeParameters(
            maxTokens: 32,
            requestedTemperature: 0.7,
            samplingConfiguration: MLXSamplingConfiguration(
                mode: .nucleus(0.9), seed: 7))

        #expect(parameters.seed == 7)
        #expect(parameters.topP == Float(0.9))
        #expect(parameters.temperature == Float(0.7))
    }

    @Test func nilConfigurationLeavesSeedAndSamplingDefaultsUnchanged() {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

        let parameters = MLXLanguageModel.Executor.makeParameters(
            maxTokens: 32,
            requestedTemperature: nil,
            samplingConfiguration: nil)

        #expect(parameters.seed == nil)
        #expect(parameters.temperature == 0.6)
        #expect(parameters.topP == 1.0)
        #expect(parameters.topK == 0)
    }

    @Test func explicitZeroTemperatureKeepsSeedButSelectsArgmax() {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

        let parameters = MLXLanguageModel.Executor.makeParameters(
            maxTokens: 32,
            requestedTemperature: 0,
            samplingConfiguration: MLXSamplingConfiguration(
                mode: .topK(40), seed: 7))

        #expect(parameters.seed == 7)
        #expect(parameters.sampler() is ArgMaxSampler)
    }
}

#endif
