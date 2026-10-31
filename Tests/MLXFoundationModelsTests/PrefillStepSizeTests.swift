// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

import Foundation
import FoundationModels
import Testing

@testable import MLXFoundationModels
import MLXLMCommon

@Suite("Prefill step size")
struct PrefillStepSizeTests {
    @Test func makeParametersSetsPrefillStepSize() {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

        let parameters = MLXLanguageModel.Executor.makeParameters(
            maxTokens: 32,
            requestedTemperature: nil,
            samplingConfiguration: nil,
            prefillStepSize: 8)

        #expect(parameters.prefill.stepSize == 8)
    }

    @Test func nilPrefillStepSizeKeepsModelDefault() {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

        let parameters = MLXLanguageModel.Executor.makeParameters(
            maxTokens: 32,
            requestedTemperature: nil,
            samplingConfiguration: nil,
            prefillStepSize: nil)

        #expect(parameters.prefill.stepSize == nil)
    }

    @Test func initStoresPrefillStepSize() {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

        let model = MLXLanguageModel(
            configuration: ModelConfiguration(id: "org/repo"),
            prefillStepSize: 8,
            weightsLocation: { _ in URL(fileURLWithPath: "/tmp") },
            load: stubLoad())

        #expect(model.prefillStepSize == 8)
        #expect(makeStubModel("org/repo").prefillStepSize == nil)
    }
}

#endif
