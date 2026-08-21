// Copyright © 2026 Apple Inc.

import MLXLLM
import XCTest

final class LLMRegistryTests: XCTestCase {

    func testFalconH1RModelConfigurationIsRegistered() {
        XCTAssertTrue(LLMRegistry.shared.contains(id: "tiiuae/Falcon-H1R-7B"))

        let configuration = LLMRegistry.falconH1R7B
        XCTAssertEqual(configuration.name, "tiiuae/Falcon-H1R-7B")
    }

    func testOfficialLFM25ConfigurationUsesScopedVariant() {
        XCTAssertTrue(LLMRegistry.shared.contains(id: "LiquidAI/LFM2.5-2.6B-MLX"))

        let configuration = LLMRegistry.lfm25_2_6b_4bit
        XCTAssertEqual(configuration.modelSubdirectory, "4bit")
        XCTAssertEqual(configuration.toolCallFormat, .lfm2)
        XCTAssertEqual(configuration.reasoningConfig, .alwaysOnThinking)
    }

    func testOfficialLFM25MoEPrecisionsAreRegisteredWithRecommendedSampling() {
        for configuration in [
            LLMRegistry.lfm25_8b_a1b_4bit,
            LLMRegistry.lfm25_8b_a1b_8bit,
        ] {
            XCTAssertTrue(LLMRegistry.shared.contains(id: configuration.name))
            XCTAssertEqual(configuration.toolCallFormat, .lfm2)
            XCTAssertEqual(configuration.reasoningConfig, .alwaysOnThinking)
            XCTAssertEqual(configuration.generationConfig?.temperature, 0.2)
            XCTAssertEqual(configuration.generationConfig?.topK, 80)
            XCTAssertEqual(configuration.generationConfig?.repetitionPenalty, 1.05)
        }
    }
}
