// Copyright © 2025 Apple Inc.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import XCTest

/// Integration tests for tool call format auto-detection with real models.
///
/// These tests verify that tool call formats are correctly auto-detected
/// when loading models from HuggingFace based on their `model_type`.
///
/// References:
/// - LFM2: https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/tool_parsers/default.py
/// - GLM4: https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/tool_parsers/glm47.py
public class ToolCallIntegrationTests: XCTestCase {

    // MARK: - Model IDs

    static let lfm2ModelId = "mlx-community/LFM2-2.6B-Exp-4bit"
    static let glm4ModelId = "mlx-community/GLM-4-9B-0414-4bit"

    // MARK: - LFM2 Format Detection

    func testLFM2ToolCallFormatAutoDetection() async throws {
        // Load LFM2 model and verify format is auto-detected
        let container = try await LLMModelFactory.shared.loadContainer(
            configuration: .init(id: Self.lfm2ModelId)
        )

        let config = await container.configuration
        XCTAssertEqual(
            config.toolCallFormat, .lfm2,
            "LFM2 model should auto-detect .lfm2 tool call format from model_type"
        )
    }

    // MARK: - GLM4 Format Detection

    func testGLM4ToolCallFormatAutoDetection() async throws {
        // Load GLM4 model and verify format is auto-detected
        let container = try await LLMModelFactory.shared.loadContainer(
            configuration: .init(id: Self.glm4ModelId)
        )

        let config = await container.configuration
        XCTAssertEqual(
            config.toolCallFormat, .glm4Moe,
            "GLM4 model should auto-detect .glm4Moe tool call format from model_type"
        )
    }
}
