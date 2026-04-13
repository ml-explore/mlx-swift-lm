// Copyright © 2026 Apple Inc.

import Foundation
import HuggingFace
import IntegrationTestHelpers
import MLXHuggingFace
import MLXLMCommon
import Testing
import Tokenizers

private let models = IntegrationTestModels(
    downloader: #hubDownloader(),
    tokenizerLoader: #huggingFaceTokenizerLoader()
)

@Suite(.serialized)
struct ToolCallIntegrationTests {

    // MARK: - LFM2

    @Test func lfm2FormatAutoDetection() async throws {
        let container = try await models.lfm2Container()
        try await ToolCallTests.lfm2FormatAutoDetection(container: container)
    }

    @Test func lfm2EndToEnd() async throws {
        let container = try await models.lfm2Container()
        try await ToolCallTests.lfm2EndToEndGeneration(container: container)
    }

    // MARK: - GLM4

    @Test func glm4FormatAutoDetection() async throws {
        let container = try await models.glm4Container()
        try await ToolCallTests.glm4FormatAutoDetection(container: container)
    }

    @Test func glm4EndToEnd() async throws {
        let container = try await models.glm4Container()
        try await ToolCallTests.glm4EndToEndGeneration(container: container)
    }

    // MARK: - Mistral3

    @Test func mistral3FormatAutoDetection() async throws {
        let container = try await models.mistral3Container()
        try await ToolCallTests.mistral3FormatAutoDetection(container: container)
    }

    @Test func mistral3EndToEnd() async throws {
        let container = try await models.mistral3Container()
        try await ToolCallTests.mistral3EndToEndGeneration(container: container)
    }

    @Test func mistral3MultiTool() async throws {
        let container = try await models.mistral3Container()
        try await ToolCallTests.mistral3MultiToolGeneration(container: container)
    }

    // MARK: - Nemotron

    @Test func nemotronFormatAutoDetection() async throws {
        let container = try await models.nemotronContainer()
        try await ToolCallTests.nemotronFormatAutoDetection(container: container)
    }

    @Test func nemotronEndToEnd() async throws {
        let container = try await models.nemotronContainer()
        try await ToolCallTests.nemotronEndToEndGeneration(container: container)
    }

    @Test func nemotronMultiTool() async throws {
        let container = try await models.nemotronContainer()
        try await ToolCallTests.nemotronMultiToolGeneration(container: container)
    }

    // MARK: - Qwen3.5

    @Test func qwen35FormatAutoDetection() async throws {
        let container = try await models.qwen35Container()
        try await ToolCallTests.qwen35FormatAutoDetection(container: container)
    }

    @Test func qwen35EndToEnd() async throws {
        let container = try await models.qwen35Container()
        try await ToolCallTests.qwen35EndToEndGeneration(container: container)
    }

    @Test func qwen35MultiTool() async throws {
        let container = try await models.qwen35Container()
        try await ToolCallTests.qwen35MultiToolGeneration(container: container)
    }
}
