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
        let container = try await models.llmContainer(for: .init(id: IntegrationTestModelIDs.lfm2))
        try await ToolCallTests.lfm2FormatAutoDetection(container: container)
    }

    @Test func lfm2EndToEnd() async throws {
        let container = try await models.llmContainer(for: .init(id: IntegrationTestModelIDs.lfm2))
        try await ToolCallTests.lfm2EndToEndGeneration(container: container)
    }

    // MARK: - GLM4

    @Test func glm4FormatAutoDetection() async throws {
        let container = try await models.llmContainer(for: .init(id: IntegrationTestModelIDs.glm4))
        try await ToolCallTests.glm4FormatAutoDetection(container: container)
    }

    // GLM-4-9B-0414 emits a markerless `get_weather\n{json-args}` tool call — its
    // chat template only lists the available tools and defines no <tool_call> /
    // <arg_key> delimiters. The .glm4 parser targets the GLM-4.5/4.7 <arg_key>
    // format, so nothing is detected. Supporting this shape needs a dedicated
    // markerless (function-name + JSON) parser plus schema-anchored streaming
    // detection; tracked as a follow-up. (glm4FormatAutoDetection still runs.)
    @Test(
        .disabled(
            "GLM-4-9B-0414 emits markerless funcname+JSON tool calls the .glm4 parser can't detect"
        ))
    func glm4EndToEnd() async throws {
        let container = try await models.llmContainer(for: .init(id: IntegrationTestModelIDs.glm4))
        try await ToolCallTests.glm4EndToEndGeneration(container: container)
    }

    // MARK: - Mistral3

    @Test func mistral3FormatAutoDetection() async throws {
        let container = try await models.llmContainer(
            for: .init(id: IntegrationTestModelIDs.mistral3))
        try await ToolCallTests.mistral3FormatAutoDetection(container: container)
    }

    @Test func mistral3EndToEnd() async throws {
        let container = try await models.llmContainer(
            for: .init(id: IntegrationTestModelIDs.mistral3))
        try await ToolCallTests.mistral3EndToEndGeneration(container: container)
    }

    @Test func mistral3MultiTool() async throws {
        let container = try await models.llmContainer(
            for: .init(id: IntegrationTestModelIDs.mistral3))
        try await ToolCallTests.mistral3MultiToolGeneration(container: container)
    }

    // MARK: - Nemotron

    @Test func nemotronFormatAutoDetection() async throws {
        let container = try await models.llmContainer(
            for: .init(id: IntegrationTestModelIDs.nemotron))
        try await ToolCallTests.nemotronFormatAutoDetection(container: container)
    }

    @Test func nemotronEndToEnd() async throws {
        let container = try await models.llmContainer(
            for: .init(id: IntegrationTestModelIDs.nemotron))
        try await ToolCallTests.nemotronEndToEndGeneration(container: container)
    }

    @Test func nemotronMultiTool() async throws {
        let container = try await models.llmContainer(
            for: .init(id: IntegrationTestModelIDs.nemotron))
        try await ToolCallTests.nemotronMultiToolGeneration(container: container)
    }

    // MARK: - Qwen3.5

    @Test func qwen35FormatAutoDetection() async throws {
        let container = try await models.llmContainer(
            for: .init(id: IntegrationTestModelIDs.qwen35))
        try await ToolCallTests.qwen35FormatAutoDetection(container: container)
    }

    @Test func qwen35EndToEnd() async throws {
        let container = try await models.llmContainer(
            for: .init(id: IntegrationTestModelIDs.qwen35))
        try await ToolCallTests.qwen35EndToEndGeneration(container: container)
    }

    @Test func qwen35StructuredToolContinuation() async throws {
        let container = try await models.llmContainer(
            for: .init(id: IntegrationTestModelIDs.qwen35))
        try await ChatSessionTests.structuredToolContinuation(container: container)
    }

    @Test func qwen35MultiTool() async throws {
        let container = try await models.llmContainer(
            for: .init(id: IntegrationTestModelIDs.qwen35))
        try await ToolCallTests.qwen35MultiToolGeneration(container: container)
    }
}
