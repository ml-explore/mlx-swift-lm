// Copyright © 2026 Apple Inc.

import Foundation
import HuggingFace
import IntegrationTestHelpers
import MLXHuggingFace
import MLXLMCommon
import Testing
import Tokenizers

private let qwenReasoningDownloader: any Downloader = #hubDownloader()
private let qwenReasoningTokenizerLoader: any TokenizerLoader = #huggingFaceTokenizerLoader()

@Suite(.serialized)
struct QwenReasoningTemplateIntegrationTests {
    enum Template: Sendable, CaseIterable {
        case qwen36, qwen38

        var modelID: String {
            switch self {
            case .qwen36: "mlx-community/Qwen3.6-27B-4bit"
            case .qwen38: "mlx-community/Qwen3.8-27B-4bit"
            }
        }

        var revision: String {
            switch self {
            case .qwen36: "c000ac2c2057d94be3fa931000c31723aac53282"
            case .qwen38: "10c35caafbb80f7dc6a7a432cdd11af10a6d4818"
            }
        }

        var supportsInlineReasoning: Bool { self == .qwen36 }
    }

    private func tokenizer(for template: Template) async throws -> any MLXLMCommon.Tokenizer {
        let directory = try await qwenReasoningDownloader.download(
            id: template.modelID, revision: template.revision,
            matching: ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja"],
            useLatest: false, progressHandler: { _ in })
        return try await qwenReasoningTokenizerLoader.load(from: directory)
    }

    // Qwen 3.8 needs reasoning_content; Qwen 3.6 also extracts reasoning from content.
    // Check the fix against both pinned templates to preserve Qwen 3.6's rendered tokens.
    @Test(arguments: Template.allCases)
    func importedHistoryUsesTheProductionTemplate(template: Template) async throws {
        try await QwenReasoningTemplateTests.importedHistory(
            tokenizer: tokenizer(for: template),
            supportsInlineReasoning: template.supportsInlineReasoning)
    }

    @Test(arguments: QwenReasoningTemplateTests.Continuation.allCases)
    func liveHybridSessionUsesOnlyMatchingTokens(
        variant: QwenReasoningTemplateTests.Continuation
    ) async throws {
        try await QwenReasoningTemplateTests.liveContinuation(
            tokenizer: tokenizer(for: .qwen38), variant: variant)
    }
}
