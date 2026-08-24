// Copyright © 2026 Apple Inc.

import Foundation
import HuggingFace
import IntegrationTestHelpers
import MLXHuggingFace
import MLXLMCommon
import Testing
import Tokenizers

@Suite(.serialized)
struct RerankerModelIntegrationTests {
    private let downloader = #hubDownloader()
    private let tokenizerLoader = #huggingFaceTokenizerLoader()

    @Test func bgeV2M3ReferenceScores() async throws {
        try await RerankerIntegrationTests.bgeV2M3(
            downloader: downloader, tokenizerLoader: tokenizerLoader)
    }

    @Test func qwen3ReferenceScores() async throws {
        try await RerankerIntegrationTests.qwen3(
            downloader: downloader, tokenizerLoader: tokenizerLoader)
    }

    @Test func jinaV3RelevanceScores() async throws {
        try await RerankerIntegrationTests.jinaV3(
            downloader: downloader, tokenizerLoader: tokenizerLoader)
    }

    /// The upstream checkpoint the MLX repo was converted from, which packages the same model
    /// with no index and the projector inside `model.safetensors`.
    @Test func jinaV3SourceCheckpointRelevanceScores() async throws {
        try await RerankerIntegrationTests.jinaV3SourceCheckpoint(
            downloader: downloader, tokenizerLoader: tokenizerLoader)
    }
}
