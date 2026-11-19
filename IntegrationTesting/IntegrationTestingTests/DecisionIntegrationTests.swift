// Copyright © 2026 Apple Inc.

import Foundation
import HuggingFace
import IntegrationTestHelpers
import MLXHuggingFace
import MLXLMCommon
import Testing
import Tokenizers

/// Skipped unless `MLX_RUN_DECISION_INTEGRATION=1`: it downloads d1-3B (6 GB). If you run
/// `xcodebuild`, prefix the variable with `TEST_RUNNER_`.
@Suite(
    .serialized,
    .timeLimit(.minutes(20)),
    .enabled(if: ProcessInfo.processInfo.environment["MLX_RUN_DECISION_INTEGRATION"] == "1"))
struct DecisionModelIntegrationTests {
    private let downloader = #hubDownloader()
    private let tokenizerLoader = #huggingFaceTokenizerLoader()

    @Test func d1ReadmeExamples() async throws {
        try await DecisionIntegrationTests.d1(
            downloader: downloader, tokenizerLoader: tokenizerLoader)
    }

    @Test func d1StillChats() async throws {
        try await DecisionIntegrationTests.d1Chat(
            downloader: downloader, tokenizerLoader: tokenizerLoader)
    }
}
