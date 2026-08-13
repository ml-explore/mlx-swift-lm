// Copyright © 2026 Apple Inc.

import Foundation
import HuggingFace
import IntegrationTestHelpers
import MLXHuggingFace
import MLXLMCommon
import Testing
import Tokenizers

private let museGlimmerDownloader: any Downloader = #hubDownloader()
private let museGlimmerTokenizerLoader: any TokenizerLoader = #huggingFaceTokenizerLoader()

@Suite(.serialized)
struct MuseGlimmerTokenizerIntegrationTests {
    private static let modelID = "mlx-community/Muse-Glimmer-30B-4bit"
    private static let revision = "3e7677d7a40d348a3daba263a2b1c0aa41910710"

    @Test("Production Muse Glimmer tokenizer and chat template satisfy Onyx ATEM")
    func productionTokenizerContract() async throws {
        let directory = try await museGlimmerDownloader.download(
            id: Self.modelID,
            revision: Self.revision,
            matching: ["*.json", "*.jinja"],
            useLatest: false,
            progressHandler: { _ in })
        let tokenizer = try await museGlimmerTokenizerLoader.load(from: directory)

        try MuseGlimmerProtocolTests.realTokenizerContract(tokenizer: tokenizer)
    }
}
