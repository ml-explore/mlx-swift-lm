// Copyright © 2026 Apple Inc.

import Foundation
import HuggingFace
import IntegrationTestHelpers
import MLXHuggingFace
import MLXLMCommon
import Testing
import Tokenizers

private let harmonyDownloader: any Downloader = #hubDownloader()
private let harmonyTokenizerLoader: any TokenizerLoader = #huggingFaceTokenizerLoader()

@Suite(.serialized)
struct HarmonyTokenizerIntegrationTests {
    private static let modelID = "mlx-community/gpt-oss-20b-MXFP4-Q8"
    private static let revision = "773a7da77e569019bb0fd17a554b263738d669a3"

    @Test("Production GPT-OSS tokenizer and chat template satisfy Harmony")
    func productionTokenizerContract() async throws {
        let directory = try await harmonyDownloader.download(
            id: Self.modelID,
            revision: Self.revision,
            matching: ["*.json", "*.jinja"],
            useLatest: false,
            progressHandler: { _ in })
        let tokenizer = try await harmonyTokenizerLoader.load(from: directory)

        try HarmonyProtocolTests.realTokenizerContract(tokenizer: tokenizer)
    }
}
