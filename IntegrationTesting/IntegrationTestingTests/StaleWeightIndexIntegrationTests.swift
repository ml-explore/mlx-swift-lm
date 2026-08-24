// Copyright © 2026 Apple Inc.

import Foundation
import HuggingFace
import IntegrationTestHelpers
import MLXHuggingFace
import MLXLMCommon
import MLXVLM
import Testing
import Tokenizers

// MARK: - Stale safetensors index load regression
//
// `mlx-community/Qwen3-VL-4B-Instruct-4bit` ships a single 3.1 GB
// `model.safetensors` but kept the `model.safetensors.index.json` of its
// unquantized source repo: the index names `model-00001-of-00002.safetensors`
// and `model-00002-of-00002.safetensors` (neither of which the repo ships) and
// declares 8.9 GB. Since #408 loading honors the index, so every weight file
// resolved to a path that does not exist and the checkpoint could not be loaded
// at all (#554). `safetensorWeightURLs` now falls back to the files the
// directory actually contains, and this exercises that on the real checkpoint:
// load plus a short greedy generate, because loading the weights is only half
// the claim — they also have to be the right ones.
//
// The revision is pinned: the index/weights mismatch is a property of this
// upload, and a re-quantized revision would silently stop covering the fallback.

private let staleIndexModelID = "mlx-community/Qwen3-VL-4B-Instruct-4bit"
private let staleIndexRevision = "2fd8dacbdb8f1e54b8c005f081ec5bf79c56376b"

@Suite(.serialized)
struct StaleWeightIndexIntegrationTests {
    private let downloader = #hubDownloader()
    private let tokenizerLoader = #huggingFaceTokenizerLoader()

    @Test func qwen3VLWithStaleIndexLoadsAndGenerates() async throws {
        let context = try await VLMModelFactory.shared.load(
            from: downloader,
            using: tokenizerLoader,
            configuration: ModelConfiguration(
                id: staleIndexModelID, revision: staleIndexRevision))

        let input = try await context.processor.prepare(
            input: UserInput(chat: [.user("What is 2+2? Reply with just the number.")]))
        let stream = try generate(
            input: input,
            parameters: GenerateParameters(maxTokens: 16, temperature: 0),
            context: context)

        var text = ""
        for await event in stream {
            if case .chunk(let chunk) = event { text += chunk }
        }

        #expect(
            text.contains("4") || text.lowercased().contains("four"),
            "\(staleIndexModelID) loaded from a stale index but answered: \(text)")
    }
}
