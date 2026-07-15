// Copyright © 2026 Apple Inc.

import Foundation
import HuggingFace
import IntegrationTestHelpers
import MLXHuggingFace
import MLXLMCommon
import MLXVLM
import Testing
import Tokenizers

// MARK: - E-series (KV-sharing) target load regression
//
// The Gemma 4 "effective" targets (E2B/E4B) use num_kv_shared_layers > 0: the
// last N decoder layers reuse an earlier layer's K/V and ship no k_proj/v_proj.
// They carry vision_config so they load through VLMModelFactory. Before the
// loader fix, the VLM Gemma4 backbone declared k_proj on the shared tail and
// loadWeights failed `keyNotFound layers.{15,24}.self_attn.k_proj.weight`
// (independent of MTP — plain load). This exercises the real cached checkpoint
// end to end: load + a short greedy generate (which also drives the runtime
// shared-KV forward on those layers).
//
// Cache-gated: each test enables only when its checkpoint is already in the HF
// cache, so it runs locally where the weights exist and disables (does not fail)
// in CI without them — no env var, no multi-GB auto-download.

private let e4bTarget = "mlx-community/gemma-4-e4b-it-4bit"
private let e2bTarget = "mlx-community/gemma-4-e2b-it-4bit"

@Suite(.serialized)
struct Gemma4EseriesLoadIntegrationTests {

    @Test(.enabled(if: hfSnapshotDir(modelId: e4bTarget) != nil))
    func testE4BTargetLoadsAndGenerates() async throws {
        try await loadAndGenerate(modelId: e4bTarget)
    }

    @Test(.enabled(if: hfSnapshotDir(modelId: e2bTarget) != nil))
    func testE2BTargetLoadsAndGenerates() async throws {
        try await loadAndGenerate(modelId: e2bTarget)
    }

    private func loadAndGenerate(modelId: String) async throws {
        let dir = try #require(hfSnapshotDir(modelId: modelId))

        // The load itself is the regression surface — pre-fix this threw
        // keyNotFound on the KV-shared tail layers.
        let context = try await VLMModelFactory.shared.load(
            from: dir, using: #huggingFaceTokenizerLoader())

        let input = try await context.processor.prepare(
            input: UserInput(chat: [.user("Why is the sky blue? One sentence.")]))
        let stream = try generate(
            input: input,
            parameters: GenerateParameters(maxTokens: 16, temperature: 0),
            context: context)

        var text = ""
        for await event in stream {
            if case .chunk(let c) = event { text += c }
        }
        #expect(!text.isEmpty, "\(modelId) loaded but produced no output")
    }
}
