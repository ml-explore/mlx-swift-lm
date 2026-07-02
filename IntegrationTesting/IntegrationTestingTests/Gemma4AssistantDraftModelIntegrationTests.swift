// Copyright © 2026 Apple Inc.

import Foundation
import IntegrationTestHelpers
import MLX
import MLXLMCommon
import MLXNN
@_spi(Testing) import MLXVLM
import Testing

/// Pinned revision of the `angelsbrood/gemma4-mtp-fixtures` HF dataset for
/// byte-exact fixture parity. Bump when the dataset is regenerated.
private let fixturesRevision = "152a8ea4cd9e58da11b0c4b39542d3ad347fce06"

private func drafterForwardFixturesOrSkip(name: String) async -> URL? {
    do {
        return try await downloadDataset(
            repo: "angelsbrood/gemma4-mtp-fixtures",
            revision: fixturesRevision,
            matching: ["drafter_forward/*.safetensors"]
        )
    } catch {
        Issue.record(
            "drafter_forward fixtures unavailable for Rung 2/3 \(name) (dataset fetch failed): \(error.localizedDescription)"
        )
        return nil
    }
}

// MARK: - Gemma4 assistant integration tests
//
// Wrapped in `@Suite(.serialized)` so the three tests below (which each load
// the 31B drafter checkpoint and exercise different forward paths) execute
// sequentially. Parallel weight loads against the same checkpoint can race
// on MLX runtime state and produce non-deterministic numeric divergence in
// the Rung 2/3 forward parity assertion.

@Suite(.serialized)
struct Gemma4AssistantIntegrationTests {

    @Test
    func testGemma4AssistantConfigurationDecodesRealCheckpoint() throws {
        let drafterDir = hfSnapshotDir(modelId: "mlx-community/gemma-4-31B-it-assistant-bf16")
        guard let drafterDir else {
            Issue.record("31B drafter checkpoint not present in HF cache; skipping")
            return
        }
        let configURL = drafterDir.appendingPathComponent("config.json")
        let data = try Data(contentsOf: configURL)
        let cfg = try JSONDecoder().decode(Gemma4AssistantConfiguration.self, from: data)
        #expect(cfg.modelType == "gemma4_assistant")
        #expect(cfg.backboneHiddenSize == 5376)
        #expect(cfg.tieWordEmbeddings == true)
        #expect(cfg.useOrderedEmbeddings == false)
        #expect(cfg.textConfiguration.hiddenLayers == 4)
        #expect(cfg.textConfiguration.hiddenSize == 1024)
        #expect(cfg.textConfiguration.vocabularySize == 262_144)
        #expect(
            cfg.textConfiguration.layerTypes
                == [
                    "sliding_attention", "sliding_attention", "sliding_attention", "full_attention",
                ]
        )
    }

    @Test
    func testRung1WeightsLoadFrom31BCheckpoint() throws {
        guard
            let drafterDir = hfSnapshotDir(modelId: "mlx-community/gemma-4-31B-it-assistant-bf16")
        else {
            Issue.record("31B drafter checkpoint not in HF cache; skipping Rung 1")
            return
        }
        let configURL = drafterDir.appendingPathComponent("config.json")
        let cfg = try JSONDecoder().decode(
            Gemma4AssistantConfiguration.self, from: Data(contentsOf: configURL))
        let model = Gemma4AssistantDraftModel(cfg)
        // loadWeights enumerates *.safetensors in the directory, sanitizes via
        // the model's sanitize hook, and applies via update(parameters:verify:).
        // If any keys are missing or unexpected, update(verify: [.all]) throws.
        try loadWeights(modelDirectory: drafterDir, model: model)
    }

    @Test
    func testRung2And3ForwardMatchesFixture31BCase01() async throws {
        guard let fixturesDir = await drafterForwardFixturesOrSkip(name: "case_01") else {
            return
        }
        guard
            let drafterDir = hfSnapshotDir(modelId: "mlx-community/gemma-4-31B-it-assistant-bf16")
        else {
            Issue.record("31B drafter checkpoint not in HF cache; skipping Rung 2/3")
            return
        }

        let configURL = drafterDir.appendingPathComponent("config.json")
        let cfg = try JSONDecoder().decode(
            Gemma4AssistantConfiguration.self, from: Data(contentsOf: configURL))
        let model = Gemma4AssistantDraftModel(cfg)
        try loadWeights(modelDirectory: drafterDir, model: model)

        let fixtureURL =
            fixturesDir
            .appendingPathComponent("drafter_forward/case_01_q1_kv32_baseline.safetensors")
        let arrays = try MLX.loadArrays(url: fixtureURL)
        guard
            let inputsEmbeds = arrays["inputs/inputs_embeds"],
            let positionIds = arrays["inputs/position_ids"],
            let fullKeys = arrays["inputs/shared_kv/full_attention/keys"],
            let fullValues = arrays["inputs/shared_kv/full_attention/values"],
            let slidingKeys = arrays["inputs/shared_kv/sliding_attention/keys"],
            let slidingValues = arrays["inputs/shared_kv/sliding_attention/values"],
            let expectedLastHidden = arrays["outputs/last_hidden"],
            let expectedLogits = arrays["outputs/logits"]
        else {
            Issue.record("fixture missing expected tensor keys")
            return
        }

        let sharedKV: [String: (MLXArray, MLXArray)] = [
            "full_attention": (fullKeys, fullValues),
            "sliding_attention": (slidingKeys, slidingValues),
        ]
        // Fixture stores position_ids as an MLXArray for parity with the
        // Python tooling; the Swift API now takes a Swift Int. Convert here
        // (one-time at test setup, not in a hot path).
        let queryOffset = Int(positionIds[0, 0].item(Int32.self))
        let (lastHidden, logits) = model.forwardHidden(
            inputsEmbeds: inputsEmbeds,
            sharedKV: sharedKV,
            queryOffset: queryOffset
        )

        // Shape parity.
        #expect(lastHidden.shape == expectedLastHidden.shape)
        #expect(logits.shape == expectedLogits.shape)

        // Finiteness check.
        #expect(!isNaN(lastHidden).any().item(Bool.self), "Swift last_hidden has NaN")
        #expect(!isInf(lastHidden).any().item(Bool.self), "Swift last_hidden has Inf")
        #expect(!isNaN(logits).any().item(Bool.self), "Swift logits has NaN")
        #expect(!isInf(logits).any().item(Bool.self), "Swift logits has Inf")

        // Value parity (Rung 2/3) — bf16 tolerance per PLAN §11 (atol=1e-3, rtol=1e-3).
        // Cast both to float32 to avoid bf16 quirks in allClose.
        let lhSwift = lastHidden.asType(.float32)
        let lhExpected = expectedLastHidden.asType(.float32)
        #expect(
            allClose(lhSwift, lhExpected, rtol: 1e-3, atol: 1e-3).item(Bool.self),
            "last_hidden values diverge from fixture")

        let logitsSwift = logits.asType(.float32)
        let logitsExpected = expectedLogits.asType(.float32)
        #expect(
            allClose(logitsSwift, logitsExpected, rtol: 1e-3, atol: 1e-3).item(Bool.self),
            "logits values diverge from fixture")
    }
}
