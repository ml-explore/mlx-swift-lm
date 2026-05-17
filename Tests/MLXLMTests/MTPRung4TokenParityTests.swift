// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXVLM
import Testing

// MARK: - Helpers

private func toolsFixturesDir(file: String = #filePath) -> URL? {
    var dir = URL(fileURLWithPath: file).deletingLastPathComponent()
    let fs = FileManager.default
    for _ in 0 ..< 10 {
        let candidate = dir.appendingPathComponent("tools/fixtures")
        if fs.fileExists(atPath: candidate.path) {
            return candidate
        }
        let parent = dir.deletingLastPathComponent()
        if parent.path == dir.path { break }
        dir = parent
    }
    return nil
}

private func hfSnapshotDir(modelId: String) -> URL? {
    let home = FileManager.default.homeDirectoryForCurrentUser
    let hub = home.appendingPathComponent(".cache/huggingface/hub")
    let folderName = "models--" + modelId.replacingOccurrences(of: "/", with: "--")
    let snapshots = hub.appendingPathComponent(folderName).appendingPathComponent("snapshots")
    guard
        let entries = try? FileManager.default.contentsOfDirectory(
            at: snapshots, includingPropertiesForKeys: nil)
    else { return nil }
    return entries.first
}

// MARK: - Rung 4 — drafter `draftBlock` greedy parity with Python fixtures
//
// Rung 4 is bit-exact (no tolerance): greedy sampling at temperature=0 is
// argmax which is deterministic. Any divergence indicates a real semantic
// drift between Swift and Python implementations.
//
// These tests load the drafter and call `draftBlock(...)` directly with
// fixture inputs. They do NOT go through the full
// `MTPSpeculativeTokenIterator` — that is exercised by
// `MTPAcceptanceRateTests` once both target and drafter are available.

@Test
func testRung4DraftBlockGreedyMatchesFixture31BCase01Block2() throws {
    try assertDraftBlockMatchesFixture(name: "case_01_block2")
}

@Test
func testRung4DraftBlockGreedyMatchesFixture31BCase02Block4() throws {
    try assertDraftBlockMatchesFixture(name: "case_02_block4")
}

@Test
func testRung4DraftBlockGreedyMatchesFixture31BCase03Block6() throws {
    try assertDraftBlockMatchesFixture(name: "case_03_block6")
}

// MARK: - Implementation

private func assertDraftBlockMatchesFixture(name: String) throws {
    guard let fixturesDir = toolsFixturesDir() else {
        Issue.record("tools/fixtures dir not found; skipping Rung 4 \(name)")
        return
    }
    guard let drafterDir = hfSnapshotDir(modelId: "mlx-community/gemma-4-31B-it-assistant-bf16")
    else {
        Issue.record("31B-assistant-bf16 checkpoint not in HF cache; skipping Rung 4 \(name)")
        return
    }

    let configURL = drafterDir.appendingPathComponent("config.json")
    let cfg = try JSONDecoder().decode(
        Gemma4AssistantConfiguration.self, from: Data(contentsOf: configURL))
    let model = Gemma4AssistantDraftModel(cfg)
    try loadWeights(modelDirectory: drafterDir, model: model)

    let fixtureURL = fixturesDir.appendingPathComponent("drafter_block/\(name).safetensors")
    let arrays = try MLX.loadArrays(url: fixtureURL)
    guard
        let lastToken = arrays["inputs/last_token"],
        let lastHidden = arrays["inputs/last_hidden"],
        let positionIds = arrays["inputs/position_ids"],
        let fullKeys = arrays["inputs/shared_kv/full_attention/keys"],
        let fullValues = arrays["inputs/shared_kv/full_attention/values"],
        let slidingKeys = arrays["inputs/shared_kv/sliding_attention/keys"],
        let slidingValues = arrays["inputs/shared_kv/sliding_attention/values"],
        let expectedDrafted = arrays["outputs/drafted_tokens"]
    else {
        Issue.record("fixture \(name) missing expected tensor keys")
        return
    }

    let sharedKV: [String: (MLXArray, MLXArray)] = [
        "full_attention": (fullKeys, fullValues),
        "sliding_attention": (slidingKeys, slidingValues),
    ]

    // Read blockSize from output shape: drafted_tokens is [1, blockSize - 1].
    let blockSize = expectedDrafted.dim(1) + 1

    let drafted = model.draftBlock(
        lastToken: lastToken,
        lastHidden: lastHidden,
        sharedKV: sharedKV,
        positionIds: positionIds,
        blockSize: blockSize,
        sampler: ArgMaxSampler()
    )

    eval(drafted)
    #expect(drafted.shape == expectedDrafted.shape)

    let swift = drafted.asArray(Int.self)
    let python = expectedDrafted.asArray(Int.self)
    #expect(swift == python, "drafter tokens diverged from Python fixture for \(name)")
}
