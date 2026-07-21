// Copyright © 2026 Apple Inc.

import Foundation
import HuggingFace
import IntegrationTestHelpers
import MLX
import MLXHuggingFace
import MLXLMCommon
import MLXVLM
import Testing

/// Pinned revision of the `angelsbrood/gemma4-mtp-fixtures` HF dataset for
/// byte-exact fixture parity. Bump when the dataset is regenerated.
private let fixturesRevision = "152a8ea4cd9e58da11b0c4b39542d3ad347fce06"

private func mtpFixturesDirOrSkip(name: String) async -> URL? {
    do {
        return try await downloadDataset(
            repo: "angelsbrood/gemma4-mtp-fixtures",
            revision: fixturesRevision,
            matching: ["drafter_block/*.safetensors"]
        )
    } catch {
        Issue.record(
            "drafter_block fixtures unavailable for Rung 4 \(name) (dataset fetch failed): \(error.localizedDescription)"
        )
        return nil
    }
}

// MARK: - Shared bound drafter
//
// Rung 4 requires the drafter to be paired with a real target so that the
// target-derived constants (input embeddings, embed scale) flow through
// `draftBlock(target:...)` per round. Loading the 31B 8-bit target is
// expensive (~30–60s); the `@Suite(.serialized)` wrapper below ensures
// the three Rung 4 case methods run sequentially, and this module-level
// cache reuses one (drafter, target) pair across all three.
//
// `Gemma4AssistantDraftModel` is deliberately non-Sendable (see the
// design note at Gemma4Assistant.swift:153–155 — Embedding is a
// reference type and cross-domain access must go through
// `MTPDrafterContainer.perform`). `nonisolated(unsafe)` is appropriate
// here because the suite is serialized — only one test runs at a time,
// and the cache becomes read-only after first population.

private struct Rung4BoundDrafter {
    let drafter: Gemma4AssistantDraftModel
    // Hold the target alive so the drafter's borrowed `embed_tokens`
    // reference stays valid for the lifetime of every test in this suite.
    let target: Gemma4
}

nonisolated(unsafe) private var _rung4Cache: Result<Rung4BoundDrafter, Error>?

private func sharedBoundDrafter() async throws -> Rung4BoundDrafter {
    if let cached = _rung4Cache {
        switch cached {
        case .success(let value): return value
        case .failure(let error): throw error
        }
    }
    do {
        let result = try await loadRung4Drafter()
        _rung4Cache = .success(result)
        return result
    } catch {
        _rung4Cache = .failure(error)
        throw error
    }
}

private let rung4DrafterModelId = "mlx-community/gemma-4-31B-it-assistant-bf16"
private let rung4TargetModelId = "mlx-community/gemma-4-31b-it-8bit"

/// Pinned checkpoint revisions matching the weights that were live when the
/// `drafter_block` fixtures were generated. See the Rung 4 note below for
/// why this pin exists.
private let rung4DrafterRevision = "28e92270316e89288579ec59c17939541d9ca433"
private let rung4TargetRevision = "fe92291011fc698452920c0b558b52f790dff711"

/// Shared downloader for the Rung 4 target+drafter pair. Fetches to the
/// local HF cache on first use; subsequent tests and runs reuse the cache.
private let downloader: any Downloader = #hubDownloader()

private func loadRung4Drafter() async throws -> Rung4BoundDrafter {
    let drafterDir = try await downloader.download(
        id: rung4DrafterModelId, revision: rung4DrafterRevision,
        matching: ["*.safetensors", "*.json"],
        useLatest: false, progressHandler: { _ in })
    let targetDir = try await downloader.download(
        id: rung4TargetModelId, revision: rung4TargetRevision,
        matching: ["*.safetensors", "*.json"],
        useLatest: false, progressHandler: { _ in })

    // Drafter — bf16, no quantization.
    let drafterCfg = try JSONDecoder().decode(
        Gemma4AssistantConfiguration.self,
        from: Data(contentsOf: drafterDir.appendingPathComponent("config.json")))
    let drafter = Gemma4AssistantDraftModel(drafterCfg)
    try loadWeights(modelDirectory: drafterDir, model: drafter)

    // Target — 8-bit quantized. `loadWeights` auto-applies group quantization
    // when weights carry `.scales` keys, per Libraries/MLXLMCommon/Load.swift:40-52.
    let targetConfigData = try Data(
        contentsOf: targetDir.appendingPathComponent("config.json"))
    let targetBase = try JSONDecoder().decode(
        BaseConfiguration.self, from: targetConfigData)
    let targetCfg = try JSONDecoder().decode(
        Gemma4Configuration.self, from: targetConfigData)
    let target = Gemma4(targetCfg)
    try loadWeights(
        modelDirectory: targetDir,
        model: target,
        perLayerQuantization: targetBase.perLayerQuantization)

    return Rung4BoundDrafter(drafter: drafter, target: target)
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
//
// `case_02_block4` and `case_03_block6` previously failed against
// `fixturesRevision`: the published `mlx-community/gemma-4-31b-it-8bit` and
// `.../gemma-4-31B-it-assistant-bf16` checkpoints had moved past the weights
// that were live when these fixtures were captured, so the "expected"
// `outputs/drafted_tokens` no longer matched a live re-run of the pinned
// `mlx-vlm` reference against the same fixture inputs (same staleness class
// previously fixed for `drafter_forward` in commit `7e2f2b8`, "Stabilize
// integration test skip semantics; bump fixture dataset revision"). Fixed
// here by pinning `rung4TargetRevision`/`rung4DrafterRevision` to the exact
// commits live at fixture-generation time (confirmed via HF's per-commit
// LFS-OID listing that the weight blobs are unchanged since), rather than
// regenerating the fixtures.

@Suite(.serialized)
struct Rung4TokenParityTests {
    @Test
    func testRung4DraftBlockGreedyMatchesFixture31BCase01Block2() async throws {
        try await assertDraftBlockMatchesFixture(name: "case_01_block2")
    }

    @Test
    func testRung4DraftBlockGreedyMatchesFixture31BCase02Block4() async throws {
        try await assertDraftBlockMatchesFixture(name: "case_02_block4")
    }

    @Test
    func testRung4DraftBlockGreedyMatchesFixture31BCase03Block6() async throws {
        try await assertDraftBlockMatchesFixture(name: "case_03_block6")
    }
}

// MARK: - Implementation

private func assertDraftBlockMatchesFixture(name: String) async throws {
    guard let fixturesDir = await mtpFixturesDirOrSkip(name: name) else {
        return
    }
    let bound = try await sharedBoundDrafter()
    let model = bound.drafter
    let target = bound.target

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

    // Fixture stores position_ids as an MLXArray for parity with the Python
    // tooling; the Swift API now takes a Swift Int. Convert here (one-time
    // at test setup, not in a hot path).
    let queryOffset = Int(positionIds[0, 0].item(Int32.self))

    let drafted = model.draftBlock(
        target: target,
        lastToken: lastToken,
        lastHidden: lastHidden,
        sharedKV: sharedKV,
        queryOffset: queryOffset,
        blockSize: blockSize,
        sampler: ArgMaxSampler()
    )

    eval(drafted)
    #expect(drafted.shape == expectedDrafted.shape)

    let swift = drafted.asArray(Int.self)
    let python = expectedDrafted.asArray(Int.self)
    #expect(swift == python, "drafter tokens diverged from Python fixture for \(name)")
}
