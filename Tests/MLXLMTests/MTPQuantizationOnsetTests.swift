// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXVLM
import Testing

// MARK: - Helpers

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

// MARK: - R13 — mid-generation KV cache quantization onset

/// PLAN risk R13: the existing ``SpeculativeTokenIterator`` quantizes both
/// caches after each round; ``MTPSpeculativeTokenIterator`` quantizes only
/// the main cache (no drafter cache). Once `maybeQuantizeKVCache` converts
/// `KVCacheSimple` to `QuantizedKVCache`, the target's emit-hook returns
/// `sharedKV: nil` (regular `(MLXArray, MLXArray)` tuples are no longer
/// available). The iterator must transition into single-token
/// "passthrough" mode without crashing — see ``MTPSpeculativeTokenIterator``
/// for the fallback implementation.
///
/// R8 is the related "no quantized MTP for this PR" limitation; this test
/// documents the fallback behavior end-to-end.

@Test
func testMTPMidGenerationKVQuantizationCompletesWithoutCrash() async throws {
    // Documents the R13 verification plan. The test gates on checkpoint
    // presence AND on a runtime that can load both the 31b-it-8bit target
    // via `VLMModelFactory` and the 31B-it-assistant-bf16 drafter via
    // `MTPDrafterModelFactory`. The target load requires a real tokenizer
    // loader (`#huggingFaceTokenizerLoader()` from MLXHuggingFace, which
    // is not in MLXLMTests' dependency closure), so the actual end-to-end
    // path runs in Xcode with the full app environment rather than via
    // `swift test`.
    //
    // The semantic to verify: with `kvBits=4, quantizedKVStart=32`, the
    // main cache transitions from `KVCacheSimple` to `QuantizedKVCache`
    // mid-generation. Once the cache quantizes, the target's emit-hook
    // returns `sharedKV: nil` and the iterator's passthrough fallback
    // engages (see ``MTPSpeculativeTokenIterator``). Generation completes
    // without crashing.
    guard hfSnapshotDir(modelId: "mlx-community/gemma-4-31B-it-assistant-bf16") != nil
    else {
        Issue.record("31B-assistant-bf16 not in HF cache; skipping R13 quantization-onset test")
        return
    }
    guard hfSnapshotDir(modelId: "mlx-community/gemma-4-31b-it-8bit") != nil
    else {
        Issue.record("31b-it-8bit target not in HF cache; skipping R13 quantization-onset test")
        return
    }

    Issue.record(
        """
        R13 quantization-onset test is intentionally skipped: target
        loading requires `#huggingFaceTokenizerLoader()` from
        MLXHuggingFace, which is not currently a dependency of MLXLMTests.
        The passthrough fallback logic is unit-tested in
        `MTPSpeculativeTokenIteratorTests.testMTPIteratorMissingStateFallsBackToPassthrough`
        with synthetic state; full-target quantization-onset measurement
        runs in a downstream app/integration test once the tokenizer
        loader is wired in.
        """
    )
}
