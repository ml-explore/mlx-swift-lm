// Copyright © 2026 Apple Inc.

import Foundation
import IntegrationTestHelpers
import MLX
import MLXLMCommon
import MLXVLM
import Testing

// MARK: - R13 — mid-generation KV cache quantization onset

/// PLAN risk R13: the existing `SpeculativeTokenIterator` quantizes both
/// caches after each round; `MTPSpeculativeTokenIterator` quantizes only
/// the main cache (no drafter cache). Once `maybeQuantizeKVCache` converts
/// `KVCacheSimple` to `QuantizedKVCache`, the target's emit-hook returns
/// `sharedKV: nil` (regular `(MLXArray, MLXArray)` tuples are no longer
/// available). The iterator must transition into single-token
/// "passthrough" mode without crashing — see `MTPSpeculativeTokenIterator`
/// for the fallback implementation.
///
/// R8 is the related "no quantized MTP for this PR" limitation; this test
/// documents the fallback behavior end-to-end.

@Test(
    .disabled(
        """
        Full target+drafter end-to-end exercise (kvBits=4, quantizedKVStart=32, \
        passthrough fallback once cache quantizes mid-generation) is deferred \
        to a follow-up PR. The passthrough fallback logic is unit-tested in \
        `MTPSpeculativeTokenIteratorTests.testMTPIteratorMissingStateFallsBackToPassthrough` \
        (in MLXLMTests) with synthetic state; this test will be wired up once \
        full-target quantization-onset measurement is exercisable here.
        """
    )
)
func testMTPMidGenerationKVQuantizationCompletesWithoutCrash() async throws {
    // Body retained for future implementation. The `.disabled` trait above
    // causes Swift Testing to skip without recording an issue.
    //
    // The semantic to verify: with `kvBits=4, quantizedKVStart=32`, the
    // main cache transitions from `KVCacheSimple` to `QuantizedKVCache`
    // mid-generation. Once the cache quantizes, the target's emit-hook
    // returns `sharedKV: nil` and the iterator's passthrough fallback
    // engages. Generation completes without crashing.
    guard hfSnapshotDir(modelId: "mlx-community/gemma-4-31B-it-assistant-bf16") != nil,
        hfSnapshotDir(modelId: "mlx-community/gemma-4-31b-it-8bit") != nil
    else {
        return
    }
}
