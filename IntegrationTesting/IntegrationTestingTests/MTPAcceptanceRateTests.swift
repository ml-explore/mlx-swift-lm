// Copyright © 2026 Apple Inc.

import Foundation
import IntegrationTestHelpers
import MLX
import MLXLMCommon
import MLXVLM
import Testing

// MARK: - Acceptance-rate floor

/// PLAN §11 / R12 sets a loose floor of `>= 0.30` for the 64-token,
/// blockSize=4, temperature=0 production-correctness gate. mlx-vlm reports
/// a 3.94x speedup on the same case which implies acceptance closer to
/// 0.7-0.8; the floor leaves headroom for hardware variance and
/// non-determinism in SDPA kernel ordering.
///
/// Tests are gated by the presence of both target and drafter checkpoints
/// in the HF cache.

@Test(
    .disabled(
        """
        Full target+drafter end-to-end measurement (64 tokens, blockSize=4, \
        temperature=0, floor ≥ 0.30 per PLAN §11/R12) is deferred to a follow-up \
        PR. The Rung 4 token-parity tests in this target cover the drafter's \
        `draftBlock(...)` directly; this test will be wired up once the \
        target+iterator loop is exercisable here.
        """
    )
)
func testAcceptanceRateFloor64TokenBlock4Temp0() async throws {
    // Body retained for future implementation. The `.disabled` trait above
    // causes Swift Testing to skip without recording an issue.
    guard hfSnapshotDir(modelId: "mlx-community/gemma-4-31B-it-assistant-bf16") != nil,
        hfSnapshotDir(modelId: "mlx-community/gemma-4-31b-it-8bit") != nil
    else {
        return
    }
}
