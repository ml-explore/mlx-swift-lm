// Copyright © 2026 Apple Inc.

import Foundation
import Testing

@testable import MLXVLM

/// Unit tests for `gemma4VisionPoolingKernel`. Pins the recovered
/// kernel to Gemma 4's `pooling_kernel_size` (3) for every
/// supported soft-token budget.
struct Gemma4VisionPoolingTests {

    /// `pooling_kernel_size` for Gemma 4 (per HF + the model
    /// config's `pooling_kernel_size` field).
    private static let expectedPoolingKernel = 3

    @Test(
        "Kernel matches pooling_kernel_size (3) for every supported budget",
        arguments: [70, 140, 280, 560, 1120])
    func testKernelMatchesPoolingKernelSize(budget: Int) {
        // Per HF: max_patches = max_soft_tokens * pool², so the
        // padded patch tensor fed into the pooler has length
        // budget * pool². The kernel recovered from those values
        // must equal pool.
        let pool = Self.expectedPoolingKernel
        let paddedPatchCount = budget * pool * pool

        let kernel = gemma4VisionPoolingKernel(
            paddedPatchCount: paddedPatchCount, outputLength: budget)

        #expect(
            kernel == pool,
            """
            Expected pooling kernel \(pool) for budget \(budget) \
            (padded patch count \(paddedPatchCount)), got \(kernel). \
            Kernel must match Gemma 4's `pooling_kernel_size` config field for \
            every supported soft-token budget.
            """)
    }

    @Test("Kernel is robust to degenerate inputs")
    func testKernelHandlesDegenerateInputs() {
        // A zero `outputLength` must not divide by zero — the
        // function falls back to treating it as 1, matching the
        // upstream pattern. The exact returned value at degenerate
        // inputs is unspecified; the assertion is "does not crash".
        _ = gemma4VisionPoolingKernel(paddedPatchCount: 0, outputLength: 0)
        _ = gemma4VisionPoolingKernel(paddedPatchCount: 100, outputLength: 0)

        // A zero `paddedPatchCount` with a real `outputLength`
        // yields kernel = 1 (no patches to pool yields the minimum
        // kernel).
        #expect(gemma4VisionPoolingKernel(paddedPatchCount: 0, outputLength: 280) == 1)
    }

    /// Documents the prior bug. The original formula divided the
    /// real (un-padded) patch count by `outputLength` and floored
    /// the square root. For a 768×768 image — the aligned default
    /// for Gemma 4 (divisible by `patch_size × pooling_kernel_size
    /// = 48`) — this returns 2 even though `pooling_kernel_size`
    /// is 3.
    @Test("Pre-fix formula returns kernel=2 at the aligned default size")
    func testPreFixFormulaReproducesBug() {
        // 768×768 image at patch_size=16 → 48×48 = 2304 real patches.
        let realPatchCount = 48 * 48
        let outputLength = 280

        // The old formula, preserved here for documentation.
        let safeLength = max(outputLength, 1)
        let ratio = max(1, realPatchCount / safeLength)
        let oldKernel = Int(sqrt(Double(ratio)))

        #expect(
            oldKernel == 2,
            "Sanity check on the pre-fix formula: should reproduce the bug.")
        #expect(
            oldKernel != Self.expectedPoolingKernel,
            """
            The pre-fix kernel (\(oldKernel)) does not match Gemma 4's \
            documented `pooling_kernel_size` of \(Self.expectedPoolingKernel). \
            This is the bug `gemma4VisionPoolingKernel` was introduced to fix.
            """)
    }
}
