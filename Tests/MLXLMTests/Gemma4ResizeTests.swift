// Copyright © 2026 Apple Inc.

import CoreGraphics
import Foundation
import Testing

@testable import MLXVLM

/// Unit tests for `Gemma4ProcessorConfiguration.aspectPreservingTargetSize`.
///
/// The resize must produce dimensions whose patch grid pools exactly:
/// both sides divisible by `pooling_kernel_size × patch_size`, and the
/// patch count within `max_soft_tokens × pooling_kernel_size²`. When
/// those invariants hold the pooler's kernel derivation is exact for
/// every aspect ratio and the pooled grid covers the image fully —
/// the bug class where real patches mapped past the one-hot budget
/// (and the bottom of the image was silently dropped) cannot occur.
struct Gemma4ResizeTests {

    /// Defaults: max_soft_tokens 280, patch_size 16, pooling_kernel_size 3.
    private static let config: Gemma4ProcessorConfiguration = {
        let json = #"{"processor_class": "Gemma4Processor"}"#
        return try! JSONDecoder().decode(
            Gemma4ProcessorConfiguration.self, from: Data(json.utf8))
    }()

    private static let sampleSizes: [(Int, Int)] = [
        (800, 800),  // the size that previously truncated the pooled grid
        (768, 768),
        (960, 672),
        (672, 960),
        (1170, 2532),  // portrait screenshot
        (2532, 1170),
        (30, 30),  // tiny
        (8000, 20),  // extreme panorama
        (20, 8000),
        (1, 1),
        (48, 48),
        (4032, 3024),  // camera photo
    ]

    @Test("Both sides are divisible by pooling_kernel_size × patch_size", arguments: sampleSizes)
    func testSideAlignment(width: Int, height: Int) {
        let config = Self.config
        let sideMultiple = config.poolingKernelSize * config.patchSize
        let target = config.aspectPreservingTargetSize(
            for: CGSize(width: width, height: height))

        #expect(Int(target.width) % sideMultiple == 0)
        #expect(Int(target.height) % sideMultiple == 0)
        #expect(Int(target.width) >= sideMultiple)
        #expect(Int(target.height) >= sideMultiple)
    }

    @Test("Patch count stays within max_soft_tokens × kernel²", arguments: sampleSizes)
    func testPatchBudget(width: Int, height: Int) {
        let config = Self.config
        let target = config.aspectPreservingTargetSize(
            for: CGSize(width: width, height: height))
        let patches =
            (Int(target.width) / config.patchSize) * (Int(target.height) / config.patchSize)
        let kernelArea = config.poolingKernelSize * config.poolingKernelSize

        #expect(patches <= config.maxSoftTokens * kernelArea)
        // Side alignment guarantees the pooling kernel divides the grid
        // exactly, so every soft token pools kernel² real patches.
        #expect(patches % kernelArea == 0)
        #expect(
            config.softTokenCount(
                height: Int(target.height), width: Int(target.width)) <= config.maxSoftTokens)
    }

    @Test("Near-square inputs use the full token budget")
    func testFullBudgetAtSquare() {
        let config = Self.config
        let target = config.aspectPreservingTargetSize(for: CGSize(width: 800, height: 800))
        let softTokens = config.softTokenCount(
            height: Int(target.height), width: Int(target.width))

        // 800×800 → 768×768 → 48×48 patches → 256 soft tokens. The exact
        // value matters less than the invariant pair: close to the budget,
        // never over it.
        #expect(softTokens > config.maxSoftTokens / 2)
        #expect(softTokens <= config.maxSoftTokens)
    }

    @Test("Aspect ratio is approximately preserved for ordinary images")
    func testAspectRatioPreserved() {
        let config = Self.config
        let input = CGSize(width: 1170, height: 2532)
        let target = config.aspectPreservingTargetSize(for: input)

        let inputRatio = input.width / input.height
        let targetRatio = target.width / target.height
        // Rounding to 48-pixel multiples bounds the ratio drift.
        #expect(abs(inputRatio - targetRatio) / inputRatio < 0.25)
        #expect(target.height > target.width)
    }

    @Test("Extreme aspect ratios clamp to one pooling cell on the short side")
    func testExtremeAspectClamps() {
        let config = Self.config
        let sideMultiple = config.poolingKernelSize * config.patchSize

        let panorama = config.aspectPreservingTargetSize(for: CGSize(width: 8000, height: 20))
        #expect(Int(panorama.height) == sideMultiple)
        #expect(Int(panorama.width) <= config.maxSoftTokens * sideMultiple)

        let tower = config.aspectPreservingTargetSize(for: CGSize(width: 20, height: 8000))
        #expect(Int(tower.width) == sideMultiple)
        #expect(Int(tower.height) <= config.maxSoftTokens * sideMultiple)
    }
}
