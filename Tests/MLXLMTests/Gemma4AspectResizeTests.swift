// Copyright © 2026 Apple Inc.

import CoreGraphics
import Foundation
import Testing

@testable import MLXVLM

/// Unit tests for `gemma4VisionAspectPreservingTarget`. Verifies the resize
/// target avoids square distortion, stays within the patch budget, and aligns
/// to `patch_size * pooling_kernel_size`.
struct Gemma4AspectResizeTests {
    private static let patchSize = 16
    private static let poolingKernelSize = 3
    private static let sideMultiple = patchSize * poolingKernelSize
    private static let imageSeqLength = 280
    private static let fallback = CGSize(width: 800, height: 800)
    private static var maxPatches: Int { imageSeqLength * poolingKernelSize * poolingKernelSize }

    private func target(_ width: Double, _ height: Double) -> CGSize {
        gemma4VisionAspectPreservingTarget(
            width: width,
            height: height,
            imageSeqLength: Self.imageSeqLength,
            fallback: Self.fallback
        )
    }

    @Test(
        "Target stays within budget and aligns to pooling stride",
        arguments: [
            (1280.0, 800.0),  // wide dashboard
            (800.0, 1280.0),  // tall page
            (1000.0, 1000.0),  // square
            (1920.0, 1080.0),  // 16:9
            (2560.0, 900.0),  // ultra-wide
        ])
    func targetIsAlignedAndBounded(size: (Double, Double)) {
        let target = target(size.0, size.1)
        let width = Int(target.width)
        let height = Int(target.height)

        #expect(width % Self.sideMultiple == 0)
        #expect(height % Self.sideMultiple == 0)
        #expect(width >= Self.sideMultiple)
        #expect(height >= Self.sideMultiple)

        let patchCount = (width / Self.patchSize) * (height / Self.patchSize)
        #expect(patchCount <= Self.maxPatches)
    }

    @Test("Non-square images are not forced to square")
    func nonSquareImagesAreNotForcedToSquare() {
        let wideTarget = target(1280, 800)
        let wideAspectRatio = Double(wideTarget.width) / Double(wideTarget.height)
        #expect(wideAspectRatio > 1.5)

        let tallTarget = target(800, 1280)
        let tallAspectRatio = Double(tallTarget.width) / Double(tallTarget.height)
        #expect(tallAspectRatio < 0.67)
    }

    @Test("Degenerate extents fall back")
    func degenerateExtentsFallBack() {
        #expect(target(0, 0) == Self.fallback)
        #expect(target(-10, 100) == Self.fallback)
        #expect(target(100, 0) == Self.fallback)
    }
}
