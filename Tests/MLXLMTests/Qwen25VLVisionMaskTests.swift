// Copyright © 2026 Apple Inc.
//
// Coverage for the cumulative patch-row boundaries behind the Qwen2.5-VL vision
// attention mask, on grids with a temporal extent (`t > 1`), i.e. video.
//
// The boundaries drive `attentionMask(sequenceLength:cuSeqlens:)`, which marks
// `[start..<end, start..<end]` attendable for each temporal slice. If they are not
// a running total the covered range stops short of the patch buffer, and the
// uncovered rows attend uniformly across every frame instead of within their own —
// the mask bias is a finite `-10000`, not `-inf`, so this corrupts values silently
// rather than surfacing as NaN. Asserting the boundaries directly is therefore the
// reliable check.
//
// `t == 1` (still images) collapses to one block per frame and is unaffected.

import Foundation
import MLXLMCommon
import XCTest

@testable import MLXVLM

final class Qwen25VLVisionMaskTests: XCTestCase {

    /// A single still image: one block, unchanged behaviour.
    func testVisionCuSeqlensSingleImage() {
        XCTAssertEqual(visionCuSeqlens([THW(1, 4, 4)]), [0, 16])
    }

    /// Several still images: one block each, boundaries accumulate across frames.
    func testVisionCuSeqlensMultipleImages() {
        XCTAssertEqual(visionCuSeqlens([THW(1, 4, 4), THW(1, 2, 4)]), [0, 16, 24])
    }

    /// Video: every temporal slice is its own block.
    ///
    /// `THW(3, 4, 4)` contributes three blocks of 16 rows, `THW(2, 2, 4)` two blocks
    /// of 8 — 64 patch rows in total. Reading the accumulator inside a `map` instead
    /// of carrying it forward yields `[0, 16, 16, 16, 24, 24]`, which covers 24 of
    /// the 64 rows and leaves rows 24...63 outside every attendable block.
    func testVisionCuSeqlensTemporalGridsAccumulateAcrossSlices() {
        let frames = [THW(3, 4, 4), THW(2, 2, 4)]

        let boundaries = visionCuSeqlens(frames)

        XCTAssertEqual(boundaries, [0, 16, 32, 48, 56, 64])

        let patchRows = frames.reduce(0) { $0 + $1.t * $1.h * $1.w }
        XCTAssertEqual(
            boundaries.last, patchRows,
            "boundaries must cover every patch row, otherwise the tail attends outside its frame")
    }

    /// The boundaries must be strictly increasing — a repeated value is an empty
    /// block, which is exactly how the accumulator bug manifests.
    func testVisionCuSeqlensIsStrictlyIncreasing() {
        let boundaries = visionCuSeqlens([THW(4, 2, 2), THW(1, 4, 4)])
        for i in 1 ..< boundaries.count {
            XCTAssertGreaterThan(
                boundaries[i], boundaries[i - 1],
                "empty block at index \(i): \(boundaries)")
        }
    }
}
