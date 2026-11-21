// Copyright © 2026 Apple Inc.

import CoreMedia
import Foundation
import MLX
import MLXLMCommon
import XCTest

@testable import MLXVLM

/// Qwen2.5-VL video M-RoPE positions. Temporal ids follow transformers'
/// `get_vision_position_ids`: `arange(t) × tokens_per_second × second_per_grid_t`, truncated.
final class Qwen25VLVideoPositionTests: XCTestCase {

    /// Text, a vision start, one video of grid (3, 4, 4) with merge 2, then text.
    private func positions(videoTemporalSteps: [Double]?) -> [[Int32]] {
        let tokens = MLXArray(
            [Int32](arrayLiteral: 7, 8, 502) + [Int32](repeating: 501, count: 12) + [9, 10]
        ).expandedDimensions(axis: 0)
        let (positionIds, _) = Qwen25VL.getRopeIndex(
            inputIds: tokens, imageGridTHW: nil, videoGridTHW: [THW(3, 4, 4)],
            spatialMergeSize: 2, imageTokenId: 500, videoTokenId: 501, visionStartTokenId: 502,
            videoTemporalSteps: videoTemporalSteps)
        return (0 ..< 3).map { positionIds[$0, 0, 0...].asArray(Int32.self) }
    }

    private let heights: [Int32] = [0, 1, 2] + [3, 3, 4, 4, 3, 3, 4, 4, 3, 3, 4, 4]
    private let widths: [Int32] = [0, 1, 2] + [3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4]

    /// The transformers docstring example at 2 tokens per second and one second per grid:
    /// temporal ids step by 2. Text after the video starts after the largest id.
    func testVideoTemporalIdsScaleByTheStep() {
        let ids = positions(videoTemporalSteps: [2])
        XCTAssertEqual(ids[0], [0, 1, 2] + [3, 3, 3, 3, 5, 5, 5, 5, 7, 7, 7, 7] + [8, 9])
        XCTAssertEqual(ids[1], heights + [8, 9])
        XCTAssertEqual(ids[2], widths + [8, 9])
    }

    func testFractionalStepsTruncate() {
        let ids = positions(videoTemporalSteps: [1.5])
        XCTAssertEqual(ids[0], [0, 1, 2] + [3, 3, 3, 3, 4, 4, 4, 4, 6, 6, 6, 6] + [7, 8])
    }

    func testNoStepAdvancesByOne() {
        let ids = positions(videoTemporalSteps: nil)
        XCTAssertEqual(ids[0], [0, 1, 2] + [3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5] + [6, 7])
    }

    /// `temporal_patch_size / fps` in transformers, from the sampled frame timestamps here.
    func testSecondsPerGridComesFromTheSampledFrameInterval() {
        func seconds(_ values: [Double]) -> [CMTime] {
            values.map { CMTime(seconds: $0, preferredTimescale: 600) }
        }

        XCTAssertEqual(
            Qwen25VLProcessor.secondsPerGrid(
                timestamps: seconds([0, 0.5, 1, 1.5]), temporalPatchSize: 2),
            1, accuracy: 1e-9)
        XCTAssertEqual(
            Qwen25VLProcessor.secondsPerGrid(timestamps: seconds([0, 1, 2]), temporalPatchSize: 2),
            2, accuracy: 1e-9)
        XCTAssertEqual(
            Qwen25VLProcessor.secondsPerGrid(timestamps: seconds([3]), temporalPatchSize: 2),
            1, accuracy: 1e-9)
    }
}
