// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import XCTest

@testable import MLXLMCommon
@testable import MLXVLM

/// Covers ``QwenVL/splitPreparedInput(_:droppingFirst:imageTokenId:videoTokenId:mergeSize:)``
/// and, more importantly, the property that makes it safe to use: prefilling only
/// the split-off suffix at the carried `positionOffset` produces the *same* M-RoPE
/// positions a cold full prefill would have produced for those tokens.
///
/// These tests are deliberately weight-free. They drive the position math directly,
/// so the equality below is exact rather than inferred from generated text.
final class Qwen3VLPreparedInputSplitTests: XCTestCase {

    // Qwen3-VL's actual special-token ids; any distinct values would do.
    private let imageTokenId = 151_655
    private let videoTokenId = 151_656
    private let visionStartTokenId = 151_652
    private let visionEndTokenId = 151_653
    private let mergeSize = 2

    /// Tokens for one "turn": leading text, a vision block sized for `frame`,
    /// then trailing text (which also stands in for the turn's generated tokens).
    private func turnTokens(frame: THW, leadingText: Int, trailingText: Int) -> [Int] {
        let padCount = frame.product / (mergeSize * mergeSize)
        return Array(repeating: 1, count: leadingText)
            + [visionStartTokenId]
            + Array(repeating: imageTokenId, count: padCount)
            + [visionEndTokenId]
            + Array(repeating: 2, count: trailingText)
    }

    private func pixels(rows: Int) -> MLXArray {
        MLXArray((0 ..< (rows * 8)).map { Float($0) }).reshaped(rows, 8)
    }

    private func input(ids: [Int], frames: [THW], mask: MLXArray? = nil) -> LMInput {
        let rows = frames.reduce(0) { $0 + $1.product }
        return LMInput(
            text: .init(tokens: MLXArray(ids).expandedDimensions(axis: 0), mask: mask),
            image: LMInput.ProcessedImage(pixels: pixels(rows: rows), frames: frames))
    }

    private func split(_ input: LMInput, droppingFirst prefixTokenCount: Int) -> LMInput? {
        QwenVL.splitPreparedInput(
            input,
            droppingFirst: prefixTokenCount,
            imageTokenId: imageTokenId,
            videoTokenId: videoTokenId,
            mergeSize: mergeSize)
    }

    private func ropeIndex(ids: [Int], frames: [THW], positionOffset: Int = 0) -> (
        positions: MLXArray, delta: Int
    ) {
        let (positions, deltas) = Qwen3VLLanguage.getRopeIndex(
            inputIds: MLXArray(ids).expandedDimensions(axis: 0),
            imageGridTHW: frames,
            videoGridTHW: nil,
            spatialMergeSize: mergeSize,
            imageTokenId: imageTokenId,
            videoTokenId: videoTokenId,
            visionStartTokenId: visionStartTokenId,
            positionOffset: positionOffset)
        return (positions, deltas.asType(.int32).item(Int.self))
    }

    // MARK: - The correctness property

    /// The whole justification for reusing the cache on an append-only media turn:
    /// positions computed for the suffix alone, offset by the carried rope delta,
    /// are identical to the tail of the positions a cold full prefill computes.
    ///
    /// This is checked with an image in the *prefix* and another in the *suffix*,
    /// which is the Qwen-VL "new screenshot every turn" shape.
    func testSuffixPositionsMatchColdPrefillPositions() throws {
        let frame1 = THW(1, 4, 6)
        let frame2 = THW(1, 2, 4)
        let prefixIds = turnTokens(frame: frame1, leadingText: 3, trailingText: 4)
        let suffixIds = turnTokens(frame: frame2, leadingText: 2, trailingText: 3)
        let fullIds = prefixIds + suffixIds

        let cold = ropeIndex(ids: fullIds, frames: [frame1, frame2])

        // Turn 1 prefilled the prefix cold; `prepare` stores its rope delta as-is.
        let prefix = ropeIndex(ids: prefixIds, frames: [frame1])
        // Turn 2 resumes with the cache holding exactly the prefix. The suffix is the
        // actual split product, not a hand-built twin.
        let full = input(ids: fullIds, frames: [frame1, frame2])
        let suffix = try XCTUnwrap(split(full, droppingFirst: prefixIds.count))
        XCTAssertEqual(suffix.text.tokens.asArray(Int32.self), suffixIds.map(Int32.init))
        let suffixFrames = try XCTUnwrap(suffix.image?.frames)
        let positionOffset = prefixIds.count + prefix.delta
        let warm = ropeIndex(
            ids: suffixIds, frames: suffixFrames, positionOffset: positionOffset)

        let coldTail = cold.positions[0..., 0..., prefixIds.count ..< fullIds.count]
        XCTAssertEqual(warm.positions.shape, coldTail.shape)
        XCTAssertEqual(
            warm.positions.asArray(Int32.self), coldTail.asArray(Int32.self),
            "warm suffix positions must equal the cold prefill's positions for the same tokens")
    }

    /// Same property with two images already cached, so the offset has to survive
    /// more than one vision block.
    func testSuffixPositionsMatchColdPrefillWithMultipleCachedImages() throws {
        let frame1 = THW(1, 4, 6)
        let frame2 = THW(1, 6, 6)
        let frame3 = THW(1, 2, 4)
        let turn1 = turnTokens(frame: frame1, leadingText: 3, trailingText: 2)
        let turn2 = turnTokens(frame: frame2, leadingText: 1, trailingText: 5)
        let turn3 = turnTokens(frame: frame3, leadingText: 2, trailingText: 1)
        let prefixIds = turn1 + turn2
        let fullIds = prefixIds + turn3

        let cold = ropeIndex(ids: fullIds, frames: [frame1, frame2, frame3])
        let prefix = ropeIndex(ids: prefixIds, frames: [frame1, frame2])
        let warm = ropeIndex(
            ids: turn3, frames: [frame3],
            positionOffset: prefixIds.count + prefix.delta)

        let coldTail = cold.positions[0..., 0..., prefixIds.count ..< fullIds.count]
        XCTAssertEqual(
            warm.positions.asArray(Int32.self), coldTail.asArray(Int32.self))
    }

    /// The delta the continuation stores back must anchor the *next* turn too,
    /// mirroring `prepareContinuation`'s `ropeDeltas - cacheOffset` bookkeeping.
    func testCarriedDeltaAnchorsASecondContinuation() throws {
        let frame1 = THW(1, 4, 6)
        let frame2 = THW(1, 2, 4)
        let frame3 = THW(1, 4, 4)
        let turn1 = turnTokens(frame: frame1, leadingText: 3, trailingText: 2)
        let turn2 = turnTokens(frame: frame2, leadingText: 2, trailingText: 3)
        let turn3 = turnTokens(frame: frame3, leadingText: 1, trailingText: 2)
        let fullIds = turn1 + turn2 + turn3

        let cold = ropeIndex(ids: fullIds, frames: [frame1, frame2, frame3])

        // turn 1 cold
        let s1 = ropeIndex(ids: turn1, frames: [frame1])
        var cacheOffset = turn1.count
        var carriedDelta = s1.delta

        // turn 2 as a continuation
        let s2 = ropeIndex(
            ids: turn2, frames: [frame2], positionOffset: cacheOffset + carriedDelta)
        carriedDelta = s2.delta - cacheOffset
        cacheOffset += turn2.count

        // turn 3 as a continuation off turn 2's carried state
        let s3 = ropeIndex(
            ids: turn3, frames: [frame3], positionOffset: cacheOffset + carriedDelta)

        let coldTail = cold.positions[
            0..., 0..., (turn1.count + turn2.count) ..< fullIds.count]
        XCTAssertEqual(s3.positions.asArray(Int32.self), coldTail.asArray(Int32.self))
    }
}
