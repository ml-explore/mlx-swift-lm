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
final class QwenVLPreparedInputSplitTests: XCTestCase {

    // Qwen2.5-VL's actual special-token ids; any distinct values would do.
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
        let (positions, deltas) = Qwen25VL.getRopeIndex(
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
        // Turn 2 resumes with the cache holding exactly the prefix.
        let positionOffset = prefixIds.count + prefix.delta
        let warm = ropeIndex(
            ids: suffixIds, frames: [frame2], positionOffset: positionOffset)

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

    // MARK: - Splitting

    func testSplitKeepsOnlyTheSuffixImage() throws {
        let frame1 = THW(1, 4, 6)
        let frame2 = THW(1, 2, 4)
        let prefixIds = turnTokens(frame: frame1, leadingText: 3, trailingText: 4)
        let suffixIds = turnTokens(frame: frame2, leadingText: 2, trailingText: 3)
        let full = input(ids: prefixIds + suffixIds, frames: [frame1, frame2])

        let result = try XCTUnwrap(split(full, droppingFirst: prefixIds.count))

        XCTAssertEqual(result.text.tokens.shape, [1, suffixIds.count])
        XCTAssertEqual(result.text.tokens.asArray(Int.self), suffixIds)

        let image = try XCTUnwrap(result.image)
        XCTAssertEqual(image.frames?.count, 1)
        XCTAssertEqual(image.frames?.first?.values.0, frame2.t)
        XCTAssertEqual(image.frames?.first?.values.1, frame2.h)
        XCTAssertEqual(image.frames?.first?.values.2, frame2.w)
        XCTAssertEqual(image.pixels.shape, [frame2.product, 8])

        // the retained rows must be the *tail* of the original buffer
        let expected = pixels(rows: frame1.product + frame2.product)[
            frame1.product ..< (frame1.product + frame2.product), 0...]
        XCTAssertEqual(image.pixels.asArray(Float.self), expected.asArray(Float.self))
    }

    /// The payload the split hands back must satisfy the arity the feature merge
    /// silently assumes: one feature row group per placeholder run.
    func testSplitSuffixPlaceholderCountMatchesRetainedFrames() throws {
        let frame1 = THW(1, 4, 6)
        let frame2 = THW(1, 2, 4)
        let prefixIds = turnTokens(frame: frame1, leadingText: 3, trailingText: 4)
        let suffixIds = turnTokens(frame: frame2, leadingText: 2, trailingText: 3)
        let full = input(ids: prefixIds + suffixIds, frames: [frame1, frame2])

        let result = try XCTUnwrap(split(full, droppingFirst: prefixIds.count))
        let padCount = result.text.tokens.asArray(Int.self).filter { $0 == imageTokenId }.count
        let framePads = (result.image?.frames ?? []).reduce(0) {
            $0 + $1.product / (mergeSize * mergeSize)
        }
        XCTAssertEqual(padCount, framePads)
    }

    func testSplitAcceptsAllOnesMaskAndResizesIt() throws {
        let frame1 = THW(1, 4, 6)
        let frame2 = THW(1, 2, 4)
        let prefixIds = turnTokens(frame: frame1, leadingText: 3, trailingText: 4)
        let suffixIds = turnTokens(frame: frame2, leadingText: 2, trailingText: 3)
        let ids = prefixIds + suffixIds
        let mask = MLXArray.ones([1, ids.count]).asType(.int8)
        let full = input(ids: ids, frames: [frame1, frame2], mask: mask)

        let result = try XCTUnwrap(split(full, droppingFirst: prefixIds.count))
        XCTAssertEqual(result.text.mask?.shape, [1, suffixIds.count])
    }

    // MARK: - Refusals

    /// A temporal grid (`t > 1`) carried in the *image* payload is refused for the
    /// same reason video is: the vision full-attention mask mis-accumulates
    /// `cuSeqlens` across temporal slices, so those rows are not independently
    /// reusable. Refused whether the temporal frame is dropped with the prefix or
    /// retained in the suffix.
    func testSplitRefusesTemporalImageFrame() throws {
        let dropped = THW(2, 4, 4)
        let plain = THW(1, 2, 4)

        let prefixIds = turnTokens(frame: dropped, leadingText: 3, trailingText: 4)
        let suffixIds = turnTokens(frame: plain, leadingText: 2, trailingText: 3)
        XCTAssertNil(
            split(
                input(ids: prefixIds + suffixIds, frames: [dropped, plain]),
                droppingFirst: prefixIds.count),
            "a temporal frame in the cached prefix must refuse")

        let retainedPrefixIds = turnTokens(frame: plain, leadingText: 3, trailingText: 4)
        let retainedSuffixIds = turnTokens(frame: dropped, leadingText: 2, trailingText: 3)
        XCTAssertNil(
            split(
                input(
                    ids: retainedPrefixIds + retainedSuffixIds, frames: [plain, dropped]),
                droppingFirst: retainedPrefixIds.count),
            "a temporal frame retained in the suffix must refuse")
    }

    /// A boundary inside a vision block would hand the suffix a partial run of
    /// placeholders. The split must decline rather than guess.
    func testSplitRefusesBoundaryInsideAnImageBlock() throws {
        let frame1 = THW(1, 4, 6)
        let frame2 = THW(1, 2, 4)
        let prefixIds = turnTokens(frame: frame1, leadingText: 3, trailingText: 4)
        let suffixIds = turnTokens(frame: frame2, leadingText: 2, trailingText: 3)
        let full = input(ids: prefixIds + suffixIds, frames: [frame1, frame2])

        // land two tokens into the suffix's placeholder run
        let insideBlock = prefixIds.count + 2 + 1 + 2
        XCTAssertNil(split(full, droppingFirst: insideBlock))
    }

    func testSplitRefusesWhenFramesDoNotAccountForPlaceholders() throws {
        let frame1 = THW(1, 4, 6)
        let frame2 = THW(1, 2, 4)
        let ids =
            turnTokens(frame: frame1, leadingText: 3, trailingText: 4)
            + turnTokens(frame: frame2, leadingText: 2, trailingText: 3)
        // frames claim more placeholders than the prompt has
        let wrong = LMInput(
            text: .init(tokens: MLXArray(ids).expandedDimensions(axis: 0)),
            image: LMInput.ProcessedImage(
                pixels: pixels(rows: frame1.product + frame2.product + THW(1, 2, 2).product),
                frames: [frame1, frame2, THW(1, 2, 2)]))
        XCTAssertNil(split(wrong, droppingFirst: 12))
    }

    func testSplitRefusesWhenPixelRowsDoNotMatchFrames() throws {
        let frame1 = THW(1, 4, 6)
        let frame2 = THW(1, 2, 4)
        let prefixIds = turnTokens(frame: frame1, leadingText: 3, trailingText: 4)
        let suffixIds = turnTokens(frame: frame2, leadingText: 2, trailingText: 3)
        let wrong = LMInput(
            text: .init(tokens: MLXArray(prefixIds + suffixIds).expandedDimensions(axis: 0)),
            image: LMInput.ProcessedImage(
                pixels: pixels(rows: frame1.product), frames: [frame1, frame2]))
        XCTAssertNil(split(wrong, droppingFirst: prefixIds.count))
    }

    func testSplitRefusesMixedImageAndVideo() throws {
        let frame1 = THW(1, 4, 6)
        let frame2 = THW(1, 2, 4)
        let prefixIds = turnTokens(frame: frame1, leadingText: 3, trailingText: 4)
        let suffixIds = turnTokens(frame: frame2, leadingText: 2, trailingText: 3)
        let mixed = LMInput(
            text: .init(tokens: MLXArray(prefixIds + suffixIds).expandedDimensions(axis: 0)),
            image: LMInput.ProcessedImage(
                pixels: pixels(rows: frame1.product), frames: [frame1]),
            video: LMInput.ProcessedVideo(
                pixels: pixels(rows: frame2.product), frames: [frame2]))
        XCTAssertNil(split(mixed, droppingFirst: prefixIds.count))
    }

    /// Video is refused outright: `Qwen25VL`'s vision full-attention mask
    /// mis-accumulates `cuSeqlens` for `t > 1`, so a video's rows are not isolated
    /// from the rest of the buffer and dropping earlier rows is not neutral.
    func testSplitRefusesVideoOnly() throws {
        let frame1 = THW(2, 4, 6)
        let frame2 = THW(3, 2, 4)
        let padCount1 = frame1.product / (mergeSize * mergeSize)
        let padCount2 = frame2.product / (mergeSize * mergeSize)
        let prefixIds =
            [visionStartTokenId] + Array(repeating: videoTokenId, count: padCount1)
            + [visionEndTokenId, 1, 1]
        let suffixIds =
            [visionStartTokenId] + Array(repeating: videoTokenId, count: padCount2)
            + [visionEndTokenId, 2]
        let videoOnly = LMInput(
            text: .init(
                tokens: MLXArray(prefixIds + suffixIds).expandedDimensions(axis: 0)),
            video: LMInput.ProcessedVideo(
                pixels: pixels(rows: frame1.product + frame2.product),
                frames: [frame1, frame2]))
        XCTAssertNil(split(videoOnly, droppingFirst: prefixIds.count))
    }

    /// A video placeholder in a prompt whose payload is images-only means media this
    /// routine is not accounting for.
    func testSplitRefusesStrayVideoPlaceholder() throws {
        let frame1 = THW(1, 4, 6)
        let frame2 = THW(1, 2, 4)
        let prefixIds = turnTokens(frame: frame1, leadingText: 3, trailingText: 4)
        let suffixIds =
            turnTokens(frame: frame2, leadingText: 2, trailingText: 3) + [videoTokenId]
        let full = input(ids: prefixIds + suffixIds, frames: [frame1, frame2])
        XCTAssertNil(split(full, droppingFirst: prefixIds.count))
    }

    func testSplitRefusesNonUniformMask() throws {
        let frame1 = THW(1, 4, 6)
        let frame2 = THW(1, 2, 4)
        let prefixIds = turnTokens(frame: frame1, leadingText: 3, trailingText: 4)
        let suffixIds = turnTokens(frame: frame2, leadingText: 2, trailingText: 3)
        let ids = prefixIds + suffixIds
        var maskValues = Array(repeating: Int32(1), count: ids.count)
        maskValues[0] = 0
        let mask = MLXArray(maskValues).reshaped(1, ids.count).asType(.int8)
        let full = input(ids: ids, frames: [frame1, frame2], mask: mask)
        XCTAssertNil(split(full, droppingFirst: prefixIds.count))
    }

    /// A rank-1 suffix would route back to the cold path in
    /// `prepare(_:cache:state:windowSize:)`, which computes positions from zero.
    func testSplitRefusesRankOneTokens() throws {
        let frame1 = THW(1, 4, 6)
        let frame2 = THW(1, 2, 4)
        let prefixIds = turnTokens(frame: frame1, leadingText: 3, trailingText: 4)
        let suffixIds = turnTokens(frame: frame2, leadingText: 2, trailingText: 3)
        let flat = LMInput(
            text: .init(tokens: MLXArray(prefixIds + suffixIds)),
            image: LMInput.ProcessedImage(
                pixels: pixels(rows: frame1.product + frame2.product),
                frames: [frame1, frame2]))
        XCTAssertNil(split(flat, droppingFirst: prefixIds.count))
    }

    func testSplitRefusesPrecomputedPositionIds() throws {
        let frame1 = THW(1, 4, 6)
        let frame2 = THW(1, 2, 4)
        let prefixIds = turnTokens(frame: frame1, leadingText: 3, trailingText: 4)
        let suffixIds = turnTokens(frame: frame2, leadingText: 2, trailingText: 3)
        let ids = prefixIds + suffixIds
        let withPositions = LMInput(
            text: .init(tokens: MLXArray(ids).expandedDimensions(axis: 0)),
            image: LMInput.ProcessedImage(
                pixels: pixels(rows: frame1.product + frame2.product),
                positionIds: MLXArray(Array(0 ..< ids.count)),
                frames: [frame1, frame2]))
        XCTAssertNil(split(withPositions, droppingFirst: prefixIds.count))
    }

    func testSplitRefusesMissingFrames() throws {
        let frame1 = THW(1, 4, 6)
        let ids = turnTokens(frame: frame1, leadingText: 3, trailingText: 4)
        let noFrames = LMInput(
            text: .init(tokens: MLXArray(ids).expandedDimensions(axis: 0)),
            image: LMInput.ProcessedImage(pixels: pixels(rows: frame1.product)))
        XCTAssertNil(split(noFrames, droppingFirst: 3))
    }

    func testSplitRefusesOutOfRangeBoundaries() throws {
        let frame1 = THW(1, 4, 6)
        let ids = turnTokens(frame: frame1, leadingText: 3, trailingText: 4)
        let full = input(ids: ids, frames: [frame1])
        XCTAssertNil(split(full, droppingFirst: 0))
        XCTAssertNil(split(full, droppingFirst: ids.count))
        XCTAssertNil(split(full, droppingFirst: ids.count + 5))
    }

    /// Nothing remains for the suffix — the caller has no reason to reuse and the
    /// split must not hand back an image-free payload that looks reusable.
    func testSplitRefusesWhenAllMediaBelongsToThePrefix() throws {
        let frame1 = THW(1, 4, 6)
        let prefixIds = turnTokens(frame: frame1, leadingText: 3, trailingText: 4)
        let suffixIds = Array(repeating: 3, count: 5)
        let full = input(ids: prefixIds + suffixIds, frames: [frame1])
        XCTAssertNil(split(full, droppingFirst: prefixIds.count))
    }
}
