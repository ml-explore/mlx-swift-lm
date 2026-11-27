// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import XCTest

@testable import MLXVLM

/// The vision tower emits every image row before every video row. These tests pin how
/// `QwenVL.VisualTokenLayout` maps those rows onto a prompt that puts a video first.
final class QwenVLVisualTokenLayoutTests: XCTestCase {

    private let image = 500
    private let video = 501

    private func tokens(_ ids: [Int32]) -> MLXArray {
        MLXArray(ids).expandedDimensions(axis: 0)
    }

    func testVideoBeforeImageTakesTheVideoRows() {
        let layout = QwenVL.VisualTokenLayout(
            inputIds: tokens([7, 501, 501, 8, 500, 500]), imageTokenId: image,
            videoTokenId: video)

        XCTAssertEqual(layout.indices, [1, 2, 4, 5])
        XCTAssertEqual(layout.featureRows?.asArray(Int32.self), [2, 3, 0, 1])
    }

    func testRowsAlreadyInTokenOrderNeedNoGather() {
        let imagesOnly = QwenVL.VisualTokenLayout(
            inputIds: tokens([7, 500, 500, 8, 500]), imageTokenId: image, videoTokenId: video)
        let imageThenVideo = QwenVL.VisualTokenLayout(
            inputIds: tokens([500, 7, 501, 501]), imageTokenId: image, videoTokenId: video)
        let textOnly = QwenVL.VisualTokenLayout(
            inputIds: tokens([7, 8, 9]), imageTokenId: image, videoTokenId: video)

        XCTAssertEqual(imagesOnly.indices, [1, 2, 4])
        XCTAssertNil(imagesOnly.featureRows)
        XCTAssertEqual(imageThenVideo.indices, [0, 2, 3])
        XCTAssertNil(imageThenVideo.featureRows)
        XCTAssertEqual(textOnly.indices, [])
        XCTAssertNil(textOnly.featureRows)
    }

    /// The merge Qwen2-VL, Qwen2.5-VL and GLM-OCR share.
    func testSharedMergePlacesVideoRowsOnVideoTokens() {
        let ids = tokens([7, 501, 501, 8, 500, 500])
        let embeds = zeros([1, 6, 2])
        // Image rows 1 and 2, then video rows 3 and 4, as the vision tower emits them.
        let features = MLXArray([Float](arrayLiteral: 1, 1, 2, 2, 3, 3, 4, 4), [1, 4, 2])

        let merged = QwenVL.mergeInputIdsWithImageFeatures(
            inputIds: ids, inputEmbeds: embeds, imageFeatures: features,
            imageTokenId: image, videoTokenId: video)

        XCTAssertEqual(
            merged.asArray(Float.self), [0, 0, 3, 3, 4, 4, 0, 0, 1, 1, 2, 2])
    }
}
