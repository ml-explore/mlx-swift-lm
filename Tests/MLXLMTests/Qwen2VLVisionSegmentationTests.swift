// Copyright © 2026 Apple Inc.

// Qwen2-VL vision attention must stay inside each image. Two images in one
// prompt concatenate their patch rows. Attending across that concatenation
// mixes one image into the other. The reference attends per cu_seqlens block,
// so a block's output must not depend on any other block's inputs.

import Foundation
import MLX
import MLXLMCommon
import XCTest

@testable import MLXVLM

final class Qwen2VLVisionSegmentationTests: XCTestCase {

    private func qkv(rows: Int, seed: UInt64) -> (MLXArray, MLXArray, MLXArray) {
        let key = MLXRandom.key(seed)
        let parts = MLXRandom.split(key: key, into: 3)
        return (
            MLXRandom.normal([rows, 2, 8], key: parts[0]),
            MLXRandom.normal([rows, 2, 8], key: parts[1]),
            MLXRandom.normal([rows, 2, 8], key: parts[2])
        )
    }

    func testTwoImageBoundariesMatchReference() {
        XCTAssertEqual(visionCuSeqlens([THW(1, 16, 16), THW(1, 16, 16)]), [0, 256, 512])
    }

    func testEachImageAttendsOnlyToItself() {
        Device.withDefaultDevice(.cpu) {
            let frames = [THW(1, 4, 4), THW(1, 2, 4)]
            let cu = visionCuSeqlens(frames)
            let (q, k, v) = qkv(rows: cu.last!, seed: 7)
            let scale: Float = 1 / Float(8).squareRoot()

            let joint = qwen2VLSegmentedVisionAttention(
                queries: q, keys: k, values: v, scale: scale, cuSeqlens: cu)
            XCTAssertEqual(joint.shape, [cu.last!, 16])

            for i in 1 ..< cu.count {
                let r = cu[i - 1] ..< cu[i]
                let alone = qwen2VLSegmentedVisionAttention(
                    queries: q[r], keys: k[r], values: v[r], scale: scale,
                    cuSeqlens: [0, r.count])
                XCTAssertTrue(
                    allClose(joint[r], alone, rtol: 1e-5, atol: 1e-5).item(Bool.self),
                    "block \(i) attended outside its image")
            }
        }
    }

    func testSecondImageIndependentOfFirst() {
        Device.withDefaultDevice(.cpu) {
            let cu = visionCuSeqlens([THW(1, 4, 4), THW(1, 4, 4)])
            let (q, k, v) = qkv(rows: 32, seed: 1)
            let (q2, k2, v2) = qkv(rows: 32, seed: 2)
            let first = 0 ..< 16
            let second = 16 ..< 32
            let scale: Float = 0.25

            let a = qwen2VLSegmentedVisionAttention(
                queries: q, keys: k, values: v, scale: scale, cuSeqlens: cu)
            let b = qwen2VLSegmentedVisionAttention(
                queries: concatenated([q2[first], q[second]]),
                keys: concatenated([k2[first], k[second]]),
                values: concatenated([v2[first], v[second]]),
                scale: scale, cuSeqlens: cu)
            XCTAssertTrue(
                allClose(a[second], b[second], rtol: 1e-5, atol: 1e-5).item(Bool.self))
        }
    }
}
