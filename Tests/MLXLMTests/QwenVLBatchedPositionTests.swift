// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN
import MLXVLM
import Testing

@testable import MLXLMCommon

@Suite(.serialized)
struct QwenVLBatchedPositionTests {
    @Test(
        "Stateless Qwen VL text rows use their own positions", arguments: QwenVLTextFamily.allCases)
    func raggedTextLogits(family: QwenVLTextFamily) throws {
        try Device.withDefaultDevice(.cpu) {
            let model = try tinyQwenVLText(family)
            let prompts = [[Int32(1), 2, 3], [4, 5, 6, 7, 8, 9, 10]]
            var singles = try prompts.map { prompt in
                let cache = try model.newCache(parameters: nil)
                let output = model.nextTokenLogits(
                    .init(tokens: MLXArray(prompt)[.newAxis]), cache: cache, state: nil)
                #expect(output.state?.isEmpty != false)
                eval(output.logits, cache)
                return cache
            }
            let batch = try singles[0].indices.map { layer in
                try mergeCaches(singles.map { $0[layer].copy() })
            }
            let first = try #require(batch.first as? BatchKVCache)
            #expect(first.batchOffset.asArray(Int32.self) == [3, 7])

            for length in [1, 2, 1] {
                let tokens = MLXArray((0 ..< 2 * length).map { Int32(11 + $0) })
                    .reshaped([2, length])
                let actual = model.nextTokenLogits(.init(tokens: tokens), cache: batch, state: nil)
                #expect(actual.state?.isEmpty != false)
                for row in singles.indices {
                    let expected = model.nextTokenLogits(
                        .init(tokens: tokens[row ..< row + 1]), cache: singles[row], state: nil)
                    #expect(
                        maxAbsDifference(actual.logits[row ..< row + 1], expected.logits) < 1e-4,
                        "\(family), row \(row), chunk length \(length)")
                }
                let expectedOffsets = singles.map { $0[0].offset }
                #expect(first.batchOffset.asArray(Int.self) == expectedOffsets)
            }
            batch.forEach { $0.filter(batchIndices: batchRows(0)) }
            singles = [singles[0]]
            if case .scalar(let offset) = first.ropeOffset {
                #expect(offset == singles[0][0].offset)
            } else {
                Issue.record("A remaining single row must use the scalar RoPE path")
            }
            let tokens = MLXArray([Int32(21), 22])[.newAxis]
            let actual = model.nextTokenLogits(.init(tokens: tokens), cache: batch, state: nil)
            let expected = model.nextTokenLogits(
                .init(tokens: tokens), cache: singles[0], state: nil)
            #expect(maxAbsDifference(actual.logits, expected.logits) < 1e-4)
        }
    }

}

// `QwenVLTextFamily` and `tinyQwenVLText(_:)` live in RoPEApplicationTests.swift.
