// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN
import XCTest

@testable import MLXLMCommon

final class EvaluateTests: XCTestCase {

    /// Applies the reference filter chain (`applyTopP` → `applyMinP` → `applyTopK`)
    /// without triggering the combined `topK + topP` fast path.
    private func referenceFiltered(
        _ logProbabilities: MLXArray, topP: Float, topK: Int, minP: Float = 0.0
    ) -> MLXArray {
        let afterTopPAndMinP = TopPSampler(temperature: 1.0, topP: topP, minP: minP)
            .filtered(logProbabilities)
        return TopPSampler(temperature: 1.0, topK: topK).filtered(afterTopPAndMinP)
    }

    private func peakedLogits(vocabSize: Int, seed: UInt64) -> MLXArray {
        withRandomState(MLXRandom.RandomState(seed: seed)) {
            let background = MLXRandom.normal([1, vocabSize])
            let hotIndices = MLXArray((0 ..< 16).map { Int32(($0 * 9973 + 17) % vocabSize) })
                .reshaped(1, 16)
            let hotValues = MLXArray((0 ..< 16).map { Float(14.0 - Double($0) * 0.35) })
                .reshaped(1, 16)
            return putAlong(background, hotIndices, values: hotValues, axis: -1)
        }
    }

    func testTopPSamplerFilteredMatchesReferenceDistribution() {
        for vocabSize in [151_936, 262_144] {
            let peaked = logSoftmax(peakedLogits(vocabSize: vocabSize, seed: UInt64(vocabSize)))
            // Near-flat logits where all top-K tokens sum to far less than topP.
            let flat = withRandomState(MLXRandom.RandomState(seed: UInt64(vocabSize + 1))) {
                logSoftmax(MLXRandom.uniform(low: -1e-2, high: 1e-2, [1, vocabSize]) as MLXArray)
            }
            let batched = concatenated([peaked, flat], axis: 0)
            eval(peaked, flat, batched)

            let configurations: [(topP: Float, topK: Int, minP: Float)] = [
                (0.9, 1, 0.0),
                (0.9, 50, 0.0),
                (0.9, 50, 0.05),
                (0.95, 2048, 0.01),
            ]

            for config in configurations {
                let sampler = TopPSampler(
                    temperature: 0.7, topP: config.topP, topK: config.topK, minP: config.minP)

                for logprobs in [peaked, flat, batched] {
                    let actual = sampler.filtered(logprobs)
                    let expected = referenceFiltered(
                        logprobs, topP: config.topP, topK: config.topK, minP: config.minP)
                    eval(actual, expected)

                    XCTAssertEqual(actual.shape, expected.shape)
                    XCTAssertEqual(actual.dtype, expected.dtype)
                    XCTAssertEqual(actual.asArray(Float.self), expected.asArray(Float.self))
                }

                // On the flat row without minP, sum(topK) < topP so all K tokens survive.
                if config.minP == 0.0 {
                    let surviving = sampler.filtered(flat).asArray(Float.self)
                        .filter(\.isFinite).count
                    XCTAssertEqual(surviving, config.topK)
                }
            }

            // Ties strictly inside topK break in the same order as applyTopP.
            var tiedRow = [Float](repeating: -10.0, count: 64)
            tiedRow[2] = 8.0
            tiedRow[7] = 8.0
            tiedRow[19] = 8.0
            tiedRow[31] = 8.0
            let tiedLogprobs = logSoftmax(MLXArray(tiedRow).reshaped(1, 64))
            let tiedSampler = TopPSampler(temperature: 0.8, topP: 0.4, topK: 16)
            let actualTied = tiedSampler.filtered(tiedLogprobs)
            let expectedTied = referenceFiltered(tiedLogprobs, topP: 0.4, topK: 16)
            eval(actualTied, expectedTied)
            XCTAssertEqual(actualTied.asArray(Float.self), expectedTied.asArray(Float.self))
        }
    }

    func testSeededTopPSamplerMatchesReferenceDraws() {
        for vocabSize in [151_936, 262_144] {
            let logits = withRandomState(MLXRandom.RandomState(seed: UInt64(vocabSize + 99))) {
                MLXRandom.normal([1, vocabSize]) * 2.5
            }
            let logprobs = logSoftmax(logits)
            eval(logits, logprobs)

            let configurations: [(temp: Float, topP: Float, topK: Int, minP: Float)] = [
                (0.7, 0.9, 50, 0.0),
                (0.8, 0.95, 50, 0.02),
                (1.0, 0.85, 2048, 0.0),
            ]

            for config in configurations {
                let firstSampler = TopPSampler(
                    temperature: config.temp, topP: config.topP, topK: config.topK,
                    minP: config.minP, seed: 42)
                let secondSampler = TopPSampler(
                    temperature: config.temp, topP: config.topP, topK: config.topK,
                    minP: config.minP, seed: 42)
                let otherSeedSampler = TopPSampler(
                    temperature: config.temp, topP: config.topP, topK: config.topK,
                    minP: config.minP, seed: 43)
                let referenceState = MLXRandom.RandomState(seed: 42)

                let filteredReference = referenceFiltered(
                    logprobs, topP: config.topP, topK: config.topK, minP: config.minP)
                eval(filteredReference)

                var firstDraws: [Int] = []
                var secondDraws: [Int] = []
                var otherDraws: [Int] = []
                var expectedDraws: [Int] = []

                for _ in 0 ..< 24 {
                    firstDraws.append(firstSampler.sample(logits: logits).item(Int.self))
                    secondDraws.append(secondSampler.sample(logits: logits).item(Int.self))
                    otherDraws.append(otherSeedSampler.sample(logits: logits).item(Int.self))
                    let expected = withRandomState(referenceState) {
                        categorical(filteredReference * (1 / MLXArray(config.temp)))
                    }
                    expectedDraws.append(expected.item(Int.self))
                }

                XCTAssertEqual(firstDraws, secondDraws)
                XCTAssertEqual(firstDraws, expectedDraws)
                XCTAssertNotEqual(firstDraws, otherDraws)
            }
        }
    }
}
