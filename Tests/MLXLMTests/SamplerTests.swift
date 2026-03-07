// Copyright © 2026 Apple Inc.

import MLX
@testable import MLXLMCommon
import XCTest

final class SamplerTests: XCTestCase {

    private let logProbabilities = log(MLXArray([0.05 as Float, 0.15, 0.30, 0.50], [1, 4]))

    func testTopKFilteringKeepsOnlyHighestLogProbabilities() {
        let filtered = applyTopK(logProbabilities, topK: 2)
        let filteredValues = filtered.asArray(Float.self)
        let originalValues = logProbabilities.asArray(Float.self)

        XCTAssertTrue(filteredValues[0].isInfinite && filteredValues[0] < 0)
        XCTAssertTrue(filteredValues[1].isInfinite && filteredValues[1] < 0)
        XCTAssertEqual(filteredValues[2], originalValues[2], accuracy: 1e-6)
        XCTAssertEqual(filteredValues[3], originalValues[3], accuracy: 1e-6)
    }

    func testMinPFilteringRespectsMinimumTokensToKeep() {
        let filtered = applyMinP(logProbabilities, minP: 0.2, minTokensToKeep: 3)
        let filteredValues = filtered.asArray(Float.self)
        let originalValues = logProbabilities.asArray(Float.self)

        XCTAssertEqual(filteredValues[0], originalValues[0], accuracy: 1e-6)
        XCTAssertEqual(filteredValues[1], originalValues[1], accuracy: 1e-6)
        XCTAssertEqual(filteredValues[2], originalValues[2], accuracy: 1e-6)
        XCTAssertTrue(filteredValues[3].isInfinite && filteredValues[3] < 0)
    }

    func testTopKSamplerUsesSeededRandomState() {
        let samplerA = CategoricalSampler(
            temperature: 1.0,
            topK: 2,
            randomState: MLXRandom.RandomState(seed: 7)
        )
        let samplerB = CategoricalSampler(
            temperature: 1.0,
            topK: 2,
            randomState: MLXRandom.RandomState(seed: 7)
        )

        let tokenA = samplerA.sample(logits: logProbabilities).item(Int.self)
        let tokenB = samplerB.sample(logits: logProbabilities).item(Int.self)

        XCTAssertEqual(tokenA, tokenB)
        XCTAssertTrue([2, 3].contains(tokenA))
    }

    func testMinPSamplerUsesSeededRandomState() {
        let samplerA = CategoricalSampler(
            temperature: 1.0,
            minP: 0.2,
            minTokensToKeep: 3,
            randomState: MLXRandom.RandomState(seed: 11)
        )
        let samplerB = CategoricalSampler(
            temperature: 1.0,
            minP: 0.2,
            minTokensToKeep: 3,
            randomState: MLXRandom.RandomState(seed: 11)
        )

        let tokenA = samplerA.sample(logits: logProbabilities).item(Int.self)
        let tokenB = samplerB.sample(logits: logProbabilities).item(Int.self)

        XCTAssertEqual(tokenA, tokenB)
        XCTAssertTrue([1, 2, 3].contains(tokenA))
    }
}
