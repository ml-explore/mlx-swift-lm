// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import Testing

@testable import MLXLMCommon

@Test
func testSpeculativeAcceptanceProbabilityBoundaries() {
    #expect(
        speculativeAcceptanceProbability(
            targetLogProbability: Foundation.log(0.6),
            draftLogProbability: Foundation.log(0.3)) == 1)

    let ratio = speculativeAcceptanceProbability(
        targetLogProbability: Foundation.log(0.2),
        draftLogProbability: Foundation.log(0.5))
    #expect(abs(ratio - 0.4) < 1e-6)

    #expect(
        speculativeAcceptanceProbability(
            targetLogProbability: -.infinity,
            draftLogProbability: Foundation.log(0.5)) == 0)
    #expect(
        speculativeAcceptanceProbability(
            targetLogProbability: Foundation.log(0.5),
            draftLogProbability: -.infinity) == 0)
    #expect(
        speculativeAcceptanceProbability(
            targetLogProbability: .nan,
            draftLogProbability: Foundation.log(0.5)) == 0)
}

@Test
func testSpeculativeResidualDistributionIsNormalized() {
    let target = MLXArray([Float(log(0.2)), Float(log(0.8))])[.newAxis, 0...]
    let draft = MLXArray([Float(log(0.6)), Float(log(0.4))])[.newAxis, 0...]

    let residual = speculativeResidualLogProbabilities(target: target, draft: draft)
    let probabilities = exp(residual).asArray(Float.self)

    #expect(probabilities[0] == 0)
    #expect(abs(probabilities[1] - 1) < 1e-6)
    #expect(abs(probabilities.reduce(0, +) - 1) < 1e-6)
}

@Test
func testDistributionSamplersExposeNormalizedLogProbabilities() {
    let logits = MLXArray([Float(0), 1, 2, 3])[.newAxis, 0...]

    let categorical = CategoricalSampler(temperature: 0.7, seed: 1)
    let categoricalMass = exp(categorical.logProbabilities(logits: logits)).sum()
        .item(Float.self)
    #expect(abs(categoricalMass - 1) < 1e-5)

    let filtered = TopPSampler(temperature: 0.7, topK: 2, seed: 1)
    let filteredProbabilities = exp(filtered.logProbabilities(logits: logits))
        .asArray(Float.self)
    #expect(abs(filteredProbabilities.reduce(0, +) - 1) < 1e-5)
    #expect(filteredProbabilities.filter { $0 > 0 }.count == 2)
}
