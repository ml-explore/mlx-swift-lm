// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLMCommon

@Suite(.serialized)
struct MTPSpeculativeSamplingTests {
    /// Each emitted token follows the target distribution, whatever the draft distribution.
    @Test func emittedTokensFollowTheTarget() {
        let target: [[Float]] = [
            [0.5, 0.2, 0.2, 0.1], [0.1, 0.6, 0.2, 0.1], [0.25, 0.25, 0.25, 0.25],
        ]
        let draft: [[Float]] = [[0.1, 0.6, 0.2, 0.1], [0.4, 0.4, 0.1, 0.1]]
        let logP = log(MLXArray(target.flatMap { $0 }, [3, 4]))
        let logQ = log(MLXArray(draft.flatMap { $0 }, [2, 4]))
        let state = MLXRandom.RandomState(seed: 3)
        let draftState = MLXRandom.RandomState(seed: 4)
        let trials = 4000
        var first = [Int](repeating: 0, count: 4)
        var second = [Int](repeating: 0, count: 4)
        var secondCount = 0
        for _ in 0 ..< trials {
            let drafts = withRandomState(draftState) { categorical(logQ) }
            let result = speculativeSamplingMTPVerification(
                targetLogProbabilities: logP, draftLogProbabilities: logQ, draftTokens: drafts,
                randomState: state)
            #expect(result.tokens.count == result.accepted + 1)
            first[result.tokens[0]] += 1
            if result.tokens.count > 1 {
                second[result.tokens[1]] += 1
                secondCount += 1
            }
        }
        for token in 0 ..< 4 {
            #expect(abs(Float(first[token]) / Float(trials) - target[0][token]) < 0.03)
            #expect(abs(Float(second[token]) / Float(secondCount) - target[1][token]) < 0.05)
        }
    }

    @Test func identicalDistributionsAcceptEveryDraft() {
        let logP = log(
            MLXArray([0.7, 0.2, 0.1, 0.3, 0.3, 0.4, 0.5, 0.25, 0.25] as [Float], [3, 3]))
        let drafts = MLXArray([2, 0] as [UInt32])
        let result = speculativeSamplingMTPVerification(
            targetLogProbabilities: logP, draftLogProbabilities: logP[..<2], draftTokens: drafts,
            randomState: MLXRandom.RandomState(seed: 1))
        #expect(result.accepted == 2)
        #expect(Array(result.tokens.prefix(2)) == [2, 0])
        #expect(result.finalToken.dtype == drafts.dtype)
        #expect(result.finalToken.item(Int.self) == result.tokens.last)
    }

    @Test func distributionsMatchTheSamplers() {
        let logits = MLXArray([2, 1, 0.5, -1, 3] as [Float], [1, 5])
        let categorical = CategoricalSampler(temperature: 0.7)
        #expect(
            maxDifference(categorical.logProbabilities(logits: logits), logSoftmax(logits / 0.7))
                < 1e-5)

        // Top-k keeps the two most likely tokens.
        let topK = TopPSampler(temperature: 0.5, topK: 2)
        let probabilities = exp(topK.logProbabilities(logits: logits)).asArray(Float.self)
        let kept = exp(MLXArray([3, 2] as [Float]) / 0.5)
        let expected = (kept / kept.sum()).asArray(Float.self)
        #expect(abs(probabilities[4] - expected[0]) < 1e-5)
        #expect(abs(probabilities[0] - expected[1]) < 1e-5)
        #expect(probabilities[1] == 0 && probabilities[2] == 0 && probabilities[3] == 0)
    }
}

private func maxDifference(_ a: MLXArray, _ b: MLXArray) -> Float {
    abs(a - b).max().item(Float.self)
}
