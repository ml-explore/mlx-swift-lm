// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import Testing

@testable import MLXLMCommon

@Suite("Speculative sampling")
struct SpeculativeSamplingTests {

    static let target: [Float] = [0.5, 0.2, 0.15, 0.1, 0.05]

    /// The frequency of each token among `tokens`.
    private func frequencies(_ tokens: MLXArray, vocabulary: Int) -> [Float] {
        let rows = Float(tokens.size)
        return (0 ..< vocabulary).map {
            (tokens .== MLXArray(Int32($0))).sum().item(Float.self) / rows
        }
    }

    private func repeated(_ probabilities: [Float], rows: Int) -> MLXArray {
        broadcast(log(MLXArray(probabilities)).reshaped(1, -1), to: [rows, probabilities.count])
    }

    @Test(
        "Kept drafts and corrections follow the target distribution",
        arguments: [
            [Float(0.1), 0.4, 0.1, 0.3, 0.1],
            [0.2, 0.3, 0.5],
            [0.5, 0.2, 0.15, 0.1, 0.05],
        ])
    func emitsTarget(proposal: [Float]) {
        let rows = 200_000
        let randomState = MLXRandom.RandomState(seed: 7)
        let target = repeated(Self.target, rows: rows)
        let proposalRows = repeated(proposal, rows: rows)
        let drafts = categorical(proposalRows, key: randomState)

        let (kept, corrections) = speculativeSamplingDecisions(
            target: target, proposal: proposalRows, drafts: drafts, randomState: randomState)
        let emitted = MLX.where(kept, drafts, corrections)

        for (frequency, probability) in zip(frequencies(emitted, vocabulary: 5), Self.target) {
            #expect(abs(frequency - probability) < 0.005)
        }
        let keptShare = kept.asType(.float32).mean().item(Float.self)
        let overlap = zip(Self.target, proposal + [0, 0]).map { min($0, $1) }.reduce(0, +)
        #expect(abs(keptShare - overlap) < 0.005)
    }

    @Test("A drafter that drafts without drawing keeps each draft with its target probability")
    func deterministicDrafts() {
        let rows = 200_000
        let randomState = MLXRandom.RandomState(seed: 9)
        let drafts = MLXArray(Array(repeating: Int32(1), count: rows))

        let (kept, corrections) = speculativeSamplingDecisions(
            target: repeated(Self.target, rows: rows),
            proposal: deterministicProposal(drafts: drafts, width: 5), drafts: drafts,
            randomState: randomState)
        let emitted = MLX.where(kept, drafts, corrections)

        #expect(abs(kept.asType(.float32).mean().item(Float.self) - 0.2) < 0.005)
        for (frequency, probability) in zip(frequencies(emitted, vocabulary: 5), Self.target) {
            #expect(abs(frequency - probability) < 0.005)
        }
    }

    @Test(
        "Samplers draw from the distribution they report",
        arguments: [
            GenerateParameters(temperature: 0.7, topP: 0.9, topK: 4, minP: 0.05, seed: 3),
            GenerateParameters(temperature: 1.3, seed: 5),
        ])
    func samplerReportsItsDistribution(parameters: GenerateParameters) throws {
        let sampler = try #require(parameters.sampler() as? any DistributionLogitSampler)
        let logits: [Float] = [2.0, 1.5, 1.4, 0.3, -0.5, -1.0, 0.9, 0.0]
        let reported = exp(sampler.logProbabilities(logits: MLXArray(logits).reshaped(1, -1)))
            .asArray(Float.self)
        let rows = 200_000
        let draws = sampler.sample(
            logits: broadcast(MLXArray(logits).reshaped(1, -1), to: [rows, logits.count]))

        #expect(abs(reported.reduce(0, +) - 1) < 1e-5)
        for (frequency, probability) in zip(
            frequencies(draws, vocabulary: logits.count), reported)
        {
            #expect(abs(frequency - probability) < 0.005)
        }
    }
}
