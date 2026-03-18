// Copyright © 2025 Apple Inc.

import MLX
import MLXLMCommon
import XCTest

public class SampleTests: XCTestCase {

    private func sampleCounts(sampler: TopPSampler, logits: MLXArray, draws: Int) -> [Int: Int] {
        var counts: [Int: Int] = [:]
        for _ in 0 ..< draws {
            let token = sampler.sample(logits: logits).item(Int.self)
            counts[token, default: 0] += 1
        }
        return counts
    }

    private func frequency(_ counts: [Int: Int], token: Int, draws: Int) -> Float {
        Float(counts[token, default: 0]) / Float(draws)
    }

    private func assertOnlySampled(
        _ counts: [Int: Int], allowedTokens: Set<Int>, file: StaticString = #filePath,
        line: UInt = #line
    ) {
        for token in counts.keys {
            XCTAssertTrue(
                allowedTokens.contains(token), "Unexpected sampled token: \(token)", file: file,
                line: line)
        }
    }

    func testTopKSamplerKeepsOnlyTopToken() {
        let sampler = TopPSampler(temperature: 1.0, topK: 1)
        let logits = MLXArray([0.1 as Float, 2.0 as Float, 1.0 as Float])[.newAxis, .ellipsis]

        for _ in 0 ..< 10 {
            let token = sampler.sample(logits: logits).item(Int.self)
            XCTAssertEqual(token, 1)
        }
    }

    func testTopPSamplerLowThresholdKeepsMaxToken() {
        let probs = MLXArray([0.9 as Float, 0.0 as Float, 0.0 as Float, 0.1 as Float])[
            .newAxis, .ellipsis]
        let sampler = TopPSampler(temperature: 1.0, topP: 0.3)
        let counts = sampleCounts(sampler: sampler, logits: log(probs), draws: 200)

        XCTAssertEqual(counts[0], 200)
        assertOnlySampled(counts, allowedTokens: [0])
    }

    func testTopPSamplerPartialMassKeepsExpectedDistribution() {
        let probs = MLXArray([0.0 as Float, 0.5 as Float, 0.4 as Float, 0.1 as Float])[
            .newAxis, .ellipsis]
        let draws = 4000
        let sampler = TopPSampler(temperature: 1.0, topP: 0.6)
        let counts = sampleCounts(sampler: sampler, logits: log(probs), draws: draws)

        assertOnlySampled(counts, allowedTokens: [1, 2])
        XCTAssertEqual(frequency(counts, token: 1, draws: draws), 0.5556, accuracy: 0.06)
        XCTAssertEqual(frequency(counts, token: 2, draws: draws), 0.4444, accuracy: 0.06)
    }

    func testTopPSamplerHighThresholdKeepsExpectedDistribution() {
        let probs = MLXArray([0.0 as Float, 0.5 as Float, 0.4 as Float, 0.1 as Float])[
            .newAxis, .ellipsis]
        let draws = 4000
        let sampler = TopPSampler(temperature: 1.0, topP: 0.95)
        let counts = sampleCounts(sampler: sampler, logits: log(probs), draws: draws)

        assertOnlySampled(counts, allowedTokens: [1, 2, 3])
        XCTAssertEqual(frequency(counts, token: 1, draws: draws), 0.5, accuracy: 0.06)
        XCTAssertEqual(frequency(counts, token: 2, draws: draws), 0.4, accuracy: 0.06)
        XCTAssertEqual(frequency(counts, token: 3, draws: draws), 0.1, accuracy: 0.04)
    }

    func testTopKSamplerTopTwoKeepsExpectedDistribution() {
        let probs = MLXArray([0.6 as Float, 0.0 as Float, 0.1 as Float, 0.3 as Float])[
            .newAxis, .ellipsis]
        let draws = 4000
        let sampler = TopPSampler(temperature: 1.0, topK: 2)
        let counts = sampleCounts(sampler: sampler, logits: log(probs), draws: draws)

        assertOnlySampled(counts, allowedTokens: [0, 3])
        XCTAssertEqual(frequency(counts, token: 0, draws: draws), 0.6667, accuracy: 0.06)
        XCTAssertEqual(frequency(counts, token: 3, draws: draws), 0.3333, accuracy: 0.06)
    }

    func testMinPSamplerKeepsOnlyHighProbabilityToken() {
        let sampler = TopPSampler(temperature: 1.0, minP: 0.95)
        let logits = MLXArray([0.0 as Float, 0.0 as Float, 4.0 as Float])[.newAxis, .ellipsis]

        for _ in 0 ..< 10 {
            let token = sampler.sample(logits: logits).item(Int.self)
            XCTAssertEqual(token, 2)
        }
    }

    func testMinPSamplerLowThresholdKeepsExpectedDistribution() {
        let probs = MLXArray([0.9 as Float, 0.0 as Float, 0.0 as Float, 0.1 as Float])[
            .newAxis, .ellipsis]
        let draws = 4000
        let sampler = TopPSampler(temperature: 1.0, minP: 0.05)
        let counts = sampleCounts(sampler: sampler, logits: log(probs), draws: draws)

        assertOnlySampled(counts, allowedTokens: [0, 3])
        XCTAssertEqual(frequency(counts, token: 0, draws: draws), 0.9, accuracy: 0.05)
        XCTAssertEqual(frequency(counts, token: 3, draws: draws), 0.1, accuracy: 0.05)
    }

    func testGenerateParametersCreatesExpectedSampler() {
        XCTAssertTrue(GenerateParameters(temperature: 0.7, topK: 40).sampler() is TopPSampler)
        XCTAssertTrue(GenerateParameters(temperature: 0.7, minP: 0.1).sampler() is TopPSampler)
        XCTAssertTrue(GenerateParameters(temperature: 0).sampler() is ArgMaxSampler)
    }

    func testPresencePenaltyContextPenalizesSeenTokens() {
        var processor = PresencePenaltyContext(presencePenalty: 0.5, presenceContextSize: 20)
        processor.prompt(MLXArray([1, 1, 3]))

        let logits =
            MLXArray([1.0 as Float, 2.0 as Float, 3.0 as Float, 4.0 as Float])[.newAxis, .ellipsis]
        let processed = processor.process(logits: logits)
        let values = processed[0].asArray(Float.self)
        XCTAssertEqual(values[0], 1.0, accuracy: 1e-6)
        XCTAssertEqual(values[1], 1.5, accuracy: 1e-6)
        XCTAssertEqual(values[2], 3.0, accuracy: 1e-6)
        XCTAssertEqual(values[3], 3.5, accuracy: 1e-6)
    }

    func testFrequencyPenaltyContextPenalizesByCount() {
        var processor = FrequencyPenaltyContext(frequencyPenalty: 0.5, frequencyContextSize: 20)
        processor.prompt(MLXArray([1, 1, 3]))

        let logits =
            MLXArray([1.0 as Float, 2.0 as Float, 3.0 as Float, 4.0 as Float])[.newAxis, .ellipsis]
        let processed = processor.process(logits: logits)
        let values = processed[0].asArray(Float.self)
        XCTAssertEqual(values[0], 1.0, accuracy: 1e-6)
        XCTAssertEqual(values[1], 1.0, accuracy: 1e-6)
        XCTAssertEqual(values[2], 3.0, accuracy: 1e-6)
        XCTAssertEqual(values[3], 3.5, accuracy: 1e-6)
    }

    func testGenerateParametersCreatesExpectedPenaltyProcessor() {
        XCTAssertNotNil(GenerateParameters(repetitionPenalty: 1.1).processor())
        XCTAssertNotNil(GenerateParameters(presencePenalty: 0.5).processor())
        XCTAssertNotNil(GenerateParameters(frequencyPenalty: 0.5).processor())
        XCTAssertNotNil(
            GenerateParameters(
                repetitionPenalty: 1.1, presencePenalty: 0.5, frequencyPenalty: 0.5
            ).processor()
        )
    }

    func testPresencePenaltyContextPenalizesUniqueSeenTokens() {
        var processor = PresencePenaltyContext(presencePenalty: 0.5, presenceContextSize: 5)
        processor.prompt(MLXArray([0, 0, 0, 1, 1]))

        let logits = MLXArray.zeros([1, 4], type: Float.self)
        let processed = processor.process(logits: logits)
        let values = processed[0].asArray(Float.self)

        XCTAssertEqual(values[0], -0.5, accuracy: 1e-6)
        XCTAssertEqual(values[1], -0.5, accuracy: 1e-6)
        XCTAssertEqual(values[2], 0.0, accuracy: 1e-6)
        XCTAssertEqual(values[3], 0.0, accuracy: 1e-6)
    }

    func testFrequencyPenaltyContextPenalizesByTokenCount() {
        var processor = FrequencyPenaltyContext(frequencyPenalty: 0.5, frequencyContextSize: 5)
        processor.prompt(MLXArray([0, 0, 0, 1, 1]))

        let logits = MLXArray.zeros([1, 4], type: Float.self)
        let processed = processor.process(logits: logits)
        let values = processed[0].asArray(Float.self)

        XCTAssertEqual(values[0], -1.5, accuracy: 1e-6)
        XCTAssertEqual(values[1], -1.0, accuracy: 1e-6)
        XCTAssertEqual(values[2], 0.0, accuracy: 1e-6)
        XCTAssertEqual(values[3], 0.0, accuracy: 1e-6)
    }

    func testGenerateParametersPenaltyProcessorComposesPenaltiesInOrder() {
        var processor = GenerateParameters(
            repetitionPenalty: 1.5, repetitionContextSize: 5,
            presencePenalty: 0.5, presenceContextSize: 5,
            frequencyPenalty: 0.25, frequencyContextSize: 5
        ).processor()
        XCTAssertNotNil(processor)

        processor?.prompt(MLXArray([0, 0, 0, 1, 1]))
        let logits = MLXArray([1.0 as Float, 0.5 as Float, 0.0 as Float, -0.5 as Float])[
            .newAxis, .ellipsis
        ]
        let processed = processor?.process(logits: logits)
        guard let values = processed?[0].asArray(Float.self) else {
            XCTFail("Expected processed logits")
            return
        }
        XCTAssertEqual(values[0], -0.5833, accuracy: 1e-4)
        XCTAssertEqual(values[1], -0.6667, accuracy: 1e-4)
        XCTAssertEqual(values[2], 0.0, accuracy: 1e-4)
        XCTAssertEqual(values[3], -0.5, accuracy: 1e-4)
    }
}
