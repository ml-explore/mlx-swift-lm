// Copyright © 2025 Apple Inc.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import MLXOptimizers
import XCTest

public class EvalTests: XCTestCase {

    func testLlamaEval() throws {
        let config = LlamaConfiguration(
            hiddenSize: 64, hiddenLayers: 16, intermediateSize: 512, attentionHeads: 32,
            rmsNormEps: 0.00001, vocabularySize: 100, kvHeads: 8)
        let model = LlamaModel(config)
        quantize(model: model, groupSize: 64, bits: 4)

        let input = MLXArray([1, 2, 3, 4, 5])[.newAxis, .ellipsis]
        let output = model.callAsFunction(input, cache: nil)

        XCTAssertEqual(output.shape, [1, 5, 100])
    }

    func testLlamaLora() throws {
        let config = LlamaConfiguration(
            hiddenSize: 64, hiddenLayers: 16, intermediateSize: 512, attentionHeads: 32,
            rmsNormEps: 0.00001, vocabularySize: 100, kvHeads: 8)
        let model = LlamaModel(config)
        quantize(model: model, groupSize: 64, bits: 4)

        let optimizer = Adam(learningRate: 1e-5)

        let train = ["a", "b", "c"]
        let valid = ["x", "y", "z"]

        let tokenizer = TestTokenizer()
        let parameters = LoRATrain.Parameters(iterations: 5)

        try LoRATrain.train(
            model: model, train: train, validate: valid, optimizer: optimizer,
            tokenizer: tokenizer,
            parameters: parameters
        ) { progress in
            print(progress)
            return .more
        }

        let input = MLXArray([1, 2, 3, 4, 5])[.newAxis, .ellipsis]
        let output = model.callAsFunction(input, cache: nil)

        XCTAssertEqual(output.shape, [1, 5, 100])
    }

    func testConcurrentEvaluation() async throws {
        let config = LlamaConfiguration(
            hiddenSize: 64, hiddenLayers: 4, intermediateSize: 128, attentionHeads: 8,
            rmsNormEps: 0.00001, vocabularySize: 100, kvHeads: 4)
        let model = LlamaModel(config)
        quantize(model: model, groupSize: 64, bits: 4)

        // Force evaluation of all model weights before concurrent usage
        // This ensures all weight promises are realized and avoids race conditions
        eval(model)

        let processor = TestInputProcessor()
        let container = ModelContainer(
            context: .init(
                configuration: processor.configuration, model: model, processor: processor,
                tokenizer: processor.tokenizer))

        let numTasks = 3
        let shapes = await withTaskGroup(of: [Int].self) { group in
            var allResults: [[Int]] = []

            for taskId in 0 ..< numTasks {
                group.addTask {
                    await container.perform { context in
                        let input = MLXArray([
                            1 + taskId, 2 + taskId, 3 + taskId, 4 + taskId, 5 + taskId,
                        ])[.newAxis, .ellipsis]

                        let output = context.model.callAsFunction(input, cache: nil)
                        eval(output)

                        return output.shape
                    }
                }
            }

            for await result in group {
                allResults.append(result)
            }

            return allResults
        }

        XCTAssertEqual(shapes.count, numTasks)

        for result in shapes {
            XCTAssertEqual(result, [1, 5, 100])
        }
    }

    func testConcurrentSampling() async throws {
        let vocabSize = 100

        let numSamplers = 4
        let results = try await withThrowingTaskGroup(of: Int.self) { group in
            var samplerResults: [Int] = []

            for samplerId in 0 ..< numSamplers {
                group.addTask {
                    let logits = MLXRandom.normal([1, vocabSize])
                    return withRandomState(MLXRandom.RandomState(seed: UInt64(samplerId))) {
                        if samplerId % 2 == 0 {
                            return categorical(logits).item(Int.self)
                        } else {
                            return logits.argMax(axis: -1).item(Int.self)
                        }
                    }
                }
            }

            for try await result in group {
                samplerResults.append(result)
            }

            return samplerResults
        }

        XCTAssertEqual(results.count, numSamplers)

        for result in results {
            XCTAssertGreaterThanOrEqual(result, 0)
            XCTAssertLessThan(result, vocabSize)
        }
    }

    func testRandomStateIsolation() async throws {
        // the logit sampler will not use shared random state
        let numSamplers = 5
        let samplesPerTask = 10

        let allResults = try await withThrowingTaskGroup(of: [Int].self) { group in
            var results: [[Int]] = []

            for samplerId in 0 ..< numSamplers {
                group.addTask {
                    let logits = MLXArray.ones([1, 50])
                    var taskResults: [Int] = []
                    let sampler = CategoricalSampler(temperature: 1.0)

                    for sampleId in 0 ..< samplesPerTask {
                        let token = withRandomState(
                            MLXRandom.RandomState(seed: UInt64(samplerId * 1000 + sampleId))
                        ) {
                            return sampler.sample(logits: logits)
                        }
                        taskResults.append(token.item(Int.self))
                    }

                    return taskResults
                }
            }

            for try await result in group {
                results.append(result)
            }

            return results
        }

        XCTAssertEqual(allResults.count, numSamplers)

        for samplerResults in allResults {
            XCTAssertEqual(samplerResults.count, samplesPerTask)
        }

        let uniqueSequences = Set(allResults.map { $0.description })
        XCTAssertGreaterThan(uniqueSequences.count, 0)
    }
}
