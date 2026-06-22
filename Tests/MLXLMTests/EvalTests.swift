// Copyright © 2025 Apple Inc.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import MLXOptimizers
import XCTest

public class EvalTests: XCTestCase {

    private enum TestKVCacheMode: String {
        case fp16
        case affine4
        case varianceNormalized4Value2

        var parameters: GenerateParameters {
            switch self {
            case .fp16:
                GenerateParameters(temperature: 0)
            case .affine4:
                GenerateParameters(
                    kvBits: 4,
                    kvGroupSize: 32,
                    quantizedKVStart: 0,
                    kvQuantizationStrategy: .affine,
                    temperature: 0)
            case .varianceNormalized4Value2:
                GenerateParameters(
                    kvBits: 4,
                    kvGroupSize: 32,
                    quantizedKVStart: 0,
                    kvQuantizationStrategy: .varianceNormalized,
                    kvValueBits: 2,
                    kvTileSize: 32,
                    temperature: 0)
            }
        }
    }

    private struct DecodeResult {
        var tokens: [Int]
        var cacheBytes: Int
        var elapsed: TimeInterval

        var tokensPerSecond: Double {
            Double(tokens.count) / elapsed
        }
    }

    private func makeQualityGateLlamaModel(seed: UInt64 = 11) -> LlamaModel {
        MLXRandom.seed(seed)
        let config = LlamaConfiguration(
            hiddenSize: 64, hiddenLayers: 2, intermediateSize: 128, attentionHeads: 2,
            rmsNormEps: 0.00001, vocabularySize: 100, kvHeads: 1)
        let model = LlamaModel(config)
        eval(model)
        return model
    }

    private func applyDynamicKVQuantization(cache: inout [KVCache], mode: TestKVCacheMode) {
        let parameters = mode.parameters
        maybeQuantizeKVCache(
            cache: &cache,
            kvBits: parameters.kvBits,
            kvGroupSize: parameters.kvGroupSize,
            quantizedKVStart: parameters.quantizedKVStart,
            strategy: parameters.kvQuantizationStrategy,
            kvValueBits: parameters.kvValueBits,
            kvTileSize: parameters.kvTileSize)
    }

    private func cacheBytes(_ cache: [KVCache]) -> Int {
        let arrays = cache.flatMap { $0.state }
        if !arrays.isEmpty {
            eval(arrays)
        }
        return arrays.reduce(0) { $0 + $1.nbytes }
    }

    private func nextToken(
        model: LlamaModel,
        token: Int,
        cache: [KVCache]
    ) -> Int {
        let input = MLXArray([token])[.newAxis, .ellipsis]
        let logits = model.callAsFunction(input, cache: cache)
        let next = argMax(logits[0..., -1, 0...], axis: -1)
        eval(next)
        return next.item(Int.self)
    }

    private func generateTokens(
        model: LlamaModel,
        prompt: [Int],
        maxTokens: Int,
        mode: TestKVCacheMode
    ) -> DecodeResult {
        var cache = model.newCache(parameters: nil)
        let start = Date.timeIntervalSinceReferenceDate
        let promptLogits = model.callAsFunction(MLXArray(prompt)[.newAxis, .ellipsis], cache: cache)
        eval(promptLogits)
        applyDynamicKVQuantization(cache: &cache, mode: mode)

        var previous = argMax(promptLogits[0..., -1, 0...], axis: -1).item(Int.self)
        var generated: [Int] = []
        generated.reserveCapacity(maxTokens)

        for _ in 0 ..< maxTokens {
            generated.append(previous)
            previous = nextToken(model: model, token: previous, cache: cache)
            applyDynamicKVQuantization(cache: &cache, mode: mode)
        }

        return DecodeResult(
            tokens: generated,
            cacheBytes: cacheBytes(cache),
            elapsed: Date.timeIntervalSinceReferenceDate - start)
    }

    private func autoregressiveCrossEntropy(
        model: LlamaModel,
        sequence: [Int],
        mode: TestKVCacheMode
    ) -> Float {
        precondition(sequence.count > 1)
        var cache = model.newCache(parameters: nil)
        var loss: Float = 0

        for i in 0 ..< sequence.count - 1 {
            let input = MLXArray([sequence[i]])[.newAxis, .ellipsis]
            let target = MLXArray([sequence[i + 1]])[.newAxis, .ellipsis]
            let logits = model.callAsFunction(input, cache: cache)
            let tokenLoss = mean(crossEntropy(logits: logits, targets: target))
            eval(tokenLoss)
            loss += tokenLoss.item(Float.self)
            applyDynamicKVQuantization(cache: &cache, mode: mode)
        }

        return loss / Float(sequence.count - 1)
    }

    private func tokenAgreement(_ lhs: [Int], _ rhs: [Int]) -> Float {
        precondition(lhs.count == rhs.count)
        let matches = zip(lhs, rhs).filter { $0 == $1 }.count
        return Float(matches) / Float(lhs.count)
    }

    private func firstTokenMismatch(_ lhs: [Int], _ rhs: [Int]) -> String {
        for (index, pair) in zip(lhs, rhs).enumerated() where pair.0 != pair.1 {
            return "index \(index): \(pair.0) != \(pair.1)"
        }
        return "none"
    }

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

    func testLlamaVarianceNormalizedKVCacheGenerationPath() throws {
        let config = LlamaConfiguration(
            hiddenSize: 64, hiddenLayers: 2, intermediateSize: 128, attentionHeads: 2,
            rmsNormEps: 0.00001, vocabularySize: 100, kvHeads: 1)
        let model = LlamaModel(config)
        let parameters = GenerateParameters(
            kvBits: 4,
            quantizedKVStart: 0,
            kvQuantizationStrategy: .varianceNormalized,
            kvValueBits: 4,
            kvTileSize: 32,
            temperature: 0)

        var cache = model.newCache(parameters: parameters)
        XCTAssertTrue(cache.allSatisfy { $0 is KVCacheSimple })

        let prompt = MLXArray([1, 2, 3, 4])[.newAxis, .ellipsis]
        let output = model.callAsFunction(prompt, cache: cache)
        eval(output)
        XCTAssertEqual(output.shape, [1, 4, 100])

        maybeQuantizeKVCache(
            cache: &cache,
            kvBits: parameters.kvBits,
            kvGroupSize: parameters.kvGroupSize,
            quantizedKVStart: parameters.quantizedKVStart,
            strategy: parameters.kvQuantizationStrategy,
            kvValueBits: parameters.kvValueBits,
            kvTileSize: parameters.kvTileSize)
        XCTAssertTrue(cache.allSatisfy { $0 is VarianceNormalizedKVCache })

        let next = MLXArray([5])[.newAxis, .ellipsis]
        let nextOutput = model.callAsFunction(next, cache: cache)
        eval(nextOutput)
        XCTAssertEqual(nextOutput.shape, [1, 1, 100])
    }

    func testLlamaVarianceNormalizedKVCacheAutoregressivePerplexityTracksBaselines() throws {
        let model = makeQualityGateLlamaModel()
        let sequence = (0 ..< 128).map { (($0 * 37 + 11) % 99) + 1 }

        let fp16Loss = autoregressiveCrossEntropy(model: model, sequence: sequence, mode: .fp16)
        let affineLoss = autoregressiveCrossEntropy(
            model: model, sequence: sequence, mode: .affine4)
        let varianceNormalizedLoss = autoregressiveCrossEntropy(
            model: model, sequence: sequence, mode: .varianceNormalized4Value2)

        let affineDrift = abs(affineLoss - fp16Loss)
        let varianceNormalizedDrift = abs(varianceNormalizedLoss - fp16Loss)
        print(
            "KV perplexity gate: fp16=\(fp16Loss), affine4=\(affineLoss), "
                + "varianceNormalized4/value2=\(varianceNormalizedLoss)"
        )

        XCTAssertLessThan(affineDrift, 0.20)
        XCTAssertLessThan(varianceNormalizedDrift, 0.20)
    }

    func testLlamaVarianceNormalizedKVCacheGreedyDecodeTracksFP16ForOneHundredTokens() throws {
        let model = makeQualityGateLlamaModel()
        let prompt = [1, 7, 13, 29, 31, 43, 59, 61]
        let fp16 = generateTokens(model: model, prompt: prompt, maxTokens: 100, mode: .fp16)
        let affine = generateTokens(model: model, prompt: prompt, maxTokens: 100, mode: .affine4)
        let varianceNormalized = generateTokens(
            model: model, prompt: prompt, maxTokens: 100, mode: .varianceNormalized4Value2)

        let affineAgreement = tokenAgreement(affine.tokens, fp16.tokens)
        let varianceNormalizedAgreement = tokenAgreement(varianceNormalized.tokens, fp16.tokens)
        print(
            "KV token agreement: affine4=\(affineAgreement) "
                + "(first mismatch: \(firstTokenMismatch(affine.tokens, fp16.tokens))), "
                + "varianceNormalized4/value2=\(varianceNormalizedAgreement) "
                + "(first mismatch: \(firstTokenMismatch(varianceNormalized.tokens, fp16.tokens)))"
        )

        XCTAssertGreaterThanOrEqual(affineAgreement, 0.95)
        XCTAssertGreaterThanOrEqual(varianceNormalizedAgreement, 0.95)
    }

    func testLlamaVarianceNormalizedKVCacheAppleSiliconDecodePerformanceSmoke() throws {
        #if os(macOS) && arch(arm64)
            let model = makeQualityGateLlamaModel()
            let prompt = [3, 5, 8, 13, 21, 34, 55, 89]

            _ = generateTokens(model: model, prompt: prompt, maxTokens: 16, mode: .fp16)
            _ = generateTokens(model: model, prompt: prompt, maxTokens: 16, mode: .affine4)
            _ = generateTokens(
                model: model, prompt: prompt, maxTokens: 16, mode: .varianceNormalized4Value2)

            let fp16 = generateTokens(model: model, prompt: prompt, maxTokens: 96, mode: .fp16)
            let affine = generateTokens(model: model, prompt: prompt, maxTokens: 96, mode: .affine4)
            let varianceNormalized = generateTokens(
                model: model, prompt: prompt, maxTokens: 96, mode: .varianceNormalized4Value2)

            print(
                "KV decode smoke: fp16=\(fp16.tokensPerSecond) tok/s "
                    + "(\(fp16.cacheBytes) bytes), affine4=\(affine.tokensPerSecond) tok/s "
                    + "(\(affine.cacheBytes) bytes), varianceNormalized4/value2="
                    + "\(varianceNormalized.tokensPerSecond) tok/s "
                    + "(\(varianceNormalized.cacheBytes) bytes)"
            )

            XCTAssertGreaterThan(fp16.tokensPerSecond, 0)
            XCTAssertGreaterThan(affine.tokensPerSecond, 0)
            XCTAssertGreaterThan(varianceNormalized.tokensPerSecond, 0)
            XCTAssertLessThan(affine.cacheBytes, fp16.cacheBytes)
            XCTAssertLessThan(varianceNormalized.cacheBytes, fp16.cacheBytes)

            // This is a smoke threshold for Apple Silicon CI variance, not a benchmark claim.
            XCTAssertLessThan(varianceNormalized.elapsed, fp16.elapsed * 10)
            XCTAssertLessThan(varianceNormalized.elapsed, affine.elapsed * 10)
        #else
            throw XCTSkip("Apple Silicon decode performance smoke test requires arm64 macOS.")
        #endif
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
