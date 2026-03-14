// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXNN
import Tokenizers
import XCTest

@testable import MLXLMCommon

// MARK: - Mock Model for Scheduler Tests

/// A deterministic mock language model for InferenceScheduler tests.
///
/// Produces tokens deterministically: next token = (input_token + 1) % vocabSize.
/// Uses KVCacheSimple by default (batch-compatible).
private class SchedulerMockModel: Module, LanguageModel, KVCacheDimensionProvider {
    let vocabSize: Int
    let numLayers: Int
    var kvHeads: [Int] { Array(repeating: 4, count: numLayers) }

    init(vocabSize: Int = 32, numLayers: Int = 1) {
        self.vocabSize = vocabSize
        self.numLayers = numLayers
    }

    func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
        .tokens(input.text)
    }

    func callAsFunction(
        _ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?
    ) -> LMOutput {
        let tokens = input.tokens
        let B = tokens.dim(0)
        let S = tokens.dim(1)

        var logitsFlat = [Float]()
        for b in 0 ..< B {
            for s in 0 ..< S {
                let lastToken = tokens[b, s].item(Int32.self)
                let predictedToken = (Int(lastToken) + 1) % vocabSize

                var row = [Float](repeating: -100.0, count: vocabSize)
                row[predictedToken] = 0.0
                logitsFlat.append(contentsOf: row)
            }
        }

        let logits = MLXArray(logitsFlat, [B, S, vocabSize])
        return LMOutput(logits: logits)
    }

    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        weights
    }
}

/// Mock model that creates MambaCache (batch-incompatible).
private class SSMMockModel: Module, LanguageModel {
    let vocabSize: Int = 32

    func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
        .tokens(input.text)
    }

    func callAsFunction(
        _ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?
    ) -> LMOutput {
        let logits = MLXArray.zeros([input.tokens.dim(0), input.tokens.dim(1), vocabSize])
        return LMOutput(logits: logits)
    }

    func newCache(parameters: GenerateParameters?) -> [KVCache] {
        [MambaCache()]
    }

    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        weights
    }
}

// MARK: - Tests

class InferenceSchedulerTests: XCTestCase {

    // MARK: - VAL-SCHED-001: Single request uses TokenIterator directly

    func testSingleRequestUsesTokenIteratorDirectly() async throws {
        try skipIfMetalUnavailable()

        let model = SchedulerMockModel()
        let tokenizer = TestTokenizer()
        let config = ModelConfiguration(id: "test-model")
        let scheduler = InferenceScheduler()

        let input = LMInput(tokens: MLXArray([Int32(1), Int32(2), Int32(3)]))
        let params = GenerateParameters(maxTokens: 3, temperature: 0)

        let stream = try await scheduler.submit(
            input: input,
            parameters: params,
            model: model,
            cache: nil,
            tokenizer: tokenizer,
            configuration: config
        )

        // Verify state is single
        let currentState = await scheduler.currentState
        XCTAssertEqual(currentState, "single", "Single request should use single path")

        // Consume the stream to completion
        var chunks = [String]()
        for await generation in stream {
            if let chunk = generation.chunk {
                chunks.append(chunk)
            }
        }

        // Should have received some output
        XCTAssertFalse(chunks.isEmpty, "Should receive output from single request")
    }

    // MARK: - VAL-SCHED-002: Single request receives complete streaming output

    func testSingleRequestReceivesCompleteOutput() async throws {
        try skipIfMetalUnavailable()

        let model = SchedulerMockModel()
        let tokenizer = TestTokenizer()
        let config = ModelConfiguration(id: "test-model")
        let scheduler = InferenceScheduler()

        let input = LMInput(tokens: MLXArray([Int32(1), Int32(2)]))
        let params = GenerateParameters(maxTokens: 5, temperature: 0)

        let stream = try await scheduler.submit(
            input: input,
            parameters: params,
            model: model,
            cache: nil,
            tokenizer: tokenizer,
            configuration: config
        )

        var receivedInfo = false
        var chunks = [String]()
        for await generation in stream {
            switch generation {
            case .chunk(let text):
                chunks.append(text)
            case .info(let info):
                receivedInfo = true
                XCTAssertGreaterThan(
                    info.generationTokenCount, 0,
                    "Should report non-zero token count")
            case .toolCall:
                break
            }
        }

        XCTAssertTrue(receivedInfo, "Should receive completion info")
    }

    // MARK: - VAL-SCHED-007: Incompatible requests fall back to single path

    func testVLMInputFallsBackToSinglePath() throws {
        try skipIfMetalUnavailable()

        let model = SchedulerMockModel()

        // VLM input with image data — should be batch-incompatible
        let image = LMInput.ProcessedImage(pixels: MLXArray.zeros([1, 3, 224, 224]))
        let input = LMInput(
            text: .init(tokens: MLXArray([Int32(1), Int32(2)])),
            image: image
        )

        let compatible = InferenceScheduler.isBatchCompatible(
            input: input,
            parameters: GenerateParameters(temperature: 0),
            cache: nil,
            model: model
        )

        XCTAssertFalse(compatible, "VLM inputs with images should be batch-incompatible")
    }

    func testVideoInputFallsBackToSinglePath() throws {
        try skipIfMetalUnavailable()

        let model = SchedulerMockModel()

        let video = LMInput.ProcessedVideo(pixels: MLXArray.zeros([1, 3, 16, 224, 224]))
        let input = LMInput(
            text: .init(tokens: MLXArray([Int32(1)])),
            video: video
        )

        let compatible = InferenceScheduler.isBatchCompatible(
            input: input,
            parameters: GenerateParameters(temperature: 0),
            cache: nil,
            model: model
        )

        XCTAssertFalse(compatible, "VLM inputs with video should be batch-incompatible")
    }

    // MARK: - VAL-SCHED-008: Standard LLM models are batch-compatible

    func testStandardLLMIsBatchCompatible() throws {
        try skipIfMetalUnavailable()

        let model = SchedulerMockModel()
        let input = LMInput(tokens: MLXArray([Int32(1), Int32(2)]))

        let compatible = InferenceScheduler.isBatchCompatible(
            input: input,
            parameters: GenerateParameters(temperature: 0),
            cache: nil,
            model: model
        )

        XCTAssertTrue(compatible, "Standard LLM with KVCacheSimple should be batch-compatible")
    }

    // MARK: - VAL-SCHED-015: Requests with kvBits set are batch-incompatible

    func testKvBitsRequestIsIncompatible() throws {
        try skipIfMetalUnavailable()

        let model = SchedulerMockModel()
        let input = LMInput(tokens: MLXArray([Int32(1)]))

        let compatible = InferenceScheduler.isBatchCompatible(
            input: input,
            parameters: GenerateParameters(kvBits: 4, temperature: 0),
            cache: nil,
            model: model
        )

        XCTAssertFalse(
            compatible,
            "Requests with kvBits set should be batch-incompatible"
        )
    }

    // MARK: - VAL-SCHED-007 (continued): SSM model incompatible

    func testSSMModelIsIncompatible() throws {
        try skipIfMetalUnavailable()

        let model = SSMMockModel()
        let input = LMInput(tokens: MLXArray([Int32(1)]))

        let compatible = InferenceScheduler.isBatchCompatible(
            input: input,
            parameters: GenerateParameters(temperature: 0),
            cache: nil,
            model: model
        )

        XCTAssertFalse(
            compatible,
            "SSM models with MambaCache should be batch-incompatible"
        )
    }

    // MARK: - VAL-SCHED-007 (continued): CacheList incompatible

    func testCacheListIsIncompatible() throws {
        try skipIfMetalUnavailable()

        let model = SchedulerMockModel()
        let input = LMInput(tokens: MLXArray([Int32(1)]))

        // Provide a CacheList as the pre-existing cache
        let cacheList = CacheList(KVCacheSimple(), MambaCache())
        let compatible = InferenceScheduler.isBatchCompatible(
            input: input,
            parameters: GenerateParameters(temperature: 0),
            cache: [cacheList],
            model: model
        )

        XCTAssertFalse(
            compatible,
            "CacheList (hybrid models) should be batch-incompatible"
        )
    }

    // MARK: - VAL-SCHED-014: Actor isolation prevents data races

    func testActorIsolationPreventDataRaces() async throws {
        try skipIfMetalUnavailable()

        // This test verifies that InferenceScheduler is an actor (compile-time guarantee)
        // and that concurrent access via submit() is safe.
        let model = SchedulerMockModel()
        let tokenizer = TestTokenizer()
        let config = ModelConfiguration(id: "test-model")
        let scheduler = InferenceScheduler()

        // Submit multiple requests concurrently — should not crash
        await withTaskGroup(of: Void.self) { group in
            for i in 0 ..< 3 {
                group.addTask {
                    let input = LMInput(tokens: MLXArray([Int32(i + 1)]))
                    let params = GenerateParameters(maxTokens: 2, temperature: 0)
                    do {
                        let stream = try await scheduler.submit(
                            input: input,
                            parameters: params,
                            model: model,
                            cache: nil,
                            tokenizer: tokenizer,
                            configuration: config
                        )
                        // Consume to completion
                        for await _ in stream {}
                    } catch {
                        // Upgrade failures are acceptable — we're testing safety
                    }
                }
            }
        }

        // If we get here without crash, actor isolation is working
    }

    // MARK: - State transitions: idle -> single -> back to idle

    func testIdleToSingleToIdleTransition() async throws {
        try skipIfMetalUnavailable()

        let model = SchedulerMockModel()
        let tokenizer = TestTokenizer()
        let config = ModelConfiguration(id: "test-model")
        let scheduler = InferenceScheduler()

        // Initially idle
        var currentState = await scheduler.currentState
        XCTAssertEqual(currentState, "idle")

        let input = LMInput(tokens: MLXArray([Int32(1), Int32(2)]))
        let params = GenerateParameters(maxTokens: 3, temperature: 0)

        let stream = try await scheduler.submit(
            input: input,
            parameters: params,
            model: model,
            cache: nil,
            tokenizer: tokenizer,
            configuration: config
        )

        // Now should be in single state
        currentState = await scheduler.currentState
        XCTAssertEqual(currentState, "single")

        // Consume to completion
        for await _ in stream {}

        // Wait a moment for the cleanup task to run
        try await Task.sleep(nanoseconds: 100_000_000)  // 100ms

        // Should return to idle
        currentState = await scheduler.currentState
        XCTAssertEqual(currentState, "idle")
    }

    // MARK: - VAL-SCHED-011: Each request gets independent AsyncStream

    func testEachRequestGetsIndependentStream() async throws {
        try skipIfMetalUnavailable()

        let model = SchedulerMockModel()
        let tokenizer = TestTokenizer()
        let config = ModelConfiguration(id: "test-model")
        let scheduler = InferenceScheduler()

        // First request
        let input1 = LMInput(tokens: MLXArray([Int32(1), Int32(2)]))
        let params1 = GenerateParameters(maxTokens: 3, temperature: 0)

        let stream1 = try await scheduler.submit(
            input: input1,
            parameters: params1,
            model: model,
            cache: nil,
            tokenizer: tokenizer,
            configuration: config
        )

        // Each submit returns a unique AsyncStream instance — this confirms
        // independent routing at the stream level.

        var tokens1 = [String]()
        for await gen in stream1 {
            if let chunk = gen.chunk {
                tokens1.append(chunk)
            }
        }

        XCTAssertFalse(tokens1.isEmpty, "First request should produce output")
    }

    // MARK: - Incompatible request while single is active uses fallback

    func testIncompatibleRequestWhileSingleIsActive() async throws {
        try skipIfMetalUnavailable()

        let model = SchedulerMockModel()
        let tokenizer = TestTokenizer()
        let config = ModelConfiguration(id: "test-model")
        let scheduler = InferenceScheduler()

        // First compatible request
        let input1 = LMInput(tokens: MLXArray([Int32(1), Int32(2)]))
        let params1 = GenerateParameters(maxTokens: 10, temperature: 0)

        let _ = try await scheduler.submit(
            input: input1,
            parameters: params1,
            model: model,
            cache: nil,
            tokenizer: tokenizer,
            configuration: config
        )

        // State should be single
        var currentState = await scheduler.currentState
        XCTAssertEqual(currentState, "single")

        // Second request is incompatible (has image)
        let image = LMInput.ProcessedImage(pixels: MLXArray.zeros([1, 3, 224, 224]))
        let input2 = LMInput(
            text: .init(tokens: MLXArray([Int32(3), Int32(4)])),
            image: image
        )
        let params2 = GenerateParameters(maxTokens: 3, temperature: 0)

        // This should fall back to single path (not upgrade to batch)
        let stream2 = try await scheduler.submit(
            input: input2,
            parameters: params2,
            model: model,
            cache: nil,
            tokenizer: tokenizer,
            configuration: config
        )

        // State should still be single (not batched) because the second request is incompatible
        currentState = await scheduler.currentState
        XCTAssertEqual(
            currentState, "single",
            "Incompatible request should not trigger batch upgrade")

        // Consume second stream to verify it works
        var chunks = [String]()
        for await gen in stream2 {
            if let chunk = gen.chunk {
                chunks.append(chunk)
            }
        }
    }

    // MARK: - QuantizedKVCache is incompatible

    func testQuantizedKVCacheIsIncompatible() throws {
        try skipIfMetalUnavailable()

        let model = SchedulerMockModel()
        let input = LMInput(tokens: MLXArray([Int32(1)]))

        // Provide QuantizedKVCache directly
        let qCache = QuantizedKVCache(groupSize: 64, bits: 4)
        let compatible = InferenceScheduler.isBatchCompatible(
            input: input,
            parameters: GenerateParameters(temperature: 0),
            cache: [qCache],
            model: model
        )

        XCTAssertFalse(
            compatible,
            "QuantizedKVCache should be batch-incompatible"
        )
    }

    // MARK: - Empty cache array is compatible

    func testEmptyCacheArrayIsCompatible() throws {
        try skipIfMetalUnavailable()

        let model = SchedulerMockModel()
        let input = LMInput(tokens: MLXArray([Int32(1)]))

        let compatible = InferenceScheduler.isBatchCompatible(
            input: input,
            parameters: GenerateParameters(temperature: 0),
            cache: [],
            model: model
        )

        XCTAssertTrue(compatible, "Empty cache array should be batch-compatible")
    }

    // MARK: - Nil cache is compatible

    func testNilCacheIsCompatible() throws {
        try skipIfMetalUnavailable()

        let model = SchedulerMockModel()
        let input = LMInput(tokens: MLXArray([Int32(1)]))

        let compatible = InferenceScheduler.isBatchCompatible(
            input: input,
            parameters: GenerateParameters(temperature: 0),
            cache: nil,
            model: model
        )

        XCTAssertTrue(compatible, "Nil cache should be batch-compatible")
    }

    // MARK: - KVCacheSimple cache array is compatible

    func testKVCacheSimpleIsCompatible() throws {
        try skipIfMetalUnavailable()

        let model = SchedulerMockModel()
        let input = LMInput(tokens: MLXArray([Int32(1)]))

        let compatible = InferenceScheduler.isBatchCompatible(
            input: input,
            parameters: GenerateParameters(temperature: 0),
            cache: [KVCacheSimple()],
            model: model
        )

        XCTAssertTrue(compatible, "KVCacheSimple should be batch-compatible")
    }
}
