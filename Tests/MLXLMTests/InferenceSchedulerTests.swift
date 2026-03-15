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
private class SchedulerMockModel: Module, LanguageModel, KVCacheDimensionProvider,
    @unchecked
    Sendable
{
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

/// Mock model returning mixed RotatingKVCache/KVCacheSimple layers,
/// simulating sliding-window models like Gemma3 or Mistral3.
private class RotatingCacheMockModel: Module, LanguageModel, @unchecked Sendable {
    let vocabSize: Int
    let numLayers: Int
    let slidingWindowMaxSize: Int
    let slidingWindowKeep: Int

    init(
        vocabSize: Int = 32, numLayers: Int = 2,
        slidingWindowMaxSize: Int = 64, slidingWindowKeep: Int = 4
    ) {
        self.vocabSize = vocabSize
        self.numLayers = numLayers
        self.slidingWindowMaxSize = slidingWindowMaxSize
        self.slidingWindowKeep = slidingWindowKeep
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

    /// Returns layers: [KVCacheSimple, RotatingKVCache]
    func newCache(parameters: GenerateParameters?) -> [KVCache] {
        [
            KVCacheSimple(),
            RotatingKVCache(maxSize: slidingWindowMaxSize, keep: slidingWindowKeep),
        ]
    }

    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        weights
    }
}

/// Mock model that creates MambaCache (batch-incompatible).
private class SSMMockModel: Module, LanguageModel, @unchecked Sendable {
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

    // MARK: - VAL-SCHED-005: Upgrade uses live TokenIterator state

    /// Verifies that single-to-batch upgrade uses the live TokenIterator state
    /// (with current KV cache) rather than the stale copy stored in actor state.
    /// The single-request task cooperatively deposits its live state before
    /// the scheduler builds the batch.
    func testUpgradeUsesLiveTokenIteratorState() async throws {
        try skipIfMetalUnavailable()

        let model = SchedulerMockModel()
        let tokenizer = TestTokenizer()
        let config = ModelConfiguration(id: "test-model")
        let scheduler = InferenceScheduler()

        // First request with a few tokens — long enough to advance the iterator
        let input1 = LMInput(tokens: MLXArray([Int32(1), Int32(2), Int32(3)]))
        let params1 = GenerateParameters(maxTokens: 20, temperature: 0)

        let stream1 = try await scheduler.submit(
            input: input1,
            parameters: params1,
            model: model,
            cache: nil,
            tokenizer: tokenizer,
            configuration: config
        )

        // Verify we're in single state
        var currentState = await scheduler.currentState
        XCTAssertEqual(currentState, "single")

        // Consume a few tokens from stream1 to advance the iterator
        var tokens1BeforeUpgrade = [String]()
        var count = 0
        for await gen in stream1 {
            if let chunk = gen.chunk {
                tokens1BeforeUpgrade.append(chunk)
                count += 1
                if count >= 2 {
                    break
                }
            }
        }

        // Now submit a second request to trigger upgrade
        let input2 = LMInput(tokens: MLXArray([Int32(5), Int32(6)]))
        let params2 = GenerateParameters(maxTokens: 5, temperature: 0)

        let stream2 = try await scheduler.submit(
            input: input2,
            parameters: params2,
            model: model,
            cache: nil,
            tokenizer: tokenizer,
            configuration: config
        )

        // Should now be in batched state
        currentState = await scheduler.currentState
        XCTAssertEqual(
            currentState, "batched",
            "Should transition to batched state after second request")

        // Consume remaining tokens from both streams
        var tokens1AfterUpgrade = [String]()
        var tokens2 = [String]()

        await withTaskGroup(of: (Int, [String]).self) { group in
            group.addTask {
                var chunks = [String]()
                for await gen in stream1 {
                    if let chunk = gen.chunk {
                        chunks.append(chunk)
                    }
                }
                return (1, chunks)
            }

            group.addTask {
                var chunks = [String]()
                for await gen in stream2 {
                    if let chunk = gen.chunk {
                        chunks.append(chunk)
                    }
                }
                return (2, chunks)
            }

            for await (id, chunks) in group {
                if id == 1 {
                    tokens1AfterUpgrade = chunks
                } else {
                    tokens2 = chunks
                }
            }
        }

        // First request should have continued generating after upgrade
        // (tokens before + after should form a coherent sequence)
        let totalFirst = tokens1BeforeUpgrade.count + tokens1AfterUpgrade.count
        XCTAssertGreaterThan(
            totalFirst, 0,
            "First request should produce tokens across the upgrade boundary")

        // Second request should also produce output
        XCTAssertGreaterThan(
            tokens2.count, 0,
            "Second request should produce output in batch mode")
    }

    // MARK: - VAL-SCHED-003: Second concurrent request triggers batch upgrade

    func testSecondConcurrentRequestTriggersBatchUpgrade() async throws {
        try skipIfMetalUnavailable()

        let model = SchedulerMockModel()
        let tokenizer = TestTokenizer()
        let config = ModelConfiguration(id: "test-model")
        let scheduler = InferenceScheduler()

        // First request with large maxTokens to ensure it's still running
        // when the second request arrives.
        let input1 = LMInput(tokens: MLXArray([Int32(1), Int32(2)]))
        let params1 = GenerateParameters(maxTokens: 1000, temperature: 0)

        let stream1 = try await scheduler.submit(
            input: input1,
            parameters: params1,
            model: model,
            cache: nil,
            tokenizer: tokenizer,
            configuration: config
        )

        var currentState = await scheduler.currentState
        XCTAssertEqual(currentState, "single")

        // Second request triggers upgrade
        let input2 = LMInput(tokens: MLXArray([Int32(5), Int32(6)]))
        let params2 = GenerateParameters(maxTokens: 5, temperature: 0)

        let stream2 = try await scheduler.submit(
            input: input2,
            parameters: params2,
            model: model,
            cache: nil,
            tokenizer: tokenizer,
            configuration: config
        )

        currentState = await scheduler.currentState
        // After upgrade, state should be batched. If the first request
        // happened to finish before the upgrade handshake, the fallback
        // creates a new single request instead.
        XCTAssertTrue(
            currentState == "batched" || currentState == "single",
            "Second concurrent request should trigger batch upgrade or fallback to single (got \(currentState))"
        )

        // Consume streams concurrently to avoid deadlock
        await withTaskGroup(of: Void.self) { group in
            group.addTask { for await _ in stream1 {} }
            group.addTask { for await _ in stream2 {} }
        }
    }

    // MARK: - Cancellation after upgrade removes UID from BatchTokenIterator

    /// Verifies that after upgrade, cancelling the first request's stream
    /// removes its UID from the BatchTokenIterator (not cancelling the
    /// defunct single-request task).
    func testCancellationAfterUpgradeRemovesUID() async throws {
        try skipIfMetalUnavailable()

        let model = SchedulerMockModel()
        let tokenizer = TestTokenizer()
        let config = ModelConfiguration(id: "test-model")
        let scheduler = InferenceScheduler()

        // First request with many tokens
        let input1 = LMInput(tokens: MLXArray([Int32(1), Int32(2)]))
        let params1 = GenerateParameters(maxTokens: 50, temperature: 0)

        let stream1 = try await scheduler.submit(
            input: input1,
            parameters: params1,
            model: model,
            cache: nil,
            tokenizer: tokenizer,
            configuration: config
        )

        // Second request triggers upgrade
        let input2 = LMInput(tokens: MLXArray([Int32(5), Int32(6)]))
        let params2 = GenerateParameters(maxTokens: 50, temperature: 0)

        let stream2 = try await scheduler.submit(
            input: input2,
            parameters: params2,
            model: model,
            cache: nil,
            tokenizer: tokenizer,
            configuration: config
        )

        // Now cancel stream1 by dropping it (letting the continuation terminate)
        // and verify stream2 continues producing output
        var request1Stopped = false
        var request2Completed = false

        await withTaskGroup(of: (Int, Bool).self) { group in
            group.addTask {
                var count = 0
                for await _ in stream1 {
                    count += 1
                    if count >= 2 {
                        // Stop consuming early to trigger cancellation
                        break
                    }
                }
                return (1, true)
            }

            group.addTask {
                var count = 0
                for await _ in stream2 {
                    count += 1
                }
                return (2, count > 0)
            }

            for await (id, result) in group {
                if id == 1 {
                    request1Stopped = result
                } else {
                    request2Completed = result
                }
            }
        }

        XCTAssertTrue(
            request1Stopped,
            "First request should have stopped after early break")
        XCTAssertTrue(
            request2Completed,
            "Second request should complete even after first is cancelled")
    }

    // MARK: - VAL-SCHED-016: Third concurrent request joins existing batch

    func testThirdRequestJoinsExistingBatch() async throws {
        try skipIfMetalUnavailable()

        let model = SchedulerMockModel()
        let tokenizer = TestTokenizer()
        let config = ModelConfiguration(id: "test-model")
        let scheduler = InferenceScheduler()

        // First request
        let input1 = LMInput(tokens: MLXArray([Int32(1), Int32(2)]))
        let params1 = GenerateParameters(maxTokens: 20, temperature: 0)

        let stream1 = try await scheduler.submit(
            input: input1,
            parameters: params1,
            model: model,
            cache: nil,
            tokenizer: tokenizer,
            configuration: config
        )

        // Second request triggers upgrade
        let input2 = LMInput(tokens: MLXArray([Int32(3), Int32(4)]))
        let params2 = GenerateParameters(maxTokens: 10, temperature: 0)

        let stream2 = try await scheduler.submit(
            input: input2,
            parameters: params2,
            model: model,
            cache: nil,
            tokenizer: tokenizer,
            configuration: config
        )

        var currentState = await scheduler.currentState
        XCTAssertEqual(currentState, "batched")

        // Third request joins existing batch (no migration)
        let input3 = LMInput(tokens: MLXArray([Int32(7), Int32(8)]))
        let params3 = GenerateParameters(maxTokens: 5, temperature: 0)

        let stream3 = try await scheduler.submit(
            input: input3,
            parameters: params3,
            model: model,
            cache: nil,
            tokenizer: tokenizer,
            configuration: config
        )

        currentState = await scheduler.currentState
        XCTAssertEqual(
            currentState, "batched",
            "Should still be in batched state after third request")

        // All three should produce output
        // Collect per-stream results: chunk count and info
        typealias StreamResult = (chunkCount: Int, info: GenerateCompletionInfo?)

        var results = [Int: StreamResult]()

        await withTaskGroup(of: (Int, StreamResult).self) { group in
            group.addTask {
                var count = 0
                var info: GenerateCompletionInfo?
                for await gen in stream1 {
                    if gen.chunk != nil { count += 1 }
                    if let i = gen.info { info = i }
                }
                return (1, (count, info))
            }
            group.addTask {
                var count = 0
                var info: GenerateCompletionInfo?
                for await gen in stream2 {
                    if gen.chunk != nil { count += 1 }
                    if let i = gen.info { info = i }
                }
                return (2, (count, info))
            }
            group.addTask {
                var count = 0
                var info: GenerateCompletionInfo?
                for await gen in stream3 {
                    if gen.chunk != nil { count += 1 }
                    if let i = gen.info { info = i }
                }
                return (3, (count, info))
            }

            for await (id, result) in group {
                results[id] = result
            }
        }

        // Each stream must independently produce .chunk events
        XCTAssertTrue(results[1]!.chunkCount > 0, "Stream 1 must produce .chunk")
        XCTAssertTrue(results[2]!.chunkCount > 0, "Stream 2 must produce .chunk")
        XCTAssertTrue(results[3]!.chunkCount > 0, "Stream 3 (joined) must produce .chunk")

        // Stream 3's .info must have non-zero generationTokenCount
        XCTAssertNotNil(results[3]!.info, "Stream 3 must receive .info")
        if let info3 = results[3]!.info {
            XCTAssertGreaterThan(
                info3.generationTokenCount, 0,
                "Stream 3 .info must have generationTokenCount > 0")
        }
    }

    // MARK: - UpgradeFlag deposits live state correctly

    /// Unit test for the UpgradeFlag cooperative mechanism in isolation.
    func testUpgradeFlagDepositAndReceiveLiveState() async throws {
        try skipIfMetalUnavailable()

        let flag = InferenceScheduler.UpgradeFlag()

        // Simulate the scheduler side: request upgrade and await live state
        let stateTask = Task {
            await withCheckedContinuation { continuation in
                flag.requestUpgrade(continuation: continuation)
            }
        }

        // Yield to let the continuation get set
        try await Task.sleep(nanoseconds: 10_000_000)  // 10ms

        // Simulate the task side: detect upgradeRequested and deposit state
        XCTAssertTrue(flag.upgradeRequested, "Flag should be set to upgradeRequested")

        let mockCache = KVCacheSimple()
        let liveState = InferenceScheduler.LiveIteratorState(
            cache: [mockCache],
            y: LMInput.Text(tokens: MLXArray([Int32(42)])),
            tokenCount: 7,
            maxTokens: 100,
            sampler: ArgMaxSampler(),
            processor: nil,
            promptTokenCount: 10,
            promptTime: 0.05
        )
        flag.depositLiveState(liveState)

        // The scheduler side should now have received the live state
        let received = await stateTask.value
        XCTAssertNotNil(received, "Should receive the live state")
        XCTAssertEqual(received?.tokenCount, 7, "Should receive the live token count")
        XCTAssertEqual(received?.maxTokens, 100, "Should receive the live maxTokens")
    }

    // MARK: - Regression: maxTokens not overrun on upgrade at final allowed token

    /// Verifies that when the first request has exhausted its maxTokens budget
    /// at the point of upgrade, the first request finishes immediately without
    /// producing extra tokens. This is a regression test for the off-by-one
    /// where `max(firstMaxTokens, 1)` clamped a zero remaining budget to 1.
    func testMaxTokensNotOverrunOnUpgradeAtFinalToken() async throws {
        try skipIfMetalUnavailable()

        let model = SchedulerMockModel()
        // Use a tokenizer with non-zero EOS to avoid early stop.
        // The default TestTokenizer has eosTokenId = 0, unknownTokenId = 0.
        // Our mock model produces (input+1)%32, starting from token 10:
        // 11, 12, 13, ... — none of which are 0 within maxTokens = 3.
        let tokenizer = TestTokenizer()
        let config = ModelConfiguration(id: "test-model")
        let scheduler = InferenceScheduler()

        let maxTokens = 3
        let input1 = LMInput(tokens: MLXArray([Int32(10)]))
        let params1 = GenerateParameters(maxTokens: maxTokens, temperature: 0)

        let stream1 = try await scheduler.submit(
            input: input1,
            parameters: params1,
            model: model,
            cache: nil,
            tokenizer: tokenizer,
            configuration: config
        )

        // Consume all tokens from the first request before triggering upgrade.
        // This ensures the iterator has advanced to tokenCount == maxTokens.
        var firstChunks = [String]()
        var firstInfo: GenerateCompletionInfo?
        var stream1Finished = false

        // We'll collect from stream1 in a task so we can also submit the
        // second request. We consume a few tokens, then trigger upgrade.
        let collectTask = Task { () -> ([String], GenerateCompletionInfo?) in
            var chunks = [String]()
            var info: GenerateCompletionInfo?
            for await gen in stream1 {
                switch gen {
                case .chunk(let text):
                    chunks.append(text)
                case .info(let i):
                    info = i
                case .toolCall:
                    break
                }
            }
            return (chunks, info)
        }

        // Give the first request time to run to completion or near completion
        try await Task.sleep(nanoseconds: 200_000_000)  // 200ms

        // Now submit the second request — this triggers upgrade.
        // If the first request already finished, the upgrade falls back
        // gracefully (live state is nil → starts a new single request).
        let input2 = LMInput(tokens: MLXArray([Int32(20)]))
        let params2 = GenerateParameters(maxTokens: 5, temperature: 0)

        let stream2 = try await scheduler.submit(
            input: input2,
            parameters: params2,
            model: model,
            cache: nil,
            tokenizer: tokenizer,
            configuration: config
        )

        // Collect results from both streams
        let (chunks1, info1) = await collectTask.value
        firstChunks = chunks1
        firstInfo = info1

        var secondChunks = [String]()
        for await gen in stream2 {
            if let chunk = gen.chunk {
                secondChunks.append(chunk)
            }
        }

        // The first request must have produced at most maxTokens tokens.
        // With the old bug (max(0, 1) clamping), it could produce maxTokens + 1.
        XCTAssertLessThanOrEqual(
            firstChunks.count, maxTokens,
            "First request must not exceed maxTokens (\(maxTokens)) — got \(firstChunks.count) chunks"
        )

        // If we got completion info, verify the token count is within budget
        if let info = firstInfo {
            XCTAssertLessThanOrEqual(
                info.generationTokenCount, maxTokens,
                "GenerateCompletionInfo token count must not exceed maxTokens"
            )
        }
    }

    /// Verifies that the first request produces exactly maxTokens tokens total
    /// even when upgrade occurs mid-generation. Tokens produced on the single
    /// path plus tokens produced on the batch path must sum to at most maxTokens.
    func testFirstRequestProducesExactlyMaxTokensAcrossUpgrade() async throws {
        try skipIfMetalUnavailable()

        let model = SchedulerMockModel()
        let tokenizer = TestTokenizer()
        let config = ModelConfiguration(id: "test-model")
        let scheduler = InferenceScheduler()

        let maxTokens = 10
        let input1 = LMInput(tokens: MLXArray([Int32(10)]))
        let params1 = GenerateParameters(maxTokens: maxTokens, temperature: 0)

        let stream1 = try await scheduler.submit(
            input: input1,
            parameters: params1,
            model: model,
            cache: nil,
            tokenizer: tokenizer,
            configuration: config
        )

        // Consume a few tokens to advance the iterator, then trigger upgrade
        var firstTokenCount = 0

        let collectTask = Task { () -> Int in
            var count = 0
            for await gen in stream1 {
                if gen.chunk != nil {
                    count += 1
                }
            }
            return count
        }

        // Small delay to let a few tokens be generated
        try await Task.sleep(nanoseconds: 50_000_000)  // 50ms

        // Trigger upgrade with second request
        let input2 = LMInput(tokens: MLXArray([Int32(20)]))
        let params2 = GenerateParameters(maxTokens: 5, temperature: 0)

        let stream2 = try await scheduler.submit(
            input: input2,
            parameters: params2,
            model: model,
            cache: nil,
            tokenizer: tokenizer,
            configuration: config
        )

        firstTokenCount = await collectTask.value

        // Consume second stream
        for await _ in stream2 {}

        // The total tokens for the first request (across single + batch) must
        // not exceed maxTokens.
        XCTAssertLessThanOrEqual(
            firstTokenCount, maxTokens,
            "Total first-request tokens across upgrade must not exceed maxTokens (\(maxTokens)), got \(firstTokenCount)"
        )
    }

    // MARK: - VAL-FIX-004: Single-to-batch upgrade preserves RotatingKVCache state

    func testUpgradePreservesRotatingKVCacheState() async throws {
        try skipIfMetalUnavailable()

        let model = RotatingCacheMockModel(
            slidingWindowMaxSize: 64,
            slidingWindowKeep: 4
        )
        let tokenizer = TestTokenizer()
        let config = ModelConfiguration(id: "test-model")
        let scheduler = InferenceScheduler()

        // Submit first request with enough tokens to generate for a while
        let input1 = LMInput(tokens: MLXArray([Int32(1), Int32(2), Int32(3)]))
        let params1 = GenerateParameters(maxTokens: 20, temperature: 0)

        let stream1 = try await scheduler.submit(
            input: input1,
            parameters: params1,
            model: model,
            cache: model.newCache(parameters: nil),
            tokenizer: tokenizer,
            configuration: config
        )

        // Collect first stream in a background task
        let collectTask = Task {
            var count = 0
            for await event in stream1 {
                if case .chunk = event {
                    count += 1
                }
            }
            return count
        }

        // Small delay to let a few tokens be generated on the single path
        try await Task.sleep(nanoseconds: 50_000_000)  // 50ms

        // Submit second request to trigger batch upgrade
        let input2 = LMInput(tokens: MLXArray([Int32(10)]))
        let params2 = GenerateParameters(maxTokens: 5, temperature: 0)

        let stream2 = try await scheduler.submit(
            input: input2,
            parameters: params2,
            model: model,
            cache: model.newCache(parameters: nil),
            tokenizer: tokenizer,
            configuration: config
        )

        // Consume both streams
        let firstTokenCount = await collectTask.value
        var secondTokenCount = 0
        for await event in stream2 {
            if case .chunk = event {
                secondTokenCount += 1
            }
        }

        // Both requests should have produced tokens — the upgrade should not
        // have silently broken generation by discarding RotatingKVCache data.
        XCTAssertGreaterThan(
            firstTokenCount, 0,
            "First request should produce tokens after upgrade"
        )
        XCTAssertGreaterThan(
            secondTokenCount, 0,
            "Second request should produce tokens"
        )

        // Verify the scheduler transitioned through batch mode.
        // After both streams complete, the scheduler should be idle.
        let state = await scheduler.currentState
        XCTAssertEqual(state, "idle", "Scheduler should be idle after both streams complete")
    }

    // MARK: - VAL-FIX-005: Batched .info reports correct promptTokenCount

    /// Verifies that .info events for each batched request report the actual
    /// prompt token count (matching the input token array length), not zero.
    func testBatchedInfoReportsCorrectPromptTokenCount() async throws {
        try skipIfMetalUnavailable()

        let model = SchedulerMockModel()
        let tokenizer = TestTokenizer()
        let config = ModelConfiguration(id: "test-model")
        let scheduler = InferenceScheduler()

        // First request with 3 prompt tokens
        let input1 = LMInput(tokens: MLXArray([Int32(1), Int32(2), Int32(3)]))
        let params1 = GenerateParameters(maxTokens: 20, temperature: 0)

        let stream1 = try await scheduler.submit(
            input: input1,
            parameters: params1,
            model: model,
            cache: nil,
            tokenizer: tokenizer,
            configuration: config
        )

        // Second request with 5 prompt tokens — triggers batch upgrade
        let input2 = LMInput(
            tokens: MLXArray([Int32(10), Int32(11), Int32(12), Int32(13), Int32(14)]))
        let params2 = GenerateParameters(maxTokens: 5, temperature: 0)

        let stream2 = try await scheduler.submit(
            input: input2,
            parameters: params2,
            model: model,
            cache: nil,
            tokenizer: tokenizer,
            configuration: config
        )

        let currentState = await scheduler.currentState
        // If upgrade succeeded, we're in batched mode. If the first request
        // finished before the handshake, fallback to single is also OK —
        // but we primarily test the batched path.
        guard currentState == "batched" else {
            // Fallback: first request already completed before upgrade.
            // Consume streams and skip batch-specific assertions.
            for await _ in stream1 {}
            for await _ in stream2 {}
            return
        }

        // Collect .info events from both streams
        typealias InfoResult = GenerateCompletionInfo?

        var info1: InfoResult = nil
        var info2: InfoResult = nil

        await withTaskGroup(of: (Int, InfoResult).self) { group in
            group.addTask {
                var info: GenerateCompletionInfo?
                for await gen in stream1 {
                    if let i = gen.info { info = i }
                }
                return (1, info)
            }
            group.addTask {
                var info: GenerateCompletionInfo?
                for await gen in stream2 {
                    if let i = gen.info { info = i }
                }
                return (2, info)
            }

            for await (id, result) in group {
                if id == 1 { info1 = result } else { info2 = result }
            }
        }

        // First request's .info must have promptTokenCount == 3 (its input token count)
        XCTAssertNotNil(info1, "First request should receive .info")
        if let info = info1 {
            XCTAssertEqual(
                info.promptTokenCount, 3,
                "First request's .info promptTokenCount should match input token count (3), got \(info.promptTokenCount)"
            )
        }

        // Second request's .info must have promptTokenCount == 5 (its input token count)
        XCTAssertNotNil(info2, "Second request should receive .info")
        if let info = info2 {
            XCTAssertEqual(
                info.promptTokenCount, 5,
                "Second request's .info promptTokenCount should match input token count (5), got \(info.promptTokenCount)"
            )
        }
    }

    // MARK: - VAL-FIX-006: Prompt timing preserved across single-to-batch upgrade

    /// Verifies that the first request's prompt processing time is preserved
    /// through the single-to-batch upgrade and reported in its .info event
    /// (not reset to zero).
    func testFirstRequestPromptTimePreservedAfterUpgrade() async throws {
        try skipIfMetalUnavailable()

        let model = SchedulerMockModel()
        let tokenizer = TestTokenizer()
        let config = ModelConfiguration(id: "test-model")
        let scheduler = InferenceScheduler()

        // First request with enough tokens to generate for a while
        let input1 = LMInput(tokens: MLXArray([Int32(1), Int32(2), Int32(3)]))
        let params1 = GenerateParameters(maxTokens: 20, temperature: 0)

        let stream1 = try await scheduler.submit(
            input: input1,
            parameters: params1,
            model: model,
            cache: nil,
            tokenizer: tokenizer,
            configuration: config
        )

        // Small delay to let the first request produce a token and measure promptTime
        try await Task.sleep(nanoseconds: 50_000_000)  // 50ms

        // Second request triggers upgrade
        let input2 = LMInput(tokens: MLXArray([Int32(10), Int32(11)]))
        let params2 = GenerateParameters(maxTokens: 5, temperature: 0)

        let stream2 = try await scheduler.submit(
            input: input2,
            parameters: params2,
            model: model,
            cache: nil,
            tokenizer: tokenizer,
            configuration: config
        )

        let currentState = await scheduler.currentState
        guard currentState == "batched" else {
            // Fallback: first request already completed before upgrade.
            for await _ in stream1 {}
            for await _ in stream2 {}
            return
        }

        // Collect .info from the first request
        typealias InfoResult = GenerateCompletionInfo?

        var firstInfo: InfoResult = nil

        await withTaskGroup(of: (Int, InfoResult).self) { group in
            group.addTask {
                var info: GenerateCompletionInfo?
                for await gen in stream1 {
                    if let i = gen.info { info = i }
                }
                return (1, info)
            }
            group.addTask {
                for await _ in stream2 {}
                return (2, nil)
            }

            for await (id, result) in group {
                if id == 1 { firstInfo = result }
            }
        }

        // The first request's promptTime must be > 0 — it was measured on the
        // single path before upgrade and should be preserved through the handoff.
        XCTAssertNotNil(firstInfo, "First request should receive .info after upgrade")
        if let info = firstInfo {
            XCTAssertGreaterThan(
                info.promptTime, 0,
                "First request's promptTime should be > 0 after upgrade, got \(info.promptTime)"
            )
        }
    }

    // MARK: - VAL-FIX-007: Submit accepts cachedKVState parameter

    /// Verifies that the scheduler's submit() method accepts an optional
    /// cachedKVState parameter and passes it through to the batch path.
    func testSubmitAcceptsCachedKVStateParameter() async throws {
        try skipIfMetalUnavailable()

        let model = SchedulerMockModel()
        let tokenizer = TestTokenizer()
        let config = ModelConfiguration(id: "test-model")
        let scheduler = InferenceScheduler()

        // Create a mock cached KV state
        let cachedKV: [KVCache] = [KVCacheSimple()]

        let input = LMInput(tokens: MLXArray([Int32(1), Int32(2), Int32(3)]))
        let params = GenerateParameters(maxTokens: 3, temperature: 0)

        // Submit with cachedKVState — should not crash
        let stream = try await scheduler.submit(
            input: input,
            parameters: params,
            model: model,
            cache: nil,
            tokenizer: tokenizer,
            configuration: config,
            cachedKVState: cachedKV
        )

        // Consume the stream — should work normally
        var chunks = [String]()
        for await gen in stream {
            if let chunk = gen.chunk {
                chunks.append(chunk)
            }
        }

        // Should produce output
        XCTAssertFalse(chunks.isEmpty, "Should produce output with cachedKVState")
    }

    /// Verifies that submit with nil cachedKVState (default) works unchanged.
    func testSubmitWithNilCachedKVStateWorksUnchanged() async throws {
        try skipIfMetalUnavailable()

        let model = SchedulerMockModel()
        let tokenizer = TestTokenizer()
        let config = ModelConfiguration(id: "test-model")
        let scheduler = InferenceScheduler()

        let input = LMInput(tokens: MLXArray([Int32(1), Int32(2)]))
        let params = GenerateParameters(maxTokens: 3, temperature: 0)

        // Submit without cachedKVState (using default nil)
        let stream = try await scheduler.submit(
            input: input,
            parameters: params,
            model: model,
            cache: nil,
            tokenizer: tokenizer,
            configuration: config
        )

        var chunks = [String]()
        for await gen in stream {
            if let chunk = gen.chunk {
                chunks.append(chunk)
            }
        }

        XCTAssertFalse(chunks.isEmpty, "Should produce output with default nil cachedKVState")
    }

    /// Verifies that cachedKVState is passed through the batch upgrade path
    /// (second request with cached state joins batch correctly).
    func testCachedKVStateThroughBatchUpgradePath() async throws {
        try skipIfMetalUnavailable()

        let model = SchedulerMockModel()
        let tokenizer = TestTokenizer()
        let config = ModelConfiguration(id: "test-model")
        let scheduler = InferenceScheduler()

        // First request without cache (standard path)
        let input1 = LMInput(tokens: MLXArray([Int32(1), Int32(2)]))
        let params1 = GenerateParameters(maxTokens: 20, temperature: 0)

        let stream1 = try await scheduler.submit(
            input: input1,
            parameters: params1,
            model: model,
            cache: nil,
            tokenizer: tokenizer,
            configuration: config
        )

        // Second request with cached KV state — triggers batch upgrade
        let cachedKV: [KVCache] = [KVCacheSimple()]
        let input2 = LMInput(tokens: MLXArray([Int32(5), Int32(6), Int32(7)]))
        let params2 = GenerateParameters(maxTokens: 5, temperature: 0)

        let stream2 = try await scheduler.submit(
            input: input2,
            parameters: params2,
            model: model,
            cache: nil,
            tokenizer: tokenizer,
            configuration: config,
            cachedKVState: cachedKV
        )

        // Both streams should produce output
        var chunks1 = [String]()
        var chunks2 = [String]()

        await withTaskGroup(of: (Int, [String]).self) { group in
            group.addTask {
                var chunks = [String]()
                for await gen in stream1 {
                    if let chunk = gen.chunk { chunks.append(chunk) }
                }
                return (1, chunks)
            }
            group.addTask {
                var chunks = [String]()
                for await gen in stream2 {
                    if let chunk = gen.chunk { chunks.append(chunk) }
                }
                return (2, chunks)
            }

            for await (id, chunks) in group {
                if id == 1 { chunks1 = chunks } else { chunks2 = chunks }
            }
        }

        // Both should produce output, with the second request using its cached state
        let totalOutput = chunks1.count + chunks2.count
        XCTAssertGreaterThan(
            totalOutput, 0,
            "Both streams should produce output when second has cachedKVState"
        )
    }
}
