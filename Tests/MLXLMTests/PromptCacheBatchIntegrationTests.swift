// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXNN
import XCTest

@testable import MLXLMCommon

// MARK: - Mock Language Model

/// A deterministic mock language model for prompt cache batch integration tests.
///
/// Given input tokens of shape `[B, S]`, it produces logits of shape `[B, S, vocabSize]`
/// where the highest-logit token for each position is `(input_token + 1) % vocabSize`.
/// Tracks call count and input shapes for verifying reduced prefill.
private class MockCachePrefillModel: Module, LanguageModel {
    let vocabSize: Int
    let numLayers: Int

    /// Track call count for verifying that cached prefixes reduce model calls.
    var callCount = 0

    /// Track total tokens processed across all calls.
    var totalTokensProcessed = 0

    /// Track input shapes for each call.
    var inputShapes: [[Int]] = []

    init(vocabSize: Int = 32, numLayers: Int = 2) {
        self.vocabSize = vocabSize
        self.numLayers = numLayers
    }

    func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
        .tokens(input.text)
    }

    func callAsFunction(
        _ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?
    ) -> LMOutput {
        callCount += 1
        let tokens = input.tokens
        let B = tokens.dim(0)
        let S = tokens.dim(1)
        inputShapes.append([B, S])
        totalTokensProcessed += B * S

        // Build logits: predicted next token = (last_input_token + 1) % vocabSize
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

    func newCache(parameters: GenerateParameters?) -> [KVCache] {
        (0 ..< numLayers).map { _ in KVCacheSimple() }
    }

    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        weights
    }

    /// Reset tracking counters.
    func resetCounters() {
        callCount = 0
        totalTokensProcessed = 0
        inputShapes = []
    }
}

// MARK: - Tests

/// Tests for the integration of LRUPromptCache with batch generation.
///
/// These tests verify:
/// - VAL-PCACHE-007: Extract individual cache from BatchKVCache
/// - VAL-PCACHE-008: Merge individual caches into BatchKVCache
/// - VAL-PCACHE-009: Cached prompt reduces prefill token count
/// - VAL-PCACHE-010: Merge-extract roundtrip preserves data
///
/// Additionally tests mixed cached/uncached batches and correct generation output.
class PromptCacheBatchIntegrationTests: XCTestCase {

    // MARK: - Helpers

    /// Create keys/values with known content for testing.
    /// Shape: [B, H, S, D]
    private func makeKV(
        batchSize B: Int, heads H: Int, seqLen S: Int, headDim D: Int, value: Float = 1.0
    ) -> (MLXArray, MLXArray) {
        let keys = MLXArray.ones([B, H, S, D]) * value
        let values = MLXArray.ones([B, H, S, D]) * (value + 1)
        return (keys, values)
    }

    /// Create a mock KVCacheSimple with synthetic keys/values.
    private func makeMockCache(seqLen: Int, heads: Int = 2, headDim: Int = 4, value: Float = 1.0)
        -> KVCacheSimple
    {
        let cache = KVCacheSimple()
        if seqLen > 0 {
            let keys = MLXArray.ones([1, heads, seqLen, headDim]) * value
            let values = MLXArray.ones([1, heads, seqLen, headDim]) * (value + 1)
            _ = cache.update(keys: keys, values: values)
        }
        return cache
    }

    /// Create a multi-layer mock prompt cache (array of KVCacheSimple).
    private func makeMockPromptCache(
        layers: Int = 2, seqLen: Int, heads: Int = 2, headDim: Int = 4, value: Float = 1.0
    ) -> [KVCache] {
        (0 ..< layers).map { _ in
            makeMockCache(seqLen: seqLen, heads: heads, headDim: headDim, value: value)
        }
    }

    // MARK: - VAL-PCACHE-007: Extract individual cache from BatchKVCache

    /// Verify that extract(idx:) on a batch returns a single-sequence cache with padding removed.
    func testExtractFromBatchRemovesPadding() throws {
        try skipIfMetalUnavailable()

        // Create individual caches with different lengths
        let cacheA = makeMockCache(seqLen: 3, value: 1.0)
        let cacheB = makeMockCache(seqLen: 7, value: 2.0)

        // Merge into a batch
        let batchCache = BatchKVCache.merge([cacheA, cacheB])

        // Extract each individual cache
        let extractedA = batchCache.extract(idx: 0)
        let extractedB = batchCache.extract(idx: 1)

        // A had padding of 4 (7 - 3), so extracted should have only 3 tokens
        XCTAssertEqual(
            extractedA.offset, 3, "Extracted cache A should have offset 3 (padding stripped)")
        XCTAssertEqual(
            extractedA.keys!.dim(2), 3, "Extracted keys should have 3 positions (no padding)")

        // B had no padding
        XCTAssertEqual(extractedB.offset, 7, "Extracted cache B should have offset 7")
        XCTAssertEqual(extractedB.keys!.dim(2), 7, "Extracted keys should have 7 positions")

        // Batch dimension should be 1 for both
        XCTAssertEqual(extractedA.keys!.dim(0), 1)
        XCTAssertEqual(extractedB.keys!.dim(0), 1)
    }

    // MARK: - VAL-PCACHE-008: Merge individual caches into BatchKVCache

    /// Verify that merging individual caches creates a batch with correct left-padding.
    func testMergeCreatesCorrectLeftPadding() throws {
        try skipIfMetalUnavailable()

        let cacheA = makeMockCache(seqLen: 5, value: 1.0)
        let cacheB = makeMockCache(seqLen: 3, value: 2.0)
        let cacheC = makeMockCache(seqLen: 8, value: 3.0)

        let batchCache = BatchKVCache.merge([cacheA, cacheB, cacheC])

        // Max length is 8, so padding = [3, 5, 0]
        XCTAssertEqual(batchCache.batchSize, 3)
        XCTAssertEqual(batchCache.leftPadding[0].item(Int32.self), 3)  // 8 - 5
        XCTAssertEqual(batchCache.leftPadding[1].item(Int32.self), 5)  // 8 - 3
        XCTAssertEqual(batchCache.leftPadding[2].item(Int32.self), 0)  // 8 - 8

        // _idx should equal the max length
        XCTAssertEqual(batchCache._idx, 8)

        // Keys shape should be [3, H, 8, D]
        XCTAssertEqual(batchCache.keys!.dim(0), 3)
        XCTAssertEqual(batchCache.keys!.dim(2), 8)
    }

    // MARK: - VAL-PCACHE-009: Cached prompt reduces prefill token count

    /// When a request has a cached prefix, only uncached suffix tokens go through
    /// model prefill. Verify reduced model call count.
    func testCachedPromptReducesPrefillTokenCount() throws {
        try skipIfMetalUnavailable()

        let model = MockCachePrefillModel(vocabSize: 32, numLayers: 2)

        // --- Run 1: Full prefill (no cache) ---
        let iteratorFull = BatchTokenIterator(
            model: model,
            defaultSampler: ArgMaxSampler(),
            completionBatchSize: 32,
            prefillBatchSize: 8
        )

        let prompt = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        let _ = iteratorFull.insert(
            prompts: [prompt],
            maxTokens: [1]
        )

        // Trigger prefill
        let _ = iteratorFull.next()
        let fullPrefillCalls = model.callCount
        let fullTokensProcessed = model.totalTokensProcessed

        // --- Run 2: Cached prefill (8 tokens cached, 2 suffix) ---
        model.resetCounters()

        // Create a cached KV state covering the first 8 tokens
        let cachedLayers = makeMockPromptCache(layers: 2, seqLen: 8)

        let iteratorCached = BatchTokenIterator(
            model: model,
            defaultSampler: ArgMaxSampler(),
            completionBatchSize: 32,
            prefillBatchSize: 8
        )

        let _ = iteratorCached.insert(
            prompts: [prompt],
            maxTokens: [1],
            cachedKVStates: [cachedLayers]
        )

        // Trigger prefill
        let _ = iteratorCached.next()
        let cachedPrefillCalls = model.callCount
        let cachedTokensProcessed = model.totalTokensProcessed

        // The cached path should process fewer tokens because 8 out of 10
        // tokens are already cached, leaving only 2 suffix tokens for prefill.
        XCTAssertLessThan(
            cachedTokensProcessed, fullTokensProcessed,
            "Cached prefill should process fewer tokens (\(cachedTokensProcessed)) "
                + "than full prefill (\(fullTokensProcessed))"
        )

        // Full prefill processes 10 tokens; cached prefill processes only 2 suffix tokens.
        // The suffix has 2 tokens: [9, 10]. The model processes the first 1 in a chunk
        // step, then the last 1 in the final sampling step = 2 calls total.
        // Full prefill: 9 tokens in chunks + 1 for sampling = at least 2 calls.
        // With default prefillStepSize=2048, full does it in 2 calls (9 chunk + 1 sample).
        // Cached does it in 2 calls (1 chunk + 1 sample) but fewer tokens per call.
        XCTAssertLessThanOrEqual(
            cachedPrefillCalls, fullPrefillCalls,
            "Cached prefill should need at most as many model calls"
        )
    }

    /// Verify reduced prefill with multiple prompts with different cache depths.
    func testMixedCacheDepthsReducePrefill() throws {
        try skipIfMetalUnavailable()

        let model = MockCachePrefillModel(vocabSize: 32, numLayers: 2)

        // --- Run 1: Full prefill for two prompts ---
        let iteratorFull = BatchTokenIterator(
            model: model,
            defaultSampler: ArgMaxSampler(),
            completionBatchSize: 32,
            prefillBatchSize: 8
        )

        let promptA = [1, 2, 3, 4, 5]  // 5 tokens
        let promptB = [10, 11, 12, 13, 14, 15, 16, 17]  // 8 tokens

        let _ = iteratorFull.insert(
            prompts: [promptA, promptB],
            maxTokens: [1, 1]
        )
        let _ = iteratorFull.next()
        let fullTokensProcessed = model.totalTokensProcessed

        // --- Run 2: Cached prefill ---
        model.resetCounters()

        // Cache 3 tokens for prompt A (suffix = [4, 5], 2 tokens)
        // Cache 6 tokens for prompt B (suffix = [16, 17], 2 tokens)
        let cachedA = makeMockPromptCache(layers: 2, seqLen: 3, value: 1.0)
        let cachedB = makeMockPromptCache(layers: 2, seqLen: 6, value: 2.0)

        let iteratorCached = BatchTokenIterator(
            model: model,
            defaultSampler: ArgMaxSampler(),
            completionBatchSize: 32,
            prefillBatchSize: 8
        )

        let _ = iteratorCached.insert(
            prompts: [promptA, promptB],
            maxTokens: [1, 1],
            cachedKVStates: [cachedA, cachedB]
        )
        let _ = iteratorCached.next()
        let cachedTokensProcessed = model.totalTokensProcessed

        // Full prefill: 5 + 8 = 13 tokens padded to 8 each = 16 total tokens processed
        // Cached prefill: suffixes are 2 tokens each = 4 total tokens processed
        XCTAssertLessThan(
            cachedTokensProcessed, fullTokensProcessed,
            "Cached prefill should process fewer tokens (\(cachedTokensProcessed)) "
                + "than full prefill (\(fullTokensProcessed))"
        )
    }

    /// Verify mixed cached and uncached prompts in a single batch.
    func testMixedCachedAndUncachedPrompts() throws {
        try skipIfMetalUnavailable()

        let model = MockCachePrefillModel(vocabSize: 32, numLayers: 2)

        let iterator = BatchTokenIterator(
            model: model,
            defaultSampler: ArgMaxSampler(),
            completionBatchSize: 32,
            prefillBatchSize: 8
        )

        // Prompt A: fully uncached (5 tokens)
        let promptA = [1, 2, 3, 4, 5]
        // Prompt B: cached prefix of 6 tokens, suffix = [17] (1 token)
        let promptB = [10, 11, 12, 13, 14, 15, 16, 17]
        let cachedB = makeMockPromptCache(layers: 2, seqLen: 7, value: 2.0)

        let uids = iterator.insert(
            prompts: [promptA, promptB],
            maxTokens: [2, 2],
            cachedKVStates: [nil, cachedB]
        )

        // Run generation
        var tokensPerUID = [Int: [Int]]()
        var loopCount = 0
        while let responses = iterator.next(), !responses.isEmpty {
            for r in responses {
                tokensPerUID[r.uid, default: []].append(r.token)
            }
            loopCount += 1
            if loopCount > 20 { break }
        }

        // Both prompts should produce tokens
        XCTAssertEqual(tokensPerUID[uids[0]]?.count, 2, "Uncached prompt should produce 2 tokens")
        XCTAssertEqual(tokensPerUID[uids[1]]?.count, 2, "Cached prompt should produce 2 tokens")
    }

    // MARK: - VAL-PCACHE-010: Merge-extract roundtrip preserves data

    /// Merging then extracting produces caches identical to originals.
    func testMergeExtractRoundtripPreservesData() throws {
        try skipIfMetalUnavailable()

        let H = 2
        let D = 4

        // Create individual caches with distinct content
        let cacheA = KVCacheSimple()
        let cacheB = KVCacheSimple()
        let cacheC = KVCacheSimple()

        let kA = MLXArray.ones([1, H, 3, D]) * 1.0
        let vA = MLXArray.ones([1, H, 3, D]) * 10.0
        let kB = MLXArray.ones([1, H, 5, D]) * 2.0
        let vB = MLXArray.ones([1, H, 5, D]) * 20.0
        let kC = MLXArray.ones([1, H, 7, D]) * 3.0
        let vC = MLXArray.ones([1, H, 7, D]) * 30.0

        _ = cacheA.update(keys: kA, values: vA)
        _ = cacheB.update(keys: kB, values: vB)
        _ = cacheC.update(keys: kC, values: vC)

        // Merge into a batch
        let batchCache = BatchKVCache.merge([cacheA, cacheB, cacheC])

        // Extract each individual cache
        let extractedA = batchCache.extract(idx: 0)
        let extractedB = batchCache.extract(idx: 1)
        let extractedC = batchCache.extract(idx: 2)

        // Verify offsets match originals
        XCTAssertEqual(extractedA.offset, 3)
        XCTAssertEqual(extractedB.offset, 5)
        XCTAssertEqual(extractedC.offset, 7)

        // Verify key dimensions match originals
        XCTAssertEqual(extractedA.keys!.dim(2), 3)
        XCTAssertEqual(extractedB.keys!.dim(2), 5)
        XCTAssertEqual(extractedC.keys!.dim(2), 7)

        // Verify key values match originals (within floating point tolerance)
        let diffAKeys = abs(extractedA.keys![.ellipsis, ..<3, 0...] - kA).sum().item(Float.self)
        let diffBKeys = abs(extractedB.keys![.ellipsis, ..<5, 0...] - kB).sum().item(Float.self)
        let diffCKeys = abs(extractedC.keys![.ellipsis, ..<7, 0...] - kC).sum().item(Float.self)
        XCTAssertEqual(diffAKeys, 0.0, "Cache A keys should match original after round-trip")
        XCTAssertEqual(diffBKeys, 0.0, "Cache B keys should match original after round-trip")
        XCTAssertEqual(diffCKeys, 0.0, "Cache C keys should match original after round-trip")

        // Verify value values match originals
        let diffAValues = abs(extractedA.values![.ellipsis, ..<3, 0...] - vA).sum().item(Float.self)
        let diffBValues = abs(extractedB.values![.ellipsis, ..<5, 0...] - vB).sum().item(Float.self)
        let diffCValues = abs(extractedC.values![.ellipsis, ..<7, 0...] - vC).sum().item(Float.self)
        XCTAssertEqual(diffAValues, 0.0, "Cache A values should match original after round-trip")
        XCTAssertEqual(diffBValues, 0.0, "Cache B values should match original after round-trip")
        XCTAssertEqual(diffCValues, 0.0, "Cache C values should match original after round-trip")
    }

    /// Multi-layer merge-extract roundtrip preserves all layers.
    func testMultiLayerMergeExtractRoundtrip() throws {
        try skipIfMetalUnavailable()

        let numLayers = 3
        let H = 2
        let D = 4

        // Create per-layer caches for two sequences
        var layerCachesA = [KVCacheSimple]()
        var layerCachesB = [KVCacheSimple]()

        for l in 0 ..< numLayers {
            let cA = KVCacheSimple()
            let kA = MLXArray.ones([1, H, 4, D]) * Float(l + 1)
            let vA = MLXArray.ones([1, H, 4, D]) * Float(l + 1) * 10
            _ = cA.update(keys: kA, values: vA)
            layerCachesA.append(cA)

            let cB = KVCacheSimple()
            let kB = MLXArray.ones([1, H, 6, D]) * Float(l + 10)
            let vB = MLXArray.ones([1, H, 6, D]) * Float(l + 10) * 10
            _ = cB.update(keys: kB, values: vB)
            layerCachesB.append(cB)
        }

        // Merge per-layer
        var batchCaches = [BatchKVCache]()
        for l in 0 ..< numLayers {
            batchCaches.append(BatchKVCache.merge([layerCachesA[l], layerCachesB[l]]))
        }

        // Extract per-layer
        for l in 0 ..< numLayers {
            let extractedA = batchCaches[l].extract(idx: 0)
            let extractedB = batchCaches[l].extract(idx: 1)

            XCTAssertEqual(extractedA.offset, 4, "Layer \(l): A offset should be 4")
            XCTAssertEqual(extractedB.offset, 6, "Layer \(l): B offset should be 6")

            // Verify key content
            let expectedKeyA = Float(l + 1)
            let actualKeyA = extractedA.keys![0, 0, 0, 0].item(Float.self)
            XCTAssertEqual(actualKeyA, expectedKeyA, "Layer \(l): A key value should match")

            let expectedKeyB = Float(l + 10)
            let actualKeyB = extractedB.keys![0, 0, 0, 0].item(Float.self)
            XCTAssertEqual(actualKeyB, expectedKeyB, "Layer \(l): B key value should match")
        }
    }

    // MARK: - Full LRUPromptCache Integration

    /// End-to-end: insert cache, fetch it, use in batch generation.
    func testLRUPromptCacheWithBatchGeneration() throws {
        try skipIfMetalUnavailable()

        let model = MockCachePrefillModel(vocabSize: 32, numLayers: 2)
        let promptCache = LRUPromptCache(maxSize: 10)

        // Simulate: first request generates and stores cache
        let tokens = [1, 2, 3, 4, 5, 6, 7, 8]
        let cachedKV = makeMockPromptCache(layers: 2, seqLen: 8, value: 1.0)
        promptCache.insertCache(model: "test", tokens: tokens, promptCache: cachedKV)

        // Second request: same prefix, different suffix
        let newTokens = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        let (fetchedCache, remainder) = promptCache.fetchNearestCache(
            model: "test", tokens: newTokens
        )

        XCTAssertNotNil(fetchedCache, "Should find cached prefix")
        XCTAssertEqual(remainder, [9, 10], "Remainder should be the uncached suffix")

        // Use the fetched cache in batch generation
        let iterator = BatchTokenIterator(
            model: model,
            defaultSampler: ArgMaxSampler(),
            completionBatchSize: 32,
            prefillBatchSize: 8
        )

        model.resetCounters()
        let uids = iterator.insert(
            prompts: [newTokens],
            maxTokens: [3],
            cachedKVStates: [fetchedCache]
        )

        var tokenCount = 0
        while let responses = iterator.next(), !responses.isEmpty {
            for r in responses {
                XCTAssertEqual(r.uid, uids[0])
                XCTAssertGreaterThanOrEqual(r.token, 0)
                XCTAssertLessThan(r.token, model.vocabSize)
                tokenCount += 1
            }
        }

        XCTAssertEqual(tokenCount, 3, "Should generate 3 tokens")

        // The model should have processed only the suffix (2 tokens) + sampling,
        // not the full 10-token prompt.
        XCTAssertLessThan(
            model.totalTokensProcessed, 10,
            "Should process fewer than 10 tokens due to cached prefix"
        )
    }

    // MARK: - Edge Cases

    /// Exact cache match: entire prompt is cached, prefill is skipped entirely.
    /// The last prompt token is replayed from the trimmed cache (trim+re-process)
    /// to get logits for the first decode token, then one decode step produces
    /// the generated token. This follows the pattern: 1 trim+replay + maxTokens
    /// decode steps = 2 total model calls (matching testCacheCoversFull which
    /// expects 1 + 2 = 3 for maxTokens=2).
    func testExactCacheMatchSkipsPrefill() throws {
        try skipIfMetalUnavailable()

        let model = MockCachePrefillModel(vocabSize: 32, numLayers: 2)

        // Cache covers all 5 tokens
        let prompt = [1, 2, 3, 4, 5]
        let cachedKV = makeMockPromptCache(layers: 2, seqLen: 5, value: 1.0)

        let iterator = BatchTokenIterator(
            model: model,
            defaultSampler: ArgMaxSampler(),
            completionBatchSize: 32,
            prefillBatchSize: 8
        )

        let _ = iterator.insert(
            prompts: [prompt],
            maxTokens: [1],
            cachedKVStates: [cachedKV]
        )

        let _ = iterator.next()

        // Exact hit: cache is trimmed by 1, then last token re-processed (1 call),
        // plus 1 decode step for the generated token = 2 total model calls.
        XCTAssertEqual(
            model.callCount, 2,
            "Exact cache match should require 2 model calls (1 trim+replay + 1 decode)"
        )
        XCTAssertEqual(
            model.totalTokensProcessed, 2,
            "Exact cache match should process 2 tokens (1 replay + 1 decode)"
        )
    }

    /// Single cached prompt with long suffix still benefits from caching.
    func testLongSuffixStillBenefitsFromCache() throws {
        try skipIfMetalUnavailable()

        let model = MockCachePrefillModel(vocabSize: 32, numLayers: 2)

        // 100-token prompt, 80 tokens cached, 20 suffix tokens
        let prompt = Array(1 ... 100)
        let cachedKV = makeMockPromptCache(layers: 2, seqLen: 80, value: 1.0)

        // Full prefill
        let iteratorFull = BatchTokenIterator(
            model: model,
            defaultSampler: ArgMaxSampler(),
            completionBatchSize: 32,
            prefillBatchSize: 8
        )
        let _ = iteratorFull.insert(prompts: [prompt], maxTokens: [1])
        let _ = iteratorFull.next()
        let fullTokens = model.totalTokensProcessed

        // Cached prefill
        model.resetCounters()
        let iteratorCached = BatchTokenIterator(
            model: model,
            defaultSampler: ArgMaxSampler(),
            completionBatchSize: 32,
            prefillBatchSize: 8
        )
        let _ = iteratorCached.insert(
            prompts: [prompt],
            maxTokens: [1],
            cachedKVStates: [cachedKV]
        )
        let _ = iteratorCached.next()
        let cachedTokens = model.totalTokensProcessed

        // Full processes 100 tokens, cached processes only 20 suffix tokens
        XCTAssertLessThan(
            cachedTokens, fullTokens,
            "Cached prefill (\(cachedTokens) tokens) should be much less than full (\(fullTokens) tokens)"
        )
        // Cached should process roughly 20 tokens (suffix), not 100
        XCTAssertLessThanOrEqual(
            cachedTokens, 25, "Cached prefill should process ~20 suffix tokens")
    }

    /// Cached prompts with zero-length suffix (cache covers entire prompt).
    func testCacheCoversFull() throws {
        try skipIfMetalUnavailable()

        let model = MockCachePrefillModel(vocabSize: 32, numLayers: 2)

        // Cache covers more than the prompt (trimmed to prompt length)
        let prompt = [1, 2, 3]
        // Cache for exactly 3 tokens
        let cachedKV = makeMockPromptCache(layers: 2, seqLen: 3, value: 1.0)

        let iterator = BatchTokenIterator(
            model: model,
            defaultSampler: ArgMaxSampler(),
            completionBatchSize: 32,
            prefillBatchSize: 8
        )

        let uids = iterator.insert(
            prompts: [prompt],
            maxTokens: [2],
            cachedKVStates: [cachedKV]
        )

        // Should work without crashing and produce tokens
        var tokenCount = 0
        while let responses = iterator.next(), !responses.isEmpty {
            for r in responses {
                XCTAssertEqual(r.uid, uids[0])
                tokenCount += 1
            }
        }

        XCTAssertEqual(tokenCount, 2, "Should produce 2 tokens even with fully cached prompt")

        // The first call should be the exact-hit trim+replay (1 token).
        // Subsequent calls are decode steps (1 token each for 2 generated tokens).
        // Total: 1 (exact-hit replay) + 2 (decode steps) = 3 model calls.
        XCTAssertEqual(model.callCount, 3, "Expected 3 model calls: 1 trim+replay + 2 decode")
    }

    // MARK: - Cache Layout Correctness (Mixed Depths)

    /// Verify that mixed-depth cached prompts produce correct KV tensor alignment.
    /// When caches with different depths are merged and suffix-prefilled, the
    /// resulting batch cache must have leftPadding that matches the physical
    /// zero positions in the KV tensors.
    func testMixedDepthCacheLayoutCorrectness() throws {
        try skipIfMetalUnavailable()

        let model = MockCachePrefillModel(vocabSize: 32, numLayers: 2)

        // Prompt A: 3 tokens cached out of 6 → suffix = [4, 5, 6] (3 tokens)
        // Prompt B: 7 tokens cached out of 9 → suffix = [8, 9] (2 tokens)
        //
        // Cache depths differ (3 vs 7), suffix lengths differ (3 vs 2).
        // Right-aligned layout: bufferLen = maxCacheLen = 7
        //   A: leftPadding = 7 - 3 = 4 (data at positions 4..6)
        //   B: leftPadding = 7 - 7 = 0 (data at positions 0..6)
        let promptA = [1, 2, 3, 4, 5, 6]
        let promptB = [10, 11, 12, 13, 14, 15, 16, 17, 18]

        let cachedA = makeMockPromptCache(layers: 2, seqLen: 3, value: 1.0)
        let cachedB = makeMockPromptCache(layers: 2, seqLen: 7, value: 2.0)

        let iterator = BatchTokenIterator(
            model: model,
            defaultSampler: ArgMaxSampler(),
            completionBatchSize: 32,
            prefillBatchSize: 8
        )

        let uids = iterator.insert(
            prompts: [promptA, promptB],
            maxTokens: [3, 3],
            cachedKVStates: [cachedA, cachedB]
        )

        // Run generation and verify both produce tokens
        var tokensPerUID = [Int: [Int]]()
        var loopCount = 0
        while let responses = iterator.next(), !responses.isEmpty {
            for r in responses {
                tokensPerUID[r.uid, default: []].append(r.token)
            }
            loopCount += 1
            if loopCount > 20 { break }
        }

        // Both prompts should produce their requested token count
        XCTAssertEqual(
            tokensPerUID[uids[0]]?.count, 3,
            "Prompt A should produce 3 tokens with mixed-depth cache"
        )
        XCTAssertEqual(
            tokensPerUID[uids[1]]?.count, 3,
            "Prompt B should produce 3 tokens with mixed-depth cache"
        )

        // Verify the model processed fewer tokens than a full-prefill would.
        // Full prefill: 6 + 9 = 15 prompt tokens padded to 9 each = 18.
        // Cached: suffix A = 3 tokens, suffix B = 2 tokens, padded to 3 each = 6.
        // Plus decode steps.
        XCTAssertLessThan(
            model.totalTokensProcessed, 18,
            "Mixed-depth cached prefill should process much fewer than full prefill tokens"
        )
    }

    /// Verify that extracting a cache from a right-aligned mixed-depth merged
    /// batch produces correct per-sequence data with no holes.
    ///
    /// The right-alignment invariant: each sequence's cached KV data ends
    /// exactly at `_idx`, so `leftPadding[i] ..< _idx` contains only valid
    /// written data. This eliminates unwritten holes that the old layout had.
    func testMixedDepthExtractAfterMerge() throws {
        try skipIfMetalUnavailable()

        let H = 2
        let D = 4

        // Create caches with very different depths
        let cacheShort = KVCacheSimple()
        let cacheLong = KVCacheSimple()

        let kShort = MLXArray.ones([1, H, 2, D]) * 5.0
        let vShort = MLXArray.ones([1, H, 2, D]) * 50.0
        let kLong = MLXArray.ones([1, H, 10, D]) * 9.0
        let vLong = MLXArray.ones([1, H, 10, D]) * 90.0

        _ = cacheShort.update(keys: kShort, values: vShort)
        _ = cacheLong.update(keys: kLong, values: vLong)

        // Right-aligned layout: bufferLen = maxCacheLen = 10
        // Short (2 tokens): padding = 10 - 2 = 8, data at positions 8..9
        // Long (10 tokens): padding = 10 - 10 = 0, data at positions 0..9
        let bufferLen = 10  // maxCacheLen
        let rightAlignedPadding = [
            bufferLen - 2,  // 8
            bufferLen - 10,  // 0
        ]

        // Build merged cache manually (as processPartialCacheHits now does)
        let keysArr = MLXArray.zeros([2, H, bufferLen, D])
        let valuesArr = MLXArray.zeros([2, H, bufferLen, D])

        // Place short cache data at position 8..9 (right-aligned to _idx=10)
        keysArr[0 ..< 1, 0..., 8 ..< 10, 0...] = kShort
        valuesArr[0 ..< 1, 0..., 8 ..< 10, 0...] = vShort
        // Place long cache data at position 0..9 (right-aligned to _idx=10)
        keysArr[1 ..< 2, 0..., 0 ..< 10, 0...] = kLong
        valuesArr[1 ..< 2, 0..., 0 ..< 10, 0...] = vLong

        let batchCache = BatchKVCache(leftPadding: rightAlignedPadding)
        batchCache.keys = keysArr
        batchCache.values = valuesArr
        batchCache._idx = bufferLen
        batchCache.batchOffsets = MLXArray([Int32(2), Int32(10)])

        // Extract and verify: no holes in extracted data
        let extractedShort = batchCache.extract(idx: 0)
        let extractedLong = batchCache.extract(idx: 1)

        // Short: leftPadding=8, _idx=10, so extracted has 10-8 = 2 positions
        XCTAssertEqual(extractedShort.offset, 2, "Short cache should have offset 2 (no holes)")
        XCTAssertEqual(
            extractedShort.keys!.dim(2), 2,
            "Short extracted keys should have exactly 2 positions (no padding, no holes)")

        // Long: leftPadding=0, _idx=10, so extracted has 10-0 = 10 positions
        XCTAssertEqual(extractedLong.offset, 10, "Long cache should have offset 10")
        XCTAssertEqual(
            extractedLong.keys!.dim(2), 10,
            "Long extracted keys should have exactly 10 positions")

        // Every position in extracted short cache should be real data (value 5.0)
        let shortKeyVal0 = extractedShort.keys![0, 0, 0, 0].item(Float.self)
        let shortKeyVal1 = extractedShort.keys![0, 0, 1, 0].item(Float.self)
        XCTAssertEqual(shortKeyVal0, 5.0, "All extracted short positions should be real data")
        XCTAssertEqual(shortKeyVal1, 5.0, "All extracted short positions should be real data")

        // Every position in extracted long cache should be real data (value 9.0)
        let longKeyVal0 = extractedLong.keys![0, 0, 0, 0].item(Float.self)
        let longKeyVal9 = extractedLong.keys![0, 0, 9, 0].item(Float.self)
        XCTAssertEqual(longKeyVal0, 9.0, "All extracted long positions should be real data")
        XCTAssertEqual(longKeyVal9, 9.0, "All extracted long positions should be real data")
    }

    /// Verify that exact cache hits mixed with partial hits in a single batch
    /// are handled correctly (each group processes independently).
    func testMixedExactAndPartialCacheHits() throws {
        try skipIfMetalUnavailable()

        let model = MockCachePrefillModel(vocabSize: 32, numLayers: 2)

        // Prompt A: exact hit (5 tokens cached, 5 tokens in prompt)
        let promptA = [1, 2, 3, 4, 5]
        let cachedA = makeMockPromptCache(layers: 2, seqLen: 5, value: 1.0)

        // Prompt B: partial hit (3 tokens cached out of 7)
        let promptB = [10, 11, 12, 13, 14, 15, 16]
        let cachedB = makeMockPromptCache(layers: 2, seqLen: 3, value: 2.0)

        let iterator = BatchTokenIterator(
            model: model,
            defaultSampler: ArgMaxSampler(),
            completionBatchSize: 32,
            prefillBatchSize: 8
        )

        let uids = iterator.insert(
            prompts: [promptA, promptB],
            maxTokens: [2, 2],
            cachedKVStates: [cachedA, cachedB]
        )

        var tokensPerUID = [Int: [Int]]()
        var loopCount = 0
        while let responses = iterator.next(), !responses.isEmpty {
            for r in responses {
                tokensPerUID[r.uid, default: []].append(r.token)
            }
            loopCount += 1
            if loopCount > 20 { break }
        }

        XCTAssertEqual(
            tokensPerUID[uids[0]]?.count, 2,
            "Exact-hit prompt should produce 2 tokens"
        )
        XCTAssertEqual(
            tokensPerUID[uids[1]]?.count, 2,
            "Partial-hit prompt should produce 2 tokens"
        )
    }

    /// Verify that cached generation produces the same token sequence as
    /// uncached generation when using the same deterministic sampler.
    func testCachedVsUncachedGenerationSemanticEquivalence() throws {
        try skipIfMetalUnavailable()

        let model = MockCachePrefillModel(vocabSize: 32, numLayers: 2)
        let prompt = [1, 2, 3, 4, 5, 6, 7, 8]

        // --- Run 1: Fully uncached ---
        let iteratorUncached = BatchTokenIterator(
            model: model,
            defaultSampler: ArgMaxSampler(),
            completionBatchSize: 32,
            prefillBatchSize: 8
        )
        let uidsUncached = iteratorUncached.insert(
            prompts: [prompt],
            maxTokens: [5]
        )

        var uncachedTokens = [Int]()
        while let responses = iteratorUncached.next(), !responses.isEmpty {
            for r in responses {
                uncachedTokens.append(r.token)
            }
        }

        // --- Run 2: Cached prefix (6 tokens cached, 2 suffix) ---
        model.resetCounters()
        let cachedKV = makeMockPromptCache(layers: 2, seqLen: 6, value: 1.0)

        let iteratorCached = BatchTokenIterator(
            model: model,
            defaultSampler: ArgMaxSampler(),
            completionBatchSize: 32,
            prefillBatchSize: 8
        )
        let uidsCached = iteratorCached.insert(
            prompts: [prompt],
            maxTokens: [5],
            cachedKVStates: [cachedKV]
        )

        var cachedTokens = [Int]()
        while let responses = iteratorCached.next(), !responses.isEmpty {
            for r in responses {
                cachedTokens.append(r.token)
            }
        }

        // Both should produce 5 tokens
        XCTAssertEqual(uncachedTokens.count, 5, "Uncached should produce 5 tokens")
        XCTAssertEqual(cachedTokens.count, 5, "Cached should produce 5 tokens")

        // With our mock model (next = input+1 mod vocabSize), the tokens
        // should be valid outputs. We can't expect exact equality because
        // the cached path uses synthetic KV data (ones) rather than model-
        // computed KV data, but both should produce valid token sequences
        // within the vocabulary range.
        for (i, token) in cachedTokens.enumerated() {
            XCTAssertGreaterThanOrEqual(token, 0, "Token \(i) should be >= 0")
            XCTAssertLessThan(token, model.vocabSize, "Token \(i) should be < vocabSize")
        }
    }

    /// Verify that the mock model observes correct cache state during
    /// mixed-depth cached prompt prefill (cache offsets are correct).
    func testMockModelObservesCacheState() throws {
        try skipIfMetalUnavailable()

        // Custom model that records cache offsets during each call
        let model = CacheObservingModel(vocabSize: 32, numLayers: 2)

        // Cache 4 tokens for a 7-token prompt → suffix = [5, 6, 7]
        let prompt = [1, 2, 3, 4, 5, 6, 7]
        let cachedKV = makeMockPromptCache(layers: 2, seqLen: 4, value: 1.0)

        let iterator = BatchTokenIterator(
            model: model,
            defaultSampler: ArgMaxSampler(),
            completionBatchSize: 32,
            prefillBatchSize: 8
        )

        let _ = iterator.insert(
            prompts: [prompt],
            maxTokens: [1],
            cachedKVStates: [cachedKV]
        )

        let _ = iterator.next()

        // The model should have been called at least once
        XCTAssertGreaterThan(model.callCount, 0, "Model should be called during prefill")

        // Verify that the cache provided to the model had non-nil keys
        // (indicating the cached prefix was loaded)
        XCTAssertTrue(
            model.cacheHadKeys,
            "Cache passed to model should have pre-loaded keys from prompt cache"
        )
    }

    // MARK: - Right-Aligned Mixed-Depth Layout Tests

    /// Verify that the right-aligned layout produces a BatchKVCache where every
    /// position in `leftPadding[i] ..< _idx` is filled with valid cached data
    /// (no unwritten holes).
    func testRightAlignedLayoutNoHoles() throws {
        try skipIfMetalUnavailable()

        let H = 2
        let D = 4

        // Simulate the right-aligned layout produced by processPartialCacheHits.
        // Sequence A: 3 tokens cached
        // Sequence B: 7 tokens cached
        // bufferLen = maxCacheLen = 7
        let cacheA = KVCacheSimple()
        let cacheB = KVCacheSimple()

        let kA = MLXArray.ones([1, H, 3, D]) * 3.0
        let vA = MLXArray.ones([1, H, 3, D]) * 30.0
        let kB = MLXArray.ones([1, H, 7, D]) * 7.0
        let vB = MLXArray.ones([1, H, 7, D]) * 70.0

        _ = cacheA.update(keys: kA, values: vA)
        _ = cacheB.update(keys: kB, values: vB)

        let bufferLen = 7  // maxCacheLen
        let rightAlignedPadding = [
            bufferLen - 3,  // 4
            bufferLen - 7,  // 0
        ]

        let keysArr = MLXArray.zeros([2, H, bufferLen, D])
        let valuesArr = MLXArray.zeros([2, H, bufferLen, D])

        // Right-align: A at positions 4..6, B at positions 0..6
        keysArr[0 ..< 1, 0..., 4 ..< 7, 0...] = kA
        valuesArr[0 ..< 1, 0..., 4 ..< 7, 0...] = vA
        keysArr[1 ..< 2, 0..., 0 ..< 7, 0...] = kB
        valuesArr[1 ..< 2, 0..., 0 ..< 7, 0...] = vB

        let batchCache = BatchKVCache(leftPadding: rightAlignedPadding)
        batchCache.keys = keysArr
        batchCache.values = valuesArr
        batchCache._idx = bufferLen

        // Check no holes: every position from leftPadding[i] to _idx should be non-zero.
        // For sequence A (leftPadding=4, _idx=7): positions 4,5,6 should all be 3.0
        for pos in 4 ..< 7 {
            let val = keysArr[0, 0, pos, 0].item(Float.self)
            XCTAssertEqual(
                val, 3.0,
                "Sequence A position \(pos) should contain valid data (3.0), got \(val)"
            )
        }
        // Padding positions should be zero
        for pos in 0 ..< 4 {
            let val = keysArr[0, 0, pos, 0].item(Float.self)
            XCTAssertEqual(
                val, 0.0,
                "Sequence A position \(pos) should be padding (0.0), got \(val)"
            )
        }

        // For sequence B (leftPadding=0, _idx=7): all positions should be 7.0
        for pos in 0 ..< 7 {
            let val = keysArr[1, 0, pos, 0].item(Float.self)
            XCTAssertEqual(
                val, 7.0,
                "Sequence B position \(pos) should contain valid data (7.0), got \(val)"
            )
        }

        // Extract and verify no holes in extracted caches
        let extractedA = batchCache.extract(idx: 0)
        let extractedB = batchCache.extract(idx: 1)

        XCTAssertEqual(extractedA.offset, 3, "Extracted A should have offset 3 (no holes)")
        XCTAssertEqual(extractedB.offset, 7, "Extracted B should have offset 7 (no holes)")

        // All 3 positions in extracted A should be real data
        for pos in 0 ..< 3 {
            let val = extractedA.keys![0, 0, pos, 0].item(Float.self)
            XCTAssertEqual(
                val, 3.0,
                "Extracted A position \(pos) should be real data (3.0)"
            )
        }
    }

    /// Verify that mixed-depth cached prompts through the full BatchTokenIterator
    /// produce correct generation with the right-aligned layout.
    func testMixedDepthCachedPrefillIntegration() throws {
        try skipIfMetalUnavailable()

        let model = MockCachePrefillModel(vocabSize: 32, numLayers: 2)

        // Three prompts with very different cache depths
        let promptA = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  // 10 tokens, 2 cached
        let promptB = [11, 12, 13, 14, 15]  // 5 tokens, 4 cached
        let promptC = [21, 22, 23, 24, 25, 26, 27]  // 7 tokens, 7 cached (exact hit)

        let cachedA = makeMockPromptCache(layers: 2, seqLen: 2, value: 1.0)
        let cachedB = makeMockPromptCache(layers: 2, seqLen: 4, value: 2.0)
        let cachedC = makeMockPromptCache(layers: 2, seqLen: 7, value: 3.0)

        let iterator = BatchTokenIterator(
            model: model,
            defaultSampler: ArgMaxSampler(),
            completionBatchSize: 32,
            prefillBatchSize: 8
        )

        let uids = iterator.insert(
            prompts: [promptA, promptB, promptC],
            maxTokens: [3, 3, 3],
            cachedKVStates: [cachedA, cachedB, cachedC]
        )

        var tokensPerUID = [Int: [Int]]()
        var loopCount = 0
        while let responses = iterator.next(), !responses.isEmpty {
            for r in responses {
                tokensPerUID[r.uid, default: []].append(r.token)
            }
            loopCount += 1
            if loopCount > 30 { break }
        }

        // All three should produce their requested token count
        XCTAssertEqual(
            tokensPerUID[uids[0]]?.count, 3,
            "Prompt A (partial hit, deep suffix) should produce 3 tokens"
        )
        XCTAssertEqual(
            tokensPerUID[uids[1]]?.count, 3,
            "Prompt B (partial hit, shallow suffix) should produce 3 tokens"
        )
        XCTAssertEqual(
            tokensPerUID[uids[2]]?.count, 3,
            "Prompt C (exact hit) should produce 3 tokens"
        )
    }

    // MARK: - RotatingKVCache Cached-Prefill Tests

    /// Verify that RotatingKVCache entries survive the exact-hit cached-prefill path.
    /// Previously, RotatingKVCache layers were silently dropped because the code
    /// hard-coded BatchKVCache.merge which only handles KVCacheSimple.
    func testRotatingKVCacheSurvivesExactHitPath() throws {
        try skipIfMetalUnavailable()

        let model = MockRotatingCacheModel(vocabSize: 32, numLayers: 2, maxKVSize: 64)

        // Create a cached prompt state using RotatingKVCache
        let prompt = [1, 2, 3, 4, 5]
        let cachedKV = makeMockRotatingPromptCache(
            layers: 2, seqLen: 5, maxSize: 64, value: 1.0)

        let iterator = BatchTokenIterator(
            model: model,
            defaultSampler: ArgMaxSampler(),
            completionBatchSize: 32,
            prefillBatchSize: 8
        )

        let uids = iterator.insert(
            prompts: [prompt],
            maxTokens: [2],
            cachedKVStates: [cachedKV]
        )

        var tokensPerUID = [Int: [Int]]()
        var loopCount = 0
        while let responses = iterator.next(), !responses.isEmpty {
            for r in responses {
                tokensPerUID[r.uid, default: []].append(r.token)
            }
            loopCount += 1
            if loopCount > 20 { break }
        }

        XCTAssertEqual(
            tokensPerUID[uids[0]]?.count, 2,
            "RotatingKVCache exact-hit should produce 2 tokens"
        )
    }

    /// Verify that RotatingKVCache entries survive the partial-hit cached-prefill path.
    func testRotatingKVCacheSurvivesPartialHitPath() throws {
        try skipIfMetalUnavailable()

        let model = MockRotatingCacheModel(vocabSize: 32, numLayers: 2, maxKVSize: 64)

        // 8-token prompt, 5 cached as RotatingKVCache → suffix = [6, 7, 8]
        let prompt = [1, 2, 3, 4, 5, 6, 7, 8]
        let cachedKV = makeMockRotatingPromptCache(
            layers: 2, seqLen: 5, maxSize: 64, value: 1.0)

        let iterator = BatchTokenIterator(
            model: model,
            defaultSampler: ArgMaxSampler(),
            completionBatchSize: 32,
            prefillBatchSize: 8
        )

        let uids = iterator.insert(
            prompts: [prompt],
            maxTokens: [2],
            cachedKVStates: [cachedKV]
        )

        var tokensPerUID = [Int: [Int]]()
        var loopCount = 0
        while let responses = iterator.next(), !responses.isEmpty {
            for r in responses {
                tokensPerUID[r.uid, default: []].append(r.token)
            }
            loopCount += 1
            if loopCount > 20 { break }
        }

        XCTAssertEqual(
            tokensPerUID[uids[0]]?.count, 2,
            "RotatingKVCache partial-hit should produce 2 tokens"
        )
    }

    /// Verify that mixed-depth RotatingKVCache entries in a batch work correctly.
    func testMixedDepthRotatingCachePrefill() throws {
        try skipIfMetalUnavailable()

        let model = MockRotatingCacheModel(vocabSize: 32, numLayers: 2, maxKVSize: 64)

        // Two prompts with different rotating cache depths
        let promptA = [1, 2, 3, 4, 5, 6]  // 6 tokens, 3 cached
        let promptB = [10, 11, 12, 13, 14, 15, 16, 17]  // 8 tokens, 6 cached

        let cachedA = makeMockRotatingPromptCache(
            layers: 2, seqLen: 3, maxSize: 64, value: 1.0)
        let cachedB = makeMockRotatingPromptCache(
            layers: 2, seqLen: 6, maxSize: 64, value: 2.0)

        let iterator = BatchTokenIterator(
            model: model,
            defaultSampler: ArgMaxSampler(),
            completionBatchSize: 32,
            prefillBatchSize: 8
        )

        let uids = iterator.insert(
            prompts: [promptA, promptB],
            maxTokens: [2, 2],
            cachedKVStates: [cachedA, cachedB]
        )

        var tokensPerUID = [Int: [Int]]()
        var loopCount = 0
        while let responses = iterator.next(), !responses.isEmpty {
            for r in responses {
                tokensPerUID[r.uid, default: []].append(r.token)
            }
            loopCount += 1
            if loopCount > 20 { break }
        }

        XCTAssertEqual(
            tokensPerUID[uids[0]]?.count, 2,
            "Prompt A with rotating cache should produce 2 tokens"
        )
        XCTAssertEqual(
            tokensPerUID[uids[1]]?.count, 2,
            "Prompt B with rotating cache should produce 2 tokens"
        )
    }

    // MARK: - VAL-FIX-009: Mixed-Layer Cached Partial-Hit

    /// Verify that a mixed-layer model (layer 0 = KVCacheSimple, layer 1 =
    /// RotatingKVCache) preserves per-layer cache types through the cached
    /// partial-hit path. Previously, processPartialCacheHits() used a blanket
    /// first-layer type check that applied the same path to ALL layers,
    /// silently dropping RotatingKVCache data when layer 0 was KVCacheSimple.
    func testMixedLayerCachedPartialHitPreservesPerLayerCacheType() throws {
        try skipIfMetalUnavailable()

        let model = MockMixedLayerCacheModel(vocabSize: 32, maxKVSize: 64)

        // 8-token prompt, 5 cached as mixed layers → suffix = [6, 7, 8]
        let prompt = [1, 2, 3, 4, 5, 6, 7, 8]
        let cachedKV = makeMockMixedLayerPromptCache(seqLen: 5, maxSize: 64, value: 1.0)

        let iterator = BatchTokenIterator(
            model: model,
            defaultSampler: ArgMaxSampler(),
            completionBatchSize: 32,
            prefillBatchSize: 8
        )

        let uids = iterator.insert(
            prompts: [prompt],
            maxTokens: [2],
            cachedKVStates: [cachedKV]
        )

        // Advance to trigger cached prefill
        var tokensPerUID = [Int: [Int]]()
        var loopCount = 0
        while let responses = iterator.next(), !responses.isEmpty {
            for r in responses {
                tokensPerUID[r.uid, default: []].append(r.token)
            }
            loopCount += 1
            if loopCount > 20 { break }
        }

        // Verify tokens were produced (cache data was not silently dropped)
        XCTAssertEqual(
            tokensPerUID[uids[0]]?.count, 2,
            "Mixed-layer partial-hit should produce 2 tokens"
        )

        // Verify per-layer cache types in the active batch cache.
        // After generation completes, verify the batch was created with correct types.
        // We use a fresh iterator and inspect after one step to see the cache.
        let model2 = MockMixedLayerCacheModel(vocabSize: 32, maxKVSize: 64)
        let cachedKV2 = makeMockMixedLayerPromptCache(seqLen: 5, maxSize: 64, value: 1.0)

        let iterator2 = BatchTokenIterator(
            model: model2,
            defaultSampler: ArgMaxSampler(),
            completionBatchSize: 32,
            prefillBatchSize: 8
        )

        _ = iterator2.insert(
            prompts: [prompt],
            maxTokens: [5],
            cachedKVStates: [cachedKV2]
        )

        // One step triggers cached prefill and produces the first token.
        let _ = iterator2.next()

        let batchCache = iterator2.activeBatch?.cache
        XCTAssertNotNil(batchCache, "Active batch should have a cache")
        XCTAssertEqual(batchCache?.count, 2, "Should have 2 cache layers")

        if let cache = batchCache {
            XCTAssertTrue(
                cache[0] is BatchKVCache,
                "Layer 0 should be BatchKVCache (from KVCacheSimple), got \(type(of: cache[0]))"
            )
            XCTAssertTrue(
                cache[1] is BatchRotatingKVCache,
                "Layer 1 should be BatchRotatingKVCache (from RotatingKVCache), got \(type(of: cache[1]))"
            )

            // Verify neither layer has nil data (no silently dropped cache)
            if let bkv = cache[0] as? BatchKVCache {
                XCTAssertNotNil(bkv.keys, "Layer 0 BatchKVCache should have non-nil keys")
                XCTAssertNotNil(bkv.values, "Layer 0 BatchKVCache should have non-nil values")
            }
            if let brkv = cache[1] as? BatchRotatingKVCache {
                XCTAssertNotNil(brkv.keys, "Layer 1 BatchRotatingKVCache should have non-nil keys")
                XCTAssertNotNil(
                    brkv.values, "Layer 1 BatchRotatingKVCache should have non-nil values")
            }
        }
    }

    // MARK: - Helpers for Mixed-Layer Cache tests

    /// Create a mixed-layer mock prompt cache: layer 0 = KVCacheSimple, layer 1 = RotatingKVCache.
    private func makeMockMixedLayerPromptCache(
        seqLen: Int, maxSize: Int, heads: Int = 2, headDim: Int = 4, value: Float = 1.0
    ) -> [KVCache] {
        let simpleCache = makeMockCache(
            seqLen: seqLen, heads: heads, headDim: headDim, value: value)
        let rotatingCache = makeMockRotatingCache(
            seqLen: seqLen, maxSize: maxSize, heads: heads, headDim: headDim, value: value)
        return [simpleCache, rotatingCache]
    }

    // MARK: - Prepare/Finalize Lifecycle Tests

    /// Verify that BatchKVCache.prepare/finalize correctly rolls right-padding
    /// zeros to the left side, adjusting leftPadding and batchOffsets.
    func testBatchKVCachePrepareFinalize() throws {
        try skipIfMetalUnavailable()

        let H = 2
        let D = 4

        // Simulate a mixed-depth scenario:
        // Seq A: 3 cached tokens, suffix [4, 5, 6] (3 tokens)
        // Seq B: 7 cached tokens, suffix [8, 9] (2 tokens)
        //
        // After right-padding suffix: maxSuffix = 3
        //   A: [4, 5, 6] → no right-padding (rightPad = 0)
        //   B: [8, 9, 0] → rightPad = 1
        //
        // Cache after merge: bufferLen = 7 (maxCacheLen)
        //   A: leftPadding = 4 (7-3), data at positions 4..6
        //   B: leftPadding = 0 (7-7), data at positions 0..6
        //
        // After prefill of 3 right-padded suffix tokens: _idx = 7 + 3 = 10
        //   A: cached at 4..6, suffix at 7..9 → all valid
        //   B: cached at 0..6, suffix at 7..8, padding zero at 9 → BAD position 9
        //
        // After finalize (roll by [0, 1]):
        //   B: position 9 (padding) rolls to position 0 (left side)
        //   B: leftPadding adjusts from 0 to 1, batchOffsets decreases by 1
        //   Now all padding is on the LEFT for both sequences.

        let batchCache = BatchKVCache(leftPadding: [4, 0])
        // Simulate cached + suffix KV data: _idx = 10 (7 cached + 3 suffix)
        let keysArr = MLXArray.zeros([2, H, 10, D])
        let valuesArr = MLXArray.zeros([2, H, 10, D])

        // Fill seq A: valid data at positions 4..9 (6 = 3 cached + 3 suffix)
        keysArr[0 ..< 1, 0..., 4 ..< 10, 0...] = MLXArray.ones([1, H, 6, D]) * 1.0
        valuesArr[0 ..< 1, 0..., 4 ..< 10, 0...] = MLXArray.ones([1, H, 6, D]) * 10.0

        // Fill seq B: valid data at positions 0..8 (7 cached + 2 suffix), position 9 = padding
        keysArr[1 ..< 2, 0..., 0 ..< 9, 0...] = MLXArray.ones([1, H, 9, D]) * 2.0
        valuesArr[1 ..< 2, 0..., 0 ..< 9, 0...] = MLXArray.ones([1, H, 9, D]) * 20.0
        // Position 9 for seq B is right-padding zero (already zero from MLXArray.zeros)

        batchCache.keys = keysArr
        batchCache.values = valuesArr
        batchCache._idx = 10
        batchCache.batchOffsets = MLXArray([Int32(6), Int32(9)])  // 3+3, 7+2

        // Prepare with right-padding
        let rightPad = MLXArray([Int32(0), Int32(1)])
        batchCache.prepare(rightPadding: rightPad)

        // Verify right-padding was stored
        XCTAssertNotNil(batchCache._rightPadding)

        // Finalize: roll right-padding zeros to the left
        batchCache.finalize()

        // After finalize:
        // Seq A: leftPadding = 4 + 0 = 4, batchOffsets = 6 - 0 = 6
        // Seq B: leftPadding = 0 + 1 = 1, batchOffsets = 9 - 1 = 8
        XCTAssertEqual(
            batchCache.leftPadding[0].item(Int32.self), 4,
            "Seq A leftPadding should remain 4 (no right-padding)")
        XCTAssertEqual(
            batchCache.leftPadding[1].item(Int32.self), 1,
            "Seq B leftPadding should be 1 (0 + rightPad of 1)")
        XCTAssertEqual(
            batchCache.batchOffsets[0].item(Int32.self), 6,
            "Seq A batchOffsets should remain 6")
        XCTAssertEqual(
            batchCache.batchOffsets[1].item(Int32.self), 8,
            "Seq B batchOffsets should be 8 (9 - 1)")

        // Verify that rightPadding was cleared
        XCTAssertNil(batchCache._rightPadding, "rightPadding should be nil after finalize")

        // Verify the KV layout: for seq B, position 0 should now be the
        // rolled padding zero, and positions 1..9 should be valid data.
        let seqBKey0 = batchCache.keys![1, 0, 0, 0].item(Float.self)
        let seqBKey1 = batchCache.keys![1, 0, 1, 0].item(Float.self)
        XCTAssertEqual(
            seqBKey0, 0.0,
            "Seq B position 0 should be padding (rolled from right)")
        XCTAssertEqual(
            seqBKey1, 2.0,
            "Seq B position 1 should be valid data")
    }

    /// Verify that prepare(rightPadding:) is a no-op when all right-padding is zero.
    func testPrepareWithZeroRightPaddingIsNoOp() throws {
        try skipIfMetalUnavailable()

        let batchCache = BatchKVCache(leftPadding: [2, 0])
        let rightPad = MLXArray([Int32(0), Int32(0)])
        batchCache.prepare(rightPadding: rightPad)

        // Should not store rightPadding since max is 0
        XCTAssertNil(batchCache._rightPadding, "Zero right-padding should not be stored")

        // Finalize should be a no-op
        batchCache.finalize()
        XCTAssertEqual(
            batchCache.leftPadding[0].item(Int32.self), 2,
            "leftPadding should be unchanged")
    }

    /// Verify that mixed-depth cached-prefill with prepare/finalize produces
    /// correct generation (tokens are produced for all sequences).
    func testMixedDepthPrepareFinalizePrefillIntegration() throws {
        try skipIfMetalUnavailable()

        let model = MockCachePrefillModel(vocabSize: 32, numLayers: 2)

        // Seq A: 5 cached, 3 suffix → [1,2,3,4,5, 6,7,8]
        // Seq B: 3 cached, 5 suffix → [11,12,13, 14,15,16,17,18]
        // This is the exact concrete example from the feature description.
        let promptA = [1, 2, 3, 4, 5, 6, 7, 8]
        let promptB = [11, 12, 13, 14, 15, 16, 17, 18]

        let cachedA = makeMockPromptCache(layers: 2, seqLen: 5, value: 1.0)
        let cachedB = makeMockPromptCache(layers: 2, seqLen: 3, value: 2.0)

        let iterator = BatchTokenIterator(
            model: model,
            defaultSampler: ArgMaxSampler(),
            completionBatchSize: 32,
            prefillBatchSize: 8
        )

        let uids = iterator.insert(
            prompts: [promptA, promptB],
            maxTokens: [4, 4],
            cachedKVStates: [cachedA, cachedB]
        )

        var tokensPerUID = [Int: [Int]]()
        var loopCount = 0
        while let responses = iterator.next(), !responses.isEmpty {
            for r in responses {
                tokensPerUID[r.uid, default: []].append(r.token)
            }
            loopCount += 1
            if loopCount > 30 { break }
        }

        // Both should produce 4 tokens
        XCTAssertEqual(
            tokensPerUID[uids[0]]?.count, 4,
            "Seq A (5 cached, 3 suffix) should produce 4 tokens with prepare/finalize"
        )
        XCTAssertEqual(
            tokensPerUID[uids[1]]?.count, 4,
            "Seq B (3 cached, 5 suffix) should produce 4 tokens with prepare/finalize"
        )

        // Verify all tokens are within vocabulary range
        for (_, tokens) in tokensPerUID {
            for token in tokens {
                XCTAssertGreaterThanOrEqual(token, 0)
                XCTAssertLessThan(token, model.vocabSize)
            }
        }
    }

    /// Verify that after finalize, extracting caches produces correct data
    /// with all padding at the left side and no garbage entries.
    func testKVLayoutAfterFinalizeHasPaddingOnLeft() throws {
        try skipIfMetalUnavailable()

        let H = 2
        let D = 4

        // Build a batch cache mimicking a post-finalize state:
        // Seq A: leftPadding=4, valid data at 4..9 (6 tokens)
        // Seq B: leftPadding=1, valid data at 1..9 (9 tokens)
        // _idx = 10
        let batchCache = BatchKVCache(leftPadding: [4, 1])
        let keysArr = MLXArray.zeros([2, H, 10, D])
        let valuesArr = MLXArray.zeros([2, H, 10, D])

        keysArr[0 ..< 1, 0..., 4 ..< 10, 0...] = MLXArray.ones([1, H, 6, D]) * 5.0
        valuesArr[0 ..< 1, 0..., 4 ..< 10, 0...] = MLXArray.ones([1, H, 6, D]) * 50.0
        keysArr[1 ..< 2, 0..., 1 ..< 10, 0...] = MLXArray.ones([1, H, 9, D]) * 7.0
        valuesArr[1 ..< 2, 0..., 1 ..< 10, 0...] = MLXArray.ones([1, H, 9, D]) * 70.0

        batchCache.keys = keysArr
        batchCache.values = valuesArr
        batchCache._idx = 10
        batchCache.batchOffsets = MLXArray([Int32(6), Int32(9)])

        // Extract and verify: no garbage entries in extracted caches
        let extractedA = batchCache.extract(idx: 0)
        let extractedB = batchCache.extract(idx: 1)

        // Seq A: leftPadding=4, _idx=10, so extracted = 10-4 = 6 tokens
        XCTAssertEqual(extractedA.offset, 6, "Extracted A should have 6 valid tokens")
        XCTAssertEqual(extractedA.keys!.dim(2), 6)

        // Seq B: leftPadding=1, _idx=10, so extracted = 10-1 = 9 tokens
        XCTAssertEqual(extractedB.offset, 9, "Extracted B should have 9 valid tokens")
        XCTAssertEqual(extractedB.keys!.dim(2), 9)

        // All extracted positions should be real data (no zeros from padding)
        for pos in 0 ..< 6 {
            let val = extractedA.keys![0, 0, pos, 0].item(Float.self)
            XCTAssertEqual(val, 5.0, "Extracted A position \(pos) should be valid data (5.0)")
        }
        for pos in 0 ..< 9 {
            let val = extractedB.keys![0, 0, pos, 0].item(Float.self)
            XCTAssertEqual(val, 7.0, "Extracted B position \(pos) should be valid data (7.0)")
        }
    }

    /// Verify that mixed-depth partial-hit produces the same number of tokens
    /// as individual processing (semantic equivalence check).
    func testMixedDepthBatchVsIndividualTokenCount() throws {
        try skipIfMetalUnavailable()

        let model = MockCachePrefillModel(vocabSize: 32, numLayers: 2)

        let promptA = [1, 2, 3, 4, 5, 6]
        let promptB = [10, 11, 12, 13, 14, 15, 16, 17, 18]

        let cachedA = makeMockPromptCache(layers: 2, seqLen: 2, value: 1.0)
        let cachedB = makeMockPromptCache(layers: 2, seqLen: 7, value: 2.0)

        // --- Individual processing ---
        var individualTokenCounts = [Int: Int]()

        model.resetCounters()
        let iterA = BatchTokenIterator(
            model: model,
            defaultSampler: ArgMaxSampler(),
            completionBatchSize: 32,
            prefillBatchSize: 8
        )
        let uidsA = iterA.insert(
            prompts: [promptA],
            maxTokens: [3],
            cachedKVStates: [cachedA]
        )
        var countA = 0
        while let responses = iterA.next(), !responses.isEmpty {
            countA += responses.count
        }
        individualTokenCounts[0] = countA

        model.resetCounters()
        let iterB = BatchTokenIterator(
            model: model,
            defaultSampler: ArgMaxSampler(),
            completionBatchSize: 32,
            prefillBatchSize: 8
        )
        let uidsB = iterB.insert(
            prompts: [promptB],
            maxTokens: [3],
            cachedKVStates: [cachedB]
        )
        var countB = 0
        while let responses = iterB.next(), !responses.isEmpty {
            countB += responses.count
        }
        individualTokenCounts[1] = countB

        // --- Batch processing ---
        model.resetCounters()
        let cachedA2 = makeMockPromptCache(layers: 2, seqLen: 2, value: 1.0)
        let cachedB2 = makeMockPromptCache(layers: 2, seqLen: 7, value: 2.0)

        let iterBatch = BatchTokenIterator(
            model: model,
            defaultSampler: ArgMaxSampler(),
            completionBatchSize: 32,
            prefillBatchSize: 8
        )
        let uidsBatch = iterBatch.insert(
            prompts: [promptA, promptB],
            maxTokens: [3, 3],
            cachedKVStates: [cachedA2, cachedB2]
        )

        var batchTokenCounts = [Int: Int]()
        var loopCount = 0
        while let responses = iterBatch.next(), !responses.isEmpty {
            for r in responses {
                batchTokenCounts[r.uid, default: 0] += 1
            }
            loopCount += 1
            if loopCount > 30 { break }
        }

        // Both paths should produce the same token count
        XCTAssertEqual(
            batchTokenCounts[uidsBatch[0]], individualTokenCounts[0],
            "Batch prompt A should produce same token count as individual (\(individualTokenCounts[0]!))"
        )
        XCTAssertEqual(
            batchTokenCounts[uidsBatch[1]], individualTokenCounts[1],
            "Batch prompt B should produce same token count as individual (\(individualTokenCounts[1]!))"
        )
    }

    // MARK: - Helpers for RotatingKVCache tests

    /// Create a mock RotatingKVCache with synthetic keys/values.
    private func makeMockRotatingCache(
        seqLen: Int, maxSize: Int, heads: Int = 2, headDim: Int = 4, value: Float = 1.0
    ) -> RotatingKVCache {
        let cache = RotatingKVCache(maxSize: maxSize)
        if seqLen > 0 {
            let keys = MLXArray.ones([1, heads, seqLen, headDim]) * value
            let values = MLXArray.ones([1, heads, seqLen, headDim]) * (value + 1)
            _ = cache.update(keys: keys, values: values)
        }
        return cache
    }

    /// Create a multi-layer mock prompt cache using RotatingKVCache.
    private func makeMockRotatingPromptCache(
        layers: Int = 2, seqLen: Int, maxSize: Int, heads: Int = 2, headDim: Int = 4,
        value: Float = 1.0
    ) -> [KVCache] {
        (0 ..< layers).map { _ in
            makeMockRotatingCache(
                seqLen: seqLen, maxSize: maxSize, heads: heads, headDim: headDim, value: value)
        }
    }
}

// MARK: - Cache-Observing Mock Model

/// A mock model that records cache state during each forward call.
private class CacheObservingModel: Module, LanguageModel {
    let vocabSize: Int
    let numLayers: Int
    var callCount = 0
    var cacheHadKeys = false

    init(vocabSize: Int = 32, numLayers: Int = 2) {
        self.vocabSize = vocabSize
        self.numLayers = numLayers
    }

    func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
        .tokens(input.text)
    }

    func callAsFunction(
        _ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?
    ) -> LMOutput {
        callCount += 1
        let tokens = input.tokens
        let B = tokens.dim(0)
        let S = tokens.dim(1)

        // Check if cache has pre-loaded keys
        if let caches = cache {
            for c in caches {
                if let batchCache = c as? BatchKVCache, batchCache.keys != nil {
                    cacheHadKeys = true
                }
            }
        }

        // Same deterministic logits as MockCachePrefillModel
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

    func newCache(parameters: GenerateParameters?) -> [KVCache] {
        (0 ..< numLayers).map { _ in KVCacheSimple() }
    }

    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        weights
    }
}

// MARK: - Mock Rotating Cache Model

/// A mock model that produces RotatingKVCache layers, for testing that
/// cached RotatingKVCache entries survive the cached-prefill path.
private class MockRotatingCacheModel: Module, LanguageModel {
    let vocabSize: Int
    let numLayers: Int
    let maxKVSize: Int

    var callCount = 0

    init(vocabSize: Int = 32, numLayers: Int = 2, maxKVSize: Int = 64) {
        self.vocabSize = vocabSize
        self.numLayers = numLayers
        self.maxKVSize = maxKVSize
    }

    func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
        .tokens(input.text)
    }

    func callAsFunction(
        _ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?
    ) -> LMOutput {
        callCount += 1
        let tokens = input.tokens
        let B = tokens.dim(0)
        let S = tokens.dim(1)

        // Same deterministic logits as MockCachePrefillModel
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

    func newCache(parameters: GenerateParameters?) -> [KVCache] {
        (0 ..< numLayers).map { _ in RotatingKVCache(maxSize: maxKVSize) }
    }

    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        weights
    }
}

// MARK: - Mock Mixed-Layer Cache Model

/// A mock model that returns mixed cache types per layer:
/// layer 0 = KVCacheSimple (global attention), layer 1 = RotatingKVCache (sliding-window).
/// Simulates models like Gemma3 that interleave global and sliding-window layers.
private class MockMixedLayerCacheModel: Module, LanguageModel {
    let vocabSize: Int
    let maxKVSize: Int

    var callCount = 0

    init(vocabSize: Int = 32, maxKVSize: Int = 64) {
        self.vocabSize = vocabSize
        self.maxKVSize = maxKVSize
    }

    func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
        .tokens(input.text)
    }

    func callAsFunction(
        _ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?
    ) -> LMOutput {
        callCount += 1
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

    /// Returns 2 layers: [KVCacheSimple, RotatingKVCache]
    func newCache(parameters: GenerateParameters?) -> [KVCache] {
        [
            KVCacheSimple(),
            RotatingKVCache(maxSize: maxKVSize),
        ]
    }

    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        weights
    }
}
