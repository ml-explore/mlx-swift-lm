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

    /// Exact cache match: entire prompt is cached, only last token needs sampling.
    func testExactCacheMatchMinimalPrefill() throws {
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

        // When the cache covers the entire prompt, only the last token needs sampling.
        // This results in just 1 model call with 1 token.
        XCTAssertEqual(
            model.callCount, 1,
            "Exact cache match should require only 1 model call for sampling"
        )
        XCTAssertEqual(
            model.totalTokensProcessed, 1,
            "Exact cache match should process only 1 token"
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
    }
}
