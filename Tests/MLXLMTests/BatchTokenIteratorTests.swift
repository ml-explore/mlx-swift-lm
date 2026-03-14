// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXNN
import XCTest

@testable import MLXLMCommon

// MARK: - Mock Language Model

/// A deterministic mock language model for batch token iterator tests.
///
/// Given input tokens of shape `[B, S]`, it produces logits of shape `[B, S, vocabSize]`
/// where the highest-logit token for each position is the sum of the input tokens modulo vocabSize.
/// This provides deterministic, input-dependent output suitable for verifying batch generation.
private class MockBatchLanguageModel: Module, LanguageModel {
    let vocabSize: Int
    let numLayers: Int

    /// Optional: token that should be produced after a certain number of steps per sequence.
    /// Maps uid -> step at which to force a stop token.
    var forceStopAtStep: [Int: Int] = [:]

    /// Track call count for verifying chunked prefill.
    var callCount = 0

    /// Track input shapes for verifying chunked prefill.
    var inputShapes: [[Int]] = []

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
        callCount += 1
        inputShapes.append(input.tokens.shape)

        let tokens = input.tokens
        let B = tokens.dim(0)
        let S = tokens.dim(1)

        // Build logits: for each position, create a one-hot-ish distribution
        // where the "predicted next token" = (sum of all input tokens for that batch) % vocabSize
        // This gives deterministic output based on input content.
        var logitsFlat = [Float]()
        for b in 0 ..< B {
            for s in 0 ..< S {
                // Use the last token in the sequence as the "prediction"
                // For single-token decode: this is just the input token
                // The predicted next token = (input_token + 1) % vocabSize
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

// MARK: - Tests

class BatchTokenIteratorTests: XCTestCase {

    // MARK: - VAL-ENGINE-001: Insert returns unique UIDs

    func testInsertReturnsUniqueUIDs() throws {
        try skipIfMetalUnavailable()

        let model = MockBatchLanguageModel()
        let iterator = BatchTokenIterator(
            model: model,
            completionBatchSize: 32,
            prefillBatchSize: 8
        )

        let uids1 = iterator.insert(prompts: [[1, 2, 3]], maxTokens: [10])
        let uids2 = iterator.insert(prompts: [[4, 5]], maxTokens: [10])
        let uids3 = iterator.insert(prompts: [[6, 7, 8, 9]], maxTokens: [10])

        // All UIDs should be unique
        let allUIDs = uids1 + uids2 + uids3
        XCTAssertEqual(Set(allUIDs).count, allUIDs.count, "All UIDs must be unique")
        XCTAssertEqual(allUIDs.count, 3)
    }

    // MARK: - VAL-ENGINE-002: Per-request maxTokens respected

    func testPerRequestMaxTokensRespected() throws {
        try skipIfMetalUnavailable()

        let model = MockBatchLanguageModel()
        let iterator = BatchTokenIterator(
            model: model,
            completionBatchSize: 32,
            prefillBatchSize: 8
        )

        // Insert two prompts with different maxTokens
        let uids = iterator.insert(
            prompts: [[1, 2], [3, 4]],
            maxTokens: [2, 5]
        )

        var tokensPerUID = [Int: [Int]]()
        var finishReasons = [Int: GenerateStopReason]()

        // Run generation until complete
        while let responses = iterator.next(), !responses.isEmpty {
            for r in responses {
                tokensPerUID[r.uid, default: []].append(r.token)
                if let reason = r.finishReason {
                    finishReasons[r.uid] = reason
                }
            }
        }

        // First request (maxTokens=2) should have at most 2 tokens
        XCTAssertLessThanOrEqual(tokensPerUID[uids[0]]?.count ?? 0, 2)
        // Second request (maxTokens=5) should have at most 5 tokens
        XCTAssertLessThanOrEqual(tokensPerUID[uids[1]]?.count ?? 0, 5)

        // Both should finish with .length (no stop tokens configured)
        XCTAssertEqual(finishReasons[uids[0]], .length)
        XCTAssertEqual(finishReasons[uids[1]], .length)
    }

    // MARK: - VAL-ENGINE-003: Prompts sorted by ascending length

    func testPromptsSortedByAscendingLength() throws {
        try skipIfMetalUnavailable()

        let model = MockBatchLanguageModel()
        let iterator = BatchTokenIterator(
            model: model,
            completionBatchSize: 32,
            prefillBatchSize: 8
        )

        // Insert prompts of varying lengths (not in order)
        let _ = iterator.insert(
            prompts: [[1, 2, 3, 4, 5], [6], [7, 8, 9]],
            maxTokens: [10, 10, 10]
        )

        // Check that pendingPrompts are sorted by length ascending
        let lengths = iterator.pendingPrompts.map(\.effectiveLength)
        XCTAssertEqual(lengths, lengths.sorted(), "Pending prompts should be sorted by length")
        XCTAssertEqual(lengths, [1, 3, 5])
    }

    // MARK: - VAL-ENGINE-004: Left-padding applied for variable-length sequences
    // (Verified implicitly through the processPrompts flow — left-padding is internal)

    func testLeftPaddingApplied() throws {
        try skipIfMetalUnavailable()

        let model = MockBatchLanguageModel()
        let iterator = BatchTokenIterator(
            model: model,
            completionBatchSize: 32,
            prefillBatchSize: 8
        )

        // Insert prompts of different lengths
        let _ = iterator.insert(
            prompts: [[1], [2, 3, 4]],
            maxTokens: [1, 1]
        )

        // Calling next() triggers prefill with left-padding
        // The mock model should receive a [2, 3] shaped input for the last-token step
        // (after chunked prefill of the first tokens)
        let responses = iterator.next()
        XCTAssertNotNil(responses)

        // Verify the model was called (prefill happened)
        XCTAssertGreaterThan(model.callCount, 0)
    }

    // MARK: - VAL-ENGINE-005: Prefill processes prompts in chunks of prefillStepSize

    func testPrefillChunkedByStepSize() throws {
        try skipIfMetalUnavailable()

        let model = MockBatchLanguageModel()
        // Use a small prefillStepSize to force chunking
        let iterator = BatchTokenIterator(
            model: model,
            completionBatchSize: 32,
            prefillBatchSize: 8,
            prefillStepSize: 3
        )

        // Insert a prompt with 8 tokens — should be chunked into steps of 3
        // With 8 tokens total, prefill processes all but last token = 7 tokens
        // Chunks: 3, 3, 1 (last token), then final step for sampling
        let _ = iterator.insert(
            prompts: [[1, 2, 3, 4, 5, 6, 7, 8]],
            maxTokens: [1]
        )

        let _ = iterator.next()

        // Verify model was called multiple times for chunked prefill
        // With 8 tokens and prefillStepSize=3:
        // Chunk 1: 3 tokens, Chunk 2: 3 tokens, remaining 2 tokens: 1 for final chunk, last 1 for step
        XCTAssertGreaterThan(model.callCount, 1, "Prefill should require multiple model calls")

        // Verify no chunk exceeds prefillStepSize
        for shape in model.inputShapes {
            if shape.count >= 2 {
                XCTAssertLessThanOrEqual(
                    shape[1], 3,
                    "No prefill chunk should exceed prefillStepSize")
            }
        }
    }

    // MARK: - VAL-ENGINE-006: Prefill transitions to decode phase

    func testPrefillTransitionsToDecode() throws {
        try skipIfMetalUnavailable()

        let model = MockBatchLanguageModel()
        let iterator = BatchTokenIterator(
            model: model,
            completionBatchSize: 32,
            prefillBatchSize: 8
        )

        let uids = iterator.insert(
            prompts: [[1, 2, 3]],
            maxTokens: [3]
        )

        // First next() call triggers prefill and produces first decode token
        let responses = iterator.next()
        XCTAssertNotNil(responses)
        XCTAssertEqual(responses?.count, 1)
        XCTAssertEqual(responses?.first?.uid, uids[0])

        // The token should be a valid token (non-negative)
        if let token = responses?.first?.token {
            XCTAssertGreaterThanOrEqual(token, 0)
            XCTAssertLessThan(token, model.vocabSize)
        }
    }

    // MARK: - VAL-ENGINE-007: Each next() produces one token per active sequence

    func testNextProducesOneTokenPerSequence() throws {
        try skipIfMetalUnavailable()

        let model = MockBatchLanguageModel()
        let iterator = BatchTokenIterator(
            model: model,
            completionBatchSize: 32,
            prefillBatchSize: 8
        )

        let uids = iterator.insert(
            prompts: [[1, 2], [3, 4], [5, 6]],
            maxTokens: [5, 5, 5]
        )

        // First next() triggers prefill and returns first tokens
        let responses = iterator.next()
        XCTAssertNotNil(responses)
        XCTAssertEqual(responses?.count, 3, "Should produce exactly one token per active sequence")

        // Verify each UID appears exactly once
        let responseUIDs = Set(responses?.map(\.uid) ?? [])
        XCTAssertEqual(responseUIDs, Set(uids))
    }

    // MARK: - VAL-ENGINE-008: Stop token terminates with reason .stop

    func testStopTokenTerminatesWithStop() throws {
        try skipIfMetalUnavailable()

        let stopToken = 5
        let model = MockBatchLanguageModel(vocabSize: 32)
        let iterator = BatchTokenIterator(
            model: model,
            stopTokens: [stopToken],
            completionBatchSize: 32,
            prefillBatchSize: 8
        )

        // Insert a prompt whose mock model output will eventually produce the stop token.
        // Mock model: predicted token = (input_token + 1) % vocabSize
        // So if the input token is (stopToken - 1) = 4, the output will be 5 (stop token).
        // We need to engineer a prompt that leads to the stop token.
        let promptToken = stopToken - 1  // = 4
        let uids = iterator.insert(
            prompts: [[promptToken]],
            maxTokens: [100]
        )

        var foundStop = false
        var loopCount = 0
        while let responses = iterator.next(), !responses.isEmpty {
            for r in responses {
                if r.finishReason == .stop {
                    foundStop = true
                    XCTAssertEqual(r.uid, uids[0])
                }
            }
            loopCount += 1
            if loopCount > 50 { break }  // Safety limit
        }

        XCTAssertTrue(foundStop, "Should have found a .stop finish reason")
    }

    // MARK: - VAL-ENGINE-009: Sequences finish independently

    func testSequencesFinishIndependently() throws {
        try skipIfMetalUnavailable()

        let model = MockBatchLanguageModel()
        let iterator = BatchTokenIterator(
            model: model,
            completionBatchSize: 32,
            prefillBatchSize: 8
        )

        // Two prompts with very different maxTokens
        let uids = iterator.insert(
            prompts: [[1, 2], [3, 4]],
            maxTokens: [1, 5]
        )

        var finishedUIDs = Set<Int>()
        var tokenCounts = [Int: Int]()
        var loopCount = 0

        while let responses = iterator.next(), !responses.isEmpty {
            for r in responses {
                tokenCounts[r.uid, default: 0] += 1
                if r.finishReason != nil {
                    finishedUIDs.insert(r.uid)
                }
            }
            loopCount += 1
            if loopCount > 20 { break }
        }

        // First prompt (maxTokens=1) should finish before second (maxTokens=5)
        XCTAssertTrue(finishedUIDs.contains(uids[0]))
        XCTAssertTrue(finishedUIDs.contains(uids[1]))

        // First should have generated fewer tokens
        XCTAssertLessThanOrEqual(tokenCounts[uids[0]] ?? 0, 1)
        XCTAssertGreaterThan(tokenCounts[uids[1]] ?? 0, 1)
    }

    // MARK: - VAL-ENGINE-010: completionBatchSize limits concurrent decode sequences

    func testCompletionBatchSizeLimits() throws {
        try skipIfMetalUnavailable()

        let model = MockBatchLanguageModel()
        // Set a small completionBatchSize
        let iterator = BatchTokenIterator(
            model: model,
            completionBatchSize: 2,
            prefillBatchSize: 2
        )

        // Insert 4 prompts — only 2 should be active at a time
        let _ = iterator.insert(
            prompts: [[1], [2], [3], [4]],
            maxTokens: [3, 3, 3, 3]
        )

        // First next: should prefill and start at most completionBatchSize sequences
        let responses = iterator.next()
        XCTAssertNotNil(responses)
        XCTAssertLessThanOrEqual(
            responses?.count ?? 0, 2,
            "Active batch should not exceed completionBatchSize"
        )
    }

    // MARK: - VAL-ENGINE-011: Remove active sequence mid-generation

    func testRemoveActiveSequence() throws {
        try skipIfMetalUnavailable()

        let model = MockBatchLanguageModel()
        let iterator = BatchTokenIterator(
            model: model,
            completionBatchSize: 32,
            prefillBatchSize: 8
        )

        let uids = iterator.insert(
            prompts: [[1, 2], [3, 4], [5, 6]],
            maxTokens: [10, 10, 10]
        )

        // First next() to start generation
        let _ = iterator.next()

        // Remove the second sequence mid-generation
        iterator.remove(uids: [uids[1]])

        // Next call should not include the removed UID
        if let responses = iterator.next() {
            let responseUIDs = Set(responses.map(\.uid))
            XCTAssertFalse(
                responseUIDs.contains(uids[1]),
                "Removed UID should not appear in responses"
            )
        }
    }

    // MARK: - VAL-ENGINE-011 (continued): Remove from pending queue

    func testRemoveFromPendingQueue() throws {
        try skipIfMetalUnavailable()

        let model = MockBatchLanguageModel()
        // Small completionBatchSize so not all prompts are prefilled at once
        let iterator = BatchTokenIterator(
            model: model,
            completionBatchSize: 1,
            prefillBatchSize: 1
        )

        let uids = iterator.insert(
            prompts: [[1], [2], [3]],
            maxTokens: [10, 10, 10]
        )

        // Remove a pending prompt before it's processed
        iterator.remove(uids: [uids[2]])

        // Verify it was removed from pending
        let pendingUIDs = iterator.pendingPrompts.map(\.uid)
        XCTAssertFalse(
            pendingUIDs.contains(uids[2]),
            "Removed UID should not be in pending queue"
        )
    }

    // MARK: - VAL-ENGINE-012: close() stops all generation

    func testCloseStopsGeneration() throws {
        try skipIfMetalUnavailable()

        let model = MockBatchLanguageModel()
        let iterator = BatchTokenIterator(
            model: model,
            completionBatchSize: 32,
            prefillBatchSize: 8
        )

        let _ = iterator.insert(
            prompts: [[1, 2, 3]],
            maxTokens: [100]
        )

        // Start generation
        let _ = iterator.next()

        // Close the iterator
        iterator.close()

        // After close, next() should return nil
        let result = iterator.next()
        XCTAssertNil(result, "next() should return nil after close()")
    }

    // MARK: - Additional: UID uniqueness across multiple insertions

    func testUIDUniquenessAcrossInsertions() throws {
        try skipIfMetalUnavailable()

        let model = MockBatchLanguageModel()
        let iterator = BatchTokenIterator(
            model: model,
            completionBatchSize: 32,
            prefillBatchSize: 8
        )

        var allUIDs = [Int]()
        for _ in 0 ..< 5 {
            let uids = iterator.insert(
                prompts: [[1], [2]],
                maxTokens: [1, 1]
            )
            allUIDs.append(contentsOf: uids)
        }

        XCTAssertEqual(
            Set(allUIDs).count, allUIDs.count,
            "UIDs must be unique across all insertions"
        )
        XCTAssertEqual(allUIDs.count, 10)
    }

    // MARK: - Empty batch returns empty responses

    func testEmptyBatchReturnsEmptyResponses() throws {
        try skipIfMetalUnavailable()

        let model = MockBatchLanguageModel()
        let iterator = BatchTokenIterator(
            model: model,
            completionBatchSize: 32,
            prefillBatchSize: 8
        )

        // Don't insert anything — next() should return empty
        let responses = iterator.next()
        XCTAssertNotNil(responses)
        XCTAssertTrue(responses?.isEmpty ?? false)
    }

    // MARK: - Full generation loop produces expected token count

    func testFullGenerationLoop() throws {
        try skipIfMetalUnavailable()

        let model = MockBatchLanguageModel()
        let iterator = BatchTokenIterator(
            model: model,
            completionBatchSize: 32,
            prefillBatchSize: 8
        )

        let maxToks = 3
        let uids = iterator.insert(
            prompts: [[10, 20]],
            maxTokens: [maxToks]
        )

        var totalTokens = 0
        var loopCount = 0
        while let responses = iterator.next(), !responses.isEmpty {
            for r in responses {
                XCTAssertEqual(r.uid, uids[0])
                totalTokens += 1
            }
            loopCount += 1
            if loopCount > 20 { break }
        }

        XCTAssertEqual(totalTokens, maxToks, "Should produce exactly maxTokens tokens")
    }
}
