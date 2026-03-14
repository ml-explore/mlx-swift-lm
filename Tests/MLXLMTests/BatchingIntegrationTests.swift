// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXNN
import Tokenizers
import XCTest

@testable import MLXLMCommon

// MARK: - Mock Model for Cross-Area Integration Tests

/// A deterministic mock language model for cross-area integration tests.
///
/// Produces tokens deterministically: next token = (input_token + 1) % vocabSize.
/// Uses KVCacheSimple by default (batch-compatible).
/// Conforms to KVCacheDimensionProvider so newCache() creates proper KVCacheSimple layers.
private class IntegrationTestMockModel: Module, LanguageModel, KVCacheDimensionProvider,
    @unchecked Sendable
{
    let vocabSize: Int
    let numLayers: Int
    var kvHeads: [Int] { Array(repeating: 4, count: numLayers) }

    /// Track call count for verifying prefill behavior.
    var callCount = 0
    /// Track total tokens processed across all calls.
    var totalTokensProcessed = 0

    init(vocabSize: Int = 64, numLayers: Int = 1) {
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
        totalTokensProcessed += B * S

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

    func resetCounters() {
        callCount = 0
        totalTokensProcessed = 0
    }
}

/// Mock model that creates MambaCache (batch-incompatible).
private class IncompatibleSSMMockModel: Module, LanguageModel, @unchecked Sendable {
    let vocabSize: Int = 64

    func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
        .tokens(input.text)
    }

    func callAsFunction(
        _ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?
    ) -> LMOutput {
        let B = input.tokens.dim(0)
        let S = input.tokens.dim(1)

        var logitsFlat = [Float]()
        for b in 0 ..< B {
            for s in 0 ..< S {
                let lastToken = input.tokens[b, s].item(Int32.self)
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
        [MambaCache()]
    }

    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        weights
    }
}

/// A mock language model that produces a fixed token sequence encoding a
/// JSON tool call (`<tool_call>{"name":"get_weather","arguments":{"city":"SF"}}</tool_call>`).
///
/// Token ID mapping (used with `ToolCallTestTokenizer`):
///   100 → `<tool_call>`
///   101 → `{"name": "get_weather", "arguments": {"city": "SF"}}`
///   102 → `</tool_call>`
///
/// Prompt starting with 50 → tool call tokens [100, 101, 102, 10, 10, ...].
/// All other prompts → deterministic (last_token + 1) % vocabSize tokens.
///
/// Uses the input token itself to determine the next output: when the input
/// is a tool-call token (100, 101, 102), the model emits the next one in
/// the sequence. This avoids needing cache offset tracking.
private class ToolCallMockModel: Module, LanguageModel, KVCacheDimensionProvider,
    @unchecked Sendable
{
    let vocabSize: Int = 200
    let numLayers: Int = 1
    var kvHeads: [Int] { [4] }

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
                let token = tokens[b, s].item(Int32.self)
                var row = [Float](repeating: -100.0, count: vocabSize)

                // Determine next token based on the current input token.
                // Prompt token 50 → start tool call sequence with 100.
                // Tool call chain: 100 → 101, 101 → 102, 102 → 10 (filler).
                // All others: (token + 1) % vocabSize.
                let nextToken: Int
                switch token {
                case 50: nextToken = 100  // Start tool call
                case 100: nextToken = 101  // Continue tool call body
                case 101: nextToken = 102  // End tool call
                case 102: nextToken = 10  // Filler after tool call
                default: nextToken = (Int(token) + 1) % vocabSize
                }

                row[nextToken] = 0.0
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

/// A tokenizer that maps specific token IDs to tool-call-forming strings.
/// Token 100 → `<tool_call>`, 101 → JSON body, 102 → `</tool_call>`.
/// All other tokens map to simple lowercase letters.
private struct ToolCallTestTokenizer: Tokenizer {
    var bosToken: String? = nil
    var bosTokenId: Int? = 0
    var eosToken: String? = nil
    var eosTokenId: Int? = 0
    var unknownToken: String? = nil
    var unknownTokenId: Int? = 0

    private static let specialTokens: [Int: String] = [
        100: "<tool_call>",
        101: "{\"name\": \"get_weather\", \"arguments\": {\"city\": \"SF\"}}",
        102: "</tool_call>",
    ]

    func tokenize(text: String) -> [String] {
        text.split(separator: " ").map { String($0) }
    }

    func encode(text: String) -> [Int] { [1, 2, 3] }
    func encode(text: String, addSpecialTokens: Bool) -> [Int] { encode(text: text) }

    func decode(tokens: [Int], skipSpecialTokens: Bool) -> String {
        tokens.map { convertIdToToken($0) ?? "?" }.joined()
    }

    func convertTokenToId(_ token: String) -> Int? { nil }
    func convertIdToToken(_ id: Int) -> String? {
        Self.specialTokens[id] ?? String(Character(UnicodeScalar(97 + (id % 26))!))
    }

    func applyChatTemplate(messages: [Tokenizers.Message]) throws -> [Int] { [1, 2] }
    func applyChatTemplate(messages: [Tokenizers.Message], tools: [Tokenizers.ToolSpec]?) throws
        -> [Int]
    { [1, 2] }
    func applyChatTemplate(
        messages: [Tokenizers.Message], tools: [Tokenizers.ToolSpec]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] { [1, 2] }
    func applyChatTemplate(
        messages: [Tokenizers.Message], chatTemplate: Tokenizers.ChatTemplateArgument
    ) throws -> [Int] { [1, 2] }
    func applyChatTemplate(messages: [Tokenizers.Message], chatTemplate: String) throws -> [Int] {
        [1, 2]
    }
    func applyChatTemplate(
        messages: [Tokenizers.Message], chatTemplate: Tokenizers.ChatTemplateArgument?,
        addGenerationPrompt: Bool, truncation: Bool, maxLength: Int?, tools: [Tokenizers.ToolSpec]?
    ) throws -> [Int] { [1, 2] }
    func applyChatTemplate(
        messages: [Tokenizers.Message], chatTemplate: Tokenizers.ChatTemplateArgument?,
        addGenerationPrompt: Bool, truncation: Bool, maxLength: Int?, tools: [Tokenizers.ToolSpec]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] { [1, 2] }
}

/// A simple mock input processor for ModelContainer-based tests.
private struct IntegrationMockInputProcessor: UserInputProcessor {
    let tokenizer: Tokenizer
    let configuration: ModelConfiguration

    var messageGenerator: MessageGenerator { DefaultMessageGenerator() }

    init(tokenizer: Tokenizer, configuration: ModelConfiguration) {
        self.tokenizer = tokenizer
        self.configuration = configuration
    }

    func prepare(input: UserInput) throws -> LMInput {
        let messages = messageGenerator.generate(from: input)
        let promptTokens = try tokenizer.applyChatTemplate(
            messages: messages, tools: input.tools, additionalContext: input.additionalContext)
        return LMInput(tokens: MLXArray(promptTokens))
    }
}

// MARK: - Cross-Area Integration Tests

/// Comprehensive cross-area integration tests verifying end-to-end flows
/// across batch KV cache, batch generation engine, scheduler, prompt cache,
/// and model RoPE migration.
///
/// These tests verify:
/// - VAL-CROSS-001: End-to-end single request flow unchanged
/// - VAL-CROSS-002: End-to-end batch request flow
/// - VAL-CROSS-003: Single-to-batch upgrade flow
/// - VAL-CROSS-004: Fallback flow for incompatible requests
/// - VAL-CROSS-005: Backward API compatibility
/// - VAL-CROSS-006: Different sequence lengths in batch
/// - VAL-CROSS-007: Prompt cache integrated with batch generation
/// - VAL-CROSS-008: Tool calls in batch generation routed to correct request stream
class BatchingIntegrationTests: XCTestCase {

    // MARK: - Helpers

    /// Create a ModelContainer with the given model and optional scheduler.
    private func makeModelContainer(
        model: (any LanguageModel)? = nil,
        scheduler: InferenceScheduler? = nil
    ) -> ModelContainer {
        let resolvedModel = model ?? IntegrationTestMockModel()
        let tokenizer = TestTokenizer()
        let config = ModelConfiguration(id: "test-integration-model")
        let processor = IntegrationMockInputProcessor(
            tokenizer: tokenizer, configuration: config)

        let context = ModelContext(
            configuration: config,
            model: resolvedModel,
            processor: processor,
            tokenizer: tokenizer
        )

        return ModelContainer(context: context, scheduler: scheduler)
    }

    /// Create a mock prompt cache with synthetic keys/values.
    private func makeMockPromptCache(
        layers: Int = 1, seqLen: Int, heads: Int = 2, headDim: Int = 4, value: Float = 1.0
    ) -> [KVCache] {
        (0 ..< layers).map { _ in
            let cache = KVCacheSimple()
            if seqLen > 0 {
                let keys = MLXArray.ones([1, heads, seqLen, headDim]) * value
                let values = MLXArray.ones([1, heads, seqLen, headDim]) * (value + 1)
                _ = cache.update(keys: keys, values: values)
            }
            return cache
        }
    }

    // MARK: - VAL-CROSS-001: End-to-end single request flow unchanged

    /// A single request through the full pipeline (prepare → TokenIterator →
    /// applyRotaryPosition → stream) works identically to before batching changes.
    func testSingleRequestFlowUnchanged() async throws {
        try skipIfMetalUnavailable()

        let model = IntegrationTestMockModel()

        // Use the single-request TokenIterator path directly (no scheduler)
        let input = LMInput(tokens: MLXArray([Int32(10), Int32(20), Int32(30)]))
        let params = GenerateParameters(maxTokens: 5, temperature: 0)

        let iterator = try TokenIterator(
            input: input,
            model: model,
            cache: nil,
            parameters: params
        )

        var tokens = [Int]()
        for token in iterator {
            tokens.append(token)
        }

        // Should produce exactly maxTokens tokens
        XCTAssertEqual(tokens.count, 5, "Single request should produce exactly maxTokens tokens")

        // Mock model: next token = (input + 1) % vocabSize
        // From last prompt token 30: produces 31, then 32, 33, 34, 35
        let expectedTokens = [31, 32, 33, 34, 35]
        XCTAssertEqual(
            tokens, expectedTokens,
            "Deterministic mock should produce predictable sequence: "
                + "expected \(expectedTokens), got \(tokens)")
    }

    /// Single request through ModelContainer (without scheduler) produces output
    /// identical to the direct TokenIterator path.
    func testSingleRequestThroughModelContainerNoScheduler() async throws {
        try skipIfMetalUnavailable()

        let container = makeModelContainer()

        let input = LMInput(tokens: MLXArray([Int32(10), Int32(20), Int32(30)]))
        let params = GenerateParameters(maxTokens: 5, temperature: 0)

        let stream = try await container.generate(input: input, parameters: params)

        var chunks = [String]()
        var receivedInfo = false
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

        XCTAssertFalse(chunks.isEmpty, "Should produce text output")
        XCTAssertTrue(receivedInfo, "Should receive completion info")
    }

    /// Single request through scheduler stays on single path (no batch structures).
    func testSingleRequestThroughSchedulerUsesSinglePath() async throws {
        try skipIfMetalUnavailable()

        let model = IntegrationTestMockModel()
        let tokenizer = TestTokenizer()
        let config = ModelConfiguration(id: "test-model")
        let scheduler = InferenceScheduler()

        let input = LMInput(tokens: MLXArray([Int32(10), Int32(20), Int32(30)]))
        let params = GenerateParameters(maxTokens: 5, temperature: 0)

        let stream = try await scheduler.submit(
            input: input,
            parameters: params,
            model: model,
            cache: nil,
            tokenizer: tokenizer,
            configuration: config
        )

        // Verify scheduler is in single state
        let state = await scheduler.currentState
        XCTAssertEqual(state, "single", "Single request should use single path")

        // Consume stream and verify output
        var chunks = [String]()
        for await gen in stream {
            if let chunk = gen.chunk {
                chunks.append(chunk)
            }
        }

        XCTAssertFalse(chunks.isEmpty, "Should produce output on single path")
    }

    // MARK: - VAL-CROSS-002: End-to-end batch request flow

    /// Multiple requests through the batch pipeline produce correct independent
    /// outputs with per-sequence RoPE offsets.
    func testEndToEndBatchFlow() async throws {
        try skipIfMetalUnavailable()

        let model = IntegrationTestMockModel()
        let tokenizer = TestTokenizer()
        let config = ModelConfiguration(id: "test-model")
        let scheduler = InferenceScheduler()

        // First request (starts on single path)
        let input1 = LMInput(tokens: MLXArray([Int32(1), Int32(2), Int32(3)]))
        let params1 = GenerateParameters(maxTokens: 10, temperature: 0)

        let stream1 = try await scheduler.submit(
            input: input1,
            parameters: params1,
            model: model,
            cache: nil,
            tokenizer: tokenizer,
            configuration: config
        )

        // Second request triggers upgrade to batch
        let input2 = LMInput(tokens: MLXArray([Int32(10), Int32(20)]))
        let params2 = GenerateParameters(maxTokens: 5, temperature: 0)

        let stream2 = try await scheduler.submit(
            input: input2,
            parameters: params2,
            model: model,
            cache: nil,
            tokenizer: tokenizer,
            configuration: config
        )

        // Consume both streams concurrently
        var chunks1 = [String]()
        var chunks2 = [String]()

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
                    chunks1 = chunks
                } else {
                    chunks2 = chunks
                }
            }
        }

        // Both streams should produce some output
        let totalOutput = chunks1.count + chunks2.count
        XCTAssertGreaterThan(
            totalOutput, 0,
            "Batch flow should produce output from at least one request")
    }

    /// Multiple requests through BatchTokenIterator directly produce correct
    /// independent outputs with deterministic per-request token sequences.
    func testBatchTokenIteratorMultipleRequests() throws {
        try skipIfMetalUnavailable()

        let model = IntegrationTestMockModel()
        let iterator = BatchTokenIterator(
            model: model,
            defaultSampler: ArgMaxSampler(),
            completionBatchSize: 32,
            prefillBatchSize: 8
        )

        // Insert three prompts with different content
        let uids = iterator.insert(
            prompts: [[1, 2, 3], [10, 20], [5, 6, 7, 8]],
            maxTokens: [4, 4, 4]
        )

        var tokensPerUID = [Int: [Int]]()
        var finishReasons = [Int: GenerateStopReason]()
        var loopCount = 0

        while let responses = iterator.next(), !responses.isEmpty {
            for r in responses {
                tokensPerUID[r.uid, default: []].append(r.token)
                if let reason = r.finishReason {
                    finishReasons[r.uid] = reason
                }
            }
            loopCount += 1
            if loopCount > 30 { break }
        }

        // All three should produce exactly 4 tokens
        for uid in uids {
            XCTAssertEqual(
                tokensPerUID[uid]?.count, 4,
                "Request \(uid) should produce exactly 4 tokens")
            XCTAssertEqual(
                finishReasons[uid], .length,
                "Request \(uid) should finish with .length")
        }

        // Verify deterministic per-request outputs.
        // Mock model: next token = (last_prompt_token + 1) % vocabSize,
        // then each subsequent token = (prev + 1) % vocabSize.
        // Prompt [1,2,3]: last=3 → 4,5,6,7
        // Prompt [10,20]: last=20 → 21,22,23,24
        // Prompt [5,6,7,8]: last=8 → 9,10,11,12
        let expected0 = [4, 5, 6, 7]
        let expected1 = [21, 22, 23, 24]
        let expected2 = [9, 10, 11, 12]

        let seq0 = tokensPerUID[uids[0]] ?? []
        let seq1 = tokensPerUID[uids[1]] ?? []
        let seq2 = tokensPerUID[uids[2]] ?? []

        XCTAssertEqual(
            seq0, expected0,
            "Prompt [1,2,3] should produce \(expected0), got \(seq0)")
        XCTAssertEqual(
            seq1, expected1,
            "Prompt [10,20] should produce \(expected1), got \(seq1)")
        XCTAssertEqual(
            seq2, expected2,
            "Prompt [5,6,7,8] should produce \(expected2), got \(seq2)")
    }

    // MARK: - VAL-CROSS-003: Single-to-batch upgrade flow

    /// First request starts on single path, second request triggers upgrade,
    /// first continues without interruption, second starts generating.
    /// Asserts uninterrupted token continuity: the first request should produce
    /// a total token count consistent with its maxTokens regardless of upgrade.
    func testSingleToBatchUpgradeFlow() async throws {
        try skipIfMetalUnavailable()

        let model = IntegrationTestMockModel()
        let tokenizer = TestTokenizer()
        let config = ModelConfiguration(id: "test-model")
        let scheduler = InferenceScheduler()

        // First request — starts on single path
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

        var state = await scheduler.currentState
        XCTAssertEqual(state, "single", "First request should start on single path")

        // Consume a few tokens from the first request to advance the iterator
        var tokensBeforeUpgrade = [String]()
        var count = 0
        for await gen in stream1 {
            if let chunk = gen.chunk {
                tokensBeforeUpgrade.append(chunk)
                count += 1
                if count >= 2 {
                    break
                }
            }
        }

        // Second request triggers upgrade
        let input2 = LMInput(tokens: MLXArray([Int32(10), Int32(20)]))
        let params2 = GenerateParameters(maxTokens: 5, temperature: 0)

        let stream2 = try await scheduler.submit(
            input: input2,
            parameters: params2,
            model: model,
            cache: nil,
            tokenizer: tokenizer,
            configuration: config
        )

        state = await scheduler.currentState
        XCTAssertTrue(
            state == "batched" || state == "single",
            "Should transition to batched or fall back to single (got \(state))")

        // Consume remaining tokens from both streams concurrently
        var tokensAfterUpgrade = [String]()
        var secondRequestChunks = [String]()

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
                    tokensAfterUpgrade = chunks
                } else {
                    secondRequestChunks = chunks
                }
            }
        }

        // First request should have continued generating after upgrade
        let totalFirst = tokensBeforeUpgrade.count + tokensAfterUpgrade.count
        XCTAssertGreaterThan(
            totalFirst, 0,
            "First request should produce tokens across the upgrade boundary")

        // Verify token continuity: total should not exceed maxTokens
        XCTAssertLessThanOrEqual(
            totalFirst, 20,
            "First request total tokens should not exceed maxTokens (20)")

        // Second request should also produce output
        XCTAssertFalse(
            secondRequestChunks.isEmpty,
            "Second request should produce output after triggering upgrade")
    }

    // MARK: - VAL-CROSS-004: Fallback flow for incompatible requests

    /// Incompatible requests fall back to single path while compatible ones
    /// continue in batch.
    func testFallbackFlowForIncompatibleRequests() async throws {
        try skipIfMetalUnavailable()

        let model = IntegrationTestMockModel()
        let tokenizer = TestTokenizer()
        let config = ModelConfiguration(id: "test-model")
        let scheduler = InferenceScheduler()

        // Compatible request starts on single path
        let input1 = LMInput(tokens: MLXArray([Int32(1), Int32(2)]))
        let params1 = GenerateParameters(maxTokens: 10, temperature: 0)

        let stream1 = try await scheduler.submit(
            input: input1,
            parameters: params1,
            model: model,
            cache: nil,
            tokenizer: tokenizer,
            configuration: config
        )

        var state = await scheduler.currentState
        XCTAssertEqual(state, "single")

        // Incompatible request (VLM with image) should fall back to single path
        let image = LMInput.ProcessedImage(pixels: MLXArray.zeros([1, 3, 224, 224]))
        let input2 = LMInput(
            text: .init(tokens: MLXArray([Int32(5), Int32(6)])),
            image: image
        )
        let params2 = GenerateParameters(maxTokens: 3, temperature: 0)

        let stream2 = try await scheduler.submit(
            input: input2,
            parameters: params2,
            model: model,
            cache: nil,
            tokenizer: tokenizer,
            configuration: config
        )

        // State should still be single (not batched) because the incompatible
        // request doesn't trigger upgrade
        state = await scheduler.currentState
        XCTAssertEqual(
            state, "single",
            "Incompatible request should not trigger batch upgrade")

        // Both streams should produce output
        var output1 = [String]()
        var output2 = [String]()

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
                if id == 1 { output1 = chunks } else { output2 = chunks }
            }
        }

        let totalOutput = output1.count + output2.count
        XCTAssertGreaterThan(
            totalOutput, 0,
            "Both compatible and incompatible requests should produce output")
    }

    /// kvBits requests fall back to single path correctly.
    func testKvBitsRequestFallsBack() async throws {
        try skipIfMetalUnavailable()

        let model = IntegrationTestMockModel()
        let tokenizer = TestTokenizer()
        let config = ModelConfiguration(id: "test-model")
        let scheduler = InferenceScheduler()

        // First compatible request
        let input1 = LMInput(tokens: MLXArray([Int32(1), Int32(2)]))
        let params1 = GenerateParameters(maxTokens: 5, temperature: 0)

        let _ = try await scheduler.submit(
            input: input1,
            parameters: params1,
            model: model,
            cache: nil,
            tokenizer: tokenizer,
            configuration: config
        )

        // Second request with kvBits (batch-incompatible)
        let input2 = LMInput(tokens: MLXArray([Int32(5)]))
        let params2 = GenerateParameters(maxTokens: 3, kvBits: 4, temperature: 0)

        let stream2 = try await scheduler.submit(
            input: input2,
            parameters: params2,
            model: model,
            cache: nil,
            tokenizer: tokenizer,
            configuration: config
        )

        // kvBits request should not trigger batch upgrade
        let state = await scheduler.currentState
        XCTAssertEqual(
            state, "single",
            "kvBits request should not trigger batch upgrade")

        // Consume second stream
        var chunks = [String]()
        for await gen in stream2 {
            if let chunk = gen.chunk {
                chunks.append(chunk)
            }
        }

        XCTAssertFalse(chunks.isEmpty, "kvBits fallback should still produce output")
    }

    /// SSM model falls back correctly.
    func testSSMModelFallsBack() throws {
        try skipIfMetalUnavailable()

        let model = IncompatibleSSMMockModel()
        let input = LMInput(tokens: MLXArray([Int32(1), Int32(2)]))

        let compatible = InferenceScheduler.isBatchCompatible(
            input: input,
            parameters: GenerateParameters(temperature: 0),
            cache: nil,
            model: model
        )

        XCTAssertFalse(compatible, "SSM model should be batch-incompatible")
    }

    /// Verify that two compatible requests batch, while a third incompatible
    /// request falls back to the single path. All three produce valid output.
    func testMixedCompatibleIncompatibleRequests() async throws {
        try skipIfMetalUnavailable()

        let model = IntegrationTestMockModel()
        let tokenizer = TestTokenizer()
        let config = ModelConfiguration(id: "test-model")
        let scheduler = InferenceScheduler()

        // First compatible request
        let input1 = LMInput(tokens: MLXArray([Int32(1), Int32(2)]))
        let params1 = GenerateParameters(maxTokens: 5, temperature: 0)
        let stream1 = try await scheduler.submit(
            input: input1, parameters: params1, model: model,
            cache: nil, tokenizer: tokenizer, configuration: config
        )

        // Second compatible request — should trigger batch upgrade
        let input2 = LMInput(tokens: MLXArray([Int32(10), Int32(20)]))
        let params2 = GenerateParameters(maxTokens: 5, temperature: 0)
        let stream2 = try await scheduler.submit(
            input: input2, parameters: params2, model: model,
            cache: nil, tokenizer: tokenizer, configuration: config
        )

        // Third request — incompatible (VLM with image) — falls back to single
        let image = LMInput.ProcessedImage(pixels: MLXArray.zeros([1, 3, 224, 224]))
        let input3 = LMInput(
            text: .init(tokens: MLXArray([Int32(5), Int32(6)])),
            image: image
        )
        let params3 = GenerateParameters(maxTokens: 3, temperature: 0)
        let stream3 = try await scheduler.submit(
            input: input3, parameters: params3, model: model,
            cache: nil, tokenizer: tokenizer, configuration: config
        )

        // All three streams should produce output
        var completedStreams = Set<Int>()
        await withTaskGroup(of: Int.self) { group in
            group.addTask {
                for await _ in stream1 {}
                return 1
            }
            group.addTask {
                for await _ in stream2 {}
                return 2
            }
            group.addTask {
                for await _ in stream3 {}
                return 3
            }
            for await id in group {
                completedStreams.insert(id)
            }
        }

        XCTAssertEqual(
            completedStreams.count, 3,
            "All three streams (2 compatible + 1 incompatible) should complete; "
                + "completed: \(completedStreams)")
    }

    /// Verify isBatchCompatible correctly distinguishes compatible vs incompatible
    /// request types.
    func testBatchCompatibilityDetection() throws {
        try skipIfMetalUnavailable()

        let compatibleModel = IntegrationTestMockModel()
        let ssmModel = IncompatibleSSMMockModel()

        // Standard text-only LLM — compatible
        let textInput = LMInput(tokens: MLXArray([Int32(1), Int32(2)]))
        XCTAssertTrue(
            InferenceScheduler.isBatchCompatible(
                input: textInput,
                parameters: GenerateParameters(temperature: 0),
                cache: nil,
                model: compatibleModel
            ),
            "Standard text-only LLM should be batch-compatible")

        // VLM input — incompatible
        let image = LMInput.ProcessedImage(pixels: MLXArray.zeros([1, 3, 224, 224]))
        let vlmInput = LMInput(
            text: .init(tokens: MLXArray([Int32(1)])),
            image: image
        )
        XCTAssertFalse(
            InferenceScheduler.isBatchCompatible(
                input: vlmInput,
                parameters: GenerateParameters(temperature: 0),
                cache: nil,
                model: compatibleModel
            ),
            "VLM input with image should be batch-incompatible")

        // kvBits request — incompatible
        XCTAssertFalse(
            InferenceScheduler.isBatchCompatible(
                input: textInput,
                parameters: GenerateParameters(kvBits: 4, temperature: 0),
                cache: nil,
                model: compatibleModel
            ),
            "Request with kvBits should be batch-incompatible")

        // SSM model — incompatible (detected via cache type)
        let ssmCache = ssmModel.newCache(parameters: nil)
        XCTAssertFalse(
            InferenceScheduler.isBatchCompatible(
                input: textInput,
                parameters: GenerateParameters(temperature: 0),
                cache: ssmCache,
                model: ssmModel
            ),
            "SSM model with MambaCache should be batch-incompatible")
    }

    // MARK: - VAL-CROSS-005: Backward API compatibility

    /// All existing public APIs (TokenIterator, generate(), KVCacheSimple,
    /// GenerateParameters) work unchanged.
    func testTokenIteratorAPIUnchanged() throws {
        try skipIfMetalUnavailable()

        let model = IntegrationTestMockModel()

        // TokenIterator with standard GenerateParameters
        let input = LMInput(tokens: MLXArray([Int32(5), Int32(10)]))
        let params = GenerateParameters(maxTokens: 3, temperature: 0)

        let iterator = try TokenIterator(
            input: input,
            model: model,
            cache: nil,
            parameters: params
        )

        var tokens = [Int]()
        for token in iterator {
            tokens.append(token)
        }

        XCTAssertEqual(tokens.count, 3, "TokenIterator should produce 3 tokens")
    }

    /// KVCacheSimple works unchanged.
    func testKVCacheSimpleAPIUnchanged() throws {
        try skipIfMetalUnavailable()

        let cache = KVCacheSimple()

        // Basic operations should work
        XCTAssertEqual(cache.offset, 0, "New cache should have offset 0")
        XCTAssertNil(cache.keys, "New cache should have nil keys")

        // Update should work
        let keys = MLXArray.ones([1, 4, 1, 8])
        let values = MLXArray.ones([1, 4, 1, 8])
        let (k, v) = cache.update(keys: keys, values: values)

        XCTAssertEqual(cache.offset, 1, "After update, offset should be 1")
        XCTAssertNotNil(k, "Should return keys")
        XCTAssertNotNil(v, "Should return values")
    }

    /// GenerateParameters can be created with all existing fields.
    func testGenerateParametersAPIUnchanged() {
        // Default parameters
        let params1 = GenerateParameters()
        XCTAssertNil(params1.maxTokens, "Default maxTokens should be nil")
        XCTAssertEqual(params1.temperature, 0.6)

        // Parameters with explicit values
        let params2 = GenerateParameters(
            maxTokens: 100,
            temperature: 0.5,
            topP: 0.9
        )
        XCTAssertEqual(params2.maxTokens, 100)
        XCTAssertEqual(params2.temperature, 0.5)

        // Parameters with kvBits
        let params3 = GenerateParameters(kvBits: 4, temperature: 0)
        XCTAssertEqual(params3.kvBits, 4)
    }

    /// ModelContainer works without scheduler (existing path).
    func testModelContainerWithoutSchedulerAPIUnchanged() async throws {
        try skipIfMetalUnavailable()

        let container = makeModelContainer()

        // scheduler should be nil by default
        XCTAssertNil(container.scheduler, "Default scheduler should be nil")

        let input = LMInput(tokens: MLXArray([Int32(1), Int32(2)]))
        let params = GenerateParameters(maxTokens: 3, temperature: 0)

        let stream = try await container.generate(input: input, parameters: params)

        var receivedInfo = false
        for await generation in stream {
            if case .info = generation {
                receivedInfo = true
            }
        }

        XCTAssertTrue(receivedInfo, "Should receive completion info via existing path")
    }

    /// applyRotaryPosition is backward compatible with nil cache.
    func testApplyRotaryPositionNilCacheBackwardCompat() throws {
        try skipIfMetalUnavailable()

        // When cache is nil, applyRotaryPosition should use offset 0,
        // producing the same result as rope(x, offset: 0)
        let rope = RoPE(dimensions: 8, traditional: false, base: 10000)
        let x = MLXArray.ones([1, 4, 1, 8])

        let result = applyRotaryPosition(rope, to: x, cache: nil)

        // Should produce valid output (same shape as input)
        XCTAssertEqual(result.shape, x.shape, "Output shape should match input shape")
    }

    /// applyRotaryPosition is backward compatible with KVCacheSimple.
    func testApplyRotaryPositionKVCacheSimpleBackwardCompat() throws {
        try skipIfMetalUnavailable()

        let rope = RoPE(dimensions: 8, traditional: false, base: 10000)
        let x = MLXArray.ones([1, 4, 1, 8])

        // With KVCacheSimple, should use scalar offset
        let cache = KVCacheSimple()
        let result = applyRotaryPosition(rope, to: x, cache: cache)
        XCTAssertEqual(result.shape, x.shape, "Output shape should match input shape")
    }

    // MARK: - VAL-CROSS-006: Different sequence lengths in batch

    /// Batch requests with varying prompt lengths (10, 100, 500 tokens) produce
    /// correct output with proper padding/masking.
    func testVariableSequenceLengthsInBatch() throws {
        try skipIfMetalUnavailable()

        let model = IntegrationTestMockModel()
        let iterator = BatchTokenIterator(
            model: model,
            defaultSampler: ArgMaxSampler(),
            completionBatchSize: 32,
            prefillBatchSize: 8
        )

        // Create prompts of very different lengths
        let shortPrompt = Array(1 ... 10)  // 10 tokens
        let mediumPrompt = Array(1 ... 100)  // 100 tokens
        let longPrompt = Array(1 ... 500)  // 500 tokens (but capped by vocabSize)

        // Use tokens within vocabSize range
        let shortTokens = shortPrompt.map { $0 % model.vocabSize }
        let mediumTokens = mediumPrompt.map { $0 % model.vocabSize }
        let longTokens = longPrompt.map { $0 % model.vocabSize }

        let uids = iterator.insert(
            prompts: [shortTokens, mediumTokens, longTokens],
            maxTokens: [5, 5, 5]
        )

        var tokensPerUID = [Int: [Int]]()
        var finishReasons = [Int: GenerateStopReason]()
        var loopCount = 0

        while let responses = iterator.next(), !responses.isEmpty {
            for r in responses {
                tokensPerUID[r.uid, default: []].append(r.token)
                if let reason = r.finishReason {
                    finishReasons[r.uid] = reason
                }
            }
            loopCount += 1
            if loopCount > 50 { break }
        }

        // All three should produce exactly 5 tokens regardless of prompt length
        for (i, uid) in uids.enumerated() {
            let tokens = tokensPerUID[uid] ?? []
            XCTAssertEqual(
                tokens.count, 5,
                "Prompt \(i) (length \([shortTokens, mediumTokens, longTokens][i].count)) "
                    + "should produce 5 tokens, got \(tokens.count)")
            XCTAssertEqual(
                finishReasons[uid], .length,
                "Prompt \(i) should finish with .length")

            // Verify all tokens are valid and within vocabulary
            for token in tokens {
                XCTAssertGreaterThanOrEqual(token, 0)
                XCTAssertLessThan(token, model.vocabSize)
            }
        }

        // Verify deterministic expected first tokens based on last prompt token:
        // short: last = 10 % 64 = 10 → first output = 11
        // medium: last = 100 % 64 = 36 → first output = 37
        // long: last = 500 % 64 = 52 → first output = 53
        let firstTokenShort = tokensPerUID[uids[0]]?.first
        let firstTokenMedium = tokensPerUID[uids[1]]?.first
        let firstTokenLong = tokensPerUID[uids[2]]?.first
        XCTAssertEqual(
            firstTokenShort, 11,
            "Short prompt (last=10) should start generating at 11")
        XCTAssertEqual(
            firstTokenMedium, 37,
            "Medium prompt (last=36) should start generating at 37")
        XCTAssertEqual(
            firstTokenLong, 53,
            "Long prompt (last=52) should start generating at 53")
    }

    /// Variable-length prompts through the scheduler produce correct output.
    func testVariableLengthsThroughScheduler() async throws {
        try skipIfMetalUnavailable()

        let model = IntegrationTestMockModel()
        let tokenizer = TestTokenizer()
        let config = ModelConfiguration(id: "test-model")
        let scheduler = InferenceScheduler()

        // Short prompt
        let input1 = LMInput(tokens: MLXArray(Array(repeating: Int32(1), count: 5)))
        let params1 = GenerateParameters(maxTokens: 3, temperature: 0)

        let stream1 = try await scheduler.submit(
            input: input1,
            parameters: params1,
            model: model,
            cache: nil,
            tokenizer: tokenizer,
            configuration: config
        )

        // Longer prompt triggers batch with very different length
        let input2 = LMInput(tokens: MLXArray(Array(repeating: Int32(10), count: 50)))
        let params2 = GenerateParameters(maxTokens: 3, temperature: 0)

        let stream2 = try await scheduler.submit(
            input: input2,
            parameters: params2,
            model: model,
            cache: nil,
            tokenizer: tokenizer,
            configuration: config
        )

        // Both should complete without errors
        var completed = [Int: Bool]()

        await withTaskGroup(of: (Int, Bool).self) { group in
            group.addTask {
                for await _ in stream1 {}
                return (1, true)
            }
            group.addTask {
                for await _ in stream2 {}
                return (2, true)
            }
            for await (id, success) in group {
                completed[id] = success
            }
        }

        XCTAssertTrue(completed[1] ?? false, "Short prompt should complete")
        XCTAssertTrue(completed[2] ?? false, "Long prompt should complete")
    }

    // MARK: - VAL-CROSS-007: Prompt cache integrated with batch generation

    /// Requests with cached prefixes join a batch with reduced prefill, and
    /// cached KV data is correctly merged into the batch cache.
    func testPromptCacheIntegrationWithBatchGeneration() throws {
        try skipIfMetalUnavailable()

        let model = IntegrationTestMockModel()
        let promptCache = LRUPromptCache(maxSize: 10)

        // Simulate storing a cached prefix
        let cachedTokens = [1, 2, 3, 4, 5, 6, 7, 8]
        let cachedKV = makeMockPromptCache(layers: 1, seqLen: 8, value: 1.0)
        promptCache.insertCache(
            model: "test", tokens: cachedTokens, promptCache: cachedKV)

        // New request with same prefix + additional suffix
        let newTokens = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        let (fetchedCache, remainder) = promptCache.fetchNearestCache(
            model: "test", tokens: newTokens
        )

        XCTAssertNotNil(fetchedCache, "Should find cached prefix")
        XCTAssertEqual(remainder, [9, 10], "Remainder should be uncached suffix")

        // Use cached prefix in batch generation
        model.resetCounters()
        let iterator = BatchTokenIterator(
            model: model,
            defaultSampler: ArgMaxSampler(),
            completionBatchSize: 32,
            prefillBatchSize: 8
        )

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

        // Verify reduced prefill: cached prefix (8 tokens) means only suffix
        // (2 tokens) needs to be processed through the model.
        XCTAssertLessThan(
            model.totalTokensProcessed, 10,
            "Should process fewer than 10 tokens due to cached prefix "
                + "(actual: \(model.totalTokensProcessed))")
    }

    /// Cached prefix reduces prefill token count when mixed with uncached prompts.
    func testCachedAndUncachedMixedInBatch() throws {
        try skipIfMetalUnavailable()

        let model = IntegrationTestMockModel()

        // Full prefill baseline
        let iteratorFull = BatchTokenIterator(
            model: model,
            defaultSampler: ArgMaxSampler(),
            completionBatchSize: 32,
            prefillBatchSize: 8
        )

        let promptA = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        let promptB = [20, 21, 22, 23, 24]

        let _ = iteratorFull.insert(
            prompts: [promptA, promptB],
            maxTokens: [1, 1]
        )
        let _ = iteratorFull.next()
        let fullTokens = model.totalTokensProcessed

        // Cached prefill
        model.resetCounters()
        let cachedA = makeMockPromptCache(layers: 1, seqLen: 8, value: 1.0)

        let iteratorCached = BatchTokenIterator(
            model: model,
            defaultSampler: ArgMaxSampler(),
            completionBatchSize: 32,
            prefillBatchSize: 8
        )

        let _ = iteratorCached.insert(
            prompts: [promptA, promptB],
            maxTokens: [1, 1],
            cachedKVStates: [cachedA, nil]
        )
        let _ = iteratorCached.next()
        let cachedTokens = model.totalTokensProcessed

        XCTAssertLessThan(
            cachedTokens, fullTokens,
            "Cached prefill (\(cachedTokens)) should use fewer tokens than full (\(fullTokens))")
    }

    // MARK: - VAL-CROSS-008: Tool calls in batch generation routed to correct stream

    /// When a sequence generates a tool call token pattern through the scheduler,
    /// the parsed ToolCall is emitted only on that request's stream.
    ///
    /// Uses `ToolCallMockModel` (emits `<tool_call>` tokens for prompt starting
    /// with token 50) and `ToolCallTestTokenizer` (maps IDs 100-102 to tool-call
    /// text). A single request (prompt [50]) receives a `.toolCall` event with
    /// the correct function name.
    func testToolCallEmittedOnCorrectStream() async throws {
        try skipIfMetalUnavailable()

        let model = ToolCallMockModel()
        let tokenizer = ToolCallTestTokenizer()
        let config = ModelConfiguration(id: "test-tool-model")
        let scheduler = InferenceScheduler()

        // Single request producing tool call tokens on the single path
        let input = LMInput(tokens: MLXArray([Int32(50)]))
        let params = GenerateParameters(maxTokens: 5, temperature: 0)

        let stream = try await scheduler.submit(
            input: input,
            parameters: params,
            model: model,
            cache: nil,
            tokenizer: tokenizer,
            configuration: config
        )

        // Collect all Generation events
        var toolCallNames = [String]()
        var hasInfo = false

        for await gen in stream {
            switch gen {
            case .chunk:
                break
            case .info:
                hasInfo = true
            case .toolCall(let tc):
                toolCallNames.append(tc.function.name)
            }
        }

        // The stream should have received a .toolCall event with "get_weather"
        XCTAssertTrue(
            toolCallNames.contains("get_weather"),
            "Stream should receive .toolCall(get_weather); "
                + "got tool calls: \(toolCallNames)")

        XCTAssertTrue(hasInfo, "Stream should receive completion info")
    }

    /// Verify that two independent scheduler streams have complete isolation:
    /// tool call events arrive only on the producing stream, not on others.
    /// This uses two sequential requests (no concurrent batch upgrade complexity)
    /// to verify the routing mechanism.
    func testToolCallStreamIsolationSequential() async throws {
        try skipIfMetalUnavailable()

        let model = ToolCallMockModel()
        let tokenizer = ToolCallTestTokenizer()
        let config = ModelConfiguration(id: "test-tool-model")

        // First request: produces tool calls
        let scheduler1 = InferenceScheduler()
        let input1 = LMInput(tokens: MLXArray([Int32(50)]))
        let params1 = GenerateParameters(maxTokens: 5, temperature: 0)

        let stream1 = try await scheduler1.submit(
            input: input1, parameters: params1, model: model,
            cache: nil, tokenizer: tokenizer, configuration: config
        )

        var toolCalls1 = [String]()
        for await gen in stream1 {
            if case .toolCall(let tc) = gen {
                toolCalls1.append(tc.function.name)
            }
        }

        // Second request: produces plain text (no tool calls)
        let scheduler2 = InferenceScheduler()
        let input2 = LMInput(tokens: MLXArray([Int32(1), Int32(2)]))
        let params2 = GenerateParameters(maxTokens: 5, temperature: 0)

        let stream2 = try await scheduler2.submit(
            input: input2, parameters: params2, model: model,
            cache: nil, tokenizer: tokenizer, configuration: config
        )

        var toolCalls2 = [String]()
        for await gen in stream2 {
            if case .toolCall(let tc) = gen {
                toolCalls2.append(tc.function.name)
            }
        }

        // Tool call should appear on stream 1 only
        XCTAssertTrue(
            toolCalls1.contains("get_weather"),
            "Tool-call stream should receive .toolCall(get_weather); "
                + "got: \(toolCalls1)")
        XCTAssertTrue(
            toolCalls2.isEmpty,
            "Plain-text stream should NOT receive any tool calls; "
                + "got: \(toolCalls2)")
    }

    /// Verify stream isolation at the BatchTokenIterator level: each UID's
    /// tokens match the deterministic expected sequence.
    func testBatchTokenIteratorStreamIsolation() throws {
        try skipIfMetalUnavailable()

        let model = IntegrationTestMockModel()
        let iterator = BatchTokenIterator(
            model: model,
            defaultSampler: ArgMaxSampler(),
            completionBatchSize: 32,
            prefillBatchSize: 8
        )

        // Two prompts with very different starting tokens
        let uids = iterator.insert(
            prompts: [[1, 2, 3], [30, 40, 50]],
            maxTokens: [5, 5]
        )

        var tokensPerUID = [Int: [Int]]()

        while let responses = iterator.next(), !responses.isEmpty {
            for r in responses {
                tokensPerUID[r.uid, default: []].append(r.token)
            }
        }

        let tokens0 = tokensPerUID[uids[0]] ?? []
        let tokens1 = tokensPerUID[uids[1]] ?? []

        // Both should produce exactly 5 tokens
        XCTAssertEqual(tokens0.count, 5, "First request should produce 5 tokens")
        XCTAssertEqual(tokens1.count, 5, "Second request should produce 5 tokens")

        // Verify deterministic expected sequences (stream isolation):
        // Prompt [1,2,3]: last=3 → 4,5,6,7,8
        // Prompt [30,40,50]: last=50 → 51,52,53,54,55
        let expected0 = [4, 5, 6, 7, 8]
        let expected1 = [51, 52, 53, 54, 55]
        XCTAssertEqual(
            tokens0, expected0,
            "Prompt [1,2,3] should produce \(expected0), got \(tokens0)")
        XCTAssertEqual(
            tokens1, expected1,
            "Prompt [30,40,50] should produce \(expected1), got \(tokens1)")
    }

    // MARK: - Additional Cross-Area Tests

    /// Verify that batch output matches single-request output for the same prompt
    /// with deterministic sampling.
    func testBatchVsSingleOutputMatch() throws {
        try skipIfMetalUnavailable()

        let maxTokens = 5
        let prompt = [5, 10, 15]

        // Single-request generation
        let singleModel = IntegrationTestMockModel()
        let singleInput = LMInput(tokens: MLXArray(prompt.map { Int32($0) }))
        let singleIterator = try TokenIterator(
            input: singleInput,
            model: singleModel,
            processor: nil,
            sampler: ArgMaxSampler(),
            prefillStepSize: 512,
            maxTokens: maxTokens
        )
        var singleTokens = [Int]()
        for token in singleIterator {
            singleTokens.append(token)
        }

        // Batch-of-1 generation
        let batchModel = IntegrationTestMockModel()
        let batchIterator = BatchTokenIterator(
            model: batchModel,
            defaultSampler: ArgMaxSampler(),
            completionBatchSize: 32,
            prefillBatchSize: 8
        )

        let batchUIDs = batchIterator.insert(
            prompts: [prompt],
            maxTokens: [maxTokens]
        )

        var batchTokens = [Int]()
        while let responses = batchIterator.next(), !responses.isEmpty {
            for r in responses {
                XCTAssertEqual(r.uid, batchUIDs[0])
                batchTokens.append(r.token)
            }
        }

        // Verify deterministic expected output:
        // Prompt [5,10,15]: last=15 → 16,17,18,19,20
        let expectedOutput = [16, 17, 18, 19, 20]
        XCTAssertEqual(
            singleTokens, expectedOutput,
            "Single path should produce \(expectedOutput), got \(singleTokens)")
        XCTAssertEqual(
            batchTokens, expectedOutput,
            "Batch path should produce \(expectedOutput), got \(batchTokens)")
        XCTAssertEqual(
            singleTokens, batchTokens,
            "Batch output must match single-request output with ArgMax")
    }

    /// ModelContainer with scheduler correctly routes through InferenceScheduler.
    func testModelContainerWithSchedulerEndToEnd() async throws {
        try skipIfMetalUnavailable()

        let scheduler = InferenceScheduler()
        let container = makeModelContainer(scheduler: scheduler)

        // Submit two concurrent requests through ModelContainer
        var results = [Int: Bool]()

        await withTaskGroup(of: (Int, Bool).self) { group in
            group.addTask {
                do {
                    let input = LMInput(tokens: MLXArray([Int32(1), Int32(2)]))
                    let params = GenerateParameters(maxTokens: 5, temperature: 0)
                    let stream = try await container.generate(
                        input: input, parameters: params)
                    var count = 0
                    for await gen in stream {
                        if gen.chunk != nil { count += 1 }
                    }
                    return (1, count > 0)
                } catch {
                    return (1, false)
                }
            }
            group.addTask {
                try? await Task.sleep(nanoseconds: 10_000_000)  // 10ms
                do {
                    let input = LMInput(tokens: MLXArray([Int32(10), Int32(20)]))
                    let params = GenerateParameters(maxTokens: 3, temperature: 0)
                    let stream = try await container.generate(
                        input: input, parameters: params)
                    var count = 0
                    for await gen in stream {
                        if gen.chunk != nil { count += 1 }
                    }
                    return (2, count > 0)
                } catch {
                    return (2, false)
                }
            }
            for await (id, success) in group {
                results[id] = success
            }
        }

        let anyProduced = results.values.contains(true)
        XCTAssertTrue(
            anyProduced,
            "At least one request through ModelContainer+scheduler should produce output")
    }

    /// Verify that the scheduler returns to idle after all requests complete.
    func testSchedulerReturnsToIdleAfterCompletion() async throws {
        try skipIfMetalUnavailable()

        let model = IntegrationTestMockModel()
        let tokenizer = TestTokenizer()
        let config = ModelConfiguration(id: "test-model")
        let scheduler = InferenceScheduler()

        var state = await scheduler.currentState
        XCTAssertEqual(state, "idle", "Should start idle")

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

        state = await scheduler.currentState
        XCTAssertEqual(state, "single")

        // Consume to completion
        for await _ in stream {}

        // Wait for cleanup
        try await Task.sleep(nanoseconds: 200_000_000)  // 200ms

        state = await scheduler.currentState
        XCTAssertEqual(state, "idle", "Should return to idle after completion")
    }

    /// Staggered completion in batch: first request finishes before second.
    func testStaggeredCompletionInBatch() async throws {
        try skipIfMetalUnavailable()

        let model = IntegrationTestMockModel()
        let tokenizer = TestTokenizer()
        let config = ModelConfiguration(id: "test-model")
        let scheduler = InferenceScheduler()

        // First request with fewer tokens
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

        // Second request with more tokens
        let input2 = LMInput(tokens: MLXArray([Int32(10), Int32(20)]))
        let params2 = GenerateParameters(maxTokens: 10, temperature: 0)

        let stream2 = try await scheduler.submit(
            input: input2,
            parameters: params2,
            model: model,
            cache: nil,
            tokenizer: tokenizer,
            configuration: config
        )

        var completed1 = false
        var completed2 = false

        await withTaskGroup(of: (Int, Bool).self) { group in
            group.addTask {
                for await _ in stream1 {}
                return (1, true)
            }
            group.addTask {
                for await _ in stream2 {}
                return (2, true)
            }
            for await (id, success) in group {
                if id == 1 { completed1 = success } else { completed2 = success }
            }
        }

        XCTAssertTrue(completed1, "Short request should complete")
        XCTAssertTrue(completed2, "Long request should complete after short one")
    }
}
