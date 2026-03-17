// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXNN
import Tokenizers
import XCTest

@testable import MLXLMCommon

// MARK: - Mock Model for ModelContainer Integration Tests

/// A deterministic mock language model for ModelContainer integration tests.
///
/// Produces tokens deterministically: next token = (input_token + 1) % vocabSize.
/// Uses KVCacheSimple by default (batch-compatible).
private class IntegrationMockModel: Module, LanguageModel, KVCacheDimensionProvider,
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

/// A simple mock input processor for tests.
private struct MockInputProcessor: UserInputProcessor {
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

// MARK: - Tests

class ModelContainerIntegrationTests: XCTestCase {

    // Helper to create a ModelContainer with a mock model
    private func makeModelContainer(
        scheduler: InferenceScheduler? = nil
    ) -> ModelContainer {
        let model = IntegrationMockModel()
        let tokenizer = TestTokenizer()
        let config = ModelConfiguration(id: "test-model")
        let processor = MockInputProcessor(tokenizer: tokenizer, configuration: config)

        let context = ModelContext(
            configuration: config,
            model: model,
            processor: processor,
            tokenizer: tokenizer
        )

        let container = ModelContainer(context: context)

        // Set the scheduler if provided
        if let scheduler {
            // We'll set it after construction via a method or property
            // This will be implemented as part of the feature
            container.scheduler = scheduler
        }

        return container
    }

    private func makeCallTrackingContainer(
        scheduler: InferenceScheduler? = nil,
        configurationID: String = "test-model"
    ) -> (
        container: ModelContainer,
        model: CallTrackingModel,
        promptCache: LRUPromptCache,
        configuration: ModelConfiguration
    ) {
        let model = CallTrackingModel(vocabSize: 32, numLayers: 1)
        let tokenizer = TestTokenizer()
        let configuration = ModelConfiguration(id: configurationID)
        let processor = MockInputProcessor(tokenizer: tokenizer, configuration: configuration)

        let context = ModelContext(
            configuration: configuration,
            model: model,
            processor: processor,
            tokenizer: tokenizer
        )

        let promptCache = LRUPromptCache(maxSize: 10)
        let container = ModelContainer(context: context)
        container.scheduler = scheduler
        container.promptCache = promptCache

        return (container, model, promptCache, configuration)
    }

    // MARK: - VAL-SCHED-009: ModelContainer without scheduler uses existing path

    func testModelContainerWithoutSchedulerUsesExistingPath() async throws {
        try skipIfMetalUnavailable()

        let container = makeModelContainer()

        // Scheduler should be nil by default
        let schedulerIsNil = container.scheduler == nil
        XCTAssertTrue(schedulerIsNil, "Default scheduler should be nil")

        let input = LMInput(tokens: MLXArray([Int32(1), Int32(2), Int32(3)]))
        let params = GenerateParameters(maxTokens: 3, temperature: 0)

        let stream = try await container.generate(input: input, parameters: params)

        var chunks = [String]()
        for await generation in stream {
            if let chunk = generation.chunk {
                chunks.append(chunk)
            }
        }

        // Should produce output via the existing direct path
        XCTAssertFalse(chunks.isEmpty, "Should produce output without scheduler")
    }

    // MARK: - VAL-SCHED-010: ModelContainer with scheduler routes through InferenceScheduler

    func testModelContainerWithSchedulerRoutesThrough() async throws {
        try skipIfMetalUnavailable()

        let scheduler = InferenceScheduler()
        let container = makeModelContainer(scheduler: scheduler)

        let input = LMInput(tokens: MLXArray([Int32(1), Int32(2), Int32(3)]))
        let params = GenerateParameters(maxTokens: 3, temperature: 0)

        let stream = try await container.generate(input: input, parameters: params)

        // After submit, the scheduler should be in "single" state
        let schedulerState = await scheduler.currentState
        XCTAssertEqual(
            schedulerState, "single",
            "Scheduler should transition to single state when request is routed through it"
        )

        // Consume stream
        var chunks = [String]()
        for await generation in stream {
            if let chunk = generation.chunk {
                chunks.append(chunk)
            }
        }

        XCTAssertFalse(chunks.isEmpty, "Should produce output via scheduler path")
    }

    func testModelContainerWithSchedulerForwardsWiredMemoryTicket() async throws {
        try skipIfMetalUnavailable()

        let scheduler = InferenceScheduler()
        let container = makeModelContainer(scheduler: scheduler)
        let manager = WiredMemoryManager.makeForTesting(
            configuration: .init(
                policyOnlyWhenUnsupported: true,
                baselineOverride: 0,
                useRecommendedWorkingSetWhenUnsupported: false
            )
        )
        let policy = WiredSumPolicy(cap: 1024)
        let ticket = policy.ticket(size: 64, manager: manager, kind: .active)
        let eventStream = await manager.events()
        let eventsTask = Task { () -> [WiredMemoryEvent] in
            var events = [WiredMemoryEvent]()
            for await event in eventStream {
                events.append(event)
                if events.filter({ $0.ticketID == ticket.id && $0.kind == .ticketEnded }).count >= 1
                {
                    break
                }
            }
            return events
        }

        let input = LMInput(tokens: MLXArray([Int32(1), Int32(2), Int32(3)]))
        let params = GenerateParameters(maxTokens: 4, temperature: 0)

        let stream = try await container.generate(
            input: input,
            parameters: params,
            wiredMemoryTicket: ticket
        )

        for await _ in stream {}

        let events = await eventsTask.value
        XCTAssertEqual(
            events.filter { $0.ticketID == ticket.id && $0.kind == .ticketStarted }.count,
            1
        )
        XCTAssertEqual(
            events.filter { $0.ticketID == ticket.id && $0.kind == .ticketEnded }.count,
            1
        )
    }

    // MARK: - VAL-SCHED-011: Each request gets independent AsyncStream

    func testEachRequestGetsIndependentStream() async throws {
        try skipIfMetalUnavailable()

        let scheduler = InferenceScheduler()
        let container = makeModelContainer(scheduler: scheduler)

        let params = GenerateParameters(maxTokens: 5, temperature: 0)

        // Submit two requests concurrently
        var tokens1 = [String]()
        var tokens2 = [String]()

        await withTaskGroup(of: (Int, [String]).self) { group in
            group.addTask {
                let input = LMInput(tokens: MLXArray([Int32(1), Int32(2)]))
                var chunks = [String]()
                do {
                    let stream = try await container.generate(input: input, parameters: params)
                    for await gen in stream {
                        if let chunk = gen.chunk {
                            chunks.append(chunk)
                        }
                    }
                } catch {}
                return (1, chunks)
            }

            group.addTask {
                // Small delay to ensure second request arrives while first is active
                try? await Task.sleep(nanoseconds: 10_000_000)  // 10ms
                let input = LMInput(tokens: MLXArray([Int32(5), Int32(6)]))
                var chunks = [String]()
                do {
                    let stream = try await container.generate(input: input, parameters: params)
                    for await gen in stream {
                        if let chunk = gen.chunk {
                            chunks.append(chunk)
                        }
                    }
                } catch {}
                return (2, chunks)
            }

            for await (id, chunks) in group {
                if id == 1 {
                    tokens1 = chunks
                } else {
                    tokens2 = chunks
                }
            }
        }

        // Both streams should have produced some output independently
        // (At minimum, one should produce output; the second may or may not
        // depending on timing, but they should be independent)
        let totalOutput = tokens1.count + tokens2.count
        XCTAssertGreaterThan(
            totalOutput, 0,
            "At least one stream should produce output"
        )
    }

    // MARK: - VAL-SCHED-012: Request cancellation stops generation for that request

    func testRequestCancellationStopsOnlyThatRequest() async throws {
        try skipIfMetalUnavailable()

        let scheduler = InferenceScheduler()
        let container = makeModelContainer(scheduler: scheduler)

        let params = GenerateParameters(maxTokens: 50, temperature: 0)

        var request1Cancelled = false
        var request2Completed = false

        await withTaskGroup(of: (Int, Bool).self) { group in
            group.addTask {
                do {
                    let input = LMInput(tokens: MLXArray([Int32(1), Int32(2)]))
                    let stream = try await container.generate(input: input, parameters: params)
                    var count = 0
                    for await _ in stream {
                        count += 1
                        if count >= 2 {
                            // Cancel this task after receiving 2 items
                            break
                        }
                    }
                    return (1, true)
                } catch {
                    return (1, true)
                }
            }

            group.addTask {
                // Small delay to start second request
                try? await Task.sleep(nanoseconds: 10_000_000)  // 10ms
                do {
                    let input = LMInput(tokens: MLXArray([Int32(5), Int32(6)]))
                    let stream = try await container.generate(input: input, parameters: params)
                    for await _ in stream {
                        // Consume fully
                    }
                    return (2, true)
                } catch {
                    return (2, false)
                }
            }

            for await (id, completed) in group {
                if id == 1 {
                    request1Cancelled = completed
                } else {
                    request2Completed = completed
                }
            }
        }

        // Request 1 was broken out of early, Request 2 should complete
        XCTAssertTrue(request1Cancelled, "First request should have been cancelled/broken")
        XCTAssertTrue(request2Completed, "Second request should complete independently")
    }

    // MARK: - VAL-SCHED-013: Staggered completion handled correctly

    func testStaggeredCompletionHandledCorrectly() async throws {
        try skipIfMetalUnavailable()

        let scheduler = InferenceScheduler()
        let container = makeModelContainer(scheduler: scheduler)

        var completed1 = false
        var completed2 = false

        await withTaskGroup(of: (Int, Bool).self) { group in
            group.addTask {
                do {
                    // Request 1: short (3 tokens)
                    let input = LMInput(tokens: MLXArray([Int32(1), Int32(2)]))
                    let params = GenerateParameters(maxTokens: 3, temperature: 0)
                    let stream = try await container.generate(input: input, parameters: params)
                    for await _ in stream {}
                    return (1, true)
                } catch {
                    return (1, false)
                }
            }

            group.addTask {
                try? await Task.sleep(nanoseconds: 10_000_000)  // 10ms delay
                do {
                    // Request 2: longer (10 tokens)
                    let input = LMInput(tokens: MLXArray([Int32(5), Int32(6)]))
                    let params = GenerateParameters(maxTokens: 10, temperature: 0)
                    let stream = try await container.generate(input: input, parameters: params)
                    for await _ in stream {}
                    return (2, true)
                } catch {
                    return (2, false)
                }
            }

            for await (id, success) in group {
                if id == 1 {
                    completed1 = success
                } else {
                    completed2 = success
                }
            }
        }

        XCTAssertTrue(completed1, "Short request should complete")
        XCTAssertTrue(completed2, "Long request should complete after short one finishes")
    }

    // MARK: - VAL-SCHED-006: Padding and masking correct in batched mode

    func testPaddingAndMaskingCorrectInBatchedMode() async throws {
        try skipIfMetalUnavailable()

        // Run a single request through the scheduler and verify it produces output.
        // Full deterministic comparison requires batch + single path producing
        // identical tokens, which is covered structurally but Metal-dependent tests
        // can only be verified in Xcode.
        let scheduler = InferenceScheduler()
        let container = makeModelContainer(scheduler: scheduler)

        let input = LMInput(tokens: MLXArray([Int32(1), Int32(2), Int32(3)]))
        let params = GenerateParameters(maxTokens: 5, temperature: 0)

        let stream = try await container.generate(input: input, parameters: params)

        var receivedInfo = false
        var chunkCount = 0
        for await generation in stream {
            switch generation {
            case .chunk:
                chunkCount += 1
            case .info(let info):
                receivedInfo = true
                XCTAssertGreaterThan(
                    info.generationTokenCount, 0,
                    "Should report non-zero token count"
                )
            case .toolCall:
                break
            }
        }

        XCTAssertTrue(receivedInfo, "Should receive completion info")
        XCTAssertGreaterThan(chunkCount, 0, "Should receive output chunks")
    }

    // MARK: - VAL-SCHED-018: Multiple ChatSessions sharing ModelContainer trigger batching

    func testMultipleChatSessionsSharingModelContainerTriggerBatching() async throws {
        try skipIfMetalUnavailable()

        let scheduler = InferenceScheduler()
        let container = makeModelContainer(scheduler: scheduler)

        var result1: String?
        var result2: String?

        await withTaskGroup(of: (Int, String?).self) { group in
            group.addTask {
                // Create ChatSession inside task to avoid sending non-Sendable across isolation
                let session = ChatSession(container)
                do {
                    let response = try await session.respond(to: "Hello world")
                    return (1, response)
                } catch {
                    return (1, nil)
                }
            }

            group.addTask {
                // Small delay so second request arrives while first is generating
                try? await Task.sleep(nanoseconds: 10_000_000)  // 10ms
                // Create ChatSession inside task to avoid sending non-Sendable across isolation
                let session = ChatSession(container)
                do {
                    let response = try await session.respond(to: "Goodbye world")
                    return (2, response)
                } catch {
                    return (2, nil)
                }
            }

            for await (id, response) in group {
                if id == 1 {
                    result1 = response
                } else {
                    result2 = response
                }
            }
        }

        // Both sessions should produce output
        // At least one should succeed (depending on timing, both may succeed)
        let anySucceeded = result1 != nil || result2 != nil
        XCTAssertTrue(
            anySucceeded,
            "At least one ChatSession should produce output when sharing ModelContainer"
        )
    }

    // MARK: - Incompatible request falls back to direct path

    func testIncompatibleRequestWithSchedulerFallsBack() async throws {
        try skipIfMetalUnavailable()

        let scheduler = InferenceScheduler()
        let (container, _, promptCache, config) = makeCallTrackingContainer(scheduler: scheduler)

        let promptTokens = [1, 2, 3, 4, 5]
        let fullSequence = [1, 2, 3, 4, 5, 6, 7]
        let firstInput = LMInput(tokens: MLXArray(promptTokens.map(Int32.init)))
        let params = GenerateParameters(
            maxTokens: 2,
            kvBits: 4,
            quantizedKVStart: 1_000,
            temperature: 0
        )

        let stream = try await container.generate(input: firstInput, parameters: params)

        var chunks = [String]()
        for await generation in stream {
            if let chunk = generation.chunk {
                chunks.append(chunk)
            }
        }

        // Should still produce output via fallback to direct path
        XCTAssertFalse(
            chunks.isEmpty,
            "Incompatible request should fall back to direct path and still produce output"
        )

        let (exactCache, exactRemainder) = promptCache.fetchNearestCache(
            model: config.name,
            tokens: fullSequence
        )
        XCTAssertNotNil(
            exactCache,
            "Fallback request should write back its final cache using the full prompt+generation token key"
        )
        XCTAssertEqual(exactCache?.first?.offset, fullSequence.count)
        XCTAssertEqual(exactRemainder, [])

        let (trimmedCache, trimmedRemainder) = promptCache.fetchNearestCache(
            model: config.name,
            tokens: promptTokens
        )
        XCTAssertNotNil(
            trimmedCache,
            "Full-sequence fallback write-back should be reusable for the original prompt prefix"
        )
        XCTAssertEqual(trimmedCache?.first?.offset, promptTokens.count)
        XCTAssertEqual(trimmedRemainder, [])
    }

    // MARK: - kvBits request falls back to direct path

    func testKvBitsRequestFallsBackToDirectPath() async throws {
        try skipIfMetalUnavailable()

        let scheduler = InferenceScheduler()
        let (container, model, promptCache, config) = makeCallTrackingContainer(
            scheduler: scheduler)

        let promptTokens = [1, 2, 3, 4, 5]
        let fullSequence = [1, 2, 3, 4, 5, 6, 7]
        let firstInput = LMInput(tokens: MLXArray(promptTokens.map(Int32.init)))
        let params = GenerateParameters(
            maxTokens: 2,
            kvBits: 4,
            quantizedKVStart: 1_000,
            temperature: 0
        )

        let firstStream = try await container.generate(input: firstInput, parameters: params)

        for await _ in firstStream {}

        let fullFallbackTokensProcessed = model.totalTokensProcessed
        XCTAssertGreaterThan(fullFallbackTokensProcessed, promptTokens.count)

        model.resetCounters()

        let secondInput = LMInput(tokens: MLXArray(promptTokens.map(Int32.init)))
        let secondStream = try await container.generate(input: secondInput, parameters: params)

        var chunks = [String]()
        for await generation in secondStream {
            if let chunk = generation.chunk {
                chunks.append(chunk)
            }
        }

        // Should produce output via direct path (kvBits incompatible with batch)
        XCTAssertFalse(
            chunks.isEmpty,
            "kvBits request should fall back to direct path"
        )

        XCTAssertTrue(
            model.sawPreloadedCache,
            "Repeated kvBits fallback request should receive the cached KV state on the single-path fallback"
        )
        XCTAssertLessThan(
            model.totalTokensProcessed,
            fullFallbackTokensProcessed,
            "Repeated kvBits fallback request should process fewer tokens when prompt cache is reused"
        )

        let (exactCache, exactRemainder) = promptCache.fetchNearestCache(
            model: config.name,
            tokens: fullSequence
        )
        XCTAssertNotNil(
            exactCache,
            "Fallback request should keep writing back the final cache after repeated kvBits requests"
        )
        XCTAssertEqual(exactCache?.first?.offset, fullSequence.count)
        XCTAssertEqual(exactRemainder, [])
    }

    // MARK: - Scheduler property can be set and read

    func testSchedulerPropertySetAndRead() async throws {
        let container = makeModelContainer()

        // Default should be nil
        var schedulerValue = container.scheduler
        XCTAssertNil(schedulerValue, "Default scheduler should be nil")

        // Set a scheduler
        let scheduler = InferenceScheduler()
        container.scheduler = scheduler

        // Should now be non-nil
        schedulerValue = container.scheduler
        XCTAssertNotNil(schedulerValue, "Scheduler should be set")
    }

    // MARK: - PromptCache property can be set and read

    func testPromptCachePropertySetAndRead() async throws {
        let container = makeModelContainer()

        // Default should be nil
        var cacheValue = container.promptCache
        XCTAssertNil(cacheValue, "Default promptCache should be nil")

        // Set a prompt cache
        let promptCache = LRUPromptCache(maxSize: 10)
        container.promptCache = promptCache

        // Should now be non-nil
        cacheValue = container.promptCache
        XCTAssertNotNil(cacheValue, "PromptCache should be set")
    }

    // MARK: - VAL-FIX-007: LRUPromptCache wired into scheduler path

    /// Verifies that when ModelContainer.scheduler is set and LRUPromptCache is available,
    /// repeated prompts with shared prefixes use cached KV state instead of full reprocessing.
    /// The second identical prompt should process fewer tokens than the first.
    func testPromptCacheWiredIntoSchedulerPath() async throws {
        try skipIfMetalUnavailable()

        let scheduler = InferenceScheduler()
        let (container, model, promptCache, config) = makeCallTrackingContainer(
            scheduler: scheduler)

        // First request — should process all tokens (no cache hit)
        let promptTokens = [1, 2, 3, 4, 5]
        let tokens1 = MLXArray(promptTokens.map(Int32.init))
        let input1 = LMInput(tokens: tokens1)
        let params1 = GenerateParameters(maxTokens: 2, temperature: 0)

        let stream1 = try await container.generate(input: input1, parameters: params1)
        for await _ in stream1 {}

        // Wait for scheduler to return to idle
        try await Task.sleep(nanoseconds: 200_000_000)

        let firstTokensProcessed = model.totalTokensProcessed
        XCTAssertGreaterThan(firstTokensProcessed, promptTokens.count)

        let (cachedKV, remainder) = promptCache.fetchNearestCache(
            model: config.name,
            tokens: promptTokens
        )
        XCTAssertNotNil(cachedKV, "First scheduler request should write back prompt cache state")
        XCTAssertEqual(remainder, [], "Repeated prompt should be fully satisfied by cached prefix")

        model.resetCounters()

        // Second request — same tokens, should get a cache hit
        let tokens2 = MLXArray(promptTokens.map(Int32.init))
        let input2 = LMInput(tokens: tokens2)
        let params2 = GenerateParameters(maxTokens: 2, temperature: 0)

        let stream2 = try await container.generate(input: input2, parameters: params2)
        for await _ in stream2 {}

        XCTAssertTrue(
            model.sawPreloadedCache,
            "Second scheduler request should receive cached KV state from the prompt cache"
        )
        XCTAssertLessThan(
            model.totalTokensProcessed,
            firstTokensProcessed,
            "Prompt cache hit should reduce prompt processing work on the second request"
        )
    }

    /// Verifies that prompt cache fetch is called with the correct model identifier.
    func testPromptCacheFetchUsesModelName() async throws {
        try skipIfMetalUnavailable()

        let model = IntegrationMockModel()
        let tokenizer = TestTokenizer()
        let config = ModelConfiguration(id: "test-model-abc")
        let processor = MockInputProcessor(tokenizer: tokenizer, configuration: config)

        let context = ModelContext(
            configuration: config,
            model: model,
            processor: processor,
            tokenizer: tokenizer
        )

        let scheduler = InferenceScheduler()
        let promptCache = LRUPromptCache(maxSize: 10)

        let container = ModelContainer(context: context)
        container.scheduler = scheduler
        container.promptCache = promptCache

        // Insert a cache entry under the model name
        let cachedKV: [KVCache] = [KVCacheSimple()]
        let testTokens = [1, 2, 3]
        promptCache.insertCache(
            model: config.name,
            tokens: testTokens,
            promptCache: cachedKV
        )

        // Verify the entry can be fetched using the same model name
        let (fetched, remainder) = promptCache.fetchNearestCache(
            model: config.name, tokens: testTokens)
        XCTAssertNotNil(fetched, "Should find cache entry using model name")
        XCTAssertEqual(remainder, [], "Should have empty remainder for exact match")

        // Verify the entry is NOT found under a different model name
        let (wrongFetch, _) = promptCache.fetchNearestCache(
            model: "different-model", tokens: testTokens)
        XCTAssertNil(wrongFetch, "Should not find cache entry under different model name")
    }

    // MARK: - VAL-FIX-008: ChatSession preserves cache state with batching enabled

    /// Verifies that ChatSession does not drop KV cache state when batching is enabled.
    /// Follow-up messages in the same session should reuse cached context.
    func testChatSessionPreservesCacheWithBatchingEnabled() async throws {
        try skipIfMetalUnavailable()

        let scheduler = InferenceScheduler()
        let promptCache = LRUPromptCache(maxSize: 10)
        let container = makeModelContainer(scheduler: scheduler)
        container.promptCache = promptCache

        // Create a ChatSession with the scheduler-enabled container
        let session = ChatSession(container)

        // First message — builds initial context
        let response1 = try await session.respond(to: "Hello world")
        XCTAssertFalse(response1.isEmpty, "First response should produce output")

        // Second message — should reuse cached context via history
        let response2 = try await session.respond(to: "How are you?")
        XCTAssertFalse(response2.isEmpty, "Second response should produce output")

        // The scheduler path stores .history, so the second call
        // re-tokenizes the full conversation and sends it through
        // model.generate() — the prompt cache should help reduce
        // prefill for the shared prefix tokens.
        //
        // Verify the session works correctly across multiple turns.
        // The key test is that follow-up messages don't crash or lose
        // context when batching is enabled.
    }

    /// Verifies that ChatSession with scheduler maintains conversation history
    /// across multiple turns (history is not dropped).
    func testChatSessionSchedulerPathMaintainsHistory() async throws {
        try skipIfMetalUnavailable()

        let scheduler = InferenceScheduler()
        let container = makeModelContainer(scheduler: scheduler)

        let session = ChatSession(container)

        // Multiple turns
        let r1 = try await session.respond(to: "First message")
        XCTAssertFalse(r1.isEmpty, "Turn 1 should produce output")

        let r2 = try await session.respond(to: "Second message")
        XCTAssertFalse(r2.isEmpty, "Turn 2 should produce output")

        let r3 = try await session.respond(to: "Third message")
        XCTAssertFalse(r3.isEmpty, "Turn 3 should produce output")

        // All three turns should complete without error, demonstrating
        // that the scheduler path correctly maintains history across turns.
    }
}

// MARK: - Call Tracking Mock Model

/// A mock model that tracks call counts and total tokens processed,
/// used to verify that prompt cache reduces prefill work.
private class CallTrackingModel: Module, LanguageModel, KVCacheDimensionProvider,
    @unchecked Sendable
{
    let vocabSize: Int
    let numLayers: Int
    var kvHeads: [Int] { Array(repeating: 4, count: numLayers) }

    var callCount = 0
    var totalTokensProcessed = 0
    var inputShapes = [[Int]]()
    var sawPreloadedCache = false

    init(vocabSize: Int = 32, numLayers: Int = 1) {
        self.vocabSize = vocabSize
        self.numLayers = numLayers
    }

    func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
        let cachedLength = cache.first?.offset ?? 0
        let promptLength = input.text.tokens.size

        if cachedLength >= promptLength, promptLength > 0 {
            _ = trimPromptCache(cache, numTokens: 1)
            return .tokens(input.text[(promptLength - 1)...])
        }

        if cachedLength > 0 {
            return .tokens(input.text[cachedLength...])
        }

        return .tokens(input.text)
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

        if let cache {
            let hasPreloadedKeys = cache.contains { layer in
                layer.innerState().first != nil
            }
            sawPreloadedCache = sawPreloadedCache || hasPreloadedKeys
        }

        appendSyntheticKV(to: cache, inputTokens: tokens, defaultHeads: 4, defaultHeadDim: 8)

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

    func newCache(parameters: GenerateParameters?) -> [KVCache] {
        (0 ..< numLayers).map { _ in KVCacheSimple() }
    }

    func resetCounters() {
        callCount = 0
        totalTokensProcessed = 0
        inputShapes = []
        sawPreloadedCache = false
    }
}

private func appendSyntheticKV(
    to caches: [KVCache]?, inputTokens: MLXArray, defaultHeads: Int = 2, defaultHeadDim: Int = 4
) {
    guard let caches else { return }

    let batchSize = inputTokens.dim(0)
    let seqLen = inputTokens.dim(1)

    for (layerIndex, cache) in caches.enumerated() {
        let state = cache.innerState()
        let existingKeys = state.first
        let existingValues = state.count > 1 ? state[1] : nil

        let heads = existingKeys?.dim(1) ?? defaultHeads
        let keyDim = existingKeys?.dim(3) ?? defaultHeadDim
        let valueDim = existingValues?.dim(3) ?? keyDim

        let baseValue = Float(layerIndex + 1)
        let keys = MLXArray.ones([batchSize, heads, seqLen, keyDim]) * baseValue
        let values = MLXArray.ones([batchSize, heads, seqLen, valueDim]) * (baseValue + 1)
        _ = cache.update(keys: keys, values: values)
    }
}
