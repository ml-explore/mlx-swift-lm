// Copyright © 2025 Apple Inc.

import Foundation
import MLX
@preconcurrency @testable import MLXLMCommon
import MLXNN
import Tokenizers
import XCTest

// MARK: - Factory Resolution Order Tests

class DualPathRoutingTests: XCTestCase {

    /// Verify that ModelFactoryRegistry lists LLM before VLM by default.
    ///
    /// The default trampoline order should try MLXLLM first, then MLXVLM.
    /// This ensures dual-path models (e.g. Qwen 3.5) resolve as LLM
    /// when loaded via the generic `loadModel`/`loadModelContainer` APIs.
    func testFactoryRegistryPrefersLLMOverVLM() {
        let factories = ModelFactoryRegistry.shared.modelFactories()

        // Both factories should be available in the test environment
        guard factories.count >= 2 else {
            // In unit test context without both modules linked, we can at least
            // verify the trampoline array order via the registry's public API.
            // If only one factory is available, the ordering test is moot.
            return
        }

        // The first factory should be the LLM factory.
        // LLMModelFactory's modelRegistry is LLMRegistry; VLMModelFactory's is VLMRegistry.
        let firstFactory = factories[0]
        let secondFactory = factories[1]

        // LLMModelFactory uses LLMRegistry, VLMModelFactory uses VLMRegistry.
        // We distinguish by checking the type name of the model registry.
        let firstName = String(describing: type(of: firstFactory))
        let secondName = String(describing: type(of: secondFactory))

        XCTAssertTrue(
            firstName.contains("LLM"),
            "First factory should be LLM, got \(firstName)")
        XCTAssertTrue(
            secondName.contains("VLM"),
            "Second factory should be VLM, got \(secondName)")
    }

    // MARK: - VLM-Loaded Container Bypasses Scheduler

    /// A minimal mock model for testing the VLM guard in ModelContainer.generate().
    private class MinimalMockModel: Module, LanguageModel, KVCacheDimensionProvider,
        @unchecked Sendable
    {
        let vocabSize = 32
        var kvHeads: [Int] { [4] }

        func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
            .tokens(input.text)
        }

        func callAsFunction(
            _ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?
        ) -> LMOutput {
            let B = input.tokens.dim(0)
            let S = input.tokens.dim(1)
            // Return logits with token 0 as the highest probability (will hit EOS quickly)
            var flat = [Float](repeating: -100.0, count: B * S * vocabSize)
            for i in stride(from: 0, to: flat.count, by: vocabSize) {
                flat[i] = 0.0  // token 0 = EOS
            }
            return LMOutput(logits: MLXArray(flat, [B, S, vocabSize]))
        }

        func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
            weights
        }
    }

    /// Verify that a VLM-loaded ModelContainer with a scheduler set
    /// bypasses the scheduler and uses the direct TokenIterator path.
    func testVLMLoadedContainerBypassesScheduler() async throws {
        try skipIfMetalUnavailable()
        let model = MinimalMockModel()
        let tokenizer = TestTokenizer()
        let config = ModelConfiguration(id: "test-vlm-model")
        let processor = TestInputProcessor()

        // Create a ModelContext with loadedAsVLM = true
        let context = ModelContext(
            configuration: config,
            model: model,
            processor: processor,
            tokenizer: tokenizer,
            loadedAsVLM: true
        )

        // Create container WITH a scheduler — should be bypassed for VLM
        let scheduler = InferenceScheduler()
        let container = ModelContainer(context: context, scheduler: scheduler)

        // The scheduler should be set on the container
        XCTAssertNotNil(container.scheduler, "Scheduler should be set on container")

        // Submit a text-only request
        let input = LMInput(tokens: MLXArray([Int32(1), Int32(2), Int32(3)]))
        let params = GenerateParameters(maxTokens: 3, temperature: 0)

        let stream = try await container.generate(
            input: input,
            parameters: params
        )

        // The scheduler should NOT have been used — its state should still be idle
        let schedulerState = await scheduler.currentState
        XCTAssertEqual(
            schedulerState, "idle",
            "Scheduler should remain idle when container is VLM-loaded, got: \(schedulerState)")

        // Consume the stream to verify it completes (via direct TokenIterator path)
        var receivedOutput = false
        for await generation in stream {
            if generation.chunk != nil || generation.info != nil {
                receivedOutput = true
            }
        }
        XCTAssertTrue(receivedOutput, "Should receive output via direct TokenIterator path")
    }

    /// Verify that a non-VLM ModelContainer with a scheduler actually uses the scheduler.
    func testLLMLoadedContainerUsesScheduler() async throws {
        try skipIfMetalUnavailable()
        let model = MinimalMockModel()
        let tokenizer = TestTokenizer()
        let config = ModelConfiguration(id: "test-llm-model")
        let processor = TestInputProcessor()

        // Create a ModelContext with loadedAsVLM = false (default)
        let context = ModelContext(
            configuration: config,
            model: model,
            processor: processor,
            tokenizer: tokenizer
        )

        let scheduler = InferenceScheduler()
        let container = ModelContainer(context: context, scheduler: scheduler)

        let input = LMInput(tokens: MLXArray([Int32(1), Int32(2), Int32(3)]))
        let params = GenerateParameters(maxTokens: 3, temperature: 0)

        let stream = try await container.generate(
            input: input,
            parameters: params
        )

        // The scheduler should have been used — its state should NOT be idle
        let schedulerState = await scheduler.currentState
        XCTAssertNotEqual(
            schedulerState, "idle",
            "Scheduler should be active for LLM-loaded container, got: \(schedulerState)")

        // Consume the stream
        for await _ in stream {}
    }

    /// Verify that ModelContext defaults loadedAsVLM to false.
    func testModelContextDefaultsLoadedAsVLMToFalse() {
        let context = ModelContext(
            configuration: ModelConfiguration(id: "test"),
            model: MinimalMockModel(),
            processor: TestInputProcessor(),
            tokenizer: TestTokenizer()
        )
        XCTAssertFalse(context.loadedAsVLM, "loadedAsVLM should default to false")
    }
}
