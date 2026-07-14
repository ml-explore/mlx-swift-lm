// Copyright © 2026 Apple Inc.

import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import XCTest

/// Tests for the cancellation-before-next() ordering fix in generateLoopTask.
///
/// These tests are in a separate commit so they can be pruned before upstream PR
/// submission if the project prefers to keep tests closer to a model integration suite.
final class CancellationTests: XCTestCase {

    private func makeTinyModel() -> LlamaModel {
        let config = LlamaConfiguration(
            hiddenSize: 64, hiddenLayers: 4, intermediateSize: 128,
            attentionHeads: 8, rmsNormEps: 0.00001, vocabularySize: 100, kvHeads: 4)
        let model = LlamaModel(config)
        quantize(model: model, groupSize: 64, bits: 4)
        eval(model)
        return model
    }

    // Stream-consumer break cancels the task via onTermination. The generation task
    // must still settle (run Stream().synchronize()) before task.value returns.
    func testGenerateTaskSettlesAfterStreamCancellation() async throws {
        let model = makeTinyModel()
        let tokenizer = TestTokenizer()
        let configuration = ModelConfiguration(id: "test")

        let input = LMInput(tokens: MLXArray([1, 2, 3, 4, 5]))
        let iterator = try TokenIterator(
            input: input, model: model,
            parameters: GenerateParameters(maxTokens: 500))

        let (stream, task) = generateTask(
            promptTokenCount: input.text.tokens.size,
            modelConfiguration: configuration,
            tokenizer: tokenizer,
            iterator: iterator)

        // Consume one element then break. The for-await loop deallocation fires
        // continuation.onTermination → task.cancel().
        for await _ in stream { break }

        // Must complete without hanging. Failure here means Stream().synchronize()
        // was not reached or the task body was not allowed to run to completion.
        await task.value
    }

    // task.cancel() called before any stream consumption. The loop must observe
    // Task.isCancelled at the while condition and skip iterator.next() entirely
    // (or exit on the first check), then settle and complete.
    func testGenerateTaskSettlesAfterImmediateCancellation() async throws {
        let model = makeTinyModel()
        let tokenizer = TestTokenizer()
        let configuration = ModelConfiguration(id: "test")

        let input = LMInput(tokens: MLXArray([1, 2, 3, 4, 5]))
        let iterator = try TokenIterator(
            input: input, model: model,
            parameters: GenerateParameters(maxTokens: 500))

        let (stream, task) = generateTask(
            promptTokenCount: input.text.tokens.size,
            modelConfiguration: configuration,
            tokenizer: tokenizer,
            iterator: iterator)

        task.cancel()
        for await _ in stream {}

        await task.value
    }

    // Cancellation must be reported as .cancelled in the stream's completion info.
    func testGenerateTaskReportsCancelledStopReason() async throws {
        let model = makeTinyModel()
        let tokenizer = TestTokenizer()
        let configuration = ModelConfiguration(id: "test")

        let input = LMInput(tokens: MLXArray([1, 2, 3, 4, 5]))
        let iterator = try TokenIterator(
            input: input, model: model,
            parameters: GenerateParameters(maxTokens: 500))

        let (stream, task) = generateTask(
            promptTokenCount: input.text.tokens.size,
            modelConfiguration: configuration,
            tokenizer: tokenizer,
            iterator: iterator)

        task.cancel()

        var finalInfo: GenerateCompletionInfo?
        for await generation in stream {
            if case .info(let info) = generation { finalInfo = info }
        }

        await task.value
        XCTAssertEqual(finalInfo?.stopReason, .cancelled)
    }

    // The guard-nil exit path (iterator.next() returns nil at maxTokens) must still
    // report .length. This validates that the while loop's natural-termination path
    // feeds the correct stopReason through the post-loop check.
    func testGenerateTaskReportsLengthStopReasonAtMaxTokens() async throws {
        let model = makeTinyModel()
        let tokenizer = TestTokenizer()
        let configuration = ModelConfiguration(id: "test")

        let input = LMInput(tokens: MLXArray([1, 2, 3, 4, 5]))
        let maxTokens = 7
        let iterator = try TokenIterator(
            input: input, model: model,
            parameters: GenerateParameters(maxTokens: maxTokens))

        let (stream, task) = generateTask(
            promptTokenCount: input.text.tokens.size,
            modelConfiguration: configuration,
            tokenizer: tokenizer,
            iterator: iterator)

        var finalInfo: GenerateCompletionInfo?
        for await generation in stream {
            if case .info(let info) = generation { finalInfo = info }
        }

        await task.value
        XCTAssertEqual(finalInfo?.stopReason, .length)
        XCTAssertEqual(finalInfo?.generationTokenCount, maxTokens)
    }
}
