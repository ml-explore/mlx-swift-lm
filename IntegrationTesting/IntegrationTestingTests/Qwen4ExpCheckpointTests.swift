// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import Tokenizers
import XCTest

/// The published Qwen3.8-Flash-Next OptiQ-2bit checkpoint against mlx-lm on the same weights.
///
/// Export the reference with tools/qwen4-exp-checkpoint-reference.py, then set
/// MLX_QWEN4_EXP_CHECKPOINT to the snapshot and MLX_QWEN4_EXP_REFERENCE_FILE to the output.
/// The model takes about 80 GB, so run one test at a time.
///
/// The two sides match bit for bit through the first 34 layers, then drift by bf16 rounding
/// in shared kernels, which MoE routing amplifies. So the tests compare greedy tokens and
/// top-k membership, and only print the logit drift.
final class Qwen4ExpCheckpointTests: XCTestCase {
    private let tokenizerLoader = #huggingFaceTokenizerLoader()

    private func inputs() throws -> (checkpoint: URL, reference: [String: MLXArray]) {
        let environment = ProcessInfo.processInfo.environment
        guard let checkpoint = environment["MLX_QWEN4_EXP_CHECKPOINT"],
            let reference = environment["MLX_QWEN4_EXP_REFERENCE_FILE"]
        else {
            throw XCTSkip(
                "Set MLX_QWEN4_EXP_CHECKPOINT and MLX_QWEN4_EXP_REFERENCE_FILE to run this test")
        }
        return (URL(filePath: checkpoint), try loadArrays(url: URL(filePath: reference)))
    }

    private func load(_ checkpoint: URL) async throws -> (ModelContext, Qwen4ExpModel) {
        let start = Date.timeIntervalSinceReferenceDate
        let context = try await LLMModelFactory.shared.load(
            from: checkpoint, using: tokenizerLoader)
        let model = try XCTUnwrap(context.model as? Qwen4ExpModel)
        print(
            "Qwen4Exp load_seconds=\(Date.timeIntervalSinceReferenceDate - start), "
                + "active_gib=\(Double(Memory.activeMemory) / Double(1 << 30))")
        return (context, model)
    }

    /// Last-position logits of a prefill in the reference's 127-token chunks.
    private func prefill(_ model: Qwen4ExpModel, _ tokens: MLXArray, cache: [KVCache]) throws
        -> MLXArray
    {
        let length = tokens.dim(-1)
        var output: MLXArray?
        for offset in stride(from: 0, to: length, by: 127) {
            output = model(tokens[0..., offset ..< min(offset + 127, length)], cache: cache)
            eval(output!)
        }
        return try XCTUnwrap(output)[0..., -1].asType(.float32)
    }

    /// Greedy tokens after `logits`, decoding one token at a time.
    private func greedy(
        _ model: Qwen4ExpModel, from logits: MLXArray, cache: [KVCache], steps: Int
    ) -> [Int] {
        var logits = logits
        var tokens: [Int] = []
        for _ in 0 ..< steps {
            let next = argMax(logits, axis: -1)
            tokens.append(next.item(Int.self))
            logits = model(next.reshaped(1, 1), cache: cache)[0..., -1].asType(.float32)
            eval(logits)
        }
        return tokens
    }

    private func printDrift(_ label: String, _ logits: MLXArray, _ expected: MLXArray) {
        let error = abs(logits - expected).max().item(Float.self)
        print("Qwen4Exp \(label): logits_max_abs=\(error)")
    }

    func testPublishedCheckpointMatchesPythonGreedy() async throws {
        let (checkpoint, reference) = try inputs()
        let (context, model) = try await load(checkpoint)
        XCTAssertEqual(context.configuration.toolCallFormat, .xmlFunction)

        let tokens = try XCTUnwrap(reference["tokens_19"]).reshaped(1, -1)
        let cache = try model.newCache(parameters: nil)
        let logits = try prefill(model, tokens, cache: cache)
        printDrift("prefill 19", logits, try XCTUnwrap(reference["logits_19"]))
        let expected = try XCTUnwrap(reference["greedy_19"]).asArray(Int.self)
        XCTAssertEqual(greedy(model, from: logits, cache: cache, steps: expected.count), expected)
    }

    /// The 257-token prompt spans three prefill chunks. Its logits are flat, so drift can
    /// swap near-tied tokens; the top-1 must stay among the reference's top 5.
    func testPublishedCheckpointChunkedPrefill() async throws {
        let (checkpoint, reference) = try inputs()
        let (_, model) = try await load(checkpoint)

        let tokens = try XCTUnwrap(reference["tokens_257"]).reshaped(1, -1)
        let logits = try prefill(model, tokens, cache: try model.newCache(parameters: nil))
        let expected = try XCTUnwrap(reference["logits_257"])
        printDrift("prefill 257", logits, expected)
        let top5 = argSort(-expected, axis: -1)[0..., ..<5].asArray(Int.self)
        XCTAssertTrue(top5.contains(argMax(logits, axis: -1).item(Int.self)), "\(top5)")
    }

    func testPublishedCheckpointAnswersLikePython() async throws {
        let (checkpoint, reference) = try inputs()
        let (context, model) = try await load(checkpoint)

        let input = try await context.processor.prepare(
            input: UserInput(
                chat: [.user("What is the capital of France? Answer in one short sentence.")],
                additionalContext: ["enable_thinking": false]))
        let prompt = try XCTUnwrap(reference["tokens_chat"]).asArray(Int.self)
        XCTAssertEqual(input.text.tokens.asArray(Int.self), prompt, "chat template")

        let cache = try model.newCache(parameters: nil)
        let logits = try prefill(model, input.text.tokens.reshaped(1, -1), cache: cache)
        printDrift("chat prefill", logits, try XCTUnwrap(reference["logits_chat"]))

        // Generation stops at the end of the turn; what the reference decodes after it is noise.
        let endOfTurn = try XCTUnwrap(context.tokenizer.convertTokenToId("<|im_end|>"))
        let reply = try XCTUnwrap(reference["greedy_chat"]).asArray(Int.self)
        let answer = Array(reply.prefix(through: try XCTUnwrap(reply.firstIndex(of: endOfTurn))))
        let generated = greedy(model, from: logits, cache: cache, steps: answer.count)
        print("Qwen4Exp answer: \(context.tokenizer.decode(tokenIds: generated).debugDescription)")
        XCTAssertEqual(generated, answer)
    }
}
