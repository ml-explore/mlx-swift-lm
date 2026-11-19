// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import XCTest

/// Compares a DeepSeek-V4.1 checkpoint with the outputs that
/// `tools/deepseek-v4-reference/deepseek-v41-checkpoint-reference.py` wrote for it with mlx-lm.
/// `tools/deepseek-v4-reference/deepseek-v41-truncate.py` makes a checkpoint with Engram off.
final class DeepseekV41CheckpointTests: XCTestCase {

    func testCheckpointMatchesReference() throws {
        let environment = ProcessInfo.processInfo.environment
        guard let path = environment["MLX_DEEPSEEK_V41_CHECKPOINT"],
            let referencePath = environment["MLX_DEEPSEEK_V41_REFERENCE"]
        else {
            throw XCTSkip(
                "Set MLX_DEEPSEEK_V41_CHECKPOINT to a DeepSeek-V4.1 checkpoint and "
                    + "MLX_DEEPSEEK_V41_REFERENCE to its reference outputs")
        }
        let directory = URL(filePath: path)
        let data = try Data(contentsOf: directory.appending(component: "config.json"))
        let config = try JSONDecoder().decode(DeepseekV41Configuration.self, from: data)
        try config.validateModelConfiguration()
        let base = try JSONDecoder().decode(BaseConfiguration.self, from: data)
        let (reference, metadata) = try loadArraysAndMetadata(url: URL(filePath: referencePath))
        let chunk = Int(metadata["chunk"] ?? "") ?? 256
        // mlx-lm's own uncached and cached paths differ by this much.
        let baselineAgreement = try XCTUnwrap(
            metadata["baseline_argmax_agreement"].flatMap(Float.init))
        let baselineKL = try XCTUnwrap(metadata["baseline_kl"].flatMap(Float.init))

        let model = DeepseekV41Model(config)
        try loadWeights(
            modelDirectory: directory, model: model,
            perLayerQuantization: base.perLayerQuantization)
        model.train(false)

        let tokens = try XCTUnwrap(reference["tokens"])
        let count = tokens.dim(0)

        let full = logProbabilities(model(tokens.reshaped(1, count), cache: nil)[0])
        let fullTail = try XCTUnwrap(reference["full_tail"])
        let argmaxMatch = agreement(
            argMax(full, axis: -1), try XCTUnwrap(reference["full_argmax"]))
        let tail = full[(-fullTail.dim(0))...]
        let tailKL = meanKL(fullTail, tail)
        let tailDiff = maxDifference(fullTail, tail)

        let cache = try model.newCache(parameters: nil)
        var logits = MLXArray()
        for offset in stride(from: 0, to: count, by: chunk) {
            let end = min(offset + chunk, count)
            logits = model(tokens[offset ..< end].reshaped(1, -1), cache: cache)
            eval(logits)
            eval(cache)
        }

        // Each step feeds the reference token, so one differing argmax does not
        // change the later steps.
        let generated = try XCTUnwrap(reference["generated"])
        var outputs = [logProbabilities(logits[0, -1])]
        for step in 0 ..< generated.dim(0) {
            outputs.append(
                logProbabilities(model(generated[step].reshaped(1, 1), cache: cache)[0, -1]))
        }
        let cached = stacked(outputs)
        let cachedReference = concatenated([
            try XCTUnwrap(reference["prefill_last"]).reshaped(1, -1),
            try XCTUnwrap(reference["decode"]),
        ])
        let cachedKL = meanKL(cachedReference, cached)
        let cachedDiff = maxDifference(cachedReference, cached)
        let greedy = argMax(cached[..<generated.dim(0)], axis: -1)
        let greedyMatches = Int(
            (greedy .== generated.asType(.int32)).asType(.int32).sum().item(Int32.self))

        print(
            """
            deepseek_v41 checkpoint vs mlx-lm: \(count) tokens, chunk \(chunk)
              mlx-lm uncached vs cached: argmax agreement \(baselineAgreement), KL \(baselineKL)
              uncached: argmax agreement \(argmaxMatch), tail KL \(tailKL), \
            tail max |dlogprob| \(tailDiff)
              cached prefill + decode: KL \(cachedKL), max |dlogprob| \(cachedDiff), \
            greedy matches \(greedyMatches)/\(generated.dim(0))
            """)
        let maxKL = 2 * baselineKL + 1e-3
        XCTAssertGreaterThan(argmaxMatch, baselineAgreement - 0.03)
        XCTAssertLessThan(tailKL, maxKL)
        XCTAssertLessThan(cachedKL, maxKL)
        XCTAssertGreaterThanOrEqual(greedyMatches, generated.dim(0) - 2)
    }

    private func logProbabilities(_ logits: MLXArray) -> MLXArray {
        let result = logSoftmax(logits.asType(.float32), axis: -1)
        eval(result)
        return result
    }

    private func agreement(_ a: MLXArray, _ b: MLXArray) -> Float {
        (a.asType(.int32) .== b.asType(.int32)).asType(.float32).mean().item(Float.self)
    }

    /// The mean over rows of KL(reference || output), from log-probabilities.
    private func meanKL(_ reference: MLXArray, _ output: MLXArray) -> Float {
        (exp(reference) * (reference - output)).sum(axis: -1).mean().item(Float.self)
    }

    private func maxDifference(_ a: MLXArray, _ b: MLXArray) -> Float {
        abs(a - b).max().item(Float.self)
    }
}
