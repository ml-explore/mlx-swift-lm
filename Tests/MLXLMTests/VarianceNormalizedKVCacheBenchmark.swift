// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import Testing

@testable import MLXLMCommon

/// Opt-in microbenchmarks for the variance-normalized KV-cache hot path.
///
/// Run with:
///
/// ```sh
/// MLX_RUN_VARN_BENCHMARKS=1 swift test --filter VarianceNormalizedKVCacheBenchmark
/// ```
///
/// These tests print measurements but deliberately enforce only correctness. Absolute GPU timing
/// is not stable enough for a portable CI threshold.
@Suite(.serialized)
struct VarianceNormalizedKVCacheBenchmark {
    private func median(_ values: [Double]) -> Double {
        let sorted = values.sorted()
        return sorted[sorted.count / 2]
    }

    private func relativeRMSE(_ actual: MLXArray, _ reference: MLXArray) -> Float {
        let numerator = sqrt(mean(square(actual.asType(.float32) - reference.asType(.float32))))
        let denominator = maximum(
            sqrt(mean(square(reference.asType(.float32)))), MLXArray(Float(1e-8)))
        let result = numerator / denominator
        eval(result)
        return result.item(Float.self)
    }

    @Test func decodeContextMatrix() throws {
        guard ProcessInfo.processInfo.environment["MLX_RUN_VARN_BENCHMARKS"] == "1" else { return }

        let contexts = [512, 1_024, 2_048, 3_968, 4_096, 8_192, 32_768]
        let queryHeads = 8
        let kvHeads = 4
        let headDimension = 128
        let scale = 1 / sqrt(Float(headDimension))

        for context in contexts {
            let cache = VarianceNormalizedKVCache(
                tileSize: 128, keyBits: 4, valueBits: 2, sinkhornIterations: 8)
            let prefillQuery = MLXRandom.normal([1, queryHeads, 1, headDimension]).asType(.float16)
            let prefillKeys = MLXRandom.normal([1, kvHeads, context, headDimension]).asType(
                .float16)
            let prefillValues = MLXRandom.normal([1, kvHeads, context, headDimension]).asType(
                .float16)
            let prefillOutput = cache.updateAndAttend(
                queries: prefillQuery, keys: prefillKeys, values: prefillValues, scale: scale)
            eval(prefillOutput)

            var samples: [Double] = []
            for iteration in 0 ..< 20 {
                let query = MLXRandom.normal([1, queryHeads, 1, headDimension]).asType(.float16)
                let key = MLXRandom.normal([1, kvHeads, 1, headDimension]).asType(.float16)
                let value = MLXRandom.normal([1, kvHeads, 1, headDimension]).asType(.float16)
                let start = ContinuousClock.now
                let output = cache.updateAndAttend(
                    queries: query, keys: key, values: value, scale: scale)
                eval(output)
                let elapsed = ContinuousClock.now - start
                if iteration >= 5 {
                    samples.append(
                        Double(elapsed.components.seconds) * 1_000
                            + Double(elapsed.components.attoseconds) / 1e15)
                }
                #expect(output.shape == [1, queryHeads, 1, headDimension])
            }

            print(
                String(
                    format:
                        "[VARNBENCH] context=%6d partitions=%2d median_decode=%7.3f ms compact=%8.2f MiB",
                    context, cache.attentionPartitionCount, median(samples),
                    Double(cache.compactStorageByteCount) / 1_048_576))
        }
    }

    @Test func sinkhornIterationMatrix() throws {
        guard ProcessInfo.processInfo.environment["MLX_RUN_VARN_BENCHMARKS"] == "1" else { return }

        let context = 4_096
        let queryHeads = 8
        let kvHeads = 4
        let headDimension = 128
        let scale = 1 / sqrt(Float(headDimension))
        let query = MLXRandom.normal([1, queryHeads, 1, headDimension]).asType(.float16)
        let keys = MLXRandom.normal([1, kvHeads, context, headDimension]).asType(.float16)
        let values = MLXRandom.normal([1, kvHeads, context, headDimension]).asType(.float16)
        let reference = MLXFast.scaledDotProductAttention(
            queries: query, keys: keys, values: values, scale: scale, mask: nil)
        eval(query, keys, values, reference)

        // Warm each graph shape before recording the ingest measurement.
        for iterations in [4, 8, 16] {
            let cache = VarianceNormalizedKVCache(
                tileSize: 128, keyBits: 4, valueBits: 2, sinkhornIterations: iterations)
            eval(cache.updateAndAttend(queries: query, keys: keys, values: values, scale: scale))
        }

        for iterations in [4, 8, 16] {
            let cache = VarianceNormalizedKVCache(
                tileSize: 128, keyBits: 4, valueBits: 2, sinkhornIterations: iterations)
            let start = ContinuousClock.now
            let output = cache.updateAndAttend(
                queries: query, keys: keys, values: values, scale: scale)
            eval(output)
            let elapsed = ContinuousClock.now - start
            let milliseconds =
                Double(elapsed.components.seconds) * 1_000
                + Double(elapsed.components.attoseconds) / 1e15
            print(
                String(
                    format:
                        "[VARNBENCH] sinkhorn=%2d ingest=%7.3f ms attention_relative_rmse=%.6f",
                    iterations, milliseconds, relativeRMSE(output, reference)))
        }
    }
}
