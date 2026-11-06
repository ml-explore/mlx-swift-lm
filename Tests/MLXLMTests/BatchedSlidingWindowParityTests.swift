// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

/// A sliding-window model's batched caches must give every row the logits it gets on its own,
/// also once rows have wrapped their windows.
@Suite(.serialized)
struct BatchedSlidingWindowParityTests {

    @Test("A ragged batch of Gemma 3 prompts decodes past its window like separate sequences")
    func gemma3() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = tinyGemma3()
            let prompts: [[Int32]] = [[1, 2, 3], [4, 5, 6, 7, 8]]
            let longest = prompts.map(\.count).max()!

            var singles: [[any KVCache]] = []
            var expected: [MLXArray] = []
            for prompt in prompts {
                let cache = try model.newCache(parameters: nil)
                let logits = model(MLXArray(prompt).reshaped([1, prompt.count]), cache: cache)
                singles.append(cache)
                expected.append(logits[0..., (prompt.count - 1) ..< prompt.count, 0...])
            }

            let fresh = try prompts.map { _ in try model.newCache(parameters: nil) }
            let batched = try fresh[0].indices.map { layer in
                try mergeRows(fresh.map { $0[layer] })
            }
            #expect(batched.contains { $0 is BatchRotatingKVCache })
            #expect(batched.contains { $0 is BatchKVCache })
            batched.forEach { $0.prepare(lengths: prompts.map(\.count)) }
            let padded = prompts.flatMap { $0 + Array(repeating: 0, count: longest - $0.count) }
            let logits = model(MLXArray(padded).reshaped([prompts.count, longest]), cache: batched)
            batched.forEach { $0.finalize() }
            for (row, prompt) in prompts.enumerated() {
                let actual = logits[row ..< row + 1, (prompt.count - 1) ..< prompt.count, 0...]
                #expect(maxAbsDifference(actual, expected[row]) < 1e-4, "prefill, row \(row)")
            }

            // Twelve steps take both rows past the window of eight.
            for step in 0 ..< 12 {
                let token = Int32(10 + step % 6)
                let output = model(MLXArray([token, token]).reshaped([2, 1]), cache: batched)
                for row in prompts.indices {
                    let single = model(MLXArray([token]).reshaped([1, 1]), cache: singles[row])
                    #expect(
                        maxAbsDifference(output[row ..< row + 1], single) < 1e-4,
                        "step \(step), row \(row)")
                }
            }
            try expectExtractedRowsMatch(batched, singles)

            batched.forEach { filterRows($0, keeping: batchRows(1)) }
            let output = model(MLXArray([Int32(14)]).reshaped([1, 1]), cache: batched)
            let single = model(MLXArray([Int32(14)]).reshaped([1, 1]), cache: singles[1])
            #expect(maxAbsDifference(output, single) < 1e-4, "after filtering")
        }
    }

    @Test(
        "Prefilled rows, wrapped or not, merge and decode like separate sequences",
        arguments: [false, true])
    func mergedRows(overlong: Bool) throws {
        try Device.withDefaultDevice(.cpu) {
            let model = tinyGemma3()
            let singles = try [model.newCache(parameters: nil), model.newCache(parameters: nil)]
            if overlong {
                eval(model(MLXArray(Array(1 ... 11)).reshaped([1, 11]), cache: singles[0]))
            } else {
                eval(model(MLXArray(Array(1 ... 8)).reshaped([1, 8]), cache: singles[0]))
                for token in 9 ... 11 {
                    eval(model(MLXArray([token]).reshaped([1, 1]), cache: singles[0]))
                }
            }
            eval(model(MLXArray([1, 2, 3]).reshaped([1, 3]), cache: singles[1]))

            // The batch is built from copies, so the single caches continue on their own.
            let batched = try singles[0].indices.map { layer in
                try mergeRows(singles.map { $0[layer].copy() })
            }
            for cache in batched {
                let positioned = try #require(cache as? any BatchPositionedKVCache)
                #expect(positioned.batchOffset.asArray(Int.self) == [11, 3])
            }
            try expectExtractedRowsMatch(batched, singles)

            for step in 0 ..< 6 {
                let token = 4 + step
                let output = model(MLXArray([token, token]).reshaped([2, 1]), cache: batched)
                for row in 0 ..< 2 {
                    let expected = model(MLXArray([token]).reshaped([1, 1]), cache: singles[row])
                    #expect(
                        maxAbsDifference(output[row ..< row + 1], expected) < 1e-4,
                        "step \(step), row \(row)")
                }
            }
            try expectExtractedRowsMatch(batched, singles)

            batched.forEach { filterRows($0, keeping: batchRows(0)) }
            let continuation = MLXArray([4, 5]).reshaped([1, 2])
            let expected = model(continuation, cache: singles[0])
            #expect(maxAbsDifference(model(continuation, cache: batched), expected) < 1e-4)
        }
    }
}

func tinyGemma3() -> Gemma3TextModel {
    let configuration = Gemma3TextConfiguration(
        modelType: "gemma3_text", hiddenSize: 16, hiddenLayers: 4, intermediateSize: 32,
        attentionHeads: 2, headDim: 8, rmsNormEps: 1e-5, vocabularySize: 16, kvHeads: 1,
        ropeTheta: 1_000_000, ropeLocalBaseFreq: 10_000, ropeTraditional: false,
        queryPreAttnScalar: 8, slidingWindow: 8, slidingWindowPattern: 2,
        maxPositionEmbeddings: 128)
    return withRandomState(MLXRandom.RandomState(seed: 503)) {
        let model = Gemma3TextModel(configuration)
        eval(model)
        return model
    }
}

/// One layer's rows as a batch, on the row type.
private func mergeRows(_ caches: [any KVCache]) throws -> any KVCache {
    if let rows = caches as? [RotatingKVCache] {
        return try RotatingKVCache.merge(rows)
    }
    return KVCacheSimple.merge(try caches.map { try #require($0 as? KVCacheSimple) })
}

private func filterRows(_ cache: any KVCache, keeping indices: MLXArray) {
    switch cache {
    case let batch as BatchRotatingKVCache: batch.filter(batchIndices: indices)
    case let batch as BatchKVCache: batch.filter(batchIndices: indices)
    default: Issue.record("not a batched cache: \(type(of: cache))")
    }
}

private func extractRow(_ cache: any KVCache, _ index: Int) throws -> any KVCache {
    switch cache {
    case let batch as BatchRotatingKVCache: return batch.extract(index)
    case let batch as BatchKVCache: return batch.extract(index)
    default: throw KVCacheError(message: "not a batched cache: \(type(of: cache))")
    }
}

/// Every row extracted from `batch` holds the offset and the retained window of its single
/// cache.
private func expectExtractedRowsMatch(_ batch: [any KVCache], _ singles: [[any KVCache]])
    throws
{
    func retained(_ cache: any KVCache) throws -> [MLXArray] {
        guard let rotating = cache as? RotatingKVCache else { return cache.state }
        let window = try #require(rotating.maxSize)
        let view = try #require(rotating.logicalView(tail: window))
        return [view.0, view.1]
    }
    for (row, caches) in singles.enumerated() {
        try #require(caches.count == batch.count)
        for layer in batch.indices {
            let actual = try extractRow(batch[layer], row)
            #expect(actual.offset == caches[layer].offset, "row \(row), layer \(layer)")
            let a = try retained(actual)
            let b = try retained(caches[layer])
            try #require(a.count == b.count)
            for (actualArray, expectedArray) in zip(a, b) {
                try #require(actualArray.shape == expectedArray.shape, "row \(row), layer \(layer)")
                #expect(
                    maxAbsDifference(actualArray, expectedArray) < 1e-4,
                    "row \(row), layer \(layer)")
            }
        }
    }
}
