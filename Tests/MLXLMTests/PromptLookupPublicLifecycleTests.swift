// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

// Deliberately not `@testable`: stopping early must work through public API alone.

/// Predicts from the tokens its sliding window actually holds, so stale lookahead changes output.
private final class PublicLookupModel: Module, LanguageModel, KVCacheDimensionProvider {
    var kvHeads: [Int] { [1] }

    func newCache(parameters: GenerateParameters?) -> [KVCache] {
        [RotatingKVCache(maxSize: 8, keep: 0)]
    }

    func prepare(
        _ input: LMInput, cache: [KVCache], state: LMOutput.State?, prefill: PrefillParameters
    ) throws -> PrepareResult {
        .tokens(input.text)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        callAsFunction(.init(tokens: inputs), cache: cache, state: nil).logits
    }

    func callAsFunction(
        _ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?
    ) -> LMOutput {
        let tokens = input.tokens.asArray(Int.self)
        let base = cache?.first?.offset ?? 0
        let keys = MLXArray(
            (base ..< base + tokens.count).flatMap { Array(repeating: Float($0), count: 32) },
            [1, 1, tokens.count, 32])
        let values = MLXArray(
            tokens.flatMap { Array(repeating: Float($0), count: 32) },
            [1, 1, tokens.count, 32])
        let presented = cache?.first?.update(keys: keys, values: values) ?? (keys, values)
        let cachedPositions = presented.0[0, 0, 0..., 0].asArray(Float.self).map {
            Int($0.rounded())
        }
        let cachedTokens = presented.1[0, 0, 0..., 0].asArray(Float.self).map {
            Int($0.rounded())
        }

        var logits = [Float](repeating: -100, count: tokens.count * 8)
        for (index, token) in tokens.enumerated() {
            let position = base + index
            let window = zip(cachedPositions, cachedTokens).reduce(0) { sum, entry in
                sum + ((position - 7 ... position).contains(entry.0) ? entry.1 : 0)
            }
            let prediction = position % 9 >= 6 ? (window + 1) % 8 : (token + 1) % 8
            logits[index * 8 + prediction] = 100
        }
        return LMOutput(logits: MLXArray(logits, [1, tokens.count, 8]))
    }
}

private func drain(_ iterator: inout some TokenIteratorProtocol) -> [Int] {
    var result = [Int]()
    while let token = iterator.next() { result.append(token) }
    return result
}

@Suite(.serialized)
struct PromptLookupPublicLifecycleTests {
    @Test func finishingEarlyLeavesAReusableCache() throws {
        let prompt = Array(repeating: Array(0 ..< 8), count: 3).flatMap { $0 }
        let followUp = [3, 4, 5]
        let configuration = PromptLookupConfiguration(
            maxDraftTokens: 4, maxNGramLength: 3, minimumConfidence: 0, adaptive: false)
        let continuation = GenerateParameters(maxTokens: 24, temperature: 0)
        var stopsWithLookahead = 0

        for stopAfter in 1 ... 12 {
            let model = PublicLookupModel()
            let cache = model.newCache(parameters: nil)
            var lookup = try PromptLookupTokenIterator(
                input: LMInput(tokens: MLXArray(prompt)), model: model, cache: cache,
                parameters: GenerateParameters(maxTokens: 64, temperature: 0),
                configuration: configuration)
            var emitted = [Int]()
            while emitted.count < stopAfter, let token = lookup.next() { emitted.append(token) }
            if cache[0].offset > prompt.count + emitted.count { stopsWithLookahead += 1 }

            lookup.finish()
            lookup.finish()
            #expect(lookup.next() == nil)
            #expect(cache[0].offset == prompt.count + emitted.count)

            var resumed = try TokenIterator(
                input: LMInput(tokens: MLXArray(followUp)), model: model, cache: cache,
                parameters: continuation)
            var reference = try TokenIterator(
                input: LMInput(tokens: MLXArray(prompt + emitted + followUp)),
                model: PublicLookupModel(), parameters: continuation)
            #expect(drain(&resumed) == drain(&reference), "stopAfter: \(stopAfter)")
        }
        // Some stops must fall inside an accepted block, or the rewind above is untested.
        #expect(stopsWithLookahead > 0)
    }
}
