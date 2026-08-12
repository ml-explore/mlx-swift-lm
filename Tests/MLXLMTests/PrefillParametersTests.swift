// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import Testing

/// Unit tests for ``PrefillParameters``: the chunk-length arithmetic, the
/// chunk schedule the default `LLMModel.prepare` derives from it, and the
/// progress-callback sequence observed through a real `TokenIterator`.
struct PrefillParametersTests {

    // MARK: - chunkLength arithmetic

    @Test("balanced chunk length: minimal equal chunks under the ceiling")
    func balancedArithmetic() {
        let prefill = PrefillParameters(stepSize: nil, chunking: .balanced)
        for (count, step) in [(31884, 1024), (5000, 512), (1025, 1024), (7, 3), (100, 100)] {
            let chunk = prefill.chunkLength(forChunking: count, defaultStepSize: step)!
            #expect(chunk <= step)
            // the chunk count the ceiling dictates is not exceeded
            let minimalCount = (count + step - 1) / step
            #expect((count + chunk - 1) / chunk == minimalCount)
            // near-equal: the smallest chunk is within chunkCount-1 of the largest
            let lastChunk = count - (minimalCount - 1) * chunk
            #expect(lastChunk > chunk - minimalCount)
            #expect(lastChunk <= chunk)
        }
    }

    @Test("balanced degenerates safely: fits-in-one, empty, non-positive step")
    func balancedDegenerates() {
        let prefill = PrefillParameters(chunking: .balanced)
        // count at or under the step: one chunk of exactly count
        #expect(prefill.chunkLength(forChunking: 24, defaultStepSize: 128) == 24)
        #expect(prefill.chunkLength(forChunking: 512, defaultStepSize: 512) == 512)
        // empty and non-positive floor to 1
        #expect(prefill.chunkLength(forChunking: 0, defaultStepSize: 512) == 1)
        #expect(
            PrefillParameters(stepSize: 0).chunkLength(forChunking: 10, defaultStepSize: 512) == 1)
    }

    @Test("stepSize overrides the default; maximumStepSize caps both")
    func stepResolution() {
        // explicit stepSize wins over the model default
        #expect(
            PrefillParameters(stepSize: 8, chunking: .remainder)
                .chunkLength(forChunking: 100, defaultStepSize: 512) == 8)
        // the cap clamps an explicit stepSize
        #expect(
            PrefillParameters(stepSize: 512, chunking: .remainder)
                .chunkLength(forChunking: 100, defaultStepSize: 128, maximumStepSize: 64) == 64)
        // and clamps the default too
        #expect(
            PrefillParameters(chunking: .remainder)
                .chunkLength(forChunking: 100, defaultStepSize: 128, maximumStepSize: 64) == 64)
    }

    @Test("unchunked resolves to nil at any length")
    func unchunkedIsNil() {
        let prefill = PrefillParameters(chunking: .unchunked)
        #expect(prefill.chunkLength(forChunking: 5) == nil)
        #expect(prefill.chunkLength(forChunking: 31884) == nil)
    }

    // MARK: - forEachChunk schedule

    /// Collects the chunk ranges `forEachChunk` drives: the schedule the
    /// shipping code produces, plus the positions left for the caller's final
    /// forward.
    private static func schedule(
        total: Int, reserving: Int = 1, prefill: PrefillParameters
    ) throws -> (chunks: [Range<Int>], tail: Int) {
        var chunks: [Range<Int>] = []
        let processed = try prefill.forEachChunk(total: total, reserving: reserving) {
            chunks.append($0)
        }
        return (chunks, total - processed)
    }

    /// Pins the exact schedule of the banked measurement (31,885 tokens at a
    /// 1024 ceiling on Qwen3.6-35B-A3B): balanced runs 32 near-equal chunks
    /// and hands the iterator one token, where the legacy stride ran 31 full
    /// chunks and left a degenerate 141-token remainder. `reserving` values
    /// mirror `LLMModel.prepare`'s calls.
    @Test("schedule pin: 31,885 tokens @ 1024")
    func schedulePin() throws {
        let balanced = try Self.schedule(
            total: 31885, prefill: .init(stepSize: 1024, chunking: .balanced))
        #expect(balanced.chunks.map(\.count) == Array(repeating: 997, count: 31) + [977])
        #expect(balanced.chunks.first?.lowerBound == 0)
        #expect(balanced.chunks.last?.upperBound == 31884)
        #expect(balanced.tail == 1)

        let legacy = try Self.schedule(
            total: 31885, reserving: 1024, prefill: .init(stepSize: 1024, chunking: .remainder))
        #expect(legacy.chunks.map(\.count) == Array(repeating: 1024, count: 31))
        #expect(legacy.tail == 141)
    }

    @Test("schedule: fits-in-one-chunk totals become a single chunk; unchunked, none")
    func scheduleShortPrompt() throws {
        let balanced = try Self.schedule(
            total: 300, prefill: .init(stepSize: 512, chunking: .balanced))
        #expect(balanced.chunks.map(\.count) == [299])
        #expect(balanced.tail == 1)

        let unchunked = try Self.schedule(
            total: 300, prefill: .init(stepSize: 512, chunking: .unchunked))
        #expect(unchunked.chunks.isEmpty)
        #expect(unchunked.tail == 300)
    }

    // MARK: - progress through a real TokenIterator

    private final class ProgressLog: @unchecked Sendable {
        var events: [[Int]] = []
    }

    private static func makeModel() -> LlamaModel {
        let config = LlamaConfiguration(
            hiddenSize: 32, hiddenLayers: 2, intermediateSize: 64, attentionHeads: 4,
            rmsNormEps: 0.00001, vocabularySize: 100, kvHeads: 2)
        return LlamaModel(config)
    }

    private static func prefillEvents(
        promptTokens: Int, prefill: PrefillParameters
    ) throws -> (events: [[Int]], cacheOffsets: [Int]) {
        let model = makeModel()
        let log = ProgressLog()
        var parameters = GenerateParameters(temperature: 0)
        parameters.prefill = prefill
        parameters.prefill.progress = { processed, total in
            log.events.append([processed, total])
        }
        let prompt = MLXArray((0 ..< promptTokens).map { Int32($0 % 100) })
        let cache = model.newCache(parameters: parameters)
        _ = try TokenIterator(
            input: LMInput(tokens: prompt), model: model, cache: cache, parameters: parameters)
        return (log.events, cache.map { $0.offset })
    }

    @Test("balanced progress: equal chunk boundaries, then the terminal call")
    func progressBalanced() throws {
        let result = try Self.prefillEvents(
            promptTokens: 25, prefill: .init(stepSize: 8, chunking: .balanced))
        #expect(result.events == [[8, 25], [16, 25], [24, 25], [25, 25]])
        #expect(result.cacheOffsets.allSatisfy { $0 == 25 })

        // a non-divisible count balances below the ceiling: ceil(24/4) = 6 per chunk
        let uneven = try Self.prefillEvents(
            promptTokens: 25, prefill: .init(stepSize: 7, chunking: .balanced))
        #expect(uneven.events == [[6, 25], [12, 25], [18, 25], [24, 25], [25, 25]])
    }

    @Test("remainder progress: stride boundaries, remainder finishes at the terminal")
    func progressRemainder() throws {
        let result = try Self.prefillEvents(
            promptTokens: 26, prefill: .init(stepSize: 8, chunking: .remainder))
        #expect(result.events == [[8, 26], [16, 26], [24, 26], [26, 26]])
        #expect(result.cacheOffsets.allSatisfy { $0 == 26 })
    }

    @Test("unchunked and fits-in-one-chunk prompts report only the terminal call")
    func progressSingleForward() throws {
        let unchunked = try Self.prefillEvents(
            promptTokens: 25, prefill: .init(stepSize: 8, chunking: .unchunked))
        #expect(unchunked.events == [[25, 25]])
        #expect(unchunked.cacheOffsets.allSatisfy { $0 == 25 })

        let short = try Self.prefillEvents(
            promptTokens: 5, prefill: .init(stepSize: 512, chunking: .balanced))
        #expect(short.events == [[5, 5]])
        #expect(short.cacheOffsets.allSatisfy { $0 == 5 })
    }
}
