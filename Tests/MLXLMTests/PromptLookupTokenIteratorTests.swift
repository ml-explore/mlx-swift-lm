// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLMCommon

private let lookupTestBias = LMOutput.Key<Int>("tests.promptLookup.bias")

/// Predictions read the cache's actual causal token history, so a bad rollback changes output.
private final class LookupHistoryModel: Module, LanguageModel, KVCacheDimensionProvider {
    enum PrefillStyle: CaseIterable {
        case tokens
        case logits
    }

    let prefillStyle: PrefillStyle
    let contextSensitive: Bool
    let slidingWindow: Int?
    let additionalSlidingWindow: Int?
    let recurrent: Bool
    let softLogits: Bool
    var kvHeads: [Int] { [1] }
    private(set) var forwardWidths = [Int]()
    /// Start position, width, and whether the first cache was compressed, for each forward.
    private(set) var forwards = [(start: Int, width: Int, compressed: Bool)]()

    init(
        prefillStyle: PrefillStyle = .tokens, contextSensitive: Bool = true,
        slidingWindow: Int? = nil, additionalSlidingWindow: Int? = nil,
        recurrent: Bool = false, softLogits: Bool = false
    ) {
        self.prefillStyle = prefillStyle
        self.contextSensitive = contextSensitive
        self.slidingWindow = slidingWindow
        self.additionalSlidingWindow = additionalSlidingWindow
        self.recurrent = recurrent
        self.softLogits = softLogits
        super.init()
    }

    func newCache(parameters: GenerateParameters?) -> [KVCache] {
        var result: [KVCache]
        if let slidingWindow {
            result = [RotatingKVCache(maxSize: slidingWindow, keep: 0)]
        } else {
            result = [KVCacheSimple()]
        }
        if let additionalSlidingWindow {
            result.append(RotatingKVCache(maxSize: additionalSlidingWindow, keep: 0))
        }
        if recurrent { result.append(MambaCache()) }
        return result
    }

    func prepare(
        _ input: LMInput, cache: [KVCache], state: LMOutput.State?, prefill: PrefillParameters
    ) throws -> PrepareResult {
        switch prefillStyle {
        case .tokens:
            return .tokens(input.text)
        case .logits:
            return .logits(callAsFunction(input.text, cache: cache, state: state))
        }
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        callAsFunction(.init(tokens: inputs), cache: cache, state: nil).logits
    }

    func callAsFunction(
        _ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?
    ) -> LMOutput {
        let tokens = input.tokens.asArray(Int.self)
        let base = cache?.first?.offset ?? 0
        forwardWidths.append(tokens.count)
        forwards.append((base, tokens.count, cache?.first is QuantizedKVCache))
        let positions = Array(base ..< base + tokens.count)
        let keys = MLXArray(
            positions.flatMap { Array(repeating: Float($0), count: 32) },
            [1, 1, tokens.count, 32])
        let values = MLXArray(
            tokens.flatMap { Array(repeating: Float($0), count: 32) },
            [1, 1, tokens.count, 32])

        let presented: (MLXArray, MLXArray)
        if let quantized = cache?.first as? QuantizedKVCache {
            _ = quantized.updateQuantized(keys: keys, values: values)
            let materialized = quantized.toUnquantized().state
            presented = (materialized[0], materialized[1])
        } else if let cache = cache?.first {
            presented = cache.update(keys: keys, values: values)
        } else {
            presented = (keys, values)
        }
        let cachedPositions = presented.0[0, 0, 0..., 0].asArray(Float.self).map {
            Int($0.rounded())
        }
        let cachedTokens = presented.1[0, 0, 0..., 0].asArray(Float.self).map {
            Int($0.rounded())
        }
        for layer in cache?.dropFirst() ?? [] {
            if let recurrent = layer as? MambaCache {
                recurrent[0] = MLXArray(cachedTokens.reduce(0, +))
                recurrent.offset += tokens.count
            } else {
                _ = layer.update(keys: keys, values: values)
            }
        }

        var logits = [Float](repeating: softLogits ? -2 : -100, count: tokens.count * 8)
        for (index, token) in tokens.enumerated() {
            let position = base + index
            let contextSum = zip(cachedPositions, cachedTokens).reduce(0) { sum, entry in
                let (cachedPosition, cachedToken) = entry
                return sum
                    + ((position - 7 ... position).contains(cachedPosition) ? cachedToken : 0)
            }
            let prediction =
                contextSensitive && position % 9 >= 6
                ? (contextSum + 1) % 8 : (token + 1) % 8
            let biased = (prediction + (state?[lookupTestBias] ?? 0)) % 8
            logits[index * 8 + biased] = softLogits ? 1 : 100
        }
        return LMOutput(logits: MLXArray(logits, [1, tokens.count, 8]), state: state)
    }
}

private func lookupConfiguration(maxDraftTokens: Int = 4) -> PromptLookupConfiguration {
    PromptLookupConfiguration(
        maxDraftTokens: maxDraftTokens, maxNGramLength: 3, minNGramLength: 1,
        minimumOccurrences: 1, minimumConfidence: 0, adaptive: false)
}

private func lookupDrain(_ iterator: inout some TokenIteratorProtocol) -> [Int] {
    var result = [Int]()
    while let token = iterator.next() { result.append(token) }
    return result
}

@Suite(.serialized)
struct PromptLookupTokenIteratorTests {
    @Test(arguments: [false, true])
    func forwardsProcessorLifecycleAndCacheMetrics(recurrent: Bool) throws {
        struct EmissionProcessor: LogitProcessor, GenerationReasoningTokenCounting {
            var emittedCount = 0
            var finalized = false
            var generationReasoningTokenCount: Int? { finalized ? nil : emittedCount }

            mutating func prompt(_ prompt: MLXArray) {}
            func process(logits: MLXArray) -> MLXArray { logits }
            mutating func didSample(token: MLXArray) {}
            mutating func didEmit(token: Int) { emittedCount += 1 }
            mutating func finalizeGeneration() {
                #expect(!finalized)
                finalized = true
            }
        }

        let prompt = Array(repeating: Array(0 ..< 8), count: 3).flatMap { $0 }
        var iterator = try PromptLookupTokenIterator(
            input: LMInput(tokens: MLXArray(prompt)),
            model: LookupHistoryModel(
                contextSensitive: false, slidingWindow: 8, recurrent: recurrent),
            parameters: GenerateParameters(maxTokens: 48, temperature: 0),
            components: GenerationComponents(logitProcessorFactory: { EmissionProcessor() }),
            configuration: lookupConfiguration())

        for count in 1 ... 12 {
            let next = iterator.next()
            let token = try #require(next)
            #expect(iterator.reasoningTokenCount == count - 1)
            iterator.recordEmittedToken(token)
            #expect(iterator.reasoningTokenCount == count)
        }
        #expect(((iterator.speculativeDecodingTelemetry?.roundCount ?? 0) > 0) == !recurrent)
        iterator.finish()
        iterator.finish()
        #expect(iterator.reasoningTokenCount == nil)
        let evicted = iterator.cache.map(\.evictedTokenCount).max() ?? 0
        #expect(evicted > 0)
        #expect(iterator.evictedTokenCount == evicted)
    }

    @Test(arguments: [false, true], [0, 1, 2, 7, 48])
    func matchesAutoregressiveWithBothPrefillResults(
        returnsLogits: Bool, maxTokens: Int
    ) throws {
        let prefillStyle: LookupHistoryModel.PrefillStyle = returnsLogits ? .logits : .tokens
        let prompt = Array(repeating: Array(0 ..< 8), count: 3).flatMap { $0 }
        let parameters = GenerateParameters(maxTokens: maxTokens, temperature: 0)
        var plain = try TokenIterator(
            input: LMInput(tokens: MLXArray(prompt)),
            model: LookupHistoryModel(prefillStyle: prefillStyle), parameters: parameters)
        var lookup = try PromptLookupTokenIterator(
            input: LMInput(tokens: MLXArray(prompt)),
            model: LookupHistoryModel(prefillStyle: prefillStyle), parameters: parameters,
            configuration: lookupConfiguration())

        #expect(lookupDrain(&lookup) == lookupDrain(&plain))
        #expect(lookup.tokenCount == maxTokens)
        lookup.finalizeGeneration()
        #expect(lookup.cacheStorage.processedTokenCount <= prompt.count + maxTokens)
        #expect(lookup.cacheStorage.processedTokenCount >= prompt.count + maxTokens - 1)
    }

    @Test func rejectedDraftsDoNotChangeSubsequentContext() throws {
        let prompt = Array(repeating: Array(0 ..< 8), count: 4).flatMap { $0 }
        let parameters = GenerateParameters(maxTokens: 96, temperature: 0)
        var plain = try TokenIterator(
            input: LMInput(tokens: MLXArray(prompt)), model: LookupHistoryModel(),
            parameters: parameters)
        var lookup = try PromptLookupTokenIterator(
            input: LMInput(tokens: MLXArray(prompt)), model: LookupHistoryModel(),
            parameters: parameters, configuration: lookupConfiguration())

        #expect(lookupDrain(&lookup) == lookupDrain(&plain))
        let telemetry = try #require(lookup.speculativeDecodingTelemetry)
        #expect(telemetry.acceptedDraftTokenCount > 0)
        #expect(telemetry.rejectedDraftTokenCount > 0)
        #expect(telemetry.draftModelCallCount == 0)
        #expect(telemetry.targetModelCallCount == telemetry.roundCount)
    }

    @Test func adaptiveLookupRecoversAfterRejectionCooldown() throws {
        let prompt = Array(repeating: Array(0 ..< 8), count: 4).flatMap { $0 }
        let parameters = GenerateParameters(maxTokens: 96, temperature: 0)
        var state = LMOutput.State()
        state[lookupTestBias] = 1
        var configuration = lookupConfiguration()
        configuration.adaptive = true
        var plain = try TokenIterator(
            input: LMInput(tokens: MLXArray(prompt)),
            model: LookupHistoryModel(contextSensitive: false), state: state,
            parameters: parameters)
        var lookup = try PromptLookupTokenIterator(
            input: LMInput(tokens: MLXArray(prompt)),
            model: LookupHistoryModel(contextSensitive: false), state: state,
            parameters: parameters, configuration: configuration)

        #expect(lookupDrain(&lookup) == lookupDrain(&plain))
        let telemetry = try #require(lookup.speculativeDecodingTelemetry)
        #expect(telemetry.rejectedDraftTokenCount > 0)
        #expect(telemetry.acceptedDraftTokenCount > 0)
        #expect(lookup.suppressedLookupCount >= 16)
    }

    @Test func adaptiveLookupSkipsRepeatedMissesWithoutChangingOutput() throws {
        let prompt = Array(repeating: Array(0 ..< 8), count: 4).flatMap { $0 }
        let parameters = GenerateParameters(maxTokens: 64, temperature: 0)
        var configuration = lookupConfiguration()
        configuration.minimumOccurrences = 1000
        configuration.adaptive = true
        var plain = try TokenIterator(
            input: LMInput(tokens: MLXArray(prompt)), model: LookupHistoryModel(),
            parameters: parameters)
        var lookup = try PromptLookupTokenIterator(
            input: LMInput(tokens: MLXArray(prompt)), model: LookupHistoryModel(),
            parameters: parameters, configuration: configuration)

        #expect(lookupDrain(&lookup) == lookupDrain(&plain))
        #expect(lookup.lookupMissCount > 0)
        #expect(lookup.suppressedLookupCount > 0)
        #expect(lookup.fallbackTokenCount == 64)
        #expect(lookup.speculativeDecodingTelemetry?.roundCount == 0)
    }

    @Test func repeatedTextUsesFewerTargetForwards() throws {
        let prompt = Array(repeating: Array(0 ..< 8), count: 4).flatMap { $0 }
        let parameters = GenerateParameters(maxTokens: 80, temperature: 0)
        let plainModel = LookupHistoryModel(contextSensitive: false)
        let lookupModel = LookupHistoryModel(contextSensitive: false)
        var plain = try TokenIterator(
            input: LMInput(tokens: MLXArray(prompt)), model: plainModel, parameters: parameters)
        var lookup = try PromptLookupTokenIterator(
            input: LMInput(tokens: MLXArray(prompt)), model: lookupModel, parameters: parameters,
            configuration: lookupConfiguration())

        #expect(lookupDrain(&lookup) == lookupDrain(&plain))
        #expect(lookupModel.forwardWidths.count < plainModel.forwardWidths.count)
        #expect(try #require(lookup.speculativeDecodingTelemetry).acceptedDraftTokenCount > 0)
    }

    @Test(arguments: [1, 2, 3, 5, 9, 13, 21])
    func earlyStopRewindsTheWrappedWindow(stopAfter: Int) throws {
        let prompt = Array(repeating: Array(0 ..< 8), count: 3).flatMap { $0 }
        let model = LookupHistoryModel(contextSensitive: false, slidingWindow: 8)
        var lookup = try PromptLookupTokenIterator(
            input: LMInput(tokens: MLXArray(prompt)), model: model,
            parameters: GenerateParameters(maxTokens: 80, temperature: 0),
            configuration: lookupConfiguration())
        var emitted = [Int]()
        while emitted.count < stopAfter, let token = lookup.next() { emitted.append(token) }
        lookup.finalizeGeneration()

        let timeline = lookup.cacheStorage.processedTokenCount
        #expect(timeline <= prompt.count + emitted.count)
        #expect(timeline >= prompt.count + emitted.count - 1)
        #expect(lookup.cacheStorage.nativeAttentionOffsetsAreAligned)
        let ring = try #require(lookup.cache.first as? RotatingKVCache)
        let represented = Array((prompt + emitted).prefix(timeline))
        let view = try #require(ring.logicalView(tail: 8))
        #expect(
            view.1[0, 0, 0..., 0].asArray(Float.self).map { Int($0.rounded()) }
                == Array(represented.suffix(8)))
    }

    @Test func contextSensitiveGenerationContinuesPastWindowWraps() throws {
        let prompt = Array(repeating: Array(0 ..< 8), count: 3).flatMap { $0 }
        let parameters = GenerateParameters(maxTokens: 96, temperature: 0)
        var plain = try TokenIterator(
            input: LMInput(tokens: MLXArray(prompt)),
            model: LookupHistoryModel(slidingWindow: 8), parameters: parameters)
        var lookup = try PromptLookupTokenIterator(
            input: LMInput(tokens: MLXArray(prompt)),
            model: LookupHistoryModel(slidingWindow: 8), parameters: parameters,
            configuration: lookupConfiguration())

        #expect(lookupDrain(&lookup) == lookupDrain(&plain))
        #expect(try #require(lookup.speculativeDecodingTelemetry).roundCount > 3)
    }

    @Test func discardingAStopTokenPreservesTheRecordedTimeline() throws {
        let prompt = Array(repeating: Array(0 ..< 8), count: 3).flatMap { $0 }
        var iterator = try PromptLookupTokenIterator(
            input: LMInput(tokens: MLXArray(prompt)),
            model: LookupHistoryModel(contextSensitive: false, slidingWindow: 8),
            parameters: GenerateParameters(maxTokens: 32, temperature: 0),
            configuration: lookupConfiguration())
        let first = iterator.next()
        let second = iterator.next()
        _ = try #require(first)
        _ = try #require(second)
        iterator.discardGeneratedToken()
        iterator.finalizeGeneration()

        #expect(iterator.tokenCount == 2)
        #expect(iterator.cacheStorage.processedTokenCount == prompt.count + 2)
        #expect(iterator.cacheStorage.nativeAttentionOffsetsAreAligned)
    }

    @Test func warmCacheUsesFullHistoryAndCarriedModelState() throws {
        let prefix = Array(repeating: Array(0 ..< 8), count: 3).flatMap { $0 }
        let suffix = [0, 1, 2]
        let parameters = GenerateParameters(maxTokens: 32, temperature: 0)
        let model = LookupHistoryModel(contextSensitive: false)
        let cache = model.newCache(parameters: parameters)
        var state = LMOutput.State()
        state[lookupTestBias] = 1
        _ = model(.init(tokens: MLXArray(prefix)), cache: cache, state: state)

        var plain = try TokenIterator(
            input: LMInput(tokens: MLXArray(prefix + suffix)),
            model: LookupHistoryModel(contextSensitive: false), state: state,
            parameters: parameters)
        var lookup = try PromptLookupTokenIterator(
            input: LMInput(tokens: MLXArray(suffix)), model: model, cache: cache, state: state,
            parameters: parameters, configuration: lookupConfiguration(), history: prefix + suffix)

        #expect(lookupDrain(&lookup) == lookupDrain(&plain))
        #expect(lookup.state?[lookupTestBias] == 1)
    }

    @Test func penaltiesAndReferenceProcessorFollowAcceptedTokens() throws {
        final class RecordingProcessor: LogitProcessor, @unchecked Sendable {
            var samples = [Int]()
            func prompt(_ prompt: MLXArray) {}
            func process(logits: MLXArray) -> MLXArray { logits }
            func didSample(token: MLXArray) { samples.append(token.item(Int.self)) }
            func copy() -> Self {
                let result = RecordingProcessor()
                result.samples = samples
                return result as! Self
            }
        }
        let prompt = Array(repeating: Array(0 ..< 8), count: 3).flatMap { $0 }
        let parameters = GenerateParameters(
            maxTokens: 50, temperature: 0, repetitionPenalty: 1.5,
            presencePenalty: 0.5, frequencyPenalty: 0.3)
        let recording = RecordingProcessor()
        let components = GenerationComponents(logitProcessorFactory: { recording })
        var plain = try TokenIterator(
            input: LMInput(tokens: MLXArray(prompt)), model: LookupHistoryModel(),
            parameters: parameters)
        var lookup = try PromptLookupTokenIterator(
            input: LMInput(tokens: MLXArray(prompt)), model: LookupHistoryModel(),
            parameters: parameters, components: components, configuration: lookupConfiguration())

        let tokens = lookupDrain(&lookup)
        #expect(tokens == lookupDrain(&plain))
        #expect(Array(recording.samples.prefix(tokens.count)) == tokens)
        #expect(recording.samples.count <= tokens.count + 1)
    }

    @Test func seededSamplingMatchesAutoregressiveAndRepeats() throws {
        let prompt = Array(repeating: Array(0 ..< 8), count: 3).flatMap { $0 }
        let parameters = GenerateParameters(
            maxTokens: 64, temperature: 0.8, topP: 0.95, seed: 314)
        func run() throws -> [Int] {
            var iterator = try PromptLookupTokenIterator(
                input: LMInput(tokens: MLXArray(prompt)),
                model: LookupHistoryModel(softLogits: true), parameters: parameters,
                configuration: lookupConfiguration())
            return lookupDrain(&iterator)
        }

        let tokens = try run()
        var plain = try TokenIterator(
            input: LMInput(tokens: MLXArray(prompt)),
            model: LookupHistoryModel(softLogits: true), parameters: parameters)

        #expect(tokens == lookupDrain(&plain))
        #expect(try tokens == run())
    }

    @Test func sampledTokensRetainTheTargetDistribution() throws {
        let prompt = Array(repeating: Array(0 ..< 8), count: 3).flatMap { $0 }
        var preferred = 0
        var total = 0
        for seed in 0 ..< 12 {
            var iterator = try PromptLookupTokenIterator(
                input: LMInput(tokens: MLXArray(prompt)),
                model: LookupHistoryModel(contextSensitive: false, softLogits: true),
                parameters: GenerateParameters(maxTokens: 48, temperature: 1, seed: UInt64(seed)),
                configuration: lookupConfiguration())
            var previous = prompt.last!
            for token in lookupDrain(&iterator) {
                if token == (previous + 1) % 8 { preferred += 1 }
                previous = token
                total += 1
            }
        }

        let expected = 1 / (1 + 7 * exp(-3.0))
        #expect(abs(Double(preferred) / Double(total) - expected) < 0.08)
    }

    @Test func recurrentCacheFallsBackWithoutChangingOutput() throws {
        let prompt = Array(repeating: Array(0 ..< 8), count: 3).flatMap { $0 }
        let parameters = GenerateParameters(maxTokens: 32, temperature: 0)
        var plain = try TokenIterator(
            input: LMInput(tokens: MLXArray(prompt)), model: LookupHistoryModel(recurrent: true),
            parameters: parameters)
        var lookup = try PromptLookupTokenIterator(
            input: LMInput(tokens: MLXArray(prompt)), model: LookupHistoryModel(recurrent: true),
            parameters: parameters, configuration: lookupConfiguration())

        #expect(lookupDrain(&lookup) == lookupDrain(&plain))
        #expect(lookup.speculativeDecodingTelemetry?.roundCount == 0)
        #expect(lookup.fallbackTokenCount == 32)
        #expect(lookup.cacheStorage.nativeAttentionOffsetsAreAligned)
    }

    @Test func typedCompressionAppliesAcrossAThreshold() throws {
        let prompt = Array(repeating: Array(0 ..< 8), count: 2).flatMap { $0 }
        let compression = try AffineKVCacheConfiguration(
            bits: 8, groupSize: 32, compressionStart: prompt.count + 4)
        let parameters = GenerateParameters(
            maxTokens: 40, kvCache: .init(strategy: .affine(compression)), temperature: 0)
        var plain = try TokenIterator(
            input: LMInput(tokens: MLXArray(prompt)), model: LookupHistoryModel(),
            parameters: parameters)
        var lookup = try PromptLookupTokenIterator(
            input: LMInput(tokens: MLXArray(prompt)), model: LookupHistoryModel(),
            parameters: parameters, configuration: lookupConfiguration())

        #expect(lookupDrain(&lookup) == lookupDrain(&plain))
        #expect(lookup.cache.first is QuantizedKVCache)
        #expect(lookup.cacheStorage.nativeAttentionOffsetsAreAligned)
    }

    @Test(arguments: [0, 1, 4, 5, 9, 13])
    func roundsDoNotCrossTheCompressionBoundary(delay: Int) throws {
        let prompt = Array(repeating: Array(0 ..< 8), count: 2).flatMap { $0 }
        let start = prompt.count + delay
        let compression = try AffineKVCacheConfiguration(
            bits: 8, groupSize: 32, compressionStart: start)
        let parameters = GenerateParameters(
            maxTokens: 40, kvCache: .init(strategy: .affine(compression)), temperature: 0)
        let plainModel = LookupHistoryModel(contextSensitive: false)
        let lookupModel = LookupHistoryModel(contextSensitive: false)
        var plain = try TokenIterator(
            input: LMInput(tokens: MLXArray(prompt)), model: plainModel, parameters: parameters)
        var lookup = try PromptLookupTokenIterator(
            input: LMInput(tokens: MLXArray(prompt)), model: lookupModel,
            parameters: parameters, configuration: lookupConfiguration(maxDraftTokens: 8))

        #expect(lookupDrain(&lookup) == lookupDrain(&plain))
        #expect(try #require(lookup.speculativeDecodingTelemetry).roundCount > 1)
        // Ordinary decoding runs position p on a compressed cache exactly when p > start.
        for model in [plainModel, lookupModel] {
            for forward in model.forwards.dropFirst() {
                let last = forward.start + forward.width - 1
                #expect(forward.compressed == (forward.start > start))
                #expect(forward.compressed || last <= start)
            }
        }
    }

    @Test(arguments: [26, 128])
    func dynamicCompressionPreservesEarlyStopRestorePoints(compressionStart: Int) throws {
        let prompt = Array(repeating: Array(0 ..< 8), count: 3).flatMap { $0 }
        let compression = try AffineKVCacheConfiguration(
            bits: 8, groupSize: 32, compressionStart: compressionStart)
        var iterator = try PromptLookupTokenIterator(
            input: LMInput(tokens: MLXArray(prompt)),
            model: LookupHistoryModel(contextSensitive: false, additionalSlidingWindow: 8),
            parameters: GenerateParameters(
                maxTokens: 64, kvCache: .init(strategy: .affine(compression)), temperature: 0),
            configuration: lookupConfiguration())
        let first = iterator.next()
        let second = iterator.next()
        let emitted = try [#require(first), #require(second)]
        iterator.finalizeGeneration()

        let timeline = iterator.cacheStorage.processedTokenCount
        #expect(timeline == prompt.count + emitted.count)
        #expect(iterator.cacheStorage.nativeAttentionOffsetsAreAligned)
        let ring = try #require(iterator.cache[1] as? RotatingKVCache)
        let view = try #require(ring.logicalView(tail: 8))
        #expect(
            view.1[0, 0, 0..., 0].asArray(Float.self).map { Int($0.rounded()) }
                == Array((prompt + emitted).suffix(8)))
    }

    @Test func streamEOSFinalizesAnAcceptedBlock() async throws {
        let prompt = Array(repeating: Array(0 ..< 8), count: 3).flatMap { $0 } + [0, 1, 2, 3]
        let model = LookupHistoryModel(contextSensitive: false)
        let cache = model.newCache(parameters: nil)
        let iterator = try PromptLookupTokenIterator(
            input: LMInput(tokens: MLXArray(prompt)), model: model, cache: cache,
            parameters: GenerateParameters(maxTokens: 80, temperature: 0),
            configuration: lookupConfiguration())
        let (stream, task) = generateTask(
            promptTokenCount: prompt.count,
            modelConfiguration: ModelConfiguration(id: "lookup-eos", extraEOSTokens: ["7"]),
            tokenizer: LookupTestTokenizer(), iterator: iterator)
        var info: GenerateCompletionInfo?
        for await event in stream {
            if case .info(let completion) = event { info = completion }
        }
        await task.value

        let completion = try #require(info)
        #expect(completion.stopReason == .stop)
        #expect(completion.generationTokenCount == 3)
        #expect(cache[0].offset <= prompt.count + 4)
        #expect(cache[0].offset >= prompt.count + 3)
    }

    @Test func streamCancellationFinalizesAnAcceptedBlock() async throws {
        struct CancellingProcessor: LogitProcessor {
            var count = 0
            mutating func prompt(_ prompt: MLXArray) {}
            func process(logits: MLXArray) -> MLXArray { logits }
            mutating func didSample(token: MLXArray) {
                count += 1
                if count == 3 { withUnsafeCurrentTask { $0?.cancel() } }
            }
        }
        let prompt = Array(repeating: Array(0 ..< 8), count: 3).flatMap { $0 }
        let model = LookupHistoryModel(contextSensitive: false)
        let cache = model.newCache(parameters: nil)
        let iterator = try PromptLookupTokenIterator(
            input: LMInput(tokens: MLXArray(prompt)), model: model, cache: cache,
            parameters: GenerateParameters(maxTokens: 80, temperature: 0),
            components: GenerationComponents(logitProcessorFactory: { CancellingProcessor() }),
            configuration: lookupConfiguration())
        let (stream, task) = generateTask(
            promptTokenCount: prompt.count,
            modelConfiguration: ModelConfiguration(id: "lookup-cancel"),
            tokenizer: LookupTestTokenizer(), iterator: iterator)
        var info: GenerateCompletionInfo?
        for await event in stream {
            if case .info(let completion) = event { info = completion }
        }
        await task.value

        let completion = try #require(info)
        #expect(completion.stopReason == .cancelled)
        #expect(completion.generationTokenCount < 80)
        #expect(cache[0].offset <= prompt.count + completion.generationTokenCount)
        #expect(cache[0].offset >= prompt.count + completion.generationTokenCount - 1)
    }

    @Test func rawGenerationRoutesTheOptionalConfiguration() async throws {
        let tokenizer = LookupTestTokenizer()
        let configuration = ModelConfiguration(id: "lookup-routing")
        let context = ModelContext(
            configuration: configuration, model: LookupHistoryModel(contextSensitive: false),
            processor: TestInputProcessor(
                tokenizer: tokenizer, configuration: configuration,
                messageGenerator: DefaultMessageGenerator()),
            tokenizer: tokenizer)
        let prompt = Array(repeating: Array(0 ..< 8), count: 3).flatMap { $0 }
        let (stream, task) = try generateTokensTask(
            input: LMInput(tokens: MLXArray(prompt)),
            parameters: GenerateParameters(
                maxTokens: 24, temperature: 0, promptLookup: lookupConfiguration()),
            context: context)
        var tokens = [Int]()
        var info: GenerateCompletionInfo?
        for await item in stream {
            if let token = item.token { tokens.append(token) }
            if let completion = item.info { info = completion }
        }
        await task.value

        #expect(tokens == Array(repeating: Array(0 ..< 8), count: 3).flatMap { $0 })
        #expect(try #require(info?.speculativeDecodingTelemetry).roundCount > 0)
    }

    @Test func chatContinuationLooksUpTheReusedPromptPrefix() async throws {
        func makeSession(lookup: Bool) -> ChatSession {
            let tokenizer = LookupTestTokenizer()
            let configuration = ModelConfiguration(id: "lookup-chat")
            return ChatSession(
                ModelContext(
                    configuration: configuration,
                    model: LookupHistoryModel(contextSensitive: false),
                    processor: TestInputProcessor(
                        tokenizer: tokenizer, configuration: configuration,
                        messageGenerator: DefaultMessageGenerator()),
                    tokenizer: tokenizer),
                generateParameters: GenerateParameters(
                    maxTokens: 16, temperature: 0,
                    promptLookup: lookup ? lookupConfiguration() : nil))
        }
        let plain = makeSession(lookup: false)
        let lookup = makeSession(lookup: true)
        let prompt = String(repeating: "01234567", count: 3)
        let plainFirst = try await plain.respond(to: prompt)
        let lookupFirst = try await lookup.respond(to: prompt)
        #expect(lookupFirst == plainFirst)
        plain.generateParameters.maxTokens = 4
        lookup.generateParameters.maxTokens = 4

        let expected = try await plain.respond(to: "0123")
        var actual = ""
        var info: GenerateCompletionInfo?
        for try await item in lookup.streamDetails(to: "0123") {
            switch item {
            case .chunk(let text): actual += text
            case .info(let completion): info = completion
            default: break
            }
        }

        #expect(actual == expected)
        #expect(try #require(info).cachedPromptTokenCount > 0)
        #expect(try #require(info?.speculativeDecodingTelemetry).acceptedDraftTokenCount > 0)
    }

    @Test func invalidConfigurationDoesNotMutateTheSuppliedCache() throws {
        let cache = KVCacheSimple()
        var configuration = lookupConfiguration()
        configuration.maxDraftTokens = 0
        #expect(throws: PromptLookupError.self) {
            _ = try PromptLookupTokenIterator(
                input: LMInput(tokens: MLXArray([1, 2, 3])), model: LookupHistoryModel(),
                cache: [cache], parameters: GenerateParameters(), configuration: configuration)
        }
        #expect(cache.offset == 0)
    }
}

private struct LookupTestTokenizer: Tokenizer {
    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var eosTokenId: Int? { nil }
    var unknownToken: String? { nil }
    var unknownTokenId: Int? { nil }
    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        text.compactMap(\.wholeNumberValue)
    }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        tokenIds.map(String.init).joined()
    }
    func convertTokenToId(_ token: String) -> Int? { Int(token) }
    func convertIdToToken(_ id: Int) -> String? { String(id) }
    func applyChatTemplate(
        messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        messages.flatMap { encode(text: $0["content"] as? String ?? "", addSpecialTokens: false) }
    }
}
