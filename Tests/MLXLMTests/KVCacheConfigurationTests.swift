// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLMCommon

@Suite("KV-cache configuration")
struct KVCacheConfigurationTests {
    @Test func typedPresetsExposeStableAlgorithmIdentity() {
        let quality = KVCacheConfiguration(
            strategy: .turboQuant(.qualityFirst))
        let balanced = KVCacheConfiguration(
            strategy: .turboQuant(.balanced))
        let affine = KVCacheConfiguration(strategy: .affine(.fourBit))

        #expect(quality.strategy.identifier == .turboQuant)
        #expect(balanced.strategy.identifier == .turboQuant)
        #expect(affine.strategy.identifier == .affine)
        #expect(TurboQuantKVCacheConfiguration.qualityFirst.keyPrecision == .fp16)
        #expect(TurboQuantKVCacheConfiguration.qualityFirst.valuePrecision == .fourBit)
        #expect(TurboQuantKVCacheConfiguration.balanced.keyPrecision == .affineEightBit)
        #expect(TurboQuantKVCacheConfiguration.balanced.valuePrecision == .threeBit)
    }

    @Test func invalidTypedValuesAreRejectedAtConstruction() {
        #expect(throws: KVCacheConfigurationError.invalidCapacity(0)) {
            try KVCacheConfiguration.Capacity(maxTokens: 0)
        }
        #expect(throws: KVCacheConfigurationError.invalidPreservedPrefix(8, capacity: 8)) {
            try KVCacheConfiguration.Capacity(maxTokens: 8, preservedPrefixTokens: 8)
        }
        for bits in [1, 7, 9] {
            #expect(throws: KVCacheConfigurationError.invalidAffineBits(bits)) {
                try AffineKVCacheConfiguration(bits: bits)
            }
        }
        #expect(throws: KVCacheConfigurationError.invalidGroupSize(0)) {
            try AffineKVCacheConfiguration(bits: 4, groupSize: 0)
        }
    }

    @Test func legacyTurboSchemeResolvesToTypedConfiguration() throws {
        let parameters = GenerateParameters(
            kvGroupSize: 64,
            quantizedKVStart: 128,
            kvScheme: "turbo8v3")

        let optionalConfiguration = try parameters.resolvedKVCacheConfiguration()
        let resolved = try #require(optionalConfiguration)
        #expect(resolved.strategy.identifier == .turboQuant)
        guard case .turboQuant(let turbo) = resolved.strategy.storage else {
            Issue.record("Expected a TurboQuant strategy")
            return
        }
        #expect(turbo.keyPrecision == .affineEightBit)
        #expect(turbo.valuePrecision == .threeBit)
        #expect(turbo.compressionStart == 128)
    }

    @Test func unknownLegacySchemeIsRejected() {
        let parameters = GenerateParameters(kvScheme: "future-unregistered-scheme")
        #expect(
            throws: KVCacheConfigurationError.unsupportedLegacyScheme(
                "future-unregistered-scheme")
        ) {
            try parameters.resolvedKVCacheConfiguration()
        }
    }

    @Test func typedAndLegacyConfigurationCannotBeMixed() {
        let parameters = GenerateParameters(
            kvCache: .init(strategy: .turboQuant(.balanced)),
            kvBits: 4)
        #expect(throws: KVCacheConfigurationError.conflictingLegacyConfiguration) {
            try parameters.resolvedKVCacheConfiguration()
        }
    }

    @Test func capacityPreservesCallerSelectedPrefix() throws {
        let capacity = try KVCacheConfiguration.Capacity(
            maxTokens: 4_096, preservedPrefixTokens: 16)
        let parameters = GenerateParameters(kvCache: .init(capacity: capacity))
        let resolved = try parameters.resolvedKVCacheConfiguration()

        #expect(parameters.effectiveKVCacheCapacity == capacity)
        #expect(resolved?.capacity == capacity)
    }

    @Test func typedConfigurationRejectsAllRotatingCacheByDefault() {
        let configuration = KVCacheConfiguration(
            strategy: .turboQuant(.balanced))
        let cache: [KVCache] = [
            RotatingKVCache(maxSize: 128),
            RotatingKVCache(maxSize: 128),
        ]

        #expect(
            throws: KVCacheConfigurationError.noCompatibleLayers(strategy: .turboQuant)
        ) {
            try validateKVCacheCompatibility(cache, configuration: configuration)
        }
    }

    @Test func requireAllLayersRejectsMixedRotatingTopology() {
        let configuration = KVCacheConfiguration(
            strategy: .affine(.fourBit),
            compatibility: .requireAllLayers)
        let cache: [KVCache] = [
            KVCacheSimple(),
            RotatingKVCache(maxSize: 128),
        ]

        #expect(
            throws: KVCacheConfigurationError.incompatibleLayers(
                strategy: .affine, count: 1)
        ) {
            try validateKVCacheCompatibility(cache, configuration: configuration)
        }
    }

    @Test func runtimeReportRecursesThroughCompositeCaches() {
        let configuration = KVCacheConfiguration(strategy: .affine(.fourBit))
        let cache: [KVCache] = [
            CacheList(
                MambaCache(),
                QuantizedKVCache(groupSize: 64, bits: 4)),
            RotatingKVCache(maxSize: 128),
        ]

        let report = kvCacheRuntimeReport(cache: cache, configuration: configuration)

        #expect(report.compressedLayerCount == 1)
        #expect(report.pendingLayerCount == 0)
        #expect(report.skippedLayerCount == 1)
        #expect(report.processedTokenCount == nil)
        #expect(report.layers.count == 3)
        #expect(report.layers[0].path == [0, 0])
        #expect(report.layers[0].state == .notApplicable)
        #expect(report.layers[1].path == [0, 1])
        #expect(report.layers[1].resolvedStrategy == .affine)
        #expect(report.layers[2].reason == .slidingWindow)
    }

    @Test func typedTurboQuantDispatchRewritesNestedAttentionCache() throws {
        let simple = KVCacheSimple()
        simple.offset = 8
        let list = CacheList(MambaCache(), simple)
        var cache: [KVCache] = [list]
        let configuration = KVCacheConfiguration(
            strategy: .turboQuant(.qualityFirst),
            compatibility: .requireAllLayers)

        try validateKVCacheCompatibility(cache, configuration: configuration)
        _ = try applyKVCacheConfiguration(cache: &cache, configuration: configuration)

        #expect(list[0] is MambaCache)
        #expect(list[1] is TurboQuantKVCache)
        let report = kvCacheRuntimeReport(cache: cache, configuration: configuration)
        #expect(report.pendingLayerCount == 1)
        #expect(report.skippedLayerCount == 0)
    }

    @Test func realizedUnsupportedShapeFailsClosed() throws {
        var cache: [KVCache] = [populatedSimpleCache(headDimension: 48)]
        let configuration = KVCacheConfiguration(
            strategy: .affine(.fourBit),
            compatibility: .requireAtLeastOneLayer)

        #expect(
            throws: KVCacheConfigurationError.noCompatibleLayers(strategy: .affine)
        ) {
            _ = try applyKVCacheConfiguration(cache: &cache, configuration: configuration)
        }
        #expect(cache[0] is KVCacheSimple)
    }

    @Test func mixedRealizedShapesReportConversionAndSkip() throws {
        var cache = mixedShapeCache()
        let configuration = KVCacheConfiguration(
            strategy: .affine(.fourBit),
            compatibility: .allowPartial)

        let result = try applyKVCacheConfiguration(
            cache: &cache, configuration: configuration)

        #expect(result.convertedLayerCount == 1)
        #expect(result.alreadyCompatibleLayerCount == 0)
        #expect(result.skipped.count == 1)
        #expect(result.skipped[0].path == [1])
        #expect(result.skipped[0].reason == .unsupportedShape)
        #expect(cache[0] is QuantizedKVCache)
        #expect(cache[1] is KVCacheSimple)
    }

    @Test func requireAllLayersRejectsMixedRealizedShapesWithoutMutation() throws {
        var cache = mixedShapeCache()
        let configuration = KVCacheConfiguration(
            strategy: .affine(.fourBit),
            compatibility: .requireAllLayers)

        #expect(
            throws: KVCacheConfigurationError.incompatibleLayers(
                strategy: .affine, count: 1)
        ) {
            _ = try applyKVCacheConfiguration(cache: &cache, configuration: configuration)
        }
        #expect(cache[0] is KVCacheSimple)
        #expect(cache[1] is KVCacheSimple)
    }

    @Test func explicitFullPrecisionRejectsCompressedCache() {
        let configuration = KVCacheConfiguration(strategy: .fullPrecision)
        let cache: [KVCache] = [QuantizedKVCache(groupSize: 64, bits: 4)]

        #expect(
            throws: KVCacheConfigurationError.incompatibleLayers(
                strategy: .fullPrecision, count: 1)
        ) {
            try validateKVCacheCompatibility(cache, configuration: configuration)
        }
    }

    @Test func explicitFullPrecisionReportsSimpleCachesAsCompatible() throws {
        var cache: [KVCache] = [KVCacheSimple(), KVCacheSimple()]

        let result = try applyKVCacheConfiguration(
            cache: &cache,
            configuration: KVCacheConfiguration(strategy: .fullPrecision))

        #expect(result.convertedLayerCount == 0)
        #expect(result.alreadyCompatibleLayerCount == 2)
        #expect(result.pendingLayerCount == 0)
        #expect(result.skipped.isEmpty)
    }

    @Test func configuredCacheMustRealizeRequestedCapacity() throws {
        let configuration = KVCacheConfiguration(
            capacity: try .init(maxTokens: 128, preservedPrefixTokens: 4))
        let suppliedCapacity = try KVCacheConfiguration.Capacity(
            maxTokens: 64, preservedPrefixTokens: 4)
        let cache: [KVCache] = [suppliedCapacity.makeRotatingCache()]

        #expect(
            throws: KVCacheConfigurationError.incompatibleCapacity(expected: 128, count: 1)
        ) {
            try validateKVCacheCompatibility(cache, configuration: configuration)
        }
    }

    @Test func modelNativeSlidingWindowDoesNotConflictWithRequestedCapacity() throws {
        let configuration = KVCacheConfiguration(
            capacity: try .init(maxTokens: 128, preservedPrefixTokens: 4))
        let nativeSlidingCache = RotatingKVCache(maxSize: 512, keep: 0)
        let configuredCache = try KVCacheConfiguration.Capacity(
            maxTokens: 128, preservedPrefixTokens: 4
        ).makeRotatingCache()

        try validateKVCacheCompatibility(
            [nativeSlidingCache, configuredCache], configuration: configuration)

        #expect(nativeSlidingCache.preservedPrefixTokens == 0)
        #expect(configuredCache.preservedPrefixTokens == 4)
    }

    @Test func requestedCapacityOriginSurvivesPromptCacheRoundTrip() throws {
        let requested = try KVCacheConfiguration.Capacity(
            maxTokens: 64, preservedPrefixTokens: 4)
        let cache = requested.makeRotatingCache()
        _ = cache.update(
            keys: MLXArray.ones([1, 1, 8, 16]),
            values: MLXArray.ones([1, 1, 8, 16]))

        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("safetensors")
        defer { try? FileManager.default.removeItem(at: url) }

        try savePromptCache(url: url, cache: [cache])
        let (loaded, _) = try loadPromptCache(url: url)
        let restored = try #require(loaded.first as? RotatingKVCache)

        #expect(restored.capacityOrigin == .requested)
        try validateKVCacheCompatibility(
            loaded, configuration: KVCacheConfiguration(capacity: requested))

        let incompatible = KVCacheConfiguration(
            capacity: try .init(maxTokens: 128, preservedPrefixTokens: 4))
        #expect(
            throws: KVCacheConfigurationError.incompatibleCapacity(expected: 128, count: 1)
        ) {
            try validateKVCacheCompatibility(loaded, configuration: incompatible)
        }
    }

    @Test func legacyRotatingMetadataDefaultsToModelNativeCapacity() {
        let cache = RotatingKVCache(maxSize: 64)
        cache.capacityOrigin = .requested

        cache.metaState = ["0", "64", "256", "0", "0"]

        #expect(cache.capacityOrigin == .modelNative)
    }

    @Test func sharedRealizationPublishesConversionAndBecomesTerminal() throws {
        let affine = try AffineKVCacheConfiguration(
            bits: 4, groupSize: 64, compressionStart: 1)
        let plan = KVCachePlan(
            configuration: KVCacheConfiguration(
                strategy: .affine(affine), compatibility: .allowPartial))
        let simple = populatedSimpleCache(headDimension: 64)
        simple.offset = 1
        let storage = KVCacheStorage([simple], plan: plan)
        let observer = storage

        _ = try plan.applyAndValidate(to: storage)
        #expect(!storage.isApplicationTerminal)

        simple.offset = 2
        plan.apply(to: storage)

        #expect(storage.isApplicationTerminal)
        #expect(storage.cache[0] is QuantizedKVCache)
        #expect(observer.cache[0] is QuantizedKVCache)

        plan.apply(to: storage)
        #expect(storage.isApplicationTerminal)
        #expect(storage.cache[0] is QuantizedKVCache)
    }

    @Test func modelCacheOwnsHybridProgressAcrossPrefillAndDecode() throws {
        let model = HybridProgressModel()
        let storage = KVCacheStorage(model.newCache(parameters: nil), plan: .disabled)
        var iterator = try TokenIterator(
            input: LMInput(tokens: MLXArray([1, 2, 3])),
            model: model,
            cacheStorage: storage,
            parameters: GenerateParameters(maxTokens: 2, temperature: 0))

        #expect(storage.processedTokenCount == 3)
        let recurrent = try #require(storage.cache[0] as? MambaCache)
        let attention = try #require(storage.cache[1] as? KVCacheSimple)
        #expect(recurrent.offset == 0)
        #expect(attention.offset == 3)
        #expect(storage.nativeAttentionOffsetsAreAligned)

        _ = iterator.next()

        #expect(storage.processedTokenCount == 4)
        #expect(recurrent.offset == 0)
        #expect(attention.offset == 4)
        #expect(storage.nativeAttentionOffsetsAreAligned)
    }

    @Test func modelCacheCopyAndTrimPreserveOneTimeline() throws {
        let attention = KVCacheSimple()
        _ = attention.update(
            keys: MLXArray.zeros([1, 1, 8, 4]),
            values: MLXArray.zeros([1, 1, 8, 4]))
        let storage = KVCacheStorage([attention], plan: .disabled)

        #expect(storage.processedTokenCount == 8)
        let snapshot = storage.copy()
        let trimmed = snapshot.trim(3)

        #expect(trimmed == 3)
        #expect(snapshot.processedTokenCount == 5)
        #expect(snapshot.cache[0].offset == 5)
        #expect(storage.processedTokenCount == 8)
        #expect(storage.cache[0].offset == 8)
        let snapshotAttention = try #require(snapshot.cache[0] as? KVCacheSimple)
        #expect(snapshotAttention !== attention)
    }

    @Test func turboRuntimeReportMarksDifferentRequestedStrategy() {
        let configuration = KVCacheConfiguration(strategy: .affine(.fourBit))
        let cache: [KVCache] = [
            TurboQuantKVCache(bits: 4, keyBits: 0, valueBits: 4)
        ]

        let layer = kvCacheRuntimeReport(cache: cache, configuration: configuration).layers[0]
        #expect(layer.state == .skipped)
        #expect(layer.resolvedStrategy == .turboQuant)
        #expect(layer.reason == .differentStrategy)
    }

    @Test func turboBoundaryProtectionUsesStableAttentionPaths() throws {
        let turbo = try TurboQuantKVCacheConfiguration(
            keyPrecision: .twoBit,
            valuePrecision: .fourBit)
        let configuration = KVCacheConfiguration(
            strategy: .turboQuant(turbo),
            compatibility: .allowPartial)
        var cache: [KVCache] = (0 ..< 6).map {
            [2, 5].contains($0) ? populatedSimpleCache(headDimension: 64) : KVCacheSimple()
        }

        _ = try applyKVCacheConfiguration(cache: &cache, configuration: configuration)
        #expect(cache[2] is TurboQuantKVCache)
        #expect(cache[5] is QuantizedKVCache)

        for index in [0, 1, 3, 4] {
            cache[index].state = populatedSimpleCache(headDimension: 64).state
        }
        _ = try applyKVCacheConfiguration(cache: &cache, configuration: configuration)

        for index in [0, 1, 4, 5] { #expect(cache[index] is QuantizedKVCache) }
        for index in [2, 3] { #expect(cache[index] is TurboQuantKVCache) }

        let report = kvCacheRuntimeReport(cache: cache, configuration: configuration)
        #expect(report.layers[0].reason == .boundaryProtection)
        #expect(report.layers[3].reason == .awaitingCompressionStart)
    }

    @Test func turboBoundaryProtectionExcludesRotatingLayersFromRanking() throws {
        let rotating = (0 ..< 4).map { _ in RotatingKVCache(maxSize: 128) }
        let simple = (0 ..< 6).map { _ in populatedSimpleCache(headDimension: 64) }
        var cache: [KVCache] = [
            rotating[0], simple[0], rotating[1], simple[1], simple[2],
            simple[3], rotating[2], simple[4], simple[5], rotating[3],
        ]
        let turbo = try TurboQuantKVCacheConfiguration(
            keyPrecision: .twoBit,
            valuePrecision: .fourBit)
        let configuration = KVCacheConfiguration(
            strategy: .turboQuant(turbo),
            compatibility: .allowPartial)

        _ = try applyKVCacheConfiguration(cache: &cache, configuration: configuration)

        for index in [0, 2, 6, 9] { #expect(cache[index] is RotatingKVCache) }
        for index in [1, 3, 7, 8] { #expect(cache[index] is QuantizedKVCache) }
        for index in [4, 5] { #expect(cache[index] is TurboQuantKVCache) }

        let report = kvCacheRuntimeReport(cache: cache, configuration: configuration)
        for index in [1, 3, 7, 8] {
            #expect(report.layers[index].reason == .boundaryProtection)
        }
        for index in [0, 2, 6, 9] {
            #expect(report.layers[index].reason == .slidingWindow)
        }
    }

    private func mixedShapeCache() -> [KVCache] {
        [populatedSimpleCache(headDimension: 64), populatedSimpleCache(headDimension: 48)]
    }

    private func populatedSimpleCache(headDimension: Int) -> KVCacheSimple {
        let cache = KVCacheSimple()
        cache.state = [
            MLXArray.zeros([1, 1, 1, headDimension]),
            MLXArray.zeros([1, 1, 1, headDimension]),
        ]
        return cache
    }

    private final class HybridProgressModel: Module, LanguageModel {
        func prepare(
            _ input: LMInput,
            cache: [KVCache],
            state: LMOutput.State?,
            windowSize: Int?
        ) throws -> PrepareResult {
            .tokens(input.text)
        }

        func callAsFunction(
            _ input: LMInput.Text,
            cache: [KVCache]?,
            state: LMOutput.State?
        ) -> LMOutput {
            let length = input.cacheSequenceLength
            if let recurrent = cache?[0] as? MambaCache {
                recurrent.advance(length)
            }
            if let attention = cache?[1] as? KVCacheSimple {
                _ = attention.update(
                    keys: MLXArray.zeros([1, 1, length, 4]),
                    values: MLXArray.zeros([1, 1, length, 4]))
            }
            return LMOutput(logits: MLXArray.zeros([1, length, 8]))
        }

        func newCache(parameters: GenerateParameters?) -> [KVCache] {
            [MambaCache(), KVCacheSimple()]
        }
    }
}
