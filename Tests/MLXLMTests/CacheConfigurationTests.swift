// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLLM
import MLXNN
import Testing

@testable import MLXLMCommon

// MARK: - Helpers

/// Minimal model that uses the default ``KVCacheDimensionProvider`` cache path.
private final class DummyModel: Module, LanguageModel, KVCacheDimensionProvider {
    let kvHeads: [Int]

    init(layers: Int = 4) {
        self.kvHeads = Array(repeating: 1, count: layers)
        super.init()
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        inputs
    }

    func prepare(
        _ input: LMInput, cache: [KVCache], state: LMOutput.State?, prefill: PrefillParameters
    ) throws -> PrepareResult {
        .tokens(input.text)
    }
}

// MARK: - Unified status

private func legacyStatus(cache: [KVCache], maxKVSize: Int) throws -> KVCacheStatus {
    let plan = try GenerateParameters(maxKVSize: maxKVSize).kvCachePlan()
    return KVCacheStatus(cache: cache, plan: plan, phase: .planned)
}

@Test func cacheStatusCapacityNotRequested() {
    let status = KVCacheStatus(cache: [KVCacheSimple(), KVCacheSimple()])
    #expect(status.capacityDisposition == .notRequested)
    #expect(status.requestSource == nil)
}

@Test func cacheStatusCapacityFullyApplied() throws {
    let status = try legacyStatus(
        cache: [
            RotatingKVCache(maxSize: 4096, keep: 4),
            RotatingKVCache(maxSize: 4096, keep: 4),
        ],
        maxKVSize: 4096)
    #expect(status.capacityDisposition == .fullyApplied)
    #expect(status.capacityAppliedLayerCount == 2)
    #expect(status.requestSource == .legacy)
}

@Test func cacheStatusHybridCapacityFullyApplied() throws {
    let status = try legacyStatus(
        cache: [
            MambaCache(),
            RotatingKVCache(maxSize: 4096, keep: 4),
            MambaCache(),
            RotatingKVCache(maxSize: 4096, keep: 4),
        ],
        maxKVSize: 4096)
    #expect(status.capacityDisposition == .fullyApplied)
    #expect(status.isHybrid)
}

@Test func cacheStatusCapacityPartiallyApplied() throws {
    let status = try legacyStatus(
        cache: [
            RotatingKVCache(maxSize: 512),
            KVCacheSimple(),
            RotatingKVCache(maxSize: 512),
        ],
        maxKVSize: 4096)
    #expect(status.capacityDisposition == .partiallyApplied)
    #expect(status.capacityAppliedLayerCount == 2)
    #expect(status.capacityUnappliedLayerCount == 1)
}

@Test func cacheStatusTreatsOversizedWindowAsUnapplied() throws {
    let status = try legacyStatus(
        cache: [
            RotatingKVCache(maxSize: 8192),
            RotatingKVCache(maxSize: 4096, keep: 4),
        ],
        maxKVSize: 4096)
    #expect(status.capacityDisposition == .partiallyApplied)
    #expect(status.capacityAppliedLayerCount == 1)
    #expect(status.capacityUnappliedLayerCount == 1)
}

@Test func cacheStatusCapacityUnsupportedForPureSSM() throws {
    let status = try legacyStatus(
        cache: [MambaCache(), MambaCache()], maxKVSize: 4096)
    #expect(status.capacityDisposition == .unsupported)
}

@Test func cacheStatusCapacityIgnored() throws {
    let status = try legacyStatus(
        cache: [KVCacheSimple(), KVCacheSimple()], maxKVSize: 4096)
    #expect(status.capacityDisposition == .ignored)
    #expect(status.capacityUnappliedLayerCount == 2)
}

@Test func cacheLayerStatusClassifiesConcreteCaches() {
    let caches: [KVCache] = [
        KVCacheSimple(),
        RotatingKVCache(maxSize: 128, keep: 4),
        MambaCache(),
        CacheList(MambaCache(), RotatingKVCache(maxSize: 64, keep: 4)),
    ]

    let status = KVCacheStatus(cache: caches)
    #expect(status.layers.count == 5)
    #expect(status.layers[1].kind == .rotatingAttention(maxSize: 128, keep: 4))
    #expect(status.layers[1].capacitySource == .modelDefined)
    #expect(status.layers[2].kind == .stateSpace)
    #expect(status.layers[3].path == [3, 0])
    #expect(status.layers[4].path == [3, 1])
}

// MARK: - Default LanguageModel path

@Test func defaultLanguageModelHonorsMaxKVSize() throws {
    let model = DummyModel(layers: 3)
    var parameters = GenerateParameters()
    parameters.maxKVSize = 256

    let caches = try model.newCache(parameters: parameters)
    #expect(caches.count == 3)
    #expect(caches.allSatisfy { $0 is RotatingKVCache })
    #expect(caches.allSatisfy { $0.maxSize == 256 })

    let status = try model.cacheStatus(parameters: parameters)
    #expect(status.capacityDisposition == .fullyApplied)
    #expect(status.requestSource == .legacy)
    #expect(status.requestedConfiguration?.capacity?.maxTokens == 256)
    #expect(status.attentionMaxSizes == [256, 256, 256])
}

@Test func defaultLanguageModelUnboundedWithoutMaxKVSize() throws {
    let model = DummyModel(layers: 2)
    let caches = try model.newCache(parameters: nil)
    #expect(caches.allSatisfy { $0 is KVCacheSimple })
    #expect(caches.allSatisfy { $0.maxSize == nil })

    let status = try model.cacheStatus(parameters: nil)
    #expect(status.capacityDisposition == .notRequested)
    #expect(status.phase == .planned)
}

// MARK: - Construction helpers

@Test func makeAttentionKVCacheHonorsMaxKVSize() throws {
    var parameters = GenerateParameters()
    parameters.maxKVSize = 64
    let limited = try makeAttentionKVCache(parameters: parameters)
    #expect(limited is RotatingKVCache)
    #expect(limited.maxSize == 64)
    if let rotating = limited as? RotatingKVCache {
        #expect(rotating.keepCount == 4)
    }

    let unbounded = try makeAttentionKVCache(parameters: nil)
    #expect(unbounded is KVCacheSimple)
}

@Test func makeAttentionKVCacheClampsKeepForTinyLimit() throws {
    let parameters = GenerateParameters(maxKVSize: 1)
    let cache = try #require(
        makeAttentionKVCache(parameters: parameters) as? RotatingKVCache)
    #expect(cache.maxSize == 1)
    #expect(cache.keepCount == 0)
}

@Test func cacheConstructionRejectsInvalidLegacyCapacity() {
    let parameters = GenerateParameters(maxKVSize: 0)
    #expect(throws: KVCacheConfigurationError.invalidCapacity(0)) {
        try makeAttentionKVCache(parameters: parameters)
    }
    let model = DummyModel()
    #expect(throws: KVCacheConfigurationError.invalidCapacity(0)) {
        try model.newCache(parameters: parameters)
    }
}

@Test func slidingWindowConstructionRejectsInvalidModelConfiguration() {
    #expect(throws: KVCacheConfigurationError.invalidSlidingWindow(0)) {
        try makeSlidingWindowKVCache(parameters: nil, window: 0)
    }
    #expect(throws: KVCacheConfigurationError.invalidSlidingWindow(-1)) {
        try slidingWindowCacheKind(parameters: nil, window: -1)
    }
}

@Test func makeHybridAttentionKVCachePrefersSlidingWindow() throws {
    var parameters = GenerateParameters()
    parameters.maxKVSize = 4096

    let sliding = try makeHybridAttentionKVCache(
        parameters: parameters,
        slidingWindow: 512,
        usesSlidingWindow: true
    )
    #expect(sliding is RotatingKVCache)
    #expect(sliding.maxSize == 512)

    let cappedSliding = try makeHybridAttentionKVCache(
        parameters: parameters,
        slidingWindow: 8192,
        usesSlidingWindow: true
    )
    #expect(cappedSliding is RotatingKVCache)
    #expect(cappedSliding.maxSize == 4096)

    let full = try makeHybridAttentionKVCache(
        parameters: parameters,
        slidingWindow: 512,
        usesSlidingWindow: false
    )
    #expect(full is RotatingKVCache)
    #expect(full.maxSize == 4096)
}

@Test func typedCapacityPreservesModelNativeSlidingWindow() throws {
    let capacity = try KVCacheConfiguration.Capacity(
        maxTokens: 64, preservedPrefixTokens: 3)
    let parameters = GenerateParameters(
        kvCache: KVCacheConfiguration(capacity: capacity))

    let sliding = try #require(
        makeHybridAttentionKVCache(
            parameters: parameters,
            slidingWindow: 512,
            usesSlidingWindow: true
        ) as? RotatingKVCache)
    #expect(sliding.maxSize == 512)
    #expect(sliding.keepCount == 0)
    #expect(sliding.capacityOrigin == .modelNative)

    let full = try #require(
        makeHybridAttentionKVCache(
            parameters: parameters,
            slidingWindow: 512,
            usesSlidingWindow: false
        ) as? RotatingKVCache)
    #expect(full.maxSize == 64)
    #expect(full.keepCount == 3)
    #expect(full.capacityOrigin == .requested)

    let status = KVCacheStatus(
        cache: [sliding, full],
        configuration: parameters.kvCache)
    #expect(status.requestSource == .typed)
    #expect(status.requestedConfiguration == parameters.kvCache)
    #expect(status.capacityDisposition == .fullyApplied)
    #expect(status.capacityAppliedLayerCount == 1)
    #expect(status.attentionMaxSizes == [512, 64])
    #expect(status.layers.map(\.capacitySource) == [.modelDefined, .requested])
}

@Test func typedCapacityRequiresAnExactCallerConfiguredCache() throws {
    let capacity = try KVCacheConfiguration.Capacity(
        maxTokens: 64, preservedPrefixTokens: 3)
    let requested = RotatingKVCache(maxSize: 32, keep: 3)
    requested.capacityOrigin = .requested

    let status = KVCacheStatus(
        cache: [requested],
        configuration: KVCacheConfiguration(capacity: capacity))

    #expect(status.capacityDisposition == .ignored)
    #expect(status.capacityAppliedLayerCount == 0)
    #expect(status.capacityUnappliedLayerCount == 1)
}

@Test func typedCapacityIsUnsupportedForOnlyModelDefinedWindows() throws {
    let capacity = try KVCacheConfiguration.Capacity(maxTokens: 64)
    let status = KVCacheStatus(
        cache: [RotatingKVCache(maxSize: 512)],
        configuration: KVCacheConfiguration(capacity: capacity))

    #expect(status.capacityDisposition == .unsupported)
    #expect(status.capacityAppliedLayerCount == 0)
    #expect(status.capacityUnappliedLayerCount == 0)
}

@Test func promptCacheWithLayerCountClampsTinyLegacyLimit() throws {
    let caches = try makePromptCacheWithLayerCount(numLayers: 2, maxKVSize: 1)
    #expect(caches.count == 2)
    let keys = MLXArray.zeros([1, 1, 1, 8])
    let values = MLXArray.zeros([1, 1, 1, 8])
    for cache in caches {
        let rotating = try #require(cache as? RotatingKVCache)
        #expect(rotating.maxSize == 1)
        #expect(rotating.keepCount == 0)
        #expect(rotating.capacityOrigin == .requested)
        _ = rotating.update(keys: keys, values: values)
        _ = rotating.update(keys: keys, values: values)
        #expect(rotating.offset == 2)
    }
}

// MARK: - Model-specific overrides

@Test func qwen35TextModelHonorsMaxKVSizeOnAttentionLayers() throws {
    // Tiny hybrid: full attention every 4th layer (interval 4).
    let json = """
        {
          "model_type": "qwen3_5",
          "hidden_size": 64,
          "num_hidden_layers": 4,
          "intermediate_size": 128,
          "num_attention_heads": 4,
          "num_key_value_heads": 2,
          "head_dim": 16,
          "linear_num_value_heads": 4,
          "linear_num_key_heads": 2,
          "linear_key_head_dim": 16,
          "linear_value_head_dim": 16,
          "linear_conv_kernel_dim": 4,
          "rms_norm_eps": 1e-6,
          "vocab_size": 128,
          "full_attention_interval": 4,
          "tie_word_embeddings": true
        }
        """.data(using: .utf8)!
    let textConfig = try JSONDecoder().decode(Qwen35TextConfiguration.self, from: json)
    let model = Qwen35TextModel(textConfig)

    var parameters = GenerateParameters()
    parameters.maxKVSize = 128

    let caches = try model.newCache(parameters: parameters)
    #expect(caches.count == 4)

    // With interval 4: layers 0,1,2 linear; layer 3 full attention.
    var attentionCount = 0
    var stateSpaceCount = 0
    for cache in caches {
        if cache is MambaCache {
            stateSpaceCount += 1
            #expect(cache.maxSize == nil)
        } else {
            attentionCount += 1
            #expect(cache is RotatingKVCache)
            #expect(cache.maxSize == 128)
        }
    }
    #expect(attentionCount >= 1)
    #expect(stateSpaceCount >= 1)

    let status = try model.cacheStatus(parameters: parameters)
    #expect(status.capacityDisposition == .fullyApplied)
    #expect(status.isHybrid)

    // Without a limit, attention layers stay unbounded.
    let unbounded = try model.newCache(parameters: nil)
    for cache in unbounded where !(cache is MambaCache) {
        #expect(cache is KVCacheSimple)
        #expect(cache.maxSize == nil)
    }
}

@Test func falconH1CacheConfigurationReportsHybridFullyApplied() throws {
    let json = """
        {
            "model_type": "falcon_h1",
            "hidden_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "vocab_size": 100,
            "mamba_d_ssm": 16,
            "mamba_d_state": 8,
            "mamba_d_head": 8,
            "mamba_n_heads": 2,
            "mamba_n_groups": 1,
            "mamba_d_conv": 4,
            "mamba_chunk_size": 64,
            "attention_in_multiplier": 2.0,
            "key_multiplier": 0.5,
            "num_logits_to_keep": 1
        }
        """.data(using: .utf8)!
    let config = try JSONDecoder.json5().decode(FalconH1Configuration.self, from: json)
    let model = FalconH1Model(config)

    var parameters = GenerateParameters()
    parameters.maxKVSize = 8

    let caches = try model.newCache(parameters: parameters)
    for entry in caches {
        let list = try #require(entry as? CacheList)
        #expect(list.children[0] is MambaCache)
        #expect(list.children[1] is RotatingKVCache)
        #expect(list.children[1].maxSize == 8)
    }

    let status = try model.cacheStatus(parameters: parameters)
    #expect(status.capacityDisposition == .fullyApplied)
    #expect(status.isHybrid)
}

@Test func pureMambaReportsUnsupportedMaxKVSize() throws {
    let json = """
        {
          "model_type": "mamba2",
          "vocab_size": 64,
          "hidden_size": 32,
          "num_hidden_layers": 2,
          "state_size": 8,
          "ssm_state_size": 8,
          "conv_kernel": 4,
          "n_groups": 1,
          "head_dim": 8,
          "num_heads": 4,
          "use_bias": false,
          "use_conv_bias": true,
          "tie_word_embeddings": true
        }
        """.data(using: .utf8)!
    let config = try JSONDecoder().decode(Mamba2Configuration.self, from: json)
    let model = Mamba2Model(config)

    var parameters = GenerateParameters()
    parameters.maxKVSize = 1024

    let caches = model.newCache(parameters: parameters)
    #expect(caches.allSatisfy { $0 is MambaCache })

    let status = try model.cacheStatus(parameters: parameters)
    #expect(status.capacityDisposition == .unsupported)
}

@Test func gemma3TextAllAttentionLayersHonorMaxKVSize() throws {
    // 6 layers, pattern 3 → global at indices 2 and 5.
    let config = Gemma3TextConfiguration(
        modelType: "gemma3_text",
        hiddenSize: 64,
        hiddenLayers: 6,
        intermediateSize: 128,
        attentionHeads: 4,
        headDim: 16,
        rmsNormEps: 1e-6,
        vocabularySize: 128,
        kvHeads: 1,
        ropeTheta: 1_000_000,
        ropeLocalBaseFreq: 10_000,
        ropeTraditional: false,
        queryPreAttnScalar: 256,
        slidingWindow: 32,
        slidingWindowPattern: 3,
        maxPositionEmbeddings: 128
    )
    let model = Gemma3TextModel(config)

    var parameters = GenerateParameters()
    parameters.maxKVSize = 16

    let caches = try model.newCache(parameters: parameters)
    #expect(caches.count == 6)

    for (i, cache) in caches.enumerated() {
        let isGlobal = (i % 3 == 2)
        #expect(cache is RotatingKVCache)
        if isGlobal {
            #expect(cache.maxSize == 16)
        } else {
            #expect(cache.maxSize == 16)
        }
    }

    let status = try model.cacheStatus(parameters: parameters)
    #expect(status.capacityDisposition == .fullyApplied)
}
