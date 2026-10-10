// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

@Suite(.serialized)
struct Gemma3nSharedKVTests {
    @Test(arguments: [[12], [2, 3, 1, 6], Array(repeating: 1, count: 12)])
    func chunkingAndRotationMatchUncachedAttention(chunks: [Int]) throws {
        try withRandomState(MLXRandom.RandomState(seed: 42)) {
            let model = Gemma3nTextModel(config: try configuration())
            #expect(model.languageModel.layerIdxToCacheIdx == [0, 1, 2, 3, 2, 3])
            eval(model)
            let tokens = MLXArray(Array(1 ... 12)).reshaped(1, 12)
            let expected = model(tokens, cache: nil)
            eval(expected)

            for reserve in [0, 3] {
                let cache = try caches(model, reserve: reserve)
                let output = forward(model, tokens: tokens, chunks: chunks, cache: cache)
                #expect(allClose(output, expected, rtol: 2e-4, atol: 2e-5).item(Bool.self))
                #expect(cache.count == 4)
                #expect(cache.allSatisfy { $0.offset == 12 })
            }
        }
    }

    @Test func rewindMatchesFreshPrefill() throws {
        try withRandomState(MLXRandom.RandomState(seed: 43)) {
            let model = Gemma3nTextModel(config: try configuration())
            eval(model)
            let cache = try caches(model, reserve: 3)
            let prompt = MLXArray(Array(1 ... 12)).reshaped(1, 12)
            eval(
                forward(model, tokens: prompt, chunks: Array(repeating: 1, count: 12), cache: cache)
            )
            #expect(trimPromptCache(cache, numTokens: 2) == 2)

            let continuation = MLXArray([13, 14, 15]).reshaped(1, 3)
            let output = model(continuation, cache: cache)
            let fresh = model(
                MLXArray(Array(1 ... 10) + [13, 14, 15]).reshaped(1, 13), cache: nil)
            let expected = fresh[0..., 10..., 0...]
            eval(output, expected)
            #expect(allClose(output, expected, rtol: 2e-4, atol: 2e-5).item(Bool.self))
            #expect(cache.allSatisfy { $0.offset == 13 })
        }
    }

    @Test func stagedVerificationAndPartialCommitMatchOrdinaryDecoding() throws {
        try withRandomState(MLXRandom.RandomState(seed: 44)) {
            let model = Gemma3nTextModel(config: try configuration())
            eval(model)
            let cache = try caches(model, reserve: 3)
            let prompt = MLXArray(Array(1 ... 12)).reshaped(1, 12)
            eval(
                forward(model, tokens: prompt, chunks: Array(repeating: 1, count: 12), cache: cache)
            )
            let referenceCache = cache.map { $0.copy() }
            let storage = KVCacheStorage(cache, plan: .disabled)
            let round = try #require(storage.beginRound(maximumPositions: 3))
            let tokens = MLXArray([13, 14, 15]).reshaped(1, 3)
            let output = model(tokens, cache: round.caches)
            let expected = model(tokens, cache: referenceCache)
            eval(output, expected)
            #expect(allClose(output, expected, rtol: 2e-4, atol: 2e-5).item(Bool.self))
            #expect(cache[2].offset == 12)
            storage.commit(round, retaining: 2)
            #expect(cache.allSatisfy { $0.offset == 14 })

            let next = model(MLXArray([16]).reshaped(1, 1), cache: cache)
            let freshLogits = model(
                MLXArray(Array(1 ... 12) + [13, 14, 16]).reshaped(1, 15), cache: nil)
            let fresh = freshLogits[0..., 14..., 0...]
            eval(next, fresh)
            #expect(allClose(next, fresh, rtol: 2e-4, atol: 2e-5).item(Bool.self))
        }
    }

    @Test(arguments: [1, 3])
    func sharedAttentionUsesOriginalPositionsAndNeverReadsSerialization(count: Int) throws {
        try withRandomState(MLXRandom.RandomState(seed: 45)) {
            let cache = SerializationObservingCache()
            try checkSharedAttention(cache: cache, count: count)
            #expect(cache.stateReads == 0)
        }
    }

    @Test(arguments: [1, 3])
    func quantizedAttentionMatchesDequantizedReference(count: Int) throws {
        try withRandomState(MLXRandom.RandomState(seed: 46)) {
            let config = try configuration()
            let owner = Gemma3nAttention(config, layerIdx: 3)
            let shared = Gemma3nAttention(config, layerIdx: 5)
            let cache = QuantizedKVCache(groupSize: 32, bits: 8)
            let x = MLXRandom.normal([1, count, config.hiddenSize])
            let (_, kv) = owner.forward(x, mask: .causal, cache: cache)
            let output = shared.forward(x, mask: .causal, sharedKV: kv).0
            let (keys, values) = try #require(cache.getQuantizedState())
            let k = dequantized(
                keys.0, scales: keys.1, biases: keys.2, groupSize: cache.groupSize,
                bits: cache.bits, mode: cache.mode)
            let v = dequantized(
                values.0, scales: values.1, biases: values.2, groupSize: cache.groupSize,
                bits: cache.bits, mode: cache.mode)
            let q = queries(shared, x: x, offset: .scalar(0))
            let expected = shared.oProj(
                MLXFast.scaledDotProductAttention(
                    queries: q, keys: k, values: v, scale: 1, mask: .causal
                )
                .transposed(0, 2, 1, 3).reshaped(1, count, -1))
            eval(output, expected)
            #expect(allClose(output, expected, rtol: 2e-4, atol: 2e-5).item(Bool.self))
            #expect(cache.offset == count)
        }
    }

    @Test func modelSupportsDynamicKVQuantization() throws {
        let model = Gemma3nTextModel(config: try configuration())
        eval(model)
        var cache = try model.newCache(parameters: nil)
        eval(model(MLXArray([1, 2, 3]).reshaped(1, 3), cache: cache))
        maybeQuantizeKVCache(cache: &cache, kvBits: 4, kvGroupSize: 32, quantizedKVStart: 0)
        #expect(cache[3] is QuantizedKVCache)
        let output = model(MLXArray([4, 5]).reshaped(1, 2), cache: cache)
        eval(output)
        #expect(output.shape == [1, 2, 32])
        #expect(all(isFinite(output)).item(Bool.self))
        #expect(cache.allSatisfy { $0.offset == 5 })
    }

    @Test(arguments: [1, 3], [0, 35])
    func nativeCompressedAttentionUpdatesOnlyTheOwner(count: Int, initialCount: Int) throws {
        try withRandomState(MLXRandom.RandomState(seed: 47)) {
            let factories: [() -> KVCache] = [
                {
                    VarianceNormalizedKVCache(
                        tileSize: 32, keyBits: 4, valueBits: 4, sinkhornIterations: 2)
                },
                { TurboQuantKVCache(bits: 4) },
                { TurboQuantKVCache(bits: 4, keyBits: 0) },
                { TurboQuantKVCache(bits: 4, keyBits: 8, keyGroupSize: 32) },
            ]
            for makeCache in factories {
                let cache = makeCache()
                try checkSharedAttention(
                    cache: cache, count: count,
                    initialCount: initialCount, referenceCache: makeCache())
            }
        }
    }

    @Test func batchPositionsAreCapturedBeforeTheOwnerAdvances() throws {
        try withRandomState(MLXRandom.RandomState(seed: 48)) {
            let config = try configuration()
            let owner = Gemma3nAttention(config, layerIdx: 3)
            let shared = Gemma3nAttention(config, layerIdx: 5)
            let cache = PositionedCache()
            let x = MLXRandom.normal([2, 3, config.hiddenSize])
            let originalOffset = RoPEOffset.batch(MLXArray([5, 9]))
            let (k, v) = projections(owner, x: x, offset: originalOffset)
            let q = queries(shared, x: x, offset: originalOffset)
            let expected = shared.oProj(
                MLXFast.scaledDotProductAttention(
                    queries: q, keys: k, values: v, scale: 1, mask: .causal
                )
                .transposed(0, 2, 1, 3).reshaped(2, 3, -1))
            let (_, kv) = owner.forward(x, mask: .causal, cache: cache)
            let output = shared.forward(x, mask: .causal, sharedKV: kv).0
            eval(output, expected)
            #expect(allClose(output, expected, rtol: 2e-4, atol: 2e-5).item(Bool.self))
            #expect(cache.batchOffset.asArray(Int.self) == [8, 12])
        }
    }

    @Test(arguments: ["full_attention", "global_attention", "sliding_attention"])
    func singleAttentionTypeSupportsSharing(layerType: String) throws {
        let config = try configuration(layerTypes: Array(repeating: layerType, count: 6))
        let model = Gemma3nTextModel(config: config)
        eval(model)
        let cache = try model.newCache(parameters: nil)
        let tokens = MLXArray([1, 2, 3]).reshaped(1, 3)
        let expected = model(tokens, cache: nil)
        let output = model(tokens, cache: cache)
        eval(expected, output)
        #expect(allClose(output, expected, rtol: 2e-4, atol: 2e-5).item(Bool.self))
        #expect(model.languageModel.layerIdxToCacheIdx == [0, 1, 2, 3, 3, 3])
    }

    @Test func sharedLayersLoadModernAndLegacyCheckpointsWithoutUnusedProjections() throws {
        let model = Gemma3nTextModel(config: try configuration())
        let weights = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
        for index in [4, 5] {
            #expect(
                !weights.keys.contains {
                    $0.hasPrefix("language_model.layers.\(index).self_attn.k_proj.")
                })
            #expect(
                !weights.keys.contains {
                    $0.hasPrefix("language_model.layers.\(index).self_attn.v_proj.")
                })
        }
        var legacy = weights
        legacy["language_model.layers.4.self_attn.k_proj.weight"] = MLXArray.zeros([32, 32])
        legacy["language_model.layers.5.self_attn.k_norm.weight"] = MLXArray.ones([32])
        let settings = BaseConfiguration.PerLayerQuantization(
            quantization: nil,
            perLayerQuantization: [
                "language_model.layers.4.self_attn.k_proj": .quantize(.init(groupSize: 32, bits: 4))
            ])
        let prepared = try model.prepareCheckpoint(
            .init(weights: legacy, perLayerQuantization: settings))
        #expect(prepared.weights.count == weights.count)
        #expect(prepared.perLayerQuantization?.perLayerQuantization.isEmpty == true)
        for checkpoint in [weights, prepared.weights] {
            let loaded = Gemma3nTextModel(config: try configuration())
            try loaded.update(parameters: ModuleParameters.unflattened(checkpoint), verify: [.all])
        }
    }

    private func checkSharedAttention(
        cache: KVCache, count: Int, initialCount: Int = 35,
        referenceCache suppliedCache: KVCache? = nil
    ) throws {
        let config = try configuration()
        let owner = Gemma3nAttention(config, layerIdx: 3)
        let shared = Gemma3nAttention(config, layerIdx: 5)
        if initialCount > 0 {
            let initial = MLXRandom.normal([1, initialCount, config.hiddenSize])
            eval(owner.forward(initial, mask: .causal, cache: cache).0)
            let token = MLXRandom.normal([1, 1, config.hiddenSize])
            eval(owner.forward(token, cache: cache).0)
            if let suppliedCache {
                eval(owner.forward(initial, mask: .causal, cache: suppliedCache).0)
                eval(owner.forward(token, cache: suppliedCache).0)
            }
        }
        let prefixCount = cache.offset
        if let observing = cache as? SerializationObservingCache {
            #expect(observing.stateReads == 0)
        }
        let referenceCache = suppliedCache ?? cache.copy()
        (cache as? SerializationObservingCache)?.stateReads = 0
        let offset = cache.ropeOffset
        let x = MLXRandom.normal([1, count, config.hiddenSize])
        let (k, v) = projections(owner, x: x, offset: offset)
        let q = queries(shared, x: x, offset: offset)
        let expected = shared.oProj(
            attentionWithCacheUpdate(
                queries: q, keys: k, values: v, cache: referenceCache,
                scale: 1, mask: .causal
            )
            .transposed(0, 2, 1, 3).reshaped(1, count, -1))
        let (_, kv) = owner.forward(x, mask: .causal, cache: cache)
        let output = shared.forward(x, mask: .causal, sharedKV: kv).0
        let repeated = shared.forward(x, mask: .causal, sharedKV: kv).0
        eval(expected, output, repeated)
        #expect(allClose(output, expected, rtol: 2e-4, atol: 2e-5).item(Bool.self))
        #expect(arrayEqual(output, repeated).item(Bool.self))
        #expect(cache.offset == prefixCount + count)
    }

    private func queries(_ attention: Gemma3nAttention, x: MLXArray, offset: RoPEOffset?)
        -> MLXArray
    {
        let q = attention.qNorm(
            attention.qProj(x).reshaped(x.dim(0), x.dim(1), -1, attention.headDim)
        )
        .transposed(0, 2, 1, 3)
        return applyRotaryPosition(attention.rope, to: q, offset: offset)
    }

    private func projections(_ attention: Gemma3nAttention, x: MLXArray, offset: RoPEOffset?) -> (
        MLXArray, MLXArray
    ) {
        let k = attention.kNorm!(
            attention.kProj!(x).reshaped(x.dim(0), x.dim(1), -1, attention.headDim)
        )
        .transposed(0, 2, 1, 3)
        let v = attention.vNorm!(
            attention.vProj!(x).reshaped(x.dim(0), x.dim(1), -1, attention.headDim)
        )
        .transposed(0, 2, 1, 3)
        return (applyRotaryPosition(attention.rope, to: k, offset: offset), v)
    }

    private func caches(_ model: Gemma3nTextModel, reserve: Int) throws -> [KVCache] {
        if reserve == 0 { return try model.newCache(parameters: nil) }
        return try model.newCache(
            parameters: GenerateParameters(
                kvCache: KVCacheConfiguration(rewind: try .init(maxTokens: reserve))))
    }

    private func forward(
        _ model: Gemma3nTextModel, tokens: MLXArray, chunks: [Int], cache: [KVCache]
    ) -> MLXArray {
        var position = 0
        return concatenated(
            chunks.map { count in
                defer { position += count }
                let output = model(tokens[0..., position ..< (position + count)], cache: cache)
                eval(output)
                return output
            }, axis: 1)
    }

    private func configuration(layerTypes: [String]? = nil) throws -> Gemma3nTextConfiguration {
        var values = try #require(
            JSONSerialization.jsonObject(
                with: JSONEncoder().encode(Gemma3nTextConfiguration())) as? [String: Any])
        values.merge([
            "hidden_size": 32, "num_hidden_layers": 6,
            "intermediate_size": Array(repeating: 64, count: 6),
            "num_attention_heads": 2, "head_dim": 32, "vocab_size": 32,
            "num_key_value_heads": 1, "num_kv_shared_layers": 2, "vocab_size_per_layer_input": 32,
            "hidden_size_per_layer_input": 16, "altup_num_inputs": 2, "laurel_rank": 8,
            "sliding_window": 4,
            "layer_types": layerTypes ?? [
                "sliding_attention", "full_attention", "sliding_attention", "full_attention",
                "sliding_attention", "full_attention",
            ],
            "activation_sparsity_pattern": Array(repeating: 0, count: 6),
        ]) { _, new in new }
        return try JSONDecoder().decode(
            Gemma3nTextConfiguration.self, from: JSONSerialization.data(withJSONObject: values))
    }
}

private final class SerializationObservingCache: KVCacheSimple {
    var stateReads = 0
    override var state: [MLXArray] {
        get {
            stateReads += 1
            return super.state
        }
        set { super.state = newValue }
    }
}

private final class PositionedCache: KVCacheSimple, BatchPositionedKVCache {
    var batchOffset = MLXArray([5, 9])
    override var ropeOffset: RoPEOffset { .batch(batchOffset + 0) }
    override func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        let result = super.update(keys: keys, values: values)
        batchOffset += MLXArray(keys.dim(2))
        return result
    }
}
