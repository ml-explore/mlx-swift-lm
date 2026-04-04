//
//  Gemma4Text.swift
//  mlx-swift-lm
//
//  Created by Check Engine Chat contributors.
//

// Based on https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/models/gemma4_text.py

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Configuration

public struct Gemma4TextConfiguration: Codable {
    let modelType: String
    let hiddenSize: Int
    let hiddenLayers: Int
    let intermediateSize: Int
    let attentionHeads: Int
    let kvHeads: Int
    let headDim: Int
    let globalHeadDim: Int
    let vocabularySize: Int
    let rmsNormEps: Float
    let slidingWindow: Int
    let layerTypes: [String]
    let finalLogitSoftcapping: Float?
    let hiddenSizePerLayerInput: Int
    let vocabSizePerLayerInput: Int
    let numKVSharedLayers: Int
    let tieWordEmbeddings: Bool
    let useDoubleWideMlp: Bool
    let ropeParameters: [String: RopeLayerParams]

    struct RopeLayerParams: Codable {
        let ropeTheta: Float?
        let partialRotaryFactor: Float?

        enum CodingKeys: String, CodingKey {
            case ropeTheta = "rope_theta"
            case partialRotaryFactor = "partial_rotary_factor"
        }

        init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta)
            partialRotaryFactor = try c.decodeIfPresent(Float.self, forKey: .partialRotaryFactor)
        }
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case globalHeadDim = "global_head_dim"
        case vocabularySize = "vocab_size"
        case rmsNormEps = "rms_norm_eps"
        case slidingWindow = "sliding_window"
        case layerTypes = "layer_types"
        case finalLogitSoftcapping = "final_logit_softcapping"
        case hiddenSizePerLayerInput = "hidden_size_per_layer_input"
        case vocabSizePerLayerInput = "vocab_size_per_layer_input"
        case numKVSharedLayers = "num_kv_shared_layers"
        case tieWordEmbeddings = "tie_word_embeddings"
        case useDoubleWideMlp = "use_double_wide_mlp"
        case ropeParameters = "rope_parameters"
    }

    enum VLMCodingKeys: String, CodingKey {
        case textConfig = "text_config"
    }

    public init(from decoder: Decoder) throws {
        let nestedContainer = try decoder.container(keyedBy: VLMCodingKeys.self)
        let container =
            if nestedContainer.contains(.textConfig) {
                try nestedContainer.nestedContainer(keyedBy: CodingKeys.self, forKey: .textConfig)
            } else {
                try decoder.container(keyedBy: CodingKeys.self)
            }

        modelType = try container.decode(String.self, forKey: .modelType)
        hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        hiddenLayers = try container.decode(Int.self, forKey: .hiddenLayers)
        intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)
        attentionHeads = try container.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 8
        kvHeads = try container.decodeIfPresent(Int.self, forKey: .kvHeads) ?? 2
        headDim = try container.decodeIfPresent(Int.self, forKey: .headDim) ?? 256
        globalHeadDim = try container.decodeIfPresent(Int.self, forKey: .globalHeadDim) ?? 512
        vocabularySize = try container.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 262144
        rmsNormEps = try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        slidingWindow = try container.decodeIfPresent(Int.self, forKey: .slidingWindow) ?? 512
        layerTypes = try container.decode([String].self, forKey: .layerTypes)
        finalLogitSoftcapping =
            try container.decodeIfPresent(Float.self, forKey: .finalLogitSoftcapping)
        hiddenSizePerLayerInput =
            try container.decodeIfPresent(Int.self, forKey: .hiddenSizePerLayerInput) ?? 0
        vocabSizePerLayerInput =
            try container.decodeIfPresent(Int.self, forKey: .vocabSizePerLayerInput) ?? 262144
        numKVSharedLayers =
            try container.decodeIfPresent(Int.self, forKey: .numKVSharedLayers) ?? 0
        tieWordEmbeddings =
            try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? true
        useDoubleWideMlp =
            try container.decodeIfPresent(Bool.self, forKey: .useDoubleWideMlp) ?? false
        ropeParameters =
            try container.decodeIfPresent([String: RopeLayerParams].self, forKey: .ropeParameters)
            ?? [:]
    }

    var firstKVSharedLayerIdx: Int { hiddenLayers - numKVSharedLayers }

    func effectiveHeadDim(layerType: String) -> Int {
        layerType == "full_attention" ? globalHeadDim : headDim
    }
}

// MARK: - Attention

private class Attention: Module {
    let nHeads: Int
    let nKVHeads: Int
    let headDim: Int
    let layerIdx: Int
    let isSliding: Bool
    let isKVSharedLayer: Bool
    let kvSharedLayerIndex: Int?
    let storeFullLengthKV: Bool
    let scale: Float

    @ModuleInfo(key: "q_proj") var queryProj: Linear
    @ModuleInfo(key: "k_proj") var keyProj: Linear
    @ModuleInfo(key: "v_proj") var valueProj: Linear
    @ModuleInfo(key: "o_proj") var outputProj: Linear
    @ModuleInfo(key: "q_norm") var queryNorm: Gemma.RMSNorm
    @ModuleInfo(key: "k_norm") var keyNorm: Gemma.RMSNorm
    @ModuleInfo var rope: RoPE

    var lastKV: (MLXArray, MLXArray)?

    init(_ config: Gemma4TextConfiguration, layerIdx: Int) {
        let dim = config.hiddenSize
        let layerType = config.layerTypes[layerIdx]
        self.layerIdx = layerIdx
        self.isSliding = layerType == "sliding_attention"
        self.nHeads = config.attentionHeads
        self.nKVHeads = config.kvHeads
        self.headDim = config.effectiveHeadDim(layerType: layerType)
        self.scale = 1.0

        self._queryProj.wrappedValue = Linear(dim, nHeads * headDim, bias: false)
        self._keyProj.wrappedValue = Linear(dim, nKVHeads * headDim, bias: false)
        self._valueProj.wrappedValue = Linear(dim, nKVHeads * headDim, bias: false)
        self._outputProj.wrappedValue = Linear(nHeads * headDim, dim, bias: false)

        self._queryNorm.wrappedValue = Gemma.RMSNorm(
            dimensions: headDim, eps: config.rmsNormEps)
        self._keyNorm.wrappedValue = Gemma.RMSNorm(
            dimensions: headDim, eps: config.rmsNormEps)

        // RoPE with per-layer-type parameters and partial rotation
        let ropeKey = isSliding ? "sliding_attention" : "full_attention"
        let ropeParams = config.ropeParameters[ropeKey]
        let ropeTheta = ropeParams?.ropeTheta ?? 10_000.0
        let partialFactor = ropeParams?.partialRotaryFactor ?? 1.0
        let ropeDims = Int(Float(headDim) * partialFactor)

        self._rope.wrappedValue = RoPE(
            dimensions: ropeDims, traditional: false, base: ropeTheta)

        // KV sharing: layers beyond firstKVSharedLayerIdx share KV from earlier layers
        let firstShared = config.firstKVSharedLayerIdx
        self.isKVSharedLayer = layerIdx >= firstShared && firstShared > 0

        if isKVSharedLayer {
            let prevTypes = Array(config.layerTypes[0 ..< firstShared])
            let currentType = config.layerTypes[layerIdx]
            self.kvSharedLayerIndex = prevTypes.lastIndex(of: currentType)
            self.storeFullLengthKV = false
        } else {
            self.kvSharedLayerIndex = nil
            let prevTypes = Array(config.layerTypes[0 ..< firstShared])
            let currentType = config.layerTypes[layerIdx]
            self.storeFullLengthKV = layerIdx == prevTypes.lastIndex(of: currentType)
        }

        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache? = nil,
        sharedKV: (MLXArray, MLXArray)? = nil
    ) -> MLXArray {
        let (B, L, _) = (x.dim(0), x.dim(1), x.dim(2))

        var queries = queryProj(x).reshaped(B, L, nHeads, -1).transposed(0, 2, 1, 3)
        queries = queryNorm(queries)

        let keys: MLXArray
        let values: MLXArray

        if isKVSharedLayer, let shared = sharedKV {
            keys = shared.0
            values = shared.1
        } else {
            var k = keyProj(x).reshaped(B, L, nKVHeads, -1).transposed(0, 2, 1, 3)
            var v = valueProj(x).reshaped(B, L, nKVHeads, -1).transposed(0, 2, 1, 3)

            k = keyNorm(k)
            // Note: Python model applies RMSNormNoScale to values, but for quantized
            // models the v_norm weights are not present. Skip for compatibility.

            if let cache {
                k = rope(k, offset: cache.offset)
                let (updatedK, updatedV) = cache.update(keys: k, values: v)
                keys = updatedK
                values = updatedV
            } else {
                keys = rope(k)
                values = v
            }
        }

        if storeFullLengthKV {
            lastKV = (keys, values)
        }

        if let cache {
            queries = rope(queries, offset: cache.offset)
        } else {
            queries = rope(queries)
        }

        // Adjust mask for sliding window cache size
        var finalMask = mask
        if case .array(let maskArray) = mask {
            let keySeqLen = keys.shape[2]
            if maskArray.shape.last! != keySeqLen {
                let slicedMask = maskArray[.ellipsis, (-keySeqLen)...]
                finalMask = .array(slicedMask)
            }
        }

        let output = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: cache,
            scale: scale,
            mask: finalMask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        return outputProj(output)
    }
}

// MARK: - MLP

private class MLP: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear

    init(_ config: Gemma4TextConfiguration, layerIdx: Int) {
        let firstShared = config.firstKVSharedLayerIdx
        let isShared = layerIdx >= firstShared && firstShared > 0
        let useDoubleWide = config.useDoubleWideMlp && isShared
        let iSize = config.intermediateSize * (useDoubleWide ? 2 : 1)

        self._gateProj.wrappedValue = Linear(config.hiddenSize, iSize, bias: false)
        self._downProj.wrappedValue = Linear(iSize, config.hiddenSize, bias: false)
        self._upProj.wrappedValue = Linear(config.hiddenSize, iSize, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(geluApproximate(gateProj(x)) * upProj(x))
    }
}

// MARK: - Decoder Layer

private class TransformerBlock: Module {
    @ModuleInfo(key: "self_attn") var selfAttention: Attention
    @ModuleInfo var mlp: MLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: Gemma.RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: Gemma.RMSNorm
    @ModuleInfo(key: "pre_feedforward_layernorm") var preFeedforwardLayerNorm: Gemma.RMSNorm
    @ModuleInfo(key: "post_feedforward_layernorm") var postFeedforwardLayerNorm: Gemma.RMSNorm

    // Per-layer input gating (multimodal injection, bypassed for text-only)
    @ModuleInfo(key: "per_layer_input_gate") var perLayerInputGate: Linear?
    @ModuleInfo(key: "per_layer_projection") var perLayerProjection: Linear?
    @ModuleInfo(key: "post_per_layer_input_norm") var postPerLayerInputNorm: Gemma.RMSNorm?

    let layerType: String
    let hasPerLayerGate: Bool

    // layer_scalar is loaded as a parameter
    @ModuleInfo(key: "layer_scalar") var layerScalar: MLXArray

    init(_ config: Gemma4TextConfiguration, layerIdx: Int) {
        self.layerType = config.layerTypes[layerIdx]
        self.hasPerLayerGate = config.hiddenSizePerLayerInput > 0

        self._selfAttention.wrappedValue = Attention(config, layerIdx: layerIdx)
        self._mlp.wrappedValue = MLP(config, layerIdx: layerIdx)

        self._inputLayerNorm.wrappedValue = Gemma.RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = Gemma.RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._preFeedforwardLayerNorm.wrappedValue = Gemma.RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postFeedforwardLayerNorm.wrappedValue = Gemma.RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)

        if hasPerLayerGate {
            self._perLayerInputGate.wrappedValue = Linear(
                config.hiddenSize, config.hiddenSizePerLayerInput, bias: false)
            self._perLayerProjection.wrappedValue = Linear(
                config.hiddenSizePerLayerInput, config.hiddenSize, bias: false)
            self._postPerLayerInputNorm.wrappedValue = Gemma.RMSNorm(
                dimensions: config.hiddenSize, eps: config.rmsNormEps)
        }

        self._layerScalar.wrappedValue = MLXArray.ones([1])

        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache? = nil,
        perLayerInput: MLXArray? = nil,
        sharedKV: (MLXArray, MLXArray)? = nil
    ) -> MLXArray {
        // Attention block
        let r = selfAttention(inputLayerNorm(x), mask: mask, cache: cache, sharedKV: sharedKV)
        let attnNorm = postAttentionLayerNorm(r)
        var h = Gemma.clipResidual(x, attnNorm)

        // MLP block
        let r2 = mlp(preFeedforwardLayerNorm(h))
        let mlpNorm = postFeedforwardLayerNorm(r2)
        h = Gemma.clipResidual(h, mlpNorm)

        // Per-layer gating (text-only: perLayerInput is nil, gate is bypassed)
        if hasPerLayerGate,
           let perLayerInput,
           let gateProj = perLayerInputGate,
           let outProj = perLayerProjection,
           let postNorm = postPerLayerInputNorm
        {
            let residual = h
            var gate = geluApproximate(gateProj(h))
            gate = gate * perLayerInput
            gate = postNorm(outProj(gate))
            h = residual + gate
        }

        // Layer scalar
        h = h * layerScalar

        return h
    }
}

// MARK: - Model

private class Gemma4Model: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo var layers: [TransformerBlock]
    @ModuleInfo var norm: Gemma.RMSNorm

    let config: Gemma4TextConfiguration

    init(_ config: Gemma4TextConfiguration) {
        self.config = config

        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize,
            dimensions: config.hiddenSize
        )

        self._layers.wrappedValue = (0 ..< config.hiddenLayers).map { i in
            TransformerBlock(config, layerIdx: i)
        }

        self.norm = Gemma.RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)

        super.init()
    }

    func callAsFunction(
        _ inputs: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode? = nil,
        cache: [KVCache?]? = nil
    ) -> MLXArray {
        var h = embedTokens(inputs)
        let scale = MLXArray(sqrt(Float(config.hiddenSize)), dtype: .bfloat16)
        h = h * scale.asType(h.dtype)

        var layerCache = cache ?? Array(repeating: nil as KVCache?, count: layers.count)

        // Create attention masks
        var fullMask: MLXFast.ScaledDotProductAttentionMaskMode = .none
        var slidingMask: MLXFast.ScaledDotProductAttentionMaskMode = .none

        if mask == nil {
            let globalIdx = config.layerTypes.firstIndex(of: "full_attention")
            let slidingIdx = config.layerTypes.firstIndex(of: "sliding_attention")

            fullMask = createAttentionMask(
                h: h, cache: globalIdx.flatMap { layerCache[$0] })
            slidingMask = createAttentionMask(
                h: h, cache: slidingIdx.flatMap { layerCache[$0] },
                windowSize: config.slidingWindow)
        }

        // KV sharing store: [layerIndex: (kv, offset)]
        var sharedKVStore = [Int: ((MLXArray, MLXArray), Int)]()

        for (i, layer) in layers.enumerated() {
            let isGlobal = layer.layerType == "full_attention"

            // Resolve shared KV from earlier layer
            var layerSharedKV: (MLXArray, MLXArray)? = nil
            if layer.selfAttention.isKVSharedLayer,
               let refIdx = layer.selfAttention.kvSharedLayerIndex,
               let stored = sharedKVStore[refIdx]
            {
                layerSharedKV = stored.0
            }

            let localMask: MLXFast.ScaledDotProductAttentionMaskMode
            if let mask {
                localMask = mask
            } else if isGlobal {
                localMask = fullMask
            } else {
                localMask = slidingMask
            }

            // Text-only: no per-layer input
            h = layer(
                h, mask: localMask, cache: layerCache[i],
                perLayerInput: nil, sharedKV: layerSharedKV)

            // Store KV for sharing with later layers
            if layer.selfAttention.storeFullLengthKV,
               let kv = layer.selfAttention.lastKV
            {
                let offset = layerCache[i]?.offset ?? 0
                sharedKVStore[i] = (kv, offset)
            }
        }

        return norm(h)
    }
}

// MARK: - Public API

public class Gemma4TextModel: Module, LLMModel {

    @ModuleInfo private var model: Gemma4Model
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public let config: Gemma4TextConfiguration
    public var vocabularySize: Int { config.vocabularySize }

    let finalLogitSoftcapping: Float?

    public init(_ config: Gemma4TextConfiguration) {
        self.config = config
        self.finalLogitSoftcapping = config.finalLogitSoftcapping
        self.model = Gemma4Model(config)

        if !config.tieWordEmbeddings {
            self._lmHead.wrappedValue = Linear(
                config.hiddenSize, config.vocabularySize, bias: false)
        }

        super.init()
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var out = model(inputs, mask: nil, cache: cache)

        if let lmHead {
            out = lmHead(out)
        } else {
            out = model.embedTokens.asLinear(out)
        }

        // Final logit soft-capping: tanh(x / cap) * cap
        if let cap = finalLogitSoftcapping {
            out = tanh(out / cap) * cap
        }

        return out
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var processedWeights = weights

        // Handle VLM models with language_model prefix
        let unflattened = ModuleParameters.unflattened(weights)
        if let lm = unflattened["language_model"] {
            processedWeights = Dictionary(uniqueKeysWithValues: lm.flattened())
        }

        // Skip rotary embeddings (computed dynamically)
        processedWeights = processedWeights.filter { !$0.key.contains("rotary_emb") }

        // Tied embeddings: copy embed_tokens to lm_head if needed
        if config.tieWordEmbeddings && processedWeights["lm_head.weight"] == nil {
            ["weight", "scales", "biases"].forEach { key in
                if let embedWeight = processedWeights["model.embed_tokens.\(key)"] {
                    processedWeights["lm_head.\(key)"] = embedWeight
                }
            }
        }

        return processedWeights
    }

    public func newCache(parameters: GenerateParameters? = nil) -> [KVCache] {
        (0 ..< config.hiddenLayers).map { i in
            let layerType = config.layerTypes[i]
            if layerType == "full_attention" {
                let cache = StandardKVCache()
                cache.step = 1024
                return cache
            } else {
                return RotatingKVCache(maxSize: config.slidingWindow, keep: 0)
            }
        }
    }

    public func prepare(
        _ input: LMInput, cache: [KVCache], windowSize: Int? = nil
    ) throws -> PrepareResult {
        let promptTokens = input.text.tokens
        guard promptTokens.shape[0] > 0 else {
            return .tokens(.init(tokens: MLXArray(Int32(0))[0 ..< 0]))
        }
        return .tokens(input.text)
    }
}

// MARK: - LoRA

extension Gemma4TextModel: LoRAModel {
    public func loraLinearLayers() -> LoRALinearLayers {
        model.layers.map { ($0.selfAttention, ["q_proj", "v_proj"]) }
    }
}
