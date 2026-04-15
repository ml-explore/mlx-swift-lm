// Copyright © 2025 Apple Inc.
//
// Based on https://github.com/Blaizzy/mlx-vlm/tree/main/mlx_vlm/models/gemma4

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Configuration

public struct Gemma4TextConfiguration: Codable, Sendable {
    public let modelType: String
    public let hiddenSize: Int
    public let numHiddenLayers: Int
    public let intermediateSize: Int
    public let numAttentionHeads: Int
    public let headDim: Int
    public let globalHeadDim: Int?
    public let rmsNormEps: Float
    public let vocabularySize: Int
    public let numKeyValueHeads: Int
    public let numGlobalKeyValueHeads: Int?
    public let numKvSharedLayers: Int?
    public let slidingWindow: Int
    public let slidingWindowPattern: Int
    public let maxPositionEmbeddings: Int
    public let finalLogitSoftcapping: Float?
    public let layerTypes: [String]?
    public let ropeParameters: [String: [String: StringOrNumber]]?
    public let attentionKEqV: Bool?
    public let useDoubleWideMlp: Bool?
    public let hiddenSizePerLayerInput: Int?
    public let vocabSizePerLayerInput: Int?

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case numAttentionHeads = "num_attention_heads"
        case headDim = "head_dim"
        case globalHeadDim = "global_head_dim"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case numKeyValueHeads = "num_key_value_heads"
        case numGlobalKeyValueHeads = "num_global_key_value_heads"
        case numKvSharedLayers = "num_kv_shared_layers"
        case slidingWindow = "sliding_window"
        case slidingWindowPattern = "sliding_window_pattern"
        case maxPositionEmbeddings = "max_position_embeddings"
        case finalLogitSoftcapping = "final_logit_softcapping"
        case layerTypes = "layer_types"
        case ropeParameters = "rope_parameters"
        case attentionKEqV = "attention_k_eq_v"
        case useDoubleWideMlp = "use_double_wide_mlp"
        case hiddenSizePerLayerInput = "hidden_size_per_layer_input"
        case vocabSizePerLayerInput = "vocab_size_per_layer_input"
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
        hiddenSize = try container.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 1536
        numHiddenLayers = try container.decodeIfPresent(Int.self, forKey: .numHiddenLayers) ?? 35
        intermediateSize =
            try container.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 6144
        numAttentionHeads =
            try container.decodeIfPresent(Int.self, forKey: .numAttentionHeads) ?? 8
        headDim = try container.decodeIfPresent(Int.self, forKey: .headDim) ?? 256
        globalHeadDim = try container.decodeIfPresent(Int.self, forKey: .globalHeadDim)
        rmsNormEps = try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        vocabularySize =
            try container.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 262144
        numKeyValueHeads =
            try container.decodeIfPresent(Int.self, forKey: .numKeyValueHeads) ?? 1
        numGlobalKeyValueHeads = try container.decodeIfPresent(
            Int.self, forKey: .numGlobalKeyValueHeads)
        numKvSharedLayers = try container.decodeIfPresent(Int.self, forKey: .numKvSharedLayers)
        slidingWindow = try container.decodeIfPresent(Int.self, forKey: .slidingWindow) ?? 512
        slidingWindowPattern =
            try container.decodeIfPresent(Int.self, forKey: .slidingWindowPattern) ?? 5
        maxPositionEmbeddings =
            try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 131072
        finalLogitSoftcapping = try container.decodeIfPresent(
            Float.self, forKey: .finalLogitSoftcapping)
        layerTypes = try container.decodeIfPresent([String].self, forKey: .layerTypes)
        ropeParameters = try container.decodeIfPresent(
            [String: [String: StringOrNumber]].self, forKey: .ropeParameters)
        attentionKEqV = try container.decodeIfPresent(Bool.self, forKey: .attentionKEqV)
        useDoubleWideMlp = try container.decodeIfPresent(Bool.self, forKey: .useDoubleWideMlp)
        hiddenSizePerLayerInput = try container.decodeIfPresent(
            Int.self, forKey: .hiddenSizePerLayerInput)
        vocabSizePerLayerInput = try container.decodeIfPresent(
            Int.self, forKey: .vocabSizePerLayerInput)
    }

    /// Compute the layer types array, defaulting to the sliding_window_pattern if not provided.
    public var effectiveLayerTypes: [String] {
        if let layerTypes {
            return layerTypes
        }
        let pattern =
            Array(repeating: "sliding_attention", count: slidingWindowPattern - 1) + [
                "full_attention"
            ]
        var result: [String] = []
        for i in 0 ..< numHiddenLayers {
            result.append(pattern[i % pattern.count])
        }
        return result
    }

    /// Number of caches (= non-shared layers).
    public var numCaches: Int {
        numHiddenLayers - (numKvSharedLayers ?? 0)
    }
}

// MARK: - RMSNorm Variants

/// RMSNorm with no learnable scale parameter.
private class Gemma4RMSNormNoScale: Module, UnaryLayer {
    let eps: Float

    init(eps: Float = 1e-6) {
        self.eps = eps
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(x, weight: MLXArray.mlxNone, eps: eps)
    }
}

/// RMSNorm where weight is used directly (no +1 offset, unlike Gemma.RMSNorm).
private class Gemma4RMSNormZeroShift: Module, UnaryLayer {
    @ModuleInfo var weight: MLXArray
    let eps: Float

    init(dimensions: Int, eps: Float = 1e-6) {
        self.eps = eps
        self._weight.wrappedValue = MLXArray.ones([dimensions])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(x, weight: weight, eps: eps)
    }
}

// MARK: - MLP

private class Gemma4MLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(hiddenSize: Int, intermediateSize: Int) {
        _gateProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        _upProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        _downProj.wrappedValue = Linear(intermediateSize, hiddenSize, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(geluApproximate(gateProj(x)) * upProj(x))
    }
}

// MARK: - Attention

private class Gemma4Attention: Module {
    let isSliding: Bool
    let isKvSharedLayer: Bool
    let useKEqV: Bool
    let numHeads: Int
    let numKVHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: Gemma.RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: Gemma.RMSNorm
    @ModuleInfo(key: "v_norm") var vNorm: Gemma4RMSNormNoScale
    @ModuleInfo var rope: RoPE

    init(_ config: Gemma4TextConfiguration, layerIdx: Int) {
        let layerTypes = config.effectiveLayerTypes
        self.isSliding = layerTypes[layerIdx] == "sliding_attention"

        let firstKvSharedLayerIdx = config.numHiddenLayers - (config.numKvSharedLayers ?? 0)
        self.isKvSharedLayer =
            (config.numKvSharedLayers ?? 0) > 0 && layerIdx >= firstKvSharedLayerIdx

        // Full-attention layers use globalHeadDim when available
        let useGlobalHeadDim = !self.isSliding && (config.globalHeadDim ?? 0) > 0
        self.headDim = useGlobalHeadDim ? (config.globalHeadDim ?? config.headDim) : config.headDim
        self.numHeads = config.numAttentionHeads

        // K=V optimization for full-attention (26B/31B models)
        self.useKEqV = (config.attentionKEqV ?? false) && !self.isSliding
        if self.useKEqV, let numGlobal = config.numGlobalKeyValueHeads {
            self.numKVHeads = numGlobal
        } else {
            self.numKVHeads = config.numKeyValueHeads
        }

        self.scale = 1.0

        let dim = config.hiddenSize
        _qProj.wrappedValue = Linear(dim, numHeads * headDim, bias: false)
        _kProj.wrappedValue = Linear(dim, numKVHeads * headDim, bias: false)
        _vProj.wrappedValue = Linear(dim, numKVHeads * headDim, bias: false)
        _oProj.wrappedValue = Linear(numHeads * headDim, dim, bias: false)

        _qNorm.wrappedValue = Gemma.RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        _kNorm.wrappedValue = Gemma.RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        _vNorm.wrappedValue = Gemma4RMSNormNoScale(eps: config.rmsNormEps)

        // RoPE: ProportionalRoPE with partial_rotary_factor=1.0 is mathematically
        // equivalent to standard RoPE. Use the rope_theta from rope_parameters.
        let layerKey = self.isSliding ? "sliding_attention" : "full_attention"
        let ropeParams = config.ropeParameters?[layerKey]
        let ropeTheta: Float
        if let theta = ropeParams?["rope_theta"]?.asFloat() {
            ropeTheta = theta
        } else {
            ropeTheta = self.isSliding ? 10000.0 : 1_000_000.0
        }

        _rope.wrappedValue = RoPE(dimensions: headDim, traditional: false, base: ropeTheta)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode = .none,
        cache: KVCache? = nil
    ) -> MLXArray {
        let (B, L, _) = (x.dim(0), x.dim(1), x.dim(2))

        var queries = qProj(x).reshaped(B, L, -1, headDim)
        queries = qNorm(queries)

        var keys: MLXArray
        var values: MLXArray

        if isKvSharedLayer, let cache {
            let state = cache.state
            if state.count >= 2 {
                // Read from shared cache without updating
                keys = state[0]
                values = state[1]
            } else {
                // Fallback: compute normally if cache has no state yet
                keys = kProj(x).reshaped(B, L, -1, headDim)
                if useKEqV { values = keys } else { values = vProj(x).reshaped(B, L, -1, headDim) }
                keys = kNorm(keys)
                values = vNorm(values)
                keys = keys.transposed(0, 2, 1, 3)
                keys = applyRotaryPosition(rope, to: keys, cache: cache)
                values = values.transposed(0, 2, 1, 3)
                (keys, values) = cache.update(keys: keys, values: values)
            }
        } else {
            keys = kProj(x).reshaped(B, L, -1, headDim)
            if useKEqV { values = keys } else { values = vProj(x).reshaped(B, L, -1, headDim) }
            keys = kNorm(keys)
            values = vNorm(values)
            keys = keys.transposed(0, 2, 1, 3)
            keys = applyRotaryPosition(rope, to: keys, cache: cache)
            values = values.transposed(0, 2, 1, 3)
            if let cache {
                (keys, values) = cache.update(keys: keys, values: values)
            }
        }

        queries = queries.transposed(0, 2, 1, 3)
        queries = applyRotaryPosition(rope, to: queries, cache: cache)

        var adjustedMask = mask
        if case .array(let maskArray) = mask {
            let keysSeqLen = keys.dim(keys.ndim - 2)
            if maskArray.shape.last! != keysSeqLen {
                let slicedMask = maskArray[.ellipsis, 0 ..< keysSeqLen].asType(queries.dtype)
                adjustedMask = .array(slicedMask)
            } else {
                adjustedMask = .array(maskArray.asType(queries.dtype))
            }
        }

        let output = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            mask: adjustedMask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        return oProj(output)
    }
}

// MARK: - Decoder Layer

private class Gemma4DecoderLayer: Module {
    let hasPerLayerInput: Bool

    @ModuleInfo(key: "self_attn") var selfAttn: Gemma4Attention
    @ModuleInfo var mlp: Gemma4MLP
    @ModuleInfo(key: "input_layernorm") var inputLayernorm: Gemma.RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayernorm: Gemma.RMSNorm
    @ModuleInfo(key: "pre_feedforward_layernorm") var preFeedforwardLayernorm: Gemma.RMSNorm
    @ModuleInfo(key: "post_feedforward_layernorm") var postFeedforwardLayernorm: Gemma.RMSNorm
    @ModuleInfo(key: "layer_scalar") var layerScalar: MLXArray

    // Per-layer input gating (2B/4B models)
    @ModuleInfo(key: "per_layer_input_gate") var perLayerInputGate: Linear?
    @ModuleInfo(key: "per_layer_projection") var perLayerProjection: Linear?
    @ModuleInfo(key: "post_per_layer_input_norm") var postPerLayerInputNorm: Gemma.RMSNorm?

    init(_ config: Gemma4TextConfiguration, layerIdx: Int) {
        let hiddenSizePerLayerInput = config.hiddenSizePerLayerInput ?? 0
        self.hasPerLayerInput = hiddenSizePerLayerInput > 0

        let firstKvSharedLayerIdx = config.numHiddenLayers - (config.numKvSharedLayers ?? 0)
        let isKvSharedLayer =
            (config.numKvSharedLayers ?? 0) > 0 && layerIdx >= firstKvSharedLayerIdx
        let useDoubleWide = (config.useDoubleWideMlp ?? false) && isKvSharedLayer
        let effectiveIntermediateSize =
            config.intermediateSize * (useDoubleWide ? 2 : 1)

        _selfAttn.wrappedValue = Gemma4Attention(config, layerIdx: layerIdx)
        _mlp.wrappedValue = Gemma4MLP(
            hiddenSize: config.hiddenSize,
            intermediateSize: effectiveIntermediateSize
        )
        _inputLayernorm.wrappedValue = Gemma.RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _postAttentionLayernorm.wrappedValue = Gemma.RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _preFeedforwardLayernorm.wrappedValue = Gemma.RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _postFeedforwardLayernorm.wrappedValue = Gemma.RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)

        _layerScalar.wrappedValue = MLXArray.ones([1])

        if hasPerLayerInput {
            _perLayerInputGate.wrappedValue = Linear(
                config.hiddenSize, hiddenSizePerLayerInput, bias: false)
            _perLayerProjection.wrappedValue = Linear(
                hiddenSizePerLayerInput, config.hiddenSize, bias: false)
            _postPerLayerInputNorm.wrappedValue = Gemma.RMSNorm(
                dimensions: config.hiddenSize, eps: config.rmsNormEps)
        } else {
            _perLayerInputGate.wrappedValue = nil
            _perLayerProjection.wrappedValue = nil
            _postPerLayerInputNorm.wrappedValue = nil
        }

        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode = .none,
        cache: KVCache? = nil,
        perLayerInput: MLXArray? = nil
    ) -> MLXArray {
        var residual = x

        var h = inputLayernorm(x)
        h = selfAttn(h, mask: mask, cache: cache)
        h = postAttentionLayernorm(h)
        h = residual + h

        residual = h

        h = preFeedforwardLayernorm(h)
        h = mlp(h)
        h = postFeedforwardLayernorm(h)
        h = residual + h

        // Per-layer input gating
        if hasPerLayerInput,
            let perLayerInputGate,
            let perLayerProjection,
            let postPerLayerInputNorm,
            let perLayerInput
        {
            residual = h
            var gate = perLayerInputGate(h)
            gate = geluApproximate(gate)
            gate = gate * perLayerInput
            gate = perLayerProjection(gate)
            gate = postPerLayerInputNorm(gate)
            h = residual + gate
        }

        h = h * layerScalar

        return h
    }
}

// MARK: - Text Model

private class Gemma4TextModelInner: Module {
    let config: Gemma4TextConfiguration
    let firstKvSharedLayerIdx: Int
    let firstFullCacheIdx: Int
    let firstSlidingCacheIdx: Int
    let layerIdxToCacheIdx: [Int]
    let hasPerLayerInput: Bool

    // Scaling constants (not learnable weights)
    private let embedScale: Float
    private let embedTokensPerLayerScale: Float
    private let perLayerProjectionScale: Float
    private let perLayerInputScale: Float

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "layers") var layers: [Gemma4DecoderLayer]
    @ModuleInfo var norm: Gemma.RMSNorm

    // Per-layer embeddings (2B/4B models)
    @ModuleInfo(key: "embed_tokens_per_layer") var embedTokensPerLayer: Embedding?
    @ModuleInfo(key: "per_layer_model_projection") var perLayerModelProjection: Linear?
    @ModuleInfo(key: "per_layer_projection_norm") var perLayerProjectionNorm:
        Gemma4RMSNormZeroShift?

    init(_ config: Gemma4TextConfiguration) {
        self.config = config
        self.hasPerLayerInput = (config.hiddenSizePerLayerInput ?? 0) > 0
        self.embedScale = pow(Float(config.hiddenSize), 0.5)

        // KV sharing: compute cache mapping
        let numShared = config.numKvSharedLayers ?? 0
        self.firstKvSharedLayerIdx = config.numHiddenLayers - numShared
        let layerTypes = config.effectiveLayerTypes
        let concreteLayerTypes = Array(layerTypes.prefix(firstKvSharedLayerIdx))

        var idxToCacheIdx = Array(0 ..< firstKvSharedLayerIdx)
        if numShared > 0 {
            let sharedFullIdx =
                concreteLayerTypes.lastIndex(of: "full_attention") ?? (firstKvSharedLayerIdx - 1)
            let sharedSlidingIdx =
                concreteLayerTypes.lastIndex(of: "sliding_attention") ?? 0
            for i in firstKvSharedLayerIdx ..< config.numHiddenLayers {
                if layerTypes[i] == "full_attention" {
                    idxToCacheIdx.append(sharedFullIdx)
                } else {
                    idxToCacheIdx.append(sharedSlidingIdx)
                }
            }
        }
        self.layerIdxToCacheIdx = idxToCacheIdx

        // First cache indices by type (for mask creation)
        self.firstFullCacheIdx =
            concreteLayerTypes.firstIndex(of: "full_attention") ?? 0
        self.firstSlidingCacheIdx =
            concreteLayerTypes.firstIndex(of: "sliding_attention") ?? 0

        // Per-layer input scaling constants
        let hiddenSizePerLayerInput = config.hiddenSizePerLayerInput ?? 256
        let vocabSizePerLayerInput = config.vocabSizePerLayerInput ?? config.vocabularySize
        self.embedTokensPerLayerScale = pow(Float(hiddenSizePerLayerInput), 0.5)
        self.perLayerProjectionScale = pow(Float(config.hiddenSize), -0.5)
        self.perLayerInputScale = pow(2.0, -0.5)

        _embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize,
            dimensions: config.hiddenSize
        )
        _layers.wrappedValue = (0 ..< config.numHiddenLayers).map { idx in
            Gemma4DecoderLayer(config, layerIdx: idx)
        }
        _norm.wrappedValue = Gemma.RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)

        if hasPerLayerInput {
            _embedTokensPerLayer.wrappedValue = Embedding(
                embeddingCount: vocabSizePerLayerInput,
                dimensions: config.numHiddenLayers * hiddenSizePerLayerInput
            )
            _perLayerModelProjection.wrappedValue = Linear(
                config.hiddenSize,
                config.numHiddenLayers * hiddenSizePerLayerInput,
                bias: false
            )
            _perLayerProjectionNorm.wrappedValue = Gemma4RMSNormZeroShift(
                dimensions: hiddenSizePerLayerInput, eps: config.rmsNormEps)
        } else {
            _embedTokensPerLayer.wrappedValue = nil
            _perLayerModelProjection.wrappedValue = nil
            _perLayerProjectionNorm.wrappedValue = nil
        }

        super.init()
    }

    /// Compute per-layer token embeddings from input IDs.
    private func getPerLayerInputs(_ inputIds: MLXArray) -> MLXArray {
        guard let embedTokensPerLayer else { fatalError("embedTokensPerLayer is nil") }
        let hiddenSizePerLayerInput = config.hiddenSizePerLayerInput ?? 256
        var result = embedTokensPerLayer(inputIds)
        result = result * MLXArray(embedTokensPerLayerScale, dtype: result.dtype)
        result = result.reshaped(
            Array(inputIds.shape) + [config.numHiddenLayers, hiddenSizePerLayerInput])
        return result
    }

    /// Project embeddings into per-layer inputs and combine with token embeddings.
    private func projectPerLayerInputs(
        _ inputsEmbeds: MLXArray,
        perLayerInputs: MLXArray?
    ) -> MLXArray {
        guard let perLayerModelProjection,
            let perLayerProjectionNorm
        else { fatalError("Per-layer projection modules are nil") }
        let hiddenSizePerLayerInput = config.hiddenSizePerLayerInput ?? 256

        var perLayerProjection = perLayerModelProjection(inputsEmbeds)
        perLayerProjection =
            perLayerProjection * MLXArray(perLayerProjectionScale, dtype: inputsEmbeds.dtype)
        perLayerProjection = perLayerProjection.reshaped(
            Array(inputsEmbeds.shape.dropLast()) + [
                config.numHiddenLayers, hiddenSizePerLayerInput,
            ])
        perLayerProjection = perLayerProjectionNorm(perLayerProjection)

        guard let perLayerInputs else {
            return perLayerProjection
        }

        return (perLayerProjection + perLayerInputs)
            * MLXArray(perLayerInputScale, dtype: inputsEmbeds.dtype)
    }

    func callAsFunction(
        _ inputs: MLXArray,
        inputsEmbeds: MLXArray? = nil,
        mask: MLXFast.ScaledDotProductAttentionMaskMode? = nil,
        cache: [KVCache?]?
    ) -> MLXArray {
        var h: MLXArray
        if let inputsEmbeds {
            h = inputsEmbeds
        } else {
            h = embedTokens(inputs)
            h = h * MLXArray(embedScale, dtype: h.dtype)
        }

        // Compute per-layer inputs
        var finalPerLayerInputs: MLXArray? = nil
        if hasPerLayerInput {
            let tokenPerLayerInputs = getPerLayerInputs(inputs)
            finalPerLayerInputs = projectPerLayerInputs(h, perLayerInputs: tokenPerLayerInputs)
        }

        let cacheArray: [KVCache?] = cache ?? Array(repeating: nil, count: firstKvSharedLayerIdx)
        let layerTypes = config.effectiveLayerTypes

        let globalMask: MLXFast.ScaledDotProductAttentionMaskMode
        let slidingMask: MLXFast.ScaledDotProductAttentionMaskMode
        if let mask {
            globalMask = mask
            slidingMask = mask
        } else {
            let fullCache =
                firstFullCacheIdx < cacheArray.count ? cacheArray[firstFullCacheIdx] : nil
            let slidingCache =
                firstSlidingCacheIdx < cacheArray.count ? cacheArray[firstSlidingCacheIdx] : nil
            globalMask = createAttentionMask(h: h, cache: fullCache)
            slidingMask = createAttentionMask(
                h: h, cache: slidingCache, windowSize: config.slidingWindow)
        }

        for (i, layer) in layers.enumerated() {
            let cacheIdx = layerIdxToCacheIdx[i]
            let layerCache = cacheIdx < cacheArray.count ? cacheArray[cacheIdx] : nil
            let isGlobal = layerTypes[i] == "full_attention"
            let localMask = isGlobal ? globalMask : slidingMask

            var perLayerInput: MLXArray? = nil
            if let finalPerLayerInputs {
                // Shape: [B, L, numLayers, hiddenSizePerLayerInput] → pick layer i
                perLayerInput = finalPerLayerInputs[0..., 0..., i, 0...]
            }

            h = layer(h, mask: localMask, cache: layerCache, perLayerInput: perLayerInput)
        }

        return norm(h)
    }
}

// MARK: - Top-Level Model

public class Gemma4TextModel: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    @ModuleInfo fileprivate var model: Gemma4TextModelInner

    private let config: Gemma4TextConfiguration
    private let finalLogitSoftcapping: Float?

    public init(_ config: Gemma4TextConfiguration) {
        self.config = config
        self.vocabularySize = config.vocabularySize
        self.finalLogitSoftcapping = config.finalLogitSoftcapping

        let layerTypes = config.effectiveLayerTypes
        self.kvHeads = layerTypes.map { _ in config.numKeyValueHeads }

        _model.wrappedValue = Gemma4TextModelInner(config)
        super.init()
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let cacheArray = cache?.map { $0 as KVCache? }
        let out = model(inputs, cache: cacheArray)

        var logits = model.embedTokens.asLinear(out)
        if let cap = finalLogitSoftcapping {
            logits = tanh(logits / cap) * cap
        }
        return logits
    }

    public func newCache(parameters: GenerateParameters?) -> [any KVCache] {
        let layerTypes = config.effectiveLayerTypes
        var caches: [any KVCache] = []
        for i in 0 ..< config.numCaches {
            if layerTypes[i] == "full_attention" {
                caches.append(StandardKVCache())
            } else {
                caches.append(RotatingKVCache(maxSize: config.slidingWindow, keep: 0))
            }
        }
        return caches
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized: [String: MLXArray] = [:]
        for (key, value) in weights {
            // Strip VLM-style "model." prefix
            let strippedKey: String
            if key.hasPrefix("model.") {
                strippedKey = String(key.dropFirst("model.".count))
            } else {
                strippedKey = key
            }
            // Skip rotary embedding and clipping weights
            if strippedKey.contains("self_attn.rotary_emb") { continue }
            if strippedKey.contains("input_max") || strippedKey.contains("input_min")
                || strippedKey.contains("output_max") || strippedKey.contains("output_min")
            {
                continue
            }
            sanitized[strippedKey] = value
        }

        // Trim vocabulary if oversized
        let vocabKeys = [
            "model.embed_tokens.weight",
            "model.embed_tokens.scales",
            "model.embed_tokens.biases",
        ]
        for key in vocabKeys {
            if let tensor = sanitized[key], tensor.dim(0) > vocabularySize {
                sanitized[key] = tensor[0 ..< vocabularySize]
            }
        }

        return sanitized
    }
}

extension Gemma4TextModel: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}
