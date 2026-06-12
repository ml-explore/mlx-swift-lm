import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Compiled fusion fragments

private let diffusionGemmaRMSEps: Float = 1e-6

private let _diffusionGemmaCompiledAddRMSNorm: @Sendable (MLXArray, MLXArray, MLXArray) ->
    MLXArray = compile(shapeless: true) { residual, x, weight in
        residual + MLXFast.rmsNorm(x, weight: weight, eps: diffusionGemmaRMSEps)
    }

private let _diffusionGemmaCompiledGeluMul: @Sendable (MLXArray, MLXArray) -> MLXArray = compile(
    shapeless: true
) { gate, other in
    geluApproximate(gate) * other
}

private func diffusionGemmaAddRMSNorm(
    _ residual: MLXArray, _ x: MLXArray, _ weight: MLXArray
) -> MLXArray {
    if Device.defaultDevice().deviceType == .cpu {
        return residual + MLXFast.rmsNorm(x, weight: weight, eps: diffusionGemmaRMSEps)
    }
    return _diffusionGemmaCompiledAddRMSNorm(residual, x, weight)
}

private func diffusionGemmaGeluMul(_ gate: MLXArray, _ other: MLXArray) -> MLXArray {
    if Device.defaultDevice().deviceType == .cpu {
        return geluApproximate(gate) * other
    }
    return _diffusionGemmaCompiledGeluMul(gate, other)
}

private func diffusionGemmaLayerTypes(hiddenLayers: Int, slidingWindowPattern: Int) -> [String] {
    let pattern =
        Array(repeating: "sliding_attention", count: max(slidingWindowPattern - 1, 0))
        + ["full_attention"]
    guard !pattern.isEmpty else { return Array(repeating: "full_attention", count: hiddenLayers) }
    var result: [String] = []
    while result.count < hiddenLayers {
        result.append(contentsOf: pattern)
    }
    return Array(result.prefix(hiddenLayers))
}

private func diffusionGemmaDefaultRopeParameters() -> [String: [String: StringOrNumber]] {
    [
        "full_attention": [
            "partial_rotary_factor": .float(0.25),
            "rope_theta": .float(1_000_000.0),
            "rope_type": .string("proportional"),
        ],
        "sliding_attention": [
            "rope_theta": .float(10_000.0),
            "rope_type": .string("default"),
        ],
    ]
}

public struct DiffusionGemmaTextConfiguration: Codable, Sendable {
    var modelType: String = "diffusion_gemma_text"
    var hiddenSize: Int = 2816
    var hiddenLayers: Int = 30
    var intermediateSize: Int = 2112
    var moeIntermediateSize: Int = 704
    var attentionHeads: Int = 16
    var kvHeads: Int = 8
    var globalKVHeads: Int = 2
    var headDim: Int = 256
    var globalHeadDim: Int = 512
    var vocabularySize: Int = 262_144
    var slidingWindow: Int = 1024
    var slidingWindowPattern: Int = 6
    var maxPositionEmbeddings: Int = 262_144
    var rmsNormEps: Float = 1e-6
    var finalLogitSoftcapping: Float = 30.0
    var attentionBias: Bool = false
    var topKExperts: Int = 8
    var numExperts: Int = 128
    var tieWordEmbeddings: Bool = true
    var layerTypes: [String] = []
    var ropeParameters: [String: [String: StringOrNumber]] = diffusionGemmaDefaultRopeParameters()

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case moeIntermediateSize = "moe_intermediate_size"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case globalKVHeads = "num_global_key_value_heads"
        case headDim = "head_dim"
        case globalHeadDim = "global_head_dim"
        case vocabularySize = "vocab_size"
        case slidingWindow = "sliding_window"
        case slidingWindowPattern = "sliding_window_pattern"
        case maxPositionEmbeddings = "max_position_embeddings"
        case rmsNormEps = "rms_norm_eps"
        case finalLogitSoftcapping = "final_logit_softcapping"
        case attentionBias = "attention_bias"
        case topKExperts = "top_k_experts"
        case numExperts = "num_experts"
        case tieWordEmbeddings = "tie_word_embeddings"
        case layerTypes = "layer_types"
        case ropeParameters = "rope_parameters"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? modelType
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? hiddenSize
        hiddenLayers = try c.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? hiddenLayers
        intermediateSize =
            try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? intermediateSize
        moeIntermediateSize =
            try c.decodeIfPresent(Int.self, forKey: .moeIntermediateSize) ?? moeIntermediateSize
        attentionHeads = try c.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? attentionHeads
        kvHeads = try c.decodeIfPresent(Int.self, forKey: .kvHeads) ?? kvHeads
        globalKVHeads = try c.decodeIfPresent(Int.self, forKey: .globalKVHeads) ?? globalKVHeads
        headDim = try c.decodeIfPresent(Int.self, forKey: .headDim) ?? headDim
        globalHeadDim = try c.decodeIfPresent(Int.self, forKey: .globalHeadDim) ?? globalHeadDim
        vocabularySize =
            try c.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? vocabularySize
        slidingWindow = try c.decodeIfPresent(Int.self, forKey: .slidingWindow) ?? slidingWindow
        slidingWindowPattern =
            try c.decodeIfPresent(Int.self, forKey: .slidingWindowPattern)
            ?? slidingWindowPattern
        maxPositionEmbeddings =
            try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings)
            ?? maxPositionEmbeddings
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? rmsNormEps
        finalLogitSoftcapping =
            try c.decodeIfPresent(Float.self, forKey: .finalLogitSoftcapping)
            ?? finalLogitSoftcapping
        attentionBias = try c.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? attentionBias
        topKExperts = try c.decodeIfPresent(Int.self, forKey: .topKExperts) ?? topKExperts
        numExperts = try c.decodeIfPresent(Int.self, forKey: .numExperts) ?? numExperts
        tieWordEmbeddings =
            try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? tieWordEmbeddings
        layerTypes =
            try c.decodeIfPresent([String].self, forKey: .layerTypes)
            ?? diffusionGemmaLayerTypes(
                hiddenLayers: hiddenLayers, slidingWindowPattern: slidingWindowPattern)
        ropeParameters =
            try c.decodeIfPresent(
                [String: [String: StringOrNumber]].self, forKey: .ropeParameters)
            ?? diffusionGemmaDefaultRopeParameters()
    }
}

public struct DiffusionGemmaConfiguration: Codable, Sendable {
    var modelType: String = "diffusion_gemma"
    var textConfig: DiffusionGemmaTextConfiguration
    var canvasLength: Int = 256
    var tieWordEmbeddings: Bool = true

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case textConfig = "text_config"
        case canvasLength = "canvas_length"
        case tieWordEmbeddings = "tie_word_embeddings"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? modelType
        textConfig = try c.decode(DiffusionGemmaTextConfiguration.self, forKey: .textConfig)
        canvasLength = try c.decodeIfPresent(Int.self, forKey: .canvasLength) ?? canvasLength
        tieWordEmbeddings =
            try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? tieWordEmbeddings
    }
}

private final class DiffusionGemmaRMSNormNoScale: Module {
    let eps: Float

    init(eps: Float) {
        self.eps = eps
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(x, weight: MLXArray.mlxNone, eps: eps)
    }
}

private final class DiffusionGemmaMLP: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(_ config: DiffusionGemmaTextConfiguration) {
        _gateProj.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        _upProj.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        _downProj.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(diffusionGemmaGeluMul(gateProj(x), upProj(x)))
    }
}

private final class DiffusionGemmaRouter: Module {
    let config: DiffusionGemmaTextConfiguration
    let rootSize: Float

    @ModuleInfo(key: "proj") var proj: Linear
    @ParameterInfo(key: "scale") var scale: MLXArray
    @ParameterInfo(key: "per_expert_scale") var perExpertScale: MLXArray

    init(_ config: DiffusionGemmaTextConfiguration) {
        self.config = config
        self.rootSize = pow(Float(config.hiddenSize), -0.5)
        _proj.wrappedValue = Linear(config.hiddenSize, config.numExperts, bias: false)
        _scale.wrappedValue = MLXArray.ones([config.hiddenSize])
        _perExpertScale.wrappedValue = MLXArray.ones([config.numExperts])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray) {
        var h = MLXFast.rmsNorm(x, weight: MLXArray.mlxNone, eps: config.rmsNormEps)
        h = h * scale.asType(h.dtype) * rootSize

        let probabilities = softmax(proj(h), axis: -1, precise: true)
        let topKIndices = argPartition(probabilities, kth: -config.topKExperts, axis: -1)[
            .ellipsis, (-config.topKExperts)...]
        var topKWeights = takeAlong(probabilities, topKIndices, axis: -1)
        topKWeights = topKWeights / topKWeights.sum(axis: -1, keepDims: true)
        topKWeights = topKWeights * perExpertScale[topKIndices].asType(topKWeights.dtype)
        return (topKIndices, topKWeights)
    }
}

private final class DiffusionGemmaExperts: Module {
    @ModuleInfo(key: "gate_up_proj") var gateUpProj: SwitchLinear
    @ModuleInfo(key: "down_proj") var downProj: SwitchLinear

    init(_ config: DiffusionGemmaTextConfiguration) {
        _gateUpProj.wrappedValue = SwitchLinear(
            inputDims: config.hiddenSize,
            outputDims: 2 * config.moeIntermediateSize,
            numExperts: config.numExperts,
            bias: false)
        _downProj.wrappedValue = SwitchLinear(
            inputDims: config.moeIntermediateSize,
            outputDims: config.hiddenSize,
            numExperts: config.numExperts,
            bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, indices: MLXArray, weights: MLXArray) -> MLXArray {
        var x = MLX.expandedDimensions(x, axes: [-2, -3])
        let doSort = indices.size >= 64

        var idx = indices
        var inverseOrder = MLXArray()

        if doSort {
            (x, idx, inverseOrder) = gatherSort(x: x, indices: indices)
        }

        let gateUp = gateUpProj(x, idx, sortedIndices: doSort)
        let parts = split(gateUp, parts: 2, axis: -1)
        let activated = diffusionGemmaGeluMul(parts[0], parts[1])
        var expertOutput = downProj(activated, idx, sortedIndices: doSort)

        if doSort {
            expertOutput = scatterUnsort(
                x: expertOutput, invOrder: inverseOrder, shape: indices.shape)
        }

        expertOutput = MLX.squeezed(expertOutput, axis: -2)
        return weightedExpertSum(expertOutput, weights)
    }
}

private enum DiffusionGemmaAttentionMode {
    case encoder
    case decoder
}

private final class DiffusionGemmaAttention: Module {
    let config: DiffusionGemmaTextConfiguration
    let layerIdx: Int
    let layerType: String
    let isSliding: Bool
    let headDim: Int
    let numHeads: Int
    let numKVHeads: Int
    let mode: DiffusionGemmaAttentionMode

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear?
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm
    @ModuleInfo(key: "v_norm") var vNorm: DiffusionGemmaRMSNormNoScale
    @ModuleInfo var rope: RoPELayer

    init(
        _ config: DiffusionGemmaTextConfiguration,
        layerIdx: Int,
        mode: DiffusionGemmaAttentionMode
    ) {
        self.config = config
        self.layerIdx = layerIdx
        self.layerType = config.layerTypes[layerIdx]
        self.isSliding = layerType == "sliding_attention"
        self.headDim = isSliding ? config.headDim : config.globalHeadDim
        self.numHeads = config.attentionHeads
        self.numKVHeads = isSliding ? config.kvHeads : config.globalKVHeads
        self.mode = mode

        _qProj.wrappedValue = Linear(config.hiddenSize, numHeads * headDim, bias: false)
        _kProj.wrappedValue = Linear(config.hiddenSize, numKVHeads * headDim, bias: false)
        if isSliding {
            _vProj.wrappedValue = Linear(config.hiddenSize, numKVHeads * headDim, bias: false)
        }
        _oProj.wrappedValue = Linear(numHeads * headDim, config.hiddenSize, bias: false)
        _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        _vNorm.wrappedValue = DiffusionGemmaRMSNormNoScale(eps: config.rmsNormEps)

        let ropeConfig = config.ropeParameters[layerType]
        _rope.wrappedValue = initializeRope(
            dims: headDim,
            base: ropeConfig?["rope_theta"]?.asFloat() ?? (isSliding ? 10_000 : 1_000_000),
            traditional: false,
            scalingConfig: ropeConfig,
            maxPositionEmbeddings: nil)

        super.init()
    }

    private func cacheState(_ cache: KVCache?) -> (MLXArray, MLXArray)? {
        guard let state = cache?.state, state.count == 2 else {
            return nil
        }
        return (state[0], state[1])
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let (batch, length, _) = (x.dim(0), x.dim(1), x.dim(2))
        let offset = cache?.offset ?? 0

        var queries = qProj(x).reshaped(batch, length, numHeads, headDim)
        queries = qNorm(queries).transposed(0, 2, 1, 3)
        queries = applyRotaryPosition(rope, to: queries, offset: .scalar(offset))

        let keyRaw = kProj(x).reshaped(batch, length, numKVHeads, headDim)
        var keys = kNorm(keyRaw).transposed(0, 2, 1, 3)
        keys = applyRotaryPosition(rope, to: keys, offset: .scalar(offset))

        var values: MLXArray
        if let vProj {
            values = vProj(x).reshaped(batch, length, numKVHeads, headDim)
        } else {
            values = keyRaw
        }
        values = vNorm(values).transposed(0, 2, 1, 3)

        let finalKeys: MLXArray
        let finalValues: MLXArray
        switch mode {
        case .encoder:
            if let cache {
                (finalKeys, finalValues) = cache.update(keys: keys, values: values)
            } else {
                finalKeys = keys
                finalValues = values
            }
        case .decoder:
            if let (cachedKeys, cachedValues) = cacheState(cache) {
                finalKeys = concatenated([cachedKeys, keys], axis: 2)
                finalValues = concatenated([cachedValues, values], axis: 2)
            } else {
                finalKeys = keys
                finalValues = values
            }
        }

        let output = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: finalKeys,
            values: finalValues,
            scale: 1.0,
            mask: mask
        )
        return oProj(output.transposed(0, 2, 1, 3).reshaped(batch, length, -1))
    }
}

private final class DiffusionGemmaLayer: Module {
    let layerType: String

    @ModuleInfo(key: "self_attn") var attention: DiffusionGemmaAttention
    @ModuleInfo var mlp: DiffusionGemmaMLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm
    @ModuleInfo(key: "pre_feedforward_layernorm") var preFeedforwardLayerNorm: RMSNorm
    @ModuleInfo(key: "post_feedforward_layernorm") var postFeedforwardLayerNorm: RMSNorm
    @ModuleInfo var router: DiffusionGemmaRouter
    @ModuleInfo var experts: DiffusionGemmaExperts
    @ModuleInfo(key: "post_feedforward_layernorm_1") var postFeedforwardLayerNorm1: RMSNorm
    @ModuleInfo(key: "post_feedforward_layernorm_2") var postFeedforwardLayerNorm2: RMSNorm
    @ModuleInfo(key: "pre_feedforward_layernorm_2") var preFeedforwardLayerNorm2: RMSNorm
    @ModuleInfo(key: "layer_scalar") var layerScalar: MLXArray

    init(
        _ config: DiffusionGemmaTextConfiguration,
        layerIdx: Int,
        mode: DiffusionGemmaAttentionMode
    ) {
        precondition(
            config.rmsNormEps == diffusionGemmaRMSEps,
            "DiffusionGemma fused path requires rmsNormEps == \(diffusionGemmaRMSEps), got \(config.rmsNormEps)"
        )

        self.layerType = config.layerTypes[layerIdx]
        _attention.wrappedValue = DiffusionGemmaAttention(config, layerIdx: layerIdx, mode: mode)
        _mlp.wrappedValue = DiffusionGemmaMLP(config)
        _inputLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _preFeedforwardLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _postFeedforwardLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _router.wrappedValue = DiffusionGemmaRouter(config)
        _experts.wrappedValue = DiffusionGemmaExperts(config)
        _postFeedforwardLayerNorm1.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _postFeedforwardLayerNorm2.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _preFeedforwardLayerNorm2.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _layerScalar.wrappedValue = MLXArray.ones([1])
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        var residual = x
        var h = inputLayerNorm(x)
        h = attention(h, mask: mask, cache: cache)
        h = diffusionGemmaAddRMSNorm(residual, h, postAttentionLayerNorm.weight)

        residual = h
        let dense = postFeedforwardLayerNorm1(mlp(preFeedforwardLayerNorm(h)))

        let flat = residual.reshaped(-1, residual.dim(-1))
        let (indices, weights) = router(flat)
        var sparse = preFeedforwardLayerNorm2(flat)
        sparse = experts(sparse, indices: indices, weights: weights)
        sparse = sparse.reshaped(residual.shape)
        sparse = postFeedforwardLayerNorm2(sparse)

        h = diffusionGemmaAddRMSNorm(residual, dense + sparse, postFeedforwardLayerNorm.weight)
        return h * layerScalar.asType(h.dtype)
    }
}

private final class DiffusionGemmaTextStack: Module {
    let config: DiffusionGemmaTextConfiguration
    let mode: DiffusionGemmaAttentionMode
    let embedScale: Float

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo var layers: [DiffusionGemmaLayer]
    @ModuleInfo var norm: RMSNorm

    init(_ config: DiffusionGemmaTextConfiguration, mode: DiffusionGemmaAttentionMode) {
        self.config = config
        self.mode = mode
        self.embedScale = pow(Float(config.hiddenSize), 0.5)
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize,
            dimensions: config.hiddenSize)
        _layers.wrappedValue = (0 ..< config.hiddenLayers).map {
            DiffusionGemmaLayer(config, layerIdx: $0, mode: mode)
        }
        _norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        super.init()
    }

    func embeddings(_ tokens: MLXArray) -> MLXArray {
        let e = embedTokens(tokens)
        return (e * MLXArray(embedScale, dtype: .float32)).asType(e.dtype)
    }

    func callAsFunction(
        inputs: MLXArray? = nil,
        inputEmbeddings: MLXArray? = nil,
        cache: [KVCache]? = nil
    ) -> MLXArray {
        var h: MLXArray
        if let inputEmbeddings {
            h = inputEmbeddings
        } else if let inputs {
            h = embeddings(inputs)
        } else {
            fatalError("DiffusionGemma requires token ids or input embeddings")
        }

        for (idx, layer) in layers.enumerated() {
            let layerCache = cache?[idx]
            let mask: MLXFast.ScaledDotProductAttentionMaskMode
            switch mode {
            case .encoder:
                if layer.layerType == "sliding_attention" {
                    mask = createAttentionMask(h: h, cache: layerCache, windowSize: config.slidingWindow)
                } else {
                    mask = createAttentionMask(h: h, cache: layerCache)
                }
            case .decoder:
                mask = .none
            }
            h = layer(h, mask: mask, cache: layerCache)
        }

        return norm(h)
    }
}

private final class DiffusionGemmaEncoder: Module {
    @ModuleInfo(key: "language_model") var languageModel: DiffusionGemmaTextStack

    init(_ config: DiffusionGemmaTextConfiguration) {
        _languageModel.wrappedValue = DiffusionGemmaTextStack(config, mode: .encoder)
        super.init()
    }
}

private final class DiffusionGemmaSelfConditioning: Module {
    @ModuleInfo(key: "pre_norm") var preNorm: RMSNorm
    @ModuleInfo(key: "post_norm") var postNorm: DiffusionGemmaRMSNormNoScale
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(_ config: DiffusionGemmaTextConfiguration) {
        _preNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _postNorm.wrappedValue = DiffusionGemmaRMSNormNoScale(eps: config.rmsNormEps)
        _gateProj.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        _upProj.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        _downProj.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: false)
        super.init()
    }

    func callAsFunction(_ inputs: MLXArray, signal: MLXArray) -> MLXArray {
        let normed = preNorm(signal)
        let conditioned = downProj(diffusionGemmaGeluMul(gateProj(normed), upProj(normed)))
        return postNorm(inputs + conditioned)
    }
}

private final class DiffusionGemmaDecoder: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo var layers: [DiffusionGemmaLayer]
    @ModuleInfo var norm: RMSNorm
    @ModuleInfo(key: "self_conditioning") var selfConditioning: DiffusionGemmaSelfConditioning

    let config: DiffusionGemmaTextConfiguration
    let embedScale: Float

    init(_ config: DiffusionGemmaTextConfiguration) {
        self.config = config
        self.embedScale = pow(Float(config.hiddenSize), 0.5)
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize,
            dimensions: config.hiddenSize)
        _layers.wrappedValue = (0 ..< config.hiddenLayers).map {
            DiffusionGemmaLayer(config, layerIdx: $0, mode: .decoder)
        }
        _norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _selfConditioning.wrappedValue = DiffusionGemmaSelfConditioning(config)
        super.init()
    }

    func callAsFunction(
        canvasTokens: MLXArray,
        cache: [KVCache],
        selfConditioningLogits: MLXArray?
    ) -> MLXArray {
        var h = embedTokens(canvasTokens)
        h = (h * MLXArray(embedScale, dtype: .float32)).asType(h.dtype)

        let signal: MLXArray
        if let selfConditioningLogits {
            let probabilities = softmax(selfConditioningLogits, axis: -1, precise: true)
            signal = matmul(probabilities.asType(embedTokens.weight.dtype), embedTokens.weight)
                * MLXArray(embedScale, dtype: .float32).asType(h.dtype)
        } else {
            signal = MLXArray.zeros(h.shape, dtype: h.dtype)
        }
        h = selfConditioning(h, signal: signal)

        for (idx, layer) in layers.enumerated() {
            h = layer(h, mask: .none, cache: cache[idx])
        }
        return norm(h)
    }
}

private final class DiffusionGemmaCore: Module {
    @ModuleInfo var encoder: DiffusionGemmaEncoder
    @ModuleInfo var decoder: DiffusionGemmaDecoder

    init(_ config: DiffusionGemmaTextConfiguration) {
        _encoder.wrappedValue = DiffusionGemmaEncoder(config)
        _decoder.wrappedValue = DiffusionGemmaDecoder(config)
        super.init()
    }
}

public final class DiffusionGemmaModel: Module, LLMModel, BlockDiffusionLanguageModel,
    KVCacheDimensionProvider
{
    public let vocabularySize: Int
    public var diffusionVocabularySize: Int { vocabularySize }
    public let diffusionCanvasLength: Int
    public let diffusionMaxDenoisingSteps = 48
    public let diffusionEntropyBound: Float = 0.1
    public let kvHeads: [Int]

    let config: DiffusionGemmaConfiguration
    @ModuleInfo fileprivate var model: DiffusionGemmaCore
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ config: DiffusionGemmaConfiguration) {
        self.config = config
        self.vocabularySize = config.textConfig.vocabularySize
        self.diffusionCanvasLength = config.canvasLength
        self.kvHeads = (0 ..< config.textConfig.hiddenLayers).map { idx in
            config.textConfig.layerTypes[idx] == "sliding_attention"
                ? config.textConfig.kvHeads : config.textConfig.globalKVHeads
        }
        _model.wrappedValue = DiffusionGemmaCore(config.textConfig)
        if !config.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(
                config.textConfig.hiddenSize,
                config.textConfig.vocabularySize,
                bias: false)
        }
        super.init()
    }

    private func encodeIntoCache(_ tokens: MLXArray, cache: [KVCache], windowSize: Int?) {
        let tokens = tokens.ndim == 1 ? tokens.expandedDimensions(axis: 0) : tokens
        let chunkSize = windowSize ?? 512
        var start = 0

        while start < tokens.dim(1) {
            let end = Swift.min(start + chunkSize, tokens.dim(1))
            _ = model.encoder.languageModel(inputs: tokens[0..., start ..< end], cache: cache)
            asyncEval(cache)
            start = end
        }

        eval(cache)
    }

    public func prepareDiffusion(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws {
        encodeIntoCache(input.text.tokens, cache: cache, windowSize: windowSize)
    }

    public func acceptDiffusionTokens(_ tokens: MLXArray, cache: [KVCache], windowSize: Int?) {
        encodeIntoCache(tokens, cache: cache, windowSize: windowSize)
    }

    public func diffusionLogits(
        canvasTokens: MLXArray,
        cache: [KVCache],
        selfConditioningLogits: MLXArray?
    ) -> MLXArray {
        let hidden = model.decoder(
            canvasTokens: canvasTokens.ndim == 1
                ? canvasTokens.expandedDimensions(axis: 0) : canvasTokens,
            cache: cache,
            selfConditioningLogits: selfConditioningLogits)
        var logits: MLXArray
        if let lmHead {
            logits = lmHead(hidden)
        } else {
            logits = model.decoder.embedTokens.asLinear(hidden)
        }
        logits = logits.asType(.float32)
        logits = tanh(logits / config.textConfig.finalLogitSoftcapping)
            * config.textConfig.finalLogitSoftcapping
        return logits
    }

    public func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws
        -> PrepareResult
    {
        throw GenerateError.unsupportedAutoregressiveGeneration(String(describing: Self.self))
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        fatalError(
            "DiffusionGemmaModel does not produce autoregressive next-token logits. Use prepareDiffusion(_:cache:windowSize:) and diffusionLogits(canvasTokens:cache:selfConditioningLogits:) through BlockDiffusionTokenIterator."
        )
    }

    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        config.textConfig.layerTypes.map { layerType in
            if layerType == "sliding_attention" {
                RotatingKVCache(maxSize: config.textConfig.slidingWindow, keep: 0)
            } else {
                StandardKVCache()
            }
        }
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized = weights.filter { key, _ in
            !key.hasPrefix("model.encoder.vision_tower")
                && !key.hasPrefix("model.encoder.embedder")
                && !key.hasPrefix("model.vision_tower")
                && !key.hasPrefix("model.embed_vision")
                && !key.contains("rotary_emb")
                && !key.contains("input_min")
                && !key.contains("input_max")
                && !key.contains("output_min")
                && !key.contains("output_max")
        }

        // Official DiffusionGemma checkpoints store the shared text weights once
        // under the decoder path. Mirror the HF tied-weight map so strict loading
        // also initializes the encoder text stack.
        for (key, value) in sanitized {
            let tiedKey: String?
            if key == "model.decoder.embed_tokens.weight" {
                tiedKey = "model.encoder.language_model.embed_tokens.weight"
            } else if key == "model.decoder.norm.weight" {
                tiedKey = "model.encoder.language_model.norm.weight"
            } else if key.hasPrefix("model.decoder.layers.") {
                tiedKey = key.replacingOccurrences(
                    of: "model.decoder.layers.",
                    with: "model.encoder.language_model.layers.")
            } else {
                tiedKey = nil
            }

            if let tiedKey, sanitized[tiedKey] == nil {
                sanitized[tiedKey] = value
            }
        }

        if config.tieWordEmbeddings {
            sanitized = sanitized.filter { key, _ in
                !key.hasPrefix("lm_head.")
            }
        } else if sanitized["lm_head.weight"] == nil,
            let embedWeight = sanitized["model.decoder.embed_tokens.weight"]
        {
            sanitized["lm_head.weight"] = embedWeight
        }

        return sanitized
    }
}

extension DiffusionGemmaModel: LoRAModel {
    public var loraLayers: [Module] {
        model.encoder.languageModel.layers.map { $0.attention }
            + model.decoder.layers.map { $0.attention }
    }
}
