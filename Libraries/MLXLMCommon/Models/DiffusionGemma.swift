// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN

// MARK: - Compiled fusion fragments

private let diffusionGemmaRMSEps: Float = 1e-6

private let _diffusionGemmaCompiledAddRMSNorm:
    @Sendable (MLXArray, MLXArray, MLXArray) ->
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
    public var modelType: String = "diffusion_gemma_text"
    public var hiddenSize: Int = 2816
    public var hiddenLayers: Int = 30
    public var intermediateSize: Int = 2112
    public var moeIntermediateSize: Int = 704
    public var attentionHeads: Int = 16
    public var kvHeads: Int = 8
    public var globalKVHeads: Int = 2
    public var headDim: Int = 256
    public var globalHeadDim: Int = 512
    public var vocabularySize: Int = 262_144
    public var slidingWindow: Int = 1024
    public var slidingWindowPattern: Int = 6
    public var maxPositionEmbeddings: Int = 262_144
    public var rmsNormEps: Float = 1e-6
    public var finalLogitSoftcapping: Float = 30.0
    public var attentionBias: Bool = false
    public var topKExperts: Int = 8
    public var numExperts: Int = 128
    public var tieWordEmbeddings: Bool = true
    public var layerTypes: [String] = []
    public var ropeParameters: [String: [String: StringOrNumber]] =
        diffusionGemmaDefaultRopeParameters()

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
    public var modelType: String = "diffusion_gemma"
    public var textConfig: DiffusionGemmaTextConfiguration
    public var generationConfig: DiffusionGemmaGenerationConfiguration = .init()
    public var canvasLength: Int = 256
    public var tieWordEmbeddings: Bool = true

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case textConfig = "text_config"
        case generationConfig = "generation_config"
        case canvasLength = "canvas_length"
        case tieWordEmbeddings = "tie_word_embeddings"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? modelType
        textConfig = try c.decode(DiffusionGemmaTextConfiguration.self, forKey: .textConfig)
        generationConfig =
            try c.decodeIfPresent(
                DiffusionGemmaGenerationConfiguration.self, forKey: .generationConfig)
            ?? generationConfig
        canvasLength = try c.decodeIfPresent(Int.self, forKey: .canvasLength) ?? canvasLength
        tieWordEmbeddings =
            try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? tieWordEmbeddings
    }

    public init(
        textConfig: DiffusionGemmaTextConfiguration,
        generationConfig: DiffusionGemmaGenerationConfiguration = .init(),
        canvasLength: Int = 256,
        tieWordEmbeddings: Bool = true,
        modelType: String = "diffusion_gemma"
    ) {
        self.textConfig = textConfig
        self.generationConfig = generationConfig
        self.canvasLength = canvasLength
        self.tieWordEmbeddings = tieWordEmbeddings
        self.modelType = modelType
    }
}

public struct DiffusionGemmaGenerationConfiguration: Codable, Sendable {
    public var confidenceThreshold: Float = 0.005
    public var maxDenoisingSteps: Int = 48
    public var maxNewTokens: Int = 256
    public var samplerConfig: SamplerConfiguration = .init()
    public var stabilityThreshold: Int = 1
    public var temperatureMax: Float = 0.8
    public var temperatureMin: Float = 0.4

    enum CodingKeys: String, CodingKey {
        case confidenceThreshold = "confidence_threshold"
        case maxDenoisingSteps = "max_denoising_steps"
        case maxNewTokens = "max_new_tokens"
        case samplerConfig = "sampler_config"
        case stabilityThreshold = "stability_threshold"
        case temperatureMax = "t_max"
        case temperatureMin = "t_min"
    }

    public init() {}

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        confidenceThreshold =
            try c.decodeIfPresent(Float.self, forKey: .confidenceThreshold) ?? confidenceThreshold
        maxDenoisingSteps =
            try c.decodeIfPresent(Int.self, forKey: .maxDenoisingSteps) ?? maxDenoisingSteps
        maxNewTokens = try c.decodeIfPresent(Int.self, forKey: .maxNewTokens) ?? maxNewTokens
        samplerConfig =
            try c.decodeIfPresent(SamplerConfiguration.self, forKey: .samplerConfig)
            ?? samplerConfig
        stabilityThreshold =
            try c.decodeIfPresent(Int.self, forKey: .stabilityThreshold) ?? stabilityThreshold
        temperatureMax =
            try c.decodeIfPresent(Float.self, forKey: .temperatureMax) ?? temperatureMax
        temperatureMin =
            try c.decodeIfPresent(Float.self, forKey: .temperatureMin) ?? temperatureMin
    }

    public struct SamplerConfiguration: Codable, Sendable {
        public var entropyBound: Float = 0.1

        enum CodingKeys: String, CodingKey {
            case entropyBound = "entropy_bound"
        }

        public init() {}

        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            entropyBound = try c.decodeIfPresent(Float.self, forKey: .entropyBound) ?? entropyBound
        }
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

        let scores = proj(h)
        let topKIndices = argPartition(scores, kth: -config.topKExperts, axis: -1)[
            .ellipsis, (-config.topKExperts)...]
        var topKWeights = takeAlong(scores, topKIndices, axis: -1)
        topKWeights = softmax(topKWeights, axis: -1, precise: true)
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

private func diffusionGemmaCacheState(_ cache: KVCache?) -> (MLXArray, MLXArray)? {
    guard let cache else {
        return nil
    }

    let state =
        if let rotatingCache = cache as? RotatingKVCache {
            rotatingCache.temporalState
        } else {
            cache.state
        }

    guard state.count == 2 else {
        return nil
    }
    return (state[0], state[1])
}

private func diffusionGemmaDecoderMask(
    batch: Int,
    canvasLength: Int,
    cache: KVCache?,
    windowSize: Int?
) -> MLXFast.ScaledDotProductAttentionMaskMode {
    guard let (keys, _) = diffusionGemmaCacheState(cache) else {
        return .none
    }

    let encoderLength = keys.dim(2)
    let validEncoderLength = Swift.min(cache?.offset ?? encoderLength, encoderLength)
    let keyLength = encoderLength + canvasLength

    if let windowSize {
        let windowPrefix = Swift.max(windowSize - 1, 0)
        if encoderLength == validEncoderLength && encoderLength <= windowPrefix {
            return .none
        }

        let start = Swift.max(0, validEncoderLength - windowPrefix)
        let positions = MLXArray(Int32(0) ..< Int32(encoderLength))
        let encoderMask = (positions .>= start) & (positions .< validEncoderLength)
        let canvasMask = MLXArray.ones([canvasLength], type: Bool.self)
        let row = concatenated([encoderMask, canvasMask], axis: 0)
        return .array(
            broadcast(
                row[.newAxis, .newAxis, .newAxis, 0...],
                to: [batch, 1, canvasLength, keyLength]))
    }

    if encoderLength == validEncoderLength {
        return .none
    }

    let positions = MLXArray(Int32(0) ..< Int32(encoderLength))
    let encoderMask = positions .< validEncoderLength
    let canvasMask = MLXArray.ones([canvasLength], type: Bool.self)
    let row = concatenated([encoderMask, canvasMask], axis: 0)
    return .array(
        broadcast(
            row[.newAxis, .newAxis, .newAxis, 0...],
            to: [batch, 1, canvasLength, keyLength]))
}

private func diffusionGemmaEncoderMask(
    h: MLXArray,
    cache: KVCache?,
    windowSize: Int? = nil,
    multimodalTokenTypes: MLXArray? = nil
) -> MLXFast.ScaledDotProductAttentionMaskMode {
    guard let multimodalTokenTypes else {
        return createAttentionMask(h: h, cache: cache, windowSize: windowSize)
    }

    let offset = cache?.offset ?? 0
    let length = h.dim(1)
    let keyLength = offset + length
    if length <= 1 && keyLength <= 1 {
        return .none
    }

    let rawTypes =
        multimodalTokenTypes.ndim == 2
        ? multimodalTokenTypes[0, 0...].asArray(Int32.self)
        : multimodalTokenTypes.asArray(Int32.self)
    let types = Array(rawTypes.suffix(length))
    var visualBlockIds = Array(repeating: -1, count: length)
    var currentBlock = -1
    var previousType: Int32 = 0
    for idx in 0 ..< length {
        let type = types[idx]
        if type == 0 {
            previousType = 0
            continue
        }
        if idx == 0 || type != previousType {
            currentBlock += 1
        }
        visualBlockIds[idx] = currentBlock
        previousType = type
    }

    var values = [Bool]()
    values.reserveCapacity(length * keyLength)
    for query in 0 ..< length {
        let absoluteQuery = offset + query
        for key in 0 ..< keyLength {
            let causal = absoluteQuery >= key
            let windowed =
                if let windowSize {
                    causal && absoluteQuery < key + windowSize
                } else {
                    causal
                }
            let localKey = key - offset
            let sameVisualBlock =
                localKey >= 0 && localKey < length
                && visualBlockIds[query] >= 0
                && visualBlockIds[query] == visualBlockIds[localKey]
            values.append(windowed || sameVisualBlock)
        }
    }

    let mask = MLXArray(values, [length, keyLength])
    return .array(mask[.newAxis, .newAxis, 0..., 0...])
}

private final class DiffusionGemmaAttention: Module {
    let config: DiffusionGemmaTextConfiguration
    let layerIdx: Int
    let layerType: String
    let isSliding: Bool
    let headDim: Int
    let numHeads: Int
    let numKVHeads: Int

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear?
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm
    @ModuleInfo(key: "v_norm") var vNorm: DiffusionGemmaRMSNormNoScale
    @ModuleInfo var rope: RoPELayer

    init(_ config: DiffusionGemmaTextConfiguration, layerIdx: Int) {
        self.config = config
        self.layerIdx = layerIdx
        self.layerType = config.layerTypes[layerIdx]
        self.isSliding = layerType == "sliding_attention"
        self.headDim = isSliding ? config.headDim : config.globalHeadDim
        self.numHeads = config.attentionHeads
        self.numKVHeads = isSliding ? config.kvHeads : config.globalKVHeads

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

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?,
        mode: DiffusionGemmaAttentionMode
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

        var attentionMask = mask
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
            if let (cachedKeys, cachedValues) = diffusionGemmaCacheState(cache) {
                var encoderKeys = cachedKeys
                var encoderValues = cachedValues

                if isSliding {
                    let windowPrefix = Swift.max(config.slidingWindow - 1, 0)
                    let encoderLength = encoderKeys.dim(2)
                    if windowPrefix > 0 && encoderLength > windowPrefix {
                        let start = encoderLength - windowPrefix
                        encoderKeys = encoderKeys[0..., 0..., start..., 0...]
                        encoderValues = encoderValues[0..., 0..., start..., 0...]

                        if case .array(let maskArray) = attentionMask {
                            let keptLength = Swift.min(maskArray.dim(-1), windowPrefix + length)
                            attentionMask = .array(
                                maskArray[.ellipsis, (maskArray.dim(-1) - keptLength)...])
                        }
                    }
                }

                finalKeys = concatenated([encoderKeys, keys], axis: 2)
                finalValues = concatenated([encoderValues, values], axis: 2)
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
            mask: attentionMask
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
        layerIdx: Int
    ) {
        precondition(
            config.rmsNormEps == diffusionGemmaRMSEps,
            "DiffusionGemma fused path requires rmsNormEps == \(diffusionGemmaRMSEps), got \(config.rmsNormEps)"
        )

        self.layerType = config.layerTypes[layerIdx]
        _attention.wrappedValue = DiffusionGemmaAttention(config, layerIdx: layerIdx)
        _mlp.wrappedValue = DiffusionGemmaMLP(config)
        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
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
        cache: KVCache?,
        mode: DiffusionGemmaAttentionMode,
        layerScalar: MLXArray? = nil
    ) -> MLXArray {
        var residual = x
        var h = inputLayerNorm(x)
        h = attention(h, mask: mask, cache: cache, mode: mode)
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
        return h * (layerScalar ?? self.layerScalar).asType(h.dtype)
    }
}

private final class DiffusionGemmaEncoderLayerScalar: Module {
    @ParameterInfo(key: "layer_scalar") var layerScalar: MLXArray

    override init() {
        _layerScalar.wrappedValue = MLXArray.ones([1])
        super.init()
    }
}

private final class DiffusionGemmaEncoderLanguageModel: Module {
    @ModuleInfo var layers: [DiffusionGemmaEncoderLayerScalar]

    init(_ config: DiffusionGemmaTextConfiguration) {
        _layers.wrappedValue = (0 ..< config.hiddenLayers).map { _ in
            DiffusionGemmaEncoderLayerScalar()
        }
        super.init()
    }
}

private final class DiffusionGemmaEncoder: Module {
    let config: DiffusionGemmaTextConfiguration

    @ModuleInfo(key: "language_model") var languageModel: DiffusionGemmaEncoderLanguageModel

    init(_ config: DiffusionGemmaTextConfiguration) {
        self.config = config
        _languageModel.wrappedValue = DiffusionGemmaEncoderLanguageModel(config)
        super.init()
    }

    func callAsFunction(
        inputs: MLXArray? = nil,
        inputEmbeddings: MLXArray? = nil,
        decoder: DiffusionGemmaDecoder,
        cache: [KVCache]? = nil,
        multimodalTokenTypes: MLXArray? = nil
    ) -> MLXArray {
        var h: MLXArray
        if let inputEmbeddings {
            h = inputEmbeddings
        } else if let inputs {
            h = decoder.embeddings(inputs)
        } else {
            fatalError("DiffusionGemma requires token ids or input embeddings")
        }

        for (idx, layer) in decoder.layers.enumerated() {
            let layerCache = cache?[idx]
            let mask: MLXFast.ScaledDotProductAttentionMaskMode
            if layer.layerType == "sliding_attention" {
                mask = diffusionGemmaEncoderMask(
                    h: h,
                    cache: layerCache,
                    windowSize: config.slidingWindow,
                    multimodalTokenTypes: multimodalTokenTypes)
            } else {
                mask = diffusionGemmaEncoderMask(
                    h: h,
                    cache: layerCache,
                    multimodalTokenTypes: multimodalTokenTypes)
            }
            h = layer(
                h,
                mask: mask,
                cache: layerCache,
                mode: .encoder,
                layerScalar: languageModel.layers[idx].layerScalar)
        }

        return decoder.norm(h)
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
            DiffusionGemmaLayer(config, layerIdx: $0)
        }
        _norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _selfConditioning.wrappedValue = DiffusionGemmaSelfConditioning(config)
        super.init()
    }

    func embeddings(_ tokens: MLXArray) -> MLXArray {
        let e = embedTokens(tokens)
        return (e * MLXArray(embedScale, dtype: .float32)).asType(e.dtype)
    }

    func selfConditioningWeight() -> MLXArray {
        if let quantizedEmbedding = embedTokens as? QuantizedEmbedding {
            return dequantized(
                quantizedEmbedding.weight,
                scales: quantizedEmbedding.scales,
                biases: quantizedEmbedding.biases,
                groupSize: quantizedEmbedding.groupSize,
                bits: quantizedEmbedding.bits,
                mode: quantizedEmbedding.mode)
        }
        return embedTokens.weight
    }

    func selfConditioningEmbeddings(logits: MLXArray, weight: MLXArray) -> MLXArray {
        let probabilities = softmax(logits.asType(.float32), axis: -1, precise: true)
        return matmul(probabilities.asType(weight.dtype), weight).asType(weight.dtype)
            * MLXArray(embedScale, dtype: .float32).asType(weight.dtype)
    }

    func callAsFunction(
        canvasTokens: MLXArray,
        cache: [KVCache],
        selfConditioningLogits: MLXArray?,
        selfConditioningEmbeddings: MLXArray? = nil
    ) -> MLXArray {
        var h = embeddings(canvasTokens)

        let signal: MLXArray
        if let selfConditioningEmbeddings {
            signal = selfConditioningEmbeddings.asType(h.dtype)
        } else if let selfConditioningLogits {
            let probabilities = softmax(selfConditioningLogits, axis: -1, precise: true)
            let projected: MLXArray
            if let quantizedEmbedding = embedTokens as? QuantizedEmbedding {
                projected = quantizedMM(
                    probabilities.asType(h.dtype),
                    quantizedEmbedding.weight,
                    scales: quantizedEmbedding.scales,
                    biases: quantizedEmbedding.biases,
                    transpose: false,
                    groupSize: quantizedEmbedding.groupSize,
                    bits: quantizedEmbedding.bits,
                    mode: quantizedEmbedding.mode)
            } else {
                projected = matmul(
                    probabilities.asType(embedTokens.weight.dtype), embedTokens.weight)
            }
            signal = projected * MLXArray(embedScale, dtype: .float32).asType(h.dtype)
        } else {
            signal = MLXArray.zeros(h.shape, dtype: h.dtype)
        }
        h = selfConditioning(h, signal: signal)

        let fullMask = diffusionGemmaDecoderMask(
            batch: h.dim(0),
            canvasLength: h.dim(1),
            cache: zip(layers, cache).first { layer, _ in
                layer.layerType == "full_attention"
            }?.1,
            windowSize: nil)
        let slidingMask = diffusionGemmaDecoderMask(
            batch: h.dim(0),
            canvasLength: h.dim(1),
            cache: zip(layers, cache).first { layer, _ in
                layer.layerType == "sliding_attention"
            }?.1,
            windowSize: config.slidingWindow)

        for (idx, layer) in layers.enumerated() {
            let mask = layer.layerType == "sliding_attention" ? slidingMask : fullMask
            h = layer(h, mask: mask, cache: cache[idx], mode: .decoder)
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

public final class DiffusionGemmaLanguageCore: Module, BlockDiffusionLanguageModel,
    KVCacheDimensionProvider
{
    public let vocabularySize: Int
    public var diffusionVocabularySize: Int { vocabularySize }
    public let diffusionCanvasLength: Int
    public let diffusionMaxDenoisingSteps: Int
    public let diffusionEntropyBound: Float
    public let diffusionTemperatureMin: Float
    public let diffusionTemperatureMax: Float
    public let diffusionStabilityThreshold: Int
    public let diffusionConfidenceThreshold: Float
    public let diffusionDefaultMaxTokens: Int?
    public let kvHeads: [Int]

    let config: DiffusionGemmaConfiguration
    @ModuleInfo fileprivate var model: DiffusionGemmaCore
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ config: DiffusionGemmaConfiguration) {
        self.config = config
        self.vocabularySize = config.textConfig.vocabularySize
        self.diffusionCanvasLength = config.canvasLength
        self.diffusionMaxDenoisingSteps = config.generationConfig.maxDenoisingSteps
        self.diffusionEntropyBound = config.generationConfig.samplerConfig.entropyBound
        self.diffusionTemperatureMin = config.generationConfig.temperatureMin
        self.diffusionTemperatureMax = config.generationConfig.temperatureMax
        self.diffusionStabilityThreshold = config.generationConfig.stabilityThreshold
        self.diffusionConfidenceThreshold = config.generationConfig.confidenceThreshold
        self.diffusionDefaultMaxTokens = config.generationConfig.maxNewTokens
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

    public func inputEmbeddings(inputIds: MLXArray) -> MLXArray {
        model.decoder.embeddings(
            inputIds.ndim == 1 ? inputIds.expandedDimensions(axis: 0) : inputIds)
    }

    private func encodeIntoCache(
        _ tokens: MLXArray,
        cache: [KVCache],
        windowSize: Int?,
        multimodalTokenTypes: MLXArray? = nil
    ) {
        let tokens = tokens.ndim == 1 ? tokens.expandedDimensions(axis: 0) : tokens
        let chunkSize = windowSize ?? 512
        var start = 0

        while start < tokens.dim(1) {
            let end = Swift.min(start + chunkSize, tokens.dim(1))
            _ = model.encoder(
                inputs: tokens[0..., start ..< end],
                decoder: model.decoder,
                cache: cache,
                multimodalTokenTypes: multimodalTokenTypes?[0..., start ..< end])
            asyncEval(cache)
            start = end
        }

        eval(cache)
    }

    private func encodeEmbeddingsIntoCache(
        _ inputEmbeddings: MLXArray,
        multimodalTokenTypes: MLXArray?,
        cache: [KVCache]
    ) {
        let inputEmbeddings =
            inputEmbeddings.ndim == 2
            ? inputEmbeddings.expandedDimensions(axis: 0) : inputEmbeddings
        _ = model.encoder(
            inputEmbeddings: inputEmbeddings,
            decoder: model.decoder,
            cache: cache,
            multimodalTokenTypes: multimodalTokenTypes)
        eval(cache)
    }

    public func prepareDiffusion(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws {
        if input.image != nil || input.video != nil || input.audio != nil {
            throw GenerateError.unsupportedMultimodalGeneration(String(describing: Self.self))
        }
        encodeIntoCache(input.text.tokens, cache: cache, windowSize: windowSize)
    }

    public func prepareDiffusion(
        inputEmbeddings: MLXArray,
        multimodalTokenTypes: MLXArray?,
        cache: [KVCache],
        windowSize: Int?
    ) {
        if multimodalTokenTypes == nil {
            let chunkSize = windowSize ?? 512
            if inputEmbeddings.dim(1) > chunkSize {
                var start = 0
                while start < inputEmbeddings.dim(1) {
                    let end = Swift.min(start + chunkSize, inputEmbeddings.dim(1))
                    encodeEmbeddingsIntoCache(
                        inputEmbeddings[0..., start ..< end, 0...],
                        multimodalTokenTypes: nil,
                        cache: cache)
                    start = end
                }
                return
            }
        }
        encodeEmbeddingsIntoCache(
            inputEmbeddings,
            multimodalTokenTypes: multimodalTokenTypes,
            cache: cache)
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
        return projectLogits(hidden)
    }

    public func diffusionLogits(
        canvasTokens: MLXArray,
        cache: [KVCache],
        selfConditioningEmbeddings: MLXArray?
    ) -> MLXArray {
        let hidden = model.decoder(
            canvasTokens: canvasTokens.ndim == 1
                ? canvasTokens.expandedDimensions(axis: 0) : canvasTokens,
            cache: cache,
            selfConditioningLogits: nil,
            selfConditioningEmbeddings: selfConditioningEmbeddings)
        return projectLogits(hidden)
    }

    public func diffusionSelfConditioningWeight() -> MLXArray? {
        model.decoder.selfConditioningWeight()
    }

    public func diffusionSelfConditioningEmbeddings(logits: MLXArray, weight: MLXArray?) -> MLXArray
    {
        guard let weight else {
            return logits
        }
        return model.decoder.selfConditioningEmbeddings(logits: logits, weight: weight)
    }

    private func projectLogits(_ hidden: MLXArray) -> MLXArray {
        var logits: MLXArray
        if let lmHead {
            logits = lmHead(hidden)
        } else {
            logits = model.decoder.embedTokens.asLinear(hidden)
        }
        logits = logits.asType(.float32)
        logits =
            tanh(logits / config.textConfig.finalLogitSoftcapping)
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
            "DiffusionGemmaLanguageCore does not produce autoregressive next-token logits. Use prepareDiffusion(_:cache:windowSize:) and diffusionLogits(canvasTokens:cache:selfConditioningLogits:) through BlockDiffusionTokenIterator."
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
                && !(key.hasPrefix("model.encoder.language_model.")
                    && !key.hasSuffix(".layer_scalar"))
                && !key.contains("rotary_emb")
                && !key.contains("input_min")
                && !key.contains("input_max")
                && !key.contains("output_min")
                && !key.contains("output_max")
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

extension DiffusionGemmaLanguageCore: LoRAModel {
    public var loraLayers: [Module] {
        model.decoder.layers.map { $0.attention }
    }
}
