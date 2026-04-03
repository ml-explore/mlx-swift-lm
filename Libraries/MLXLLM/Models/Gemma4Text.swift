//
//  Gemma4Text.swift
//  mlx-swift-lm
//

// Based on https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/gemma4.py

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Configuration

public struct Gemma4TextConfiguration: Codable {
    let modelType: String
    let hiddenSize: Int
    let numHiddenLayers: Int
    let intermediateSize: Int
    let numAttentionHeads: Int
    let numKeyValueHeads: Int
    let headDim: Int
    let globalHeadDim: Int
    let rmsNormEps: Float
    let vocabSize: Int
    let hiddenActivation: String
    let slidingWindow: Int
    let maxPositionEmbeddings: Int
    let finalLogitSoftcapping: Float
    let layerTypes: [String]
    let ropeParameters: [String: RopeParameterSet]
    let numKvSharedLayers: Int
    let hiddenSizePerLayerInput: Int
    let vocabSizePerLayerInput: Int
    let useDoubleWideMlp: Bool
    let attentionBias: Bool

    struct RopeParameterSet: Codable {
        let ropeTheta: Float?
        let ropeType: String?
        let partialRotaryFactor: Float?

        enum CodingKeys: String, CodingKey {
            case ropeTheta = "rope_theta"
            case ropeType = "rope_type"
            case partialRotaryFactor = "partial_rotary_factor"
        }
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case globalHeadDim = "global_head_dim"
        case rmsNormEps = "rms_norm_eps"
        case vocabSize = "vocab_size"
        case hiddenActivation = "hidden_activation"
        case slidingWindow = "sliding_window"
        case maxPositionEmbeddings = "max_position_embeddings"
        case finalLogitSoftcapping = "final_logit_softcapping"
        case layerTypes = "layer_types"
        case ropeParameters = "rope_parameters"
        case numKvSharedLayers = "num_kv_shared_layers"
        case hiddenSizePerLayerInput = "hidden_size_per_layer_input"
        case vocabSizePerLayerInput = "vocab_size_per_layer_input"
        case useDoubleWideMlp = "use_double_wide_mlp"
        case attentionBias = "attention_bias"
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
        intermediateSize = try container.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 6144
        numAttentionHeads = try container.decodeIfPresent(Int.self, forKey: .numAttentionHeads) ?? 8
        numKeyValueHeads = try container.decodeIfPresent(Int.self, forKey: .numKeyValueHeads) ?? 1
        headDim = try container.decodeIfPresent(Int.self, forKey: .headDim) ?? 256
        globalHeadDim = try container.decodeIfPresent(Int.self, forKey: .globalHeadDim) ?? 512
        rmsNormEps = try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        vocabSize = try container.decodeIfPresent(Int.self, forKey: .vocabSize) ?? 262144
        hiddenActivation =
            try container.decodeIfPresent(String.self, forKey: .hiddenActivation)
            ?? "gelu_pytorch_tanh"
        slidingWindow = try container.decodeIfPresent(Int.self, forKey: .slidingWindow) ?? 512
        maxPositionEmbeddings =
            try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 131072
        finalLogitSoftcapping =
            try container.decodeIfPresent(Float.self, forKey: .finalLogitSoftcapping) ?? 30.0
        layerTypes = try container.decode([String].self, forKey: .layerTypes)
        ropeParameters = try container.decode(
            [String: RopeParameterSet].self, forKey: .ropeParameters)
        numKvSharedLayers =
            try container.decodeIfPresent(Int.self, forKey: .numKvSharedLayers) ?? 0
        hiddenSizePerLayerInput =
            try container.decodeIfPresent(Int.self, forKey: .hiddenSizePerLayerInput) ?? 0
        vocabSizePerLayerInput =
            try container.decodeIfPresent(Int.self, forKey: .vocabSizePerLayerInput) ?? 262144
        useDoubleWideMlp =
            try container.decodeIfPresent(Bool.self, forKey: .useDoubleWideMlp) ?? true
        attentionBias = try container.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
    }
}

// MARK: - RMSNorm (Gemma 4 style — plain weight, no +1 offset)

class Gemma4RMSNorm: Module, UnaryLayer {
    let weight: MLXArray
    let eps: Float

    init(dimensions: Int, eps: Float = 1e-6) {
        self.weight = MLXArray.ones([dimensions])
        self.eps = eps
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(x, weight: self.weight, eps: self.eps)
    }
}

// MARK: - RMSNoScale (for v_norm — no learned scale)

class Gemma4RMSNoScale: Module {
    let eps: Float

    init(eps: Float = 1e-6) {
        self.eps = eps
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(x, weight: MLXArray.mlxNone, eps: eps)
    }
}

// MARK: - ProportionalRoPE

class ProportionalRoPE: Module {
    let headDim: Int
    let rotaryDims: Int
    let _freqs: MLXArray

    init(headDim: Int, partialRotaryFactor: Float, base: Float) {
        self.headDim = headDim
        self.rotaryDims = Int(partialRotaryFactor * Float(headDim))

        let ropeAngles = rotaryDims / 2
        let indices = MLXArray(stride(from: 0, to: 2 * ropeAngles, by: 2)).asType(.float32)
        self._freqs = 1.0 / MLX.pow(base, indices / Float(headDim))
        super.init()
    }

    func callAsFunction(_ x: MLXArray, offset: Int) -> MLXArray {
        let rot = x[.ellipsis, 0 ..< rotaryDims]
        let passthrough = x[.ellipsis, rotaryDims...]

        let rotated = MLXFast.RoPE(
            rot,
            dimensions: rotaryDims,
            traditional: false,
            base: nil,
            scale: 1.0,
            offset: offset,
            freqs: _freqs
        )
        return concatenated([rotated, passthrough], axis: -1)
    }
}

// MARK: - Attention

class Gemma4Attention: Module {
    let isSliding: Bool
    let headDim: Int
    let nHeads: Int
    let nKVHeads: Int
    let scale: Float
    let isKvSharedLayer: Bool

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: Gemma4RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: Gemma4RMSNorm
    @ModuleInfo(key: "v_norm") var vNorm: Gemma4RMSNoScale

    // RoPE — either standard RoPE or ProportionalRoPE
    private let slidingRope: RoPE?
    private let proportionalRope: ProportionalRoPE?
    private let globalStandardRope: RoPE?

    init(_ config: Gemma4TextConfiguration, layerIdx: Int) {
        self.isSliding = config.layerTypes[layerIdx] == "sliding_attention"
        self.headDim = isSliding ? config.headDim : config.globalHeadDim
        self.nHeads = config.numAttentionHeads
        self.nKVHeads = config.numKeyValueHeads
        self.scale = 1.0  // QK-norm replaces 1/sqrt(d) scaling

        let firstKvSharedIdx = config.numHiddenLayers - config.numKvSharedLayers
        self.isKvSharedLayer = config.numKvSharedLayers > 0 && layerIdx >= firstKvSharedIdx

        self._qProj.wrappedValue = Linear(
            config.hiddenSize, nHeads * headDim, bias: config.attentionBias)
        self._kProj.wrappedValue = Linear(
            config.hiddenSize, nKVHeads * headDim, bias: config.attentionBias)
        self._vProj.wrappedValue = Linear(
            config.hiddenSize, nKVHeads * headDim, bias: config.attentionBias)
        self._oProj.wrappedValue = Linear(
            nHeads * headDim, config.hiddenSize, bias: config.attentionBias)

        self._qNorm.wrappedValue = Gemma4RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        self._kNorm.wrappedValue = Gemma4RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        self._vNorm.wrappedValue = Gemma4RMSNoScale(eps: config.rmsNormEps)

        if isSliding {
            let ropeParams = config.ropeParameters["sliding_attention"]
            self.slidingRope = RoPE(
                dimensions: headDim,
                traditional: false,
                base: ropeParams?.ropeTheta ?? 10000.0
            )
            self.proportionalRope = nil
            self.globalStandardRope = nil
        } else {
            let ropeParams = config.ropeParameters["full_attention"]
            let partialFactor = ropeParams?.partialRotaryFactor ?? 1.0
            let ropeTheta = ropeParams?.ropeTheta ?? 1_000_000.0

            if partialFactor < 1.0 {
                self.proportionalRope = ProportionalRoPE(
                    headDim: headDim, partialRotaryFactor: partialFactor, base: ropeTheta)
                self.slidingRope = nil
                self.globalStandardRope = nil
            } else {
                self.globalStandardRope = RoPE(
                    dimensions: headDim, traditional: false, base: ropeTheta)
                self.proportionalRope = nil
                self.slidingRope = nil
            }
        }

        super.init()
    }

    private func applyRope(_ x: MLXArray, offset: Int) -> MLXArray {
        if let rope = slidingRope {
            return rope(x, offset: offset)
        } else if let rope = proportionalRope {
            return rope(x, offset: offset)
        } else if let rope = globalStandardRope {
            return rope(x, offset: offset)
        }
        return x
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache? = nil
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        var queries = qProj(x)
        queries = queries.reshaped(B, L, nHeads, headDim).transposed(0, 2, 1, 3)
        queries = qNorm(queries)

        // Capture offset before cache update
        let offset = cache?.offset ?? 0

        var keys: MLXArray
        var values: MLXArray

        if isKvSharedLayer, let cache = cache, cache.offset > 0 {
            let state = cache.state
            keys = state[0]
            values = state[1]
        } else {
            keys = kProj(x)
            keys = keys.reshaped(B, L, nKVHeads, headDim).transposed(0, 2, 1, 3)
            keys = kNorm(keys)

            values = vProj(x)
            values = values.reshaped(B, L, nKVHeads, headDim).transposed(0, 2, 1, 3)
            values = vNorm(values)

            keys = applyRope(keys, offset: offset)

            if let cache = cache, !isKvSharedLayer {
                (keys, values) = cache.update(keys: keys, values: values)
            }
        }

        queries = applyRope(queries, offset: offset)

        let output = MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: values, scale: scale, mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        return oProj(output)
    }
}

// MARK: - MLP

class Gemma4MLP: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(_ config: Gemma4TextConfiguration, layerIdx: Int) {
        let firstKvSharedIdx = config.numHiddenLayers - config.numKvSharedLayers
        let isKvShared = config.numKvSharedLayers > 0 && layerIdx >= firstKvSharedIdx
        let useDouble = config.useDoubleWideMlp && isKvShared
        let intermediate = config.intermediateSize * (useDouble ? 2 : 1)

        self._gateProj.wrappedValue = Linear(config.hiddenSize, intermediate, bias: false)
        self._upProj.wrappedValue = Linear(config.hiddenSize, intermediate, bias: false)
        self._downProj.wrappedValue = Linear(intermediate, config.hiddenSize, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(geluApproximate(gateProj(x)) * upProj(x))
    }
}

// MARK: - Decoder Layer

class Gemma4DecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: Gemma4Attention
    @ModuleInfo var mlp: Gemma4MLP
    @ModuleInfo(key: "input_layernorm") var inputLayernorm: Gemma4RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayernorm: Gemma4RMSNorm
    @ModuleInfo(key: "pre_feedforward_layernorm") var preFeedforwardLayernorm: Gemma4RMSNorm
    @ModuleInfo(key: "post_feedforward_layernorm") var postFeedforwardLayernorm: Gemma4RMSNorm

    let hiddenSizePerLayerInput: Int
    @ModuleInfo(key: "per_layer_input_gate") var perLayerInputGate: Linear?
    @ModuleInfo(key: "per_layer_projection") var perLayerProjection: Linear?
    @ModuleInfo(key: "post_per_layer_input_norm") var postPerLayerInputNorm: Gemma4RMSNorm?

    @ModuleInfo(key: "layer_scalar") var layerScalar: MLXArray

    init(_ config: Gemma4TextConfiguration, layerIdx: Int) {
        self._selfAttn.wrappedValue = Gemma4Attention(config, layerIdx: layerIdx)
        self._mlp.wrappedValue = Gemma4MLP(config, layerIdx: layerIdx)

        self._inputLayernorm.wrappedValue = Gemma4RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postAttentionLayernorm.wrappedValue = Gemma4RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._preFeedforwardLayernorm.wrappedValue = Gemma4RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postFeedforwardLayernorm.wrappedValue = Gemma4RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)

        self.hiddenSizePerLayerInput = config.hiddenSizePerLayerInput
        if hiddenSizePerLayerInput > 0 {
            self._perLayerInputGate.wrappedValue = Linear(
                config.hiddenSize, config.hiddenSizePerLayerInput, bias: false)
            self._perLayerProjection.wrappedValue = Linear(
                config.hiddenSizePerLayerInput, config.hiddenSize, bias: false)
            self._postPerLayerInputNorm.wrappedValue = Gemma4RMSNorm(
                dimensions: config.hiddenSize, eps: config.rmsNormEps)
        } else {
            self._perLayerInputGate.wrappedValue = nil
            self._perLayerProjection.wrappedValue = nil
            self._postPerLayerInputNorm.wrappedValue = nil
        }

        self._layerScalar.wrappedValue = MLXArray.ones([1])

        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache? = nil,
        perLayerInput: MLXArray? = nil
    ) -> MLXArray {
        // Self-attention
        var residual = x
        var h = inputLayernorm(x)
        h = selfAttn(h, mask: mask, cache: cache)
        h = postAttentionLayernorm(h)
        var out = residual + h

        // FFN
        residual = out
        h = preFeedforwardLayernorm(out)
        h = mlp(h)
        h = postFeedforwardLayernorm(h)
        out = residual + h

        // Per-Layer Embedding injection
        if hiddenSizePerLayerInput > 0,
            let perLayerInput = perLayerInput,
            let gate = perLayerInputGate,
            let proj = perLayerProjection,
            let norm = postPerLayerInputNorm
        {
            residual = out
            h = gate(out)
            h = geluApproximate(h)
            h = h * perLayerInput
            h = proj(h)
            h = norm(h)
            out = residual + h
        }

        out = out * layerScalar
        return out
    }
}

// MARK: - Language Model

class Gemma4LanguageModel: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo var layers: [Gemma4DecoderLayer]
    @ModuleInfo var norm: Gemma4RMSNorm

    // PLE components
    @ModuleInfo(key: "embed_tokens_per_layer") var embedTokensPerLayer: [Embedding]?
    @ModuleInfo(key: "per_layer_model_projection") var perLayerModelProjection: Linear?
    @ModuleInfo(key: "per_layer_projection_norm") var perLayerProjectionNorm: Gemma4RMSNorm?

    let config: Gemma4TextConfiguration
    let firstKvSharedLayerIdx: Int
    let layerIdxToCacheIdx: [Int]
    let firstSlidingIdx: Int
    let firstFullIdx: Int

    init(_ config: Gemma4TextConfiguration) {
        self.config = config

        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabSize, dimensions: config.hiddenSize)

        // PLE embeddings (split per-layer to stay under Metal 4GB buffer limit)
        if config.hiddenSizePerLayerInput > 0 {
            self._embedTokensPerLayer.wrappedValue = (0 ..< config.numHiddenLayers).map { _ in
                Embedding(
                    embeddingCount: config.vocabSizePerLayerInput,
                    dimensions: config.hiddenSizePerLayerInput)
            }
            self._perLayerModelProjection.wrappedValue = Linear(
                config.hiddenSize,
                config.numHiddenLayers * config.hiddenSizePerLayerInput,
                bias: false)
            self._perLayerProjectionNorm.wrappedValue = Gemma4RMSNorm(
                dimensions: config.hiddenSizePerLayerInput, eps: config.rmsNormEps)
        } else {
            self._embedTokensPerLayer.wrappedValue = nil
            self._perLayerModelProjection.wrappedValue = nil
            self._perLayerProjectionNorm.wrappedValue = nil
        }

        self._layers.wrappedValue = (0 ..< config.numHiddenLayers).map { i in
            Gemma4DecoderLayer(config, layerIdx: i)
        }

        self._norm.wrappedValue = Gemma4RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)

        // KV cache index mapping
        self.firstKvSharedLayerIdx =
            config.numKvSharedLayers > 0
            ? config.numHiddenLayers - config.numKvSharedLayers
            : config.numHiddenLayers

        let concreteTypes = Array(config.layerTypes.prefix(firstKvSharedLayerIdx))
        var mapping = [Int]()
        for (i, lt) in config.layerTypes.enumerated() {
            if i < firstKvSharedLayerIdx {
                mapping.append(i)
            } else {
                let idx =
                    concreteTypes.count - 1
                    - concreteTypes.reversed().firstIndex(of: lt)!
                mapping.append(idx)
            }
        }
        self.layerIdxToCacheIdx = mapping

        self.firstSlidingIdx = config.layerTypes.firstIndex(of: "sliding_attention") ?? 0
        self.firstFullIdx = config.layerTypes.firstIndex(of: "full_attention") ?? 0

        super.init()
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var h = embedTokens(inputs)
        let scale = MLXArray(sqrt(Float(config.hiddenSize)), dtype: .bfloat16)
        h = h * scale.asType(h.dtype)

        // Compute per-layer inputs
        var perLayerInputs: MLXArray? = nil
        if config.hiddenSizePerLayerInput > 0 {
            perLayerInputs = getPerLayerInputs(inputs)
            perLayerInputs = projectPerLayerInputs(h, perLayerInputs: perLayerInputs!)
        }

        let globalMask = createAttentionMask(h: h, cache: cache?[firstFullIdx])
        let slidingMask = createAttentionMask(
            h: h, cache: cache?[firstSlidingIdx], windowSize: config.slidingWindow)

        for (i, layer) in layers.enumerated() {
            let isGlobal = config.layerTypes[i] == "full_attention"
            let mask = isGlobal ? globalMask : slidingMask

            let pli: MLXArray? =
                if let perLayerInputs = perLayerInputs {
                    perLayerInputs[0..., 0..., i, 0...]
                } else {
                    nil
                }

            h = layer(h, mask: mask, cache: cache?[layerIdxToCacheIdx[i]], perLayerInput: pli)
        }

        h = norm(h)

        // Tied embeddings for output projection
        let out = embedTokens.asLinear(h)
        return tanh(out / config.finalLogitSoftcapping) * config.finalLogitSoftcapping
    }

    private func getPerLayerInputs(_ inputIds: MLXArray) -> MLXArray {
        guard let embeddings = embedTokensPerLayer else { fatalError("PLE not initialized") }
        let mask = inputIds .< config.vocabSizePerLayerInput
        let tokens = MLX.where(mask, inputIds, MLXArray.zeros(like: inputIds))
        let scale = sqrt(Float(config.hiddenSizePerLayerInput))
        let chunks = embeddings.map { emb in emb(tokens) * scale }
        return stacked(chunks, axis: -2)
    }

    private func projectPerLayerInputs(_ inputsEmbeds: MLXArray, perLayerInputs: MLXArray)
        -> MLXArray
    {
        guard let projection = perLayerModelProjection,
            let norm = perLayerProjectionNorm
        else { fatalError("PLE projection not initialized") }

        var proj = projection(inputsEmbeds)
        proj = proj * pow(Float(config.hiddenSize), -0.5)

        let shape =
            Array(inputsEmbeds.shape.dropLast())
            + [config.numHiddenLayers, config.hiddenSizePerLayerInput]
        proj = proj.reshaped(shape)
        proj = norm(proj)

        return (proj + perLayerInputs) * pow(2.0, -0.5)
    }
}

// MARK: - Top-level Model

public class Gemma4TextModel: Module, LLMModel {
    @ModuleInfo(key: "language_model") var languageModel: Gemma4LanguageModel

    public let config: Gemma4TextConfiguration
    public var vocabularySize: Int { config.vocabSize }

    public init(_ config: Gemma4TextConfiguration) {
        self.config = config
        self._languageModel.wrappedValue = Gemma4LanguageModel(config)
        super.init()
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        languageModel(inputs, cache: cache)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var processed = weights

        // Handle VLM-style weights under language_model key
        let unflattened = ModuleParameters.unflattened(weights)
        if let modelWeights = unflattened["model"] {
            let modelFlat = Dictionary(uniqueKeysWithValues: modelWeights.flattened())
            if let lmWeights = ModuleParameters.unflattened(modelFlat)["language_model"] {
                processed = Dictionary(
                    uniqueKeysWithValues: lmWeights.flattened().map {
                        ("language_model.\($0.0)", $0.1)
                    })
            }
        } else if !weights.keys.contains(where: { $0.hasPrefix("language_model.") }) {
            // Keys without prefix — add language_model. prefix
            processed = Dictionary(
                uniqueKeysWithValues: weights.map {
                    ("language_model.\($0.key)", $0.value)
                })
        }

        // Split embed_tokens_per_layer if it's a single large tensor
        let pleKey = "language_model.embed_tokens_per_layer.weight"
        if let bigWeight = processed[pleKey] {
            processed.removeValue(forKey: pleKey)
            let pleDim = config.hiddenSizePerLayerInput
            for i in 0 ..< config.numHiddenLayers {
                let chunk = bigWeight[0..., (i * pleDim) ..< ((i + 1) * pleDim)]
                processed["language_model.embed_tokens_per_layer.\(i).weight"] = chunk
            }
        }

        return processed
    }

    public func newCache(parameters: GenerateParameters? = nil) -> [KVCache] {
        let firstKvSharedIdx =
            config.numKvSharedLayers > 0
            ? config.numHiddenLayers - config.numKvSharedLayers
            : config.numHiddenLayers

        var caches = [KVCache]()
        for lt in config.layerTypes.prefix(firstKvSharedIdx) {
            if lt == "full_attention" {
                let cache = StandardKVCache()
                cache.step = 1024
                caches.append(cache)
            } else {
                caches.append(RotatingKVCache(maxSize: config.slidingWindow, keep: 0))
            }
        }
        return caches
    }

    public func prepare(
        _ input: LMInput, cache: [KVCache], windowSize: Int? = nil
    ) throws -> PrepareResult {
        .tokens(input.text)
    }
}

extension Gemma4TextModel: LoRAModel {
    public var loraLayers: [Module] {
        languageModel.layers
    }
}
