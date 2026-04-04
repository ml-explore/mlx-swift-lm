//
//  Gemma4Text.swift
//  mlx-swift-lm
//
//  Port of https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/gemma4/language.py

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
    let headDim: Int
    let globalHeadDim: Int
    let rmsNormEps: Float
    let vocabularySize: Int
    let kvHeads: Int
    let globalKVHeads: Int?
    let slidingWindow: Int
    let layerTypes: [String]
    let maxPositionEmbeddings: Int
    let finalLogitSoftcapping: Float?
    let numKVSharedLayers: Int
    let ropeParameters: [String: [String: StringOrNumber]]
    let vocabSizePerLayerInput: Int
    let hiddenSizePerLayerInput: Int
    let enableMoeBlock: Bool
    let attentionKEqV: Bool
    let useDoubleWideMlp: Bool
    let tieWordEmbeddings: Bool

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case headDim = "head_dim"
        case globalHeadDim = "global_head_dim"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case kvHeads = "num_key_value_heads"
        case globalKVHeads = "num_global_key_value_heads"
        case slidingWindow = "sliding_window"
        case layerTypes = "layer_types"
        case maxPositionEmbeddings = "max_position_embeddings"
        case finalLogitSoftcapping = "final_logit_softcapping"
        case numKVSharedLayers = "num_kv_shared_layers"
        case ropeParameters = "rope_parameters"
        case vocabSizePerLayerInput = "vocab_size_per_layer_input"
        case hiddenSizePerLayerInput = "hidden_size_per_layer_input"
        case enableMoeBlock = "enable_moe_block"
        case attentionKEqV = "attention_k_eq_v"
        case useDoubleWideMlp = "use_double_wide_mlp"
        case tieWordEmbeddings = "tie_word_embeddings"
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
        hiddenSize = try container.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 2560
        hiddenLayers = try container.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 42
        intermediateSize =
            try container.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 10240
        attentionHeads = try container.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 8
        headDim = try container.decodeIfPresent(Int.self, forKey: .headDim) ?? 256
        globalHeadDim = try container.decodeIfPresent(Int.self, forKey: .globalHeadDim) ?? 512
        rmsNormEps = try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        vocabularySize = try container.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 262144
        kvHeads = try container.decodeIfPresent(Int.self, forKey: .kvHeads) ?? 2
        globalKVHeads = try container.decodeIfPresent(Int.self, forKey: .globalKVHeads)
        slidingWindow = try container.decodeIfPresent(Int.self, forKey: .slidingWindow) ?? 512
        layerTypes = try container.decodeIfPresent([String].self, forKey: .layerTypes) ?? []
        maxPositionEmbeddings =
            try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 131072
        finalLogitSoftcapping =
            try container.decodeIfPresent(Float.self, forKey: .finalLogitSoftcapping)
        numKVSharedLayers =
            try container.decodeIfPresent(Int.self, forKey: .numKVSharedLayers) ?? 0
        ropeParameters =
            try container.decodeIfPresent(
                [String: [String: StringOrNumber]].self, forKey: .ropeParameters)
            ?? [:]
        vocabSizePerLayerInput =
            try container.decodeIfPresent(Int.self, forKey: .vocabSizePerLayerInput) ?? 0
        hiddenSizePerLayerInput =
            try container.decodeIfPresent(Int.self, forKey: .hiddenSizePerLayerInput) ?? 0
        enableMoeBlock =
            try container.decodeIfPresent(Bool.self, forKey: .enableMoeBlock) ?? false
        attentionKEqV =
            try container.decodeIfPresent(Bool.self, forKey: .attentionKEqV) ?? false
        useDoubleWideMlp =
            try container.decodeIfPresent(Bool.self, forKey: .useDoubleWideMlp) ?? false
        tieWordEmbeddings =
            try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? true
    }

    var slidingWindowPattern: Int {
        // Derive pattern from layerTypes: count of sliding + 1 full per cycle
        guard let firstFull = layerTypes.firstIndex(of: "full_attention") else {
            return layerTypes.count
        }
        return firstFull + 1
    }

    var firstKVSharedLayerIdx: Int {
        hiddenLayers - numKVSharedLayers
    }
}

// MARK: - Custom RMSNorm Variants

/// RMSNorm without learnable scale — uses MLXFast with ones weight for GPU-accelerated path
class Gemma4RMSNormNoScale: Module, UnaryLayer {
    let eps: Float
    private var _onesWeight: MLXArray?

    init(eps: Float = 1e-6) {
        self.eps = eps
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Use MLXFast.rmsNorm with a ones weight vector for the fast Metal kernel path
        // Lazily create and cache the weight to match input dimensions
        let dim = x.dim(-1)
        if _onesWeight == nil || _onesWeight!.dim(0) != dim {
            _onesWeight = MLXArray.ones([dim])
        }
        return MLXFast.rmsNorm(x, weight: _onesWeight!, eps: self.eps)
    }
}

/// RMSNorm where weight is used directly (no +1 offset like standard Gemma)
class Gemma4RMSNormZeroShift: Module, UnaryLayer {
    let weight: MLXArray
    let eps: Float

    init(dimensions: Int, eps: Float = 1e-6) {
        self.weight = MLXArray.ones([dimensions])
        self.eps = eps
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Use weight directly instead of 1.0 + weight
        return MLXFast.rmsNorm(x, weight: self.weight, eps: self.eps)
    }
}

// MARK: - ProportionalRoPE

/// Proportional RoPE for Gemma 4 full-attention layers.
/// Frequencies spaced over full head dim but rotation applied to only partial_rotary_factor
/// fraction of dimensions across the two halves (HF rotate_half convention).
class Gemma4ProportionalRoPE: Module, OffsetLayer {
    let dims: Int
    let traditional: Bool
    let rotatedDims: Int
    let rotHalf: Int
    let half: Int
    let _freqs: MLXArray?

    init(
        dims: Int,
        traditional: Bool = false,
        base: Float = 10000.0,
        factor: Float = 1.0,
        partialRotaryFactor: Float = 1.0
    ) {
        self.dims = dims
        self.traditional = traditional
        self.half = dims / 2

        let ropeAngles = Int(partialRotaryFactor * Float(dims) / 2.0)
        self.rotatedDims = 2 * ropeAngles
        self.rotHalf = ropeAngles

        if rotatedDims > 0 {
            let exponents = MLXArray(stride(from: 0, to: rotatedDims, by: 2))
                .asType(.float32) / Float(dims)
            self._freqs = factor * MLX.pow(base, exponents)
        } else {
            self._freqs = nil
        }
        super.init()
        self.freeze(keys: ["freqs"])
    }

    func callAsFunction(_ x: MLXArray, offset: Int = 0) -> MLXArray {
        guard rotatedDims > 0 else { return x }

        // Use .ellipsis for efficient single-op slicing (not per-axis 0...)
        let left = x[.ellipsis, ..<half]
        let right = x[.ellipsis, half...]

        // Gather dims to rotate from each half, apply RoPE, then scatter back
        let rotated = MLXFast.RoPE(
            concatenated([left[.ellipsis, ..<rotHalf],
                          right[.ellipsis, ..<rotHalf]], axis: -1),
            dimensions: rotatedDims,
            traditional: traditional,
            base: nil,
            scale: 1.0,
            offset: offset,
            freqs: _freqs
        )

        // Reassemble: rotated portions + unrotated remainders
        if rotHalf < half {
            return concatenated([
                rotated[.ellipsis, ..<rotHalf],
                left[.ellipsis, rotHalf...],
                rotated[.ellipsis, rotHalf...],
                right[.ellipsis, rotHalf...],
            ], axis: -1)
        } else {
            return concatenated([
                rotated[.ellipsis, ..<rotHalf],
                rotated[.ellipsis, rotHalf...],
            ], axis: -1)
        }
    }
}

// MARK: - Logit Softcapping

// Compiled logit softcapping — matches Python's @partial(mx.compile, shapeless=True)
private func makeCompiledSoftcap(_ softcapValue: Float) -> @Sendable (MLXArray) -> MLXArray {
    compile(shapeless: true) { (x: MLXArray) -> MLXArray in
        MLX.tanh(x / softcapValue) * softcapValue
    }
}

private func logitSoftcap(_ x: MLXArray, softcap: Float) -> MLXArray {
    return MLX.tanh(x / softcap) * softcap
}

// MARK: - Attention

class Gemma4Attention: Module {
    let config: Gemma4TextConfiguration
    let layerIdx: Int
    let layerType: String
    let isSliding: Bool
    let effectiveHeadDim: Int
    let nHeads: Int
    let nKVHeads: Int
    let useKEqV: Bool
    let isKVSharedLayer: Bool
    let scale: Float

    @ModuleInfo(key: "q_proj") var queryProj: Linear
    @ModuleInfo(key: "k_proj") var keyProj: Linear
    @ModuleInfo(key: "v_proj") var valueProj: Linear?
    @ModuleInfo(key: "o_proj") var outputProj: Linear

    @ModuleInfo(key: "q_norm") var queryNorm: Gemma4RMSNormZeroShift
    @ModuleInfo(key: "k_norm") var keyNorm: Gemma4RMSNormZeroShift
    @ModuleInfo(key: "v_norm") var vNorm: Gemma4RMSNormNoScale

    var rope: any OffsetLayer

    init(_ config: Gemma4TextConfiguration, layerIdx: Int) {
        self.config = config
        self.layerIdx = layerIdx
        self.layerType = layerIdx < config.layerTypes.count
            ? config.layerTypes[layerIdx] : "sliding_attention"
        self.isSliding = layerType == "sliding_attention"

        self.effectiveHeadDim =
            (layerType == "full_attention" && config.globalHeadDim > 0)
            ? config.globalHeadDim : config.headDim

        self.nHeads = config.attentionHeads

        // K-eq-V for full attention layers when enabled
        self.useKEqV = config.attentionKEqV && !isSliding
        if useKEqV, let globalKV = config.globalKVHeads {
            self.nKVHeads = globalKV
        } else {
            self.nKVHeads = config.kvHeads
        }

        self.scale = 1.0  // Gemma 4 uses RMSNorm on Q/K instead of scaling

        let dim = config.hiddenSize
        self._queryProj.wrappedValue = Linear(dim, nHeads * effectiveHeadDim, bias: false)
        self._keyProj.wrappedValue = Linear(dim, nKVHeads * effectiveHeadDim, bias: false)
        if !useKEqV {
            self._valueProj.wrappedValue = Linear(dim, nKVHeads * effectiveHeadDim, bias: false)
        }
        self._outputProj.wrappedValue = Linear(nHeads * effectiveHeadDim, dim, bias: false)

        self._queryNorm.wrappedValue = Gemma4RMSNormZeroShift(
            dimensions: effectiveHeadDim, eps: config.rmsNormEps)
        self._keyNorm.wrappedValue = Gemma4RMSNormZeroShift(
            dimensions: effectiveHeadDim, eps: config.rmsNormEps)
        self._vNorm.wrappedValue = Gemma4RMSNormNoScale(eps: config.rmsNormEps)

        // KV sharing
        self.isKVSharedLayer = config.firstKVSharedLayerIdx > 0
            && layerIdx >= config.firstKVSharedLayerIdx

        // Initialize RoPE based on layer type
        let ropeKey = isSliding ? "sliding_attention" : "full_attention"
        let ropeParams = config.ropeParameters[ropeKey] ?? [:]
        let ropeTheta = ropeParams["rope_theta"]?.asFloat() ?? 10000.0
        let ropeType: String = {
            if let typeValue = ropeParams["rope_type"], case .string(let s) = typeValue {
                return s
            }
            return "default"
        }()

        if ropeType == "proportional" {
            let partialFactor = ropeParams["partial_rotary_factor"]?.asFloat() ?? 1.0
            let factor = ropeParams["factor"]?.asFloat() ?? 1.0
            self.rope = Gemma4ProportionalRoPE(
                dims: effectiveHeadDim,
                traditional: false,
                base: ropeTheta,
                factor: factor,
                partialRotaryFactor: partialFactor
            )
        } else {
            self.rope = RoPE(
                dimensions: effectiveHeadDim, traditional: false, base: ropeTheta)
        }

        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache? = nil
    ) -> MLXArray {
        let (B, L, _) = (x.dim(0), x.dim(1), x.dim(2))
        let offset = cache?.offset ?? 0

        var queries = queryProj(x)
            .reshaped(B, L, nHeads, effectiveHeadDim)
        queries = queryNorm(queries)
        queries = queries.transposed(0, 2, 1, 3)
        queries = rope(queries, offset: offset)

        // For KV shared layers, read from the already-updated cache
        if isKVSharedLayer, let cache {
            let cachedState = cache.state
            let cachedKeys = cachedState[0]
            let cachedValues = cachedState[1]

            let output = MLXFast.scaledDotProductAttention(
                queries: queries,
                keys: cachedKeys,
                values: cachedValues,
                scale: scale,
                mask: mask
            )
            .transposed(0, 2, 1, 3)
            .reshaped(B, L, -1)
            return outputProj(output)
        }

        // Compute K and V
        var keys = keyProj(x)
            .reshaped(B, L, nKVHeads, effectiveHeadDim)

        let values: MLXArray
        if useKEqV {
            // V derived from raw K before normalization
            values = vNorm(keys).transposed(0, 2, 1, 3)
        } else {
            values = vNorm(
                valueProj!(x).reshaped(B, L, nKVHeads, effectiveHeadDim)
            ).transposed(0, 2, 1, 3)
        }

        keys = keyNorm(keys)
        keys = keys.transposed(0, 2, 1, 3)
        keys = rope(keys, offset: offset)

        let output = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: cache,
            scale: scale,
            mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)
        return outputProj(output)
    }
}

// MARK: - MLP

class Gemma4MLP: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear

    init(dimensions: Int, hiddenDimensions: Int) {
        self._gateProj.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        self._downProj.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
        self._upProj.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        return downProj(geluApproximate(gateProj(x)) * upProj(x))
    }
}

// MARK: - Decoder Layer

class Gemma4DecoderLayer: Module {
    let config: Gemma4TextConfiguration
    let layerIdx: Int
    let layerType: String

    @ModuleInfo(key: "self_attn") var selfAttention: Gemma4Attention
    @ModuleInfo var mlp: Gemma4MLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: Gemma4RMSNormZeroShift
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: Gemma4RMSNormZeroShift
    @ModuleInfo(key: "pre_feedforward_layernorm") var preFeedforwardLayerNorm: Gemma4RMSNormZeroShift
    @ModuleInfo(key: "post_feedforward_layernorm") var postFeedforwardLayerNorm: Gemma4RMSNormZeroShift

    // Per-layer input gating
    @ModuleInfo(key: "per_layer_input_gate") var perLayerInputGate: Linear?
    @ModuleInfo(key: "per_layer_projection") var perLayerProjection: Linear?
    @ModuleInfo(key: "post_per_layer_input_norm") var postPerLayerInputNorm: Gemma4RMSNormZeroShift?

    // Layer scalar
    @ModuleInfo(key: "layer_scalar") var layerScalar: MLXArray?

    init(_ config: Gemma4TextConfiguration, layerIdx: Int) {
        self.config = config
        self.layerIdx = layerIdx
        self.layerType = layerIdx < config.layerTypes.count
            ? config.layerTypes[layerIdx] : "sliding_attention"

        self._selfAttention.wrappedValue = Gemma4Attention(config, layerIdx: layerIdx)

        // MLP with optional double-wide for KV-shared layers
        let isKVShared = config.firstKVSharedLayerIdx > 0
            && layerIdx >= config.firstKVSharedLayerIdx
        let mlpIntermediate =
            config.useDoubleWideMlp && isKVShared
            ? config.intermediateSize * 2 : config.intermediateSize
        self.mlp = Gemma4MLP(dimensions: config.hiddenSize, hiddenDimensions: mlpIntermediate)

        self._inputLayerNorm.wrappedValue = Gemma4RMSNormZeroShift(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = Gemma4RMSNormZeroShift(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._preFeedforwardLayerNorm.wrappedValue = Gemma4RMSNormZeroShift(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postFeedforwardLayerNorm.wrappedValue = Gemma4RMSNormZeroShift(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)

        // Per-layer input gating (2B/4B models)
        if config.hiddenSizePerLayerInput > 0 {
            self._perLayerInputGate.wrappedValue = Linear(
                config.hiddenSize, config.hiddenSizePerLayerInput, bias: false)
            self._perLayerProjection.wrappedValue = Linear(
                config.hiddenSizePerLayerInput, config.hiddenSize, bias: false)
            self._postPerLayerInputNorm.wrappedValue = Gemma4RMSNormZeroShift(
                dimensions: config.hiddenSize, eps: config.rmsNormEps)
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
        // Self-attention block
        let inputNorm = inputLayerNorm(x)
        let attnOut = selfAttention(inputNorm, mask: mask, cache: cache)
        let attnNorm = postAttentionLayerNorm(attnOut)
        var h = x + attnNorm

        // MLP block
        let residual = h
        h = preFeedforwardLayerNorm(h)
        h = mlp(h)
        h = postFeedforwardLayerNorm(h)
        h = residual + h

        // Per-layer input gating
        if let gate = perLayerInputGate,
            let proj = perLayerProjection,
            let norm = postPerLayerInputNorm,
            let pli = perLayerInput
        {
            let gateResidual = h
            var gated = gate(h)
            gated = geluApproximate(gated)
            gated = gated * pli
            gated = proj(gated)
            gated = norm(gated)
            h = gateResidual + gated
        }

        // Layer scalar
        if let scalar = layerScalar {
            h = h * scalar
        }

        return h
    }
}

// MARK: - Scaled Linear (for per-layer projection)

/// Linear layer with output scaling — extends Linear so it gets quantized properly
class Gemma4ScaledLinear: Linear {
    let scalar: Float

    init(inFeatures: Int, outFeatures: Int, scalar: Float) {
        self.scalar = scalar
        super.init(weight: MLXArray.zeros([outFeatures, inFeatures]), bias: nil)
    }

    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        return super.callAsFunction(x) * scalar
    }
}

// MARK: - Inner Model

public class Gemma4TextModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo var layers: [Gemma4DecoderLayer]
    @ModuleInfo var norm: Gemma4RMSNormZeroShift

    let config: Gemma4TextConfiguration
    let embedScale: Float
    let layerIdxToCacheIdx: [Int]
    let firstFullCacheIdx: Int
    let firstSlidingCacheIdx: Int

    // Per-layer embeddings
    @ModuleInfo(key: "embed_tokens_per_layer") var embedTokensPerLayer: Embedding?
    @ModuleInfo(key: "per_layer_model_projection") var perLayerModelProjection:
        Gemma4ScaledLinear?
    @ModuleInfo(key: "per_layer_projection_norm") var perLayerProjectionNorm:
        Gemma4RMSNormZeroShift?

    init(_ config: Gemma4TextConfiguration) {
        self.config = config
        self.embedScale = sqrt(Float(config.hiddenSize))

        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)

        self._layers.wrappedValue = (0..<config.hiddenLayers).map { i in
            Gemma4DecoderLayer(config, layerIdx: i)
        }

        self.norm = Gemma4RMSNormZeroShift(dimensions: config.hiddenSize, eps: config.rmsNormEps)

        // Build cache index mapping for KV sharing
        let firstShared = config.firstKVSharedLayerIdx
        let concreteLayers = Array(config.layerTypes.prefix(firstShared))

        var mapping = Array(0..<firstShared)
        if firstShared < config.hiddenLayers {
            // Find the last concrete full and sliding attention layer indices
            let sharedFullIdx =
                concreteLayers.lastIndex(of: "full_attention") ?? (concreteLayers.count - 1)
            let sharedSlidingIdx =
                concreteLayers.lastIndex(of: "sliding_attention") ?? 0

            for i in firstShared..<config.hiddenLayers {
                if i < config.layerTypes.count && config.layerTypes[i] == "full_attention" {
                    mapping.append(sharedFullIdx)
                } else {
                    mapping.append(sharedSlidingIdx)
                }
            }
        }
        self.layerIdxToCacheIdx = mapping

        self.firstFullCacheIdx =
            concreteLayers.firstIndex(of: "full_attention") ?? 0
        self.firstSlidingCacheIdx =
            concreteLayers.firstIndex(of: "sliding_attention") ?? 0

        // Per-layer embeddings
        if config.hiddenSizePerLayerInput > 0 {
            self._embedTokensPerLayer.wrappedValue = Embedding(
                embeddingCount: config.vocabSizePerLayerInput,
                dimensions: config.hiddenLayers * config.hiddenSizePerLayerInput)
            self._perLayerModelProjection.wrappedValue = Gemma4ScaledLinear(
                inFeatures: config.hiddenSize,
                outFeatures: config.hiddenLayers * config.hiddenSizePerLayerInput,
                scalar: pow(Float(config.hiddenSize), -0.5))
            self._perLayerProjectionNorm.wrappedValue = Gemma4RMSNormZeroShift(
                dimensions: config.hiddenSizePerLayerInput, eps: config.rmsNormEps)
        }

        super.init()
    }

    // Pre-computed constants
    private lazy var perLayerEmbedScale: Float = sqrt(Float(config.hiddenSizePerLayerInput))
    private let perLayerInputScale: Float = 0.7071067811865476  // pow(2.0, -0.5)

    func getPerLayerInputs(_ inputIds: MLXArray) -> MLXArray {
        guard let embedPerLayer = embedTokensPerLayer else {
            fatalError("Per-layer embeddings not initialized")
        }
        var result = embedPerLayer(inputIds)
        result = result * perLayerEmbedScale
        let shape = inputIds.shape
        return result.reshaped(
            shape[0], shape.count > 1 ? shape[1] : 1,
            config.hiddenLayers, config.hiddenSizePerLayerInput)
    }

    func projectPerLayerInputs(
        _ inputsEmbeds: MLXArray, perLayerInputs: MLXArray?
    ) -> MLXArray {
        guard let proj = perLayerModelProjection,
            let projNorm = perLayerProjectionNorm
        else {
            fatalError("Per-layer projection not initialized")
        }

        var perLayerProjection = proj(inputsEmbeds)
        let shape = inputsEmbeds.shape
        perLayerProjection = perLayerProjection.reshaped(
            shape[0], shape[1], config.hiddenLayers, config.hiddenSizePerLayerInput)
        perLayerProjection = projNorm(perLayerProjection)

        guard let pli = perLayerInputs else {
            return perLayerProjection
        }

        return (perLayerProjection + pli) * perLayerInputScale
    }

    func callAsFunction(
        _ inputs: MLXArray, cache: [KVCache]? = nil
    ) -> MLXArray {
        var h = embedTokens(inputs)
        h = h * embedScale

        // Per-layer inputs — pre-slice all layers upfront to avoid per-layer slicing overhead
        var perLayerSlices: [MLXArray]? = nil
        if config.hiddenSizePerLayerInput > 0 {
            var perLayerInputs = getPerLayerInputs(inputs)
            perLayerInputs = projectPerLayerInputs(h, perLayerInputs: perLayerInputs)
            // Pre-slice into per-layer arrays once
            perLayerSlices = (0..<config.hiddenLayers).map { i in
                perLayerInputs[0..., 0..., i, 0...]
            }
        }

        // Create masks once for the entire forward pass
        let globalMask: MLXFast.ScaledDotProductAttentionMaskMode
        let slidingMask: MLXFast.ScaledDotProductAttentionMaskMode

        if let cache, firstFullCacheIdx < cache.count {
            globalMask = createAttentionMask(h: h, cache: cache[firstFullCacheIdx])
        } else {
            globalMask = createAttentionMask(h: h, cache: nil as KVCache?)
        }

        if let cache, firstSlidingCacheIdx < cache.count {
            slidingMask = createAttentionMask(
                h: h, cache: cache[firstSlidingCacheIdx], windowSize: config.slidingWindow)
        } else {
            slidingMask = createAttentionMask(
                h: h, cache: nil as KVCache?, windowSize: config.slidingWindow)
        }

        for (i, layer) in layers.enumerated() {
            let cacheIdx = layerIdxToCacheIdx[i]
            let c: KVCache? = cache != nil && cacheIdx < cache!.count ? cache![cacheIdx] : nil
            let mask = layer.layerType == "full_attention" ? globalMask : slidingMask
            h = layer(h, mask: mask, cache: c, perLayerInput: perLayerSlices?[i])
        }

        return norm(h)
    }
}

// MARK: - Outer Model

public class Gemma4TextModel: Module, LLMModel {

    @ModuleInfo public var model: Gemma4TextModelInner
    public let config: Gemma4TextConfiguration
    public var vocabularySize: Int { config.vocabularySize }

    // Pre-compiled softcap function — avoids graph retracing per token
    private let compiledSoftcap: (@Sendable (MLXArray) -> MLXArray)?

    public init(_ config: Gemma4TextConfiguration) {
        self.config = config
        self.model = Gemma4TextModelInner(config)
        if let softcap = config.finalLogitSoftcapping {
            self.compiledSoftcap = makeCompiledSoftcap(softcap)
        } else {
            self.compiledSoftcap = nil
        }
        super.init()
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var out = model(inputs, cache: cache)
        out = model.embedTokens.asLinear(out)
        if let softcap = compiledSoftcap {
            out = softcap(out)
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

        // Filter out rotary embeddings, rope frequencies, and quantization range params
        processedWeights = processedWeights.filter { key, _ in
            !key.contains("self_attn.rotary_emb")
                && !key.contains("rope.freqs")
                && !key.contains("input_max")
                && !key.contains("input_min")
                && !key.contains("output_max")
                && !key.contains("output_min")
        }

        // Truncate vocab if needed
        let expectedVocab = config.vocabularySize
        let keysToCheck = [
            "model.embed_tokens.weight", "model.embed_tokens.scales",
            "model.embed_tokens.biases",
        ]
        for key in keysToCheck {
            if let tensor = processedWeights[key], tensor.dim(0) > expectedVocab {
                processedWeights[key] = tensor[0..<expectedVocab]
            }
        }

        return processedWeights
    }

    public func newCache(parameters: GenerateParameters? = nil) -> [KVCache] {
        // Only create caches for concrete (non-shared) layers
        let firstShared = config.firstKVSharedLayerIdx
        var caches = [KVCache]()

        for i in 0..<firstShared {
            let layerType =
                i < config.layerTypes.count ? config.layerTypes[i] : "sliding_attention"
            if layerType == "full_attention" {
                let cache = StandardKVCache()
                cache.step = 1024
                caches.append(cache)
            } else {
                caches.append(
                    RotatingKVCache(maxSize: config.slidingWindow, keep: 0))
            }
        }

        return caches
    }

    public func prepare(
        _ input: LMInput, cache: [KVCache], windowSize: Int? = nil
    ) throws -> PrepareResult {
        let promptTokens = input.text.tokens
        guard promptTokens.shape[0] > 0 else {
            let emptyToken = MLXArray(Int32(0))[0..<0]
            return .tokens(.init(tokens: emptyToken))
        }
        return .tokens(input.text)
    }
}

// MARK: - LoRA

extension Gemma4TextModel: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}
