// Copyright © 2025 Apple Inc.

import CoreImage
import MLX
import MLXLMCommon
import MLXNN

// Based on https://github.com/Blaizzy/mlx-vlm/tree/main/mlx_vlm/models/gemma4

// MARK: - RoPE Configuration

/// Single layer-type RoPE config (e.g. for "full_attention" or "sliding_attention")
public struct RoPEParameterEntry: Codable, Sendable {
    let ropeTheta: Float?
    let ropeType: String?
    let partialRotaryFactor: Float?

    enum CodingKeys: String, CodingKey {
        case ropeTheta = "rope_theta"
        case ropeType = "rope_type"
        case partialRotaryFactor = "partial_rotary_factor"
    }
}

/// ProportionalRoPE for Gemma 4 full-attention layers.
///
/// Frequencies are computed relative to the full head dimension (not just the
/// rotated portion), and rotation is applied to the first rotated_dims//2
/// elements of each half of the head — matching HF's rotate_half convention.
private class ProportionalRoPE: Module, OffsetLayer {
    let dims: Int
    let traditional: Bool
    let rotatedDims: Int
    let _freqs: MLXArray?

    init(dims: Int, traditional: Bool = false, base: Float = 10000.0, partialRotaryFactor: Float = 1.0, factor: Float = 1.0) {
        self.dims = dims
        self.traditional = traditional

        let ropeAngles = Int(partialRotaryFactor * Float(dims) / 2)
        self.rotatedDims = 2 * ropeAngles

        if rotatedDims > 0 {
            let exponents = MLXArray(stride(from: 0, to: rotatedDims, by: 2)).asType(.float32) / Float(dims)
            self._freqs = factor * MLX.pow(base, exponents)
        } else {
            self._freqs = nil
        }
        super.init()
    }

    func callAsFunction(_ x: MLXArray, offset: Int = 0) -> MLXArray {
        guard rotatedDims > 0 else { return x }

        let head = x[0..., 0..., 0..., ..<dims]
        let tail = x[0..., 0..., 0..., dims...]

        let half = dims / 2
        let left = head[0..., 0..., 0..., ..<half]
        let right = head[0..., 0..., 0..., half...]

        let halfRotated = rotatedDims / 2
        var rotated = concatenated(
            [left[0..., 0..., 0..., ..<halfRotated], right[0..., 0..., 0..., ..<halfRotated]],
            axis: -1
        )

        rotated = MLXFast.RoPE(
            rotated,
            dimensions: rotatedDims,
            traditional: traditional,
            base: nil,
            scale: 1.0,
            offset: offset,
            freqs: _freqs!
        )

        let newLeft = concatenated(
            [rotated[0..., 0..., 0..., ..<halfRotated], left[0..., 0..., 0..., halfRotated...]],
            axis: -1
        )
        let newRight = concatenated(
            [rotated[0..., 0..., 0..., halfRotated...], right[0..., 0..., 0..., halfRotated...]],
            axis: -1
        )
        let newHead = concatenated([newLeft, newRight], axis: -1)

        if tail.dim(-1) == 0 {
            return newHead
        }
        return concatenated([newHead, tail], axis: -1)
    }
}

// MARK: - Text Configuration

public struct Gemma4TextConfiguration: Codable, Sendable {
    public let modelType: String
    public let hiddenSize: Int
    public let hiddenLayers: Int
    public let intermediateSize: Int
    public let slidingWindow: Int

    public let vocabularySize: Int
    public let rmsNormEps: Float
    public let finalLogitSoftcapping: Float?

    private let _attentionHeads: Int?
    private let _kvHeads: Int?
    private let _headDim: Int?
    private let _globalHeadDim: Int?
    private let _globalPartialRotaryFactor: Float?
    private let _numGlobalKeyValueHeads: Int?
    private let _numKvSharedLayers: Int?
    private let _hiddenSizePerLayerInput: Int?
    private let _vocabSizePerLayerInput: Int?
    private let _attentionKEqV: Bool?
    private let _useDoubleWideMlp: Bool?

    public let ropeTraditional: Bool
    public let maxPositionEmbeddings: Int
    public let slidingWindowPattern: Int
    public let layerTypes: [String]?

    /// Per-layer-type RoPE parameters dict
    public let ropeParameters: [String: RoPEParameterEntry]?

    public var attentionHeads: Int { _attentionHeads ?? 8 }
    public var kvHeads: Int { _kvHeads ?? 1 }
    public var headDim: Int { _headDim ?? 256 }
    public var globalHeadDim: Int { _globalHeadDim ?? headDim }
    public var globalPartialRotaryFactor: Float { _globalPartialRotaryFactor ?? 0.25 }
    public var numGlobalKeyValueHeads: Int? { _numGlobalKeyValueHeads }
    public var numKvSharedLayers: Int { _numKvSharedLayers ?? 0 }
    public var hiddenSizePerLayerInput: Int { _hiddenSizePerLayerInput ?? 0 }
    public var vocabSizePerLayerInput: Int { _vocabSizePerLayerInput ?? vocabularySize }
    public var attentionKEqV: Bool { _attentionKEqV ?? false }
    public var useDoubleWideMlp: Bool { _useDoubleWideMlp ?? true }

    /// Compute the effective layer_types array (from config or generated from sliding_window_pattern)
    public var effectiveLayerTypes: [String] {
        if let lt = layerTypes, !lt.isEmpty { return lt }
        var pattern = Array(repeating: "sliding_attention", count: slidingWindowPattern - 1)
        pattern.append("full_attention")
        var result: [String] = []
        while result.count < hiddenLayers {
            result.append(contentsOf: pattern)
        }
        return Array(result.prefix(hiddenLayers))
    }

    /// Index of the first KV-shared layer
    public var firstKvSharedLayerIdx: Int {
        hiddenLayers - numKvSharedLayers
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case slidingWindow = "sliding_window"
        case vocabularySize = "vocab_size"
        case rmsNormEps = "rms_norm_eps"
        case finalLogitSoftcapping = "final_logit_softcapping"
        case _attentionHeads = "num_attention_heads"
        case _kvHeads = "num_key_value_heads"
        case _headDim = "head_dim"
        case _globalHeadDim = "global_head_dim"
        case _globalPartialRotaryFactor = "global_partial_rotary_factor"
        case _numGlobalKeyValueHeads = "num_global_key_value_heads"
        case _numKvSharedLayers = "num_kv_shared_layers"
        case _hiddenSizePerLayerInput = "hidden_size_per_layer_input"
        case _vocabSizePerLayerInput = "vocab_size_per_layer_input"
        case _attentionKEqV = "attention_k_eq_v"
        case _useDoubleWideMlp = "use_double_wide_mlp"
        case ropeTraditional = "rope_traditional"
        case maxPositionEmbeddings = "max_position_embeddings"
        case slidingWindowPattern = "sliding_window_pattern"
        case layerTypes = "layer_types"
        case ropeParameters = "rope_parameters"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try container.decode(String.self, forKey: .modelType)
        hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        hiddenLayers = try container.decode(Int.self, forKey: .hiddenLayers)
        intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)
        slidingWindow = try container.decodeIfPresent(Int.self, forKey: .slidingWindow) ?? 512
        vocabularySize = try container.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 262144
        rmsNormEps = try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        finalLogitSoftcapping = try container.decodeIfPresent(Float.self, forKey: .finalLogitSoftcapping)
        _attentionHeads = try container.decodeIfPresent(Int.self, forKey: ._attentionHeads)
        _kvHeads = try container.decodeIfPresent(Int.self, forKey: ._kvHeads)
        _headDim = try container.decodeIfPresent(Int.self, forKey: ._headDim)
        _globalHeadDim = try container.decodeIfPresent(Int.self, forKey: ._globalHeadDim)
        _globalPartialRotaryFactor = try container.decodeIfPresent(Float.self, forKey: ._globalPartialRotaryFactor)
        _numGlobalKeyValueHeads = try container.decodeIfPresent(Int.self, forKey: ._numGlobalKeyValueHeads)
        _numKvSharedLayers = try container.decodeIfPresent(Int.self, forKey: ._numKvSharedLayers)
        _hiddenSizePerLayerInput = try container.decodeIfPresent(Int.self, forKey: ._hiddenSizePerLayerInput)
        _vocabSizePerLayerInput = try container.decodeIfPresent(Int.self, forKey: ._vocabSizePerLayerInput)
        _attentionKEqV = try container.decodeIfPresent(Bool.self, forKey: ._attentionKEqV)
        _useDoubleWideMlp = try container.decodeIfPresent(Bool.self, forKey: ._useDoubleWideMlp)
        ropeTraditional = try container.decodeIfPresent(Bool.self, forKey: .ropeTraditional) ?? false
        maxPositionEmbeddings = try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 131072
        slidingWindowPattern = try container.decodeIfPresent(Int.self, forKey: .slidingWindowPattern) ?? 5
        layerTypes = try container.decodeIfPresent([String].self, forKey: .layerTypes)
        ropeParameters = try container.decodeIfPresent([String: RoPEParameterEntry].self, forKey: .ropeParameters)
    }
}

// MARK: - Vision Configuration

public struct Gemma4VisionConfiguration: Codable, Sendable {
    public let modelType: String
    public let hiddenLayers: Int
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let attentionHeads: Int
    public let kvHeads: Int
    public let headDim: Int
    public let patchSize: Int
    public let rmsNormEps: Float
    public let defaultOutputLength: Int
    public let positionEmbeddingSize: Int
    public let poolingKernelSize: Int
    public let standardize: Bool
    public let useClippedLinears: Bool

    public let ropeParameters: RoPEParameterEntry?

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenLayers = "num_hidden_layers"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case patchSize = "patch_size"
        case rmsNormEps = "rms_norm_eps"
        case defaultOutputLength = "default_output_length"
        case positionEmbeddingSize = "position_embedding_size"
        case poolingKernelSize = "pooling_kernel_size"
        case standardize
        case useClippedLinears = "use_clipped_linears"
        case ropeParameters = "rope_parameters"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try container.decodeIfPresent(String.self, forKey: .modelType) ?? "gemma4_vision"
        hiddenLayers = try container.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 16
        hiddenSize = try container.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 768
        intermediateSize = try container.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 3072
        attentionHeads = try container.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 12
        kvHeads = try container.decodeIfPresent(Int.self, forKey: .kvHeads) ?? 12
        headDim = try container.decodeIfPresent(Int.self, forKey: .headDim) ?? 64
        patchSize = try container.decodeIfPresent(Int.self, forKey: .patchSize) ?? 16
        rmsNormEps = try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        defaultOutputLength = try container.decodeIfPresent(Int.self, forKey: .defaultOutputLength) ?? 280
        positionEmbeddingSize = try container.decodeIfPresent(Int.self, forKey: .positionEmbeddingSize) ?? 10240
        poolingKernelSize = try container.decodeIfPresent(Int.self, forKey: .poolingKernelSize) ?? 3
        standardize = try container.decodeIfPresent(Bool.self, forKey: .standardize) ?? false
        useClippedLinears = try container.decodeIfPresent(Bool.self, forKey: .useClippedLinears) ?? false
        ropeParameters = try container.decodeIfPresent(RoPEParameterEntry.self, forKey: .ropeParameters)
    }
}

// MARK: - Model Configuration

public struct Gemma4Configuration: Codable, Sendable {
    public let textConfiguration: Gemma4TextConfiguration
    public let visionConfiguration: Gemma4VisionConfiguration
    public let modelType: String
    public let quantization: BaseConfiguration.Quantization?

    private let _vocabularySize: Int?
    private let _padTokenId: Int?
    private let _imageTokenId: Int?
    private let _visionSoftTokensPerImage: Int?

    public var vocabularySize: Int {
        _vocabularySize ?? textConfiguration.vocabularySize
    }

    public var hiddenSize: Int {
        textConfiguration.hiddenSize
    }

    public var padTokenId: Int {
        _padTokenId ?? 0
    }

    public var imageTokenId: Int {
        _imageTokenId ?? 258880
    }

    public var visionSoftTokensPerImage: Int {
        _visionSoftTokensPerImage ?? 280
    }

    enum CodingKeys: String, CodingKey {
        case textConfiguration = "text_config"
        case visionConfiguration = "vision_config"
        case modelType = "model_type"
        case quantization
        case _vocabularySize = "vocab_size"
        case _padTokenId = "pad_token_id"
        case _imageTokenId = "image_token_id"
        case _visionSoftTokensPerImage = "vision_soft_tokens_per_image"
    }
}

// MARK: - RMSNorm variants

/// Gemma4 RMSNorm with scale_shift=0 (weight used directly, no +1 offset).
/// This differs from Gemma3's RMSNorm which adds 1 to weight.
private class Gemma4RMSNorm: Module, UnaryLayer {
    let weight: MLXArray
    let eps: Float

    init(dimensions: Int, eps: Float = 1e-6) {
        self.weight = MLXArray.ones([dimensions])
        self.eps = eps
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        return MLXFast.rmsNorm(x, weight: self.weight, eps: self.eps)
    }
}

/// RMSNorm without learnable scale (parameter-free)
private class RMSNormNoScale: Module, UnaryLayer {
    let eps: Float

    init(eps: Float = 1e-6) {
        self.eps = eps
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let xFloat = x.asType(.float32)
        let variance = mean(xFloat * xFloat, axis: -1, keepDims: true)
        let result = xFloat * MLX.rsqrt(variance + eps)
        return result.asType(x.dtype)
    }
}

/// Vision RMSNorm with learned scale (full float32 computation)
private class VisionRMSNorm: Module, UnaryLayer {
    let weight: MLXArray
    let eps: Float

    init(dimensions: Int, eps: Float = 1e-6) {
        self.weight = MLXArray.ones([dimensions])
        self.eps = eps
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let xFloat = x.asType(.float32)
        let variance = mean(xFloat * xFloat, axis: -1, keepDims: true)
        let normed = xFloat * MLX.rsqrt(variance + eps)
        let result = normed * weight.asType(.float32)
        return result.asType(x.dtype)
    }
}

/// Vision RMSNorm without learned scale (parameter-free, full float32)
private class VisionRMSNormNoScale: Module, UnaryLayer {
    let eps: Float

    init(eps: Float = 1e-6) {
        self.eps = eps
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let xFloat = x.asType(.float32)
        let variance = mean(xFloat * xFloat, axis: -1, keepDims: true)
        return (xFloat * MLX.rsqrt(variance + eps)).asType(x.dtype)
    }
}

// MARK: - Text Attention

private class Gemma4Attention: Module {
    let numHeads: Int
    let numKVHeads: Int
    let headDim: Int
    let layerIdx: Int
    let layerType: String
    let isSliding: Bool
    let useKEqV: Bool
    let isKvSharedLayer: Bool
    let scale: Float

    @ModuleInfo(key: "q_proj") var queryProj: Linear
    @ModuleInfo(key: "k_proj") var keyProj: Linear
    @ModuleInfo(key: "v_proj") var valueProj: Linear?
    @ModuleInfo(key: "o_proj") var outputProj: Linear

    @ModuleInfo(key: "q_norm") var queryNorm: Gemma.RMSNorm
    @ModuleInfo(key: "k_norm") var keyNorm: Gemma.RMSNorm
    @ModuleInfo(key: "v_norm") var valueNorm: RMSNormNoScale

    @ModuleInfo var rope: OffsetLayer

    init(config: Gemma4TextConfiguration, layerIdx: Int) {
        let dim = config.hiddenSize
        let layerTypes = config.effectiveLayerTypes
        self.layerIdx = layerIdx
        self.layerType = layerTypes[layerIdx]
        self.isSliding = layerType == "sliding_attention"
        self.numHeads = config.attentionHeads

        // Global attention layers may use a different head_dim
        if layerType == "full_attention" && config.globalHeadDim != config.headDim {
            self.headDim = config.globalHeadDim
        } else {
            self.headDim = config.headDim
        }

        // K-eq-V: for full attention layers with attention_k_eq_v
        self.useKEqV = config.attentionKEqV && !isSliding

        // Determine numKVHeads
        if useKEqV, let globalKVHeads = config.numGlobalKeyValueHeads {
            self.numKVHeads = globalKVHeads
        } else {
            self.numKVHeads = config.kvHeads
        }

        // Gemma4 uses scale=1.0 (pre-attn scalar is baked into q_norm/k_norm)
        self.scale = 1.0

        // KV sharing
        let firstKvSharedIdx = config.firstKvSharedLayerIdx
        self.isKvSharedLayer = layerIdx >= firstKvSharedIdx && firstKvSharedIdx > 0

        self._queryProj.wrappedValue = Linear(dim, numHeads * headDim, bias: false)
        self._keyProj.wrappedValue = Linear(dim, numKVHeads * headDim, bias: false)
        if !useKEqV {
            self._valueProj.wrappedValue = Linear(dim, numKVHeads * headDim, bias: false)
        } else {
            self._valueProj.wrappedValue = nil
        }
        self._outputProj.wrappedValue = Linear(numHeads * headDim, dim, bias: false)

        self._queryNorm.wrappedValue = Gemma.RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        self._keyNorm.wrappedValue = Gemma.RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        self._valueNorm.wrappedValue = RMSNormNoScale(eps: config.rmsNormEps)

        // Initialize RoPE based on layer type
        let layerKey = isSliding ? "sliding_attention" : "full_attention"
        if let ropeParams = config.ropeParameters?[layerKey] {
            let ropeType = ropeParams.ropeType ?? "default"
            let ropeTheta = ropeParams.ropeTheta ?? 10000.0

            if ropeType == "proportional" {
                let partialFactor = ropeParams.partialRotaryFactor ?? 1.0
                self.rope = ProportionalRoPE(
                    dims: headDim,
                    traditional: config.ropeTraditional,
                    base: ropeTheta,
                    partialRotaryFactor: partialFactor
                )
            } else {
                self.rope = RoPE(dimensions: headDim, traditional: config.ropeTraditional, base: ropeTheta)
            }
        } else {
            // Default: sliding uses local base, full uses global
            let base: Float = isSliding ? 10000.0 : 1_000_000.0
            self.rope = RoPE(dimensions: headDim, traditional: config.ropeTraditional, base: base)
        }
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache? = nil,
        sharedKV: (MLXArray, MLXArray)? = nil,
        offset: Int? = nil
    ) -> (MLXArray, (MLXArray, MLXArray)?, Int) {
        let (B, L, _) = (x.dim(0), x.dim(1), x.dim(2))

        var queries = queryProj(x)
        queries = queries.reshaped(B, L, numHeads, headDim)
        queries = queryNorm(queries)

        var keys: MLXArray
        var values: MLXArray
        var effectiveOffset: Int

        if let sharedKV = sharedKV, let offset = offset {
            // KV-shared layer: reuse keys and values from the earlier layer
            keys = sharedKV.0
            values = sharedKV.1
            effectiveOffset = offset
        } else {
            effectiveOffset = cache?.offset ?? 0

            keys = keyProj(x).reshaped(B, L, numKVHeads, headDim)

            if useKEqV {
                // K-eq-V: values come from raw k_proj before k_norm
                values = keys
            } else {
                values = valueProj!(x).reshaped(B, L, numKVHeads, headDim)
            }

            keys = keyNorm(keys)
            values = valueNorm(values)
            values = values.transposed(0, 2, 1, 3)

            keys = keys.transposed(0, 2, 1, 3)
            keys = rope(keys, offset: effectiveOffset)
        }

        queries = queries.transposed(0, 2, 1, 3)
        queries = rope(queries, offset: effectiveOffset)

        // Use cache if available (and not a shared layer)
        let finalKeys: MLXArray
        let finalValues: MLXArray
        if sharedKV == nil, let cache = cache {
            let (ck, cv) = cache.update(keys: keys, values: values)
            finalKeys = ck
            finalValues = cv
        } else {
            finalKeys = keys
            finalValues = values
        }

        let output = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: finalKeys,
            values: finalValues,
            scale: scale,
            mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        return (outputProj(output), (finalKeys, finalValues), effectiveOffset)
    }
}

// MARK: - Text MLP

private class Gemma4MLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear

    init(dimensions: Int, hiddenDimensions: Int) {
        self._gateProj.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        self._downProj.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
        self._upProj.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(geluApproximate(gateProj(x)) * upProj(x))
    }
}

// MARK: - Decoder Layer

private class Gemma4DecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttention: Gemma4Attention
    @ModuleInfo var mlp: Gemma4MLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: Gemma.RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: Gemma.RMSNorm
    @ModuleInfo(key: "pre_feedforward_layernorm") var preFeedforwardLayerNorm: Gemma.RMSNorm
    @ModuleInfo(key: "post_feedforward_layernorm") var postFeedforwardLayerNorm: Gemma.RMSNorm

    // Per-layer input gating (PLE)
    @ModuleInfo(key: "per_layer_input_gate") var perLayerInputGate: Linear?
    @ModuleInfo(key: "per_layer_projection") var perLayerProjection: Linear?
    @ModuleInfo(key: "post_per_layer_input_norm") var postPerLayerInputNorm: Gemma.RMSNorm?

    // Layer scalar
    @ModuleInfo(key: "layer_scalar") var layerScalar: MLXArray

    let layerType: String

    init(config: Gemma4TextConfiguration, layerIdx: Int) {
        let layerTypes = config.effectiveLayerTypes
        self.layerType = layerTypes[layerIdx]

        self._selfAttention.wrappedValue = Gemma4Attention(config: config, layerIdx: layerIdx)

        // MLP: use double-wide intermediate size for KV-shared layers
        let firstKvShared = config.firstKvSharedLayerIdx
        let isKvSharedLayer = layerIdx >= firstKvShared && firstKvShared > 0
        let useDoubleWide = config.useDoubleWideMlp && isKvSharedLayer
        let mlpIntermediate = config.intermediateSize * (useDoubleWide ? 2 : 1)
        self.mlp = Gemma4MLP(dimensions: config.hiddenSize, hiddenDimensions: mlpIntermediate)

        self._inputLayerNorm.wrappedValue = Gemma.RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = Gemma.RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._preFeedforwardLayerNorm.wrappedValue = Gemma.RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postFeedforwardLayerNorm.wrappedValue = Gemma.RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)

        // Per-layer input embeddings (PLE) for 2B/4B models
        let hiddenSizePLI = config.hiddenSizePerLayerInput
        if hiddenSizePLI > 0 {
            self._perLayerInputGate.wrappedValue = Linear(config.hiddenSize, hiddenSizePLI, bias: false)
            self._perLayerProjection.wrappedValue = Linear(hiddenSizePLI, config.hiddenSize, bias: false)
            self._postPerLayerInputNorm.wrappedValue = Gemma.RMSNorm(
                dimensions: config.hiddenSize, eps: config.rmsNormEps)
        } else {
            self._perLayerInputGate.wrappedValue = nil
            self._perLayerProjection.wrappedValue = nil
            self._postPerLayerInputNorm.wrappedValue = nil
        }

        self._layerScalar.wrappedValue = MLXArray.ones([1])
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache? = nil,
        perLayerInput: MLXArray? = nil,
        sharedKV: (MLXArray, MLXArray)? = nil,
        offset: Int? = nil
    ) -> (MLXArray, (MLXArray, MLXArray)?, Int) {
        var residual = x

        var h = inputLayerNorm(x)
        let (attnOut, kvs, newOffset) = selfAttention(
            h, mask: mask, cache: cache, sharedKV: sharedKV, offset: offset)
        h = postAttentionLayerNorm(attnOut)
        h = residual + h

        residual = h
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
            residual = h
            var gateOut = gate(h)
            gateOut = geluApproximate(gateOut)
            gateOut = gateOut * pli
            gateOut = proj(gateOut)
            gateOut = norm(gateOut)
            h = residual + gateOut
        }

        h = h * layerScalar

        return (h, kvs, newOffset)
    }
}

// MARK: - Gemma4 Text Model

private class Gemma4TextModel: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo var layers: [Gemma4DecoderLayer]
    @ModuleInfo var norm: Gemma.RMSNorm

    // Per-layer input embeddings (PLE)
    @ModuleInfo(key: "embed_tokens_per_layer") var embedTokensPerLayer: Embedding?
    @ModuleInfo(key: "per_layer_model_projection") var perLayerModelProjection: Linear?
    @ModuleInfo(key: "per_layer_projection_norm") var perLayerProjectionNorm: Gemma4RMSNorm?

    let config: Gemma4TextConfiguration
    let embedScale: Float
    let hiddenSizePerLayerInput: Int

    /// Maps layer index to its cache index (for KV sharing)
    let previousKvs: [Int]

    init(_ config: Gemma4TextConfiguration) {
        self.config = config
        self.embedScale = sqrtf(Float(config.hiddenSize))
        self.hiddenSizePerLayerInput = config.hiddenSizePerLayerInput

        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize,
            dimensions: config.hiddenSize
        )

        self._layers.wrappedValue = (0 ..< config.hiddenLayers).map { layerIdx in
            Gemma4DecoderLayer(config: config, layerIdx: layerIdx)
        }

        self.norm = Gemma.RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)

        // Per-layer input embeddings (2B/4B models)
        if hiddenSizePerLayerInput > 0 {
            self._embedTokensPerLayer.wrappedValue = Embedding(
                embeddingCount: config.vocabSizePerLayerInput,
                dimensions: config.hiddenLayers * hiddenSizePerLayerInput
            )
            self._perLayerModelProjection.wrappedValue = Linear(
                config.hiddenSize,
                config.hiddenLayers * hiddenSizePerLayerInput,
                bias: false
            )
            self._perLayerProjectionNorm.wrappedValue = Gemma4RMSNorm(
                dimensions: hiddenSizePerLayerInput, eps: config.rmsNormEps
            )
        } else {
            self._embedTokensPerLayer.wrappedValue = nil
            self._perLayerModelProjection.wrappedValue = nil
            self._perLayerProjectionNorm.wrappedValue = nil
        }

        // Build KV sharing map
        var kvMap = Array(0..<config.hiddenLayers)
        let layerTypes = config.effectiveLayerTypes
        if config.numKvSharedLayers > 0 {
            let n = config.hiddenLayers
            let m = n - config.numKvSharedLayers
            var kvsByType: [String: Int] = [:]
            for i in 0..<m {
                kvsByType[layerTypes[i]] = i
            }
            for j in m..<n {
                if let idx = kvsByType[layerTypes[j]] {
                    kvMap[j] = idx
                }
            }
        }
        self.previousKvs = kvMap
    }

    /// Get per-layer input embeddings from token IDs
    func getPerLayerInputs(_ inputIds: MLXArray) -> MLXArray {
        guard let embedPerLayer = embedTokensPerLayer else {
            fatalError("embed_tokens_per_layer not available")
        }
        var result = embedPerLayer(inputIds)
        result = result * sqrtf(Float(hiddenSizePerLayerInput))
        // Reshape to [B, L, num_layers, hidden_size_per_layer_input]
        let shape = inputIds.shape
        return result.reshaped(shape[0], shape[1], config.hiddenLayers, hiddenSizePerLayerInput)
    }

    /// Project per-layer inputs through the model projection
    func projectPerLayerInputs(
        _ inputEmbeds: MLXArray,
        perLayerInputs: MLXArray?
    ) -> MLXArray {
        guard let proj = perLayerModelProjection,
              let norm = perLayerProjectionNorm else {
            fatalError("per_layer_model_projection or norm not available")
        }

        var perLayerProjection = proj(inputEmbeds)
        perLayerProjection = perLayerProjection * pow(Float(config.hiddenSize), -0.5)
        // Reshape to [B, L, num_layers, hidden_size_per_layer_input]
        let shape = inputEmbeds.shape
        perLayerProjection = perLayerProjection.reshaped(
            shape[0], shape[1], config.hiddenLayers, hiddenSizePerLayerInput)
        perLayerProjection = norm(perLayerProjection)

        if let pli = perLayerInputs {
            return (perLayerProjection + pli) * pow(2.0, -0.5)
        }
        return perLayerProjection
    }

    func callAsFunction(
        _ inputs: MLXArray? = nil,
        inputEmbedding: MLXArray? = nil,
        cache: [KVCache?]? = nil,
        perLayerInputs: MLXArray? = nil
    ) -> MLXArray {
        var h: MLXArray
        if let inputEmbedding = inputEmbedding {
            h = inputEmbedding
        } else if let inputs = inputs {
            h = embedTokens(inputs)
        } else {
            fatalError("Either inputs or inputEmbedding must be provided")
        }

        // Apply embedding scaling
        let scale = MLXArray(embedScale).asType(h.dtype)
        h = h * scale

        // Handle PLE
        var pliArray: [MLXArray?]
        if hiddenSizePerLayerInput > 0 {
            var effectivePLI = perLayerInputs
            if effectivePLI == nil, let inputs = inputs {
                effectivePLI = getPerLayerInputs(inputs)
            }
            let projected = projectPerLayerInputs(h, perLayerInputs: effectivePLI)
            pliArray = (0..<config.hiddenLayers).map { i in
                projected[0..., 0..., i, 0...]
            }
        } else {
            pliArray = Array(repeating: nil, count: config.hiddenLayers)
        }

        // Build cache list, extending with nil for shared layers
        var fullCache: [KVCache?]
        if let cache = cache {
            fullCache = cache + Array(repeating: nil as KVCache?, count: config.hiddenLayers - cache.count)
        } else {
            fullCache = Array(repeating: nil as KVCache?, count: config.hiddenLayers)
        }

        // Create masks
        let layerTypes = config.effectiveLayerTypes
        let firstFullIdx = layerTypes.firstIndex(of: "full_attention") ?? 0
        let firstSlidingIdx = layerTypes.firstIndex(of: "sliding_attention") ?? 0

        let globalMask = createAttentionMask(
            h: h,
            cache: firstFullIdx < fullCache.count ? fullCache[firstFullIdx] : nil
        )
        let slidingWindowMask = createAttentionMask(
            h: h,
            cache: firstSlidingIdx < fullCache.count ? fullCache[firstSlidingIdx] : nil,
            windowSize: config.slidingWindow
        )

        // Apply each layer with KV sharing
        var intermediates: [(kv: (MLXArray, MLXArray)?, offset: Int?)] =
            Array(repeating: (nil, nil), count: config.hiddenLayers)

        for (i, layer) in layers.enumerated() {
            let c = fullCache[i]
            let isGlobal = layerTypes[i] == "full_attention"
            let mask = isGlobal ? globalMask : slidingWindowMask

            let prevIdx = previousKvs[i]
            let sharedKV = prevIdx != i ? intermediates[prevIdx].kv : nil
            let sharedOffset = prevIdx != i ? intermediates[prevIdx].offset : nil

            let (newH, kvs, newOffset) = layer(
                h, mask: mask, cache: c,
                perLayerInput: pliArray[i],
                sharedKV: sharedKV,
                offset: sharedOffset
            )
            h = newH
            intermediates[i] = (kvs, newOffset)
        }
        return norm(h)
    }
}

// MARK: - Language Model

private class Gemma4LanguageModel: Module, KVCacheDimensionProvider {
    @ModuleInfo var model: Gemma4TextModel
    @ModuleInfo(key: "lm_head") var lmHead: Module  // Can be Linear or QuantizedLinear

    let config: Gemma4TextConfiguration
    var kvHeads: [Int]

    init(_ config: Gemma4TextConfiguration) {
        self.config = config
        self.model = Gemma4TextModel(config)
        self._lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabularySize, bias: false)
        self.kvHeads = Array(repeating: config.kvHeads, count: config.hiddenLayers)
    }

    /// Creates appropriate cache types for each layer (only for non-shared layers)
    public func newCache(parameters: GenerateParameters?) -> [any KVCache] {
        var caches: [any KVCache] = []
        let layerTypes = config.effectiveLayerTypes
        let firstKvShared = config.firstKvSharedLayerIdx

        for i in 0..<firstKvShared {
            if layerTypes[i] == "full_attention" {
                caches.append(StandardKVCache())
            } else {
                caches.append(RotatingKVCache(maxSize: config.slidingWindow, keep: 0))
            }
        }
        return caches
    }

    func callAsFunction(
        _ inputs: MLXArray? = nil,
        cache: [KVCache]? = nil,
        inputEmbedding: MLXArray? = nil,
        mask: MLXFast.ScaledDotProductAttentionMaskMode? = nil,
        perLayerInputs: MLXArray? = nil
    ) -> LMOutput {
        let optionalCache = cache?.map { $0 as KVCache? }
        let out = model(
            inputs, inputEmbedding: inputEmbedding,
            cache: optionalCache, perLayerInputs: perLayerInputs
        )

        // Call the lmHead (works whether it's Linear or QuantizedLinear)
        var finalLogits: MLXArray
        if let linear = lmHead as? Linear {
            finalLogits = linear(out)
        } else if let quantized = lmHead as? QuantizedLinear {
            finalLogits = quantized(out)
        } else {
            fatalError("lmHead must be Linear or QuantizedLinear")
        }

        // Apply final logit softcapping if configured
        if let softcap = config.finalLogitSoftcapping, softcap > 0 {
            let scale = MLXArray(softcap)
            finalLogits = tanh(finalLogits / scale) * scale
        }

        return LMOutput(logits: finalLogits)
    }

    func sanitize(
        weights: [String: MLXArray], quantizationConfig: BaseConfiguration.Quantization? = nil
    ) -> [String: MLXArray] {
        var processedWeights = weights

        // Check if we have quantized weights
        let hasQuantizedLmHead = hasQuantizedWeights(
            layerPath: "language_model.lm_head", in: weights)

        if hasQuantizedLmHead {
            let q = quantizationConfig?.asTuple ?? (64, 4, .affine)

            quantize(model: self) { path, module in
                let fullPath = "language_model.\(path)"
                if weights["\(fullPath).scales"] != nil
                    && weights["\(fullPath).weight"]?.dtype == .uint32
                {
                    return q
                }
                return nil
            }
        } else {
            // Handle weight tying for regular (non-quantized) lm_head
            if processedWeights["language_model.lm_head.weight"] == nil {
                if let embedWeight = processedWeights["language_model.model.embed_tokens.weight"] {
                    processedWeights["language_model.lm_head.weight"] = embedWeight
                }
            }
        }

        // Remove unused precomputed rotary freqs and clipping params
        return processedWeights.filter { key, _ in
            !key.contains("self_attn.rotary_emb") &&
            !key.contains("input_max") && !key.contains("input_min") &&
            !key.contains("output_max") && !key.contains("output_min")
        }
    }

    private func hasQuantizedWeights(layerPath: String, in weights: [String: MLXArray]) -> Bool {
        let scalesKey = "\(layerPath).scales"
        let biasesKey = "\(layerPath).biases"
        let weightKey = "\(layerPath).weight"

        return weights[scalesKey] != nil && weights[biasesKey] != nil
            && weights[weightKey]?.dtype == .uint32
    }
}

// MARK: - 2D RoPE for Vision

/// Applies multidimensional RoPE matching the Python apply_multidimensional_rope.
/// Splits head_dim into ndim parts and applies rotate_half independently per spatial dim.
private func applyMultidimensionalRoPE(
    _ inputs: MLXArray, positions: MLXArray, baseFrequency: Float = 100.0
) -> MLXArray {
    let headDim = inputs.dim(-1)

    if positions.ndim == 2 {
        // 1D fallback - standard rotary embedding
        let half = headDim / 2
        let freqExponents = (2.0 / Float(headDim)) * MLXArray(0..<half).asType(.float32)
        let timescale = MLX.pow(baseFrequency, freqExponents)
        let sinusoidInp = expandedDimensions(positions, axis: -1).asType(.float32) / timescale
        let cosVal = MLX.cos(sinusoidInp)
        let sinVal = MLX.sin(sinusoidInp)
        let cosValFull = expandedDimensions(concatenated([cosVal, cosVal], axis: -1), axis: 2).asType(inputs.dtype)
        let sinValFull = expandedDimensions(concatenated([sinVal, sinVal], axis: -1), axis: 2).asType(inputs.dtype)
        return inputs * cosValFull + rotateHalf(inputs) * sinValFull
    }

    let ndim = positions.dim(-1)
    let channelsPerDim = 2 * (headDim / (2 * ndim))
    let halfPerDim = channelsPerDim / 2

    var resultParts: [MLXArray] = []
    for d in 0..<ndim {
        let startIdx = d * channelsPerDim
        let endIdx = (d + 1) * channelsPerDim
        let xPart = inputs[0..., 0..., 0..., startIdx..<endIdx]

        let freqExponents = (2.0 / Float(channelsPerDim)) * MLXArray(0..<halfPerDim).asType(.float32)
        let timescale = MLX.pow(baseFrequency, freqExponents)
        let sinusoidInp = positions[0..., 0..., d..<(d+1)].asType(.float32) / timescale
        let cosD = MLX.cos(sinusoidInp)
        let sinD = MLX.sin(sinusoidInp)
        let cosDFull = expandedDimensions(
            concatenated([cosD, cosD], axis: -1), axis: 2).asType(inputs.dtype)
        let sinDFull = expandedDimensions(
            concatenated([sinD, sinD], axis: -1), axis: 2).asType(inputs.dtype)

        let yPart = xPart * cosDFull + rotateHalf(xPart) * sinDFull
        resultParts.append(yPart)
    }

    return concatenated(resultParts, axis: -1)
}

/// rotate_half: [-x2, x1] matching PyTorch's convention
private func rotateHalf(_ x: MLXArray) -> MLXArray {
    let half = x.dim(-1) / 2
    let x1 = x[0..., 0..., 0..., ..<half]
    let x2 = x[0..., 0..., 0..., half...]
    return concatenated([-x2, x1], axis: -1)
}

// MARK: - Vision Model Components

private class Gemma4VisionAttention: Module {
    @ModuleInfo(key: "q_proj") var queryProj: Linear
    @ModuleInfo(key: "k_proj") var keyProj: Linear
    @ModuleInfo(key: "v_proj") var valueProj: Linear
    @ModuleInfo(key: "o_proj") var outputProj: Linear

    @ModuleInfo(key: "q_norm") var queryNorm: VisionRMSNorm
    @ModuleInfo(key: "k_norm") var keyNorm: VisionRMSNorm
    @ModuleInfo(key: "_v_norm") var valueNorm: VisionRMSNormNoScale

    let numHeads: Int
    let numKVHeads: Int
    let headDim: Int
    let ropeBaseFrequency: Float

    init(config: Gemma4VisionConfiguration) {
        self.numHeads = config.attentionHeads
        self.numKVHeads = config.kvHeads
        self.headDim = config.headDim
        self.ropeBaseFrequency = config.ropeParameters?.ropeTheta ?? 100.0

        self._queryProj.wrappedValue = Linear(config.hiddenSize, numHeads * headDim, bias: false)
        self._keyProj.wrappedValue = Linear(config.hiddenSize, numKVHeads * headDim, bias: false)
        self._valueProj.wrappedValue = Linear(config.hiddenSize, numKVHeads * headDim, bias: false)
        self._outputProj.wrappedValue = Linear(numHeads * headDim, config.hiddenSize, bias: false)

        self._queryNorm.wrappedValue = VisionRMSNorm(dimensions: headDim)
        self._keyNorm.wrappedValue = VisionRMSNorm(dimensions: headDim)
        self._valueNorm.wrappedValue = VisionRMSNormNoScale()
    }

    func callAsFunction(
        _ x: MLXArray, positions: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode = .none
    ) -> MLXArray {
        let (B, L, _) = (x.dim(0), x.dim(1), x.dim(2))

        var q = queryProj(x).reshaped(B, L, numHeads, headDim)
        var k = keyProj(x).reshaped(B, L, numKVHeads, headDim)
        var v = valueProj(x).reshaped(B, L, numKVHeads, headDim)

        q = queryNorm(q)
        k = keyNorm(k)
        v = valueNorm(v)

        // Apply 2D RoPE
        q = applyMultidimensionalRoPE(q, positions: positions, baseFrequency: ropeBaseFrequency)
        k = applyMultidimensionalRoPE(k, positions: positions, baseFrequency: ropeBaseFrequency)

        // Transpose to [B, H, L, D] for SDPA
        q = q.transposed(0, 2, 1, 3)
        k = k.transposed(0, 2, 1, 3)
        v = v.transposed(0, 2, 1, 3)

        let output = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v,
            scale: 1.0,
            mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        return outputProj(output)
    }
}

private class Gemma4VisionMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(config: Gemma4VisionConfiguration) {
        self._gateProj.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        self._upProj.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        self._downProj.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(geluApproximate(gateProj(x)) * upProj(x))
    }
}

private class Gemma4VisionTransformerBlock: Module {
    @ModuleInfo(key: "self_attn") var selfAttention: Gemma4VisionAttention
    @ModuleInfo var mlp: Gemma4VisionMLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: Gemma4RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: Gemma4RMSNorm
    @ModuleInfo(key: "pre_feedforward_layernorm") var preFeedforwardLayerNorm: Gemma4RMSNorm
    @ModuleInfo(key: "post_feedforward_layernorm") var postFeedforwardLayerNorm: Gemma4RMSNorm

    init(config: Gemma4VisionConfiguration) {
        self._selfAttention.wrappedValue = Gemma4VisionAttention(config: config)
        self.mlp = Gemma4VisionMLP(config: config)
        self._inputLayerNorm.wrappedValue = Gemma4RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = Gemma4RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._preFeedforwardLayerNorm.wrappedValue = Gemma4RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postFeedforwardLayerNorm.wrappedValue = Gemma4RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(
        _ x: MLXArray, positions: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode = .none
    ) -> MLXArray {
        let normed = inputLayerNorm(x)
        let attnOut = selfAttention(normed, positions: positions, mask: mask)
        let h = x + postAttentionLayerNorm(attnOut)

        let normedH = preFeedforwardLayerNorm(h)
        let ffwOut = mlp(normedH)
        return h + postFeedforwardLayerNorm(ffwOut)
    }
}

/// One-hot encoding utility
private func oneHot(_ indices: MLXArray, numClasses: Int) -> MLXArray {
    return (expandedDimensions(indices, axis: -1) .== MLXArray(0..<numClasses)).asType(.float32)
}

private class Gemma4VisionPatchEmbedder: Module {
    @ModuleInfo(key: "input_proj") var inputProj: Linear
    @ModuleInfo(key: "position_embedding_table") var positionEmbeddingTable: MLXArray

    let hiddenSize: Int
    let patchSize: Int
    let positionEmbeddingSize: Int

    init(config: Gemma4VisionConfiguration) {
        self.hiddenSize = config.hiddenSize
        self.patchSize = config.patchSize
        self.positionEmbeddingSize = config.positionEmbeddingSize

        self._inputProj.wrappedValue = Linear(3 * patchSize * patchSize, hiddenSize, bias: false)
        self._positionEmbeddingTable.wrappedValue = MLXArray.ones(
            [2, positionEmbeddingSize, hiddenSize])
    }

    /// Compute position embeddings from patch grid positions
    func positionEmbeddings(_ patchPositions: MLXArray, paddingPositions: MLXArray) -> MLXArray {
        let oh = oneHot(patchPositions, numClasses: positionEmbeddingSize)
        // [B, numPatches, 2, posSize] -> [B, 2, numPatches, posSize]
        let ohTransposed = oh.transposed(0, 2, 1, 3).asType(positionEmbeddingTable.dtype)
        // [B, 2, numPatches, posSize] @ [2, posSize, hiddenSize] -> sum over dim=1
        var posEmb = matmul(ohTransposed, positionEmbeddingTable)
        posEmb = posEmb.sum(axis: 1) // [B, numPatches, hiddenSize]
        posEmb = MLX.where(
            expandedDimensions(paddingPositions, axis: -1),
            MLXArray(Float(0.0)),
            posEmb
        )
        return posEmb
    }

    /// Patchify: pixel_values [B, C, H, W] -> patches [B, numPatches, C*p*p]
    func patchify(_ pixelValues: MLXArray) -> MLXArray {
        let (B, C, H, W) = (pixelValues.dim(0), pixelValues.dim(1), pixelValues.dim(2), pixelValues.dim(3))
        let p = patchSize
        let pH = H / p
        let pW = W / p

        // Reshape: [B, C, pH, p, pW, p] -> permute to [B, pH, pW, p, p, C] -> [B, pH*pW, p*p*C]
        var patches = pixelValues.reshaped(B, C, pH, p, pW, p)
        patches = patches.transposed(0, 2, 4, 3, 5, 1) // [B, pH, pW, p, p, C]
        patches = patches.reshaped(B, pH * pW, C * p * p)
        patches = 2 * (patches - 0.5)
        return inputProj(patches.asType(inputProj.weight.dtype))
    }

    func callAsFunction(
        _ pixelValues: MLXArray,
        patchPositions: MLXArray,
        paddingPositions: MLXArray
    ) -> MLXArray {
        let hiddenStates = patchify(pixelValues)
        let posEmb = positionEmbeddings(patchPositions, paddingPositions: paddingPositions)
        return hiddenStates + posEmb
    }
}

private class Gemma4VisionPooler: Module {
    let hiddenSize: Int
    let defaultOutputLength: Int
    let rootHiddenSize: Float

    init(config: Gemma4VisionConfiguration) {
        self.hiddenSize = config.hiddenSize
        self.defaultOutputLength = config.defaultOutputLength
        self.rootHiddenSize = sqrtf(Float(config.hiddenSize))
    }

    func avgPoolByPositions(_ x: MLXArray, patchPositions: MLXArray, length: Int) -> (MLXArray, MLXArray) {
        let inputSeqLen = x.dim(1)
        let k = Int(sqrtf(Float(inputSeqLen / length)))
        let kSquared = Float(k * k)

        let clamped = clip(patchPositions, min: 0)
        let maxX = expandedDimensions(clamped[0..., 0..., 0..<1].max(axis: 1), axis: -1) + 1
        var kernelIdxs = MLX.floor(clamped.asType(.float32) / Float(k)).asType(.int32)
        kernelIdxs = kernelIdxs[0..., 0..., 0..<1] + (maxX / k) * kernelIdxs[0..., 0..., 1..<2]
        let weights = oneHot(kernelIdxs.squeezed(axis: -1), numClasses: length) / kSquared

        // output = einsum("bLl,bLd->bld", weights, x)
        let output = matmul(weights.transposed(0, 2, 1), x).asType(x.dtype)
        let mask = MLX.logicalNot(MLX.all(weights .== 0, axis: 1))
        return (output, mask)
    }

    func callAsFunction(
        _ hiddenStates: MLXArray,
        patchPositions: MLXArray,
        paddingPositions: MLXArray,
        outputLength: Int? = nil
    ) -> (MLXArray, MLXArray) {
        // Zero out padding tokens before pooling
        var hs = MLX.where(
            expandedDimensions(paddingPositions, axis: -1),
            MLXArray(Float(0.0)),
            hiddenStates
        )

        let length = outputLength ?? defaultOutputLength
        var mask: MLXArray
        if hs.dim(1) == length {
            mask = paddingPositions
        } else {
            (hs, mask) = avgPoolByPositions(hs, patchPositions: patchPositions, length: length)
        }
        hs = hs * rootHiddenSize
        return (hs, mask)
    }
}

private class Gemma4VisionEncoder: Module {
    @ModuleInfo var layers: [Gemma4VisionTransformerBlock]

    init(config: Gemma4VisionConfiguration) {
        self._layers.wrappedValue = (0..<config.hiddenLayers).map { _ in
            Gemma4VisionTransformerBlock(config: config)
        }
    }

    func callAsFunction(
        _ hiddenStates: MLXArray, positions: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode
    ) -> MLXArray {
        var h = hiddenStates
        for layer in layers {
            h = layer(h, positions: positions, mask: mask)
        }
        return h
    }
}

private class Gemma4VisionModel: Module {
    @ModuleInfo(key: "patch_embedder") var patchEmbedder: Gemma4VisionPatchEmbedder
    @ModuleInfo var encoder: Gemma4VisionEncoder
    @ModuleInfo var pooler: Gemma4VisionPooler

    @ModuleInfo(key: "std_bias") var stdBias: MLXArray?
    @ModuleInfo(key: "std_scale") var stdScale: MLXArray?

    let config: Gemma4VisionConfiguration
    let maxPatches: Int

    init(config: Gemma4VisionConfiguration) {
        self.config = config
        self.maxPatches = config.defaultOutputLength * config.poolingKernelSize * config.poolingKernelSize

        self._patchEmbedder.wrappedValue = Gemma4VisionPatchEmbedder(config: config)
        self.encoder = Gemma4VisionEncoder(config: config)
        self.pooler = Gemma4VisionPooler(config: config)

        if config.standardize {
            self._stdBias.wrappedValue = MLXArray.zeros([config.hiddenSize])
            self._stdScale.wrappedValue = MLXArray.ones([config.hiddenSize])
        } else {
            self._stdBias.wrappedValue = nil
            self._stdScale.wrappedValue = nil
        }
    }

    /// Compute patch positions and padding mask for a single image
    func patchPositionsSingle(H: Int, W: Int) -> (positions: MLXArray, paddingMask: MLXArray, numReal: Int) {
        let p = config.patchSize
        let pH = H / p
        let pW = W / p
        let numPatches = pH * pW

        // Build grid positions
        var posArray: [[Int32]] = []
        for y in 0..<pH {
            for x in 0..<pW {
                posArray.append([Int32(x), Int32(y)])
            }
        }

        let numPadding = maxPatches - numPatches
        if numPadding > 0 {
            for _ in 0..<numPadding {
                posArray.append([-1, -1])
            }
        }

        let positions = MLXArray(posArray.flatMap { $0 }).reshaped(maxPatches, 2)

        var paddingArray = [Bool](repeating: false, count: maxPatches)
        if numPadding > 0 {
            for i in numPatches..<maxPatches {
                paddingArray[i] = true
            }
        }
        let paddingMask = MLXArray(paddingArray)

        return (positions, paddingMask, min(numPatches, maxPatches))
    }

    func callAsFunction(_ pixelValues: MLXArray) -> MLXArray {
        let (B, _, H, W) = (pixelValues.dim(0), pixelValues.dim(1), pixelValues.dim(2), pixelValues.dim(3))

        let numReal = min((H / config.patchSize) * (W / config.patchSize), maxPatches)
        let (positions, paddingMask, _) = patchPositionsSingle(H: H, W: W)

        // Tile for batch
        let patchPositions = tiled(expandedDimensions(positions, axis: 0), repetitions: [B, 1, 1])
        let paddingPositions = tiled(expandedDimensions(paddingMask, axis: 0), repetitions: [B, 1])

        var inputsEmbeds = patchEmbedder(
            pixelValues,
            patchPositions: patchPositions[0..., ..<numReal, 0...],
            paddingPositions: paddingPositions[0..., ..<numReal]
        )

        let numPadding = maxPatches - numReal
        if numPadding > 0 {
            let padEmbeds = MLXArray.zeros([B, numPadding, config.hiddenSize]).asType(inputsEmbeds.dtype)
            inputsEmbeds = concatenated([inputsEmbeds, padEmbeds], axis: 1)
        }

        // Build bidirectional attention mask [B, 1, L, L]
        let validMask = MLX.logicalNot(paddingPositions)
        let attnMask2d = expandedDimensions(validMask, axis: 1) * expandedDimensions(validMask, axis: 2)
        let negInf = MLXArray(Float(-1e9)).asType(inputsEmbeds.dtype)
        let zeroVal = MLXArray(Float(0.0)).asType(inputsEmbeds.dtype)
        let attnMaskValues = MLX.where(attnMask2d, zeroVal, negInf)
        let attnMask = expandedDimensions(attnMaskValues, axis: 1)

        var hiddenStates = encoder(inputsEmbeds, positions: patchPositions, mask: .array(attnMask))

        let (pooled, poolMask) = pooler(
            hiddenStates, patchPositions: patchPositions,
            paddingPositions: paddingPositions
        )

        // Extract valid (non-padding) tokens
        let validMaskForPool: MLXArray
        if poolMask.dim(1) == config.defaultOutputLength {
            validMaskForPool = poolMask
        } else {
            validMaskForPool = MLX.logicalNot(poolMask)
        }

        // For simplicity, take all valid tokens per batch item and concatenate
        // In practice with single-image batches, this is straightforward
        var allReal: [MLXArray] = []
        for i in 0..<B {
            let maskI = validMaskForPool[i]
            let nValid = maskI.asType(.int32).sum().item(Int.self)
            allReal.append(pooled[i, ..<nValid])
        }

        hiddenStates = expandedDimensions(concatenated(allReal, axis: 0), axis: 0)

        if config.standardize, let bias = stdBias, let scale = stdScale {
            hiddenStates = (hiddenStates - bias) * scale
        }

        return hiddenStates
    }
}

// MARK: - Multimodal Embedder

/// Projects soft tokens from vision into language model space.
/// Replaces Gemma3's avg-pool + einsum projector with a linear projection + RMSNorm.
private class Gemma4MultimodalEmbedder: Module {
    @ModuleInfo(key: "embedding_projection") var embeddingProjection: Linear
    @ModuleInfo(key: "embedding_pre_projection_norm") var embeddingPreProjectionNorm: RMSNormNoScale

    init(embeddingDim: Int, textHiddenSize: Int, eps: Float = 1e-6) {
        self._embeddingProjection.wrappedValue = Linear(embeddingDim, textHiddenSize, bias: false)
        self._embeddingPreProjectionNorm.wrappedValue = RMSNormNoScale(eps: eps)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let normed = embeddingPreProjectionNorm(x)
        return embeddingProjection(normed)
    }
}

// MARK: - Masked Scatter

private func maskedScatter(
    inputTensor: MLXArray,
    mask: MLXArray,
    source: MLXArray
) -> MLXArray {
    let maskFlat = mask.flattened().asType(.int32)
    let indices = cumsum(maskFlat, axis: 0) - 1
    let sourceFlat = source.flattened()
    let aligned = sourceFlat[indices % sourceFlat.dim(0)]
    let result = MLX.where(maskFlat, aligned, inputTensor.flattened())
    return result.reshaped(inputTensor.shape)
}

// MARK: - Gemma 4 VLM Model

public class Gemma4: Module, VLMModel, KVCacheDimensionProvider {
    @ModuleInfo(key: "vision_tower") private var visionTower: Gemma4VisionModel
    @ModuleInfo(key: "language_model") private var languageModel: Gemma4LanguageModel
    @ModuleInfo(key: "embed_vision") private var embedVision: Gemma4MultimodalEmbedder

    public let config: Gemma4Configuration

    public var vocabularySize: Int { config.vocabularySize }
    public var kvHeads: [Int] { languageModel.kvHeads }

    public func newCache(parameters: GenerateParameters?) -> [any KVCache] {
        return languageModel.newCache(parameters: parameters)
    }

    public init(_ config: Gemma4Configuration) {
        self.config = config

        self._visionTower.wrappedValue = Gemma4VisionModel(config: config.visionConfiguration)
        self._languageModel.wrappedValue = Gemma4LanguageModel(config.textConfiguration)
        self._embedVision.wrappedValue = Gemma4MultimodalEmbedder(
            embeddingDim: config.visionConfiguration.hiddenSize,
            textHiddenSize: config.textConfiguration.hiddenSize,
            eps: config.visionConfiguration.rmsNormEps
        )
    }

    private func getInputEmbeddings(
        inputIds: MLXArray? = nil,
        pixelValues: MLXArray? = nil
    ) -> (MLXArray, MLXArray?) {
        guard let pixelValues else {
            let embeds = languageModel.model.embedTokens(inputIds!)
            return (embeds, nil)
        }

        var inputsEmbeds = languageModel.model.embedTokens(inputIds!)
        inputsEmbeds = inputsEmbeds * MLXArray(languageModel.model.embedScale).asType(inputsEmbeds.dtype)

        // Get per-layer inputs for PLE (mask out image tokens)
        var perLayerInputs: MLXArray? = nil
        if languageModel.model.hiddenSizePerLayerInput > 0 {
            let imageMaskIds = inputIds! .== MLXArray(config.imageTokenId)
            let textMask = MLX.logicalNot(imageMaskIds)
            let perLayerTokens = MLX.where(textMask, inputIds!, MLXArray.zeros(like: inputIds!))
            perLayerInputs = languageModel.model.getPerLayerInputs(perLayerTokens)
        }

        // Process image through vision tower + embedder
        let imageFeatures = visionTower(pixelValues)
        let projectedFeatures = embedVision(imageFeatures).asType(inputsEmbeds.dtype)

        // Scatter image features into embedding
        let imageMask = inputIds! .== MLXArray(config.imageTokenId)
        let embedDim = inputsEmbeds.dim(2)
        var imageMaskExpanded = expandedDimensions(imageMask, axis: -1)
        imageMaskExpanded = repeated(imageMaskExpanded, count: embedDim, axis: -1)

        inputsEmbeds = maskedScatter(
            inputTensor: inputsEmbeds,
            mask: imageMaskExpanded,
            source: projectedFeatures
        )

        return (inputsEmbeds, perLayerInputs)
    }

    public func prepare(_ input: LMInput, cache: [any KVCache], windowSize: Int?) throws
        -> PrepareResult
    {
        guard let imagePixels = input.image?.pixels else {
            // Text-only input
            let convertedCache = cache.compactMap { $0 as KVCache }
            let result = languageModel(
                input.text.tokens, cache: convertedCache, inputEmbedding: nil, mask: nil)
            return .logits(result)
        }

        let (inputEmbeddings, perLayerInputs) = getInputEmbeddings(
            inputIds: input.text.tokens,
            pixelValues: imagePixels
        )

        let convertedCache = cache.compactMap { $0 as KVCache }
        let result = languageModel(
            nil,
            cache: convertedCache,
            inputEmbedding: inputEmbeddings,
            mask: nil,
            perLayerInputs: perLayerInputs
        )

        return .logits(result)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [any KVCache]?) -> MLXArray {
        return languageModel(inputs, cache: cache).logits
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        // Remap weights: "model.xxx" -> "xxx", and "language_model.xxx" -> "language_model.model.xxx"
        var remapped = [String: MLXArray]()
        for (k, v) in weights {
            var newKey = k
            if newKey.hasPrefix("model.") {
                newKey = String(newKey.dropFirst(6))
            }
            if newKey.hasPrefix("language_model.") && !newKey.hasPrefix("language_model.model.") {
                let rest = String(newKey.dropFirst(15))
                newKey = "language_model.model." + rest
            }
            remapped[newKey] = v
        }

        // Handle language model sanitization (quantization, weight tying, etc.)
        let processedWeights = languageModel.sanitize(
            weights: remapped, quantizationConfig: config.quantization)

        return processedWeights
    }
}

// MARK: - Processor

public struct Gemma4Processor: UserInputProcessor {
    private let config: Gemma4ProcessorConfiguration
    private let tokenizer: any Tokenizer

    public init(_ config: Gemma4ProcessorConfiguration, tokenizer: any Tokenizer) {
        self.config = config
        self.tokenizer = tokenizer
    }

    /// Aspect-ratio preserving resize that fits within patch budget
    private func aspectRatioPreservingResize(_ image: CIImage) -> CIImage {
        let width = Int(image.extent.width)
        let height = Int(image.extent.height)
        let patchSize = config.patchSize
        let poolingKernelSize = config.poolingKernelSize
        let maxSoftTokens = config.maxSoftTokens
        let maxPatches = maxSoftTokens * poolingKernelSize * poolingKernelSize
        let targetPx = maxPatches * patchSize * patchSize
        let factor = sqrt(Double(targetPx) / Double(height * width))
        let sideMult = poolingKernelSize * patchSize

        var targetHeight = Int(floor(factor * Double(height) / Double(sideMult))) * sideMult
        var targetWidth = Int(floor(factor * Double(width) / Double(sideMult))) * sideMult

        let maxSideLength = (maxPatches / (poolingKernelSize * poolingKernelSize)) * sideMult

        if targetHeight == 0 && targetWidth == 0 {
            targetHeight = sideMult
            targetWidth = sideMult
        } else if targetHeight == 0 {
            targetHeight = sideMult
            targetWidth = min(Int(floor(Double(width) / Double(height))) * sideMult, maxSideLength)
        } else if targetWidth == 0 {
            targetWidth = sideMult
            targetHeight = min(Int(floor(Double(height) / Double(width))) * sideMult, maxSideLength)
        }

        if targetHeight == height && targetWidth == width {
            return image
        }

        return MediaProcessing.resampleBicubic(
            image, to: CGSize(width: targetWidth, height: targetHeight))
    }

    public func preprocess(images: [CIImage], processing: UserInput.Processing?) throws -> (
        MLXArray, THW
    ) {
        let processedImages = images.map { image -> MLXArray in
            let srgbImage = MediaProcessing.inSRGBToneCurveSpace(image)
            let resizedImage = aspectRatioPreservingResize(srgbImage)

            // Rescale to [0, 1] then convert to channel-first MLXArray
            let rescaledImage = MediaProcessing.apply(resizedImage, processing: UserInput.Processing())
            let array = MediaProcessing.asMLXArray(rescaledImage)
            // array is [1, H, W, C], we need [1, C, H, W]
            return array.transposed(0, 3, 1, 2)
        }

        let pixelValues = concatenated(processedImages)

        let H = Int(pixelValues.dim(2))
        let W = Int(pixelValues.dim(3))

        return (pixelValues, THW(images.count, H, W))
    }

    public func prepare(input: UserInput) async throws -> LMInput {
        let messages = Qwen2VLMessageGenerator().generate(from: input)

        var promptTokens = try tokenizer.applyChatTemplate(
            messages: messages, tools: input.tools,
            additionalContext: input.additionalContext)

        var processedImage: LMInput.ProcessedImage?

        if !input.images.isEmpty {
            let imagePixelsAndFrames = try input.images.map {
                try preprocess(images: [$0.asCIImage()], processing: input.processing)
            }

            let imagePixelsConcatenated = concatenated(imagePixelsAndFrames.map { $0.0 })
            processedImage = LMInput.ProcessedImage(
                pixels: imagePixelsConcatenated,
                frames: imagePixelsAndFrames.map { $0.1 }
            )

            // Compute number of soft tokens per image
            let patchSize = config.patchSize
            let poolingKernelSize = config.poolingKernelSize

            // Expand image tokens in prompt
            let boiTokenId = 255999  // <start_of_image>
            let imageTokenId = 258880 // <image_soft_token>
            let eoiTokenId = 258882  // <end_of_image>

            var expandedTokens: [Int] = []
            var imageIdx = 0

            for token in promptTokens {
                if token == boiTokenId {
                    // Replace with boi + N image tokens + eoi
                    expandedTokens.append(boiTokenId)

                    let numSoftTokens: Int
                    if imageIdx < imagePixelsAndFrames.count {
                        let frame = imagePixelsAndFrames[imageIdx].1
                        let H = frame.h
                        let W = frame.w
                        let numPatches = (H / patchSize) * (W / patchSize)
                        numSoftTokens = numPatches / (poolingKernelSize * poolingKernelSize)
                    } else {
                        numSoftTokens = config.maxSoftTokens
                    }

                    expandedTokens.append(contentsOf: Array(repeating: imageTokenId, count: numSoftTokens))
                    expandedTokens.append(eoiTokenId)
                    imageIdx += 1
                } else {
                    expandedTokens.append(token)
                }
            }

            promptTokens = expandedTokens
        }

        let promptArray = MLXArray(promptTokens).expandedDimensions(axis: 0)
        let mask = ones(like: promptArray).asType(.int8)
        return LMInput(
            text: .init(tokens: promptArray, mask: mask),
            image: processedImage
        )
    }
}

// MARK: - Processor Configuration

public struct Gemma4ProcessorConfiguration: Codable, Sendable {
    public let processorClass: String
    public let patchSize: Int
    public let maxSoftTokens: Int
    public let poolingKernelSize: Int

    public let doResize: Bool
    public let doRescale: Bool
    public let doNormalize: Bool

    public let imageSeqLength: Int?

    enum CodingKeys: String, CodingKey {
        case processorClass = "processor_class"
        case patchSize = "patch_size"
        case maxSoftTokens = "max_soft_tokens"
        case poolingKernelSize = "pooling_kernel_size"
        case doResize = "do_resize"
        case doRescale = "do_rescale"
        case doNormalize = "do_normalize"
        case imageSeqLength = "image_seq_length"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        processorClass = try container.decode(String.self, forKey: .processorClass)
        patchSize = try container.decodeIfPresent(Int.self, forKey: .patchSize) ?? 16
        maxSoftTokens = try container.decodeIfPresent(Int.self, forKey: .maxSoftTokens) ?? 280
        poolingKernelSize = try container.decodeIfPresent(Int.self, forKey: .poolingKernelSize) ?? 3
        doResize = try container.decodeIfPresent(Bool.self, forKey: .doResize) ?? true
        doRescale = try container.decodeIfPresent(Bool.self, forKey: .doRescale) ?? true
        doNormalize = try container.decodeIfPresent(Bool.self, forKey: .doNormalize) ?? false
        imageSeqLength = try container.decodeIfPresent(Int.self, forKey: .imageSeqLength)
    }
}

// MARK: - LoRA Extension

extension Gemma4: LoRAModel {
    public var loraLayers: [Module] {
        languageModel.model.layers
    }
}
