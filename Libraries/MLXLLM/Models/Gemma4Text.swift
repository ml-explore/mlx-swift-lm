//
//  Gemma4Text.swift
//  mlx-swift-lm
//
//  Gemma 4 text model — rewritten from Apple's canonical mlx-lm reference:
//  https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/gemma4_text.py
//
//  Architecture features:
//  - Explicit layer_types (sliding_attention / full_attention)
//  - Per-layer embeddings (PLE) with ScaledLinear projection
//  - ProportionalRoPE for global attention layers
//  - Global head dimension (different from sliding head_dim)
//  - KV cache sharing for final layers via previous_kvs mapping
//  - Final logit soft-capping
//  - Double-wide MLP for KV-shared layers
//  - MoE support (26B model)
//  - K-eq-V attention (26B/31B models)
//

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
    let numGlobalKeyValueHeads: Int?
    let slidingWindow: Int
    let slidingWindowPattern: Int
    let maxPositionEmbeddings: Int
    let finalLogitSoftcapping: Float?
    let numKvSharedLayers: Int
    let useDoubleWideMlp: Bool
    let tieWordEmbeddings: Bool
    let attentionKEqV: Bool

    // Per-layer embeddings
    let hiddenSizePerLayerInput: Int?
    let vocabSizePerLayerInput: Int?

    // Layer types
    let layerTypes: [String]?

    // RoPE parameters per attention type
    let ropeParameters: RopeParametersConfig?

    // MoE
    let enableMoeBlock: Bool
    let numExperts: Int?
    let topKExperts: Int?
    let moeIntermediateSize: Int?

    struct RopeParametersConfig: Codable {
        let fullAttention: RopeConfig?
        let slidingAttention: RopeConfig?

        enum CodingKeys: String, CodingKey {
            case fullAttention = "full_attention"
            case slidingAttention = "sliding_attention"
        }
    }

    struct RopeConfig: Codable {
        let ropeTheta: Float?
        let ropeType: String?
        let partialRotaryFactor: Float?
        let factor: Float?

        enum CodingKeys: String, CodingKey {
            case ropeTheta = "rope_theta"
            case ropeType = "rope_type"
            case partialRotaryFactor = "partial_rotary_factor"
            case factor
        }
    }

    // Derived helpers
    var slidingRopeTheta: Float {
        ropeParameters?.slidingAttention?.ropeTheta ?? 10_000.0
    }

    var globalRopeTheta: Float {
        ropeParameters?.fullAttention?.ropeTheta ?? 1_000_000.0
    }

    var globalPartialRotaryFactor: Float {
        ropeParameters?.fullAttention?.partialRotaryFactor ?? 1.0
    }

    /// Resolved layer types array — computed from config or default pattern
    var resolvedLayerTypes: [String] {
        if let types = layerTypes { return types }
        let pattern =
            Array(repeating: "sliding_attention", count: slidingWindowPattern - 1)
            + ["full_attention"]
        let repeated = Array(
            (0..<((hiddenLayers / pattern.count) + 1)).flatMap { _ in pattern }
        )
        return Array(repeated.prefix(hiddenLayers))
    }

    func isGlobalLayer(_ index: Int) -> Bool {
        resolvedLayerTypes[index] == "full_attention"
    }

    func isKvSharedLayer(_ index: Int) -> Bool {
        index >= (hiddenLayers - numKvSharedLayers)
    }

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
        case numGlobalKeyValueHeads = "num_global_key_value_heads"
        case slidingWindow = "sliding_window"
        case slidingWindowPattern = "sliding_window_pattern"
        case maxPositionEmbeddings = "max_position_embeddings"
        case finalLogitSoftcapping = "final_logit_softcapping"
        case numKvSharedLayers = "num_kv_shared_layers"
        case useDoubleWideMlp = "use_double_wide_mlp"
        case tieWordEmbeddings = "tie_word_embeddings"
        case attentionKEqV = "attention_k_eq_v"
        case hiddenSizePerLayerInput = "hidden_size_per_layer_input"
        case vocabSizePerLayerInput = "vocab_size_per_layer_input"
        case layerTypes = "layer_types"
        case ropeParameters = "rope_parameters"
        case enableMoeBlock = "enable_moe_block"
        case numExperts = "num_experts"
        case topKExperts = "top_k_experts"
        case moeIntermediateSize = "moe_intermediate_size"
    }

    enum VLMCodingKeys: String, CodingKey {
        case textConfig = "text_config"
    }

    public init(from decoder: Swift.Decoder) throws {
        let nestedContainer = try decoder.container(keyedBy: VLMCodingKeys.self)
        let container =
            if nestedContainer.contains(.textConfig) {
                try nestedContainer.nestedContainer(keyedBy: CodingKeys.self, forKey: .textConfig)
            } else {
                try decoder.container(keyedBy: CodingKeys.self)
            }

        modelType = try container.decode(String.self, forKey: .modelType)
        hiddenSize = try container.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 1536
        hiddenLayers = try container.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 35
        intermediateSize =
            try container.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 6144
        attentionHeads = try container.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 8
        headDim = try container.decodeIfPresent(Int.self, forKey: .headDim) ?? 256
        globalHeadDim = try container.decodeIfPresent(Int.self, forKey: .globalHeadDim) ?? 512
        rmsNormEps = try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        vocabularySize =
            try container.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 262144
        kvHeads = try container.decodeIfPresent(Int.self, forKey: .kvHeads) ?? 1
        numGlobalKeyValueHeads = try container.decodeIfPresent(
            Int.self, forKey: .numGlobalKeyValueHeads)
        slidingWindow = try container.decodeIfPresent(Int.self, forKey: .slidingWindow) ?? 512
        slidingWindowPattern =
            try container.decodeIfPresent(Int.self, forKey: .slidingWindowPattern) ?? 5
        maxPositionEmbeddings =
            try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 131072
        finalLogitSoftcapping = try container.decodeIfPresent(
            Float.self, forKey: .finalLogitSoftcapping)
        numKvSharedLayers =
            try container.decodeIfPresent(Int.self, forKey: .numKvSharedLayers) ?? 0
        useDoubleWideMlp =
            try container.decodeIfPresent(Bool.self, forKey: .useDoubleWideMlp) ?? false
        tieWordEmbeddings =
            try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? true
        attentionKEqV =
            try container.decodeIfPresent(Bool.self, forKey: .attentionKEqV) ?? false
        hiddenSizePerLayerInput = try container.decodeIfPresent(
            Int.self, forKey: .hiddenSizePerLayerInput)
        vocabSizePerLayerInput = try container.decodeIfPresent(
            Int.self, forKey: .vocabSizePerLayerInput)
        layerTypes = try container.decodeIfPresent([String].self, forKey: .layerTypes)
        ropeParameters = try container.decodeIfPresent(
            RopeParametersConfig.self, forKey: .ropeParameters)
        enableMoeBlock =
            try container.decodeIfPresent(Bool.self, forKey: .enableMoeBlock) ?? false
        numExperts = try container.decodeIfPresent(Int.self, forKey: .numExperts)
        topKExperts = try container.decodeIfPresent(Int.self, forKey: .topKExperts)
        moeIntermediateSize = try container.decodeIfPresent(
            Int.self, forKey: .moeIntermediateSize)
    }
}

// MARK: - RMSNorm (standard, weight used directly — NOT Gemma's 1+weight convention)

/// Standard RMSNorm matching Python `nn.RMSNorm`: `rms_norm(x, weight, eps)`.
/// Gemma 4 weights (converted via mlx_vlm) store norm weights in direct format,
/// not the Gemma 2/3 convention where `1+weight` is used.
class Gemma4RMSNorm: Module, UnaryLayer {
    let weight: MLXArray
    let eps: Float

    init(dimensions: Int, eps: Float = 1e-6) {
        self.weight = MLXArray.ones([dimensions])
        self.eps = eps
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(x, weight: weight, eps: eps)
    }
}

// MARK: - RMSNorm without learnable weight (for v_norm)

class RMSNormNoScale: Module {
    let eps: Float

    init(eps: Float = 1e-6) {
        self.eps = eps
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(x, weight: MLXArray.mlxNone, eps: eps)
    }
}

// MARK: - ScaledLinear (linear with output scaling)

/// Python: `class ScaledLinear` — `(x @ weight.T) * scalar`
/// Used for `per_layer_model_projection` where scalar = hidden_size^-0.5
class ScaledLinear: Module {
    var weight: MLXArray
    let scalar: Float

    init(inputDimensions: Int, outputDimensions: Int, scalar: Float) {
        self.weight = MLXArray.zeros([outputDimensions, inputDimensions])
        self.scalar = scalar
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        matmul(x, weight.T) * scalar
    }
}

// MARK: - ProportionalRoPE for global attention layers

/// Not a Module — freqs are computed, not loaded from weights.
/// Matches Python's `initialize_rope` → `ProportionalRoPE` in rope_utils.py.
final class ProportionalRoPE {
    let dims: Int
    let traditional: Bool
    let rotatedDims: Int
    let freqs: MLXArray?

    init(
        dims: Int, traditional: Bool = false, base: Float = 10000.0,
        factor: Float = 1.0, partialRotaryFactor: Float = 1.0
    ) {
        self.dims = dims
        self.traditional = traditional

        let ropeAngles = Int(partialRotaryFactor * Float(dims) / 2.0)
        self.rotatedDims = 2 * ropeAngles

        if rotatedDims > 0 {
            // Python: exponents = arange(0, rotated_dims, 2) / dims
            let exponents = MLXArray(stride(from: 0, to: Float(rotatedDims), by: 2))
                .asType(.float32) / Float(dims)
            self.freqs = factor * pow(MLXArray(base), exponents)
        } else {
            self.freqs = nil
        }
    }

    func callAsFunction(_ x: MLXArray, offset: Int) -> MLXArray {
        guard rotatedDims > 0, let freqs else { return x }

        let head = x[.ellipsis, 0..<dims]
        let tail = dims < x.dim(-1) ? x[.ellipsis, dims...] : nil

        let half = dims / 2
        let left = head[.ellipsis, 0..<half]
        let right = head[.ellipsis, half..<dims]

        let rotHalf = rotatedDims / 2
        let leftRot = left[.ellipsis, 0..<rotHalf]
        let rightRot = right[.ellipsis, 0..<rotHalf]
        let toRotate = concatenated([leftRot, rightRot], axis: -1)

        let rotated = MLXFast.RoPE(
            toRotate, dimensions: rotatedDims, traditional: traditional,
            base: nil, scale: 1.0, offset: offset, freqs: freqs)

        let newLeft = concatenated([
            rotated[.ellipsis, 0..<rotHalf],
            left[.ellipsis, rotHalf...],
        ], axis: -1)

        let newRight = concatenated([
            rotated[.ellipsis, rotHalf...],
            right[.ellipsis, rotHalf...],
        ], axis: -1)

        let newHead = concatenated([newLeft, newRight], axis: -1)

        if let tail, tail.dim(-1) > 0 {
            return concatenated([newHead, tail], axis: -1)
        }
        return newHead
    }
}

// MARK: - Attention

/// Matches Python `class Attention` exactly.
/// Returns `(output, (keys, values), offset)` to support KV sharing.
class Gemma4Attention: Module {
    let config: Gemma4TextConfiguration
    let layerIdx: Int
    let layerType: String
    let isSliding: Bool
    let headDim: Int
    let nHeads: Int
    let nKVHeads: Int
    let useKEqV: Bool
    let scale: Float

    @ModuleInfo(key: "q_proj") var queryProj: Linear
    @ModuleInfo(key: "k_proj") var keyProj: Linear
    @ModuleInfo(key: "v_proj") var valueProj: Linear?
    @ModuleInfo(key: "o_proj") var outputProj: Linear

    @ModuleInfo(key: "q_norm") var queryNorm: Gemma4RMSNorm
    @ModuleInfo(key: "k_norm") var keyNorm: Gemma4RMSNorm
    @ModuleInfo(key: "v_norm") var valueNorm: RMSNormNoScale

    // Standard RoPE for sliding, nil for global
    @ModuleInfo var rope: OffsetLayer?
    // ProportionalRoPE for global, nil for sliding
    let proportionalRope: ProportionalRoPE?

    init(_ config: Gemma4TextConfiguration, layerIdx: Int) {
        self.config = config
        self.layerIdx = layerIdx
        self.layerType = config.resolvedLayerTypes[layerIdx]
        self.isSliding = layerType == "sliding_attention"

        // Python: head_dim = global_head_dim for full_attention, else head_dim
        self.headDim =
            (!isSliding && config.globalHeadDim > 0)
            ? config.globalHeadDim : config.headDim

        self.nHeads = config.attentionHeads

        // Python: K-eq-V for full attention layers (26B/31B)
        self.useKEqV = config.attentionKEqV && !isSliding
        if useKEqV, let globalKVHeads = config.numGlobalKeyValueHeads {
            self.nKVHeads = globalKVHeads
        } else {
            self.nKVHeads = config.kvHeads
        }

        // Python: self.scale = 1.0
        self.scale = 1.0

        let dim = config.hiddenSize
        self._queryProj.wrappedValue = Linear(dim, nHeads * headDim, bias: false)
        self._keyProj.wrappedValue = Linear(dim, nKVHeads * headDim, bias: false)
        if !useKEqV {
            self._valueProj.wrappedValue = Linear(dim, nKVHeads * headDim, bias: false)
        }
        self._outputProj.wrappedValue = Linear(nHeads * headDim, dim, bias: false)

        self._queryNorm.wrappedValue = Gemma4RMSNorm(
            dimensions: headDim, eps: config.rmsNormEps)
        self._keyNorm.wrappedValue = Gemma4RMSNorm(
            dimensions: headDim, eps: config.rmsNormEps)
        self._valueNorm.wrappedValue = RMSNormNoScale(eps: config.rmsNormEps)

        // RoPE setup per layer type
        if isSliding {
            self.rope = initializeRope(
                dims: headDim, base: config.slidingRopeTheta, traditional: false,
                scalingConfig: nil, maxPositionEmbeddings: nil)
            self.proportionalRope = nil
        } else {
            self.rope = nil
            self.proportionalRope = ProportionalRoPE(
                dims: headDim,
                traditional: false,
                base: config.globalRopeTheta,
                partialRotaryFactor: config.globalPartialRotaryFactor
            )
        }

        super.init()
    }

    private func applyRope(_ x: MLXArray, offset: Int) -> MLXArray {
        if let standardRope = rope {
            return standardRope(x, offset: offset)
        } else if let propRope = proportionalRope {
            return propRope(x, offset: offset)
        }
        return x
    }

    /// Returns `(output, sharedKV, offset)`.
    /// `sharedKV` is non-nil only for non-shared layers with regular (non-quantized) caches,
    /// enabling KV sharing. When caches are quantized, each layer operates independently.
    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache? = nil,
        sharedKV: (MLXArray, MLXArray)? = nil,
        offset inputOffset: Int? = nil
    ) -> (MLXArray, (MLXArray, MLXArray)?, Int) {
        let (B, L, _) = (x.dim(0), x.dim(1), x.dim(2))

        var queries = queryProj(x).reshaped(B, L, nHeads, headDim)
        queries = queryNorm(queries)

        if let (sharedK, sharedV) = sharedKV {
            // KV-shared layer: reuse keys/values from earlier layer (no cache update)
            let offset = inputOffset ?? 0

            queries = queries.transposed(0, 2, 1, 3)
            queries = applyRope(queries, offset: offset)

            let output = MLXFast.scaledDotProductAttention(
                queries: queries, keys: sharedK, values: sharedV,
                scale: scale, mask: mask
            )
            .transposed(0, 2, 1, 3)
            .reshaped(B, L, -1)

            return (outputProj(output), (sharedK, sharedV), offset)
        }

        // Compute own KV
        var keys = keyProj(x).reshaped(B, L, nKVHeads, headDim)
        var values: MLXArray
        if useKEqV {
            values = keys
        } else {
            values = valueProj!(x).reshaped(B, L, nKVHeads, headDim)
        }

        let offset = cache?.offset ?? 0

        keys = keyNorm(keys)
        keys = keys.transposed(0, 2, 1, 3)
        keys = applyRope(keys, offset: offset)

        values = valueNorm(values)
        values = values.transposed(0, 2, 1, 3)

        queries = queries.transposed(0, 2, 1, 3)
        queries = applyRope(queries, offset: offset)

        // Use attentionWithCacheUpdate which handles both regular and quantized caches
        let attnOutput = attentionWithCacheUpdate(
            queries: queries, keys: keys, values: values,
            cache: cache, scale: scale, mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        // For KV sharing: only return shareable KV from regular (non-quantized) caches
        var returnedKV: (MLXArray, MLXArray)? = nil
        if let cache, !(cache is QuantizedKVCacheProtocol) {
            // After attentionWithCacheUpdate, the cache contains the full K/V history.
            // Re-read from cache state for sharing with downstream layers.
            // For KVCacheSimple/RotatingKVCache, the state arrays are [keys, values].
            let state = cache.state
            if state.count >= 2 {
                returnedKV = (state[0], state[1])
            }
        }

        return (outputProj(attnOutput), returnedKV, offset)
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
        downProj(geluApproximate(gateProj(x)) * upProj(x))
    }
}

// MARK: - Decoder Layer

/// Matches Python `class DecoderLayer`.
class Gemma4DecoderLayer: Module {
    let config: Gemma4TextConfiguration
    let layerIdx: Int
    let layerType: String

    @ModuleInfo(key: "self_attn") var selfAttn: Gemma4Attention
    @ModuleInfo var mlp: Gemma4MLP

    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: Gemma4RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: Gemma4RMSNorm
    @ModuleInfo(key: "pre_feedforward_layernorm") var preFeedforwardLayerNorm: Gemma4RMSNorm
    @ModuleInfo(key: "post_feedforward_layernorm") var postFeedforwardLayerNorm: Gemma4RMSNorm

    // Per-layer input gating (PLE)
    let hasPerLayerInput: Bool
    @ModuleInfo(key: "per_layer_input_gate") var perLayerInputGate: Linear?
    @ModuleInfo(key: "per_layer_projection") var perLayerProjection: Linear?
    @ModuleInfo(key: "post_per_layer_input_norm") var postPerLayerInputNorm: Gemma4RMSNorm?

    // Layer scalar
    // swiftlint:disable:next identifier_name
    var layer_scalar: MLXArray

    init(_ config: Gemma4TextConfiguration, layerIdx: Int) {
        self.config = config
        self.layerIdx = layerIdx
        self.layerType = config.resolvedLayerTypes[layerIdx]

        self._selfAttn.wrappedValue = Gemma4Attention(config, layerIdx: layerIdx)

        // Double-wide MLP for KV-shared layers
        let mlpDim: Int
        if config.useDoubleWideMlp && config.isKvSharedLayer(layerIdx) {
            mlpDim = config.intermediateSize * 2
        } else {
            mlpDim = config.intermediateSize
        }
        self.mlp = Gemma4MLP(dimensions: config.hiddenSize, hiddenDimensions: mlpDim)

        self._inputLayerNorm.wrappedValue = Gemma4RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = Gemma4RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._preFeedforwardLayerNorm.wrappedValue = Gemma4RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postFeedforwardLayerNorm.wrappedValue = Gemma4RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)

        // Per-layer input gating
        if let perLayerDim = config.hiddenSizePerLayerInput, perLayerDim > 0 {
            self.hasPerLayerInput = true
            self._perLayerInputGate.wrappedValue = Linear(
                config.hiddenSize, perLayerDim, bias: false)
            self._perLayerProjection.wrappedValue = Linear(
                perLayerDim, config.hiddenSize, bias: false)
            self._postPerLayerInputNorm.wrappedValue = Gemma4RMSNorm(
                dimensions: config.hiddenSize, eps: config.rmsNormEps)
        } else {
            self.hasPerLayerInput = false
        }

        self.layer_scalar = MLXArray([Float(1.0)])

        super.init()
    }

    /// Returns `(output, sharedKV, offset)`.
    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache? = nil,
        perLayerInput: MLXArray? = nil,
        sharedKV: (MLXArray, MLXArray)? = nil,
        offset: Int? = nil
    ) -> (MLXArray, (MLXArray, MLXArray)?, Int) {
        // Attention block
        var residual = x
        var h = inputLayerNorm(x)
        let (attnOut, kvs, newOffset) = selfAttn(
            h, mask: mask, cache: cache, sharedKV: sharedKV, offset: offset)
        h = postAttentionLayerNorm(attnOut)
        h = residual + h

        // MLP block
        residual = h
        h = preFeedforwardLayerNorm(h)
        h = mlp(h)
        h = postFeedforwardLayerNorm(h)
        h = residual + h

        // Per-layer input gating (AFTER attention + MLP)
        if hasPerLayerInput, let signal = perLayerInput,
            let gate = perLayerInputGate, let proj = perLayerProjection,
            let norm = postPerLayerInputNorm
        {
            residual = h
            let gated = geluApproximate(gate(h)) * signal
            let projected = proj(gated)
            let normed = norm(projected)
            h = residual + normed
        }

        // Scale entire block output
        h = h * layer_scalar

        return (h, kvs, newOffset)
    }
}

// MARK: - Gemma4TextModel (inner model)

public class Gemma4Model: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo var layers: [Gemma4DecoderLayer]
    @ModuleInfo var norm: Gemma4RMSNorm

    // Per-layer embeddings (PLE)
    @ModuleInfo(key: "embed_tokens_per_layer") var embedTokensPerLayer: Embedding?
    @ModuleInfo(key: "per_layer_model_projection") var perLayerModelProjection: ScaledLinear?
    // Python uses nn.RMSNorm (standard, with 1+weight) — that's Gemma4RMSNorm
    @ModuleInfo(key: "per_layer_projection_norm") var perLayerProjectionNorm: Gemma4RMSNorm?

    let config: Gemma4TextConfiguration
    let embedScale: Float
    let hiddenSizePerLayerInput: Int

    /// Maps each layer index to the index of the layer whose KV it should share.
    /// For non-shared layers, previousKVs[i] == i.
    let previousKVs: [Int]

    init(_ config: Gemma4TextConfiguration) {
        self.config = config
        self.embedScale = sqrt(Float(config.hiddenSize))
        self.hiddenSizePerLayerInput = config.hiddenSizePerLayerInput ?? 0

        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize,
            dimensions: config.hiddenSize
        )

        // Per-layer embeddings
        if let perLayerDim = config.hiddenSizePerLayerInput, perLayerDim > 0,
            let perLayerVocab = config.vocabSizePerLayerInput
        {
            self._embedTokensPerLayer.wrappedValue = Embedding(
                embeddingCount: perLayerVocab,
                dimensions: config.hiddenLayers * perLayerDim
            )
            self._perLayerModelProjection.wrappedValue = ScaledLinear(
                inputDimensions: config.hiddenSize,
                outputDimensions: config.hiddenLayers * perLayerDim,
                scalar: pow(Float(config.hiddenSize), -0.5)
            )
            // Python line 448-449: self.per_layer_projection_norm = nn.RMSNorm(...)
            // This is standard RMSNorm — in Gemma family that's 1+weight
            self._perLayerProjectionNorm.wrappedValue = Gemma4RMSNorm(
                dimensions: perLayerDim, eps: config.rmsNormEps
            )
        }

        self._layers.wrappedValue = (0..<config.hiddenLayers).map { layerIdx in
            Gemma4DecoderLayer(config, layerIdx: layerIdx)
        }

        self.norm = Gemma4RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)

        // Build KV sharing map (Python lines 458-466)
        var prevKVs = Array(0..<config.hiddenLayers)
        if config.numKvSharedLayers > 0 {
            let n = config.hiddenLayers
            let m = n - config.numKvSharedLayers
            let layerTypes = config.resolvedLayerTypes
            // Find last non-shared layer of each type
            var kvsByType = [String: Int]()
            for i in 0..<m {
                kvsByType[layerTypes[i]] = i
            }
            // Shared layers map to the non-shared layer of the same type
            for j in m..<n {
                if let mapped = kvsByType[layerTypes[j]] {
                    prevKVs[j] = mapped
                }
            }
        }
        self.previousKVs = prevKVs

        super.init()
    }

    /// Python: `_get_per_layer_inputs` — token PLE from per-layer embedding table.
    private func getPerLayerInputs(_ inputIds: MLXArray) -> MLXArray {
        guard let embedPerLayer = embedTokensPerLayer else {
            fatalError("getPerLayerInputs called without per-layer embeddings")
        }
        let perLayerDim = hiddenSizePerLayerInput
        // Python: result = embed_tokens_per_layer(input_ids)
        //         result = result * embed_tokens_per_layer_scale
        let scale = sqrt(Float(perLayerDim))
        var result = embedPerLayer(inputIds) * scale
        // Unflatten: [B, L, num_layers * perLayerDim] → [B, L, num_layers, perLayerDim]
        let shape = result.shape
        result = result.reshaped(shape[0], shape[1], config.hiddenLayers, perLayerDim)
        return result
    }

    /// Python: `_project_per_layer_inputs`
    private func projectPerLayerInputs(
        _ inputEmbeddings: MLXArray, perLayerInputs: MLXArray?
    ) -> MLXArray {
        guard let projection = perLayerModelProjection,
            let projNorm = perLayerProjectionNorm
        else {
            fatalError("projectPerLayerInputs called without projection modules")
        }

        let perLayerDim = hiddenSizePerLayerInput

        // Python: per_layer_projection = self.per_layer_model_projection(input_embeddings)
        // ScaledLinear handles the hidden_size^-0.5 scaling internally
        var perLayerProjection = projection(inputEmbeddings)
        // Unflatten
        let shape = perLayerProjection.shape
        perLayerProjection = perLayerProjection.reshaped(
            shape[0], shape[1], config.hiddenLayers, perLayerDim)
        // Normalize per-layer slices
        perLayerProjection = projNorm(perLayerProjection)

        guard let perLayerInputs else {
            return perLayerProjection
        }

        // Python: (per_layer_projection + per_layer_inputs) * self.per_layer_input_scale
        let inputScale: Float = pow(2.0, -0.5)  // 0.7071
        return (perLayerProjection + perLayerInputs) * inputScale
    }

    func callAsFunction(
        _ inputs: MLXArray, cache: [KVCache?]? = nil
    ) -> MLXArray {
        // Embed and scale
        let inputEmbeddings = embedTokens(inputs)
        var h = inputEmbeddings * MLXArray(embedScale, dtype: .bfloat16).asType(
            inputEmbeddings.dtype)

        // Compute per-layer signals if PLE is available
        var perLayerSignals: [MLXArray?]
        if hiddenSizePerLayerInput > 0, embedTokensPerLayer != nil {
            let tokenPLE = getPerLayerInputs(inputs)
            let combined = projectPerLayerInputs(h, perLayerInputs: tokenPLE)
            // Split into per-layer slices: [B, L, num_layers, dim] → array of [B, L, dim]
            perLayerSignals = (0..<config.hiddenLayers).map { i in
                combined[.ellipsis, i, 0...]
            }
        } else {
            perLayerSignals = Array(repeating: nil, count: config.hiddenLayers)
        }

        // Pad cache list for shared layers (Python lines 568-570)
        var layerCache: [KVCache?]
        if let cache {
            layerCache =
                cache + Array(repeating: nil as KVCache?, count: config.hiddenLayers - cache.count)
        } else {
            layerCache = Array(repeating: nil as KVCache?, count: config.hiddenLayers)
        }

        // Create masks per layer type
        let masks = makeMasks(h: h, cache: layerCache)

        // Apply each layer with KV sharing
        // Python: intermediates stores (kvs, offset) per layer
        var intermediates: [(kvs: (MLXArray, MLXArray)?, offset: Int?)] =
            Array(repeating: (nil, nil), count: config.hiddenLayers)

        for idx in 0..<layers.count {
            let layer = layers[idx]
            let c = layerCache[idx]
            let mask = masks[idx]
            let prevIdx = previousKVs[idx]
            let perLayerInput = perLayerSignals[idx]

            // Get shared KV from the layer this one maps to
            let sharedKV = intermediates[prevIdx].kvs
            let sharedOffset = intermediates[prevIdx].offset

            let (output, kvs, offset) = layer(
                h, mask: mask, cache: c, perLayerInput: perLayerInput,
                sharedKV: sharedKV, offset: sharedOffset)

            h = output
            intermediates[idx] = (kvs, offset)
        }

        return norm(h)
    }

    /// Creates attention masks per layer, matching Python `_make_masks`.
    private func makeMasks(h: MLXArray, cache: [KVCache?])
        -> [MLXFast.ScaledDotProductAttentionMaskMode]
    {
        var maskByType = [String: MLXFast.ScaledDotProductAttentionMaskMode]()
        var masks = [MLXFast.ScaledDotProductAttentionMaskMode]()

        for (i, layer) in layers.enumerated() {
            let layerType = layer.layerType
            if maskByType[layerType] == nil {
                if layerType == "full_attention" {
                    maskByType[layerType] = createAttentionMask(h: h, cache: cache[i])
                } else {
                    maskByType[layerType] = createAttentionMask(
                        h: h, cache: cache[i], windowSize: config.slidingWindow)
                }
            }
            masks.append(maskByType[layerType]!)
        }
        return masks
    }
}

// MARK: - Top-level Model

public class Gemma4TextModel: Module, LLMModel {

    @ModuleInfo public var model: Gemma4Model
    @ModuleInfo(key: "lm_head") var lmHead: Linear

    public let config: Gemma4TextConfiguration
    public var vocabularySize: Int { config.vocabularySize }

    public init(_ config: Gemma4TextConfiguration) {
        self.config = config
        self.model = Gemma4Model(config)
        self._lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabularySize, bias: false)
        super.init()
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var out = model(inputs, cache: cache)
        if config.tieWordEmbeddings {
            out = model.embedTokens.asLinear(out)
        } else {
            out = lmHead(out)
        }
        if let softcap = config.finalLogitSoftcapping {
            out = tanh(out / softcap) * softcap
        }
        return out
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var processedWeights = weights

        // VLM models converted using mlx_vlm.convert will still have
        // the weights under a language_model key
        let unflattened = ModuleParameters.unflattened(weights)
        if let lm = unflattened["language_model"] {
            processedWeights = Dictionary(uniqueKeysWithValues: lm.flattened())
        }

        // Filter out rotary_emb and quantization stats
        processedWeights = processedWeights.filter { key, _ in
            !key.contains("self_attn.rotary_emb")
                && !key.contains("input_max")
                && !key.contains("input_min")
                && !key.contains("output_max")
                && !key.contains("output_min")
        }

        // Vocab truncation
        let expectedVocab = config.vocabularySize
        let keysToCheck = [
            "model.embed_tokens.weight", "model.embed_tokens.scales",
            "model.embed_tokens.biases",
            "lm_head.weight", "lm_head.scales", "lm_head.biases",
        ]
        for key in keysToCheck {
            if let tensor = processedWeights[key], tensor.dim(0) > expectedVocab {
                processedWeights[key] = tensor[0..<expectedVocab]
            }
        }

        // Per-layer embedding vocab truncation
        if let perLayerVocab = config.vocabSizePerLayerInput {
            for key in [
                "model.embed_tokens_per_layer.weight",
                "model.embed_tokens_per_layer.scales",
                "model.embed_tokens_per_layer.biases",
            ] {
                if let tensor = processedWeights[key], tensor.dim(0) > perLayerVocab {
                    processedWeights[key] = tensor[0..<perLayerVocab]
                }
            }
        }

        // Tie word embeddings
        if config.tieWordEmbeddings && processedWeights["lm_head.weight"] == nil {
            ["weight", "scales", "biases"].forEach { key in
                if let embedWeight = processedWeights["model.embed_tokens.\(key)"] {
                    processedWeights["lm_head.\(key)"] = embedWeight
                }
            }
        }

        return processedWeights
    }

    /// Creates caches for all layers. KV sharing is an optimization in the forward pass:
    /// when caches are regular (non-quantized), shared layers reuse KV from their
    /// reference layer instead of computing their own. When caches become quantized
    /// (via maybeQuantizeKVCache), each layer uses its own cache independently.
    public func newCache(parameters: GenerateParameters? = nil) -> [KVCache] {
        var caches = [KVCache]()
        for i in 0..<config.hiddenLayers {
            if config.isGlobalLayer(i) {
                let cache = StandardKVCache()
                cache.step = 1024
                caches.append(cache)
            } else {
                caches.append(
                    RotatingKVCache(maxSize: config.slidingWindow, keep: 0)
                )
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

extension Gemma4TextModel: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}
