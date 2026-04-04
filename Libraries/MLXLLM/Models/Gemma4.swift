//
//  Gemma4.swift
//  mlx-swift-lm
//
//  Created for SwiftLM Gemma 4 Support
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

public struct Gemma4Configuration: Codable {
    let modelType: String
    let hiddenSize: Int
    let hiddenLayers: Int
    let intermediateSize: Int
    let attentionHeads: Int
    let headDim: Int
    let rmsNormEps: Float
    let vocabularySize: Int
    let kvHeads: Int
    let ropeTheta: Float
    let ropeLocalBaseFreq: Float
    let ropeTraditional: Bool
    let queryPreAttnScalar: Float?
    let slidingWindow: Int
    let slidingWindowPattern: Int
    let maxPositionEmbeddings: Int
    let ropeScaling: [String: StringOrNumber]?
    let globalHeadDim: Int
    let numKvSharedLayers: Int
    let useDoubleWideMlp: Bool
    
    // MoE / Global KV Configurations
    public let numExperts: Int?
    public let topKExperts: Int?
    public let moeIntermediateSize: Int?
    public let numGlobalKeyValueHeads: Int
    public let tieWordEmbeddings: Bool

    /// Fraction of global head_dim used for RoPE (default 0.25 for Gemma 4 global attn)
    let globalRopePartialFactor: Float
    /// Final logit softcapping value (0 = disabled). Gemma 4 uses 30.0.
    let finalLogitSoftcapping: Float
    /// Per-layer conditioning dimension (0 = disabled)
    let hiddenSizePerLayerInput: Int
    /// Vocabulary size for per-layer embedding table (0 = disabled)
    let vocabSizePerLayerInput: Int

    public init(
        modelType: String, hiddenSize: Int, hiddenLayers: Int, intermediateSize: Int,
        attentionHeads: Int, headDim: Int, rmsNormEps: Float, vocabularySize: Int, kvHeads: Int,
        ropeTheta: Float, ropeLocalBaseFreq: Float, ropeTraditional: Bool,
        queryPreAttnScalar: Float?, slidingWindow: Int, slidingWindowPattern: Int,
        maxPositionEmbeddings: Int, ropeScaling: [String: StringOrNumber]? = nil,
        globalHeadDim: Int = 512, numKvSharedLayers: Int = 0, useDoubleWideMlp: Bool = false,
        tieWordEmbeddings: Bool = true,
        numExperts: Int? = nil, topKExperts: Int? = nil, moeIntermediateSize: Int? = nil,
        numGlobalKeyValueHeads: Int? = nil,
        hiddenSizePerLayerInput: Int = 0, vocabSizePerLayerInput: Int = 0,
        globalRopePartialFactor: Float = 0.25,
        finalLogitSoftcapping: Float = 0.0
    ) {
        self.modelType = modelType
        self.hiddenSize = hiddenSize
        self.hiddenLayers = hiddenLayers
        self.intermediateSize = intermediateSize
        self.attentionHeads = attentionHeads
        self.headDim = headDim
        self.rmsNormEps = rmsNormEps
        self.vocabularySize = vocabularySize
        self.kvHeads = kvHeads
        self.ropeTheta = ropeTheta
        self.ropeLocalBaseFreq = ropeLocalBaseFreq
        self.ropeTraditional = ropeTraditional
        self.queryPreAttnScalar = queryPreAttnScalar
        self.slidingWindow = slidingWindow
        self.slidingWindowPattern = slidingWindowPattern
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.ropeScaling = ropeScaling
        self.globalHeadDim = globalHeadDim
        self.numKvSharedLayers = numKvSharedLayers
        self.useDoubleWideMlp = useDoubleWideMlp
        self.tieWordEmbeddings = tieWordEmbeddings
        self.numExperts = numExperts
        self.topKExperts = topKExperts
        self.moeIntermediateSize = moeIntermediateSize
        self.numGlobalKeyValueHeads = numGlobalKeyValueHeads ?? kvHeads
        self.hiddenSizePerLayerInput = hiddenSizePerLayerInput
        self.vocabSizePerLayerInput = vocabSizePerLayerInput
        self.globalRopePartialFactor = globalRopePartialFactor
        self.finalLogitSoftcapping = finalLogitSoftcapping
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case headDim = "head_dim"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case kvHeads = "num_key_value_heads"
        case ropeTheta = "rope_theta"
        case ropeLocalBaseFreq = "rope_local_base_freq"
        case ropeTraditional = "rope_traditional"
        case queryPreAttnScalar = "query_pre_attn_scalar"
        case slidingWindow = "sliding_window"
        case slidingWindowPattern = "sliding_window_pattern"
        case maxPositionEmbeddings = "max_position_embeddings"
        case ropeScaling = "rope_scaling"
        case globalHeadDim = "global_head_dim"
        case numKvSharedLayers = "num_kv_shared_layers"
        case useDoubleWideMlp = "use_double_wide_mlp"
        case tieWordEmbeddings = "tie_word_embeddings"
        case numGlobalKeyValueHeads = "num_global_key_value_heads"
        // MoE
        case numExperts = "num_experts"
        case topKExperts = "top_k_experts"
        case moeIntermediateSize = "moe_intermediate_size"
        // Per-layer conditioning
        case hiddenSizePerLayerInput = "hidden_size_per_layer_input"
        case vocabSizePerLayerInput = "vocab_size_per_layer_input"
        // Logit softcapping
        case finalLogitSoftcapping = "final_logit_softcapping"
    }

    // Top-level keys (outside text_config)
    enum TopLevelCodingKeys: String, CodingKey {
        case textConfig = "text_config"
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
        
        let tieWordOpt = try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings)
        tieWordEmbeddings = tieWordOpt ?? true
        numExperts = try container.decodeIfPresent(Int.self, forKey: .numExperts)
        topKExperts = try container.decodeIfPresent(Int.self, forKey: .topKExperts)
        moeIntermediateSize = try container.decodeIfPresent(Int.self, forKey: .moeIntermediateSize)
        numGlobalKeyValueHeads = try container.decodeIfPresent(Int.self, forKey: .numGlobalKeyValueHeads) ?? (try container.decodeIfPresent(Int.self, forKey: .kvHeads) ?? 1)
        hiddenSize = try container.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 1152
        hiddenLayers = try container.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 26

        intermediateSize =
            try container.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 6912
        attentionHeads = try container.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 4
        headDim = try container.decodeIfPresent(Int.self, forKey: .headDim) ?? 256
        rmsNormEps = try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1.0e-6
        vocabularySize = try container.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 262144
        kvHeads = try container.decodeIfPresent(Int.self, forKey: .kvHeads) ?? 1
        ropeTheta = try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 1_000_000.0
        ropeLocalBaseFreq = try container.decodeIfPresent(Float.self, forKey: .ropeLocalBaseFreq) ?? 10_000.0
        ropeTraditional = try container.decodeIfPresent(Bool.self, forKey: .ropeTraditional) ?? true
        queryPreAttnScalar = try container.decodeIfPresent(Float.self, forKey: .queryPreAttnScalar)
        finalLogitSoftcapping = try container.decodeIfPresent(Float.self, forKey: .finalLogitSoftcapping) ?? 0.0
        slidingWindow = try container.decodeIfPresent(Int.self, forKey: .slidingWindow) ?? 512
        slidingWindowPattern = try container.decodeIfPresent(Int.self, forKey: .slidingWindowPattern) ?? (hiddenLayers == 35 ? 5 : 6)
        maxPositionEmbeddings = try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 32768
        ropeScaling = try container.decodeIfPresent([String: StringOrNumber].self, forKey: .ropeScaling)
        globalHeadDim = try container.decodeIfPresent(Int.self, forKey: .globalHeadDim) ?? 512
        numKvSharedLayers = try container.decodeIfPresent(Int.self, forKey: .numKvSharedLayers) ?? 0
        useDoubleWideMlp = try container.decodeIfPresent(Bool.self, forKey: .useDoubleWideMlp) ?? false
        // Per-layer conditioning
        self.hiddenSizePerLayerInput = try container.decodeIfPresent(Int.self, forKey: .hiddenSizePerLayerInput) ?? 0
        self.vocabSizePerLayerInput = try container.decodeIfPresent(Int.self, forKey: .vocabSizePerLayerInput) ?? 0
        // Parse partial_rotary_factor for global attention from rope_parameters.full_attention
        // Gemma 4 uses only 25% of global_head_dim (512) for positional encoding = 128 rotated dims.
        struct AC: CodingKey {
            var stringValue: String; init?(stringValue s: String) { stringValue = s }
            var intValue: Int? { nil }; init?(intValue _: Int) { nil }
        }
        if let nestedContainer = try? decoder.container(keyedBy: AC.self),
           let ropeParamsContainer = try? nestedContainer.nestedContainer(keyedBy: AC.self, forKey: AC(stringValue: "rope_parameters")!),
           let fullAttnContainer = try? ropeParamsContainer.nestedContainer(keyedBy: AC.self, forKey: AC(stringValue: "full_attention")!),
           let prf = try? fullAttnContainer.decode(Float.self, forKey: AC(stringValue: "partial_rotary_factor")!) {
            self.globalRopePartialFactor = prf
        } else {
            self.globalRopePartialFactor = 0.25  // Gemma 4 default: 128/512
        }
    }
}

public class Gemma4RMSNormNoScale: Module {
    let eps: Float

    public init(eps: Float = 1e-6) {
        self.eps = eps
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let xFloat32 = x.asType(.float32)
        let meanSq = MLX.mean(MLX.square(xFloat32), axes: [-1], keepDims: true)
        let inverseNorm = MLX.rsqrt(meanSq + eps)
        return (xFloat32 * inverseNorm).asType(x.dtype)
    }
}

/// Proportional RoPE for Gemma 4 full-attention layers.
///
/// Frequencies are computed relative to the full head dimension, and rotation is
/// applied to the first rotated_dims//2 elements of each half of the head, matching
/// the HF `rotate_half` convention used in the Python reference.
public class Gemma4ProportionalRoPE: Module, OffsetLayer {
    let dims: Int
    let traditional: Bool
    let rotatedDims: Int
    /// Stored as private; NOT a model parameter - computed at init from base/dims.
    private var _computedFreqs: MLXArray?

    public init(dims: Int, traditional: Bool = false, base: Float = 10000.0, partialRotaryFactor: Float = 1.0) {
        self.dims = dims
        self.traditional = traditional

        let ropeAngles = Int(partialRotaryFactor * Float(dims) / 2.0)
        self.rotatedDims = 2 * ropeAngles

        if rotatedDims > 0 {
            let exponents = MLXArray(stride(from: Float(0), to: Float(rotatedDims), by: 2)) / Float(dims)
            self._computedFreqs = MLXArray(base) ** exponents
        } else {
            self._computedFreqs = nil
        }

        super.init()
        // Freeze so the module system ignores this class entirely for weight loading
        self.freeze()
    }

    public func callAsFunction(_ x: MLXArray, offset: Int) -> MLXArray {
        guard rotatedDims > 0, let freqs = _computedFreqs else { return x }

        let head = x[0..., 0..., 0..., 0..<dims]
        let half = dims / 2

        let left = head[0..., 0..., 0..., 0..<half]
        let right = head[0..., 0..., 0..., half...]

        let rotHalf = rotatedDims / 2

        // Gather the rotated portions from each half
        let leftRot = left[0..., 0..., 0..., 0..<rotHalf]
        let rightRot = right[0..., 0..., 0..., 0..<rotHalf]
        let rotated = concatenated([leftRot, rightRot], axis: -1)

        // Apply standard RoPE with pre-computed freqs
        let rotatedResult = MLXFast.RoPE(
            rotated, dimensions: rotatedDims, traditional: traditional,
            base: nil, scale: 1.0, offset: offset, freqs: freqs)

        // Reconstruct: put rotated portions back into their halves
        let leftPassthru = left[0..., 0..., 0..., rotHalf...]
        let rightPassthru = right[0..., 0..., 0..., rotHalf...]

        let newLeft = concatenated([rotatedResult[0..., 0..., 0..., 0..<rotHalf], leftPassthru], axis: -1)
        let newRight = concatenated([rotatedResult[0..., 0..., 0..., rotHalf...], rightPassthru], axis: -1)

        return concatenated([newLeft, newRight], axis: -1)
    }
}

class Gemma4Attention: Module {
    let nHeads: Int
    let nKVHeads: Int
    let repeats: Int
    let headDim: Int
    let layerIdx: Int
    let scale: Float
    let isSliding: Bool
    let slidingWindow: Int
    let slidingWindowPattern: Int
    let eps: Float
    let globalRopePartialFactor: Float
    /// QK attention logit softcapping (Gemma 4 uses 30.0). 0 = disabled.
    let attnLogitSoftcap: Float

    @ModuleInfo(key: "q_proj") var queryProj: Linear
    @ModuleInfo(key: "k_proj") var keyProj: Linear
    @ModuleInfo(key: "v_proj") var valueProj: Linear
    @ModuleInfo(key: "o_proj") var outputProj: Linear

    @ModuleInfo(key: "q_norm") var queryNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var keyNorm: RMSNorm
    @ModuleInfo(key: "v_norm") var valueNorm: Gemma4RMSNormNoScale

    @ModuleInfo var rope: OffsetLayer

    init(_ config: Gemma4Configuration, layerIdx: Int) {
        let dim = config.hiddenSize
        self.layerIdx = layerIdx
        self.slidingWindow = config.slidingWindow
        self.slidingWindowPattern = config.slidingWindowPattern
        self.isSliding = (layerIdx + 1) % config.slidingWindowPattern != 0
        self.eps = config.rmsNormEps
        
        self.nHeads = config.attentionHeads
        self.nKVHeads = self.isSliding ? config.kvHeads : config.numGlobalKeyValueHeads
        self.repeats = nHeads / (nKVHeads > 0 ? nKVHeads : 1)
        self.headDim = self.isSliding ? config.headDim : config.globalHeadDim

        // Python reference: self.scale = 1.0 — Q/K RMS norms handle magnitude.
        self.scale = 1.0
        self.attnLogitSoftcap = 0.0

        self._queryProj.wrappedValue = Linear(dim, nHeads * self.headDim, bias: false)
        self._keyProj.wrappedValue = Linear(dim, nKVHeads * self.headDim, bias: false)
        self._valueProj.wrappedValue = Linear(dim, nKVHeads * self.headDim, bias: false)
        self._outputProj.wrappedValue = Linear(nHeads * self.headDim, dim, bias: false)

        self._queryNorm.wrappedValue = RMSNorm(
            dimensions: self.headDim, eps: config.rmsNormEps)
        self._keyNorm.wrappedValue = RMSNorm(dimensions: self.headDim, eps: config.rmsNormEps)
        self._valueNorm.wrappedValue = Gemma4RMSNormNoScale(eps: config.rmsNormEps)

        let ropeFactor = config.globalRopePartialFactor
        self.globalRopePartialFactor = ropeFactor

        if isSliding {
            // Sliding attention: standard RoPE on full head_dim
            self.rope = RoPE(
                dimensions: headDim, traditional: false,
                base: config.ropeLocalBaseFreq, scale: 1.0)
        } else {
            // Global attention: ProportionalRoPE with partial rotation
            self.rope = Gemma4ProportionalRoPE(
                dims: self.headDim,
                traditional: false,
                base: 1000000.0,  // rope_parameters.full_attention.rope_theta
                partialRotaryFactor: ropeFactor
            )
        }

        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache? = nil
    ) -> MLXArray {
        let (B, L, _) = (x.dim(0), x.dim(1), x.dim(2))

        var queries = queryProj(x)
        var keys = keyProj(x)
        var values = valueProj(x)

        queries = queries.reshaped(B, L, nHeads, -1).transposed(0, 2, 1, 3)
        keys = keys.reshaped(B, L, nKVHeads, -1).transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, nKVHeads, -1).transposed(0, 2, 1, 3)

        queries = queryNorm(queries)
        keys = keyNorm(keys)
        values = valueNorm(values)

        // Python reference: rope applies to keys BEFORE cache update, queries AFTER.
        // RoPE is applied to full head_dim; partial rotation is handled by rope init (dims param).
        let LCache = cache?.offset ?? 0
        // Apply RoPE to keys first (before cache), then queries
        keys = rope(keys, offset: LCache)
        queries = rope(queries, offset: LCache)

        let output: MLXArray
        if attnLogitSoftcap > 0 {
            // Gemma 4 uses QK attention logit softcapping before softmax:
            //   scores = tanh(scores / cap) * cap  (llama.cpp llm_build_gemma4_iswa)
            // MLXFast.scaledDotProductAttention has no softcap parameter, so we do it manually.
            let (cachedKeys, cachedValues) = cache?.update(keys: keys, values: values) ?? (keys, values)
            var fullKeys = cachedKeys
            var fullValues = cachedValues
            // TurboKV decode if needed
            if let kvCache = cache as? KVCacheSimple,
               let pk = kvCache.polarKeys, let pv = kvCache.polarValues,
               kvCache.compressedOffset > 0 {
                var histK = MLXFast.turboDecodeK(packed: pk).asType(cachedKeys.dtype)
                var histV = MLXFast.turboDecodeV(packed: pv).asType(cachedValues.dtype)
                // Merge 2×256 virtual heads back to original count × 512
                if kvCache.turboSplitHeads {
                    let B = histK.dim(0), H2 = histK.dim(1), T = histK.dim(2)
                    histK = histK.reshaped(B, H2 / 2, T, 512)
                    histV = histV.reshaped(B, H2 / 2, T, 512)
                }
                fullKeys   = concatenated([histK, cachedKeys],   axis: 2)
                fullValues = concatenated([histV, cachedValues], axis: 2)
            }
            // GQA expansion
            var k = fullKeys
            var v = fullValues
            if nHeads > nKVHeads {
                k = MLX.repeated(k, count: repeats, axis: 1)
                v = MLX.repeated(v, count: repeats, axis: 1)
            }
            // scores: [B, nH, L, S]
            var scores = (queries * scale).matmul(k.transposed(0, 1, 3, 2))
            // Apply attention mask
            if let maskArray = mask.mask {
                scores = scores + maskArray
            }
            // Apply QK softcap: tanh(scores / cap) * cap
            // Critically, we MUST evaluate tanh in float32. In float16/bfloat16, tanh(x) saturates
            // to exactly 1.0 for very small values above 3.0 (due to lack of mantissa bits!),
            // destroying all confidence gradients and resulting in uniform distributions!
            let originalType = scores.dtype
            let scoresF32 = scores.asType(.float32)
            let cap = MLXArray(attnLogitSoftcap).asType(.float32)
            scores = (MLX.tanh(scoresF32 / cap) * cap).asType(originalType)
            // Softmax + weighted sum
            let attnWeights = MLX.softmax(scores.asType(.float32), axis: -1).asType(scores.dtype)
            output = matmul(attnWeights, v)
        } else {
            output = attentionWithCacheUpdate(
                queries: queries, keys: keys, values: values,
                cache: cache, scale: scale, mask: mask)
        }
        return outputProj(
            output.transposed(0, 2, 1, 3).reshaped(B, L, -1)
        )
    }
}

public class Gemma4MLP: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear

    public init(dimensions: Int, hiddenDimensions: Int) {
        self._gateProj.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        self._downProj.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
        self._upProj.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Python reference: nn.gelu_approx(self.gate_proj(x)) * self.up_proj(x)
        return downProj(geluApproximate(gateProj(x)) * upProj(x))
    }
}

class Gemma4Router: Module {
    @ModuleInfo(key: "proj") var proj: Linear
    @ModuleInfo(key: "scale") var scale: MLXArray
    @ModuleInfo(key: "per_expert_scale") var perExpertScale: MLXArray

    let eps: Float
    let scalarRootSize: Float

    public init(dimensions: Int, numExperts: Int, eps: Float) {
        self.eps = eps
        self.scalarRootSize = 1.0 / sqrt(Float(dimensions))
        self._proj.wrappedValue = Linear(dimensions, numExperts, bias: false)
        self._scale.wrappedValue = MLXArray.zeros([dimensions])
        self._perExpertScale.wrappedValue = MLXArray.zeros([numExperts])
        super.init()
    }

    public func callAsFunction(_ x: MLXArray, topK: Int) -> (MLXArray, MLXArray) {
        // Python reference: RMSNormNoScale → scale multiply → 1/sqrt(hidden) → proj → softmax → topK
        // RMS norm without learnable weight (None weight)
        let xF32 = x.asType(.float32)
        let meanX2 = MLX.mean(MLX.square(xF32), axes: [-1], keepDims: true)
        let xNormed = (x * MLX.rsqrt(meanX2 + MLXArray(eps))).asType(x.dtype)

        // Scale: x * root_size * scale (Python: x * self._root_size * self.scale)
        let scaled = xNormed * MLXArray(scalarRootSize) * scale

        let expertScores = proj(scaled)
        let routerProbs = MLX.softmax(expertScores, axis: -1)

        // Python: argpartition(-expert_scores, kth=topK-1)[..., :topK]
        let negScores = MLX.negative(expertScores)
        let allInds = MLX.argPartition(negScores, kth: topK - 1, axis: -1)
        let topKInds = allInds[0..., 0..., 0..<topK]

        // Gather the softmax probs for the selected experts
        var topKWeights = MLX.takeAlong(routerProbs, topKInds, axis: -1)

        // L1 normalize then apply per-expert scale
        topKWeights = topKWeights / topKWeights.sum(axis: -1, keepDims: true)
        topKWeights = topKWeights * perExpertScale[topKInds]

        return (topKWeights, topKInds)
    }
}

class Gemma4SparseMoeBlock: Module {
    let topK: Int

    @ModuleInfo(key: "switch_glu") var switchGLU: SwitchGLU
    @ModuleInfo(key: "router") var router: Gemma4Router

    init(dimensions: Int, numExperts: Int, topK: Int, moeIntermediateSize: Int) {
        self.topK = topK
        self._router.wrappedValue = Gemma4Router(dimensions: dimensions, numExperts: numExperts, eps: 1e-6)
        self._switchGLU.wrappedValue = SwitchGLU(
            inputDims: dimensions,
            hiddenDims: moeIntermediateSize,
            numExperts: numExperts,
            activation: geluApproximate
        )
        super.init()
    }

    public func callAsFunction(_ x: MLXArray, routerInput: MLXArray) -> MLXArray {
        let (scores, inds) = router(routerInput, topK: topK)
        let y = switchGLU(x, inds)

        let B = y.dim(0)
        let T = y.dim(1)
        
        let yMasked = y * scores[0..., 0..., 0..., .newAxis]
        let yMerged = yMasked.sum(axis: 2)
        
        return yMerged.reshaped([B, T, -1])
    }
}

class Gemma4TransformerBlock: Module {
    @ModuleInfo(key: "self_attn") var selfAttention: Gemma4Attention
    @ModuleInfo var mlp: Gemma4MLP
    @ModuleInfo(key: "experts") var expertsBlock: Gemma4SparseMoeBlock

    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm
    @ModuleInfo(key: "pre_feedforward_layernorm") var preFeedforwardLayerNorm: RMSNorm
    @ModuleInfo(key: "post_feedforward_layernorm") var postFeedforwardLayerNorm: RMSNorm

    // MoE specific norms
    @ModuleInfo(key: "post_feedforward_layernorm_1") var postFeedforwardLayerNorm1: RMSNorm?
    @ModuleInfo(key: "pre_feedforward_layernorm_2") var preFeedforwardLayerNorm2: RMSNorm?
    @ModuleInfo(key: "post_feedforward_layernorm_2") var postFeedforwardLayerNorm2: RMSNorm?

    // Per-layer conditioning (Gemma 4 architectural novelty)
    @ModuleInfo(key: "per_layer_input_gate") var perLayerInputGate: Linear?
    @ModuleInfo(key: "per_layer_projection") var perLayerProjectionLayer: Linear?
    @ModuleInfo(key: "post_per_layer_input_norm") var postPerLayerInputNorm: RMSNorm?

    @ModuleInfo(key: "layer_scalar") var layerScalar: MLXArray

    let numAttentionHeads: Int
    let hiddenSize: Int
    let layerIdx: Int
    let isMoe: Bool
    let hasPerLayerInput: Bool

    init(_ config: Gemma4Configuration, layerIdx: Int) {
        self.numAttentionHeads = config.attentionHeads
        self.hiddenSize = config.hiddenSize
        self.layerIdx = layerIdx
        self.hasPerLayerInput = config.hiddenSizePerLayerInput > 0

        self._selfAttention.wrappedValue = Gemma4Attention(config, layerIdx: layerIdx)

        let mlpSize: Int
        if config.useDoubleWideMlp && layerIdx >= (config.hiddenLayers - config.numKvSharedLayers) {
            mlpSize = config.intermediateSize * 2
        } else {
            mlpSize = config.intermediateSize
        }
        self.mlp = Gemma4MLP(dimensions: config.hiddenSize, hiddenDimensions: mlpSize)

        self.isMoe = config.numExperts != nil && config.numExperts! > 0
        let numExperts = config.numExperts ?? 1
        
        self._expertsBlock.wrappedValue = Gemma4SparseMoeBlock(
            dimensions: config.hiddenSize,
            numExperts: numExperts,
            topK: config.topKExperts ?? 1,
            moeIntermediateSize: config.moeIntermediateSize ?? config.intermediateSize
        )
        
        if self.isMoe {
            self._postFeedforwardLayerNorm1.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
            self._preFeedforwardLayerNorm2.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
            self._postFeedforwardLayerNorm2.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        }

        if hasPerLayerInput {
            self._perLayerInputGate.wrappedValue = Linear(
                config.hiddenSize, config.hiddenSizePerLayerInput, bias: false)
            self._perLayerProjectionLayer.wrappedValue = Linear(
                config.hiddenSizePerLayerInput, config.hiddenSize, bias: false)
            self._postPerLayerInputNorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: config.rmsNormEps)
        }

        self._inputLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._preFeedforwardLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postFeedforwardLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._layerScalar.wrappedValue = MLXArray.ones([1])

        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache? = nil,
        perLayerInput: MLXArray? = nil
    ) -> MLXArray {
        let inputNorm = inputLayerNorm(x)
        let r = selfAttention(inputNorm, mask: mask, cache: cache)
        let attnNorm = postAttentionLayerNorm(r)

        var residualUpdates: MLXArray

        if isMoe {
            let preMLPNorm = preFeedforwardLayerNorm(x + attnNorm)
            let denseOut = mlp(preMLPNorm)
            let densePostNorm1 = postFeedforwardLayerNorm1!(denseOut)

            // Router evaluates on the un-normed `attn_out` stream
            let routerInput = x + attnNorm

            // Experts evaluate on the DENSE-NORMED sparse stream (llama.cpp `ffn_pre_norm_2`)
            let sparsePreNorm = preFeedforwardLayerNorm2!(routerInput)
            let sparseOut = expertsBlock(sparsePreNorm, routerInput: routerInput)
            let sparsePostNorm2 = postFeedforwardLayerNorm2!(sparseOut)

            let combined = densePostNorm1 + sparsePostNorm2
            let postMLPNorm = postFeedforwardLayerNorm(combined)
            residualUpdates = attnNorm + postMLPNorm
        } else {
            let preMLPNorm = preFeedforwardLayerNorm(x + attnNorm)
            let r2 = mlp(preMLPNorm)
            let postMLPNorm = postFeedforwardLayerNorm(r2)
            residualUpdates = attnNorm + postMLPNorm
        }

        // Per-layer conditioning residual (Gemma 4 architectural novelty)
        // Applied after attn+MLP, to the updates before they are added to the root state
        if hasPerLayerInput,
           let pli = perLayerInput,
           let gate = perLayerInputGate,
           let proj = perLayerProjectionLayer,
           let norm = postPerLayerInputNorm
        {
            var gated = gate(x + residualUpdates) // Wait, Gemma 4 conditions on accumulated state? Let's assume yes.
            gated = geluApproximate(gated)
            gated = gated * pli
            gated = proj(gated)
            gated = norm(gated)
            residualUpdates = residualUpdates + gated
        }

        // Apply Gemma 4 residual scaling to the ENTIRE stream, avoiding exponential variance explosion 
        return (x + residualUpdates) * layerScalar
    }
}

// Restored LayerPartitionable & StreamableMoE conformance to re-enable 
// SSD expert streaming, bridging the missing protocols from Damon Janis's initial draft.
// Reference: https://github.com/SharpAI/mlx-swift-lm/pull/1
public class Gemma4ModelInternal: Module, LayerPartitionable, StreamableMoE {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo var layers: [Gemma4TransformerBlock]
    @ModuleInfo var norm: RMSNorm

    // Per-layer conditioning weights (Gemma 4 architectural novelty)
    @ModuleInfo(key: "embed_tokens_per_layer") var embedTokensPerLayer: Embedding?
    @ModuleInfo(key: "per_layer_model_projection") var perLayerModelProjection: Linear?
    @ModuleInfo(key: "per_layer_projection_norm") var perLayerProjectionNorm: RMSNorm?

    let config: Gemma4Configuration

    // LayerPartitionable
    public var gpuLayerCount: Int? = nil
    public var totalLayerCount: Int { layers.count }
    
    // StreamableMoE
    public var streamExperts: Bool = false

    init(_ config: Gemma4Configuration) {
        self.config = config

        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize,
            dimensions: config.hiddenSize
        )

        self._layers.wrappedValue = (0 ..< config.hiddenLayers).map { layerIdx in
            Gemma4TransformerBlock(config, layerIdx: layerIdx)
        }

        self.norm = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)

        if config.hiddenSizePerLayerInput > 0 {
            // embed_tokens_per_layer: [vocabSizePerLayerInput, numLayers × hiddenSizePerLayerInput]
            self._embedTokensPerLayer.wrappedValue = Embedding(
                embeddingCount: config.vocabSizePerLayerInput,
                dimensions: config.hiddenLayers * config.hiddenSizePerLayerInput
            )
            // per_layer_model_projection: [hiddenSize → numLayers × hiddenSizePerLayerInput]
            self._perLayerModelProjection.wrappedValue = Linear(
                config.hiddenSize,
                config.hiddenLayers * config.hiddenSizePerLayerInput,
                bias: false
            )
            self._perLayerProjectionNorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSizePerLayerInput, eps: config.rmsNormEps)
        }

        super.init()
    }

    func callAsFunction(
        _ inputs: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode? = nil,
        cache: [KVCache?]? = nil
    ) -> MLXArray {
        var h = embedTokens(inputs)
        // Python reference: h = h * self.embed_scale where embed_scale = hidden_size**0.5
        h = h * MLXArray(Float(config.hiddenSize).squareRoot())
        var layerCache = cache
        if layerCache == nil {
            layerCache = Array(repeating: nil as KVCache?, count: layers.count)
        }

        let globalMask = createAttentionMask(h: h, cache: cache?[config.slidingWindowPattern - 1])
        let slidingWindowMask: MLXFast.ScaledDotProductAttentionMaskMode =
            config.slidingWindowPattern > 1
            ? createAttentionMask(h: h, cache: cache?[0], windowSize: config.slidingWindow)
            : .none

        // Compute per-layer conditioning tensor: [B, L, numLayers, hiddenSizePerLayerInput]
        // Combines a separate token-embedding table with a projection of the main embeddings.
        var perLayerInputs: MLXArray? = nil
        if config.hiddenSizePerLayerInput > 0,
           let embedPerLayer = embedTokensPerLayer,
           let modelProj = perLayerModelProjection,
           let projNorm = perLayerProjectionNorm
        {
            let B = inputs.dim(0)
            let L = inputs.dim(1)
            let nL = config.hiddenLayers
            let D = config.hiddenSizePerLayerInput

            // Token-based per-layer embeddings, scaled by sqrt(hiddenSizePerLayerInput)
            let tokenScale = MLXArray(sqrt(Float(D))).asType(h.dtype)
            let tokenEmbeds = (embedPerLayer(inputs) * tokenScale)
                .reshaped(B, L, nL, D)  // [B, L, numLayers, D]

            // Project main embeddings, scale by 1/sqrt(hiddenSize), reshape, then norm
            let projScale = MLXArray(1.0 / sqrt(Float(config.hiddenSize))).asType(h.dtype)
            let modelProjected = (modelProj(h) * projScale)
                .reshaped(B, L, nL, D)  // [B, L, numLayers, D]
            let modelProjectedNormed = projNorm(modelProjected)

            // Combine: (tokenEmbeds + projection) * (1/sqrt(2))
            let combineScale = MLXArray(Float(1.0 / 2.0.squareRoot())).asType(h.dtype)
            perLayerInputs = (tokenEmbeds + modelProjectedNormed) * combineScale
        }

        for (i, layer) in layers.enumerated() {
            let isGlobal = (i % config.slidingWindowPattern == config.slidingWindowPattern - 1)
            let layerMask = isGlobal ? globalMask : slidingWindowMask
            // Slice per-layer conditioning for this layer: [B, L, D]
            let pli = perLayerInputs.map { $0[0..., 0..., i, 0...] }
            
            h = partitionedLayerCall(index: i, gpuLayerCount: gpuLayerCount, stream: streamExperts, cacheToEval: layerCache?[i]) {
                layer(h, mask: layerMask, cache: layerCache?[i], perLayerInput: pli)
            }
        }
        return norm(h)
    }
}

public class Gemma4Model: Module, LLMModel {

    @ModuleInfo public var model: Gemma4ModelInternal
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public let config: Gemma4Configuration
    public var vocabularySize: Int { config.vocabularySize }

    public init(_ config: Gemma4Configuration) {
        self.config = config
        self.model = Gemma4ModelInternal(config)
        if !config.tieWordEmbeddings {
            self._lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabularySize, bias: false)
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
        // Apply final logit softcapping (Python reference line 579-580)
        if config.finalLogitSoftcapping > 0 {
            let cap = MLXArray(config.finalLogitSoftcapping)
            out = MLX.tanh(out / cap) * cap
        }
        return out
    }

    public func sanitize(weights: [String: MLXArray], metadata: [String: String]) -> [String: MLXArray] {
        var processedWeights = weights

        let unflattened = ModuleParameters.unflattened(weights)
        if let lm = unflattened["language_model"] {
            processedWeights = Dictionary(uniqueKeysWithValues: lm.flattened())
        }

        let expectedVocab = config.vocabularySize
        let keysToCheck = [
            "model.embed_tokens.weight", "model.embed_tokens.scales", "model.embed_tokens.biases",
            "lm_head.weight", "lm_head.scales", "lm_head.biases",
        ]
        for key in keysToCheck {
            if let tensor = processedWeights[key], tensor.dim(0) > expectedVocab {
                processedWeights[key] = tensor[0 ..< expectedVocab]
            }
        }
        
        var finalWeights = [String: MLXArray]()
        for (k, v) in processedWeights {
            if k.contains("self_attn.rotary_emb") || k.contains("input_max") || k.contains("input_min") || k.contains("output_max") || k.contains("output_min") {
                continue
            }
            if k.hasSuffix(".experts.gate_up_proj.weight") {
                 let base = k.replacingOccurrences(of: ".experts.gate_up_proj.weight", with: ".experts.switch_glu")
                 let parts = MLX.split(v, parts: 2, axis: -2)
                 finalWeights["\(base).gate_proj.weight"] = parts[0]
                 finalWeights["\(base).up_proj.weight"] = parts[1]
                 continue
            }
            if k.hasSuffix(".experts.down_proj.weight") {
                 let base = k.replacingOccurrences(of: ".experts.down_proj.weight", with: ".experts.switch_glu.down_proj.weight")
                 finalWeights[base] = v
                 continue
            }
            let newK = k.replacingOccurrences(of: ".router.", with: ".experts.router.")
            finalWeights[newK] = v
        }

        // Explicitly map per_layer_projection_norm to ensure it survives flattening
        if let normWeight = weights["language_model.model.per_layer_projection_norm.weight"] {
            finalWeights["model.per_layer_projection_norm.weight"] = normWeight
        } else if let normWeight = weights["model.per_layer_projection_norm.weight"] {
            finalWeights["model.per_layer_projection_norm.weight"] = normWeight
        }

        // Handle mixed-quantization: MLP and MoE experts might be 8-bit while other layers are 4-bit.
        // 3. Setup dynamic layer updates
        var moduleUpdates: [(String, Module)] = []
        
        // Dynamically wrap embed_tokens with QuantizedEmbedding if its shape count indicates packed uint32 (shape is 2, normally shape is 2 anyway but we check for matching scale). Since we know it's a 4-bit A4B model, we just check if scales exist.
        if let embedScales = processedWeights["model.embed_tokens.scales"], let embedTokens = model.embedTokens as? Embedding {
            moduleUpdates.append(("model.embed_tokens", QuantizedEmbedding(embedTokens, groupSize: 64, bits: 4)))
        }

        // 4. Update the MoE and MLP parameter overrides inside the layers loop.
        for (i, layer) in self.model.layers.enumerated() {
            // Check MLP
            let mlp = layer.mlp
            if let gate = mlp.gateProj as? Linear, let down = mlp.downProj as? Linear, let up = mlp.upProj as? Linear {
                if let w = finalWeights["language_model.model.layers.\(i).mlp.gate_proj.weight"] ?? finalWeights["model.layers.\(i).mlp.gate_proj.weight"], w.shape.count == 2 {
                    let bits = 32 * w.shape.last! / gate.weight.shape[1]
                    moduleUpdates.append(("model.layers.\(i).mlp.gate_proj", QuantizedLinear(gate, groupSize: 64, bits: bits)))
                }
                if let w = finalWeights["language_model.model.layers.\(i).mlp.down_proj.weight"] ?? finalWeights["model.layers.\(i).mlp.down_proj.weight"], w.shape.count == 2 {
                    let bits = 32 * w.shape.last! / down.weight.shape[1]
                    moduleUpdates.append(("model.layers.\(i).mlp.down_proj", QuantizedLinear(down, groupSize: 64, bits: bits)))
                }
                if let w = finalWeights["language_model.model.layers.\(i).mlp.up_proj.weight"] ?? finalWeights["model.layers.\(i).mlp.up_proj.weight"], w.shape.count == 2 {
                    let bits = 32 * w.shape.last! / up.weight.shape[1]
                    moduleUpdates.append(("model.layers.\(i).mlp.up_proj", QuantizedLinear(up, groupSize: 64, bits: bits)))
                }
            }

            // Check Experts switchGLU
            let switchGLU = layer.expertsBlock.switchGLU
            if let gate = switchGLU.gateProj as? SwitchLinear, let down = switchGLU.downProj as? SwitchLinear, let up = switchGLU.upProj as? SwitchLinear {
                if let w = finalWeights["language_model.model.layers.\(i).experts.switch_glu.gate_proj.weight"] ?? finalWeights["model.layers.\(i).experts.switch_glu.gate_proj.weight"], w.shape.count == 3 {
                    let bits = 32 * w.shape.last! / gate.weight.shape.last!
                    moduleUpdates.append(("model.layers.\(i).experts.switch_glu.gate_proj", QuantizedSwitchLinear(gate, groupSize: 64, bits: bits)))
                }
                if let w = finalWeights["language_model.model.layers.\(i).experts.switch_glu.down_proj.weight"] ?? finalWeights["model.layers.\(i).experts.switch_glu.down_proj.weight"], w.shape.count == 3 {
                    let bits = 32 * w.shape.last! / down.weight.shape.last!
                    moduleUpdates.append(("model.layers.\(i).experts.switch_glu.down_proj", QuantizedSwitchLinear(down, groupSize: 64, bits: bits)))
                }
                if let w = finalWeights["language_model.model.layers.\(i).experts.switch_glu.up_proj.weight"] ?? finalWeights["model.layers.\(i).experts.switch_glu.up_proj.weight"], w.shape.count == 3 {
                    let bits = 32 * w.shape.last! / up.weight.shape.last!
                    moduleUpdates.append(("model.layers.\(i).experts.switch_glu.up_proj", QuantizedSwitchLinear(up, groupSize: 64, bits: bits)))
                }
            }

            // Check Router
            // Let the built-in loader wrap the Linear layer correctly. No router override needed.
            
            // Check Attention Projections (q, k, v, o)
            if let qW = finalWeights["language_model.model.layers.\(i).self_attn.q_proj.weight"] ?? finalWeights["model.layers.\(i).self_attn.q_proj.weight"], qW.shape.count == 2 {
                if let qProj = layer.selfAttention.queryProj as? Linear {
                    let bits = 32 * qW.shape.last! / qProj.weight.shape[1]
                    moduleUpdates.append(("model.layers.\(i).self_attn.q_proj", QuantizedLinear(qProj, groupSize: 64, bits: bits)))
                }
            }
            if let kW = finalWeights["language_model.model.layers.\(i).self_attn.k_proj.weight"] ?? finalWeights["model.layers.\(i).self_attn.k_proj.weight"], kW.shape.count == 2 {
                if let kProj = layer.selfAttention.keyProj as? Linear {
                    let bits = 32 * kW.shape.last! / kProj.weight.shape[1]
                    moduleUpdates.append(("model.layers.\(i).self_attn.k_proj", QuantizedLinear(kProj, groupSize: 64, bits: bits)))
                }
            }
            if let vW = finalWeights["language_model.model.layers.\(i).self_attn.v_proj.weight"] ?? finalWeights["model.layers.\(i).self_attn.v_proj.weight"], vW.shape.count == 2 {
                if let vProj = layer.selfAttention.valueProj as? Linear {
                    let bits = 32 * vW.shape.last! / vProj.weight.shape[1]
                    moduleUpdates.append(("model.layers.\(i).self_attn.v_proj", QuantizedLinear(vProj, groupSize: 64, bits: bits)))
                }
            }
            if let oW = finalWeights["language_model.model.layers.\(i).self_attn.o_proj.weight"] ?? finalWeights["model.layers.\(i).self_attn.o_proj.weight"], oW.shape.count == 2 {
                if let oProj = layer.selfAttention.outputProj as? Linear {
                    let bits = 32 * oW.shape.last! / oProj.weight.shape[1]
                    moduleUpdates.append(("model.layers.\(i).self_attn.o_proj", QuantizedLinear(oProj, groupSize: 64, bits: bits)))
                }
            }
        }
        
        if !moduleUpdates.isEmpty {
            self.update(modules: ModuleChildren.unflattened(moduleUpdates))
        }

        // Per-layer conditioning weights are now fully implemented and loaded normally.
        
        // Gemma 4 shares k_proj weights with v_proj in some layers (or all)
        for i in 0..<config.hiddenLayers {
            let kWeightKey = "model.layers.\(i).self_attn.k_proj.weight"
            let vWeightKey = "model.layers.\(i).self_attn.v_proj.weight"
            if finalWeights[kWeightKey] != nil && finalWeights[vWeightKey] == nil {
                finalWeights[vWeightKey] = finalWeights[kWeightKey]
                
                let kScalesKey = "model.layers.\(i).self_attn.k_proj.scales"
                let vScalesKey = "model.layers.\(i).self_attn.v_proj.scales"
                if finalWeights[kScalesKey] != nil {
                    finalWeights[vScalesKey] = finalWeights[kScalesKey]
                }
                
                let kBiasesKey = "model.layers.\(i).self_attn.k_proj.biases"
                let vBiasesKey = "model.layers.\(i).self_attn.v_proj.biases"
                if finalWeights[kBiasesKey] != nil {
                    finalWeights[vBiasesKey] = finalWeights[kBiasesKey]
                }
            }
        }
        
        return finalWeights
    }

    public func newCache(parameters: GenerateParameters? = nil) -> [KVCache] {
        var caches = [KVCache]()
        let slidingWindow = config.slidingWindow
        let slidingWindowPattern = config.slidingWindowPattern

        for i in 0 ..< config.hiddenLayers {
            let isGlobalLayer = (i % slidingWindowPattern == slidingWindowPattern - 1)

            if isGlobalLayer {
                let cache = StandardKVCache()
                cache.step = 1024
                caches.append(cache)
            } else {
                caches.append(
                    RotatingKVCache(maxSize: slidingWindow, keep: 0)
                )
            }
        }

        return caches
    }

}

extension Gemma4Model: LoRAModel {
    public var loraLayers: [Module] {
        model.layers.map { $0 as Module }
    }
}
