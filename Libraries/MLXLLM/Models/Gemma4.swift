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
    let queryPreAttnScalar: Float
    let slidingWindow: Int
    let slidingWindowPattern: Int
    let maxPositionEmbeddings: Int
    let ropeScaling: [String: StringOrNumber]?
    let globalHeadDim: Int
    let numKvSharedLayers: Int
    let useDoubleWideMlp: Bool
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
        queryPreAttnScalar: Float, slidingWindow: Int, slidingWindowPattern: Int,
        maxPositionEmbeddings: Int, ropeScaling: [String: StringOrNumber]? = nil,
        globalHeadDim: Int = 512, numKvSharedLayers: Int = 0, useDoubleWideMlp: Bool = false,
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
        ropeTraditional = try container.decodeIfPresent(Bool.self, forKey: .ropeTraditional) ?? false
        queryPreAttnScalar = try container.decodeIfPresent(Float.self, forKey: .queryPreAttnScalar) ?? 256
        finalLogitSoftcapping = try container.decodeIfPresent(Float.self, forKey: .finalLogitSoftcapping) ?? 0.0
        slidingWindow = try container.decodeIfPresent(Int.self, forKey: .slidingWindow) ?? 512
        slidingWindowPattern = try container.decodeIfPresent(Int.self, forKey: .slidingWindowPattern) ?? (hiddenLayers == 35 ? 5 : 6)
        maxPositionEmbeddings = try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 32768
        ropeScaling = try container.decodeIfPresent([String: StringOrNumber].self, forKey: .ropeScaling)
        globalHeadDim = try container.decodeIfPresent(Int.self, forKey: .globalHeadDim) ?? 512
        numKvSharedLayers = try container.decodeIfPresent(Int.self, forKey: .numKvSharedLayers) ?? 0
        useDoubleWideMlp = try container.decodeIfPresent(Bool.self, forKey: .useDoubleWideMlp) ?? false
        numGlobalKeyValueHeads = try container.decodeIfPresent(Int.self, forKey: .numGlobalKeyValueHeads) ?? kvHeads
        // MoE Support
        self.numExperts = try container.decodeIfPresent(Int.self, forKey: .numExperts)
        self.topKExperts = try container.decodeIfPresent(Int.self, forKey: .topKExperts)
        self.moeIntermediateSize = try container.decodeIfPresent(Int.self, forKey: .moeIntermediateSize)
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

    // Optional MoE configurations
    public let numExperts: Int?
    public let topKExperts: Int?
    public let moeIntermediateSize: Int?
    
    public let numGlobalKeyValueHeads: Int
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
    /// QK attention logit softcapping (Gemma 4 uses 30.0). 0 = disabled.
    let attnLogitSoftcap: Float

    @ModuleInfo(key: "q_proj") var queryProj: Linear
    @ModuleInfo(key: "k_proj") var keyProj: Linear
    @ModuleInfo(key: "v_proj") var valueProj: Linear
    @ModuleInfo(key: "o_proj") var outputProj: Linear

    @ModuleInfo(key: "q_norm") var queryNorm: Gemma.RMSNorm
    @ModuleInfo(key: "k_norm") var keyNorm: Gemma.RMSNorm

    @ModuleInfo var rope: OffsetLayer

    init(_ config: Gemma4Configuration, layerIdx: Int) {
        let dim = config.hiddenSize
        self.layerIdx = layerIdx
        self.slidingWindow = config.slidingWindow
        self.slidingWindowPattern = config.slidingWindowPattern
        self.isSliding = (layerIdx + 1) % config.slidingWindowPattern != 0
        
        self.nHeads = config.attentionHeads
        self.nKVHeads = self.isSliding ? config.kvHeads : config.numGlobalKeyValueHeads
        self.repeats = nHeads / (nKVHeads > 0 ? nKVHeads : 1)
        self.headDim = self.isSliding ? config.headDim : config.globalHeadDim

        self.scale = pow(config.queryPreAttnScalar, -0.5)
        // Gemma 4 does NOT apply softcapping to attention logits (only to final logits).
        // Set to 0.0 to skip the manual attention path and use MLXFast.scaledDotProductAttention.
        self.attnLogitSoftcap = 0.0

        self._queryProj.wrappedValue = Linear(dim, nHeads * self.headDim, bias: false)
        self._keyProj.wrappedValue = Linear(dim, nKVHeads * self.headDim, bias: false)
        self._valueProj.wrappedValue = Linear(dim, nKVHeads * self.headDim, bias: false)
        self._outputProj.wrappedValue = Linear(nHeads * self.headDim, dim, bias: false)

        self._queryNorm.wrappedValue = Gemma.RMSNorm(
            dimensions: self.headDim, eps: config.rmsNormEps)
        self._keyNorm.wrappedValue = Gemma.RMSNorm(dimensions: self.headDim, eps: config.rmsNormEps)

        if isSliding {
            self.rope = initializeRope(
                dims: headDim, base: config.ropeLocalBaseFreq, traditional: false,
                scalingConfig: nil, maxPositionEmbeddings: nil)
        } else {
            // Global attention: Gemma 4 uses partial_rotary_factor to rotate only a fraction
            // of the head_dim. E.g. globalHeadDim=512, partialFactor=0.25 → 128 rotated dims.
            let globalRopeDims = Int((Float(headDim) * config.globalRopePartialFactor).rounded())
            self.rope = initializeRope(
                dims: globalRopeDims, base: config.ropeTheta, traditional: false,
                scalingConfig: config.ropeScaling,
                maxPositionEmbeddings: config.maxPositionEmbeddings)
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

        if let cache {
            queries = rope(queries, offset: cache.offset)
            keys = rope(keys, offset: cache.offset)
        } else {
            queries = rope(queries, offset: 0)
            keys = rope(keys, offset: 0)
        }

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
                let histK = MLXFast.turboDecodeK(packed: pk).asType(cachedKeys.dtype)
                let histV = MLXFast.turboDecodeV(packed: pv).asType(cachedValues.dtype)
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
            let cap = MLXArray(attnLogitSoftcap).asType(scores.dtype)
            scores = MLX.tanh(scores / cap) * cap
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

class Gemma4Router: Module {
    @ModuleInfo(key: "proj") var proj: Linear
    @ModuleInfo(key: "scale") var scale: MLXArray
    @ModuleInfo(key: "per_expert_scale") var perExpertScale: MLXArray

    public init(dimensions: Int, numExperts: Int) {
        self._proj.wrappedValue = Linear(dimensions, numExperts, bias: false)
        self._scale.wrappedValue = MLXArray.ones([dimensions])
        self._perExpertScale.wrappedValue = MLXArray.ones([numExperts])
        super.init()
    }

    func callAsFunction(_ x: MLXArray, topK: Int) -> (MLXArray, MLXArray) {
        let xNormed = x * scale
        var logits = proj(xNormed)
        logits = logits * perExpertScale
        
        let probs = MLX.softmax(logits, axis: -1, precise: true)
        let kth = probs.dim(-1) - topK
        let inds = MLX.argPartition(probs, kth: kth, axis: -1)[.ellipsis, (kth)...]
        var scores = MLX.takeAlong(probs, inds, axis: -1)
        
        // L1 normalize topK scores
        scores = scores / scores.sum(axis: -1, keepDims: true)
        
        return (scores, inds)
    }
}

class Gemma4SparseMoeBlock: Module, UnaryLayer {
    let topK: Int

    @ModuleInfo(key: "router") var router: Gemma4Router
    @ModuleInfo(key: "switch_glu") var switchGLU: SwitchGLU

    init(dimensions: Int, numExperts: Int, topK: Int, moeIntermediateSize: Int) {
        self.topK = topK
        self._router.wrappedValue = Gemma4Router(dimensions: dimensions, numExperts: numExperts)
        self._switchGLU.wrappedValue = SwitchGLU(
            inputDims: dimensions,
            hiddenDims: moeIntermediateSize,
            numExperts: numExperts,
            activation: geluApproximate
        )
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (scores, inds) = router(x, topK: topK)
        let y = switchGLU(x, inds)
        let combined = (y * scores[.ellipsis, .newAxis]).sum(axis: -2)
        return combined
    }
}

class Gemma4TransformerBlock: Module {
    @ModuleInfo(key: "self_attn") var selfAttention: Gemma4Attention
    @ModuleInfo var mlp: Gemma4MLP
    @ModuleInfo(key: "experts") var expertsBlock: Gemma4SparseMoeBlock?

    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: Gemma.RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: Gemma.RMSNorm
    @ModuleInfo(key: "pre_feedforward_layernorm") var preFeedforwardLayerNorm: Gemma.RMSNorm
    @ModuleInfo(key: "post_feedforward_layernorm") var postFeedforwardLayerNorm: Gemma.RMSNorm

    // MoE specific norms
    @ModuleInfo(key: "post_feedforward_layernorm_1") var postFeedforwardLayerNorm1: Gemma.RMSNorm?
    @ModuleInfo(key: "pre_feedforward_layernorm_2") var preFeedforwardLayerNorm2: Gemma.RMSNorm?
    @ModuleInfo(key: "post_feedforward_layernorm_2") var postFeedforwardLayerNorm2: Gemma.RMSNorm?

    // Per-layer conditioning (Gemma 4 architectural novelty)
    @ModuleInfo(key: "per_layer_input_gate") var perLayerInputGate: Linear?
    @ModuleInfo(key: "per_layer_projection") var perLayerProjectionLayer: Linear?
    @ModuleInfo(key: "post_per_layer_input_norm") var postPerLayerInputNorm: Gemma.RMSNorm?

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
        if isMoe {
            self._expertsBlock.wrappedValue = Gemma4SparseMoeBlock(
                dimensions: config.hiddenSize,
                numExperts: config.numExperts!,
                topK: config.topKExperts ?? 8,
                moeIntermediateSize: config.moeIntermediateSize ?? config.intermediateSize
            )
            self._postFeedforwardLayerNorm1.wrappedValue = Gemma.RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
            self._preFeedforwardLayerNorm2.wrappedValue = Gemma.RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
            self._postFeedforwardLayerNorm2.wrappedValue = Gemma.RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        }

        if hasPerLayerInput {
            self._perLayerInputGate.wrappedValue = Linear(
                config.hiddenSize, config.hiddenSizePerLayerInput, bias: false)
            self._perLayerProjectionLayer.wrappedValue = Linear(
                config.hiddenSizePerLayerInput, config.hiddenSize, bias: false)
            self._postPerLayerInputNorm.wrappedValue = Gemma.RMSNorm(
                dimensions: config.hiddenSize, eps: config.rmsNormEps)
        }

        self._inputLayerNorm.wrappedValue = Gemma.RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = Gemma.RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._preFeedforwardLayerNorm.wrappedValue = Gemma.RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postFeedforwardLayerNorm.wrappedValue = Gemma.RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
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

        let h = Gemma.clipResidual(x, attnNorm)
        var out: MLXArray

        if isMoe {
            let preMLPNorm = preFeedforwardLayerNorm(h)
            let denseOut = mlp(preMLPNorm)
            let densePostNorm1 = postFeedforwardLayerNorm1!(denseOut)

            let sparsePreNorm = preFeedforwardLayerNorm2!(h)
            let sparseOut = expertsBlock!(sparsePreNorm)
            let sparsePostNorm2 = postFeedforwardLayerNorm2!(sparseOut)

            let combined = densePostNorm1 + sparsePostNorm2
            let postMLPNorm = postFeedforwardLayerNorm(combined)
            out = Gemma.clipResidual(h, postMLPNorm)
        } else {
            let preMLPNorm = preFeedforwardLayerNorm(h)
            let r2 = mlp(preMLPNorm)
            let postMLPNorm = postFeedforwardLayerNorm(r2)
            out = Gemma.clipResidual(h, postMLPNorm)
        }

        // Per-layer conditioning residual (Gemma 4 architectural novelty)
        // Applied after attn+MLP, before layer_scalar: residual += norm(proj(gelu(gate(h)) * perLayerEmbed))
        if hasPerLayerInput,
           let pli = perLayerInput,
           let gate = perLayerInputGate,
           let proj = perLayerProjectionLayer,
           let norm = postPerLayerInputNorm
        {
            let residual = out
            var gated = gate(out)             // [B, L, hiddenSizePerLayerInput]
            gated = geluApproximate(gated)   // activation
            gated = gated * pli              // element-wise with per-layer token embedding
            gated = proj(gated)              // [B, L, hiddenSize]
            gated = norm(gated)
            out = residual + gated
        }

        return out * layerScalar
    }
}

public class Gemma4ModelInternal: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo var layers: [Gemma4TransformerBlock]
    @ModuleInfo var norm: Gemma.RMSNorm

    // Per-layer conditioning weights (Gemma 4 architectural novelty)
    @ModuleInfo(key: "embed_tokens_per_layer") var embedTokensPerLayer: Embedding?
    @ModuleInfo(key: "per_layer_model_projection") var perLayerModelProjection: Linear?
    @ModuleInfo(key: "per_layer_projection_norm") var perLayerProjectionNorm: Gemma.RMSNorm?

    let config: Gemma4Configuration

    init(_ config: Gemma4Configuration) {
        self.config = config

        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize,
            dimensions: config.hiddenSize
        )

        self._layers.wrappedValue = (0 ..< config.hiddenLayers).map { layerIdx in
            Gemma4TransformerBlock(config, layerIdx: layerIdx)
        }

        self.norm = Gemma.RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)

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
            self._perLayerProjectionNorm.wrappedValue = Gemma.RMSNorm(
                dimensions: config.hiddenSizePerLayerInput, eps: config.rmsNormEps)
        }

        super.init()
    }

    func callAsFunction(
        _ inputs: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode? = nil,
        cache: [KVCache?]? = nil
    ) -> MLXArray {
        var h = embedTokens(inputs)
        let scale = MLXArray(sqrt(Float(config.hiddenSize)), dtype: .bfloat16)
        h = h * scale.asType(h.dtype)

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
            h = layer(h, mask: layerMask, cache: layerCache?[i], perLayerInput: pli)
        }
        return norm(h)
    }
}

public class Gemma4Model: Module, LLMModel {

    @ModuleInfo public var model: Gemma4ModelInternal
    @ModuleInfo(key: "lm_head") var lmHead: Linear

    public let config: Gemma4Configuration
    public var vocabularySize: Int { config.vocabularySize }

    public init(_ config: Gemma4Configuration) {
        self.config = config
        self.model = Gemma4ModelInternal(config)
        self._lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabularySize, bias: false)
        super.init()
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var out = model(inputs, mask: nil, cache: cache)
        out = lmHead(out)
        // Apply final logit softcapping: logits = tanh(logits / cap) * cap
        // Critical for Gemma 4 output quality — without this logits blow up.
        if config.finalLogitSoftcapping > 0 {
            let cap = config.finalLogitSoftcapping
            out = MLX.tanh(out / cap) * cap
        }
        return out
    }

    public func sanitize(weights: [String: MLXArray])
        -> [String: MLXArray]
    {
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

        if processedWeights["lm_head.weight"] == nil {
            ["weight", "scales", "biases"].forEach { key in
                if let embedWeight = processedWeights["model.embed_tokens.\(key)"] {
                    processedWeights["lm_head.\(key)"] = embedWeight
                }
            }
        }
        
        // Remap router keys to be nested under experts block
        var finalWeights = [String: MLXArray]()
        for (k, v) in processedWeights {
            let newK = k.replacingOccurrences(of: ".router.", with: ".experts.router.")
            finalWeights[newK] = v
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
        model.layers
    }
}
