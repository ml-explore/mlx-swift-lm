//
//  GLM4MOELite.swift
//  LLM
//
//  Port of https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/glm4_moe_lite.py
//  Created by Ronald Mannak on 2025/1/7.
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - MultiLinear

/// Multi-head linear layer where each head has its own weight matrix
public class MultiLinear: Module, Quantizable {
    @ModuleInfo(key: "weight") var weight: MLXArray

    let groups: Int
    let inputDimensions: Int
    let outputDimensions: Int

    public init(groups: Int, inputDimensions: Int, outputDimensions: Int) {
        self.groups = groups
        self.inputDimensions = inputDimensions
        self.outputDimensions = outputDimensions

        let scale = sqrt(1.0 / Float(inputDimensions))
        self._weight.wrappedValue = MLXRandom.uniform(
            low: -scale,
            high: scale,
            [groups, outputDimensions, inputDimensions]
        )

        super.init()
    }

    /// Initializer for subclasses to provide weight directly
    public init(groups: Int, inputDimensions: Int, outputDimensions: Int, weight: MLXArray) {
        self.groups = groups
        self.inputDimensions = inputDimensions
        self.outputDimensions = outputDimensions
        self._weight.wrappedValue = weight
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // x shape: [B, numHeads, L, inputDims]
        // weight shape: [numHeads, outputDims, inputDims]
        // output shape: [B, numHeads, L, outputDims]
        return x.matmul(weight.swappedAxes(-1, -2))
    }

    public func toQuantized(groupSize: Int = 64, bits: Int = 4, mode: QuantizationMode) -> Module {
        QuantizedMultiLinear(self, groupSize: groupSize, bits: bits, mode: mode)
    }
}

public class QuantizedMultiLinear: MultiLinear, Quantized {
    @ModuleInfo(key: "scales") var scales: MLXArray
    @ModuleInfo(key: "biases") var biases: MLXArray?

    public let groupSize: Int
    public let bits: Int
    public let mode: QuantizationMode

    public init(
        _ other: MultiLinear, groupSize: Int = 64, bits: Int = 4, mode: QuantizationMode = .affine
    ) {
        self.groupSize = groupSize
        self.bits = bits
        self.mode = mode

        let (quantizedWeight, scales, biases) = MLX.quantized(
            other.weight, groupSize: groupSize, bits: bits, mode: mode)

        self._scales.wrappedValue = scales
        self._biases.wrappedValue = biases

        super.init(
            groups: other.groups,
            inputDimensions: other.inputDimensions,
            outputDimensions: other.outputDimensions,
            weight: quantizedWeight
        )

        self.freeze()
    }

    /// Initializer for loading quantized weights directly
    public init(
        groups: Int, inputDimensions: Int, outputDimensions: Int,
        weight: MLXArray, scales: MLXArray, biases: MLXArray?,
        groupSize: Int, bits: Int, mode: QuantizationMode
    ) {
        self.groupSize = groupSize
        self.bits = bits
        self.mode = mode
        self._scales.wrappedValue = scales
        self._biases.wrappedValue = biases

        super.init(
            groups: groups,
            inputDimensions: inputDimensions,
            outputDimensions: outputDimensions,
            weight: weight
        )

        self.freeze()
    }

    override public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Dequantize and compute
        let dequantized = MLX.dequantized(
            weight, scales: scales, biases: biases,
            groupSize: groupSize, bits: bits
        )
        return x.matmul(dequantized.swappedAxes(-1, -2))
    }
}

// MARK: - GLM4MoELiteAttention

class GLM4MoELiteAttention: Module {
    let config: GLM4MoELiteConfiguration
    let hiddenSize: Int
    let numHeads: Int
    let maxPositionEmbeddings: Int
    let ropeTheta: Float
    let qLoraRank: Int?
    let qkRopeHeadDim: Int
    let kvLoraRank: Int
    let vHeadDim: Int
    let qkNopeHeadDim: Int
    let qHeadDim: Int
    var scale: Float

    let rope: OffsetLayer
    @ModuleInfo(key: "q_proj") var qProj: Linear?
    @ModuleInfo(key: "q_a_proj") var qAProj: Linear?
    @ModuleInfo(key: "q_a_layernorm") var qALayerNorm: RMSNorm?
    @ModuleInfo(key: "q_b_proj") var qBProj: Linear?
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "kv_a_proj_with_mqa") var kvAProjWithMqa: Linear
    @ModuleInfo(key: "kv_a_layernorm") var kvALayerNorm: RMSNorm
    @ModuleInfo(key: "embed_q") var embedQ: MultiLinear
    @ModuleInfo(key: "unembed_out") var unembedOut: MultiLinear

    init(_ config: GLM4MoELiteConfiguration) {
        self.config = config
        self.hiddenSize = config.hiddenSize
        self.numHeads = config.attentionHeads
        self.maxPositionEmbeddings = config.maxPositionEmbeddings
        self.ropeTheta = config.ropeTheta
        self.qLoraRank = config.qLoraRank
        self.qkRopeHeadDim = config.qkRopeHeadDim
        self.kvLoraRank = config.kvLoraRank
        self.vHeadDim = config.vHeadDim
        self.qkNopeHeadDim = config.qkNopeHeadDim
        self.qHeadDim = config.qkNopeHeadDim + config.qkRopeHeadDim
        self.scale = pow(Float(qHeadDim), -0.5)

        if let qLoraRank {
            _qAProj.wrappedValue = Linear(hiddenSize, qLoraRank, bias: config.attentionBias)
            _qALayerNorm.wrappedValue = RMSNorm(dimensions: qLoraRank, eps: config.rmsNormEps)
            _qBProj.wrappedValue = Linear(qLoraRank, numHeads * qHeadDim, bias: false)
        } else {
            _qProj.wrappedValue = Linear(hiddenSize, numHeads * qHeadDim, bias: false)
        }

        _kvAProjWithMqa.wrappedValue = Linear(
            hiddenSize,
            kvLoraRank + qkRopeHeadDim,
            bias: config.attentionBias
        )
        _kvALayerNorm.wrappedValue = RMSNorm(dimensions: kvLoraRank, eps: config.rmsNormEps)

        // MultiLinear for embed_q and unembed_out
        _embedQ.wrappedValue = MultiLinear(
            groups: numHeads,
            inputDimensions: qkNopeHeadDim,
            outputDimensions: kvLoraRank
        )
        _unembedOut.wrappedValue = MultiLinear(
            groups: numHeads,
            inputDimensions: kvLoraRank,
            outputDimensions: vHeadDim
        )

        _oProj.wrappedValue = Linear(
            numHeads * vHeadDim, hiddenSize, bias: config.attentionBias)

        if let ropeScaling = config.ropeScaling,
            let mscaleAllDim = ropeScaling["mscale_all_dim"]?.asFloat(),
            let scalingFactor = ropeScaling["factor"]?.asFloat(),
            mscaleAllDim != 0,
            scalingFactor > 1
        {
            let s = 0.1 * mscaleAllDim * log(scalingFactor) + 1.0
            self.scale = self.scale * s * s
        }

        var ropeScaling = config.ropeScaling
        if let ropeType = ropeScaling?["type"] ?? ropeScaling?["rope_type"],
            case .string(let value) = ropeType,
            value == "deepseek_yarn"
        {
            var updated = ropeScaling ?? [:]
            updated["type"] = .string("yarn")
            ropeScaling = updated
        }

        self.rope = initializeRope(
            dims: qkRopeHeadDim,
            base: ropeTheta,
            traditional: config.ropeTraditional,
            scalingConfig: ropeScaling,
            maxPositionEmbeddings: maxPositionEmbeddings
        )
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L, _) = (x.dim(0), x.dim(1), x.dim(2))

        var q: MLXArray
        if qLoraRank == nil {
            q = qProj!(x)
        } else {
            q = qBProj!(qALayerNorm!(qAProj!(x)))
        }

        q = q.reshaped(B, L, numHeads, qHeadDim).transposed(0, 2, 1, 3)
        let splitQ = split(q, indices: [qkNopeHeadDim], axis: -1)
        var qNope = splitQ[0]
        var qPe = splitQ[1]

        var compressedKv = kvAProjWithMqa(x)
        let splitCompressedKv = split(compressedKv, indices: [kvLoraRank], axis: -1)
        compressedKv = splitCompressedKv[0]
        var kPe = splitCompressedKv[1]
        kPe = kPe.reshaped(B, L, 1, qkRopeHeadDim).transposed(0, 2, 1, 3)

        let kvLatent = kvALayerNorm(compressedKv)

        // Apply RoPE
        let offset = cache?.offset ?? 0
        qPe = rope(qPe, offset: offset)
        kPe = rope(kPe, offset: offset)

        // Expand kv_latent for attention: (B, L, kv_lora_rank) -> (B, 1, L, kv_lora_rank)
        let kvLatentExpanded = MLX.expandedDimensions(kvLatent, axis: 1)

        // Use embed_q to project q_nope from qk_nope_head_dim to kv_lora_rank
        qNope = embedQ(qNope)

        // Build keys: concatenate kv_latent with k_pe
        var keys = concatenated([kvLatentExpanded, kPe], axis: -1)

        // Update cache - store keys only, values are derived from keys
        // Python: keys, _ = cache.update_and_fetch(keys, mx.zeros((B, 1, L, 0)))
        if let cache {
            (keys, _) = cache.update(keys: keys, values: MLX.zeros([B, 1, L, 0]))
        }

        // Values are the kv_latent part of keys (excluding rope dimensions)
        // Python: values = keys[..., :-self.qk_rope_head_dim]
        // keys shape: (B, 1, L, kvLoraRank + qkRopeHeadDim)
        // values shape: (B, 1, L, kvLoraRank)
        let values = keys[.ellipsis, ..<kvLoraRank]

        // Build queries: concatenate projected q_nope with q_pe
        let queries = concatenated([qNope, qPe], axis: -1)

        // Compute attention
        var output = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            mask: mask
        )

        // Use unembed_out to project output from kv_lora_rank to v_head_dim
        output = unembedOut(output)

        output = output.transposed(0, 2, 1, 3).reshaped(B, L, -1)

        return oProj(output)
    }
}

class GLM4MoELiteMLP: Module, UnaryLayer {
    let hiddenSize: Int
    let intermediateSize: Int

    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(_ config: GLM4MoELiteConfiguration, hiddenSize: Int? = nil, intermediateSize: Int? = nil) {
        self.hiddenSize = hiddenSize ?? config.hiddenSize
        self.intermediateSize = intermediateSize ?? config.intermediateSize

        _gateProj.wrappedValue = Linear(self.hiddenSize, self.intermediateSize, bias: false)
        _upProj.wrappedValue = Linear(self.hiddenSize, self.intermediateSize, bias: false)
        _downProj.wrappedValue = Linear(self.intermediateSize, self.hiddenSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}

class GLM4MoELiteGate: Module {
    let topK: Int
    let normTopkProb: Bool
    let nRoutedExperts: Int
    let routedScalingFactor: Float
    let nGroup: Int
    let topkGroup: Int

    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "e_score_correction_bias") var eScoreCorrectionBias: MLXArray

    init(_ config: GLM4MoELiteConfiguration) {
        guard let nRoutedExperts = config.nRoutedExperts else {
            fatalError("GLM4MoELiteGate requires nRoutedExperts")
        }

        precondition(config.topkMethod == "noaux_tc", "Unsupported topk method.")

        self.topK = config.numExpertsPerTok
        self.normTopkProb = config.normTopkProb
        self.nRoutedExperts = nRoutedExperts
        self.routedScalingFactor = config.routedScalingFactor
        self.nGroup = config.nGroup
        self.topkGroup = config.topkGroup

        _weight.wrappedValue = zeros([nRoutedExperts, config.hiddenSize])
        _eScoreCorrectionBias.wrappedValue = zeros([nRoutedExperts])

        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray) {
        let hiddenStates = x.matmul(weight.T)
        let originalScores = sigmoid(hiddenStates.asType(.float32))
        var selectionScores = originalScores + eScoreCorrectionBias

        if nGroup > 1 {
            selectionScores = unflatten(selectionScores, axis: -1, shape: [nGroup, -1])
            let groupScores = top(selectionScores, k: 2, axis: -1).sum(axis: -1, keepDims: true)
            let k = nGroup - topkGroup
            let groupIdx = argPartition(groupScores, kth: k - 1, axis: -2)[.ellipsis, ..<k, 0...]
            selectionScores = putAlong(
                selectionScores, stopGradient(groupIdx), values: MLXArray(0.0), axis: -2)
            selectionScores = flattened(selectionScores, start: -2, end: -1)
        }

        let k = topK
        let inds = argPartition(-selectionScores, kth: k - 1, axis: -1)[.ellipsis, ..<k]
        var selectedScores = takeAlong(originalScores, inds, axis: -1)

        if topK > 1, normTopkProb {
            let denominator = selectedScores.sum(axis: -1, keepDims: true)
            selectedScores = selectedScores / denominator
        }
        selectedScores = selectedScores * routedScalingFactor

        return (inds, selectedScores)
    }
}

class GLM4MoELiteMoE: Module, UnaryLayer {
    let numExpertsPerTok: Int
    let gate: GLM4MoELiteGate

    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
    @ModuleInfo(key: "shared_experts") var sharedExperts: GLM4MoELiteMLP?

    init(_ config: GLM4MoELiteConfiguration) {
        guard let nRoutedExperts = config.nRoutedExperts else {
            fatalError("GLM4MoELiteMoE requires nRoutedExperts")
        }

        self.numExpertsPerTok = config.numExpertsPerTok
        self.gate = GLM4MoELiteGate(config)

        _switchMLP.wrappedValue = SwitchGLU(
            inputDims: config.hiddenSize,
            hiddenDims: config.moeIntermediateSize,
            numExperts: nRoutedExperts
        )

        if let shared = config.nSharedExperts, shared > 0 {
            let intermediateSize = config.moeIntermediateSize * shared
            _sharedExperts.wrappedValue = GLM4MoELiteMLP(
                config, intermediateSize: intermediateSize
            )
        }

        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (inds, scores) = gate(x)
        var y = switchMLP(x, inds)
        y = (y * scores[.ellipsis, .newAxis]).sum(axis: -2).asType(y.dtype)
        if let sharedExperts {
            y = y + sharedExperts(x)
        }
        return y
    }
}

class GLM4MoELiteDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: GLM4MoELiteAttention
    let mlp: UnaryLayer

    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(_ config: GLM4MoELiteConfiguration, layerIdx: Int) {
        _attention.wrappedValue = GLM4MoELiteAttention(config)

        if config.nRoutedExperts != nil,
            layerIdx >= config.firstKDenseReplace,
            layerIdx % config.moeLayerFreq == 0
        {
            self.mlp = GLM4MoELiteMoE(config)
        } else {
            self.mlp = GLM4MoELiteMLP(config)
        }

        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let r = attention(inputLayerNorm(x), mask: mask, cache: cache)
        let h = x + r
        let r2 = mlp(postAttentionLayerNorm(h))
        return h + r2
    }
}

public class GLM4MoELiteModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    let layers: [GLM4MoELiteDecoderLayer]
    let norm: RMSNorm

    init(_ config: GLM4MoELiteConfiguration) {
        precondition(config.vocabularySize > 0)

        _embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)

        self.layers = (0 ..< config.hiddenLayers)
            .map { idx in
                GLM4MoELiteDecoderLayer(config, layerIdx: idx)
            }
        self.norm = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var h = embedTokens(inputs)

        let mask = createAttentionMask(h: h, cache: cache?.first)

        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[i])
        }

        return norm(h)
    }
}

public class GLM4MoELiteModel: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    public let model: GLM4MoELiteModelInner
    let configuration: GLM4MoELiteConfiguration

    @ModuleInfo(key: "lm_head") var lmHead: Linear

    public init(_ args: GLM4MoELiteConfiguration) {
        self.configuration = args
        self.vocabularySize = args.vocabularySize
        self.kvHeads = (0 ..< args.hiddenLayers).map { _ in args.kvHeads }
        self.model = GLM4MoELiteModelInner(args)

        _lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabularySize, bias: false)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let out = model(inputs, cache: cache)
        return lmHead(out)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized = weights

        for l in 0 ..< configuration.hiddenLayers {
            let prefix = "model.layers.\(l)"
            for n in ["gate_proj", "down_proj", "up_proj"] {
                for k in ["weight", "scales", "biases"] {
                    let key = "\(prefix).mlp.experts.0.\(n).\(k)"
                    if sanitized[key] != nil, let nRoutedExperts = configuration.nRoutedExperts {
                        let toJoin = (0 ..< nRoutedExperts).map { e in
                            sanitized.removeValue(
                                forKey: "\(prefix).mlp.experts.\(e).\(n).\(k)")!
                        }
                        sanitized["\(prefix).mlp.switch_mlp.\(n).\(k)"] = MLX.stacked(toJoin)
                    }
                }
            }
        }

        let numMptLayers = configuration.numNextnPredictLayers
        if numMptLayers > 0 {
            sanitized = sanitized.filter { key, _ in
                for idx in 0 ..< numMptLayers {
                    if key.hasPrefix("model.layers.\(configuration.hiddenLayers + idx)") {
                        return false
                    }
                }
                return true
            }
        }

        return sanitized
    }
}

public struct GLM4MoELiteConfiguration: Codable, Sendable {
    var modelType: String
    var vocabularySize: Int
    var hiddenSize: Int
    var intermediateSize: Int
    var moeIntermediateSize: Int
    var hiddenLayers: Int
    var attentionHeads: Int
    var kvHeads: Int
    var nSharedExperts: Int?
    var nRoutedExperts: Int?
    var routedScalingFactor: Float
    var kvLoraRank: Int
    var qLoraRank: Int?
    var qkRopeHeadDim: Int
    var qkNopeHeadDim: Int
    var vHeadDim: Int
    var topkMethod: String
    var scoringFunc: String
    var normTopkProb: Bool
    var nGroup: Int
    var topkGroup: Int
    var numExpertsPerTok: Int
    var moeLayerFreq: Int
    var firstKDenseReplace: Int
    var maxPositionEmbeddings: Int
    var rmsNormEps: Float
    var ropeTheta: Float
    var ropeScaling: [String: StringOrNumber]?
    var ropeTraditional: Bool
    var attentionBias: Bool
    var attentionDropout: Float
    var partialRotaryFactor: Float
    var tieWordEmbeddings: Bool
    var numNextnPredictLayers: Int

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabularySize = "vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case moeIntermediateSize = "moe_intermediate_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case nSharedExperts = "n_shared_experts"
        case nRoutedExperts = "n_routed_experts"
        case routedScalingFactor = "routed_scaling_factor"
        case kvLoraRank = "kv_lora_rank"
        case qLoraRank = "q_lora_rank"
        case qkRopeHeadDim = "qk_rope_head_dim"
        case qkNopeHeadDim = "qk_nope_head_dim"
        case vHeadDim = "v_head_dim"
        case topkMethod = "topk_method"
        case scoringFunc = "scoring_func"
        case normTopkProb = "norm_topk_prob"
        case nGroup = "n_group"
        case topkGroup = "topk_group"
        case numExpertsPerTok = "num_experts_per_tok"
        case moeLayerFreq = "moe_layer_freq"
        case firstKDenseReplace = "first_k_dense_replace"
        case maxPositionEmbeddings = "max_position_embeddings"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case ropeScaling = "rope_scaling"
        case ropeTraditional = "rope_traditional"
        case attentionBias = "attention_bias"
        case attentionDropout = "attention_dropout"
        case partialRotaryFactor = "partial_rotary_factor"
        case tieWordEmbeddings = "tie_word_embeddings"
        case numNextnPredictLayers = "num_nextn_predict_layers"
    }

    public init(from decoder: Decoder) throws {
        let container: KeyedDecodingContainer<GLM4MoELiteConfiguration.CodingKeys> =
            try decoder.container(keyedBy: GLM4MoELiteConfiguration.CodingKeys.self)

        self.modelType = try container.decode(String.self, forKey: .modelType)
        self.vocabularySize = try container.decode(Int.self, forKey: .vocabularySize)
        self.hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        self.intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)
        self.moeIntermediateSize = try container.decode(Int.self, forKey: .moeIntermediateSize)
        self.hiddenLayers = try container.decode(Int.self, forKey: .hiddenLayers)
        self.attentionHeads = try container.decode(Int.self, forKey: .attentionHeads)
        self.kvHeads = try container.decode(Int.self, forKey: .kvHeads)
        self.nSharedExperts = try container.decodeIfPresent(Int.self, forKey: .nSharedExperts)
        self.nRoutedExperts = try container.decodeIfPresent(Int.self, forKey: .nRoutedExperts)
        self.routedScalingFactor = try container.decode(Float.self, forKey: .routedScalingFactor)
        self.kvLoraRank = try container.decode(Int.self, forKey: .kvLoraRank)
        self.qLoraRank = try container.decodeIfPresent(Int.self, forKey: .qLoraRank)
        self.qkRopeHeadDim = try container.decode(Int.self, forKey: .qkRopeHeadDim)
        self.qkNopeHeadDim = try container.decode(Int.self, forKey: .qkNopeHeadDim)
        self.vHeadDim = try container.decode(Int.self, forKey: .vHeadDim)
        self.topkMethod =
            try container.decodeIfPresent(String.self, forKey: .topkMethod) ?? "noaux_tc"
        self.scoringFunc =
            try container.decodeIfPresent(String.self, forKey: .scoringFunc) ?? "sigmoid"
        self.normTopkProb =
            try container.decodeIfPresent(Bool.self, forKey: .normTopkProb) ?? true
        self.nGroup = try container.decodeIfPresent(Int.self, forKey: .nGroup) ?? 1
        self.topkGroup = try container.decodeIfPresent(Int.self, forKey: .topkGroup) ?? 1
        self.numExpertsPerTok =
            try container.decodeIfPresent(Int.self, forKey: .numExpertsPerTok) ?? 4
        self.moeLayerFreq =
            try container.decodeIfPresent(Int.self, forKey: .moeLayerFreq) ?? 1
        self.firstKDenseReplace =
            try container.decodeIfPresent(Int.self, forKey: .firstKDenseReplace) ?? 1
        self.maxPositionEmbeddings = try container.decode(Int.self, forKey: .maxPositionEmbeddings)
        self.rmsNormEps = try container.decode(Float.self, forKey: .rmsNormEps)
        self.ropeTheta = try container.decode(Float.self, forKey: .ropeTheta)
        self.ropeScaling = try container.decodeIfPresent(
            [String: StringOrNumber].self, forKey: .ropeScaling)
        self.ropeTraditional =
            try container.decodeIfPresent(Bool.self, forKey: .ropeTraditional) ?? true
        self.attentionBias = try container.decode(Bool.self, forKey: .attentionBias)
        self.attentionDropout =
            try container.decodeIfPresent(Float.self, forKey: .attentionDropout) ?? 0.0
        self.partialRotaryFactor =
            try container.decodeIfPresent(Float.self, forKey: .partialRotaryFactor) ?? 1.0
        self.tieWordEmbeddings =
            try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings)
            ?? false
        self.numNextnPredictLayers =
            try container.decodeIfPresent(Int.self, forKey: .numNextnPredictLayers) ?? 1
    }
}

// MARK: - LoRA

extension GLM4MoELiteModel: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}
