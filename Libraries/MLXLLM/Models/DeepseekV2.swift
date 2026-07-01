// port of https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/deepseek_v2.py

import Foundation
import MLX
import MLXLMCommon
import MLXNN

public struct DeepseekV2Configuration: Codable, Sendable {
    var vocabSize: Int = 102400
    var hiddenSize: Int = 4096
    var intermediateSize: Int = 11008
    var moeIntermediateSize: Int = 1407
    var numHiddenLayers: Int = 30
    var numAttentionHeads: Int = 32
    var numKeyValueHeads: Int = 32
    var nSharedExperts: Int?
    var nRoutedExperts: Int?
    var routedScalingFactor: Float = 1.0
    var kvLoraRank: Int = 512
    // Optional: DeepSeek-V2-Lite sets `q_lora_rank: null` (direct q_proj, no
    // low-rank q compression); full DeepSeek-V2 sets it to 1536. nil ⇒ q_proj.
    var qLoraRank: Int?
    var qkRopeHeadDim: Int = 64
    var vHeadDim: Int = 128
    var qkNopeHeadDim: Int = 128
    var topkMethod: String = "gready"
    var nGroup: Int?
    var topkGroup: Int?
    var numExpertsPerTok: Int?
    var moeLayerFreq: Int = 1
    var firstKDenseReplace: Int = 0
    var maxPositionEmbeddings: Int = 2048
    var rmsNormEps: Float = 1e-6
    var ropeTheta: Float = 10000.0
    var ropeScaling: [String: StringOrNumber]?
    var attentionBias: Bool = false

    enum CodingKeys: String, CodingKey {
        case vocabSize = "vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case moeIntermediateSize = "moe_intermediate_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case nSharedExperts = "n_shared_experts"
        case nRoutedExperts = "n_routed_experts"
        case routedScalingFactor = "routed_scaling_factor"
        case kvLoraRank = "kv_lora_rank"
        case qLoraRank = "q_lora_rank"
        case qkRopeHeadDim = "qk_rope_head_dim"
        case vHeadDim = "v_head_dim"
        case qkNopeHeadDim = "qk_nope_head_dim"
        case topkMethod = "topk_method"
        case nGroup = "n_group"
        case topkGroup = "topk_group"
        case numExpertsPerTok = "num_experts_per_tok"
        case moeLayerFreq = "moe_layer_freq"
        case firstKDenseReplace = "first_k_dense_replace"
        case maxPositionEmbeddings = "max_position_embeddings"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case ropeScaling = "rope_scaling"
        case attentionBias = "attention_bias"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.vocabSize = try c.decodeIfPresent(Int.self, forKey: .vocabSize) ?? 102400
        self.hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 4096
        self.intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 11008
        self.moeIntermediateSize =
            try c.decodeIfPresent(Int.self, forKey: .moeIntermediateSize) ?? 1407
        self.numHiddenLayers = try c.decodeIfPresent(Int.self, forKey: .numHiddenLayers) ?? 30
        self.numAttentionHeads = try c.decodeIfPresent(Int.self, forKey: .numAttentionHeads) ?? 32
        self.numKeyValueHeads = try c.decodeIfPresent(Int.self, forKey: .numKeyValueHeads) ?? 32
        self.nSharedExperts = try c.decodeIfPresent(Int.self, forKey: .nSharedExperts)
        self.nRoutedExperts = try c.decodeIfPresent(Int.self, forKey: .nRoutedExperts)
        self.routedScalingFactor =
            try c.decodeIfPresent(Float.self, forKey: .routedScalingFactor) ?? 1.0
        self.kvLoraRank = try c.decodeIfPresent(Int.self, forKey: .kvLoraRank) ?? 512
        self.qLoraRank = try c.decodeIfPresent(Int.self, forKey: .qLoraRank)
        self.qkRopeHeadDim = try c.decodeIfPresent(Int.self, forKey: .qkRopeHeadDim) ?? 64
        self.vHeadDim = try c.decodeIfPresent(Int.self, forKey: .vHeadDim) ?? 128
        self.qkNopeHeadDim = try c.decodeIfPresent(Int.self, forKey: .qkNopeHeadDim) ?? 128
        self.topkMethod = try c.decodeIfPresent(String.self, forKey: .topkMethod) ?? "gready"
        self.nGroup = try c.decodeIfPresent(Int.self, forKey: .nGroup)
        self.topkGroup = try c.decodeIfPresent(Int.self, forKey: .topkGroup)
        self.numExpertsPerTok = try c.decodeIfPresent(Int.self, forKey: .numExpertsPerTok)
        self.moeLayerFreq = try c.decodeIfPresent(Int.self, forKey: .moeLayerFreq) ?? 1
        self.firstKDenseReplace = try c.decodeIfPresent(Int.self, forKey: .firstKDenseReplace) ?? 0
        self.maxPositionEmbeddings =
            try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 2048
        self.rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        self.ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10000.0
        self.ropeScaling = try c.decodeIfPresent(
            [String: StringOrNumber].self, forKey: .ropeScaling)
        self.attentionBias = try c.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
    }
}

class DeepseekV2Attention: Module {
    let config: DeepseekV2Configuration
    let numHeads: Int
    let qLoraRank: Int?
    let qkRopeHeadDim: Int
    let kvLoraRank: Int
    let vHeadDim: Int
    let qkNopeHeadDim: Int
    let qHeadDim: Int
    var scale: Float

    let rope: RoPELayer
    @ModuleInfo(key: "q_proj") var qProj: Linear?
    @ModuleInfo(key: "q_a_proj") var qAProj: Linear?
    @ModuleInfo(key: "q_a_layernorm") var qALayerNorm: RMSNorm?
    @ModuleInfo(key: "q_b_proj") var qBProj: Linear?
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "kv_a_proj_with_mqa") var kvAProjWithMqa: Linear
    @ModuleInfo(key: "kv_a_layernorm") var kvALayerNorm: RMSNorm
    @ModuleInfo(key: "kv_b_proj") var kvBProj: Linear

    init(config: DeepseekV2Configuration) {
        self.config = config
        self.numHeads = config.numAttentionHeads
        self.qLoraRank = (config.qLoraRank ?? 0) > 0 ? config.qLoraRank : nil
        self.qkRopeHeadDim = config.qkRopeHeadDim
        self.kvLoraRank = config.kvLoraRank
        self.vHeadDim = config.vHeadDim
        self.qkNopeHeadDim = config.qkNopeHeadDim
        self.qHeadDim = config.qkNopeHeadDim + config.qkRopeHeadDim
        self.scale = pow(Float(qHeadDim), -0.5)

        if let qLoraRank = self.qLoraRank {
            self._qAProj.wrappedValue = Linear(
                config.hiddenSize, qLoraRank, bias: config.attentionBias)
            self._qALayerNorm.wrappedValue = RMSNorm(dimensions: qLoraRank, eps: 1e-6)
            self._qBProj.wrappedValue = Linear(qLoraRank, numHeads * qHeadDim, bias: false)
        } else {
            self._qProj.wrappedValue = Linear(
                config.hiddenSize, numHeads * qHeadDim, bias: false)
        }

        self._kvAProjWithMqa.wrappedValue = Linear(
            config.hiddenSize, kvLoraRank + qkRopeHeadDim, bias: config.attentionBias)
        self._kvALayerNorm.wrappedValue = RMSNorm(dimensions: kvLoraRank, eps: 1e-6)
        self._kvBProj.wrappedValue = Linear(
            kvLoraRank, numHeads * (qHeadDim - qkRopeHeadDim + vHeadDim), bias: false)
        self._oProj.wrappedValue = Linear(
            numHeads * vHeadDim, config.hiddenSize, bias: config.attentionBias)

        // YaRN mscale applied to the attention scale, mirroring deepseek_v2.py.
        if let ropeScaling = config.ropeScaling {
            let mScaleAllDim = ropeScaling["mscale_all_dim"]?.asFloat() ?? 0.0
            let scalingFactor = ropeScaling["factor"]?.asFloat() ?? 1.0
            if mScaleAllDim != 0, scalingFactor > 1 {
                let s = 0.1 * mScaleAllDim * log(scalingFactor) + 1.0
                self.scale = self.scale * s * s
            }
        }

        self.rope = initializeRope(
            dims: qkRopeHeadDim, base: config.ropeTheta, traditional: true,
            scalingConfig: config.ropeScaling, maxPositionEmbeddings: config.maxPositionEmbeddings)
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
        let qNope = splitQ[0]
        var qPe = splitQ[1]

        var compressedKv = kvAProjWithMqa(x)
        let splitKv = split(compressedKv, indices: [kvLoraRank], axis: -1)
        compressedKv = splitKv[0]
        var kPe = splitKv[1]
        kPe = kPe.reshaped(B, L, 1, qkRopeHeadDim).transposed(0, 2, 1, 3)

        var kv = kvBProj(kvALayerNorm(compressedKv))
        kv = kv.reshaped(B, L, numHeads, -1).transposed(0, 2, 1, 3)
        let splitKv2 = split(kv, indices: [qkNopeHeadDim], axis: -1)
        let kNope = splitKv2[0]
        var values = splitKv2[1]

        let offset = cache?.ropeOffset
        qPe = applyRotaryPosition(rope, to: qPe, offset: offset)
        kPe = applyRotaryPosition(rope, to: kPe, offset: offset)
        kPe = repeated(kPe, count: numHeads, axis: 1)

        var keys: MLXArray
        if let cache = cache {
            (keys, values) = cache.update(
                keys: concatenated([kNope, kPe], axis: -1), values: values)
        } else {
            keys = concatenated([kNope, kPe], axis: -1)
        }

        let queries = concatenated([qNope, qPe], axis: -1)

        let output = attentionWithCacheUpdate(
            queries: queries, keys: keys, values: values,
            cache: cache, scale: scale, mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        return oProj(output)
    }
}

class DeepseekV2MLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(config: DeepseekV2Configuration, hiddenSize: Int? = nil, intermediateSize: Int? = nil) {
        let h = hiddenSize ?? config.hiddenSize
        let i = intermediateSize ?? config.intermediateSize
        self._gateProj.wrappedValue = Linear(h, i, bias: false)
        self._upProj.wrappedValue = Linear(h, i, bias: false)
        self._downProj.wrappedValue = Linear(i, h, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}

class DeepseekV2MoEGate: Module {
    let topK: Int
    let nRoutedExperts: Int
    let routedScalingFactor: Float
    let topkMethod: String
    let nGroup: Int
    let topkGroup: Int

    var weight: MLXArray

    init(config: DeepseekV2Configuration) {
        self.topK = config.numExpertsPerTok ?? 1
        self.nRoutedExperts = config.nRoutedExperts ?? 1
        self.routedScalingFactor = config.routedScalingFactor
        self.topkMethod = config.topkMethod
        self.nGroup = config.nGroup ?? 1
        self.topkGroup = config.topkGroup ?? 1
        self.weight = zeros([nRoutedExperts, config.hiddenSize])
    }

    func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray) {
        let (bsz, seqLen, _) = (x.dim(0), x.dim(1), x.dim(2))

        let gates = x.matmul(weight.T)
        var scores = softmax(gates, axis: -1, precise: true)

        if topkMethod == "group_limited_greedy" {
            // Mask out all but the top `topkGroup` expert groups (by the group's
            // max score), then top-k over the surviving experts.
            scores = scores.reshaped(bsz, seqLen, nGroup, -1)
            let groupScores = MLX.max(scores, axis: -1, keepDims: true)
            let k = nGroup - topkGroup
            var groupIdx = argPartition(groupScores, kth: k - 1, axis: -2)[.ellipsis, ..<k, 0...]
            groupIdx = broadcast(groupIdx, to: [bsz, seqLen, k, nRoutedExperts / nGroup])
            scores = putAlong(scores, stopGradient(groupIdx), values: MLXArray(0.0), axis: -2)
            scores = flattened(scores, start: -2, end: -1)
        }

        let inds = argPartition(-scores, kth: topK - 1, axis: -1)[.ellipsis, ..<topK]
        scores = takeAlong(scores, inds, axis: -1)
        scores = scores * routedScalingFactor
        return (inds, scores)
    }
}

class DeepseekV2MoE: Module, UnaryLayer {
    let numExpertsPerTok: Int
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
    let gate: DeepseekV2MoEGate
    @ModuleInfo(key: "shared_experts") var sharedExperts: DeepseekV2MLP?

    init(config: DeepseekV2Configuration) {
        self.numExpertsPerTok = config.numExpertsPerTok ?? 1
        self._switchMLP.wrappedValue = SwitchGLU(
            inputDims: config.hiddenSize,
            hiddenDims: config.moeIntermediateSize,
            numExperts: config.nRoutedExperts ?? 1)
        self.gate = DeepseekV2MoEGate(config: config)

        if let shared = config.nSharedExperts {
            self._sharedExperts.wrappedValue = DeepseekV2MLP(
                config: config, intermediateSize: config.moeIntermediateSize * shared)
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (inds, scores) = gate(x)
        var y = switchMLP(x, inds)
        y = weightedExpertSum(y, scores)
        if let shared = sharedExperts {
            y = y + shared(x)
        }
        return y
    }
}

class DeepseekV2DecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: DeepseekV2Attention
    var mlp: UnaryLayer
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(config: DeepseekV2Configuration, layerIdx: Int) {
        self._selfAttn.wrappedValue = DeepseekV2Attention(config: config)

        if config.nRoutedExperts != nil,
            layerIdx >= config.firstKDenseReplace,
            layerIdx % config.moeLayerFreq == 0
        {
            self.mlp = DeepseekV2MoE(config: config)
        } else {
            self.mlp = DeepseekV2MLP(config: config)
        }

        self._inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let r = selfAttn(inputLayerNorm(x), mask: mask, cache: cache)
        let h = x + r
        return h + mlp(postAttentionLayerNorm(h))
    }
}

public class DeepseekV2ModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    fileprivate let layers: [DeepseekV2DecoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    init(config: DeepseekV2Configuration) {
        precondition(config.vocabSize > 0)
        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabSize, dimensions: config.hiddenSize)
        self.layers = (0 ..< config.numHiddenLayers).map {
            DeepseekV2DecoderLayer(config: config, layerIdx: $0)
        }
        self._norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(_ x: MLXArray, cache: [KVCache]?) -> MLXArray {
        var h = embedTokens(x)
        let mask = createAttentionMask(h: h, cache: cache?.first)
        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[i])
        }
        return norm(h)
    }
}

public class DeepseekV2Model: Module, LLMModel, KVCacheDimensionProvider, LoRAModel {
    public var kvHeads: [Int]
    let config: DeepseekV2Configuration
    public let model: DeepseekV2ModelInner
    @ModuleInfo(key: "lm_head") var lmHead: Linear

    public init(_ config: DeepseekV2Configuration) {
        self.config = config
        self.kvHeads = Array(repeating: config.numKeyValueHeads, count: config.numHiddenLayers)
        self.model = DeepseekV2ModelInner(config: config)
        self._lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabSize, bias: false)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        lmHead(model(inputs, cache: cache))
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var newWeights = weights
        for l in 0 ..< config.numHiddenLayers {
            let prefix = "model.layers.\(l)"
            for projName in ["gate_proj", "down_proj", "up_proj"] {
                for key in ["weight", "scales", "biases"] {
                    let firstKey = "\(prefix).mlp.experts.0.\(projName).\(key)"
                    if weights[firstKey] != nil {
                        let joined = (0 ..< (config.nRoutedExperts ?? 1)).map {
                            newWeights.removeValue(
                                forKey: "\(prefix).mlp.experts.\($0).\(projName).\(key)")!
                        }
                        newWeights["\(prefix).mlp.switch_mlp.\(projName).\(key)"] = stacked(joined)
                    }
                }
            }
        }
        return newWeights.filter { key, _ in !key.contains("rotary_emb.inv_freq") }
    }

    public var loraLayers: [Module] {
        model.layers
    }
}
