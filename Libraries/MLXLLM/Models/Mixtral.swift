import Foundation
import MLX
import MLXLMCommon
import MLXNN

// Port of https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/mixtral.py

public struct MixtralConfiguration: Codable, Sendable {
    var modelType: String = "mixtral"
    var vocabularySize: Int = 32000
    var hiddenSize: Int = 4096
    var intermediateSize: Int = 14336
    var hiddenLayers: Int = 32
    var attentionHeads: Int = 32
    var kvHeads: Int = 8
    var numLocalExperts: Int = 8
    var numExpertsPerToken: Int = 2
    var rmsNormEps: Float = 1e-5
    var ropeTheta: Float = 1_000_000
    var ropeTraditional: Bool = false
    var tieWordEmbeddings: Bool = false

    var headDim: Int { hiddenSize / attentionHeads }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabularySize = "vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case numLocalExperts = "num_local_experts"
        case numExpertsPerToken = "num_experts_per_tok"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case ropeTraditional = "rope_traditional"
        case tieWordEmbeddings = "tie_word_embeddings"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.modelType = try container.decodeIfPresent(String.self, forKey: .modelType) ?? "mixtral"
        self.vocabularySize =
            try container.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 32000
        self.hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        self.intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)
        self.hiddenLayers = try container.decode(Int.self, forKey: .hiddenLayers)
        self.attentionHeads = try container.decode(Int.self, forKey: .attentionHeads)
        self.kvHeads = try container.decodeIfPresent(Int.self, forKey: .kvHeads) ?? attentionHeads
        self.numLocalExperts =
            try container.decodeIfPresent(Int.self, forKey: .numLocalExperts) ?? 8
        self.numExpertsPerToken =
            try container.decodeIfPresent(Int.self, forKey: .numExpertsPerToken) ?? 2
        self.rmsNormEps = try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-5
        self.ropeTheta = try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 1_000_000
        self.ropeTraditional =
            try container.decodeIfPresent(Bool.self, forKey: .ropeTraditional) ?? false
        self.tieWordEmbeddings =
            try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
    }
}

class MixtralAttention: Module {
    let args: MixtralConfiguration
    let scale: Float

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear

    let rope: RoPE

    init(_ args: MixtralConfiguration) {
        self.args = args

        let dim = args.hiddenSize
        let heads = args.attentionHeads
        let kvHeads = args.kvHeads
        let headDim = args.headDim
        self.scale = pow(Float(headDim), -0.5)

        self._wq.wrappedValue = Linear(dim, heads * headDim, bias: false)
        self._wk.wrappedValue = Linear(dim, kvHeads * headDim, bias: false)
        self._wv.wrappedValue = Linear(dim, kvHeads * headDim, bias: false)
        self._wo.wrappedValue = Linear(heads * headDim, dim, bias: false)

        self.rope = RoPE(
            dimensions: headDim, traditional: args.ropeTraditional, base: args.ropeTheta)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        var queries = wq(x).reshaped(B, L, args.attentionHeads, -1).transposed(0, 2, 1, 3)
        var keys = wk(x).reshaped(B, L, args.kvHeads, -1).transposed(0, 2, 1, 3)
        let values = wv(x).reshaped(B, L, args.kvHeads, -1).transposed(0, 2, 1, 3)

        let offset = cache?.ropeOffset
        queries = applyRotaryPosition(rope, to: queries, offset: offset)
        keys = applyRotaryPosition(rope, to: keys, offset: offset)

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

        return wo(output)
    }
}

class MixtralSparseMoeBlock: Module {
    let numExperts: Int
    let topK: Int

    @ModuleInfo(key: "gate") var gate: Linear
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU

    init(_ args: MixtralConfiguration) {
        self.numExperts = args.numLocalExperts
        self.topK = args.numExpertsPerToken

        self._gate.wrappedValue = Linear(args.hiddenSize, numExperts, bias: false)
        self._switchMLP.wrappedValue = SwitchGLU(
            inputDims: args.hiddenSize, hiddenDims: args.intermediateSize, numExperts: numExperts)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let gates = gate(x)

        let k = topK
        let inds = MLX.stopGradient(MLX.argPartition(-gates, kth: k - 1, axis: -1)[.ellipsis, ..<k])
        // Mixtral applies the softmax over the *selected* expert logits (after top-k),
        // unlike the softmax-then-select ordering used by e.g. Qwen3MoE.
        let scores = MLX.softmax(MLX.takeAlong(gates, inds, axis: -1), axis: -1, precise: true)

        let y = switchMLP(x, inds)
        return weightedExpertSum(y, scores)
    }
}

class MixtralDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: MixtralAttention
    @ModuleInfo(key: "block_sparse_moe") var blockSparseMoe: MixtralSparseMoeBlock
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(_ args: MixtralConfiguration) {
        self._selfAttn.wrappedValue = MixtralAttention(args)
        self._blockSparseMoe.wrappedValue = MixtralSparseMoeBlock(args)
        self._inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let r = selfAttn(inputLayerNorm(x), mask: mask, cache: cache)
        let h = x + r
        let out = h + blockSparseMoe(postAttentionLayerNorm(h))
        return out
    }
}

public class MixtralModelInner: Module {
    let args: MixtralConfiguration

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    fileprivate let layers: [MixtralDecoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    init(_ args: MixtralConfiguration) {
        self.args = args
        precondition(args.vocabularySize > 0)

        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)
        self.layers = (0 ..< args.hiddenLayers).map { _ in MixtralDecoderLayer(args) }
        self._norm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var h = embedTokens(inputs)

        let mask = createAttentionMask(h: h, cache: cache?.first)

        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[i])
        }

        return norm(h)
    }
}

public class MixtralModel: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    public let model: MixtralModelInner
    let configuration: MixtralConfiguration

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ args: MixtralConfiguration) {
        self.configuration = args
        self.vocabularySize = args.vocabularySize
        self.kvHeads = Array(repeating: args.kvHeads, count: args.hiddenLayers)
        self.model = MixtralModelInner(args)

        if !args.tieWordEmbeddings {
            self._lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabularySize, bias: false)
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let out = model(inputs, cache: cache)
        if let lmHead {
            return lmHead(out)
        }
        return model.embedTokens.asLinear(out)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitizedWeights = weights

        if configuration.tieWordEmbeddings {
            sanitizedWeights["lm_head.weight"] = nil
        }

        if sanitizedWeights["model.layers.0.block_sparse_moe.experts.0.w1.weight"] == nil {
            return sanitizedWeights
        }

        for l in 0 ..< configuration.hiddenLayers {
            let prefix = "model.layers.\(l)"
            for (n, m) in [("w1", "gate_proj"), ("w2", "down_proj"), ("w3", "up_proj")] {
                for k in ["weight", "scales", "biases"] {
                    if sanitizedWeights["\(prefix).block_sparse_moe.experts.0.\(n).\(k)"] != nil {
                        let toJoin = (0 ..< configuration.numLocalExperts).map { e in
                            sanitizedWeights.removeValue(
                                forKey: "\(prefix).block_sparse_moe.experts.\(e).\(n).\(k)")!
                        }
                        sanitizedWeights["\(prefix).block_sparse_moe.switch_mlp.\(m).\(k)"] =
                            MLX.stacked(toJoin)
                    }
                }
            }
        }

        return sanitizedWeights
    }
}

// MARK: - LoRA

extension MixtralModel: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}
