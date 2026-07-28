//
//  Hunyuan.swift
//  mlx-swift-lm
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// port of https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/hunyuan_v1_dense.py

class HunyuanAttention: Module {
    let args: HunyuanConfiguration
    let scale: Float

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear

    @ModuleInfo(key: "query_layernorm") var queryNorm: RMSNorm?
    @ModuleInfo(key: "key_layernorm") var keyNorm: RMSNorm?

    let rope: DynamicNTKAlphaRoPE

    public init(_ args: HunyuanConfiguration) {
        self.args = args

        let dim = args.hiddenSize
        let heads = args.attentionHeads
        let kvHeads = args.kvHeads
        let headDim = args.headDim
        self.scale = pow(Float(headDim), -0.5)

        _wq.wrappedValue = Linear(dim, heads * headDim, bias: args.attentionBias)
        _wk.wrappedValue = Linear(dim, kvHeads * headDim, bias: args.attentionBias)
        _wv.wrappedValue = Linear(dim, kvHeads * headDim, bias: args.attentionBias)
        _wo.wrappedValue = Linear(heads * headDim, dim, bias: args.attentionBias)

        if args.useQkNorm {
            _queryNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.rmsNormEps)
            _keyNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.rmsNormEps)
        }

        var scalingAlpha: Float = 1.0
        if let alpha = args.ropeScaling?["alpha"]?.asFloat() {
            scalingAlpha = alpha
        }
        self.rope = DynamicNTKAlphaRoPE(
            dimensions: headDim, base: args.ropeTheta, scalingAlpha: scalingAlpha)
    }

    public func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        var queries = wq(x)
        var keys = wk(x)
        var values = wv(x)

        queries = queries.reshaped(B, L, args.attentionHeads, -1).transposed(0, 2, 1, 3)
        keys = keys.reshaped(B, L, args.kvHeads, -1).transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, args.kvHeads, -1).transposed(0, 2, 1, 3)

        let offset = cache?.ropeOffset
        queries = applyRotaryPosition(rope, to: queries, offset: offset)
        keys = applyRotaryPosition(rope, to: keys, offset: offset)

        if let queryNorm, let keyNorm {
            queries = queryNorm(queries)
            keys = keyNorm(keys)
        }

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

class HunyuanMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    @ModuleInfo(key: "up_proj") var up: Linear

    public init(dimensions: Int, hiddenDimensions: Int) {
        _gate.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        _down.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
        _up.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        down(silu(gate(x)) * up(x))
    }
}

class HunyuanTransformerBlock: Module {
    @ModuleInfo(key: "self_attn") var attention: HunyuanAttention
    let mlp: HunyuanMLP

    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    public init(_ args: HunyuanConfiguration) {
        _attention.wrappedValue = HunyuanAttention(args)
        self.mlp = HunyuanMLP(
            dimensions: args.hiddenSize, hiddenDimensions: args.intermediateSize)
        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    public func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        var r = attention(inputLayerNorm(x), mask: mask, cache: cache)
        let h = x + r
        r = mlp(postAttentionLayerNorm(h))
        let out = h + r
        return out
    }
}

public class HunyuanModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    fileprivate let layers: [HunyuanTransformerBlock]
    let norm: RMSNorm

    public init(_ args: HunyuanConfiguration) {
        precondition(args.vocabularySize > 0)

        _embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)

        self.layers = (0 ..< args.hiddenLayers).map { _ in
            HunyuanTransformerBlock(args)
        }
        self.norm = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var h = embedTokens(inputs)

        let mask = createAttentionMask(h: h, cache: cache?.first)

        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[i])
        }

        return norm(h)
    }
}

public class HunyuanModel: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    public let model: HunyuanModelInner
    let configuration: HunyuanConfiguration

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ args: HunyuanConfiguration) {
        self.configuration = args
        self.vocabularySize = args.vocabularySize
        self.kvHeads = (0 ..< args.hiddenLayers).map { _ in args.kvHeads }
        self.model = HunyuanModelInner(args)

        if !args.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabularySize, bias: false)
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var out = model(inputs, cache: cache)
        if let lmHead {
            out = lmHead(out)
        } else {
            out = model.embedTokens.asLinear(out)
        }
        return out
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var weights = weights

        if configuration.tieWordEmbeddings {
            weights["lm_head.weight"] = nil
        }

        return weights
    }
}

public struct HunyuanConfiguration: Codable, Sendable {
    public var hiddenSize: Int
    public var hiddenLayers: Int
    public var intermediateSize: Int
    public var attentionHeads: Int
    public var kvHeads: Int
    public var rmsNormEps: Float
    public var vocabularySize: Int
    public var headDim: Int
    public var ropeTheta: Float = 10000
    public var attentionBias: Bool = false
    public var useQkNorm: Bool = true
    public var ropeScaling: [String: StringOrNumber]? = nil
    public var tieWordEmbeddings: Bool = false
    public var maxPositionEmbeddings: Int = 32768

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case headDim = "head_dim"
        case ropeTheta = "rope_theta"
        case attentionBias = "attention_bias"
        case useQkNorm = "use_qk_norm"
        case ropeScaling = "rope_scaling"
        case tieWordEmbeddings = "tie_word_embeddings"
        case maxPositionEmbeddings = "max_position_embeddings"
    }

    /// Some Hunyuan configs (e.g. Hy-MT2-7B) spell `head_dim` as `attention_head_dim`.
    private enum AltCodingKeys: String, CodingKey {
        case attentionHeadDim = "attention_head_dim"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        self.hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        self.hiddenLayers = try container.decode(Int.self, forKey: .hiddenLayers)
        self.intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)
        self.attentionHeads = try container.decode(Int.self, forKey: .attentionHeads)
        self.kvHeads = try container.decode(Int.self, forKey: .kvHeads)
        self.rmsNormEps = try container.decode(Float.self, forKey: .rmsNormEps)
        self.vocabularySize = try container.decode(Int.self, forKey: .vocabularySize)
        let alt = try decoder.container(keyedBy: AltCodingKeys.self)
        self.headDim =
            try container.decodeIfPresent(Int.self, forKey: .headDim)
            ?? alt.decodeIfPresent(Int.self, forKey: .attentionHeadDim)
            ?? (hiddenSize / attentionHeads)
        self.ropeTheta =
            try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10000
        self.attentionBias =
            try container.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        self.useQkNorm =
            try container.decodeIfPresent(Bool.self, forKey: .useQkNorm) ?? true
        self.ropeScaling = try container.decodeIfPresent(
            [String: StringOrNumber].self, forKey: .ropeScaling)
        self.tieWordEmbeddings =
            try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        self.maxPositionEmbeddings =
            try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 32768
    }
}

// MARK: - LoRA

extension HunyuanModel: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}
