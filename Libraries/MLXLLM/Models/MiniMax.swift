//
//  MiniMax.swift
//  LLM
//
//  Port of https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/minimax.py
//  Created by Ronald Mannak on 2025/1/8.
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

class MiniMaxAttention: Module {
    let args: MiniMaxConfiguration
    let scale: Float

    let numAttentionHeads: Int
    let numKeyValueHeads: Int
    let headDim: Int

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear

    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm?
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm?

    let rope: RoPE

    init(_ args: MiniMaxConfiguration) {
        self.args = args
        self.numAttentionHeads = args.attentionHeads
        self.numKeyValueHeads = args.kvHeads
        self.headDim = args.headDim ?? (args.hiddenSize / args.attentionHeads)
        self.scale = pow(Float(headDim), -0.5)

        _wq.wrappedValue = Linear(
            args.hiddenSize, numAttentionHeads * headDim, bias: false)
        _wk.wrappedValue = Linear(
            args.hiddenSize, numKeyValueHeads * headDim, bias: false)
        _wv.wrappedValue = Linear(
            args.hiddenSize, numKeyValueHeads * headDim, bias: false)
        _wo.wrappedValue = Linear(
            numAttentionHeads * headDim, args.hiddenSize, bias: false)

        if args.useQkNorm {
            _qNorm.wrappedValue = RMSNorm(
                dimensions: numAttentionHeads * headDim, eps: args.rmsNormEps)
            _kNorm.wrappedValue = RMSNorm(
                dimensions: numKeyValueHeads * headDim, eps: args.rmsNormEps)
        }

        self.rope = RoPE(
            dimensions: args.rotaryDim,
            traditional: false,
            base: args.ropeTheta
        )
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        var queries = wq(x)
        var keys = wk(x)
        let values = wv(x)

        if let qNorm, let kNorm {
            queries = qNorm(queries)
            keys = kNorm(keys)
        }

        var q = queries.reshaped(B, L, numAttentionHeads, -1).transposed(0, 2, 1, 3)
        var k = keys.reshaped(B, L, numKeyValueHeads, -1).transposed(0, 2, 1, 3)
        let v = values.reshaped(B, L, numKeyValueHeads, -1).transposed(0, 2, 1, 3)

        if let cache {
            q = rope(q, offset: cache.offset)
            k = rope(k, offset: cache.offset)
        } else {
            q = rope(q)
            k = rope(k)
        }

        let output = attentionWithCacheUpdate(
            queries: q,
            keys: k,
            values: v,
            cache: cache,
            scale: scale,
            mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        return wo(output)
    }
}

class MiniMaxSparseMoeBlock: Module {
    let numExpertsPerTok: Int

    @ModuleInfo(key: "gate") var gate: Linear
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
    @ParameterInfo(key: "e_score_correction_bias") var eScoreCorrectionBias: MLXArray

    init(_ args: MiniMaxConfiguration) {
        self.numExpertsPerTok = args.numExpertsPerTok

        _gate.wrappedValue = Linear(args.hiddenSize, args.numLocalExperts, bias: false)
        _switchMLP.wrappedValue = SwitchGLU(
            inputDims: args.hiddenSize,
            hiddenDims: args.intermediateSize,
            numExperts: args.numLocalExperts
        )
        _eScoreCorrectionBias.wrappedValue = MLXArray.zeros([args.numLocalExperts])
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let gates = gate(x.asType(.float32))

        var scores = sigmoid(gates)
        let originalScores = scores
        scores = scores + eScoreCorrectionBias

        let k = numExpertsPerTok
        let inds = argPartition(-scores, kth: k - 1, axis: -1)[.ellipsis, ..<k]
        scores = takeAlong(originalScores, inds, axis: -1)

        scores = scores / (scores.sum(axis: -1, keepDims: true) + 1e-20)
        scores = scores.asType(x.dtype)

        let y = switchMLP(x, inds)
        return (y * scores[.ellipsis, .newAxis]).sum(axis: -2)
    }
}

class MiniMaxDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: MiniMaxAttention
    @ModuleInfo(key: "block_sparse_moe") var blockSparseMoe: MiniMaxSparseMoeBlock

    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(_ args: MiniMaxConfiguration) {
        _selfAttn.wrappedValue = MiniMaxAttention(args)
        _blockSparseMoe.wrappedValue = MiniMaxSparseMoeBlock(args)
        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        var hidden = x + selfAttn(inputLayerNorm(x), mask: mask, cache: cache)
        hidden = hidden + blockSparseMoe(postAttentionLayerNorm(hidden))
        return hidden
    }
}

public class MiniMaxModelInner: Module {
    let args: MiniMaxConfiguration

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    let layers: [MiniMaxDecoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    init(_ args: MiniMaxConfiguration) {
        self.args = args

        _embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)
        self.layers = (0 ..< args.hiddenLayers).map { _ in MiniMaxDecoderLayer(args) }
        _norm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
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

public class MiniMaxModel: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    public let model: MiniMaxModelInner
    let configuration: MiniMaxConfiguration
    let modelType: String

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ args: MiniMaxConfiguration) {
        self.configuration = args
        self.vocabularySize = args.vocabularySize
        self.kvHeads = Array(repeating: args.kvHeads, count: args.hiddenLayers)
        self.modelType = args.modelType
        self.model = MiniMaxModelInner(args)

        if !args.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabularySize, bias: false)
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

        func dequant(weight: MLXArray, scaleInv: MLXArray) -> MLXArray {
            let dtype = weight.dtype
            let bs = 128
            let (m, n) = (weight.shape[0], weight.shape[1])
            let padBottom = (bs - m % bs) % bs
            let padSide = (bs - n % bs) % bs

            var padded = padded(
                weight, widths: [.init((0, padBottom)), .init((0, padSide))])
            padded = padded.reshaped(
                [(m + padBottom) / bs, bs, (n + padSide) / bs, bs])
            let scaled = padded * scaleInv[0..., .newAxis, 0..., .newAxis]
            return scaled.reshaped([m + padBottom, n + padSide])[0 ..< m, 0 ..< n]
                .asType(dtype)
        }

        var newWeights: [String: MLXArray] = [:]
        for (key, value) in sanitizedWeights {
            if key.contains("weight_scale_inv") {
                let weightKey = key.replacingOccurrences(of: "_scale_inv", with: "")
                if let weight = sanitizedWeights[weightKey] {
                    newWeights[weightKey] = dequant(weight: weight, scaleInv: value)
                }
            } else if newWeights[key] == nil {
                newWeights[key] = value
            }
        }

        sanitizedWeights = newWeights.isEmpty ? sanitizedWeights : newWeights

        if sanitizedWeights["model.layers.0.block_sparse_moe.experts.0.w1.weight"] == nil {
            return sanitizedWeights
        }

        for layerIndex in 0 ..< configuration.hiddenLayers {
            let prefix = "model.layers.\(layerIndex)"
            for (orig, updated) in [("w1", "gate_proj"), ("w2", "down_proj"), ("w3", "up_proj")] {
                for key in ["weight", "scales", "biases"] {
                    let firstKey = "\(prefix).block_sparse_moe.experts.0.\(orig).\(key)"
                    if sanitizedWeights[firstKey] != nil {
                        let toJoin = (0 ..< configuration.numLocalExperts).map { expertIndex in
                            sanitizedWeights.removeValue(
                                forKey:
                                    "\(prefix).block_sparse_moe.experts.\(expertIndex).\(orig).\(key)"
                            )!
                        }
                        sanitizedWeights[
                            "\(prefix).block_sparse_moe.switch_mlp.\(updated).\(key)"
                        ] = MLX.stacked(toJoin)
                    }
                }
            }
        }

        return sanitizedWeights
    }
}

// MARK: - Configuration

public struct MiniMaxConfiguration: Codable, Sendable {
    var modelType: String = "minimax"
    var hiddenSize: Int
    var intermediateSize: Int
    var attentionHeads: Int
    var kvHeads: Int
    var maxPositionEmbeddings: Int
    var numExpertsPerTok: Int
    var numLocalExperts: Int
    var sharedIntermediateSize: Int
    var hiddenLayers: Int
    var rmsNormEps: Float
    var ropeTheta: Float
    var rotaryDim: Int
    var vocabularySize: Int
    var tieWordEmbeddings: Bool = false
    var scoringFunc: String = "sigmoid"
    var headDim: Int?
    var useQkNorm: Bool = true

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case maxPositionEmbeddings = "max_position_embeddings"
        case numExpertsPerTok = "num_experts_per_tok"
        case numLocalExperts = "num_local_experts"
        case sharedIntermediateSize = "shared_intermediate_size"
        case hiddenLayers = "num_hidden_layers"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case rotaryDim = "rotary_dim"
        case vocabularySize = "vocab_size"
        case tieWordEmbeddings = "tie_word_embeddings"
        case scoringFunc = "scoring_func"
        case headDim = "head_dim"
        case useQkNorm = "use_qk_norm"
    }
}

// MARK: - LoRA

extension MiniMaxModel: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}
