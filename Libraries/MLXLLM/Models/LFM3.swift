//
//  LFM3.swift
//  mlx-swift-lm
//

// Port of https://huggingface.co/LiquidAI/LFM3-1.7B/blob/main/modeling_lfm3.py
//
// LFM3 is LFM2 with sliding-window attention as a third layer type, a sigmoid
// gate per attention head, partial RoPE on full-attention layers, and an
// optional RMSNorm on each block output ("sandwich norm").

import Foundation
import MLX
import MLXLMCommon
import MLXNN

public struct LFM3Configuration: Codable, Sendable {
    public enum LayerType: String, Codable, Sendable {
        case conv
        case fullAttention = "full_attention"
        case slidingAttention = "sliding_attention"
    }

    public struct RopeParameters: Codable, Sendable {
        let ropeType: String
        let ropeTheta: Float
        let partialRotaryFactor: Float

        enum CodingKeys: String, CodingKey {
            case ropeType = "rope_type"
            case ropeTheta = "rope_theta"
            case partialRotaryFactor = "partial_rotary_factor"
        }

        init(ropeTheta: Float, partialRotaryFactor: Float = 1.0) {
            self.ropeType = "default"
            self.ropeTheta = ropeTheta
            self.partialRotaryFactor = partialRotaryFactor
        }

        public init(from decoder: Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            self.ropeType =
                try container.decodeIfPresent(String.self, forKey: .ropeType) ?? "default"
            self.ropeTheta = try container.decode(Float.self, forKey: .ropeTheta)
            self.partialRotaryFactor =
                try container.decodeIfPresent(Float.self, forKey: .partialRotaryFactor) ?? 1.0
        }
    }

    let modelType: String
    let vocabularySize: Int
    let hiddenSize: Int
    let hiddenLayers: Int
    let attentionHeads: Int
    let kvHeads: Int
    let headDim: Int
    let intermediateSize: Int
    let maxPositionEmbeddings: Int?
    let normEps: Float
    let convBias: Bool
    let convLCache: Int
    let layerTypes: [LayerType]
    let slidingWindow: Int
    let useSandwichNorm: Bool
    let tieWordEmbeddings: Bool
    let ropeParameters: [String: RopeParameters]

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabularySize = "vocab_size"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case intermediateSize = "intermediate_size"
        case maxPositionEmbeddings = "max_position_embeddings"
        case normEps = "norm_eps"
        case convBias = "conv_bias"
        case convLCache = "conv_L_cache"
        case layerTypes = "layer_types"
        case slidingWindow = "sliding_window"
        case useSandwichNorm = "use_sandwich_norm"
        case tieWordEmbeddings = "tie_word_embeddings"
        case ropeParameters = "rope_parameters"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        self.modelType = try container.decodeIfPresent(String.self, forKey: .modelType) ?? "lfm3"
        self.vocabularySize =
            try container.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 128_000
        self.hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        self.hiddenLayers = try container.decode(Int.self, forKey: .hiddenLayers)
        self.attentionHeads = try container.decode(Int.self, forKey: .attentionHeads)
        self.kvHeads = try container.decode(Int.self, forKey: .kvHeads)
        self.headDim =
            try container.decodeIfPresent(Int.self, forKey: .headDim)
            ?? hiddenSize / attentionHeads
        self.intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)
        self.maxPositionEmbeddings = try container.decodeIfPresent(
            Int.self, forKey: .maxPositionEmbeddings)
        self.normEps = try container.decodeIfPresent(Float.self, forKey: .normEps) ?? 1e-5
        self.convBias = try container.decodeIfPresent(Bool.self, forKey: .convBias) ?? false
        self.convLCache = try container.decodeIfPresent(Int.self, forKey: .convLCache) ?? 7
        self.slidingWindow =
            try container.decodeIfPresent(Int.self, forKey: .slidingWindow) ?? 256
        self.useSandwichNorm =
            try container.decodeIfPresent(Bool.self, forKey: .useSandwichNorm) ?? true
        self.tieWordEmbeddings =
            try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? true

        // Same default pattern as the reference: full attention every fourth layer from
        // the third, sliding attention in between, short convolutions elsewhere.
        let hiddenLayers = self.hiddenLayers
        self.layerTypes =
            try container.decodeIfPresent([LayerType].self, forKey: .layerTypes)
            ?? (0 ..< hiddenLayers).map { i in
                i % 4 == 2 ? .fullAttention : (i % 4 == 0 && i > 0 ? .slidingAttention : .conv)
            }
        guard layerTypes.count == hiddenLayers else {
            throw DecodingError.dataCorruptedError(
                forKey: .layerTypes, in: container,
                debugDescription:
                    "layer_types has \(layerTypes.count) entries but num_hidden_layers is \(hiddenLayers)"
            )
        }

        self.ropeParameters =
            try container.decodeIfPresent([String: RopeParameters].self, forKey: .ropeParameters)
            ?? [
                LayerType.fullAttention.rawValue: .init(
                    ropeTheta: 1_000_000, partialRotaryFactor: 0.5),
                LayerType.slidingAttention.rawValue: .init(ropeTheta: 10_000),
            ]
        for layerType in Set(layerTypes) where layerType != .conv {
            guard let rope = ropeParameters[layerType.rawValue] else {
                throw DecodingError.dataCorruptedError(
                    forKey: .ropeParameters, in: container,
                    debugDescription: "rope_parameters has no entry for \(layerType.rawValue)")
            }
            // Scaled RoPE types are not ported; running them as default would be wrong.
            guard rope.ropeType == "default" else {
                throw DecodingError.dataCorruptedError(
                    forKey: .ropeParameters, in: container,
                    debugDescription: "Unsupported LFM3 rope_type '\(rope.ropeType)'")
            }
        }
    }
}

class LFM3Attention: Module {
    let attentionHeads: Int
    let kvHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "gate_proj") var gateProj: Linear

    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    let rope: RoPELayer

    init(_ args: LFM3Configuration, layerType: LFM3Configuration.LayerType) {
        self.attentionHeads = args.attentionHeads
        self.kvHeads = args.kvHeads
        self.headDim = args.headDim
        self.scale = pow(Float(headDim), -0.5)

        let dim = args.hiddenSize
        _qProj.wrappedValue = Linear(dim, attentionHeads * headDim, bias: false)
        _kProj.wrappedValue = Linear(dim, kvHeads * headDim, bias: false)
        _vProj.wrappedValue = Linear(dim, kvHeads * headDim, bias: false)
        _oProj.wrappedValue = Linear(attentionHeads * headDim, dim, bias: false)
        _gateProj.wrappedValue = Linear(dim, attentionHeads, bias: false)

        _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.normEps)
        _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.normEps)

        // Partial RoPE rotates the first dimensions of each half of the head with the
        // full-head frequencies, which is what ProportionalRoPE does. RoPE(dimensions:)
        // would rotate a contiguous prefix with different frequencies instead.
        let rope = args.ropeParameters[layerType.rawValue]!
        if rope.partialRotaryFactor < 1 {
            self.rope = initializeRope(
                dims: headDim, base: rope.ropeTheta, traditional: false,
                scalingConfig: [
                    "type": .string("proportional"),
                    "partial_rotary_factor": .float(rope.partialRotaryFactor),
                ],
                maxPositionEmbeddings: nil)
        } else {
            self.rope = RoPE(dimensions: headDim, traditional: false, base: rope.ropeTheta)
        }
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        var queries = qNorm(qProj(x).reshaped(B, L, attentionHeads, -1)).transposed(0, 2, 1, 3)
        var keys = kNorm(kProj(x).reshaped(B, L, kvHeads, -1)).transposed(0, 2, 1, 3)
        let values = vProj(x).reshaped(B, L, kvHeads, -1).transposed(0, 2, 1, 3)

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

        // One sigmoid gate per head, computed from the block input.
        let gate = sigmoid(gateProj(x)).expandedDimensions(axis: -1)
        return oProj((output * gate).reshaped(B, L, -1))
    }
}

class LFM3ShortConv: Module {
    let hiddenSize: Int
    let lCache: Int

    @ModuleInfo(key: "conv") var conv: Conv1d
    @ModuleInfo(key: "in_proj") var inProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear

    init(_ args: LFM3Configuration) {
        self.hiddenSize = args.hiddenSize
        self.lCache = args.convLCache

        _conv.wrappedValue = Conv1d(
            inputChannels: hiddenSize,
            outputChannels: hiddenSize,
            kernelSize: lCache,
            groups: hiddenSize,
            bias: args.convBias
        )
        _inProj.wrappedValue = Linear(hiddenSize, 3 * hiddenSize, bias: args.convBias)
        _outProj.wrappedValue = Linear(hiddenSize, hiddenSize, bias: args.convBias)
    }

    func callAsFunction(_ x: MLXArray, cache: MambaCache?) -> MLXArray {
        let BCx = inProj(x).split(parts: 3, axis: -1)
        let (B, C) = (BCx[0], BCx[1])
        var Bx = B * BCx[2]

        // The cache holds the last `lCache - 1` inputs, so the conv needs no padding.
        let state =
            cache?[0] ?? MLXArray.zeros([Bx.dim(0), lCache - 1, hiddenSize], dtype: Bx.dtype)
        Bx = concatenated([state, Bx], axis: -2)
        if let cache {
            cache[0] = contiguous(Bx[0..., (Bx.dim(1) - (lCache - 1))..., 0...])
            cache.advance(x.dim(1))
        }

        return outProj(C * conv(Bx))
    }
}

class LFM3MLP: Module, UnaryLayer {
    @ModuleInfo(key: "w1") var w1: Linear
    @ModuleInfo(key: "w2") var w2: Linear
    @ModuleInfo(key: "w3") var w3: Linear

    init(dim: Int, hiddenDim: Int) {
        _w1.wrappedValue = Linear(dim, hiddenDim, bias: false)
        _w2.wrappedValue = Linear(hiddenDim, dim, bias: false)
        _w3.wrappedValue = Linear(dim, hiddenDim, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        w2(silu(w1(x)) * w3(x))
    }
}

class LFM3DecoderLayer: Module {
    let layerType: LFM3Configuration.LayerType

    @ModuleInfo(key: "self_attn") var attention: LFM3Attention?
    @ModuleInfo(key: "conv") var conv: LFM3ShortConv?
    @ModuleInfo(key: "feed_forward") var feedForward: LFM3MLP
    @ModuleInfo(key: "operator_norm") var operatorNorm: RMSNorm
    @ModuleInfo(key: "ffn_norm") var ffnNorm: RMSNorm
    @ModuleInfo(key: "post_operator_norm") var postOperatorNorm: RMSNorm?
    @ModuleInfo(key: "post_ffn_norm") var postFFNNorm: RMSNorm?

    init(_ args: LFM3Configuration, layerIdx: Int) {
        self.layerType = args.layerTypes[layerIdx]

        if layerType == .conv {
            _conv.wrappedValue = LFM3ShortConv(args)
        } else {
            _attention.wrappedValue = LFM3Attention(args, layerType: layerType)
        }

        _feedForward.wrappedValue = LFM3MLP(dim: args.hiddenSize, hiddenDim: args.intermediateSize)
        _operatorNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.normEps)
        _ffnNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.normEps)
        if args.useSandwichNorm {
            _postOperatorNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.normEps)
            _postFFNNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.normEps)
        }
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        var r: MLXArray
        if let attention {
            r = attention(operatorNorm(x), mask: mask, cache: cache)
        } else {
            r = conv!(operatorNorm(x), cache: cache as? MambaCache)
        }
        if let postOperatorNorm {
            r = postOperatorNorm(r)
        }
        let h = x + r

        var f = feedForward(ffnNorm(h))
        if let postFFNNorm {
            f = postFFNNorm(f)
        }
        return h + f
    }
}

public class LFM3ModelInner: Module {
    let args: LFM3Configuration
    let firstFullAttentionIdx: Int?
    let firstSlidingAttentionIdx: Int?

    let layers: [LFM3DecoderLayer]

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "embedding_norm") var embeddingNorm: RMSNorm

    public init(_ args: LFM3Configuration) {
        self.args = args
        self.firstFullAttentionIdx = args.layerTypes.firstIndex(of: .fullAttention)
        self.firstSlidingAttentionIdx = args.layerTypes.firstIndex(of: .slidingAttention)

        _embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)

        self.layers = (0 ..< args.hiddenLayers).map { i in
            LFM3DecoderLayer(args, layerIdx: i)
        }

        _embeddingNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.normEps)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var h = embedTokens(inputs)
        let n = h.dim(1)

        // One mask per attention type, sized by the first cache of that type.
        let fullMask = firstFullAttentionIdx.map {
            makeAttentionMask(n: n, cache: cache?[$0])
        }
        let slidingMask = firstSlidingAttentionIdx.map {
            makeAttentionMask(n: n, cache: cache?[$0], windowSize: args.slidingWindow)
        }

        for (i, layer) in layers.enumerated() {
            let mask: MLXFast.ScaledDotProductAttentionMaskMode =
                switch layer.layerType {
                case .fullAttention: fullMask!
                case .slidingAttention: slidingMask!
                case .conv: .none
                }
            h = layer(h, mask: mask, cache: cache?[i])
        }

        return embeddingNorm(h)
    }
}

public class LFM3Model: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    public let model: LFM3ModelInner
    let configuration: LFM3Configuration

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ args: LFM3Configuration) {
        self.configuration = args
        self.vocabularySize = args.vocabularySize
        self.kvHeads = args.layerTypes.map { $0 == .conv ? 0 : args.kvHeads }
        self.model = LFM3ModelInner(args)
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

    public func prepareCheckpoint(_ checkpoint: ModelCheckpoint) throws -> ModelCheckpoint {
        // MLX conversions nest the text model under `language_model`. Keep only the text
        // model, which drops any vision or audio tensors.
        var checkpoint = try checkpoint.mapNames { name in
            var name = name
            if name.hasPrefix("language_model.") {
                name.removeFirst("language_model.".count)
            }
            return name.hasPrefix("model.") || name.hasPrefix("lm_head.") ? name : nil
        }
        var weights = filterLMHeadWeights(
            from: checkpoint.weights, tiedWordEmbeddings: configuration.tieWordEmbeddings)

        // PyTorch stores the depthwise conv as [C, 1, K]; MLX expects [C, K, 1].
        for (name, value) in weights where name.hasSuffix("conv.conv.weight") {
            if value.dim(-1) > value.dim(1) {
                weights[name] = value.transposed(0, 2, 1)
            }
        }
        checkpoint.weights = weights
        return checkpoint
    }

    public func sanitize(weights: [String: MLXArray]) throws -> [String: MLXArray] {
        try prepareCheckpoint(.init(weights: weights)).weights
    }

    public func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
        try configuration.layerTypes.map { layerType in
            switch layerType {
            case .conv:
                MambaCache()
            case .fullAttention, .slidingAttention:
                try makeHybridAttentionKVCache(
                    parameters: parameters,
                    slidingWindow: configuration.slidingWindow,
                    usesSlidingWindow: layerType == .slidingAttention)
            }
        }
    }
}

extension LFM3Model: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}
