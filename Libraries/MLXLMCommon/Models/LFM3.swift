//
//  LFM3.swift
//  mlx-swift-lm
//

// Port of https://huggingface.co/LiquidAI/LFM3-1.7B/blob/main/modeling_lfm3.py
//
// LFM3 is LFM2 with sliding-window attention as a third layer type, a sigmoid
// gate per attention head, partial RoPE on full-attention layers, and an
// optional RMSNorm on each block output ("sandwich norm").
//
// The configuration and backbone live here so that MLXLLM's `LFM3Model` and a future
// MLXVLM wrapper share one implementation.

import Foundation
import MLX
import MLXNN

public struct LFM3Configuration: Codable, Sendable {
    public enum LayerType: String, Codable, Sendable {
        case conv
        case fullAttention = "full_attention"
        case slidingAttention = "sliding_attention"
    }

    package struct RopeParameters: Codable, Sendable {
        package let ropeType: String
        package let ropeTheta: Float
        package let partialRotaryFactor: Float

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

        package init(from decoder: Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            self.ropeType =
                try container.decodeIfPresent(String.self, forKey: .ropeType) ?? "default"
            self.ropeTheta = try container.decode(Float.self, forKey: .ropeTheta)
            self.partialRotaryFactor =
                try container.decodeIfPresent(Float.self, forKey: .partialRotaryFactor) ?? 1.0
        }
    }

    package let modelType: String
    public let vocabularySize: Int
    public let hiddenSize: Int
    public let hiddenLayers: Int
    public let attentionHeads: Int
    public let kvHeads: Int
    public let headDim: Int
    public let intermediateSize: Int
    package let maxPositionEmbeddings: Int?
    package let normEps: Float
    package let convBias: Bool
    package let convLCache: Int
    public let layerTypes: [LayerType]
    public let slidingWindow: Int
    package let useSandwichNorm: Bool
    public let tieWordEmbeddings: Bool
    package let ropeParameters: [String: RopeParameters]

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

    /// The reference reads the LFM2 spelling `tie_embedding` before `tie_word_embeddings`.
    private enum AltCodingKeys: String, CodingKey {
        case tieEmbedding = "tie_embedding"
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
        for (key, value) in [
            (CodingKeys.hiddenSize, hiddenSize), (.attentionHeads, attentionHeads),
            (.kvHeads, kvHeads),
        ] where value <= 0 {
            throw DecodingError.dataCorruptedError(
                forKey: key, in: container, debugDescription: "\(key.rawValue) must be positive")
        }
        guard attentionHeads % kvHeads == 0 else {
            throw DecodingError.dataCorruptedError(
                forKey: .kvHeads, in: container,
                debugDescription: "num_attention_heads must be divisible by num_key_value_heads")
        }
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
        let alt = try decoder.container(keyedBy: AltCodingKeys.self)
        self.tieWordEmbeddings =
            try alt.decodeIfPresent(Bool.self, forKey: .tieEmbedding)
            ?? container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? true

        for (key, value) in [
            (CodingKeys.vocabularySize, vocabularySize), (.hiddenLayers, hiddenLayers),
            (.intermediateSize, intermediateSize), (.convLCache, convLCache),
            (.slidingWindow, slidingWindow),
        ] where value <= 0 {
            throw DecodingError.dataCorruptedError(
                forKey: key, in: container, debugDescription: "\(key.rawValue) must be positive")
        }
        guard headDim > 0, headDim % 2 == 0 else {
            throw DecodingError.dataCorruptedError(
                forKey: .headDim, in: container,
                debugDescription: "LFM3 head_dim must be positive and even")
        }
        guard normEps.isFinite, normEps > 0 else {
            throw DecodingError.dataCorruptedError(
                forKey: .normEps, in: container,
                debugDescription: "norm_eps must be finite and positive")
        }
        if let maxPositionEmbeddings, maxPositionEmbeddings <= 0 {
            throw DecodingError.dataCorruptedError(
                forKey: .maxPositionEmbeddings, in: container,
                debugDescription: "max_position_embeddings must be positive")
        }

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
            guard rope.ropeTheta.isFinite, rope.ropeTheta > 0,
                rope.partialRotaryFactor.isFinite, (0 ... 1).contains(rope.partialRotaryFactor)
            else {
                throw DecodingError.dataCorruptedError(
                    forKey: .ropeParameters, in: container,
                    debugDescription:
                        "LFM3 requires finite positive rope_theta and partial_rotary_factor in [0, 1]"
                )
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
        if let mask = cache?.makeMask(N: x.dim(1)) {
            Bx = which(mask[.ellipsis, .newAxis], Bx, 0)
        }

        // The cache holds the last `lCache - 1` inputs, so the conv needs no padding.
        let state =
            cache?[0] ?? MLXArray.zeros([Bx.dim(0), lCache - 1, hiddenSize], dtype: Bx.dtype)
        Bx = concatenated([state, Bx], axis: -2)
        if let cache {
            if let lengths = cache.currentLengths {
                let ends = clip(lengths, min: 0, max: x.dim(1))
                let positions = (ends[0..., .newAxis] + MLXArray(0 ..< lCache - 1))[
                    .ellipsis, .newAxis]
                cache[0] = contiguous(takeAlong(Bx, positions, axis: 1))
            } else {
                cache[0] = contiguous(Bx[0..., (Bx.dim(1) - (lCache - 1))..., 0...])
            }
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

package class LFM3DecoderLayer: Module {
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
            precondition(!(cache is ArraysCache), "LFM3 attention requires an attention cache")
            r = attention(operatorNorm(x), mask: mask, cache: cache)
        } else {
            precondition(
                cache == nil || cache is MambaCache, "LFM3 convolution requires MambaCache")
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

/// Attention masks for one backbone pass, one per attention layer type.
///
/// A `nil` entry uses the default: causal for full attention, and causal within the
/// sliding window for sliding attention, both sized by the first cache of that type.
/// For no mask, write `MLXFast.ScaledDotProductAttentionMaskMode.none`: a bare `.none` is `nil`.
/// An `.array` mask must match the keys the cache returns. A sliding cache drops keys outside
/// the window, so build that mask with `makeAttentionMask(n:cache:windowSize:returnArray:)`.
package struct LFM3AttentionMasks {
    package var fullAttention: MLXFast.ScaledDotProductAttentionMaskMode?
    package var slidingAttention: MLXFast.ScaledDotProductAttentionMaskMode?

    package init(
        fullAttention: MLXFast.ScaledDotProductAttentionMaskMode? = nil,
        slidingAttention: MLXFast.ScaledDotProductAttentionMaskMode? = nil
    ) {
        self.fullAttention = fullAttention
        self.slidingAttention = slidingAttention
    }
}

/// The LFM3 text backbone: token embedding, decoder layers, and the final norm.
///
/// The caller owns the caches, one per layer from `newCache(parameters:)`. Positions come
/// from the caches, so a sequence can be fed in chunks of token ids or embeddings.
public class LFM3ModelInner: Module {
    let args: LFM3Configuration
    package let layers: [LFM3DecoderLayer]
    @ModuleInfo(key: "embed_tokens") package var embedTokens: Embedding
    @ModuleInfo(key: "embedding_norm") var embeddingNorm: RMSNorm
    package var hiddenSize: Int { args.hiddenSize }

    package init(_ args: LFM3Configuration) {
        self.args = args
        let backbone = LFM3EmbeddingModelInner(args)
        layers = backbone.layers
        _embeddingNorm.wrappedValue = backbone.embeddingNorm
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)
    }

    /// Token embeddings `[B, L, hidden]` for token ids `[B, L]`.
    package func embed(_ inputs: MLXArray) -> MLXArray { embedTokens(inputs) }

    /// Final normed hidden states `[B, L, hidden]` for token ids `[B, L]`.
    package func callAsFunction(
        _ inputs: MLXArray, cache: [KVCache]? = nil, masks: LFM3AttentionMasks = .init()
    ) -> MLXArray {
        callAsFunction(embeddings: embed(inputs), cache: cache, masks: masks)
    }

    /// Final normed hidden states `[B, L, hidden]` for input embeddings `[B, L, hidden]`.
    package func callAsFunction(
        embeddings: MLXArray, cache: [KVCache]? = nil, masks: LFM3AttentionMasks = .init()
    ) -> MLXArray {
        lfm3HiddenStates(
            embeddings, args: args, layers: layers, norm: embeddingNorm, cache: cache, masks: masks)
    }

    /// Vocabulary logits for final hidden states, through `lmHead` when the checkpoint has
    /// one and the tied token embedding otherwise.
    package func projectLogits(_ hiddenStates: MLXArray, lmHead: Linear? = nil) -> MLXArray {
        lmHead?(hiddenStates) ?? embedTokens.asLinear(hiddenStates)
    }

    /// One cache per layer: a conv state for conv layers, a rotating cache bounded by the
    /// sliding window for sliding attention, and a full or `maxKVSize` cache otherwise.
    package func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
        try lfm3Cache(args, parameters: parameters)
    }
}

/// An embedding-only backbone for speech modules. It has no token or vocabulary projection API.
package final class LFM3EmbeddingModelInner: Module {
    let args: LFM3Configuration
    package let layers: [LFM3DecoderLayer]
    @ModuleInfo(key: "embedding_norm") var embeddingNorm: RMSNorm

    package init(_ args: LFM3Configuration) {
        self.args = args
        layers = (0 ..< args.hiddenLayers).map { LFM3DecoderLayer(args, layerIdx: $0) }
        _embeddingNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.normEps)
    }

    package func callAsFunction(
        embeddings: MLXArray, cache: [KVCache]? = nil, masks: LFM3AttentionMasks = .init()
    ) -> MLXArray {
        lfm3HiddenStates(
            embeddings, args: args, layers: layers, norm: embeddingNorm, cache: cache, masks: masks)
    }

    package func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
        try lfm3Cache(args, parameters: parameters)
    }
}

private func lfm3HiddenStates(
    _ embeddings: MLXArray, args: LFM3Configuration, layers: [LFM3DecoderLayer],
    norm: RMSNorm, cache: [KVCache]?, masks: LFM3AttentionMasks
) -> MLXArray {
    precondition(
        cache == nil || cache?.count == layers.count, "LFM3 requires one cache per layer")
    precondition(
        embeddings.ndim == 3 && embeddings.dim(-1) == args.hiddenSize && embeddings.dim(1) > 0,
        "LFM3 expects nonempty embeddings shaped [batch, sequence, hidden_size]")
    var h = embeddings
    let n = h.dim(1)

    let fullMask =
        masks.fullAttention
        ?? args.layerTypes.firstIndex(of: .fullAttention).map {
            makeAttentionMask(n: n, cache: cache?[$0])
        }
    let slidingMask =
        masks.slidingAttention
        ?? args.layerTypes.firstIndex(of: .slidingAttention).map {
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

    return norm(h)
}

private func lfm3Cache(_ args: LFM3Configuration, parameters: GenerateParameters?) throws
    -> [KVCache]
{
    try args.layerTypes.map { layerType in
        switch layerType {
        case .conv:
            MambaCache()
        case .fullAttention, .slidingAttention:
            try makeHybridAttentionKVCache(
                parameters: parameters,
                slidingWindow: args.slidingWindow,
                usesSlidingWindow: layerType == .slidingAttention)
        }
    }
}

extension LFM3ModelInner {
    package static func validate(tokenizer: any Tokenizer) throws {
        guard tokenizer.convertTokenToId("<|startoftext|>") == 124894,
            tokenizer.convertTokenToId(".reduce") == 96744
        else { return }
        // Probe vocabulary entries affected by ignore_merges and code-point splitting.
        for (text, pieces) in [
            (".reduce", [".reduce"]), ("line.\r\nThis", ["line", ".čĊ", "This"]),
        ] {
            let expected = pieces.compactMap { tokenizer.convertTokenToId($0) }
            guard expected.count == pieces.count,
                !expected.contains(where: { $0 == tokenizer.unknownTokenId })
            else { continue }
            guard tokenizer.encode(text: text, addSpecialTokens: false) == expected else {
                throw LFM3TokenizerError.incompatible
            }
        }
    }
}

private enum LFM3TokenizerError: LocalizedError {
    case incompatible

    var errorDescription: String? {
        "LFM3 requires a tokenizer that honors BPE ignore_merges and splits regexes by code point. "
            + "Use a tokenizer containing swift-transformers fixes #397 and #398, or a compatible TokenizerLoader."
    }
}
