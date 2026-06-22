// Copyright © 2026 Apple Inc.

// MLX Swift port of LiquidAI's LFM2.5 *bidirectional* (encoder) backbone + retrieval heads.
//
// Powers two on-device retrieval models that share the LFM2.5-350M hybrid backbone
// (short-conv + GQA attention, SwiGLU MLP, RMSNorm):
//   - LFM2.5-Embedding-350M  — CLS pooling -> 1024-d sentence vector (cosine).
//   - LFM2.5-ColBERT-350M     — per-token Dense 1024->128 projection (MaxSim).
//
// Encoder patches relative to the *causal* generative `LFM2.swift` in MLXLLM:
//   1. attention is bidirectional (additive pad-only mask, no causal mask),
//   2. the short conv is non-causal / centered (symmetric padding = kernel/2),
//   3. no LM head; a pooling/projection head is used instead.
//
// Both models share `model_type: "lfm2"` and differ only by the custom `"mlx"`
// block in config.json (`head: "embedding"` vs `"colbert"`). A single model class
// branches on that head. Ported from Models/mlx/lfm2_bidirectional.py and the
// standalone Models/swift reference; kept self-contained to match this module's
// convention (Qwen3/Gemma3 are reimplemented here rather than imported from MLXLLM).

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Configuration

public struct LFM2BidirectionalConfiguration: Decodable, Sendable {
    public let vocabSize: Int
    public let hiddenSize: Int
    public let numHiddenLayers: Int
    public let numAttentionHeads: Int
    public let numKeyValueHeads: Int
    public let normEps: Float
    public let convBias: Bool
    public let convLCache: Int
    public let blockFFDim: Int
    public let blockMultipleOf: Int
    public let blockFFNDimMultiplier: Float
    public let blockAutoAdjustFFDim: Bool
    public let ropeTheta: Float
    public let layerTypes: [String]
    public let maxPositionEmbeddings: Int?
    public let mlx: MLXHead

    /// The custom `"mlx"` block that selects the retrieval head and carries the
    /// prompt/prefix conventions used by callers (the library does not inject them).
    public struct MLXHead: Codable, Sendable {
        public enum Kind: String, Codable, Sendable {
            case embedding
            case colbert
        }
        public let head: Kind
        // embedding
        public let pooling: String?
        public let prompts: [String: String]?
        // colbert
        public let projDim: Int?
        public let queryPrefix: String?
        public let documentPrefix: String?
        public let queryLength: Int?
        public let documentLength: Int?

        enum CodingKeys: String, CodingKey {
            case head, pooling, prompts
            case projDim = "proj_dim"
            case queryPrefix = "query_prefix"
            case documentPrefix = "document_prefix"
            case queryLength = "query_length"
            case documentLength = "document_length"
        }
    }

    public var headDim: Int { hiddenSize / numAttentionHeads }

    /// Indices of the `full_attention` layers (the rest are short-conv layers).
    public var attnLayerIdxs: [Int] {
        layerTypes.enumerated().compactMap { $0.element == "full_attention" ? $0.offset : nil }
    }

    private struct RopeParameters: Codable {
        let ropeTheta: Float?
        enum CodingKeys: String, CodingKey { case ropeTheta = "rope_theta" }
    }

    enum CodingKeys: String, CodingKey {
        case vocabSize = "vocab_size"
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case normEps = "norm_eps"
        case blockNormEps = "block_norm_eps"
        case convBias = "conv_bias"
        case convLCache = "conv_L_cache"
        case blockFFDim = "block_ff_dim"
        case intermediateSize = "intermediate_size"
        case blockMultipleOf = "block_multiple_of"
        case blockFFNDimMultiplier = "block_ffn_dim_multiplier"
        case blockAutoAdjustFFDim = "block_auto_adjust_ff_dim"
        case ropeTheta = "rope_theta"
        case ropeParameters = "rope_parameters"
        case layerTypes = "layer_types"
        case maxPositionEmbeddings = "max_position_embeddings"
        case mlx
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        vocabSize = try c.decode(Int.self, forKey: .vocabSize)
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        numHiddenLayers = try c.decode(Int.self, forKey: .numHiddenLayers)
        numAttentionHeads = try c.decode(Int.self, forKey: .numAttentionHeads)
        numKeyValueHeads =
            try c.decodeIfPresent(Int.self, forKey: .numKeyValueHeads) ?? numAttentionHeads
        normEps =
            try c.decodeIfPresent(Float.self, forKey: .normEps)
            ?? c.decodeIfPresent(Float.self, forKey: .blockNormEps) ?? 1e-5
        convBias = try c.decodeIfPresent(Bool.self, forKey: .convBias) ?? false
        convLCache = try c.decodeIfPresent(Int.self, forKey: .convLCache) ?? 3
        blockFFDim =
            try c.decodeIfPresent(Int.self, forKey: .blockFFDim)
            ?? c.decode(Int.self, forKey: .intermediateSize)
        blockMultipleOf = try c.decodeIfPresent(Int.self, forKey: .blockMultipleOf) ?? 256
        blockFFNDimMultiplier =
            try c.decodeIfPresent(Float.self, forKey: .blockFFNDimMultiplier) ?? 1.0
        blockAutoAdjustFFDim =
            try c.decodeIfPresent(Bool.self, forKey: .blockAutoAdjustFFDim) ?? true
        layerTypes = try c.decode([String].self, forKey: .layerTypes)
        maxPositionEmbeddings = try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings)

        let theta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta)
        let ropeParameters = try c.decodeIfPresent(RopeParameters.self, forKey: .ropeParameters)
        ropeTheta = theta ?? ropeParameters?.ropeTheta ?? 1_000_000.0

        // The custom `mlx` block is present on the converted checkpoints. If a config
        // omits it (e.g. a raw SentenceTransformer LFM2.5 export), default to a
        // CLS-pooled embedding head — a plain encoder with no projection — rather than
        // throwing. ColBERT still requires the block, since the projection head can't
        // be inferred from config.json alone.
        mlx =
            try c.decodeIfPresent(MLXHead.self, forKey: .mlx)
            ?? MLXHead(
                head: .embedding, pooling: "cls", prompts: nil, projDim: nil,
                queryPrefix: nil, documentPrefix: nil, queryLength: nil, documentLength: nil)
    }
}

// MARK: - Blocks

/// Grouped-query attention with per-head q/k RMSNorm and RoPE. Bidirectional
/// (the only mask is an optional additive pad mask passed by the backbone).
private final class LFM2BiAttention: Module {
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear
    @ModuleInfo(key: "q_layernorm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_layernorm") var kNorm: RMSNorm

    let rope: RoPE
    let nHeads: Int
    let nKVHeads: Int
    let scale: Float

    init(_ c: LFM2BidirectionalConfiguration) {
        let dim = c.hiddenSize
        let hd = c.headDim
        nHeads = c.numAttentionHeads
        nKVHeads = c.numKeyValueHeads
        scale = pow(Float(hd), -0.5)

        _qProj.wrappedValue = Linear(dim, nHeads * hd, bias: false)
        _kProj.wrappedValue = Linear(dim, nKVHeads * hd, bias: false)
        _vProj.wrappedValue = Linear(dim, nKVHeads * hd, bias: false)
        _outProj.wrappedValue = Linear(nHeads * hd, dim, bias: false)
        _qNorm.wrappedValue = RMSNorm(dimensions: hd, eps: c.normEps)
        _kNorm.wrappedValue = RMSNorm(dimensions: hd, eps: c.normEps)
        rope = RoPE(dimensions: hd, traditional: false, base: c.ropeTheta)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))
        var q = qNorm(qProj(x).reshaped(B, L, nHeads, -1)).transposed(0, 2, 1, 3)
        var k = kNorm(kProj(x).reshaped(B, L, nKVHeads, -1)).transposed(0, 2, 1, 3)
        let v = vProj(x).reshaped(B, L, nKVHeads, -1).transposed(0, 2, 1, 3)
        q = rope(q)
        k = rope(k)
        let o = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: scale, mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)
        return outProj(o)
    }
}

/// Non-causal gated short convolution (centered, symmetric padding). Padded
/// positions are zeroed via the `keep` mask so they cannot leak through the kernel.
private final class LFM2BiShortConv: Module {
    @ModuleInfo(key: "conv") var conv: Conv1d
    @ModuleInfo(key: "in_proj") var inProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear

    init(_ c: LFM2BidirectionalConfiguration) {
        _conv.wrappedValue = Conv1d(
            inputChannels: c.hiddenSize, outputChannels: c.hiddenSize,
            kernelSize: c.convLCache, padding: c.convLCache / 2,
            groups: c.hiddenSize, bias: c.convBias)
        _inProj.wrappedValue = Linear(c.hiddenSize, 3 * c.hiddenSize, bias: c.convBias)
        _outProj.wrappedValue = Linear(c.hiddenSize, c.hiddenSize, bias: c.convBias)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, keep: MLXArray?) -> MLXArray {
        let s = inProj(x).split(parts: 3, axis: -1)  // [B, C, x]
        var Bx = s[0] * s[2]
        if let keep {
            Bx = Bx * keep.expandedDimensions(axis: -1)
        }
        var convOut = conv(Bx)
        if convOut.dim(1) != Bx.dim(1) {
            convOut = convOut[0..., 0 ..< Bx.dim(1), 0...]
        }
        return outProj(s[1] * convOut)
    }
}

/// SwiGLU MLP with the LFM2 `block_auto_adjust_ff_dim` width adjustment.
private final class LFM2BiMLP: Module, UnaryLayer {
    @ModuleInfo(key: "w1") var w1: Linear
    @ModuleInfo(key: "w2") var w2: Linear
    @ModuleInfo(key: "w3") var w3: Linear

    init(_ c: LFM2BidirectionalConfiguration) {
        var ff = c.blockFFDim
        if c.blockAutoAdjustFFDim {
            ff = Int(Float(2 * ff) / 3.0)
            ff = Int(c.blockFFNDimMultiplier * Float(ff))
            let m = c.blockMultipleOf
            ff = m * ((ff + m - 1) / m)
        }
        _w1.wrappedValue = Linear(c.hiddenSize, ff, bias: false)
        _w2.wrappedValue = Linear(ff, c.hiddenSize, bias: false)
        _w3.wrappedValue = Linear(c.hiddenSize, ff, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { w2(silu(w1(x)) * w3(x)) }
}

/// One hybrid block: `operator_norm` -> (attention | short-conv) -> residual,
/// then `ffn_norm` -> SwiGLU -> residual.
private final class LFM2BiDecoderLayer: Module {
    let isAttention: Bool
    @ModuleInfo(key: "self_attn") var attn: LFM2BiAttention?
    @ModuleInfo(key: "conv") var conv: LFM2BiShortConv?
    @ModuleInfo(key: "feed_forward") var feedForward: LFM2BiMLP
    @ModuleInfo(key: "operator_norm") var operatorNorm: RMSNorm
    @ModuleInfo(key: "ffn_norm") var ffnNorm: RMSNorm

    init(_ c: LFM2BidirectionalConfiguration, idx: Int) {
        isAttention = c.attnLayerIdxs.contains(idx)
        if isAttention {
            _attn.wrappedValue = LFM2BiAttention(c)
        } else {
            _conv.wrappedValue = LFM2BiShortConv(c)
        }
        _feedForward.wrappedValue = LFM2BiMLP(c)
        _operatorNorm.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: c.normEps)
        _ffnNorm.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: c.normEps)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, keep: MLXArray?
    ) -> MLXArray {
        let r =
            isAttention ? attn!(operatorNorm(x), mask: mask) : conv!(operatorNorm(x), keep: keep)
        let h = x + r
        return h + feedForward(ffnNorm(h))
    }
}

// MARK: - Backbone

/// Token ids `(B, L)` -> last hidden state `(B, L, hidden)` after `embedding_norm`.
/// Builds the bidirectional additive pad mask and the conv keep-mask from
/// `attentionMask` (1 = real token, 0 = padding); when `attentionMask` is nil the
/// sequence is assumed unpadded (single-sequence encoding).
private final class LFM2BiBackbone: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "embedding_norm") var embeddingNorm: RMSNorm
    fileprivate let layers: [LFM2BiDecoderLayer]

    init(_ c: LFM2BidirectionalConfiguration) {
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: c.vocabSize, dimensions: c.hiddenSize)
        _embeddingNorm.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: c.normEps)
        layers = (0 ..< c.numHiddenLayers).map { LFM2BiDecoderLayer(c, idx: $0) }
        super.init()
    }

    func callAsFunction(_ ids: MLXArray, attentionMask: MLXArray?) -> MLXArray {
        var h = embedTokens(ids)

        var maskMode: MLXFast.ScaledDotProductAttentionMaskMode = .none
        var keep: MLXArray? = nil
        if let attentionMask {
            // conv keep-mask: (B, L) float, 1 = real, 0 = pad
            keep = attentionMask.asType(h.dtype)
            // additive attention mask: (B, 1, 1, L), 0 where real, -inf where pad.
            // log(1) = 0, log(0) = -inf — matches the Bert idiom in this module.
            let additive = attentionMask.asType(h.dtype).expandedDimensions(axes: [1, 2]).log()
            maskMode = .array(additive)
        }

        for layer in layers {
            h = layer(h, mask: maskMode, keep: keep)
        }
        return embeddingNorm(h)
    }
}

// MARK: - Model

/// LFM2.5 bidirectional encoder for retrieval. One class serves both heads,
/// selected by the `"mlx"` block in config.json:
///
/// - `embedding`: returns the backbone hidden states `(B, L, hidden)`; callers pool
///   with `.cls` + `normalize` to get a 1024-d sentence vector (cosine similarity).
/// - `colbert`: applies a per-token Dense projection -> `(B, L, projDim)`; callers
///   pool with `.none` + `normalize` to get per-token multi-vectors (MaxSim).
public final class LFM2BidirectionalModel: Module, EmbeddingModel {
    @ModuleInfo(key: "model") private var model: LFM2BiBackbone
    @ModuleInfo(key: "dense") private var dense: Linear?

    public let configuration: LFM2BidirectionalConfiguration
    private let head: LFM2BidirectionalConfiguration.MLXHead.Kind

    public var vocabularySize: Int { configuration.vocabSize }

    public var poolingStrategy: Pooling.Strategy? {
        head == .colbert ? Pooling.Strategy.none : Pooling.Strategy.cls
    }

    public init(_ c: LFM2BidirectionalConfiguration) {
        self.configuration = c
        self.head = c.mlx.head
        self._model.wrappedValue = LFM2BiBackbone(c)
        if c.mlx.head == .colbert {
            self._dense.wrappedValue = Linear(c.hiddenSize, c.mlx.projDim ?? 128, bias: false)
        }
        super.init()
    }

    public func callAsFunction(
        _ inputs: MLXArray,
        positionIds: MLXArray? = nil,
        tokenTypeIds: MLXArray? = nil,
        attentionMask: MLXArray? = nil
    ) -> EmbeddingModelOutput {
        var inp = inputs
        if inp.ndim == 1 {
            inp = inp.reshaped(1, -1)
        }
        // Reshape a 1-D mask to match the 1-D input reshape above, so the additive
        // mask the backbone builds broadcasts cleanly over (B, heads, L, L) in SDPA.
        var mask = attentionMask
        if let m = mask, m.ndim == 1 {
            mask = m.reshaped(1, -1)
        }
        let lhs = model(inp, attentionMask: mask)  // (B, L, hidden)

        switch head {
        case .embedding:
            // CLS == BOS at position 0; caller pools .cls + normalize.
            return EmbeddingModelOutput(hiddenStates: lhs, pooledOutput: nil)
        case .colbert:
            // Per-token projection; caller pools .none + normalize (per-token L2), then MaxSim.
            var tok = dense!(lhs)
            // Zero padded positions so they don't pollute MaxSim for batched callers
            // (matches the reference's `tok * attention_mask`). `l2Normalized` keeps
            // zeros as zeros, so a later `.none` + normalize remains correct.
            if let mask {
                tok = tok * mask.asType(tok.dtype).expandedDimensions(axis: -1)
            }
            return EmbeddingModelOutput(hiddenStates: tok, pooledOutput: nil)
        }
    }

    /// Weights from `convert.py` are already `model.`-prefixed (and the ColBERT head
    /// is the bare key `dense.weight`), so this only applies the depthwise Conv1d
    /// transpose guard — `(O, 1, K)` -> `(O, K, 1)` — for robustness against weights
    /// re-converted straight from Hugging Face. It must NOT blanket-prepend `model.`
    /// (that would corrupt `dense.weight`).
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized = [String: MLXArray]()
        for (name, param) in weights {
            var p = param
            if name.contains("conv.weight") {
                if param.shape[param.shape.count - 1] > param.dim(1) {
                    p = param.transposed(0, 2, 1)
                }
            }
            sanitized[name] = p
        }
        return sanitized
    }
}
