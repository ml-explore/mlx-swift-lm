// Copyright © 2026 Apple Inc.

import CoreImage
import Foundation
import MLX
import MLXLMCommon
import MLXNN

// Port of https://github.com/Blaizzy/mlx-vlm/tree/main/mlx_vlm/models/muse_glimmer
//
// Muse-Glimmer pairs a 52-layer dense text transformer with a 50-layer
// native-resolution ViT. Several pieces are non-standard; each is called out at
// its definition, but the ones most likely to bite a future reader are:
//
//   - `CenteredRMSNorm` scales by `1 + weight` (the checkpoint stores a scale
//     centered at zero), and the block norms alternate between two eps values.
//   - Full-attention layers get *no* positional encoding (`layer_rope_theta` is
//     0 for them), while sliding layers use theta 500000.
//   - Attention output is gated: `output * sigmoid(gate_proj(x))`.
//   - Queries are multiplied by `qk_scale_factor` (3.87) *in addition to* the
//     usual `1/sqrt(head_dim)` softmax scale.
//   - The ViT does window attention by permuting tokens into contiguous windows
//     and running chunked SDPA — no banded mask is ever materialized.

// MARK: - Configuration

public struct MuseGlimmerTextConfiguration: Codable, Sendable {
    public let modelType: String
    public let vocabularySize: Int
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let hiddenLayers: Int
    public let attentionHeads: Int
    public let kvHeads: Int
    public let headDimensions: Int
    public let maxPositionEmbeddings: Int
    public let rmsNormEps: Float
    public let postNormEps: Float
    public let attentionBias: Bool
    public let slidingWindow: Int
    public let qkScaleFactor: Float
    public let outputMultiplier: Float
    public let finalLogitSoftcapping: Float
    public let tieWordEmbeddings: Bool

    /// Per-layer attention kind: `sliding_attention` or `full_attention`.
    public let layerTypes: [String]

    /// Per-layer RoPE base. A value of `0` means the layer is NoPE — see
    /// `MuseGlimmerTextAttention`.
    public let layerRopeTheta: [Float]

    /// Fallback theta used only to construct the (unused) RoPE on NoPE layers.
    public var ropeTheta: Float { ropeParameters?["rope_theta"]?.asFloat() ?? 500_000.0 }

    public let ropeParameters: [String: StringOrNumber]?

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabularySize = "vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case headDimensions = "head_dim"
        case maxPositionEmbeddings = "max_position_embeddings"
        case rmsNormEps = "rms_norm_eps"
        case postNormEps = "post_norm_eps"
        case attentionBias = "attention_bias"
        case slidingWindow = "sliding_window"
        case qkScaleFactor = "qk_scale_factor"
        case outputMultiplier = "output_multiplier"
        case finalLogitSoftcapping = "final_logit_softcapping"
        case tieWordEmbeddings = "tie_word_embeddings"
        case layerTypes = "layer_types"
        case layerRopeTheta = "layer_rope_theta"
        case ropeParameters = "rope_parameters"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        self.modelType =
            try container.decodeIfPresent(String.self, forKey: .modelType) ?? "muse_glimmer_text"
        self.vocabularySize =
            try container.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 202048
        self.hiddenSize = try container.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 6656
        self.intermediateSize =
            try container.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 19968
        let hiddenLayers = try container.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 52
        self.hiddenLayers = hiddenLayers
        self.attentionHeads =
            try container.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 32
        self.kvHeads = try container.decodeIfPresent(Int.self, forKey: .kvHeads) ?? 2
        self.headDimensions =
            try container.decodeIfPresent(Int.self, forKey: .headDimensions) ?? 128
        self.maxPositionEmbeddings =
            try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 131072
        self.rmsNormEps = try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-5
        self.postNormEps = try container.decodeIfPresent(Float.self, forKey: .postNormEps) ?? 1e-8
        self.attentionBias =
            try container.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        self.slidingWindow =
            try container.decodeIfPresent(Int.self, forKey: .slidingWindow) ?? 2048
        self.qkScaleFactor =
            try container.decodeIfPresent(Float.self, forKey: .qkScaleFactor) ?? 3.87
        self.outputMultiplier =
            try container.decodeIfPresent(Float.self, forKey: .outputMultiplier)
            ?? 0.196_116_135_138_184_04
        self.finalLogitSoftcapping =
            try container.decodeIfPresent(Float.self, forKey: .finalLogitSoftcapping) ?? 20.0
        self.tieWordEmbeddings =
            try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false

        self.ropeParameters =
            try container.decodeIfPresent(
                [String: StringOrNumber].self, forKey: .ropeParameters)

        // Mirrors `TextConfig.__post_init__`: full attention every 4th layer
        // counting back from the last, so the final layer is always full.
        if let decoded = try container.decodeIfPresent([String].self, forKey: .layerTypes) {
            self.layerTypes = decoded
        } else {
            self.layerTypes = (0 ..< hiddenLayers).map { index in
                (hiddenLayers - 1 - index) % 4 == 0 ? "full_attention" : "sliding_attention"
            }
        }

        // Full-attention layers are NoPE: theta 0. Sliding layers use the base theta.
        if let decoded = try container.decodeIfPresent([Float].self, forKey: .layerRopeTheta) {
            self.layerRopeTheta = decoded
        } else {
            let baseTheta = self.ropeParameters?["rope_theta"]?.asFloat() ?? 500_000.0
            self.layerRopeTheta = self.layerTypes.map {
                $0 == "full_attention" ? 0 : baseTheta
            }
        }
    }
}

public struct MuseGlimmerVisionConfiguration: Codable, Sendable {
    public let modelType: String
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let attentionHeads: Int
    public let hiddenLayers: Int
    public let patchSize: Int
    public let patchTemporal: Int
    public let mergeSize: Int
    public let positionEmbeddingHeight: Int
    public let positionEmbeddingWidth: Int
    public let layerNormEps: Float
    public var ropeTheta: Float { ropeParameters?["rope_theta"]?.asFloat() ?? 10_000.0 }

    public let ropeParameters: [String: StringOrNumber]?

    /// Per-layer attention kind: `window_attention` or `full_attention`.
    public let layerTypes: [String]

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case hiddenLayers = "num_hidden_layers"
        case patchSize = "patch_size"
        case patchTemporal = "patch_temporal"
        case mergeSize = "merge_size"
        case positionEmbeddingHeight = "pos_emb_height"
        case positionEmbeddingWidth = "pos_emb_width"
        case layerNormEps = "layer_norm_eps"
        case layerTypes = "layer_types"
        case ropeParameters = "rope_parameters"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        self.modelType =
            try container.decodeIfPresent(String.self, forKey: .modelType) ?? "muse_glimmer_vision"
        self.hiddenSize = try container.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 1536
        self.intermediateSize =
            try container.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 8960
        self.attentionHeads =
            try container.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 16
        let hiddenLayers = try container.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 50
        self.hiddenLayers = hiddenLayers
        self.patchSize = try container.decodeIfPresent(Int.self, forKey: .patchSize) ?? 14
        self.patchTemporal = try container.decodeIfPresent(Int.self, forKey: .patchTemporal) ?? 2
        self.mergeSize = try container.decodeIfPresent(Int.self, forKey: .mergeSize) ?? 2
        self.positionEmbeddingHeight =
            try container.decodeIfPresent(Int.self, forKey: .positionEmbeddingHeight) ?? 32
        self.positionEmbeddingWidth =
            try container.decodeIfPresent(Int.self, forKey: .positionEmbeddingWidth) ?? 32
        self.layerNormEps =
            try container.decodeIfPresent(Float.self, forKey: .layerNormEps) ?? 1e-5

        self.ropeParameters =
            try container.decodeIfPresent(
                [String: StringOrNumber].self, forKey: .ropeParameters)

        // Mirrors `VisionConfig.__post_init__`: full attention every 4th layer
        // counting forward, plus the last layer.
        if let decoded = try container.decodeIfPresent([String].self, forKey: .layerTypes) {
            self.layerTypes = decoded
        } else {
            self.layerTypes = (0 ..< hiddenLayers).map { index in
                (index + 1) % 4 == 0 || index == hiddenLayers - 1
                    ? "full_attention" : "window_attention"
            }
        }
    }
}

public struct MuseGlimmerConfiguration: Codable, Sendable {
    public let textConfiguration: MuseGlimmerTextConfiguration
    public let visionConfiguration: MuseGlimmerVisionConfiguration
    public let modelType: String
    public let imageTokenId: Int
    public let videoTokenId: Int
    public let outHiddenSize: Int
    public let projectorHiddenSize: Int

    public var vocabularySize: Int { textConfiguration.vocabularySize }

    enum CodingKeys: String, CodingKey {
        case textConfiguration = "text_config"
        case visionConfiguration = "vision_config"
        case modelType = "model_type"
        case imageTokenId = "image_token_id"
        case videoTokenId = "video_token_id"
        case outHiddenSize = "out_hidden_size"
        case projectorHiddenSize = "projector_hidden_size"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.textConfiguration = try container.decode(
            MuseGlimmerTextConfiguration.self, forKey: .textConfiguration)
        self.visionConfiguration = try container.decode(
            MuseGlimmerVisionConfiguration.self, forKey: .visionConfiguration)
        self.modelType =
            try container.decodeIfPresent(String.self, forKey: .modelType) ?? "muse_glimmer"
        self.imageTokenId =
            try container.decodeIfPresent(Int.self, forKey: .imageTokenId) ?? 200092
        self.videoTokenId =
            try container.decodeIfPresent(Int.self, forKey: .videoTokenId) ?? 200091
        self.outHiddenSize =
            try container.decodeIfPresent(Int.self, forKey: .outHiddenSize) ?? 6144
        self.projectorHiddenSize =
            try container.decodeIfPresent(Int.self, forKey: .projectorHiddenSize) ?? 4096
    }
}

// MARK: - Norms

/// RMS norm with no learnable scale — `mx.fast.rms_norm(x, None, eps)`.
///
/// Holds no `MLXArray`, so it contributes no parameters. That matters: the
/// checkpoint has no weights for `embed_norm`, `qk_norm` or
/// `perception_emb_norm`, and weight loading verifies the module tree against
/// the checkpoint keys exactly.
private class MuseRMSNormNoScale: Module, UnaryLayer {
    let eps: Float

    init(eps: Float) {
        self.eps = eps
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(x, weight: MLXArray.mlxNone, eps: eps)
    }
}

/// RMS norm whose checkpoint scale is centered at zero, so the effective scale
/// is `1 + weight`.
///
/// Not the same as `MLXNN.RMSNorm` (which uses `weight` directly) and not
/// mean-subtracting despite the name. The fp32 cast ordering follows the
/// reference exactly — folding `1 + weight` in bf16 instead drifts enough to
/// change decode choices.
private class MuseCenteredRMSNorm: Module, UnaryLayer {
    @ModuleInfo var weight: MLXArray
    let eps: Float

    init(dimensions: Int, eps: Float) {
        self._weight.wrappedValue = MLXArray.zeros([dimensions])
        self.eps = eps
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let dtype = x.dtype
        let f = x.asType(.float32)
        let variance = mean(f * f, axis: -1, keepDims: true)
        var out = f * rsqrt(variance + eps)
        out = out * (1.0 + weight.asType(.float32))
        return out.asType(dtype)
    }
}

// MARK: - Text tower

private class MuseGlimmerTextMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(_ config: MuseGlimmerTextConfiguration) {
        self._gateProj.wrappedValue = Linear(
            config.hiddenSize, config.intermediateSize, bias: false)
        self._upProj.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        self._downProj.wrappedValue = Linear(
            config.intermediateSize, config.hiddenSize, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}

private class MuseGlimmerTextAttention: Module {
    @ModuleInfo(key: "q_proj") var queryProj: Linear
    @ModuleInfo(key: "k_proj") var keyProj: Linear
    @ModuleInfo(key: "v_proj") var valueProj: Linear
    @ModuleInfo(key: "o_proj") var outputProj: Linear

    /// Gates the attention output. Distinct from `mlp.gate_proj` — same
    /// checkpoint key name, different module.
    @ModuleInfo(key: "gate_proj") var gateProj: Linear

    @ModuleInfo(key: "qk_norm") var qkNorm: MuseRMSNormNoScale
    @ModuleInfo var rope: RoPELayer

    let heads: Int
    let kvHeads: Int
    let headDimensions: Int

    /// The ordinary softmax scale. `qkScaleFactor` below is applied *on top* of
    /// this, not instead of it.
    let scale: Float
    let qkScaleFactor: Float

    /// False on full-attention layers, whose `layer_rope_theta` is 0 (NoPE).
    let useRope: Bool
    let isSliding: Bool

    init(_ config: MuseGlimmerTextConfiguration, layerIndex: Int) {
        self.heads = config.attentionHeads
        self.kvHeads = config.kvHeads
        self.headDimensions = config.headDimensions
        self.scale = pow(Float(config.headDimensions), -0.5)
        self.qkScaleFactor = config.qkScaleFactor

        let theta = config.layerRopeTheta[layerIndex]
        self.useRope = theta != 0
        self.isSliding = config.layerTypes[layerIndex] == "sliding_attention"

        let dim = config.hiddenSize
        let queryDim = config.attentionHeads * config.headDimensions
        let kvDim = config.kvHeads * config.headDimensions

        self._queryProj.wrappedValue = Linear(dim, queryDim, bias: config.attentionBias)
        self._keyProj.wrappedValue = Linear(dim, kvDim, bias: config.attentionBias)
        self._valueProj.wrappedValue = Linear(dim, kvDim, bias: config.attentionBias)
        self._gateProj.wrappedValue = Linear(dim, queryDim, bias: false)
        self._outputProj.wrappedValue = Linear(queryDim, dim, bias: config.attentionBias)
        self._qkNorm.wrappedValue = MuseRMSNormNoScale(eps: config.rmsNormEps)

        let effectiveTheta = self.useRope ? theta : config.ropeTheta
        self._rope.wrappedValue = initializeRope(
            dims: config.headDimensions,
            base: effectiveTheta,
            traditional: false,
            scalingConfig: [
                "rope_type": .string("default"),
                "rope_theta": .float(effectiveTheta),
            ],
            maxPositionEmbeddings: config.maxPositionEmbeddings
        )
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let (batch, length) = (x.dim(0), x.dim(1))

        var queries = queryProj(x).reshaped(batch, length, heads, headDimensions)
        var keys = keyProj(x).reshaped(batch, length, kvHeads, headDimensions)
        var values = valueProj(x).reshaped(batch, length, kvHeads, headDimensions)

        queries = (qkNorm(queries) * qkScaleFactor).transposed(0, 2, 1, 3)
        keys = qkNorm(keys).transposed(0, 2, 1, 3)
        values = values.transposed(0, 2, 1, 3)

        if useRope {
            let offset = cache?.offset ?? 0
            queries = rope(queries, offset: offset)
            keys = rope(keys, offset: offset)
        }

        var output = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: cache,
            scale: scale,
            mask: mask
        )
        output = output.transposed(0, 2, 1, 3).reshaped(batch, length, -1)

        // Gate is computed from the layer input, not the attention output.
        output = output * sigmoid(gateProj(x))
        return outputProj(output)
    }
}

private class MuseGlimmerDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttention: MuseGlimmerTextAttention
    @ModuleInfo var mlp: MuseGlimmerTextMLP

    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: MuseCenteredRMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: MuseCenteredRMSNorm
    @ModuleInfo(key: "pre_feedforward_layernorm") var preFeedforwardLayerNorm: MuseCenteredRMSNorm
    @ModuleInfo(key: "post_feedforward_layernorm") var postFeedforwardLayerNorm: MuseCenteredRMSNorm

    let isSliding: Bool

    init(_ config: MuseGlimmerTextConfiguration, layerIndex: Int) {
        self._selfAttention.wrappedValue = MuseGlimmerTextAttention(
            config, layerIndex: layerIndex)
        self._mlp.wrappedValue = MuseGlimmerTextMLP(config)

        // Note the two different eps values: the "pre" norms use rms_norm_eps
        // and the "post" norms use the much smaller post_norm_eps.
        self._inputLayerNorm.wrappedValue = MuseCenteredRMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = MuseCenteredRMSNorm(
            dimensions: config.hiddenSize, eps: config.postNormEps)
        self._preFeedforwardLayerNorm.wrappedValue = MuseCenteredRMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postFeedforwardLayerNorm.wrappedValue = MuseCenteredRMSNorm(
            dimensions: config.hiddenSize, eps: config.postNormEps)

        self.isSliding = config.layerTypes[layerIndex] == "sliding_attention"
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        var h =
            x
            + postAttentionLayerNorm(
                selfAttention(inputLayerNorm(x), mask: mask, cache: cache))
        h = h + postFeedforwardLayerNorm(mlp(preFeedforwardLayerNorm(h)))
        return h
    }
}

private class MuseGlimmerTextModel: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    /// Applied to the token embeddings before layer 0. Parameterless.
    @ModuleInfo(key: "embed_norm") var embedNorm: MuseRMSNormNoScale

    @ModuleInfo var layers: [MuseGlimmerDecoderLayer]

    /// The final norm is a plain `RMSNorm` (scale used directly), unlike the
    /// centered norms inside the blocks.
    @ModuleInfo var norm: RMSNorm

    let config: MuseGlimmerTextConfiguration
    private let fullAttentionIndex: Int
    private let slidingAttentionIndex: Int?

    init(_ config: MuseGlimmerTextConfiguration) {
        self.config = config
        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
        self._embedNorm.wrappedValue = MuseRMSNormNoScale(eps: config.rmsNormEps)
        self._layers.wrappedValue = (0 ..< config.hiddenLayers).map {
            MuseGlimmerDecoderLayer(config, layerIndex: $0)
        }
        self._norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)

        self.fullAttentionIndex = config.layerTypes.firstIndex(of: "full_attention") ?? 0
        self.slidingAttentionIndex = config.layerTypes.firstIndex(of: "sliding_attention")
        super.init()
    }

    func callAsFunction(
        _ inputs: MLXArray?,
        cache: [KVCache?]? = nil,
        inputEmbedding: MLXArray? = nil
    ) -> MLXArray {
        var h: MLXArray
        if let inputEmbedding {
            h = inputEmbedding
        } else {
            h = embedNorm(embedTokens(inputs!))
        }

        let layerCache = cache ?? Array(repeating: nil, count: layers.count)

        // Two masks only: one for the full-attention layers and one for the
        // sliding layers, each derived from the first cache of its own kind so
        // the offsets are right.
        let fullMask = createAttentionMask(h: h, cache: layerCache[fullAttentionIndex])
        var slidingMask = MLXFast.ScaledDotProductAttentionMaskMode.none
        if let slidingAttentionIndex {
            slidingMask = createAttentionMask(
                h: h, cache: layerCache[slidingAttentionIndex],
                windowSize: config.slidingWindow)
        }

        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: layer.isSliding ? slidingMask : fullMask, cache: layerCache[i])
        }
        return norm(h)
    }
}

private class MuseGlimmerLanguageModel: Module, KVCacheDimensionProvider {
    @ModuleInfo var model: MuseGlimmerTextModel
    @ModuleInfo(key: "lm_head") var lmHead: Linear

    let config: MuseGlimmerTextConfiguration
    var kvHeads: [Int]

    init(_ config: MuseGlimmerTextConfiguration) {
        self.config = config
        self._model.wrappedValue = MuseGlimmerTextModel(config)
        self._lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabularySize, bias: false)
        self.kvHeads = Array(repeating: config.kvHeads, count: config.hiddenLayers)
        super.init()
    }

    /// Sliding layers get a rotating cache bounded by the architecture window;
    /// full layers honor the requested capacity and are otherwise unbounded.
    func newCache(parameters: GenerateParameters?) throws -> [any KVCache] {
        try config.layerTypes.map { layerType in
            try makeHybridAttentionKVCache(
                parameters: parameters,
                slidingWindow: config.slidingWindow,
                usesSlidingWindow: layerType == "sliding_attention")
        }
    }

    func callAsFunction(
        _ inputs: MLXArray? = nil,
        cache: [KVCache]? = nil,
        inputEmbedding: MLXArray? = nil
    ) -> LMOutput {
        let optionalCache = cache?.map { $0 as KVCache? }
        let out = model(inputs, cache: optionalCache, inputEmbedding: inputEmbedding)

        var logits = lmHead(out) * config.outputMultiplier
        let cap = config.finalLogitSoftcapping
        if cap > 0 {
            logits = tanh(logits / cap) * cap
        }
        return LMOutput(logits: logits)
    }
}

// MARK: - Vision helpers

/// Cumulative per-frame sequence boundaries, matching `vision._cu_seqlens`.
func museCuSeqlens(_ grid: [THW]) -> [Int] {
    var result = [0]
    for frame in grid {
        for _ in 0 ..< frame.t {
            result.append(result[result.count - 1] + frame.h * frame.w)
        }
    }
    return result
}

/// Permutation that makes each attention window contiguous, plus the window
/// boundaries on the permuted sequence. Port of `vision._window_index`.
///
/// Returns `nil` for the permutation when it is the identity, which happens for
/// any grid within a single window (images up to 448×448) — the reference keeps
/// that fast path and so do we.
func museWindowIndex(_ grid: [THW], windowSize: Int) -> (MLXArray?, [Int]) {
    var indices = [Int32]()
    var cumulative = [0]
    var offset = 0

    for frame in grid {
        let (t, h, w) = (frame.t, frame.h, frame.w)
        // Python's `%` on a negative left operand returns a non-negative result;
        // Swift's does not, hence the explicit form.
        let padH = (windowSize - h % windowSize) % windowSize
        let padW = (windowSize - w % windowSize) % windowSize
        let nH = (h + padH) / windowSize
        let nW = (w + padW) / windowSize

        // Walk the padded (t, nH, windowSize, nW, windowSize) layout in the
        // transposed (t, nH, nW, windowSize, windowSize) order, skipping the
        // padding sentinels. That is exactly what the reference's pad/reshape/
        // transpose/filter pipeline produces.
        for time in 0 ..< t {
            for blockH in 0 ..< nH {
                for blockW in 0 ..< nW {
                    var length = 0
                    for row in 0 ..< windowSize {
                        let y = blockH * windowSize + row
                        if y >= h { continue }
                        for column in 0 ..< windowSize {
                            let x = blockW * windowSize + column
                            if x >= w { continue }
                            indices.append(
                                Int32(offset + time * h * w + y * w + x))
                            length += 1
                        }
                    }
                    cumulative.append(cumulative[cumulative.count - 1] + length)
                }
            }
        }
        offset += t * h * w
    }

    // Drop zero-length windows (repeated cumulative values).
    var boundaries = [cumulative[0]]
    for value in cumulative.dropFirst() where value != boundaries[boundaries.count - 1] {
        boundaries.append(value)
    }

    let isIdentity = indices.enumerated().allSatisfy { Int32($0.offset) == $0.element }
    return (isIdentity ? nil : MLXArray(indices), boundaries)
}

/// One-indexed `(width, height)` position pairs in raster order, tiled over the
/// temporal axis. Port of `vision._position_ids`.
///
/// The one-indexing and the width-before-height ordering both matter: the rotary
/// helper reads column 0 as width and column 1 as height.
func musePositionIds(_ grid: [THW]) -> MLXArray {
    var values = [Int32]()
    for frame in grid {
        var frameValues = [Int32]()
        for row in 0 ..< frame.h {
            for column in 0 ..< frame.w {
                frameValues.append(Int32(column + 1))
                frameValues.append(Int32(row + 1))
            }
        }
        for _ in 0 ..< frame.t {
            values.append(contentsOf: frameValues)
        }
    }
    let total = grid.reduce(0) { $0 + $1.product }
    return MLXArray(values, [total, 2])
}

/// SDPA that keeps the fused kernel reachable. Port of `base.ensure_fused_sdpa`.
///
/// The ViT's head dimension is 96, which is not one of the fused kernel's
/// supported sizes, so pad to 128 and slice back. Zero-padding q/k contributes
/// nothing to any dot product and the padded v channels are discarded, so this
/// is exact rather than approximate.
func museFusedSDPA(_ q: MLXArray, _ k: MLXArray, _ v: MLXArray, scale: Float) -> MLXArray {
    let d = q.dim(-1)
    guard let target = [64, 80, 128].first(where: { d <= $0 }), target != d else {
        return MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: scale, mask: .none)
    }
    let widths =
        Array(repeating: IntOrPair(0), count: q.ndim - 1) + [IntOrPair((0, target - d))]
    let output = MLXFast.scaledDotProductAttention(
        queries: padded(q, widths: widths),
        keys: padded(k, widths: widths),
        values: padded(v, widths: widths),
        scale: scale,
        mask: .none
    )
    return output[.ellipsis, ..<d]
}

// MARK: - Vision tower

/// Bilinearly resamples the learned `side` × `side` position table onto the
/// actual patch grid, as a gather of four corners with explicit weights.
///
/// Ported literally from `PatchEmbedder._position_embeddings`, including the
/// half-pixel convention and the validity masks, which *zero* an out-of-range
/// corner's contribution rather than clamping it. A CoreImage resize or a
/// generic interpolation helper would not reproduce either.
///
/// `lookup` performs the table gather, so the caller can hand in an `Embedding`
/// module (the model) or index a raw array (the tests).
func museInterpolatedPositionEmbeddings(
    grid: [THW], side: Int, lookup: (MLXArray) -> MLXArray
) -> MLXArray {
    let total = grid.reduce(0) { $0 + $1.product }
    var indices = [Int32](repeating: 0, count: 4 * total)
    var weights = [Float](repeating: 0, count: 4 * total)

    /// Corner index, fractional offset and in-range flags for one axis.
    func axis(_ count: Int) -> ([Int], [Int], [Float], [Bool], [Bool]) {
        var low = [Int](repeating: 0, count: count)
        var high = [Int](repeating: 0, count: count)
        var delta = [Float](repeating: 0, count: count)
        var validLow = [Bool](repeating: false, count: count)
        var validHigh = [Bool](repeating: false, count: count)
        for i in 0 ..< count {
            let g = (Float(i) + 0.5) * (Float(side) / Float(count)) - 0.5
            let floored = Int(g.rounded(.down))
            delta[i] = g - Float(floored)
            validLow[i] = floored >= 0 && floored < side
            validHigh[i] = floored + 1 >= 0 && floored + 1 < side
            low[i] = min(max(floored, 0), side - 1)
            high[i] = min(max(floored + 1, 0), side - 1)
        }
        return (low, high, delta, validLow, validHigh)
    }

    var cursor = 0
    for frame in grid {
        let (t, h, w) = (frame.t, frame.h, frame.w)
        let (h0, h1, dh, validH0, validH1) = axis(h)
        let (w0, w1, dw, validW0, validW1) = axis(w)
        let frameCount = h * w

        for i in 0 ..< h {
            for j in 0 ..< w {
                let local = i * w + j
                let corners = [
                    (h0[i], w0[j], (1 - dh[i]) * (1 - dw[j]), validH0[i] && validW0[j]),
                    (h0[i], w1[j], (1 - dh[i]) * dw[j], validH0[i] && validW1[j]),
                    (h1[i], w0[j], dh[i] * (1 - dw[j]), validH1[i] && validW0[j]),
                    (h1[i], w1[j], dh[i] * dw[j], validH1[i] && validW1[j]),
                ]
                for (corner, entry) in corners.enumerated() {
                    let (y, x, weight, valid) = entry
                    // Tiled across the temporal axis, matching np.tile.
                    for time in 0 ..< t {
                        let position = cursor + time * frameCount + local
                        indices[corner * total + position] = Int32(y * side + x)
                        weights[corner * total + position] = valid ? weight : 0
                    }
                }
            }
        }
        cursor += t * frameCount
    }

    let table = lookup(MLXArray(indices, [4, total]))
    let weightArray = MLXArray(weights, [4, total])
    return (table * expandedDimensions(weightArray, axis: -1)).sum(axis: 0)
}

/// 2×2 patch merge. The channel axis is moved *before* the two merge axes, so
/// each output `dim * merge * merge` vector is channel-major: all four spatial
/// values for channel 0, then channel 1, and so on.
///
/// The naive `(merge, merge, dim)` ordering is a different permutation of the
/// same numbers and would yield plausible-looking garbage, so this is asserted
/// against the reference on a ramp in the tests.
func musePixelShuffle(_ hiddenStates: MLXArray, grid: [THW], mergeSize merge: Int) -> MLXArray {
    let dim = hiddenStates.dim(-1)
    var outputs = [MLXArray]()
    var offset = 0
    for frame in grid {
        let (t, h, w) = (frame.t, frame.h, frame.w)
        let count = t * h * w
        var chunk = hiddenStates[offset ..< (offset + count)]
        chunk = chunk.reshaped(t, h / merge, merge, w / merge, merge, dim)
        chunk = chunk.transposed(0, 1, 3, 5, 2, 4)
        outputs.append(chunk.reshaped(-1, dim * merge * merge))
        offset += count
    }
    return outputs.count == 1 ? outputs[0] : concatenated(outputs, axis: 0)
}

/// Splits a `[1, C, H, W]` image into flattened patch vectors.
///
/// Two details differ from ``QwenVL/patchify(images:mergeSize:patchSize:temporalPatchSize:)``
/// and both are load-bearing: tokens come out in plain raster order (the vision
/// tower does its own 2×2 merge grouping later), and each token's values are laid
/// out `(temporal, channels, patchH, patchW)` rather than channels-first.
func museGlimmerPatchify(
    _ pixels: MLXArray, gridH: Int, gridW: Int, patchSize: Int, temporalPatchSize: Int
) -> MLXArray {
    let channels = pixels.dim(-3)
    var patches = pixels.reshaped(channels, gridH, patchSize, gridW, patchSize)
    patches = patches.transposed(1, 3, 0, 2, 4)
    patches = patches.expandedDimensions(axis: 2)
    // A size-1 axis, so tiling is `np.repeat` along that axis.
    patches = tiled(patches, repetitions: [1, 1, temporalPatchSize, 1, 1, 1])
    return patches.reshaped(
        gridH * gridW, temporalPatchSize * channels * patchSize * patchSize)
}

private class MuseGlimmerPatchEmbedder: Module {
    /// A `Linear`, not a convolution — the processor pre-patchifies, so this
    /// consumes flattened `temporal * channels * patch * patch` vectors.
    @ModuleInfo(key: "patch_embedding") var patchEmbedding: Linear
    @ModuleInfo(key: "position_embedding_table") var positionEmbeddingTable: Embedding

    let side: Int

    init(_ config: MuseGlimmerVisionConfiguration) {
        let patchDim = config.patchTemporal * 3 * config.patchSize * config.patchSize
        self._patchEmbedding.wrappedValue = Linear(patchDim, config.hiddenSize, bias: false)
        self._positionEmbeddingTable.wrappedValue = Embedding(
            embeddingCount: config.positionEmbeddingHeight * config.positionEmbeddingWidth,
            dimensions: config.hiddenSize)
        self.side = config.positionEmbeddingHeight
        super.init()
    }

    func callAsFunction(_ pixelValues: MLXArray, grid: [THW]) -> MLXArray {
        let embeddings = patchEmbedding(pixelValues)
        let positions = museInterpolatedPositionEmbeddings(grid: grid, side: side) {
            positionEmbeddingTable($0)
        }.asType(embeddings.dtype)
        return embeddings + positions
    }
}

private class MuseGlimmerVisionAttention: Module {
    @ModuleInfo(key: "q_proj") var queryProj: Linear
    @ModuleInfo(key: "k_proj") var keyProj: Linear
    @ModuleInfo(key: "v_proj") var valueProj: Linear
    @ModuleInfo(key: "proj") var outputProj: Linear

    let heads: Int
    let headDimensions: Int
    let scale: Float

    init(_ config: MuseGlimmerVisionConfiguration) {
        self.heads = config.attentionHeads
        self.headDimensions = config.hiddenSize / config.attentionHeads
        self.scale = pow(Float(self.headDimensions), -0.5)

        self._queryProj.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: true)
        self._keyProj.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: true)
        self._valueProj.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: true)
        self._outputProj.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: true)
        super.init()
    }

    func callAsFunction(
        _ hiddenStates: MLXArray,
        splitPoints: [Int],
        cos: MLXArray,
        sin: MLXArray
    ) -> MLXArray {
        let length = hiddenStates.dim(0)
        var q = queryProj(hiddenStates).reshaped(1, length, heads, -1)
        var k = keyProj(hiddenStates).reshaped(1, length, heads, -1)
        let v = valueProj(hiddenStates).reshaped(1, length, heads, -1)

        // 2D rope: cos/sin are already full head-width, so this is a plain
        // rotate-half application (no tiling, unlike Qwen 2.5-VL).
        let cosExpanded = expandedDimensions(expandedDimensions(cos, axis: 0), axis: 2)
        let sinExpanded = expandedDimensions(expandedDimensions(sin, axis: 0), axis: 2)
        let qDtype = q.dtype
        let kDtype = k.dtype
        q =
            (q.asType(.float32) * cosExpanded
            + QwenVL.rotateHalf(q).asType(.float32) * sinExpanded).asType(qDtype)
        k =
            (k.asType(.float32) * cosExpanded
            + QwenVL.rotateHalf(k).asType(.float32) * sinExpanded).asType(kDtype)

        let queries = q.transposed(0, 2, 1, 3)
        let keys = k.transposed(0, 2, 1, 3)
        let values = v.transposed(0, 2, 1, 3)

        var output: MLXArray
        if splitPoints.isEmpty {
            output = museFusedSDPA(queries, keys, values, scale: scale)
        } else {
            // Windowed (or per-frame) attention as independent chunks along the
            // sequence axis. No mask is built: the permutation already made each
            // window contiguous.
            let qs = split(queries, indices: splitPoints, axis: 2)
            let ks = split(keys, indices: splitPoints, axis: 2)
            let vs = split(values, indices: splitPoints, axis: 2)
            output = concatenated(
                (0 ..< qs.count).map { museFusedSDPA(qs[$0], ks[$0], vs[$0], scale: scale) },
                axis: 2)
        }

        output = output.transposed(0, 2, 1, 3).reshaped(length, -1)
        return outputProj(output)
    }
}

private class MuseGlimmerVisionMLP: Module, UnaryLayer {
    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear

    init(_ config: MuseGlimmerVisionConfiguration) {
        self._fc1.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: true)
        self._fc2.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: true)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Exact (erf-based) gelu, matching nn.gelu in the reference.
        fc2(geluExact(fc1(x)))
    }
}

/// Exact erf-based GELU, matching `nn.gelu`. `MLXNN.gelu` is already exact; this
/// alias exists to make the choice explicit against `geluApproximate`.
private func geluExact(_ x: MLXArray) -> MLXArray { MLXNN.gelu(x) }

private class MuseGlimmerVisionBlock: Module {
    @ModuleInfo(key: "norm1") var norm1: LayerNorm
    @ModuleInfo(key: "norm2") var norm2: LayerNorm
    @ModuleInfo(key: "attn") var attention: MuseGlimmerVisionAttention
    @ModuleInfo var mlp: MuseGlimmerVisionMLP

    init(_ config: MuseGlimmerVisionConfiguration) {
        self._norm1.wrappedValue = LayerNorm(
            dimensions: config.hiddenSize, eps: config.layerNormEps)
        self._norm2.wrappedValue = LayerNorm(
            dimensions: config.hiddenSize, eps: config.layerNormEps)
        self._attention.wrappedValue = MuseGlimmerVisionAttention(config)
        self._mlp.wrappedValue = MuseGlimmerVisionMLP(config)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, splitPoints: [Int], cos: MLXArray, sin: MLXArray
    ) -> MLXArray {
        var h = x + attention(norm1(x), splitPoints: splitPoints, cos: cos, sin: sin)
        h = h + mlp(norm2(h))
        return h
    }
}

private class MuseGlimmerVisionModel: Module {
    @ModuleInfo(key: "patch_embedder") var patchEmbedder: MuseGlimmerPatchEmbedder
    @ModuleInfo(key: "ln_pre") var lnPre: LayerNorm
    @ModuleInfo var layers: [MuseGlimmerVisionBlock]
    @ModuleInfo(key: "ln_post") var lnPost: LayerNorm

    let config: MuseGlimmerVisionConfiguration
    private let ropeSpatialDim: Int

    init(_ config: MuseGlimmerVisionConfiguration) {
        self.config = config
        self._patchEmbedder.wrappedValue = MuseGlimmerPatchEmbedder(config)
        self._lnPre.wrappedValue = LayerNorm(
            dimensions: config.hiddenSize, eps: config.layerNormEps)
        self._layers.wrappedValue = (0 ..< config.hiddenLayers).map { _ in
            MuseGlimmerVisionBlock(config)
        }
        self._lnPost.wrappedValue = LayerNorm(
            dimensions: config.hiddenSize, eps: config.layerNormEps)
        self.ropeSpatialDim = (config.hiddenSize / config.attentionHeads) / 2
        super.init()
    }

    /// 2D rope frequencies. `freqs` is `[w, h, w, h]` over quarter-width blocks,
    /// which is what makes a plain rotate-half pair width with width and height
    /// with height.
    private func rotary(_ positionIds: MLXArray) -> (MLXArray, MLXArray) {
        let exponent =
            MLXArray(stride(from: 0, to: ropeSpatialDim, by: 2)).asType(.float32)
            / Float(ropeSpatialDim)
        let inverseFreq = 1.0 / pow(MLXArray(config.ropeTheta), exponent)

        let width = positionIds[0..., 0].asType(.float32)
        let height = positionIds[0..., 1].asType(.float32)
        let freqW = expandedDimensions(width, axis: -1) * expandedDimensions(inverseFreq, axis: 0)
        let freqH = expandedDimensions(height, axis: -1) * expandedDimensions(inverseFreq, axis: 0)
        let freqs = concatenated([freqW, freqH, freqW, freqH], axis: -1)
        return (cos(freqs), sin(freqs))
    }

    func callAsFunction(_ pixelValues: MLXArray, grid: [THW]) -> MLXArray {
        let fullBoundaries = museCuSeqlens(grid)
        let fullSplitPoints = Array(fullBoundaries.dropFirst().dropLast())

        // The window side reuses `pos_emb_height` (32). That looks accidental in
        // the reference, but every published checkpoint was trained with it, so
        // it is reproduced rather than "fixed".
        let windowSide = config.positionEmbeddingHeight
        let (windowIndex, windowBoundaries) = museWindowIndex(grid, windowSize: windowSide)
        let windowSplitPoints = Array(windowBoundaries.dropFirst().dropLast())

        var hiddenStates = lnPre(patchEmbedder(pixelValues, grid: grid))
        var positions = musePositionIds(grid)
        if let windowIndex {
            hiddenStates = hiddenStates[windowIndex]
            positions = positions[windowIndex]
        }
        let (cosValues, sinValues) = rotary(positions)

        for (index, block) in layers.enumerated() {
            let splitPoints =
                config.layerTypes[index] == "full_attention"
                ? fullSplitPoints : windowSplitPoints
            hiddenStates = block(
                hiddenStates, splitPoints: splitPoints, cos: cosValues, sin: sinValues)
        }

        if let windowIndex {
            hiddenStates = hiddenStates[argSort(windowIndex, axis: 0)]
        }
        hiddenStates = lnPost(hiddenStates)
        return musePixelShuffle(hiddenStates, grid: grid, mergeSize: config.mergeSize)
    }
}

// MARK: - Vision → text bridge

private class MuseGlimmerVisionAdapter: Module, UnaryLayer {
    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear

    init(_ config: MuseGlimmerConfiguration) {
        // Both biasless, unlike the vision tower's projections.
        self._fc1.wrappedValue = Linear(
            config.outHiddenSize, config.projectorHiddenSize, bias: false)
        self._fc2.wrappedValue = Linear(
            config.projectorHiddenSize, config.projectorHiddenSize, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // gelu after *both* layers.
        geluExact(fc2(geluExact(fc1(x))))
    }
}

// MARK: - Model

public class MuseGlimmer: Module, VLMModel, KVCacheDimensionProvider {
    @ModuleInfo(key: "language_model") private var languageModel: MuseGlimmerLanguageModel
    @ModuleInfo(key: "vision_tower") private var visionTower: MuseGlimmerVisionModel
    @ModuleInfo(key: "vision_adapter") private var visionAdapter: MuseGlimmerVisionAdapter
    @ModuleInfo(key: "vision_projection") private var visionProjection: Linear

    /// Parameterless, so it carries no checkpoint weights.
    @ModuleInfo(key: "perception_emb_norm") private var perceptionEmbNorm: MuseRMSNormNoScale

    public let config: MuseGlimmerConfiguration

    public var vocabularySize: Int { config.vocabularySize }
    public var kvHeads: [Int] { languageModel.kvHeads }
    public var toolCallFormat: ToolCallFormat? { .atem }
    public var reasoningConfig: ReasoningConfig? {
        ReasoningConfig(
            startDelimiter: "to=self<|message|>", endDelimiter: "<|eom|>",
            promptStrategy: .none, isSpecialToken: true)
    }

    public init(_ config: MuseGlimmerConfiguration) {
        self.config = config
        self._languageModel.wrappedValue = MuseGlimmerLanguageModel(config.textConfiguration)
        self._visionTower.wrappedValue = MuseGlimmerVisionModel(config.visionConfiguration)
        self._visionAdapter.wrappedValue = MuseGlimmerVisionAdapter(config)
        self._visionProjection.wrappedValue = Linear(
            config.projectorHiddenSize, config.textConfiguration.hiddenSize, bias: false)
        self._perceptionEmbNorm.wrappedValue = MuseRMSNormNoScale(
            eps: config.textConfiguration.rmsNormEps)
        super.init()
    }

    public func newCache(parameters: GenerateParameters?) throws -> [any KVCache] {
        try languageModel.newCache(parameters: parameters)
    }

    private func encodeImage(_ pixelValues: MLXArray, grid: [THW]) -> MLXArray {
        let dtype = visionTower.patchEmbedder.patchEmbedding.weight.dtype
        var features = visionTower(pixelValues.asType(dtype), grid: grid)
        features = visionAdapter(features)
        features = visionProjection(features)
        return perceptionEmbNorm(features)
    }

    /// Token embeddings with any image features spliced in, always shaped
    /// `[1, length, hidden]`.
    ///
    /// The rank is part of the contract: ``prepare(_:cache:state:prefill:)``
    /// chunks along axis 1, so a 2-D result would silently chunk along the hidden
    /// dimension instead of the sequence.
    private func inputEmbeddings(
        inputIds: MLXArray, pixelValues: MLXArray?, grid: [THW]
    ) throws -> MLXArray {
        let ids = inputIds.ndim == 1 ? inputIds.expandedDimensions(axis: 0) : inputIds
        let textModel = languageModel.model
        var embeddings = textModel.embedNorm(textModel.embedTokens(ids))
        if embeddings.ndim == 2 {
            embeddings = embeddings.expandedDimensions(axis: 0)
        }

        guard let pixelValues else { return embeddings }

        // Image features are already normalized by `perception_emb_norm`; they
        // deliberately do not go through `embed_norm`.
        let imageFeatures = encodeImage(pixelValues, grid: grid).asType(embeddings.dtype)

        let imageTokenId = config.imageTokenId
        let videoTokenId = config.videoTokenId
        var positions = [Int32]()
        for (index, id) in ids.asArray(Int.self).enumerated()
        where id == imageTokenId || id == videoTokenId {
            positions.append(Int32(index))
        }

        // Mirrors the reference's guard. This is the earliest signal that the
        // processor's token expansion and the vision grid disagree.
        guard positions.count == imageFeatures.dim(0) else {
            throw VLMError.processing(
                "Image features and image tokens do not match: tokens=\(positions.count), "
                    + "features=\(imageFeatures.dim(0))")
        }

        embeddings[0..., MLXArray(positions), 0...] = imageFeatures.expandedDimensions(axis: 0)
        return embeddings
    }

    public func prepare(
        _ input: LMInput, cache: [any KVCache], state: LMOutput.State?,
        prefill: PrefillParameters
    ) throws -> PrepareResult {
        let convertedCache = cache.compactMap { $0 as KVCache }

        var tokens = input.text.tokens
        if tokens.ndim == 1 { tokens = tokens.expandedDimensions(axis: 0) }

        let embeddings = try inputEmbeddings(
            inputIds: tokens,
            pixelValues: input.image?.pixels,
            grid: input.image?.frames ?? []
        )

        // Chunked prefill is not optional here: a full-resolution image can add
        // thousands of tokens on top of the prompt, and a single pass through 52
        // layers at hidden 6656 will exhaust memory.
        //
        // No mask is passed — the text model derives both the sliding and the
        // full mask from the per-layer caches, which a single `.causal` mode
        // could not express.
        let totalPositions = embeddings.dim(1)
        let processed = try prefill.forEachChunk(total: totalPositions) { range in
            _ = languageModel(
                nil, cache: convertedCache, inputEmbedding: embeddings[0..., range, 0...])
            asyncEval(cache)
        }
        if processed > 0 { eval(cache) }
        let result = languageModel(
            nil, cache: convertedCache, inputEmbedding: embeddings[0..., processed..., 0...])
        prefill.progress?(totalPositions, totalPositions)
        return .logits(result)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [any KVCache]?) -> MLXArray {
        languageModel(inputs, cache: cache).logits
    }

    /// Key renaming only — a byte-for-byte port of `muse_glimmer.Model.sanitize`.
    ///
    /// The `mlx-community` repos are already published in the target layout, so
    /// this is a no-op there; the prefix rewrites cover the original
    /// `transformers` checkpoint. Notably there are no transposes: the patch
    /// embedder is a `Linear`, not a convolution.
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var result = [String: MLXArray](minimumCapacity: weights.count)
        for (key, value) in weights {
            if key.contains("rotary_emb.inv_freq") { continue }
            var key = key
            if key.hasPrefix("model.language_model.") {
                key = "language_model.model." + key.dropFirst("model.language_model.".count)
            } else if key.hasPrefix("model.") {
                key = String(key.dropFirst("model.".count))
            } else if key.hasPrefix("lm_head.") {
                key = "language_model." + key
            }
            result[key] = value
        }
        return result
    }
}

extension MuseGlimmer: LoRAModel {
    public var loraLayers: [Module] {
        languageModel.model.layers
    }
}

// MARK: - Processor

public struct MuseGlimmerProcessorConfiguration: Codable, Sendable {
    public struct ImageProcessor: Codable, Sendable {
        public let patchSize: Int
        public let temporalPatchSize: Int
        public let mergeSize: Int
        public let maxImageTokens: Int
        public let imageMean: [CGFloat]
        public let imageStd: [CGFloat]

        enum CodingKeys: String, CodingKey {
            case patchSize = "patch_size"
            case temporalPatchSize = "temporal_patch_size"
            case mergeSize = "merge_size"
            case maxImageTokens = "max_image_tokens"
            case imageMean = "image_mean"
            case imageStd = "image_std"
        }

        public init(from decoder: Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            self.patchSize = try container.decodeIfPresent(Int.self, forKey: .patchSize) ?? 14
            self.temporalPatchSize =
                try container.decodeIfPresent(Int.self, forKey: .temporalPatchSize) ?? 2
            self.mergeSize = try container.decodeIfPresent(Int.self, forKey: .mergeSize) ?? 2
            self.maxImageTokens =
                try container.decodeIfPresent(Int.self, forKey: .maxImageTokens) ?? 4096
            self.imageMean =
                try container.decodeIfPresent([CGFloat].self, forKey: .imageMean) ?? [
                    0.5, 0.5, 0.5,
                ]
            self.imageStd =
                try container.decodeIfPresent([CGFloat].self, forKey: .imageStd) ?? [0.5, 0.5, 0.5]
        }

        /// Builds a configuration in code, chiefly so a caller can lower
        /// ``maxImageTokens`` from the checkpoint's 4096. See the note in
        /// ``MuseGlimmerProcessor/preprocess(image:processing:)`` on why that
        /// budget dominates time-to-first-token.
        public init(
            patchSize: Int = 14,
            temporalPatchSize: Int = 2,
            mergeSize: Int = 2,
            maxImageTokens: Int = 4096,
            imageMean: [CGFloat] = [0.5, 0.5, 0.5],
            imageStd: [CGFloat] = [0.5, 0.5, 0.5]
        ) {
            self.patchSize = patchSize
            self.temporalPatchSize = temporalPatchSize
            self.mergeSize = mergeSize
            self.maxImageTokens = maxImageTokens
            self.imageMean = imageMean
            self.imageStd = imageStd
        }

        public var imageMeanTuple: (CGFloat, CGFloat, CGFloat) {
            (imageMean[0], imageMean[1], imageMean[2])
        }
        public var imageStdTuple: (CGFloat, CGFloat, CGFloat) {
            (imageStd[0], imageStd[1], imageStd[2])
        }
    }

    /// `processor_config.json` nests the image-processor fields under
    /// `image_processor` with `processor_class` at the root.
    public let imageProcessor: ImageProcessor

    enum CodingKeys: String, CodingKey {
        case imageProcessor = "image_processor"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.imageProcessor =
            try container.decodeIfPresent(ImageProcessor.self, forKey: .imageProcessor)
            ?? ImageProcessor()
    }

    public init(imageProcessor: ImageProcessor = ImageProcessor()) {
        self.imageProcessor = imageProcessor
    }
}

/// Picks the patch grid whose aspect ratio best matches the source, under the
/// token budget. Port of `processing_muse_glimmer.smart_resize`.
///
/// `factor` is `patch_size * merge_size` (28), so the grid is measured in *merged*
/// tokens and `maxTokens` bounds the post-merge sequence length.
///
/// The reference selects with `min()` over a Python `set`, whose iteration order
/// makes ties non-deterministic. Candidates are sorted here so the choice is
/// stable; ties are rare but would otherwise change the token count run to run.
func museSmartResize(height: Int, width: Int, factor: Int, maxTokens: Int) -> (Int, Int) {
    var idealH = Float(height) / Float(factor)
    var idealW = Float(width) / Float(factor)
    let ratio = idealH > 0 ? idealW / idealH : 1.0

    if idealH * idealW > Float(maxTokens) {
        idealH = (Float(maxTokens) / ratio).squareRoot()
        idealW = idealH * ratio
    }

    let hOptions = Set([Int(idealH.rounded(.down)), Int(idealH.rounded(.up))])
    let wOptions = Set([Int(idealW.rounded(.down)), Int(idealW.rounded(.up))])
    var candidates = [(Int, Int)]()
    for h in hOptions.sorted() {
        for w in wOptions.sorted() where h >= 1 && w >= 1 && h * w <= maxTokens {
            candidates.append((h, w))
        }
    }
    if candidates.isEmpty {
        candidates = [
            (
                max(1, Int(idealH.rounded(.toNearestOrEven))),
                max(1, Int(idealW.rounded(.toNearestOrEven)))
            )
        ]
    }

    let sourceRatio = Float(height) / Float(width)
    let best = candidates.min { lhs, rhs in
        abs(Float(lhs.0) / Float(lhs.1) - sourceRatio)
            < abs(Float(rhs.0) / Float(rhs.1) - sourceRatio)
    }!
    return (best.0 * factor, best.1 * factor)
}

public struct MuseGlimmerProcessor: UserInputProcessor {
    private let config: MuseGlimmerProcessorConfiguration
    private let tokenizer: any Tokenizer

    /// Resolved lazily from the tokenizer where possible; the literals are the
    /// published ids and act as a fallback.
    private static let defaultImageTokenId = 200092
    private static let defaultImageStartTokenId = 200080
    private static let defaultImageEndTokenId = 200081

    public init(_ config: MuseGlimmerProcessorConfiguration, tokenizer: any Tokenizer) {
        self.config = config
        self.tokenizer = tokenizer
    }

    /// Resizes, normalizes and patchifies one image.
    ///
    /// The patch layout is `(temporal, channels, patchH, patchW)` inside each
    /// token and the tokens themselves are in plain raster order — both differ
    /// from `QwenVL.patchify`, and both matter because the vision tower does its
    /// own 2×2 merge grouping downstream.
    public func preprocess(image: CIImage, processing: UserInput.Processing?) throws -> (
        MLXArray, THW
    ) {
        let imageConfig = config.imageProcessor
        let factor = imageConfig.patchSize * imageConfig.mergeSize

        var userProcessing = processing ?? UserInput.Processing()

        // The token budget dominates latency, because each merged token becomes a
        // prompt token that has to be prefilled through the full text stack — an
        // image at the default 4096 can cost tens of seconds before the first
        // output token. `maxPixels` is the documented per-call hook for exactly
        // this, and one merged token covers `factor * factor` pixels, so the two
        // budgets convert directly.
        var maxTokens = imageConfig.maxImageTokens
        if let maxPixels = userProcessing.maxPixels {
            maxTokens = min(maxTokens, max(1, maxPixels / (factor * factor)))
        }

        // A caller-supplied `resize` bounds the source rather than replacing the
        // target: an off-grid size would break patch/merge alignment and the token
        // count, but silently ignoring the request would leave a caller no way to
        // trade resolution for latency.
        var sourceHeight = Int(image.extent.height.rounded())
        var sourceWidth = Int(image.extent.width.rounded())
        if let resize = userProcessing.resize, resize.width > 0, resize.height > 0 {
            sourceHeight = min(sourceHeight, Int(resize.height.rounded()))
            sourceWidth = min(sourceWidth, Int(resize.width.rounded()))
        }

        let (targetHeight, targetWidth) = museSmartResize(
            height: sourceHeight,
            width: sourceWidth,
            factor: factor,
            maxTokens: maxTokens
        )
        let targetSize = CGSize(width: targetWidth, height: targetHeight)

        // Hand `MediaProcessing.apply` the grid-aligned size, so any resize it
        // performs matches what the patchifier below assumes.
        userProcessing.resize = targetSize

        let processed = MediaProcessing.apply(image, processing: userProcessing)
        let srgb = MediaProcessing.inSRGBToneCurveSpace(processed)
        // The reference resamples with PIL's LANCZOS.
        let resized = MediaProcessing.resampleLanczos(srgb, to: targetSize)
        let normalized = MediaProcessing.normalize(
            resized, mean: imageConfig.imageMeanTuple, std: imageConfig.imageStdTuple)
        let pixels = MediaProcessing.asMLXArray(normalized)

        let patchSize = imageConfig.patchSize
        let gridH = targetHeight / patchSize
        let gridW = targetWidth / patchSize
        let patches = museGlimmerPatchify(
            pixels, gridH: gridH, gridW: gridW, patchSize: patchSize,
            temporalPatchSize: imageConfig.temporalPatchSize)

        return (patches, THW(1, gridH, gridW))
    }

    private func tokenId(_ token: String, default fallback: Int) -> Int {
        tokenizer.convertTokenToId(token) ?? fallback
    }

    public func prepare(input: UserInput) async throws -> LMInput {
        // The released processor has no video patchification path. Reject it
        // explicitly so a rendered `<|video|>` placeholder cannot reach the
        // model without corresponding features.
        guard input.videos.isEmpty else { throw VLMError.singleMediaTypeAllowed }

        let messages = Qwen2VLMessageGenerator().generate(from: input)
        var promptTokens = try tokenizer.applyChatTemplate(
            messages: messages, tools: input.tools, additionalContext: input.additionalContext)

        guard !input.images.isEmpty else {
            let promptArray = MLXArray(promptTokens).expandedDimensions(axis: 0)
            let mask = ones(like: promptArray).asType(.int8)
            return LMInput(text: .init(tokens: promptArray, mask: mask))
        }

        let processed = try input.images.map {
            try preprocess(image: $0.asCIImage(), processing: input.processing)
        }
        let pixels = concatenated(processed.map { $0.0 })
        let frames = processed.map { $0.1 }

        // The template emits one `<|patch|>` per image; expand each into
        // `<|image_start|>`, one `<|patch|>` per merged token, `<|image_end|>`.
        let imageTokenId = tokenId("<|patch|>", default: Self.defaultImageTokenId)
        let imageStartTokenId = tokenId(
            "<|image_start|>", default: Self.defaultImageStartTokenId)
        let imageEndTokenId = tokenId("<|image_end|>", default: Self.defaultImageEndTokenId)
        let mergeLength = config.imageProcessor.mergeSize * config.imageProcessor.mergeSize

        var expanded = [Int]()
        var imageIndex = 0
        for token in promptTokens {
            guard token == imageTokenId else {
                expanded.append(token)
                continue
            }
            guard imageIndex < frames.count else {
                throw VLMError.processing(
                    "More image placeholders in the prompt than images supplied")
            }
            let count = frames[imageIndex].product / mergeLength
            expanded.append(imageStartTokenId)
            expanded.append(contentsOf: Array(repeating: imageTokenId, count: count))
            expanded.append(imageEndTokenId)
            imageIndex += 1
        }
        guard imageIndex == frames.count else {
            throw VLMError.processing(
                "Prompt has \(imageIndex) image placeholders but \(frames.count) images were "
                    + "supplied")
        }
        promptTokens = expanded

        let promptArray = MLXArray(promptTokens).expandedDimensions(axis: 0)
        let mask = ones(like: promptArray).asType(.int8)
        return LMInput(
            text: .init(tokens: promptArray, mask: mask),
            image: .init(pixels: pixels, frames: frames)
        )
    }
}
