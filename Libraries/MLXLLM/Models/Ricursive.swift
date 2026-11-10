// Ricursive (Ripple-65B) text model.
//
// Port of the reference `modeling_ricursive.py` in ricursiveAI/Ripple-65B-Preview (Apache-2.0).
// Structurally a Gemma 3 block (four norms per layer, per-head Q/K norm, a sliding/full
// attention mix) with gated attention, a value residual, clamped SwiGLU, and RoPE on the
// sliding layers only.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

public struct RicursiveConfiguration: Codable, Sendable {
    public var modelType: String
    public var vocabularySize: Int
    public var hiddenSize: Int
    public var intermediateSize: Int
    public var hiddenLayers: Int
    public var attentionHeads: Int
    public var kvHeads: Int
    public var headDim: Int
    public var rmsNormEps: Float
    public var ropeTheta: Float
    public var slidingWindow: Int
    public var layerTypes: [String]
    public var useGatedAttention: Bool
    public var gateFullPrecision: Bool
    public var useValueResidual: Bool
    public var ffnClampThreshold: Float?
    public var embedScale: Float?
    public var hiddenAct: String
    public var tieWordEmbeddings: Bool

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabularySize = "vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case slidingWindow = "sliding_window"
        case layerTypes = "layer_types"
        case useGatedAttention = "use_gated_attention"
        case gateFullPrecision = "gate_full_precision"
        case useValueResidual = "use_valres"
        case ffnClampThreshold = "ffn_clamp_threshold"
        case embedScale = "embed_scale"
        case hiddenAct = "hidden_act"
        case tieWordEmbeddings = "tie_word_embeddings"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decode(String.self, forKey: .modelType)
        vocabularySize = try c.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 100_352
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 3072
        intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 24576
        let layers = try c.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 50
        hiddenLayers = layers
        attentionHeads = try c.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 48
        kvHeads = try c.decodeIfPresent(Int.self, forKey: .kvHeads) ?? 8
        headDim = try c.decodeIfPresent(Int.self, forKey: .headDim) ?? 128
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10_000
        slidingWindow = try c.decodeIfPresent(Int.self, forKey: .slidingWindow) ?? 1024
        // The reference default: every fifth layer is full attention.
        layerTypes =
            try c.decodeIfPresent([String].self, forKey: .layerTypes)
            ?? (0 ..< layers).map { $0 % 5 == 4 ? "full_attention" : "sliding_attention" }
        useGatedAttention = try c.decodeIfPresent(Bool.self, forKey: .useGatedAttention) ?? true
        gateFullPrecision = try c.decodeIfPresent(Bool.self, forKey: .gateFullPrecision) ?? true
        useValueResidual = try c.decodeIfPresent(Bool.self, forKey: .useValueResidual) ?? false
        ffnClampThreshold = try c.decodeIfPresent(Float.self, forKey: .ffnClampThreshold)
        embedScale = try c.decodeIfPresent(Float.self, forKey: .embedScale)
        hiddenAct = try c.decodeIfPresent(String.self, forKey: .hiddenAct) ?? "silu"
        tieWordEmbeddings = try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
    }

    func isSliding(layer: Int) -> Bool { layerTypes[layer] == "sliding_attention" }
}

extension RicursiveConfiguration: ModelConfigurationValidating {
    public func validateModelConfiguration() throws {
        func fail(_ message: String) -> Error {
            NSError(
                domain: "RicursiveConfiguration", code: 1,
                userInfo: [NSLocalizedDescriptionKey: message])
        }
        guard layerTypes.count == hiddenLayers else {
            throw fail("layer_types needs one entry per layer")
        }
        guard layerTypes.allSatisfy({ $0 == "sliding_attention" || $0 == "full_attention" }) else {
            throw fail("unsupported Ricursive layer_types")
        }
        guard kvHeads > 0, attentionHeads % kvHeads == 0 else {
            throw fail("num_attention_heads must be a multiple of num_key_value_heads")
        }
        guard headDim % 2 == 0 else { throw fail("head_dim must be even for RoPE") }
        guard slidingWindow > 0 else { throw fail("sliding_window must be positive") }
        guard hiddenAct == "silu" else { throw fail("Ricursive only uses hidden_act silu") }
        guard !tieWordEmbeddings else { throw fail("Ricursive has a separate lm_head") }
    }
}

private final class RicursiveAttention: Module {
    let nHeads: Int
    let nKVHeads: Int
    let headDim: Int
    let scale: Float
    let isSliding: Bool
    let gateFullPrecision: Bool
    // Leading underscore keeps this out of the module's parameters.
    private let _inverseFrequencies: MLXArray

    @ModuleInfo(key: "q_proj") var queryProj: Linear
    @ModuleInfo(key: "k_proj") var keyProj: Linear
    @ModuleInfo(key: "v_proj") var valueProj: Linear
    @ModuleInfo(key: "o_proj") var outputProj: Linear
    @ModuleInfo(key: "g_proj") var gateProj: Linear?
    @ModuleInfo(key: "q_norm") var queryNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var keyNorm: RMSNorm
    @ParameterInfo(key: "value_res_lambda") var valueResidualLambda: MLXArray?

    init(_ config: RicursiveConfiguration, layer: Int) {
        let dim = config.hiddenSize
        nHeads = config.attentionHeads
        nKVHeads = config.kvHeads
        headDim = config.headDim
        scale = pow(Float(config.headDim), -0.5)
        isSliding = config.isSliding(layer: layer)
        gateFullPrecision = config.gateFullPrecision
        _inverseFrequencies =
            1
            / pow(
                MLXArray(config.ropeTheta),
                MLXArray(stride(from: 0, to: config.headDim, by: 2).map { Float($0) })
                    / Float(config.headDim))

        _queryProj.wrappedValue = Linear(dim, nHeads * headDim, bias: false)
        _keyProj.wrappedValue = Linear(dim, nKVHeads * headDim, bias: false)
        _valueProj.wrappedValue = Linear(dim, nKVHeads * headDim, bias: false)
        _outputProj.wrappedValue = Linear(nHeads * headDim, dim, bias: false)
        _gateProj.wrappedValue =
            config.useGatedAttention ? Linear(dim, nHeads * headDim, bias: false) : nil
        _queryNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        _keyNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        _valueResidualLambda.wrappedValue = config.useValueResidual ? MLXArray.zeros([1]) : nil
        super.init()
    }

    /// RoPE in float32 on `[B, heads, T, headDim]`. Positions cannot go through bfloat16,
    /// which loses integer precision after 256. The reference casts cos and sin to the model
    /// precision before widening them again, so the same rounding is applied here.
    private func rope(_ x: MLXArray, offset: Int) -> MLXArray {
        let length = x.dim(2)
        let positions = MLXArray(Int32(offset) ..< Int32(offset + length)).asType(.float32)
        let frequencies = positions[0..., .newAxis] * _inverseFrequencies[.newAxis, 0...]
        let angles = concatenated([frequencies, frequencies], axis: -1)
        let cosine = cos(angles).asType(x.dtype).asType(.float32)
        let sine = sin(angles).asType(x.dtype).asType(.float32)
        let source = x.asType(.float32)
        let half = source.dim(-1) / 2
        let rotated = concatenated(
            [-source[.ellipsis, half...], source[.ellipsis, ..<half]], axis: -1)
        return (source * cosine + rotated * sine).asType(x.dtype)
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?,
        valueResidual: MLXArray?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        var valueInput = x
        if let valueResidual, let lambda = valueResidualLambda {
            let blend = sigmoid(lambda[0]).asType(x.dtype)
            let start = valueResidual.asType(x.dtype)
            valueInput = start + blend * (x - start)
        }

        var queries = queryNorm(queryProj(x).reshaped(B, L, nHeads, headDim))
        var keys = keyNorm(keyProj(x).reshaped(B, L, nKVHeads, headDim))
        let values = valueProj(valueInput).reshaped(B, L, nKVHeads, headDim)

        queries = queries.transposed(0, 2, 1, 3)
        keys = keys.transposed(0, 2, 1, 3)

        if isSliding {
            let offset = cache?.offset ?? 0
            queries = rope(queries, offset: offset)
            keys = rope(keys, offset: offset)
        }

        var output = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values.transposed(0, 2, 1, 3),
            cache: cache,
            scale: scale,
            mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        if let gateProj {
            var gate = gateProj(x)
            if gateFullPrecision {
                gate = gate.asType(.float32)
            }
            output = output * sigmoid(gate).asType(output.dtype)
        }
        return outputProj(output)
    }
}

private final class RicursiveMLP: Module, UnaryLayer {
    let clampThreshold: Float?

    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(_ config: RicursiveConfiguration) {
        // The reference treats a zero threshold as no clamp.
        clampThreshold = (config.ffnClampThreshold ?? 0) != 0 ? config.ffnClampThreshold : nil
        _gateProj.wrappedValue = Linear(
            config.hiddenSize, config.intermediateSize, bias: false)
        _upProj.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        _downProj.wrappedValue = Linear(
            config.intermediateSize, config.hiddenSize, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var gate = gateProj(x)
        var up = upProj(x)
        if let clampThreshold {
            let limit = MLXArray(clampThreshold).asType(gate.dtype)
            gate = minimum(gate, limit)
            up = clip(up, min: -limit, max: limit)
        }
        return downProj(silu(gate) * up)
    }
}

private final class RicursiveBlock: Module {
    let isSliding: Bool

    @ModuleInfo(key: "self_attn") var selfAttention: RicursiveAttention
    @ModuleInfo var mlp: RicursiveMLP
    @ModuleInfo(key: "input_layernorm") var inputNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionNorm: RMSNorm
    @ModuleInfo(key: "pre_feedforward_layernorm") var preFeedforwardNorm: RMSNorm
    @ModuleInfo(key: "post_feedforward_layernorm") var postFeedforwardNorm: RMSNorm

    init(_ config: RicursiveConfiguration, layer: Int) {
        isSliding = config.isSliding(layer: layer)
        let (dim, eps) = (config.hiddenSize, config.rmsNormEps)
        _selfAttention.wrappedValue = RicursiveAttention(config, layer: layer)
        _mlp.wrappedValue = RicursiveMLP(config)
        _inputNorm.wrappedValue = RMSNorm(dimensions: dim, eps: eps)
        _postAttentionNorm.wrappedValue = RMSNorm(dimensions: dim, eps: eps)
        _preFeedforwardNorm.wrappedValue = RMSNorm(dimensions: dim, eps: eps)
        _postFeedforwardNorm.wrappedValue = RMSNorm(dimensions: dim, eps: eps)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?,
        valueResidual: MLXArray?
    ) -> MLXArray {
        let attended = selfAttention(
            inputNorm(x), mask: mask, cache: cache, valueResidual: valueResidual)
        let h = x + postAttentionNorm(attended)
        return h + postFeedforwardNorm(mlp(preFeedforwardNorm(h)))
    }
}

private final class RicursiveBackbone: Module {
    let config: RicursiveConfiguration
    let firstSliding: Int?
    let firstFull: Int?

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo var layers: [RicursiveBlock]
    @ModuleInfo var norm: RMSNorm

    init(_ config: RicursiveConfiguration) {
        self.config = config
        firstSliding = config.layerTypes.firstIndex(of: "sliding_attention")
        firstFull = config.layerTypes.firstIndex(of: "full_attention")
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
        _layers.wrappedValue = (0 ..< config.hiddenLayers).map {
            RicursiveBlock(config, layer: $0)
        }
        _norm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        super.init()
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var h = embedTokens(inputs)
        if let embedScale = config.embedScale {
            // The reference multiplies by a Python scalar, which takes the tensor's precision.
            h = h * MLXArray(embedScale).asType(h.dtype)
        }
        let valueResidual = config.useValueResidual ? h : nil

        // Masks depend only on the attention kind, so build one per kind.
        let slidingMask = firstSliding.map {
            createAttentionMask(h: h, cache: cache?[$0], windowSize: config.slidingWindow)
        }
        let fullMask = firstFull.map { createAttentionMask(h: h, cache: cache?[$0]) }

        for (i, layer) in layers.enumerated() {
            let mask = (layer.isSliding ? slidingMask : fullMask) ?? .none
            h = layer(h, mask: mask, cache: cache?[i], valueResidual: valueResidual)
        }
        return norm(h)
    }
}

public final class RicursiveModel: Module, LLMModel {
    public let vocabularySize: Int
    public let configuration: RicursiveConfiguration

    @ModuleInfo fileprivate var model: RicursiveBackbone
    @ModuleInfo(key: "lm_head") var lmHead: Linear

    public init(_ configuration: RicursiveConfiguration) {
        self.configuration = configuration
        vocabularySize = configuration.vocabularySize
        _model.wrappedValue = RicursiveBackbone(configuration)
        _lmHead.wrappedValue = Linear(
            configuration.hiddenSize, configuration.vocabularySize, bias: false)
        super.init()
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        projectLogits(hiddenStates(inputs, cache: cache))
    }

    public var loraLayers: [Module] { model.layers }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        weights.filter { !$0.key.contains("rotary_emb.inv_freq") }
    }

    /// Sliding layers keep a rotating window; full layers keep the whole context.
    public func newCache(parameters: GenerateParameters? = nil) throws -> [KVCache] {
        try (0 ..< configuration.hiddenLayers).map { layer in
            if configuration.isSliding(layer: layer) {
                return try makeSlidingWindowKVCache(
                    parameters: parameters, window: configuration.slidingWindow)
            }
            let cache = try makeAttentionKVCache(parameters: parameters)
            if let simple = cache as? StandardKVCache {
                simple.step = 1024
            }
            return cache
        }
    }
}

extension RicursiveModel: HiddenStateLanguageModel {
    package func hiddenStates(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        model(inputs, cache: cache)
    }

    package func projectLogits(_ hiddenStates: MLXArray) -> MLXArray {
        lmHead(hiddenStates)
    }
}
