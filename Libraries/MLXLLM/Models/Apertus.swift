import Foundation
import MLX
import MLXFast
import MLXLMCommon
import MLXNN

// MARK: - Configuration

public struct ApertusConfiguration: Codable, Sendable {
    public var hiddenSize: Int
    public var intermediateSize: Int
    public var numHiddenLayers: Int
    public var numAttentionHeads: Int
    public var numKeyValueHeads: Int
    public var rmsNormEps: Float
    public var vocabSize: Int
    public var ropeTheta: Float
    public var ropeTraditional: Bool
    public var ropeScaling: [String: StringOrNumber]?

    public init(
        hiddenSize: Int = 4096,
        intermediateSize: Int = 21504,
        numHiddenLayers: Int = 32,
        numAttentionHeads: Int = 32,
        numKeyValueHeads: Int? = 8,
        rmsNormEps: Float = 1e-5,
        vocabSize: Int = 131072,
        ropeTheta: Float = 1_000_000.0,
        ropeTraditional: Bool = false,
        ropeScaling: [String: StringOrNumber]? = nil
    ) {
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.numHiddenLayers = numHiddenLayers
        self.numAttentionHeads = numAttentionHeads
        self.numKeyValueHeads = numKeyValueHeads ?? numAttentionHeads
        self.rmsNormEps = rmsNormEps
        self.vocabSize = vocabSize
        self.ropeTheta = ropeTheta
        self.ropeTraditional = ropeTraditional
        self.ropeScaling = ropeScaling
    }

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case rmsNormEps = "rms_norm_eps"
        case vocabSize = "vocab_size"
        case ropeTheta = "rope_theta"
        case ropeTraditional = "rope_traditional"
        case ropeScaling = "rope_scaling"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        // Required fields
        let hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        let intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)
        let numHiddenLayers = try container.decode(Int.self, forKey: .numHiddenLayers)
        let numAttentionHeads = try container.decode(Int.self, forKey: .numAttentionHeads)
        let rmsNormEps = try container.decode(Float.self, forKey: .rmsNormEps)
        let vocabSize = try container.decode(Int.self, forKey: .vocabSize)

        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.numHiddenLayers = numHiddenLayers
        self.numAttentionHeads = numAttentionHeads
        self.rmsNormEps = rmsNormEps
        self.vocabSize = vocabSize

        // Optional fields with defaults
        self.numKeyValueHeads =
            try container.decodeIfPresent(Int.self, forKey: .numKeyValueHeads) ?? numAttentionHeads
        self.ropeTheta =
            try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 1_000_000.0
        self.ropeTraditional =
            try container.decodeIfPresent(Bool.self, forKey: .ropeTraditional) ?? false
        self.ropeScaling = try container.decodeIfPresent(
            [String: StringOrNumber].self, forKey: .ropeScaling)
    }
}

// MARK: - Layers

// Expanded Integral of the Exponential Linear Unit
public class XIELU: Module, UnaryLayer {
    @ModuleInfo(key: "alpha_p") var alphaPParam: MLXArray
    @ModuleInfo(key: "alpha_n") var alphaNParam: MLXArray
    @ModuleInfo(key: "beta") var betaParam: MLXArray
    @ModuleInfo(key: "eps") var epsParam: MLXArray

    override public init() {
        self._alphaPParam.wrappedValue = MLXArray(0.55)
        self._alphaNParam.wrappedValue = MLXArray(0.55)
        self._betaParam.wrappedValue = MLXArray(0.5)
        self._epsParam.wrappedValue = MLXArray(-1e-6)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let alphaP = softplus(alphaPParam)
        let alphaN = betaParam + softplus(alphaNParam)

        let posTerm = alphaP * square(x) + betaParam * x
        let negTerm = alphaN * (exp(minimum(x, epsParam)) - 1) - alphaN * x + betaParam * x

        return MLX.where(x .> 0, posTerm, negTerm)
    }
}

public class ApertusAttention: Module {
    let args: ApertusConfiguration
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    // Apertus Specific: RMSNorm on Q and K
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    let rope: RoPE

    init(args: ApertusConfiguration) {
        self.args = args

        let dim = args.hiddenSize
        let heads = args.numAttentionHeads
        let kvHeads = args.numKeyValueHeads
        let headDim = args.hiddenSize / heads
        self.scale = pow(Float(headDim), -0.5)

        self._qProj.wrappedValue = Linear(dim, heads * headDim, bias: false)
        self._kProj.wrappedValue = Linear(dim, kvHeads * headDim, bias: false)
        self._vProj.wrappedValue = Linear(dim, kvHeads * headDim, bias: false)
        self._oProj.wrappedValue = Linear(heads * headDim, dim, bias: false)

        // Norm applies to the head dimension
        self._qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.rmsNormEps)
        self._kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.rmsNormEps)

        self.rope = RoPE(
            dimensions: headDim,
            traditional: args.ropeTraditional,
            base: args.ropeTheta
        )
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache? = nil
    ) -> MLXArray {
        let B = x.dim(0)
        let L = x.dim(1)
        let heads = args.numAttentionHeads
        let kvHeads = args.numKeyValueHeads
        let headDim = args.hiddenSize / heads

        var queries = qProj(x)
        var keys = kProj(x)
        var values = vProj(x)

        // 1. Reshape to [B, L, Heads, HeadDim] to apply Norms
        queries = queries.reshaped([B, L, heads, headDim])
        keys = keys.reshaped([B, L, kvHeads, headDim])
        values = values.reshaped([B, L, kvHeads, headDim])

        // 2. Apply QK-Norms (Apertus Specific)
        queries = qNorm(queries)
        keys = kNorm(keys)

        // 3. Transpose to [B, Heads, L, HeadDim] for RoPE and SDPA
        queries = queries.transposed(0, 2, 1, 3)
        keys = keys.transposed(0, 2, 1, 3)
        values = values.transposed(0, 2, 1, 3)

        // 4. RoPE
        if let cache = cache {
            queries = rope(queries, offset: cache.offset)
            keys = rope(keys, offset: cache.offset)

            // Update cache (expects [B, H, L, D])
            let (k, v) = cache.update(keys: keys, values: values)
            keys = k
            values = v
        } else {
            queries = rope(queries)
            keys = rope(keys)
        }

        // 5. Attention (SDPA expects [B, H, L, D])
        let output = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            mask: mask
        )

        // 6. Transpose back to [B, L, Heads, HeadDim] and fuse
        let outputFused =
            output
            .transposed(0, 2, 1, 3)
            .reshaped([B, L, heads * headDim])

        return oProj(outputFused)
    }
}

public class ApertusMLP: Module {
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear
    @ModuleInfo(key: "act_fn") var act: XIELU

    public init(dim: Int, hiddenDim: Int) {
        self._upProj.wrappedValue = Linear(dim, hiddenDim, bias: false)
        self._downProj.wrappedValue = Linear(hiddenDim, dim, bias: false)
        self._act.wrappedValue = XIELU()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        return downProj(act(upProj(x)))
    }
}

public class ApertusBlock: Module {
    @ModuleInfo(key: "self_attn") public var selfAttn: ApertusAttention
    @ModuleInfo(key: "mlp") public var mlp: ApertusMLP
    @ModuleInfo(key: "attention_layernorm") public var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "feedforward_layernorm") public var postAttentionLayerNorm: RMSNorm

    public init(args: ApertusConfiguration) {
        self._selfAttn.wrappedValue = ApertusAttention(args: args)
        self._mlp.wrappedValue = ApertusMLP(dim: args.hiddenSize, hiddenDim: args.intermediateSize)
        self._inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    public func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache? = nil
    ) -> MLXArray {
        let r = selfAttn(inputLayerNorm(x), mask: mask, cache: cache)
        let h = x + r
        return h + mlp(postAttentionLayerNorm(h))
    }
}

public class ApertusModel: Module {
    @ModuleInfo(key: "embed_tokens") public var embedTokens: Embedding
    @ModuleInfo(key: "layers") public var layers: [ApertusBlock]
    @ModuleInfo(key: "norm") public var norm: RMSNorm

    public init(args: ApertusConfiguration) {
        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabSize,
            dimensions: args.hiddenSize
        )
        self._layers.wrappedValue = (0 ..< args.numHiddenLayers).map { _ in
            ApertusBlock(args: args)
        }
        self._norm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    public func callAsFunction(
        _ inputs: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: [KVCache]? = nil
    ) -> MLXArray {
        var h = embedTokens(inputs)

        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[i])
        }

        return norm(h)
    }
}

public class ApertusForCausalLM: Module, LLMModel, LoRAModel, KVCacheDimensionProvider {
    public let config: ApertusConfiguration
    public let kvHeads: [Int]
    @ModuleInfo(key: "model") public var model: ApertusModel
    @ModuleInfo(key: "lm_head") public var lmHead: Linear

    public init(config: ApertusConfiguration) {
        self.config = config
        self.kvHeads = (0 ..< config.numHiddenLayers).map { _ in config.numKeyValueHeads }
        self._model.wrappedValue = ApertusModel(args: config)
        self._lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabSize, bias: false)
    }

    public func callAsFunction(
        _ inputs: MLXArray,
        cache: [KVCache]? = nil
    ) -> MLXArray {
        let mask = createAttentionMask(h: inputs, cache: cache)
        let out = model(inputs, mask: mask, cache: cache)
        return lmHead(out)
    }

    // MARK: - LLMModel Protocol Conformance

    public var vocabularySize: Int { config.vocabSize }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        return weights
    }

    // MARK: - LoRAModel Protocol Conformance

    public var loraLayers: [Module] {
        return model.layers
    }
}
