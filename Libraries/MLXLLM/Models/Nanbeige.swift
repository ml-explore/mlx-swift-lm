//
//  Nanbeige.swift
//  LLM
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// Port of https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/nanbeige.py
//
// Nanbeige4.2 is a Looped Transformer: the same decoder-layer stack is applied
// `num_loops` times per forward pass with shared weights. Each loop pass has
// its own KV cache entries (cache count = num_loops * num_hidden_layers) and,
// unless `skip_loop_final_norm` is set, ends with the final RMSNorm.

class NanbeigeAttention: Module {
    let args: NanbeigeConfiguration
    let scale: Float

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear

    let rope: RoPELayer

    public init(_ args: NanbeigeConfiguration) {
        self.args = args

        let dim = args.hiddenSize
        let heads = args.attentionHeads
        let kvHeads = args.kvHeads
        let headDim = args.resolvedHeadDimensions
        self.scale = pow(Float(headDim), -0.5)

        _wq.wrappedValue = Linear(dim, heads * headDim, bias: args.attentionBias)
        _wk.wrappedValue = Linear(dim, kvHeads * headDim, bias: args.attentionBias)
        _wv.wrappedValue = Linear(dim, kvHeads * headDim, bias: args.attentionBias)
        _wo.wrappedValue = Linear(heads * headDim, dim, bias: args.attentionBias)

        self.rope = initializeRope(
            dims: headDim,
            base: args.ropeTheta,
            traditional: false,
            scalingConfig: args.ropeScaling,
            maxPositionEmbeddings: args.maxPositionEmbeddings
        )
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

class NanbeigeMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    @ModuleInfo(key: "up_proj") var up: Linear

    public init(_ args: NanbeigeConfiguration) {
        _gate.wrappedValue = Linear(args.hiddenSize, args.intermediateSize, bias: args.mlpBias)
        _down.wrappedValue = Linear(args.intermediateSize, args.hiddenSize, bias: args.mlpBias)
        _up.wrappedValue = Linear(args.hiddenSize, args.intermediateSize, bias: args.mlpBias)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        down(silu(gate(x)) * up(x))
    }
}

class NanbeigeTransformerBlock: Module {
    @ModuleInfo(key: "self_attn") var attention: NanbeigeAttention
    let mlp: NanbeigeMLP

    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    public init(_ args: NanbeigeConfiguration) {
        _attention.wrappedValue = NanbeigeAttention(args)
        self.mlp = NanbeigeMLP(args)
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
        return h + r
    }
}

public class NanbeigeModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    fileprivate let layers: [NanbeigeTransformerBlock]
    let norm: RMSNorm

    let numLoops: Int
    let skipLoopFinalNorm: Bool

    /// One KV cache per (loop pass, layer) pair — the single owner of the
    /// loops × layers layout that `callAsFunction` indexes and `newCache`
    /// allocates.
    var cacheSlotCount: Int { numLoops * layers.count }

    public init(_ args: NanbeigeConfiguration) {
        precondition(args.vocabularySize > 0)
        // Mirror the Python reference's __post_init__ rejection so a direct
        // construction cannot silently run reference-only features that have
        // no dedicated weights. The factory path throws earlier (and
        // recoverably) via validateModelConfiguration.
        precondition(
            !args.enableDoubleLoopSplit && !args.loopShareKV && !args.enableDepthAttention,
            "enable_double_loop_split, loop_share_kv, and enable_depth_attention are not supported"
        )

        self.numLoops = args.effectiveNumLoops
        self.skipLoopFinalNorm = args.skipLoopFinalNorm

        _embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)

        self.layers = (0 ..< args.hiddenLayers)
            .map { _ in
                NanbeigeTransformerBlock(args)
            }
        self.norm = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var h = embedTokens(inputs)

        let mask = createAttentionMask(h: h, cache: cache?.first)

        // The layer stack is applied numLoops times with shared weights; loop
        // pass `p` owns cache slots `p * layers.count ..< (p + 1) * layers.count`.
        for loopIndex in 0 ..< numLoops {
            let cacheBase = loopIndex * layers.count
            for (i, layer) in layers.enumerated() {
                h = layer(h, mask: mask, cache: cache?[cacheBase + i])
            }
            if !skipLoopFinalNorm {
                h = norm(h)
            }
        }

        if skipLoopFinalNorm {
            h = norm(h)
        }
        return h
    }
}

public class NanbeigeModel: Module, LLMModel {
    public let vocabularySize: Int

    public let model: NanbeigeModelInner
    let configuration: NanbeigeConfiguration

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ args: NanbeigeConfiguration) {
        self.configuration = args
        self.vocabularySize = args.vocabularySize
        self.model = NanbeigeModelInner(args)

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

    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        let count = model.cacheSlotCount
        if let maxKVSize = parameters?.maxKVSize {
            return (0 ..< count).map { _ in
                RotatingKVCache(maxSize: maxKVSize, keep: 4)
            }
        } else {
            return (0 ..< count).map { _ in KVCacheSimple() }
        }
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var weights = weights.filter { key, _ in
            // Remove unused precomputed rotary freqs
            !key.contains("self_attn.rotary_emb.inv_freq")
        }

        if configuration.tieWordEmbeddings {
            weights["lm_head.weight"] = nil
        }

        return weights
    }
}

public struct NanbeigeConfiguration: Codable, Sendable {
    var hiddenSize: Int
    var hiddenLayers: Int
    var intermediateSize: Int
    var attentionHeads: Int
    var rmsNormEps: Float
    var vocabularySize: Int
    var kvHeads: Int
    var headDim: Int?
    var maxPositionEmbeddings: Int?
    var ropeTheta: Float = 10_000
    var ropeScaling: [String: StringOrNumber]? = nil
    var attentionBias = false
    var mlpBias = false
    var tieWordEmbeddings = false
    var numLoops: Int = 1
    var loopLossWeights: [Float]? = nil
    var skipLoopFinalNorm = false
    // Features of the reference implementation without dedicated weights which
    // would otherwise load silently and produce incorrect results:
    var enableDoubleLoopSplit = false
    var loopShareKV = false
    var enableDepthAttention = false

    var resolvedHeadDimensions: Int {
        headDim ?? (hiddenSize / attentionHeads)
    }

    /// Checkpoints trained with per-loop losses store one weight per extra loop.
    var effectiveNumLoops: Int {
        if let loopLossWeights, !loopLossWeights.isEmpty {
            return loopLossWeights.count + 1
        }
        return numLoops
    }

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case kvHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case maxPositionEmbeddings = "max_position_embeddings"
        case ropeTheta = "rope_theta"
        case ropeScaling = "rope_scaling"
        case attentionBias = "attention_bias"
        case mlpBias = "mlp_bias"
        case tieWordEmbeddings = "tie_word_embeddings"
        case numLoops = "num_loops"
        case loopLossWeights = "loop_loss_weights"
        case skipLoopFinalNorm = "skip_loop_final_norm"
        case enableDoubleLoopSplit = "enable_double_loop_split"
        case loopShareKV = "loop_share_kv"
        case enableDepthAttention = "enable_depth_attention"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        self.hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        self.hiddenLayers = try container.decode(Int.self, forKey: .hiddenLayers)
        self.intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)
        self.attentionHeads = try container.decode(Int.self, forKey: .attentionHeads)
        self.rmsNormEps = try container.decode(Float.self, forKey: .rmsNormEps)
        self.vocabularySize = try container.decode(Int.self, forKey: .vocabularySize)
        self.kvHeads = try container.decode(Int.self, forKey: .kvHeads)
        self.headDim = try container.decodeIfPresent(Int.self, forKey: .headDim)
        self.maxPositionEmbeddings = try container.decodeIfPresent(
            Int.self, forKey: .maxPositionEmbeddings)
        self.ropeTheta = try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10_000
        self.ropeScaling = try container.decodeIfPresent(
            [String: StringOrNumber].self, forKey: .ropeScaling)
        self.attentionBias =
            try container.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        self.mlpBias = try container.decodeIfPresent(Bool.self, forKey: .mlpBias) ?? false
        self.tieWordEmbeddings =
            try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        self.numLoops = try container.decodeIfPresent(Int.self, forKey: .numLoops) ?? 1
        self.loopLossWeights = try container.decodeIfPresent(
            [Float].self, forKey: .loopLossWeights)
        self.skipLoopFinalNorm =
            try container.decodeIfPresent(Bool.self, forKey: .skipLoopFinalNorm) ?? false
        self.enableDoubleLoopSplit =
            try container.decodeIfPresent(Bool.self, forKey: .enableDoubleLoopSplit) ?? false
        self.loopShareKV = try container.decodeIfPresent(Bool.self, forKey: .loopShareKV) ?? false
        self.enableDepthAttention =
            try container.decodeIfPresent(Bool.self, forKey: .enableDepthAttention) ?? false
    }
}

extension NanbeigeConfiguration: ModelConfigurationValidating {
    public func validateModelConfiguration() throws {
        try validateRoPEConfiguration(ropeScaling, context: "NanbeigeConfiguration.rope_scaling")

        if enableDoubleLoopSplit || loopShareKV || enableDepthAttention {
            throw ModelFactoryError.invalidConfiguration(
                "NanbeigeConfiguration: enable_double_loop_split, loop_share_kv, and "
                    + "enable_depth_attention are not yet supported.")
        }
    }
}

// MARK: - LoRA

extension NanbeigeModel: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}
