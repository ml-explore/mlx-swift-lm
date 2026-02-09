//
//  Qwen3_5Text.swift
//  mlx-swift-lm
//
//  Created by John Mai on 2026/2/9.
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

public struct Qwen3_5TextConfiguration: Codable, Sendable {
    var modelType: String = "qwen3_5_text"
    var hiddenSize: Int
    var hiddenLayers: Int
    var intermediateSize: Int
    var attentionHeads: Int
    var kvHeads: Int
    var linearNumValueHeads: Int
    var linearNumKeyHeads: Int
    var linearKeyHeadDim: Int
    var linearValueHeadDim: Int
    var linearConvKernelDim: Int
    var rmsNormEps: Float
    var vocabularySize: Int
    var ropeTheta: Float
    var partialRotaryFactor: Float
    var maxPositionEmbeddings: Int
    var tieWordEmbeddings: Bool
    var attentionBias: Bool
    var headDim: Int?
    var ropeScaling: [String: StringOrNumber]?
    var fullAttentionInterval: Int

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case linearNumValueHeads = "linear_num_value_heads"
        case linearNumKeyHeads = "linear_num_key_heads"
        case linearKeyHeadDim = "linear_key_head_dim"
        case linearValueHeadDim = "linear_value_head_dim"
        case linearConvKernelDim = "linear_conv_kernel_dim"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case ropeTheta = "rope_theta"
        case partialRotaryFactor = "partial_rotary_factor"
        case maxPositionEmbeddings = "max_position_embeddings"
        case tieWordEmbeddings = "tie_word_embeddings"
        case attentionBias = "attention_bias"
        case headDim = "head_dim"
        case ropeScaling = "rope_scaling"
        case fullAttentionInterval = "full_attention_interval"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        self.modelType =
            try container.decodeIfPresent(String.self, forKey: .modelType) ?? "qwen3_5_text"
        self.hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        self.hiddenLayers = try container.decode(Int.self, forKey: .hiddenLayers)
        self.intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)
        self.attentionHeads = try container.decode(Int.self, forKey: .attentionHeads)
        self.kvHeads = try container.decode(Int.self, forKey: .kvHeads)
        self.linearNumValueHeads = try container.decode(Int.self, forKey: .linearNumValueHeads)
        self.linearNumKeyHeads = try container.decode(Int.self, forKey: .linearNumKeyHeads)
        self.linearKeyHeadDim = try container.decode(Int.self, forKey: .linearKeyHeadDim)
        self.linearValueHeadDim = try container.decode(Int.self, forKey: .linearValueHeadDim)
        self.linearConvKernelDim = try container.decode(Int.self, forKey: .linearConvKernelDim)
        self.rmsNormEps = try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        self.vocabularySize = try container.decode(Int.self, forKey: .vocabularySize)
        self.ropeTheta =
            try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10000
        self.partialRotaryFactor =
            try container.decodeIfPresent(Float.self, forKey: .partialRotaryFactor) ?? 0.25
        self.maxPositionEmbeddings =
            try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 32768
        self.tieWordEmbeddings =
            try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        self.attentionBias =
            try container.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        self.headDim = try container.decodeIfPresent(Int.self, forKey: .headDim)
        self.ropeScaling = try container.decodeIfPresent(
            [String: StringOrNumber].self, forKey: .ropeScaling)
        self.fullAttentionInterval =
            try container.decodeIfPresent(Int.self, forKey: .fullAttentionInterval) ?? 4
    }
}

final class Qwen3_5GatedDeltaNet: Module {
    let hiddenSize: Int
    let numVHeads: Int
    let numKHeads: Int
    let headKDim: Int
    let headVDim: Int
    let keyDim: Int
    let valueDim: Int
    let convKernelSize: Int
    let convDim: Int

    @ModuleInfo(key: "conv1d") var conv1d: Conv1d
    @ModuleInfo(key: "in_proj_qkv") var inProjQKV: Linear
    @ModuleInfo(key: "in_proj_z") var inProjZ: Linear
    @ModuleInfo(key: "in_proj_b") var inProjB: Linear
    @ModuleInfo(key: "in_proj_a") var inProjA: Linear

    @ParameterInfo(key: "dt_bias") var dtBias: MLXArray
    @ParameterInfo(key: "A_log") var aLog: MLXArray

    @ModuleInfo(key: "norm") var norm: Qwen3NextRMSNormGated
    @ModuleInfo(key: "out_proj") var outProj: Linear

    init(_ args: Qwen3_5TextConfiguration) {
        self.hiddenSize = args.hiddenSize
        self.numVHeads = args.linearNumValueHeads
        self.numKHeads = args.linearNumKeyHeads
        self.headKDim = args.linearKeyHeadDim
        self.headVDim = args.linearValueHeadDim
        self.keyDim = headKDim * numKHeads
        self.valueDim = headVDim * numVHeads
        self.convKernelSize = args.linearConvKernelDim
        self.convDim = keyDim * 2 + valueDim

        precondition(
            numVHeads % numKHeads == 0,
            "num_v_heads (\(numVHeads)) must be divisible by num_k_heads (\(numKHeads))")

        _conv1d.wrappedValue = Conv1d(
            inputChannels: convDim,
            outputChannels: convDim,
            kernelSize: convKernelSize,
            stride: 1,
            padding: 0,
            dilation: 1,
            groups: convDim,
            bias: false
        )

        _inProjQKV.wrappedValue = Linear(hiddenSize, keyDim * 2 + valueDim, bias: false)
        _inProjZ.wrappedValue = Linear(hiddenSize, valueDim, bias: false)
        _inProjB.wrappedValue = Linear(hiddenSize, numVHeads, bias: false)
        _inProjA.wrappedValue = Linear(hiddenSize, numVHeads, bias: false)

        _dtBias.wrappedValue = MLXArray.ones([numVHeads])
        let a = MLXRandom.uniform(low: 0, high: 16, [numVHeads])
        _aLog.wrappedValue = log(a)

        _norm.wrappedValue = Qwen3NextRMSNormGated(dimensions: headVDim, eps: args.rmsNormEps)
        _outProj.wrappedValue = Linear(valueDim, hiddenSize, bias: false)

        super.init()
    }

    func callAsFunction(
        _ inputs: MLXArray,
        mask: MLXArray? = nil,
        cache: MambaCache? = nil
    ) -> MLXArray {
        let B = inputs.dim(0)
        let S = inputs.dim(1)

        var qkv = inProjQKV(inputs)
        let z = inProjZ(inputs).reshaped(B, S, numVHeads, headVDim)
        let b = inProjB(inputs)
        let a = inProjA(inputs)

        let convState: MLXArray
        if let cacheState = cache?[0] {
            convState = cacheState
        } else {
            convState = MLXArray.zeros([B, convKernelSize - 1, convDim], dtype: inputs.dtype)
        }

        if let mask {
            qkv = MLX.where(mask[.ellipsis, .newAxis], qkv, 0)
        }

        let convInput = concatenated([convState, qkv], axis: 1)
        if let cache {
            cache[0] = convInput[0..., (-(convKernelSize - 1))...]
        }

        let convOut = silu(conv1d(convInput))

        let convSplit = MLX.split(convOut, indices: [keyDim, 2 * keyDim], axis: -1)
        var q = convSplit[0].reshaped(B, S, numKHeads, headKDim)
        var k = convSplit[1].reshaped(B, S, numKHeads, headKDim)
        let v = convSplit[2].reshaped(B, S, numVHeads, headVDim)

        var state = cache?[1]
        let invScale = pow(Float(headKDim), -0.5)
        q = pow(invScale, 2) * MLXFast.rmsNorm(q, weight: MLXArray.mlxNone, eps: 1e-6)
        k = invScale * MLXFast.rmsNorm(k, weight: MLXArray.mlxNone, eps: 1e-6)

        var out: MLXArray

        (out, state) = gatedDeltaUpdate(
            q: q,
            k: k,
            v: v,
            a: a,
            b: b,
            aLog: aLog,
            dtBias: dtBias,
            state: state,
            mask: mask
        )

        if let cache {
            cache[1] = state
        }

        out = norm(out, gate: z)
        return outProj(out.reshaped(B, S, -1))
    }
}

final class Qwen3_5Attention: Module {
    let attentionHeads: Int
    let kvHeads: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    let rope: OffsetLayer

    init(_ args: Qwen3_5TextConfiguration) {
        let headDim = args.headDim ?? (args.hiddenSize / args.attentionHeads)
        self.attentionHeads = args.attentionHeads
        self.kvHeads = args.kvHeads
        self.scale = pow(Float(headDim), -0.5)

        _qProj.wrappedValue = Linear(
            args.hiddenSize, args.attentionHeads * headDim * 2, bias: args.attentionBias)
        _kProj.wrappedValue = Linear(
            args.hiddenSize, args.kvHeads * headDim, bias: args.attentionBias)
        _vProj.wrappedValue = Linear(
            args.hiddenSize, args.kvHeads * headDim, bias: args.attentionBias)
        _oProj.wrappedValue = Linear(
            args.attentionHeads * headDim, args.hiddenSize, bias: args.attentionBias)

        _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.rmsNormEps)
        _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.rmsNormEps)

        let ropeDims = Int(Float(headDim) * args.partialRotaryFactor)
        self.rope = initializeRope(
            dims: max(1, ropeDims),
            base: args.ropeTheta,
            traditional: false,
            scalingConfig: args.ropeScaling,
            maxPositionEmbeddings: args.maxPositionEmbeddings
        )

        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let B = x.dim(0)
        let L = x.dim(1)

        let qProjOutput = qProj(x)
        let qSplit = qProjOutput.reshaped(B, L, attentionHeads, -1).split(parts: 2, axis: -1)
        var queries = qSplit[0]
        let gate = qSplit[1].reshaped(B, L, -1)

        var keys = kProj(x)
        var values = vProj(x)

        queries = qNorm(queries).transposed(0, 2, 1, 3)
        keys = kNorm(keys.reshaped(B, L, kvHeads, -1)).transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, kvHeads, -1).transposed(0, 2, 1, 3)

        if let cache {
            queries = rope(queries, offset: cache.offset)
            keys = rope(keys, offset: cache.offset)
        } else {
            queries = rope(queries, offset: 0)
            keys = rope(keys, offset: 0)
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

        return oProj(sigmoidMultiply(output, gate))
    }
}

final class Qwen3_5DecoderLayer: Module {
    let isLinear: Bool

    @ModuleInfo(key: "self_attn") var selfAttn: Qwen3_5Attention?
    @ModuleInfo(key: "linear_attn") var linearAttn: Qwen3_5GatedDeltaNet?

    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    @ModuleInfo(key: "mlp") var mlp: Qwen3NextMLP

    init(_ args: Qwen3_5TextConfiguration, layerIdx: Int) {
        self.isLinear = (layerIdx + 1) % args.fullAttentionInterval != 0

        if isLinear {
            _linearAttn.wrappedValue = Qwen3_5GatedDeltaNet(args)
        } else {
            _selfAttn.wrappedValue = Qwen3_5Attention(args)
        }

        _mlp.wrappedValue = Qwen3NextMLP(
            dimensions: args.hiddenSize,
            hiddenDimensions: args.intermediateSize
        )
        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize,
            eps: args.rmsNormEps
        )
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize,
            eps: args.rmsNormEps
        )

        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        attentionMask: MLXFast.ScaledDotProductAttentionMaskMode,
        ssmMask: MLXArray?,
        cache: KVCache?
    ) -> MLXArray {
        let h: MLXArray
        if isLinear {
            h = linearAttn!(inputLayerNorm(x), mask: ssmMask, cache: cache as? MambaCache)
        } else {
            h = selfAttn!(inputLayerNorm(x), mask: attentionMask, cache: cache)
        }

        let r = x + h
        let out = r + mlp(postAttentionLayerNorm(r))

        return out
    }
}

public class Qwen3_5TextModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    let layers: [Qwen3_5DecoderLayer]
    let norm: RMSNorm

    let ssmIdx: Int
    let faIdx: Int

    init(_ args: Qwen3_5TextConfiguration) {
        precondition(args.vocabularySize > 0)

        _embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize,
            dimensions: args.hiddenSize
        )

        self.layers = (0 ..< args.hiddenLayers).map { layerIdx in
            Qwen3_5DecoderLayer(args, layerIdx: layerIdx)
        }

        self.norm = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)

        self.ssmIdx = 0
        self.faIdx = args.fullAttentionInterval - 1

        super.init()
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache?]? = nil) -> MLXArray {
        var hiddenStates = embedTokens(inputs)

        var cacheArray = cache
        if cacheArray == nil {
            cacheArray = Array(repeating: nil as KVCache?, count: layers.count)
        }

        let faMask = createAttentionMask(h: hiddenStates, cache: cacheArray?[faIdx])
        let ssmMask = createSSMMask(h: hiddenStates, cache: cacheArray?[ssmIdx] as? MambaCache)

        for (i, layer) in layers.enumerated() {
            let mask = layer.isLinear ? ssmMask : nil
            let attnMask =
                layer.isLinear
                ? MLXFast.ScaledDotProductAttentionMaskMode.none : faMask
            hiddenStates = layer(
                hiddenStates, attentionMask: attnMask, ssmMask: mask, cache: cacheArray?[i])
        }

        return norm(hiddenStates)
    }
}

public class Qwen3_5TextModel: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    public let model: Qwen3_5TextModelInner
    let configuration: Qwen3_5TextConfiguration

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ args: Qwen3_5TextConfiguration) {
        self.configuration = args
        self.vocabularySize = args.vocabularySize
        self.kvHeads = (0 ..< args.hiddenLayers).map { _ in args.kvHeads }
        self.model = Qwen3_5TextModelInner(args)

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
        return model.layers.map { layer in
            if layer.isLinear {
                return MambaCache()
            }
            return KVCacheSimple()
        }
    }

    public func makeCache() -> [KVCache] {
        return newCache(parameters: nil)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitizedWeights = weights

        if configuration.tieWordEmbeddings {
            sanitizedWeights["lm_head.weight"] = nil
        }

        return sanitizedWeights
    }
}

extension Qwen3_5TextModel: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}
