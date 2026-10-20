// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// Port of https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/mellum.py

// MARK: - Attention

final class MellumAttention: Module {
    let attentionHeads: Int
    let kvHeads: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    let rope: RoPELayer

    init(_ args: MellumConfiguration, kind: MellumConfiguration.AttentionKind) {
        let dim = args.hiddenSize
        let headDim = args.headDim
        self.attentionHeads = args.attentionHeads
        self.kvHeads = args.kvHeads
        self.scale = pow(Float(headDim), -0.5)

        _qProj.wrappedValue = Linear(dim, attentionHeads * headDim, bias: args.attentionBias)
        _kProj.wrappedValue = Linear(dim, kvHeads * headDim, bias: args.attentionBias)
        _vProj.wrappedValue = Linear(dim, kvHeads * headDim, bias: args.attentionBias)
        _oProj.wrappedValue = Linear(attentionHeads * headDim, dim, bias: args.attentionBias)

        _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.rmsNormEps)
        _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.rmsNormEps)

        let rope = args.rope(for: kind)
        self.rope = initializeRope(
            dims: headDim,
            base: rope.theta,
            traditional: false,
            scalingConfig: rope.scaling,
            maxPositionEmbeddings: args.maxPositionEmbeddings)
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
            queries: queries, keys: keys, values: values, cache: cache, scale: scale, mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        return oProj(output)
    }
}

// MARK: - Feed-forward

final class MellumMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(dimensions: Int, hiddenDimensions: Int) {
        _gateProj.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        _upProj.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        _downProj.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}

final class MellumSparseMoeBlock: Module, UnaryLayer {
    let topK: Int
    let normTopkProb: Bool

    @ModuleInfo(key: "gate") var gate: Linear
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU

    init(_ args: MellumConfiguration) {
        self.topK = args.numExpertsPerToken
        self.normTopkProb = args.normTopkProb

        _gate.wrappedValue = Linear(args.hiddenSize, args.numExperts, bias: false)
        _switchMLP.wrappedValue = SwitchGLU(
            inputDims: args.hiddenSize,
            hiddenDims: args.moeIntermediateSize,
            numExperts: args.numExperts)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let probabilities = MLX.softmax(gate(x), axis: -1, precise: true)
        let (indices, scores) = moeRouterTopK(probabilities, k: topK, normalize: normTopkProb)

        let tokens = x.size / x.dim(-1)
        return switchMLP.callAndWeightedReduce(
            x.reshaped(tokens, -1),
            indices.reshaped(tokens, topK),
            weights: scores.reshaped(tokens, topK),
            fuseSortedReduction: true
        ).reshaped(x.shape)
    }
}

// MARK: - Decoder

final class MellumDecoderLayer: Module {
    let kind: MellumConfiguration.AttentionKind

    @ModuleInfo(key: "self_attn") var selfAttn: MellumAttention
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    let mlp: UnaryLayer

    init(_ args: MellumConfiguration, layerIndex: Int) {
        self.kind = args.layerTypes[layerIndex]

        _selfAttn.wrappedValue = MellumAttention(args, kind: kind)
        _inputLayerNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)

        switch args.mlpLayerTypes[layerIndex] {
        case .sparse:
            self.mlp = MellumSparseMoeBlock(args)
        case .dense:
            self.mlp = MellumMLP(
                dimensions: args.hiddenSize, hiddenDimensions: args.intermediateSize)
        }
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let h = x + selfAttn(inputLayerNorm(x), mask: mask, cache: cache)
        return h + mlp(postAttentionLayerNorm(h))
    }
}

public final class MellumModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    let layers: [MellumDecoderLayer]
    let norm: RMSNorm
    let slidingWindow: Int?

    private let fullAttentionIndex: Int?
    private let slidingAttentionIndex: Int?

    init(_ args: MellumConfiguration) {
        precondition(args.vocabularySize > 0)

        _embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)
        self.layers = (0 ..< args.hiddenLayers).map { MellumDecoderLayer(args, layerIndex: $0) }
        self.norm = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
        self.slidingWindow = args.slidingWindow
        self.fullAttentionIndex = args.layerTypes.firstIndex(of: .full)
        self.slidingAttentionIndex = args.layerTypes.firstIndex(of: .sliding)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var h = embedTokens(inputs)

        let fullMask = createAttentionMask(
            h: h, cache: fullAttentionIndex.flatMap { cache?[$0] })
        let slidingMask = createAttentionMask(
            h: h, cache: slidingAttentionIndex.flatMap { cache?[$0] }, windowSize: slidingWindow)

        for (i, layer) in layers.enumerated() {
            let mask = layer.kind == .full ? fullMask : slidingMask
            h = layer(h, mask: mask, cache: cache?[i])
        }

        return norm(h)
    }
}

// MARK: - Model

public final class MellumModel: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    public let model: MellumModelInner
    let configuration: MellumConfiguration

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ args: MellumConfiguration) {
        self.configuration = args
        self.vocabularySize = args.vocabularySize
        self.kvHeads = Array(repeating: args.kvHeads, count: args.hiddenLayers)
        self.model = MellumModelInner(args)

        if !args.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabularySize, bias: false)
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        projectLogits(hiddenStates(inputs, cache: cache))
    }

    public func nextTokenLogits(
        _ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?
    ) -> LMOutput {
        let hidden = hiddenStates(input.tokens, cache: cache)
        let projection: Module = lmHead ?? model.embedTokens
        return .init(
            logits: projectLogits(
                quantizedVocabularyProjectionInput(hidden, projection: projection)))
    }

    public func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
        try configuration.layerTypes.map { kind in
            try makeHybridAttentionKVCache(
                parameters: parameters,
                slidingWindow: configuration.slidingWindow,
                usesSlidingWindow: kind == .sliding)
        }
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var weights = filterLMHeadWeights(
            from: weights, tiedWordEmbeddings: configuration.tieWordEmbeddings)

        // Stack per-expert Hugging Face projections into SwitchGLU weights.
        for layer in 0 ..< configuration.hiddenLayers {
            let prefix = "model.layers.\(layer).mlp"
            for projection in ["gate_proj", "up_proj", "down_proj"] {
                let experts = (0 ..< configuration.numExperts).compactMap {
                    weights.removeValue(forKey: "\(prefix).experts.\($0).\(projection).weight")
                }
                if !experts.isEmpty {
                    weights["\(prefix).switch_mlp.\(projection).weight"] = MLX.stacked(experts)
                }
            }
        }
        return weights
    }
}

extension MellumModel: HiddenStateLanguageModel {
    package func hiddenStates(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        model(inputs, cache: cache)
    }

    package func projectLogits(_ hiddenStates: MLXArray) -> MLXArray {
        lmHead?(hiddenStates) ?? model.embedTokens.asLinear(hiddenStates)
    }
}

// MARK: - Configuration

public struct MellumConfiguration: Codable, Sendable {
    public enum AttentionKind: String, Codable, Sendable {
        case full = "full_attention"
        case sliding = "sliding_attention"
    }

    public enum FeedForwardKind: String, Codable, Sendable {
        case dense
        case sparse
    }

    var hiddenSize: Int
    var hiddenLayers: Int
    var intermediateSize: Int
    var attentionHeads: Int
    var kvHeads: Int
    var headDim: Int
    var numExperts: Int
    var numExpertsPerToken: Int
    var moeIntermediateSize: Int
    var normTopkProb: Bool = true
    var rmsNormEps: Float = 1e-6
    var vocabularySize: Int
    var maxPositionEmbeddings: Int = 131_072
    var slidingWindow: Int?
    var attentionBias: Bool = false
    var tieWordEmbeddings: Bool = false
    var layerTypes: [AttentionKind]
    var mlpLayerTypes: [FeedForwardKind]
    var ropeParameters: [String: [String: StringOrNumber]]

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case numExperts = "num_experts"
        case numExpertsPerToken = "num_experts_per_tok"
        case moeIntermediateSize = "moe_intermediate_size"
        case normTopkProb = "norm_topk_prob"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case maxPositionEmbeddings = "max_position_embeddings"
        case slidingWindow = "sliding_window"
        case attentionBias = "attention_bias"
        case tieWordEmbeddings = "tie_word_embeddings"
        case layerTypes = "layer_types"
        case mlpLayerTypes = "mlp_layer_types"
        case ropeParameters = "rope_parameters"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        hiddenLayers = try container.decode(Int.self, forKey: .hiddenLayers)
        intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)
        attentionHeads = try container.decode(Int.self, forKey: .attentionHeads)
        kvHeads = try container.decode(Int.self, forKey: .kvHeads)
        headDim =
            try container.decodeIfPresent(Int.self, forKey: .headDim)
            ?? hiddenSize / attentionHeads
        numExperts = try container.decode(Int.self, forKey: .numExperts)
        numExpertsPerToken = try container.decode(Int.self, forKey: .numExpertsPerToken)
        moeIntermediateSize = try container.decode(Int.self, forKey: .moeIntermediateSize)
        vocabularySize = try container.decode(Int.self, forKey: .vocabularySize)
        slidingWindow = try container.decodeIfPresent(Int.self, forKey: .slidingWindow)

        if let value = try container.decodeIfPresent(Bool.self, forKey: .normTopkProb) {
            normTopkProb = value
        }
        if let value = try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) {
            rmsNormEps = value
        }
        if let value = try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) {
            maxPositionEmbeddings = value
        }
        if let value = try container.decodeIfPresent(Bool.self, forKey: .attentionBias) {
            attentionBias = value
        }
        if let value = try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) {
            tieWordEmbeddings = value
        }

        layerTypes =
            try container.decodeIfPresent([AttentionKind].self, forKey: .layerTypes)
            ?? Array(repeating: .full, count: hiddenLayers)
        mlpLayerTypes =
            try container.decodeIfPresent([FeedForwardKind].self, forKey: .mlpLayerTypes)
            ?? Array(repeating: .sparse, count: hiddenLayers)
        ropeParameters =
            try container.decodeIfPresent(
                [String: [String: StringOrNumber]].self, forKey: .ropeParameters)
            ?? [
                AttentionKind.full.rawValue: ["rope_theta": .float(500_000)],
                AttentionKind.sliding.rawValue: ["rope_theta": .float(10_000)],
            ]
    }

    /// The RoPE base and scaling of one attention kind.
    func rope(for kind: AttentionKind) -> (theta: Float, scaling: [String: StringOrNumber]?) {
        let parameters = ropeParameters[kind.rawValue]
        return (parameters?["rope_theta"]?.asFloat() ?? 10_000, parameters)
    }
}

extension MellumConfiguration: ModelConfigurationValidating {
    public func validateModelConfiguration() throws {
        guard layerTypes.count == hiddenLayers, mlpLayerTypes.count == hiddenLayers else {
            throw ModelFactoryError.invalidConfiguration(
                "layer_types and mlp_layer_types need \(hiddenLayers) entries, got "
                    + "\(layerTypes.count) and \(mlpLayerTypes.count)")
        }
        guard attentionHeads % kvHeads == 0 else {
            throw ModelFactoryError.invalidConfiguration(
                "num_attention_heads (\(attentionHeads)) must be a multiple of "
                    + "num_key_value_heads (\(kvHeads))")
        }
        guard
            !mlpLayerTypes.contains(.sparse)
                || (numExpertsPerToken > 0 && numExpertsPerToken <= numExperts)
        else {
            throw ModelFactoryError.invalidConfiguration(
                "num_experts_per_tok (\(numExpertsPerToken)) must be in 1...\(numExperts)")
        }
        for kind in Set(layerTypes) {
            guard let parameters = ropeParameters[kind.rawValue] else {
                throw ModelFactoryError.invalidConfiguration(
                    "rope_parameters has no entry for \(kind.rawValue)")
            }
            try validateRoPEConfiguration(
                parameters, context: "MellumConfiguration.rope_parameters.\(kind.rawValue)")
        }
    }
}

// MARK: - LoRA

extension MellumModel: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}

// MARK: - Chat conventions

extension MellumModel {
    public var reasoningConfig: ReasoningConfig? { QwenReasoningProtocol.tagged }
}
