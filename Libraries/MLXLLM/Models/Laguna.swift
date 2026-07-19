// Copyright © 2025 Apple Inc.

// port of https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/laguna.py
//
// Laguna (Poolside Laguna XS 2.1): a Mixture-of-Experts decoder with GQA,
// per-head QK-norm, per-head softplus attention output gating, a mix of
// sliding-window and full-attention layers (each with its own RoPE), a sigmoid
// top-k router with an `e_score_correction_bias`, and a shared expert. The
// released checkpoints are NVFP4 (4-bit) quantized on the expert / shared-expert
// projections; every other projection stays full precision. Quantization mode,
// group size and bits are read from `config.json` and applied by the loader to
// any module that ships a matching `.scales` tensor, so no model-side handling
// of the quantization format is required here.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Attention

private class LagunaAttention: Module {
    let nHeads: Int
    let nKVHeads: Int
    let headDim: Int
    let scale: Float
    let gatingEnabled: Bool
    let gatePerHead: Bool
    let isSliding: Bool

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear
    @ModuleInfo(key: "g_proj") var gProj: Linear?

    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    let rope: RoPELayer

    init(_ config: LagunaConfiguration, layerIdx: Int) {
        let dim = config.hiddenSize
        self.nHeads = config.heads(forLayer: layerIdx)
        self.nKVHeads = config.numKeyValueHeads
        self.headDim = config.headDim
        self.scale = pow(Float(headDim), -0.5)
        self.gatingEnabled = config.gatingEnabled
        self.gatePerHead = config.gatePerHead

        let layerType = config.layerType(forLayer: layerIdx)
        self.isSliding = layerType == "sliding_attention"

        self._wq.wrappedValue = Linear(dim, nHeads * headDim, bias: config.qkvBias)
        self._wk.wrappedValue = Linear(dim, nKVHeads * headDim, bias: config.qkvBias)
        self._wv.wrappedValue = Linear(dim, nKVHeads * headDim, bias: config.qkvBias)
        self._wo.wrappedValue = Linear(nHeads * headDim, dim, bias: config.attentionBias)

        if gatingEnabled {
            let gateDim = gatePerHead ? nHeads : nHeads * headDim
            self._gProj.wrappedValue = Linear(dim, gateDim, bias: false)
        }

        self._qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        self._kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)

        // Per-layer-type RoPE: full-attention layers use YaRN with a partial
        // rotary factor, sliding-attention layers use plain RoPE over the full
        // head. The base and partial factor come from the per-type sub-dict.
        let ropeConfig = config.ropeParameters(forLayer: layerIdx)
        let base = ropeConfig?["rope_theta"]?.asFloat() ?? config.ropeTheta
        let partial = ropeConfig?["partial_rotary_factor"]?.asFloat() ?? 1.0
        let ropeDims = Int(Float(headDim) * partial)
        self.rope = initializeRope(
            dims: ropeDims,
            base: base,
            traditional: false,
            scalingConfig: ropeConfig,
            maxPositionEmbeddings: config.maxPositionEmbeddings
        )

        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        var queries = wq(x)
        var keys = wk(x)
        var values = wv(x)

        queries = qNorm(queries.reshaped(B, L, nHeads, headDim)).transposed(0, 2, 1, 3)
        keys = kNorm(keys.reshaped(B, L, nKVHeads, headDim)).transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, nKVHeads, headDim).transposed(0, 2, 1, 3)

        let offset = cache?.ropeOffset
        queries = applyRotaryPosition(rope, to: queries, offset: offset)
        keys = applyRotaryPosition(rope, to: keys, offset: offset)

        var output = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: cache,
            scale: scale,
            mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        if gatingEnabled, let gProj {
            // Per-head softplus gate computed in float32, then broadcast across
            // the head dimension (or applied elementwise for a per-element gate).
            let gate = softplus(gProj(x).asType(.float32)).asType(output.dtype)
            if gatePerHead {
                output =
                    (output.reshaped(B, L, nHeads, headDim) * gate[.ellipsis, .newAxis])
                    .reshaped(B, L, -1)
            } else {
                output = output * gate
            }
        }

        return wo(output)
    }
}

// MARK: - Dense MLP (also used as the shared expert)

private class LagunaMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(dimensions: Int, hiddenDimensions: Int) {
        self._gateProj.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        self._upProj.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        self._downProj.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}

// MARK: - MoE

private class LagunaMoEGate: Module {
    let topK: Int
    let normTopkProb: Bool
    let routerLogitSoftcapping: Float

    var weight: MLXArray
    var e_score_correction_bias: MLXArray

    init(_ config: LagunaConfiguration) {
        self.topK = config.numExpertsPerTok
        self.normTopkProb = config.normTopkProb
        self.routerLogitSoftcapping = config.moeRouterLogitSoftcapping
        self.weight = zeros([config.numExperts, config.hiddenSize])
        self.e_score_correction_bias = zeros([config.numExperts])
    }

    func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray) {
        var logits = x.matmul(weight.T).asType(.float32)
        if routerLogitSoftcapping > 0 {
            logits = tanh(logits / routerLogitSoftcapping) * routerLogitSoftcapping
        }

        let scores = sigmoid(logits)
        let scoresForChoice = scores + e_score_correction_bias.asType(scores.dtype)

        let inds = argPartition(-scoresForChoice, kth: topK - 1, axis: -1)[.ellipsis, ..<topK]
        var weights = takeAlong(scores, inds, axis: -1)
        if normTopkProb {
            weights = weights / weights.sum(axis: -1, keepDims: true)
        }
        return (inds, weights)
    }
}

private class LagunaSparseMoeBlock: Module, UnaryLayer {
    let routedScalingFactor: Float

    @ModuleInfo(key: "gate") var gate: LagunaMoEGate
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
    @ModuleInfo(key: "shared_expert") var sharedExpert: LagunaMLP

    init(_ config: LagunaConfiguration) {
        self.routedScalingFactor = config.moeRoutedScalingFactor
        self._gate.wrappedValue = LagunaMoEGate(config)
        self._switchMLP.wrappedValue = SwitchGLU(
            inputDims: config.hiddenSize,
            hiddenDims: config.moeIntermediateSize,
            numExperts: config.numExperts
        )
        self._sharedExpert.wrappedValue = LagunaMLP(
            dimensions: config.hiddenSize,
            hiddenDimensions: config.sharedExpertIntermediateSize
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (inds, weights) = gate(x)
        var y = switchMLP(x, inds)
        y = weightedExpertSum(y, weights.asType(y.dtype))
        if routedScalingFactor != 1 {
            y = y * routedScalingFactor
        }
        return y + sharedExpert(x)
    }
}

// MARK: - Decoder Layer

private class LagunaDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: LagunaAttention
    let mlp: UnaryLayer
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    let attentionType: String

    init(_ config: LagunaConfiguration, layerIdx: Int) {
        self._selfAttn.wrappedValue = LagunaAttention(config, layerIdx: layerIdx)

        if config.isSparse(layer: layerIdx) {
            self.mlp = LagunaSparseMoeBlock(config)
        } else {
            self.mlp = LagunaMLP(
                dimensions: config.hiddenSize, hiddenDimensions: config.intermediateSize)
        }

        self._inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)

        self.attentionType = config.layerType(forLayer: layerIdx)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let r = selfAttn(inputLayerNorm(x), mask: mask, cache: cache)
        let h = x + r
        let r2 = mlp(postAttentionLayerNorm(h))
        return h + r2
    }
}

// MARK: - Model

private class LagunaModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    fileprivate let layers: [LagunaDecoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    let layerTypes: [String]
    let slidingWindow: Int
    let fullAttentionIdx: Int
    let slidingAttentionIdx: Int

    init(_ config: LagunaConfiguration) {
        precondition(config.vocabSize > 0)

        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabSize, dimensions: config.hiddenSize)

        self.layers = (0 ..< config.numHiddenLayers).map {
            LagunaDecoderLayer(config, layerIdx: $0)
        }
        self._norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)

        self.layerTypes = (0 ..< config.numHiddenLayers).map { config.layerType(forLayer: $0) }
        self.slidingWindow = config.slidingWindow
        self.fullAttentionIdx = layerTypes.firstIndex(of: "full_attention") ?? 0
        self.slidingAttentionIdx = layerTypes.firstIndex(of: "sliding_attention") ?? 0
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var h = embedTokens(inputs)

        let fullMask = createAttentionMask(h: h, cache: cache?[fullAttentionIdx])
        let slidingMask = createAttentionMask(
            h: h, cache: cache?[slidingAttentionIdx], windowSize: slidingWindow)

        for (i, layer) in layers.enumerated() {
            let mask = layerTypes[i] == "full_attention" ? fullMask : slidingMask
            h = layer(h, mask: mask, cache: cache?[i])
        }

        return norm(h)
    }
}

public class LagunaModel: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    fileprivate let model: LagunaModelInner
    let config: LagunaConfiguration

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ config: LagunaConfiguration) {
        self.config = config
        self.vocabularySize = config.vocabSize
        self.kvHeads = (0 ..< config.numHiddenLayers).map { _ in config.numKeyValueHeads }
        self.model = LagunaModelInner(config)
        if !config.tieWordEmbeddings {
            self._lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabSize, bias: false)
        }
        super.init()

        // Checkpoints quantize only switch_mlp/shared_expert; attention, router,
        // embeddings, lm_head and the dense layer-0 MLP stay full precision. Pre-
        // quantize per layer so the loader quantize(model:) pass finds nothing to
        // convert and never descends a decoder layer with no quantized submodule.
        if let groupSize = config.quantGroupSize, let bits = config.quantBits {
            let mode = config.quantMode ?? .affine
            for layer in model.layers {
                quantize(model: layer) { path, _ in
                    if path.contains("switch_mlp") || path.contains("shared_expert") {
                        return (groupSize: groupSize, bits: bits, mode: mode)
                    }
                    return nil
                }
            }
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        let out = model(inputs, cache: cache)
        if let lmHead {
            return lmHead(out)
        } else {
            return model.embedTokens.asLinear(out)
        }
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var weights = weights
        if config.tieWordEmbeddings {
            weights["lm_head.weight"] = nil
        }
        // Drop precomputed rotary tables if a checkpoint ships them.
        return weights.filter { !$0.key.contains("rotary_emb.inv_freq") }
    }

    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        (0 ..< config.numHiddenLayers).map { i in
            config.layerType(forLayer: i) == "full_attention"
                ? KVCacheSimple()
                : RotatingKVCache(maxSize: config.slidingWindow)
        }
    }
}

// MARK: - LoRA

extension LagunaModel: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}

// MARK: - Configuration

/// Attention output gating mode. In `config.json` this is either a bool
/// (`true` enables per-head gating) or a string (`"per-head"` / `"per-element"`).
public enum LagunaGating: Codable, Sendable {
    case disabled
    case perHead
    case perElement

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let flag = try? container.decode(Bool.self) {
            self = flag ? .perHead : .disabled
        } else if let value = try? container.decode(String.self) {
            switch value {
            case "per-element", "per_element": self = .perElement
            case "false", "none", "": self = .disabled
            default: self = .perHead
            }
        } else {
            self = .perHead
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .disabled: try container.encode(false)
        case .perHead: try container.encode("per-head")
        case .perElement: try container.encode("per-element")
        }
    }

    var enabled: Bool { self != .disabled }
    var perHead: Bool { self == .perHead }
}

private struct LagunaQuantizationBlock: Decodable {
    let groupSize: Int
    let bits: Int
    let mode: QuantizationMode?
    enum CodingKeys: String, CodingKey {
        case groupSize = "group_size"
        case bits
        case mode
    }
    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.groupSize = try c.decode(Int.self, forKey: .groupSize)
        self.bits = try c.decode(Int.self, forKey: .bits)
        if let m = try c.decodeIfPresent(String.self, forKey: .mode) {
            self.mode = QuantizationMode(rawValue: m)
        } else {
            self.mode = nil
        }
    }
}

private enum LagunaQuantizationCodingKeys: String, CodingKey {
    case quantization
}

public struct LagunaConfiguration: Codable, Sendable {
    var vocabSize: Int
    var hiddenSize: Int
    var intermediateSize: Int
    var numHiddenLayers: Int
    var numAttentionHeads: Int
    var numKeyValueHeads: Int
    var headDim: Int
    var maxPositionEmbeddings: Int
    var rmsNormEps: Float
    var attentionBias: Bool
    var qkvBias: Bool
    var tieWordEmbeddings: Bool
    var ropeTheta: Float
    var slidingWindow: Int

    var layerTypes: [String]?
    var numAttentionHeadsPerLayer: [Int]?
    var mlpLayerTypes: [String]?
    var mlpOnlyLayers: [Int]
    var ropeParametersByType: [String: [String: StringOrNumber]]?

    var gating: LagunaGating

    // MoE
    var numExperts: Int
    var numExpertsPerTok: Int
    var moeIntermediateSize: Int
    var sharedExpertIntermediateSize: Int
    var moeRoutedScalingFactor: Float
    var normTopkProb: Bool
    var decoderSparseStep: Int
    var moeRouterLogitSoftcapping: Float
    var quantGroupSize: Int?
    var quantBits: Int?
    var quantMode: QuantizationMode?

    var gatingEnabled: Bool { gating.enabled }
    var gatePerHead: Bool { gating.perHead }

    func layerType(forLayer i: Int) -> String {
        if let layerTypes, i < layerTypes.count { return layerTypes[i] }
        return "full_attention"
    }

    func heads(forLayer i: Int) -> Int {
        if let numAttentionHeadsPerLayer, i < numAttentionHeadsPerLayer.count {
            return numAttentionHeadsPerLayer[i]
        }
        return numAttentionHeads
    }

    func isSparse(layer i: Int) -> Bool {
        if let mlpLayerTypes, i < mlpLayerTypes.count {
            return mlpLayerTypes[i] == "sparse"
        }
        if mlpOnlyLayers.contains(i) { return false }
        return numExperts > 0 && (i + 1) % max(decoderSparseStep, 1) == 0
    }

    /// RoPE parameters for a layer, resolved from the per-type mapping when present.
    func ropeParameters(forLayer i: Int) -> [String: StringOrNumber]? {
        guard let ropeParametersByType else { return nil }
        return ropeParametersByType[layerType(forLayer: i)]
    }

    enum CodingKeys: String, CodingKey {
        case vocabSize = "vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case maxPositionEmbeddings = "max_position_embeddings"
        case rmsNormEps = "rms_norm_eps"
        case attentionBias = "attention_bias"
        case qkvBias = "qkv_bias"
        case tieWordEmbeddings = "tie_word_embeddings"
        case ropeTheta = "rope_theta"
        case slidingWindow = "sliding_window"
        case layerTypes = "layer_types"
        case numAttentionHeadsPerLayer = "num_attention_heads_per_layer"
        case mlpLayerTypes = "mlp_layer_types"
        case mlpOnlyLayers = "mlp_only_layers"
        case ropeParametersByType = "rope_parameters"
        case gating
        case numExperts = "num_experts"
        case numExpertsPerTok = "num_experts_per_tok"
        case moeIntermediateSize = "moe_intermediate_size"
        case sharedExpertIntermediateSize = "shared_expert_intermediate_size"
        case moeRoutedScalingFactor = "moe_routed_scaling_factor"
        case normTopkProb = "norm_topk_prob"
        case decoderSparseStep = "decoder_sparse_step"
        case moeRouterLogitSoftcapping = "moe_router_logit_softcapping"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)

        self.vocabSize = try c.decodeIfPresent(Int.self, forKey: .vocabSize) ?? 100352
        self.hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 2048
        self.intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 8192
        self.numHiddenLayers = try c.decodeIfPresent(Int.self, forKey: .numHiddenLayers) ?? 40
        self.numAttentionHeads = try c.decodeIfPresent(Int.self, forKey: .numAttentionHeads) ?? 48
        self.numKeyValueHeads = try c.decodeIfPresent(Int.self, forKey: .numKeyValueHeads) ?? 8
        self.headDim = try c.decodeIfPresent(Int.self, forKey: .headDim) ?? 128
        self.maxPositionEmbeddings =
            try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 262144
        self.rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        self.attentionBias = try c.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        self.qkvBias = try c.decodeIfPresent(Bool.self, forKey: .qkvBias) ?? false
        self.tieWordEmbeddings =
            try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        self.ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10000
        self.slidingWindow = try c.decodeIfPresent(Int.self, forKey: .slidingWindow) ?? 512

        self.layerTypes = try c.decodeIfPresent([String].self, forKey: .layerTypes)
        self.numAttentionHeadsPerLayer =
            try c.decodeIfPresent([Int].self, forKey: .numAttentionHeadsPerLayer)
        self.mlpLayerTypes = try c.decodeIfPresent([String].self, forKey: .mlpLayerTypes)
        self.mlpOnlyLayers = try c.decodeIfPresent([Int].self, forKey: .mlpOnlyLayers) ?? [0]
        self.ropeParametersByType =
            try c.decodeIfPresent([String: [String: StringOrNumber]].self, forKey: .ropeParametersByType)

        self.gating = try c.decodeIfPresent(LagunaGating.self, forKey: .gating) ?? .perHead

        self.numExperts = try c.decodeIfPresent(Int.self, forKey: .numExperts) ?? 0
        self.numExpertsPerTok = try c.decodeIfPresent(Int.self, forKey: .numExpertsPerTok) ?? 8
        self.moeIntermediateSize =
            try c.decodeIfPresent(Int.self, forKey: .moeIntermediateSize) ?? 512
        self.sharedExpertIntermediateSize =
            try c.decodeIfPresent(Int.self, forKey: .sharedExpertIntermediateSize) ?? 512
        self.moeRoutedScalingFactor =
            try c.decodeIfPresent(Float.self, forKey: .moeRoutedScalingFactor) ?? 1.0
        self.normTopkProb = try c.decodeIfPresent(Bool.self, forKey: .normTopkProb) ?? true
        self.decoderSparseStep = try c.decodeIfPresent(Int.self, forKey: .decoderSparseStep) ?? 1
        self.moeRouterLogitSoftcapping =
            try c.decodeIfPresent(Float.self, forKey: .moeRouterLogitSoftcapping) ?? 0.0

        if let q = try decoder.container(keyedBy: LagunaQuantizationCodingKeys.self)
            .decodeIfPresent(LagunaQuantizationBlock.self, forKey: .quantization) {
            self.quantGroupSize = q.groupSize
            self.quantBits = q.bits
            self.quantMode = q.mode
        } else {
            self.quantGroupSize = nil
            self.quantBits = nil
            self.quantMode = nil
        }
    }
}
