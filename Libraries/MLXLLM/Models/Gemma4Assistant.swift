// Copyright © 2025 Apple Inc.
//
// Port of https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma4_assistant/modeling_gemma4_assistant.py
//
// Gemma 4 MTP assistant model — a small 4-layer transformer (~78M params)
// used as a native drafter for Multi-Token Prediction speculative decoding.
//
// Architecture:
// 1. pre_projection: Linear(2 * backbone_hidden, drafter_hidden) — down-projects
//    [backbone_last_hidden, token_embedding] into drafter space
// 2. model: 4-layer Gemma4-style transformer with shared KV from backbone
// 3. post_projection: Linear(drafter_hidden, backbone_hidden) — up-projects back
// 4. lm_head or centroid masked_embedding: produces logits over vocabulary

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Configuration

public struct Gemma4AssistantConfiguration: Codable, Sendable {
    var modelType: String = "gemma4_assistant"
    var backboneHiddenSize: Int = 2560
    var numCentroids: Int = 2048
    var centroidIntermediateTopK: Int = 32
    var useOrderedEmbeddings: Bool = true
    var tieWordEmbeddings: Bool = true
    var textConfig: Gemma4AssistantTextConfiguration

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case backboneHiddenSize = "backbone_hidden_size"
        case numCentroids = "num_centroids"
        case centroidIntermediateTopK = "centroid_intermediate_top_k"
        case useOrderedEmbeddings = "use_ordered_embeddings"
        case tieWordEmbeddings = "tie_word_embeddings"
        case textConfig = "text_config"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.modelType =
            try container.decodeIfPresent(String.self, forKey: .modelType) ?? "gemma4_assistant"
        self.backboneHiddenSize =
            try container.decodeIfPresent(Int.self, forKey: .backboneHiddenSize) ?? 2560
        self.numCentroids =
            try container.decodeIfPresent(Int.self, forKey: .numCentroids) ?? 2048
        self.centroidIntermediateTopK =
            try container.decodeIfPresent(Int.self, forKey: .centroidIntermediateTopK) ?? 32
        self.useOrderedEmbeddings =
            try container.decodeIfPresent(Bool.self, forKey: .useOrderedEmbeddings) ?? true
        self.tieWordEmbeddings =
            try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? true
        self.textConfig = try container.decode(
            Gemma4AssistantTextConfiguration.self, forKey: .textConfig)
    }
}

public struct Gemma4AssistantTextConfiguration: Codable, Sendable {
    var modelType: String = "gemma4_text"
    var hiddenSize: Int = 256
    var numHiddenLayers: Int = 4
    var intermediateSize: Int = 2048
    var numAttentionHeads: Int = 4
    var headDim: Int = 256
    var globalHeadDim: Int = 512
    var rmsNormEps: Float = 1e-6
    var vocabSize: Int = 262144
    var numKeyValueHeads: Int = 2
    var numGlobalKeyValueHeads: Int?
    var numKvSharedLayers: Int = 4
    var slidingWindow: Int = 512
    var maxPositionEmbeddings: Int = 131072
    var attentionKeqV: Bool = false
    var layerTypes: [String] = [
        "sliding_attention", "sliding_attention", "sliding_attention", "full_attention",
    ]
    var tieWordEmbeddings: Bool = true
    var hiddenActivation: String = "gelu_pytorch_tanh"
    var ropeParameters: [String: [String: StringOrNumber]]?

    var slidingRopeTheta: Float = 10000.0
    var fullRopeTheta: Float = 1_000_000.0
    var fullPartialRotaryFactor: Float = 0.25

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case numAttentionHeads = "num_attention_heads"
        case headDim = "head_dim"
        case globalHeadDim = "global_head_dim"
        case rmsNormEps = "rms_norm_eps"
        case vocabSize = "vocab_size"
        case numKeyValueHeads = "num_key_value_heads"
        case numGlobalKeyValueHeads = "num_global_key_value_heads"
        case numKvSharedLayers = "num_kv_shared_layers"
        case slidingWindow = "sliding_window"
        case maxPositionEmbeddings = "max_position_embeddings"
        case attentionKeqV = "attention_k_eq_v"
        case layerTypes = "layer_types"
        case tieWordEmbeddings = "tie_word_embeddings"
        case hiddenActivation = "hidden_activation"
        case ropeParameters = "rope_parameters"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.modelType =
            try container.decodeIfPresent(String.self, forKey: .modelType) ?? "gemma4_text"
        self.hiddenSize = try container.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 256
        self.numHiddenLayers =
            try container.decodeIfPresent(Int.self, forKey: .numHiddenLayers) ?? 4
        self.intermediateSize =
            try container.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 2048
        self.numAttentionHeads =
            try container.decodeIfPresent(Int.self, forKey: .numAttentionHeads) ?? 4
        self.headDim = try container.decodeIfPresent(Int.self, forKey: .headDim) ?? 256
        self.globalHeadDim =
            try container.decodeIfPresent(Int.self, forKey: .globalHeadDim) ?? 512
        self.rmsNormEps = try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        self.vocabSize = try container.decodeIfPresent(Int.self, forKey: .vocabSize) ?? 262144
        self.numKeyValueHeads =
            try container.decodeIfPresent(Int.self, forKey: .numKeyValueHeads) ?? 2
        self.numGlobalKeyValueHeads =
            try container.decodeIfPresent(Int.self, forKey: .numGlobalKeyValueHeads)
        self.numKvSharedLayers =
            try container.decodeIfPresent(Int.self, forKey: .numKvSharedLayers) ?? 4
        self.slidingWindow =
            try container.decodeIfPresent(Int.self, forKey: .slidingWindow) ?? 512
        self.maxPositionEmbeddings =
            try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 131072
        self.attentionKeqV =
            try container.decodeIfPresent(Bool.self, forKey: .attentionKeqV) ?? false
        self.layerTypes =
            try container.decodeIfPresent([String].self, forKey: .layerTypes)
            ?? ["sliding_attention", "sliding_attention", "sliding_attention", "full_attention"]
        self.tieWordEmbeddings =
            try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? true
        self.hiddenActivation =
            try container.decodeIfPresent(String.self, forKey: .hiddenActivation)
            ?? "gelu_pytorch_tanh"
        self.ropeParameters =
            try container.decodeIfPresent(
                [String: [String: StringOrNumber]].self, forKey: .ropeParameters)

        if let ropeParams = ropeParameters {
            if let sliding = ropeParams["sliding_attention"] {
                self.slidingRopeTheta = sliding["rope_theta"]?.asFloat() ?? 10000.0
            }
            if let full = ropeParams["full_attention"] {
                self.fullRopeTheta = full["rope_theta"]?.asFloat() ?? 1_000_000.0
                self.fullPartialRotaryFactor =
                    full["partial_rotary_factor"]?.asFloat() ?? 0.25
            }
        }
    }
}

// MARK: - Centroid Masked Embedder

/// Centroid-based masked embedder for efficient vocabulary logit computation.
///
/// Instead of computing logits over the full 262K vocabulary, this groups tokens
/// into centroids and only computes logits for tokens in the top-K predicted centroids.
private class CentroidMaskedEmbedder: Module {
    let numCentroids: Int
    let centroidIntermediateTopK: Int
    let hiddenSize: Int
    let vocabSize: Int
    let vocabSizePerCentroid: Int

    @ModuleInfo var centroids: Linear
    @ModuleInfo(key: "token_ordering") var tokenOrdering: MLXArray

    init(_ config: Gemma4AssistantConfiguration) {
        let textConfig = config.textConfig
        self.numCentroids = config.numCentroids
        self.centroidIntermediateTopK = config.centroidIntermediateTopK
        self.hiddenSize = textConfig.hiddenSize
        self.vocabSize = textConfig.vocabSize
        self.vocabSizePerCentroid = vocabSize / numCentroids

        self._centroids.wrappedValue = Linear(hiddenSize, numCentroids, bias: false)
        self._tokenOrdering.wrappedValue = MLXArray.zeros([vocabSize], dtype: .int64)

        super.init()
    }

    func callAsFunction(_ hiddenStates: MLXArray, lmHeadWeight: MLXArray) -> MLXArray {
        let batch = hiddenStates.dim(0)
        let seqLen = hiddenStates.dim(1)

        let centroidLogits = centroids(hiddenStates)

        let sorted = argSort(centroidLogits, axis: -1)
        let n = centroidLogits.dim(-1)
        let topKIndices = sorted[.ellipsis, (n - centroidIntermediateTopK)...]

        let ordering = tokenOrdering.asType(.int32)
        let canonicalPositions = ordering.reshaped(numCentroids, vocabSizePerCentroid)
        let selectedCanonical = canonicalPositions[topKIndices]

        let selectedFlat = selectedCanonical.reshaped(-1)
        let selectedEmbeddings = lmHeadWeight[selectedFlat].reshaped(
            batch, seqLen,
            centroidIntermediateTopK * vocabSizePerCentroid,
            hiddenSize
        )

        let query = hiddenStates.expandedDimensions(axis: -2)
        let selectedLogits = matmul(query, selectedEmbeddings.transposed(0, 1, 3, 2))
            .squeezed(axis: -2)

        let maskValue = selectedLogits.min().item(Float.self) - 1.0
        var output = MLXArray.full([batch, seqLen, vocabSize], values: MLXArray(maskValue))

        let scatterIdx = selectedCanonical.reshaped(batch, seqLen, -1)

        let numUpdates = scatterIdx.dim(2)
        let numRows = batch * seqLen
        var flatBase = output.reshaped(-1)
        let flatIndices = scatterIdx.reshaped(numRows, numUpdates).asType(.int32)
        let rowOffsets = (MLXArray(0 ..< Int32(numRows)) * Int32(vocabSize))
            .expandedDimensions(axis: 1)
        let absIndices = (flatIndices + rowOffsets).reshaped(-1)
        let flatUpdates = selectedLogits.reshaped(-1)
        flatBase = MLXArray.full([numRows * vocabSize], values: output.min())
        flatBase = flatBase.at[absIndices].add(flatUpdates - output.min())
        output = flatBase.reshaped(batch, seqLen, vocabSize)

        return output
    }
}

// MARK: - Attention

private class Gemma4AssistantAttention: Module {
    let layerType: String
    let isSliding: Bool
    let effectiveHeadDim: Int
    let nHeads: Int
    let nKvHeads: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    init(_ config: Gemma4AssistantTextConfiguration, layerIdx: Int) {
        self.layerType = config.layerTypes[layerIdx]
        self.isSliding = layerType == "sliding_attention"
        self.effectiveHeadDim = isSliding ? config.headDim : config.globalHeadDim
        self.nHeads = config.numAttentionHeads
        self.nKvHeads = config.numKeyValueHeads
        self.scale = 1.0

        let dim = config.hiddenSize
        self._qProj.wrappedValue = Linear(dim, nHeads * effectiveHeadDim, bias: false)
        self._kProj.wrappedValue = Linear(dim, nKvHeads * effectiveHeadDim, bias: false)
        self._vProj.wrappedValue = Linear(dim, nKvHeads * effectiveHeadDim, bias: false)
        self._oProj.wrappedValue = Linear(nHeads * effectiveHeadDim, dim, bias: false)
        self._qNorm.wrappedValue = RMSNorm(dimensions: effectiveHeadDim, eps: config.rmsNormEps)
        self._kNorm.wrappedValue = RMSNorm(dimensions: effectiveHeadDim, eps: config.rmsNormEps)

        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode? = nil,
        sharedKV: (MLXArray, MLXArray)? = nil
    ) -> MLXArray {
        let (B, L, _) = (x.dim(0), x.dim(1), x.dim(2))

        var queries = qProj(x).reshaped(B, L, nHeads, effectiveHeadDim)
        queries = qNorm(queries)
        queries = queries.transposed(0, 2, 1, 3)

        let keys: MLXArray
        let values: MLXArray

        if let (sharedK, sharedV) = sharedKV {
            keys = sharedK
            values = sharedV
        } else {
            var k = kProj(x).reshaped(B, L, nKvHeads, effectiveHeadDim)
            k = kNorm(k)
            keys = k.transposed(0, 2, 1, 3)
            values = vProj(x).reshaped(B, L, nKvHeads, effectiveHeadDim).transposed(0, 2, 1, 3)
        }

        var adjustedMask = mask
        if case .array(let maskArray) = mask {
            let keysSeqLen = keys.dim(2)
            if maskArray.dim(-1) != keysSeqLen {
                adjustedMask = .array(maskArray[.ellipsis, 0 ..< keysSeqLen])
            }
        }

        let output = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            mask: adjustedMask ?? .none
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        return oProj(output)
    }
}

// MARK: - MLP

private class Gemma4AssistantMLP: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(_ config: Gemma4AssistantTextConfiguration) {
        self._gateProj.wrappedValue = Linear(
            config.hiddenSize, config.intermediateSize, bias: false)
        self._upProj.wrappedValue = Linear(
            config.hiddenSize, config.intermediateSize, bias: false)
        self._downProj.wrappedValue = Linear(
            config.intermediateSize, config.hiddenSize, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(geluApproximate(gateProj(x)) * upProj(x))
    }
}

// MARK: - Decoder Layer

private class Gemma4AssistantDecoderLayer: Module {
    let layerType: String

    @ModuleInfo(key: "self_attn") var selfAttn: Gemma4AssistantAttention
    @ModuleInfo var mlp: Gemma4AssistantMLP
    @ModuleInfo(key: "input_layernorm") var inputLayernorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayernorm: RMSNorm
    @ModuleInfo(key: "pre_feedforward_layernorm") var preFeedforwardLayernorm: RMSNorm
    @ModuleInfo(key: "post_feedforward_layernorm") var postFeedforwardLayernorm: RMSNorm

    init(_ config: Gemma4AssistantTextConfiguration, layerIdx: Int) {
        self.layerType = config.layerTypes[layerIdx]
        self._selfAttn.wrappedValue = Gemma4AssistantAttention(config, layerIdx: layerIdx)
        self._mlp.wrappedValue = Gemma4AssistantMLP(config)
        self._inputLayernorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postAttentionLayernorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._preFeedforwardLayernorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postFeedforwardLayernorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode? = nil,
        sharedKV: (MLXArray, MLXArray)? = nil
    ) -> MLXArray {
        let residual = x
        let h = inputLayernorm(x)
        let attnOut = selfAttn(h, mask: mask, sharedKV: sharedKV)
        let postAttn = postAttentionLayernorm(attnOut)
        var out = residual + postAttn

        let residual2 = out
        out = preFeedforwardLayernorm(out)
        out = mlp(out)
        out = postFeedforwardLayernorm(out)
        out = residual2 + out

        return out
    }
}

// MARK: - Text Model Inner

private class Gemma4AssistantTextModelInner: Module {
    let config: Gemma4AssistantTextConfiguration

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "layers") var layers: [Gemma4AssistantDecoderLayer]
    @ModuleInfo var norm: RMSNorm

    init(_ config: Gemma4AssistantTextConfiguration) {
        self.config = config
        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabSize, dimensions: config.hiddenSize)
        self._layers.wrappedValue = (0 ..< config.numHiddenLayers).map {
            Gemma4AssistantDecoderLayer(config, layerIdx: $0)
        }
        self._norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        super.init()
    }

    func callAsFunction(
        _ inputsEmbeds: MLXArray,
        sharedKVStates: [String: (MLXArray, MLXArray)]? = nil
    ) -> MLXArray {
        var h = inputsEmbeds

        for layer in layers {
            let sharedKV = sharedKVStates?[layer.layerType]
            h = layer(h, mask: nil, sharedKV: sharedKV)
        }

        return norm(h)
    }
}

// MARK: - Gemma4 Assistant Model

/// Output of the Gemma 4 MTP assistant model forward pass.
public struct Gemma4AssistantOutput {
    /// Up-projected hidden state back in backbone dimension space.
    public let projectedState: MLXArray
    /// Logits over the vocabulary.
    public let logits: MLXArray
}

/// Gemma 4 MTP assistant model for speculative decoding.
///
/// This model is NOT a standalone language model — it operates as a drafter
/// alongside a Gemma 4 backbone model, sharing KV cache and embeddings.
///
/// The forward pass:
/// 1. Takes concatenated `[backbone_last_hidden, token_embedding]` as input
/// 2. Down-projects via `pre_projection` from `2 * backbone_hidden_size` to drafter `hidden_size`
/// 3. Runs through 4 transformer layers (with shared KV from backbone)
/// 4. Up-projects via `post_projection` back to `backbone_hidden_size`
/// 5. Produces logits via `lm_head` (or centroid `masked_embedding` for E2B/E4B)
public class Gemma4AssistantModel: Module, LLMModel, KVCacheDimensionProvider {
    public var vocabularySize: Int { config.textConfig.vocabSize }
    public var kvHeads: [Int] {
        (0 ..< config.textConfig.numHiddenLayers).map { _ in config.textConfig.numKeyValueHeads }
    }

    let config: Gemma4AssistantConfiguration

    @ModuleInfo(key: "model") fileprivate var textModel: Gemma4AssistantTextModelInner
    @ModuleInfo(key: "lm_head") var lmHead: Linear
    @ModuleInfo(key: "pre_projection") var preProjection: Linear
    @ModuleInfo(key: "post_projection") var postProjection: Linear
    @ModuleInfo(key: "masked_embedding") fileprivate var maskedEmbedding: CentroidMaskedEmbedder?

    public init(_ config: Gemma4AssistantConfiguration) {
        self.config = config
        let hiddenSize = config.textConfig.hiddenSize

        self._textModel.wrappedValue = Gemma4AssistantTextModelInner(config.textConfig)
        self._lmHead.wrappedValue = Linear(
            hiddenSize, config.textConfig.vocabSize, bias: false)
        self._preProjection.wrappedValue = Linear(
            2 * config.backboneHiddenSize, hiddenSize, bias: false)
        self._postProjection.wrappedValue = Linear(
            hiddenSize, config.backboneHiddenSize, bias: false)

        if config.useOrderedEmbeddings {
            self._maskedEmbedding.wrappedValue = CentroidMaskedEmbedder(config)
        }
    }

    /// Forward pass for MTP drafting.
    ///
    /// - Parameters:
    ///   - inputsEmbeds: Concatenated `[backbone_hidden, token_embedding]` of shape
    ///     `[B, L, 2 * backbone_hidden_size]`
    ///   - sharedKVStates: Per-layer-type shared KV from backbone's last layers
    /// - Returns: Projected state (in backbone dim space) and logits
    public func assistantForward(
        inputsEmbeds: MLXArray,
        sharedKVStates: [String: (MLXArray, MLXArray)]? = nil
    ) -> Gemma4AssistantOutput {
        let projected = preProjection(inputsEmbeds)
        let lastHidden = textModel(projected, sharedKVStates: sharedKVStates)
        let projectedState = postProjection(lastHidden)

        let logits: MLXArray
        if let maskedEmbedding {
            logits = maskedEmbedding(lastHidden, lmHeadWeight: lmHead.weight)
        } else {
            logits = lmHead(lastHidden)
        }

        return Gemma4AssistantOutput(projectedState: projectedState, logits: logits)
    }

    /// LanguageModel conformance — limited standalone usage.
    ///
    /// In practice this model is used via ``forward(inputsEmbeds:sharedKVStates:)``
    /// during MTP speculative decoding. This conformance exists primarily for
    /// weight loading through the model factory.
    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let embeddings = textModel.embedTokens(inputs)
        let zeros = MLXArray.zeros(like: embeddings)
        let dummyInput = concatenated([zeros, embeddings], axis: -1)
        let output = assistantForward(inputsEmbeds: dummyInput)
        return output.logits
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized = [String: MLXArray]()
        for (key, value) in weights {
            if key.contains("self_attn.rotary_emb") {
                continue
            }
            sanitized[key] = value
        }
        return sanitized
    }
}

// MARK: - MTP Draft Model

extension Gemma4AssistantModel: MTPDraftModel {
    public func forward(
        inputsEmbeds: MLXArray,
        sharedKVStates: [String: (MLXArray, MLXArray)]?
    ) -> MTPDraftOutput {
        let result = assistantForward(
            inputsEmbeds: inputsEmbeds, sharedKVStates: sharedKVStates)
        return MTPDraftOutput(projectedState: result.projectedState, logits: result.logits)
    }
}

// MARK: - LoRA

extension Gemma4AssistantModel: LoRAModel {
    public var loraLayers: [Module] {
        textModel.layers.map { $0.selfAttn }
    }
}
