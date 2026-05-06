//
//  Gemma4Assistant.swift
//  mlx-swift-lm
//
//  Multi-Token Prediction (MTP) drafter for Gemma 4 speculative decoding.
//
//  Reference: https://ai.google.dev/gemma/docs/mtp/overview
//  Python port: Blaizzy/mlx-vlm/mlx_vlm/speculative/drafters/gemma4_assistant
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Configuration

/// Drafter configuration for Gemma 4 Multi-Token Prediction (assistant) models.
///
/// The drafter is a small (4-layer for E2B/E4B/26B-A4B/31B variants) model that
/// shares the target's input embeddings and last-layer hidden state, and reads
/// K/V from the target's last full-attention and last sliding-attention layers.
public struct Gemma4AssistantConfiguration: Codable, Sendable {
    public var modelType: String = "gemma4_assistant"
    public var backboneHiddenSize: Int = 1536
    public var useOrderedEmbeddings: Bool = false
    public var numCentroids: Int = 2048
    public var centroidIntermediateTopK: Int = 32
    public var tieWordEmbeddings: Bool = true
    public var blockSize: Int = 4
    public var textConfig: Gemma4TextConfiguration

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case backboneHiddenSize = "backbone_hidden_size"
        case useOrderedEmbeddings = "use_ordered_embeddings"
        case numCentroids = "num_centroids"
        case centroidIntermediateTopK = "centroid_intermediate_top_k"
        case tieWordEmbeddings = "tie_word_embeddings"
        case blockSize = "block_size"
        case textConfig = "text_config"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.modelType =
            try container.decodeIfPresent(String.self, forKey: .modelType) ?? "gemma4_assistant"
        self.backboneHiddenSize =
            try container.decodeIfPresent(Int.self, forKey: .backboneHiddenSize) ?? 1536
        self.useOrderedEmbeddings =
            try container.decodeIfPresent(Bool.self, forKey: .useOrderedEmbeddings) ?? false
        self.numCentroids =
            try container.decodeIfPresent(Int.self, forKey: .numCentroids) ?? 2048
        self.centroidIntermediateTopK =
            try container.decodeIfPresent(Int.self, forKey: .centroidIntermediateTopK) ?? 32
        self.tieWordEmbeddings =
            try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? true
        self.blockSize =
            try container.decodeIfPresent(Int.self, forKey: .blockSize) ?? 4

        var textCfg: Gemma4TextConfiguration
        if let decoded = try container.decodeIfPresent(
            Gemma4TextConfiguration.self, forKey: .textConfig)
        {
            textCfg = decoded
        } else {
            textCfg = try Gemma4TextConfiguration(from: decoder)
        }
        // HF Gemma4AssistantConfig.__post_init__: when num_kv_shared_layers is 0
        // the assistant shares K/V across all layers.
        if textCfg.numKvSharedLayers == 0 {
            textCfg.numKvSharedLayers = textCfg.numHiddenLayers
        }
        self.textConfig = textCfg
    }
}

// MARK: - Bidirectional masks for the drafter forward

/// Bi-directional full-attention mask.
///
/// Returns nil when no masking is needed (B=1 unbatched case ⇒ no-op).
private func bidirectionalFullMask(
    queryLen: Int,
    kvLen: Int,
    kvValidLen: Int?,
    dtype: DType
) -> MLXArray? {
    guard let valid = kvValidLen, valid < kvLen else { return nil }
    let kIdx = MLXArray(0 ..< kvLen)
    let inside = kIdx .< valid
    let bias = MLX.where(
        inside,
        MLXArray(0.0).asType(dtype),
        MLXArray(-Float.infinity).asType(dtype)
    )
    // [kvLen] → [1, 1, 1, kvLen]
    return bias.expandedDimensions(axes: [0, 1, 2])
}

/// Bi-directional sliding-window mask.
///
/// For each query position `q ∈ [queryOffset, queryOffset + queryLen)`,
/// allows attention to KV positions `k ∈ (q - window, q + window)`.
private func bidirectionalSwaMask(
    queryLen: Int,
    queryOffset: Int,
    kvLen: Int,
    window: Int,
    kvValidLen: Int?,
    dtype: DType
) -> MLXArray? {
    if kvValidLen == nil && kvLen <= window && queryOffset + queryLen <= kvLen + window {
        return nil
    }
    let qStart = queryOffset
    let qEnd = queryOffset + queryLen
    let qIdx = MLXArray(qStart ..< qEnd).expandedDimensions(axis: 1)
    let kIdx = MLXArray(0 ..< kvLen).expandedDimensions(axis: 0)
    let dist = qIdx - kIdx
    var inside = (dist .> -window) .&& (dist .< window)
    if let valid = kvValidLen {
        inside = inside .&& (kIdx .< valid)
    }
    let bias = MLX.where(
        inside,
        MLXArray(0.0).asType(dtype),
        MLXArray(-Float.infinity).asType(dtype)
    )
    // [queryLen, kvLen] → [1, 1, queryLen, kvLen]
    return bias.expandedDimensions(axes: [0, 1])
}

private func makeDrafterMasks(
    sharedKV: [String: (MLXArray, MLXArray)],
    queryLen: Int,
    queryOffset: Int,
    slidingWindow: Int,
    dtype: DType
) -> [String: MLXFast.ScaledDotProductAttentionMaskMode] {
    var masks = [String: MLXFast.ScaledDotProductAttentionMaskMode]()
    for (layerType, kv) in sharedKV {
        let kvLen = kv.0.dim(-2)
        // KV beyond `queryOffset` belongs to bonus / draft tokens themselves
        // (which have not been sampled yet) — must not be attended.
        let kvValid: Int? = (queryOffset < kvLen) ? queryOffset : nil

        // When the helper returns nil, no mask is needed; omit the entry —
        // callers fall back to `.none` via `masks[lt] ?? .none`.
        if layerType == "sliding_attention" {
            if let bias = bidirectionalSwaMask(
                queryLen: queryLen,
                queryOffset: queryOffset,
                kvLen: kvLen,
                window: slidingWindow,
                kvValidLen: kvValid,
                dtype: dtype)
            {
                masks[layerType] = .array(bias)
            }
        } else {
            if let bias = bidirectionalFullMask(
                queryLen: queryLen,
                kvLen: kvLen,
                kvValidLen: kvValid,
                dtype: dtype)
            {
                masks[layerType] = .array(bias)
            }
        }
    }
    return masks
}

// MARK: - Sparse LM head (centroid-routed) for E2B / E4B drafters

/// Centroid-routed sparse softmax head. Mirrors HF
/// `Gemma4AssistantMaskedEmbedder`. The drafter learns a `centroids` linear
/// that scores `numCentroids` token clusters, and a `tokenOrdering` buffer
/// that maps each cluster to a contiguous block of canonical token IDs. At
/// inference the top-K clusters' tokens (default 32 × 128 = 4096 of 262144)
/// are materialised and scored densely; the rest of the vocab is filled with
/// `min(selected) - 1` so it loses any argmax / sampling competition.
private final class Gemma4AssistantMaskedEmbedder: Module {

    let hiddenSize: Int
    let vocabSize: Int
    let numCentroids: Int
    let topK: Int
    let vocabSizePerCentroid: Int

    @ModuleInfo var centroids: Linear
    @ParameterInfo(key: "token_ordering") var tokenOrdering: MLXArray

    init(_ config: Gemma4AssistantConfiguration) {
        let textCfg = config.textConfig
        self.hiddenSize = textCfg.hiddenSize
        self.vocabSize = textCfg.vocabSize
        self.numCentroids = config.numCentroids
        self.topK = config.centroidIntermediateTopK
        self.vocabSizePerCentroid = textCfg.vocabSize / config.numCentroids

        self._centroids.wrappedValue = Linear(hiddenSize, numCentroids, bias: false)
        self._tokenOrdering.wrappedValue = MLXArray.zeros([textCfg.vocabSize], dtype: .int32)

        super.init()
    }

    /// hidden: [B, L, hidden_size], lmHeadWeight: [vocab_size, hidden_size]
    /// Returns: [B, L, vocab_size]
    func callAsFunction(_ hidden: MLXArray, lmHeadWeight: MLXArray) -> MLXArray {
        let B = hidden.dim(0)
        let L = hidden.dim(1)

        // Cluster scores → top-K cluster indices.
        let centroidLogits = centroids(hidden)  // [B, L, num_centroids]
        let topkIdx = MLX.argPartition(
            centroidLogits, kth: numCentroids - topK, axis: -1)[
                .ellipsis, (numCentroids - topK)...
            ]  // [B, L, top_k]

        // Reshape ordering to [num_centroids, vocab_size_per_centroid].
        let ordering = tokenOrdering.reshaped([numCentroids, vocabSizePerCentroid])

        // selected_canonical: [B, L, top_k, vsc]
        let selectedCanonical = ordering[topkIdx]

        // Gather embeddings: lmHeadWeight[selected_canonical] → [B, L, top_k * vsc, hidden]
        let flatIdx = selectedCanonical.reshaped([-1])
        let selectedEmb =
            lmHeadWeight[flatIdx]
            .reshaped([B, L, topK * vocabSizePerCentroid, hiddenSize])

        // selected_logits = h @ E.T
        let h4 = hidden.expandedDimensions(axis: -2)  // [B, L, 1, hidden]
        let eT = selectedEmb.swappedAxes(-1, -2)  // [B, L, hidden, top_k*vsc]
        let selectedLogits = matmul(h4, eT).squeezed(axis: -2)  // [B, L, top_k*vsc]

        let maskValue = selectedLogits.min().item(Float.self) - 1.0

        let scatterIdx = selectedCanonical.reshaped([B, L, -1])  // [B, L, top_k*vsc]
        let outArr = MLXArray.full(
            [B, L, vocabSize],
            values: MLXArray(maskValue).asType(hidden.dtype),
            dtype: hidden.dtype
        )
        return putAlong(outArr, scatterIdx, values: selectedLogits, axis: -1)
    }
}

// MARK: - Drafter inner module (mirrors `_DraftInner` in the Python port)

private class Gemma4DraftInner: Module {
    let config: Gemma4TextConfiguration

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "layers") var layers: [Gemma4DecoderLayer]
    @ModuleInfo var norm: RMSNorm

    init(_ config: Gemma4TextConfiguration) {
        self.config = config
        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabSize, dimensions: config.hiddenSize)
        self._layers.wrappedValue = (0 ..< config.numHiddenLayers).map {
            Gemma4DecoderLayer(config, layerIdx: $0)
        }
        self._norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        super.init()
    }
}

// MARK: - Public drafter model

/// Gemma 4 Multi-Token Prediction drafter.
///
/// Tightly coupled to a ``Gemma4Model`` target: every drafter layer reads K/V
/// from the target's last full-attention and last sliding-attention layers
/// (no own KV cache); the drafter's only recurrent state is the target's
/// last hidden, projected through ``postProjection``.
///
/// Conforms to ``BaseLanguageModel`` so weights can be loaded via the standard
/// ``loadWeights(modelDirectory:model:quantization:perLayerQuantization:)``
/// helper, but does NOT implement the autoregressive ``LanguageModel`` API:
/// the drafter's forward needs the target's hidden state + shared K/V and is
/// only ever called from within ``MTPSpeculativeTokenIterator``.
public class Gemma4AssistantDraftModel: Module, BaseLanguageModel {

    public let config: Gemma4AssistantConfiguration

    fileprivate var model: Gemma4DraftInner

    @ModuleInfo(key: "pre_projection") fileprivate var preProjection: Linear
    @ModuleInfo(key: "post_projection") fileprivate var postProjection: Linear
    @ModuleInfo(key: "lm_head") fileprivate var lmHead: Linear?
    @ModuleInfo(key: "masked_embedding") fileprivate var maskedEmbedding:
        Gemma4AssistantMaskedEmbedder?

    // Bound state (set by ``bind(target:)``)
    private var targetEmbedTokens: Embedding?
    private var targetEmbedScale: Float = 1.0

    public init(_ config: Gemma4AssistantConfiguration) {
        self.config = config
        let textCfg = config.textConfig
        self.model = Gemma4DraftInner(textCfg)
        self._preProjection.wrappedValue = Linear(
            2 * config.backboneHiddenSize, textCfg.hiddenSize, bias: false)
        self._postProjection.wrappedValue = Linear(
            textCfg.hiddenSize, config.backboneHiddenSize, bias: false)
        if !config.tieWordEmbeddings {
            self._lmHead.wrappedValue = Linear(
                textCfg.hiddenSize, textCfg.vocabSize, bias: false)
        }
        if config.useOrderedEmbeddings {
            self._maskedEmbedding.wrappedValue =
                Gemma4AssistantMaskedEmbedder(config)
        }
        super.init()
    }

    /// Wire the target's input embeddings + scale into this drafter. Must be
    /// called before ``draftBlock(...)``. Convenience wrapper that forwards
    /// to the protocol-driven ``bindMTP(target:)``.
    @discardableResult
    public func bind(target: Gemma4Model) -> Self {
        bindMTP(target: target)
        return self
    }

    /// Drafter has no KV cache of its own.
    public func newCache(parameters: GenerateParameters?) -> [any KVCache] { [] }

    private func computeLogits(_ hidden: MLXArray) -> MLXArray {
        if let masked = maskedEmbedding {
            return masked(hidden, lmHeadWeight: model.embedTokens.weight)
        }
        if config.tieWordEmbeddings {
            return model.embedTokens.asLinear(hidden)
        }
        return lmHead!(hidden)
    }

    /// One drafter forward.
    ///
    /// - Parameters:
    ///   - inputsEmbeds: `[B, L, 2 * backboneHiddenSize]` (already concatenated
    ///     `[targetEmbed(token), lastHidden]`).
    ///   - sharedKV: target's last full-/sliding-attention K/V keyed by layer
    ///     type.
    ///   - positionOffset: drafter's RoPE position (held constant across draft
    ///     steps within a block).
    /// - Returns: `(newLastHidden, logits)` where `newLastHidden` is the
    ///   `postProjection`-mapped output ready to feed into the next draft step.
    fileprivate func forward(
        inputsEmbeds: MLXArray,
        sharedKV: [String: (MLXArray, MLXArray)],
        positionOffset: Int
    ) -> (MLXArray, MLXArray) {
        let textCfg = config.textConfig
        var h = preProjection(inputsEmbeds)
        let queryLen = h.dim(1)

        let masks = makeDrafterMasks(
            sharedKV: sharedKV,
            queryLen: queryLen,
            queryOffset: positionOffset,
            slidingWindow: textCfg.slidingWindow,
            dtype: h.dtype
        )

        let offset = Gemma4PositionOffset.scalar(positionOffset)
        for layer in model.layers {
            let lt = layer.layerType
            guard let kv = sharedKV[lt] else {
                fatalError("Drafter layer of type \(lt) has no shared KV provided")
            }
            let mask = masks[lt] ?? .none
            let (out, _, _) = layer(
                h,
                mask: mask,
                cache: nil,
                perLayerInput: nil,
                sharedKV: kv,
                positionOffset: offset
            )
            h = out
        }

        let normed = model.norm(h)
        let lastHidden = postProjection(normed)
        let logits = computeLogits(normed)
        return (lastHidden, logits)
    }

    /// Autoregressive K-step drafting.
    ///
    /// - Parameters:
    ///   - bonusToken: most recently accepted target token (will be the first
    ///     verify position, not drafted here).
    ///   - targetLastHidden: `[B, 1, backboneHiddenSize]` — target's last
    ///     hidden state at the bonus token's position.
    ///   - sharedKV: target's last full-/sliding-attention K/V.
    ///   - positionOffset: bonus token's absolute position (used as RoPE
    ///     position for all draft steps within this block).
    ///   - blockSize: total verify-block size; the drafter generates
    ///     `blockSize - 1` candidate tokens.
    /// - Returns: drafted token IDs `[Int]` of length `blockSize - 1` (greedy).
    public func draftBlock(
        bonusToken: Int,
        targetLastHidden: MLXArray,
        sharedKV: [String: (MLXArray, MLXArray)],
        positionOffset: Int,
        blockSize: Int
    ) -> [Int] {
        precondition(blockSize >= 2, "blockSize must be ≥ 2")
        guard let inputEmbed = targetEmbedTokens else {
            fatalError("bind(target:) must be called before draftBlock(...)")
        }

        var hPrev = targetLastHidden
        var tok = MLXArray([Int32(bonusToken)]).reshaped([1, 1])
        var out: [Int] = []
        out.reserveCapacity(blockSize - 1)

        for _ in 0 ..< (blockSize - 1) {
            let tokEmbed = inputEmbed(tok) * targetEmbedScale
            let inputsEmbeds = concatenated([tokEmbed, hPrev], axis: -1)
            let (newHidden, logits) = forward(
                inputsEmbeds: inputsEmbeds,
                sharedKV: sharedKV,
                positionOffset: positionOffset
            )
            let nextTok = argMax(logits[0..., -1, 0...], axis: -1)
            asyncEval(nextTok)
            let id = nextTok.item(Int.self)
            out.append(id)
            hPrev = newHidden
            tok = MLXArray([Int32(id)]).reshaped([1, 1])
        }
        return out
    }

    /// Sanitize HF / mlx-community checkpoint weights.
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var out = [String: MLXArray]()
        for (k, v) in weights {
            // `lm_head.weight` is dropped when tied to `embed_tokens.weight`
            // (matches HF / Python ref).
            if k == "lm_head.weight" && config.tieWordEmbeddings {
                continue
            }
            // Cast cluster-routing index buffer to int32 (loaded as int64).
            if k == "masked_embedding.token_ordering" {
                out[k] = v.asType(.int32)
                continue
            }
            out[k] = v
        }
        return out
    }
}

// MARK: - Protocol conformances for MTP iterator

extension Gemma4AssistantDraftModel: MTPDrafterModel {
    public func bindMTP(target: any MTPTargetModel) {
        targetEmbedTokens = target.inputEmbeddings
        targetEmbedScale = target.inputEmbedScale
    }
}

extension Gemma4Model: MTPTargetModel {}

// MARK: - Loader

extension Gemma4AssistantDraftModel {

    /// Load a Gemma 4 MTP drafter from a local model directory. The directory
    /// must contain `config.json` and `*.safetensors` files.
    ///
    /// Resolve a HF `ModelConfiguration` to a local directory using the
    /// existing factory infrastructure first:
    /// ```swift
    /// let resolved = try await resolve(
    ///     configuration: ModelConfiguration(
    ///         id: "mlx-community/gemma-4-E2B-it-assistant-bf16"),
    ///     from: HubDownloader.shared, useLatest: false,
    ///     progressHandler: { _ in })
    /// let drafter = try Gemma4AssistantDraftModel.load(
    ///     from: resolved.modelDirectory)
    /// ```
    public static func load(from modelDirectory: URL) throws -> Gemma4AssistantDraftModel {
        let configURL = modelDirectory.appendingPathComponent("config.json")
        let data = try Data(contentsOf: configURL)
        let decoder = JSONDecoder()
        let config = try decoder.decode(Gemma4AssistantConfiguration.self, from: data)
        let baseConfig = try? decoder.decode(BaseConfiguration.self, from: data)

        let drafter = Gemma4AssistantDraftModel(config)
        try loadWeights(
            modelDirectory: modelDirectory,
            model: drafter,
            perLayerQuantization: baseConfig?.perLayerQuantization
        )
        return drafter
    }
}
