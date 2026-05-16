// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Configuration

/// Configuration for the Gemma 4 MTP speculative drafter.
///
/// Mirrors mlx-vlm's `Gemma4AssistantConfig`
/// (`mlx_vlm/speculative/drafters/gemma4_assistant/config.py` at SHA d49d428).
/// Decodes from a HF `config.json` shape.
public struct Gemma4AssistantConfiguration: Codable, Sendable {
    public let textConfiguration: Gemma4TextConfiguration
    public let backboneHiddenSize: Int
    public let modelType: String
    public let tieWordEmbeddings: Bool
    public let useOrderedEmbeddings: Bool
    public let numCentroids: Int
    public let centroidIntermediateTopK: Int
    public let blockSize: Int
    /// Per mlx-vlm config.py: unused by MTP (drafter consumes shared K/V, not
    /// per-layer hidden captures) but kept for API parity with DFlash.
    public let targetLayerIds: [Int]

    enum CodingKeys: String, CodingKey {
        case textConfiguration = "text_config"
        case backboneHiddenSize = "backbone_hidden_size"
        case modelType = "model_type"
        case tieWordEmbeddings = "tie_word_embeddings"
        case useOrderedEmbeddings = "use_ordered_embeddings"
        case numCentroids = "num_centroids"
        case centroidIntermediateTopK = "centroid_intermediate_top_k"
        case blockSize = "block_size"
        case targetLayerIds = "target_layer_ids"
    }

    public init(from decoder: any Swift.Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        textConfiguration = try c.decode(Gemma4TextConfiguration.self, forKey: .textConfiguration)
        backboneHiddenSize =
            try c.decodeIfPresent(Int.self, forKey: .backboneHiddenSize) ?? 1536
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "gemma4_assistant"
        tieWordEmbeddings =
            try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? true
        useOrderedEmbeddings =
            try c.decodeIfPresent(Bool.self, forKey: .useOrderedEmbeddings) ?? false
        numCentroids = try c.decodeIfPresent(Int.self, forKey: .numCentroids) ?? 2048
        centroidIntermediateTopK =
            try c.decodeIfPresent(Int.self, forKey: .centroidIntermediateTopK) ?? 32
        blockSize = try c.decodeIfPresent(Int.self, forKey: .blockSize) ?? 4
        targetLayerIds = try c.decodeIfPresent([Int].self, forKey: .targetLayerIds) ?? []
    }
}

// MARK: - MaskedEmbedder (use_ordered_embeddings)

/// Centroid-routed sparse LM head used when `use_ordered_embeddings=true`.
///
/// Mirrors mlx-vlm's
/// `mlx_vlm/speculative/drafters/gemma4_assistant/masked_embedder.py`.
///
/// **Phase A note:** the 26B-A4B-bf16 and 31B-bf16 reference checkpoints
/// both ship `use_ordered_embeddings=false`, so this module's forward path
/// is not exercised by any current verification fixture. The module is
/// declared (and its weights would load correctly) but `callAsFunction` is
/// a TODO — implementation lands when a checkpoint with the centroid path
/// becomes available for verification.
public final class Gemma4AssistantMaskedEmbedder: Module {
    @ModuleInfo(key: "centroids") public var centroids: Linear
    @ParameterInfo(key: "token_ordering") public var tokenOrdering: MLXArray

    public let hiddenSize: Int
    public let vocabSize: Int
    public let numCentroids: Int
    public let topK: Int
    public let vocabSizePerCentroid: Int

    public init(config: Gemma4AssistantConfiguration) {
        let textCfg = config.textConfiguration
        self.hiddenSize = textCfg.hiddenSize
        self.vocabSize = textCfg.vocabularySize
        self.numCentroids = config.numCentroids
        self.topK = config.centroidIntermediateTopK
        self.vocabSizePerCentroid = max(1, vocabSize / max(numCentroids, 1))

        self._centroids.wrappedValue = Linear(hiddenSize, numCentroids, bias: false)
        self._tokenOrdering.wrappedValue = MLXArray.zeros([vocabSize], dtype: .int32)
        super.init()
    }

    public func callAsFunction(_ hiddenStates: MLXArray, lmHeadWeight: MLXArray) -> MLXArray {
        fatalError(
            "Gemma4AssistantMaskedEmbedder forward not implemented yet — requires a "
                + "use_ordered_embeddings=true checkpoint to verify; current fixtures "
                + "use the tied-lm_head path.")
    }
}

// MARK: - Inner draft module

/// The drafter's inner module — mirrors the Python `_DraftInner` so weight
/// keys (`model.embed_tokens.weight`, `model.layers.N.*`, `model.norm.weight`)
/// load without remapping.
///
/// Class is public so it can appear as a property type on
/// ``Gemma4AssistantDraftModel``; members are internal because they reference
/// internal-scoped Gemma 4 building blocks (`Gemma4TextDecoderLayer`,
/// `Gemma4RMSNormZeroShift`).
public final class Gemma4AssistantDraftInner: Module {
    public let config: Gemma4TextConfiguration

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "layers") var layers: [Gemma4TextDecoderLayer]
    @ModuleInfo(key: "norm") var norm: Gemma4RMSNormZeroShift

    public init(_ textConfig: Gemma4TextConfiguration) {
        self.config = textConfig
        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: textConfig.vocabularySize,
            dimensions: textConfig.hiddenSize)
        self._layers.wrappedValue = (0 ..< textConfig.hiddenLayers).map {
            Gemma4TextDecoderLayer(config: textConfig, layerIdx: $0, kvSharedOnly: true)
        }
        self._norm.wrappedValue = Gemma4RMSNormZeroShift(
            dimensions: textConfig.hiddenSize, eps: textConfig.rmsNormEps)
        super.init()
    }
}

// MARK: - Drafter

/// The Gemma 4 MTP drafter — a 4-layer Q-only assistant that cross-attends to
/// the target model's last full-attention and last sliding-attention K/V to
/// propose K-1 tokens per round.
///
/// Mirrors mlx-vlm's `Gemma4AssistantDraftModel` at
/// `mlx_vlm/speculative/drafters/gemma4_assistant/gemma4_assistant.py`
/// (SHA d49d428). Conforms to ``MTPDrafterModel``; consumed by
/// `MTPSpeculativeTokenIterator` (Phase B).
public final class Gemma4AssistantDraftModel: Module, MTPDrafterModel {
    public let config: Gemma4AssistantConfiguration

    @ModuleInfo(key: "model") public var model: Gemma4AssistantDraftInner
    @ModuleInfo(key: "pre_projection") public var preProjection: Linear
    @ModuleInfo(key: "post_projection") public var postProjection: Linear
    @ModuleInfo(key: "lm_head") public var lmHead: Linear?
    @ModuleInfo(key: "masked_embedding") public var maskedEmbedding: Gemma4AssistantMaskedEmbedder?

    // Bind-time state. Read-only during eval (consistent with PR #283's
    // "no mutation during eval" invariant). Not `Sendable` because Embedding
    // is a reference type; cross-domain access must go through
    // `MTPDrafterContainer.perform`.
    private var boundInputEmbed: Embedding?
    private var boundEmbedScale: Float = 1.0
    private var boundTargetLayerTypes: [String] = []

    public init(_ config: Gemma4AssistantConfiguration) {
        self.config = config
        let textCfg = config.textConfiguration
        self._model.wrappedValue = Gemma4AssistantDraftInner(textCfg)
        self._preProjection.wrappedValue = Linear(
            2 * config.backboneHiddenSize, textCfg.hiddenSize, bias: false)
        self._postProjection.wrappedValue = Linear(
            textCfg.hiddenSize, config.backboneHiddenSize, bias: false)
        if !config.tieWordEmbeddings {
            self._lmHead.wrappedValue = Linear(
                textCfg.hiddenSize, textCfg.vocabularySize, bias: false)
        }
        if config.useOrderedEmbeddings {
            self._maskedEmbedding.wrappedValue = Gemma4AssistantMaskedEmbedder(config: config)
        }
        super.init()
    }

    public func bind(target: any LanguageModel) {
        // Mirrors mlx-vlm's bind walk (gemma4_assistant.py:79-101): for Gemma 4
        // the input embedding lives under `.language_model.model.embed_tokens`.
        // `Gemma4TextLanguageModel` is not itself a `LanguageModel`, so we
        // only match the top-level `Gemma4` here.
        guard let g4 = target as? Gemma4 else {
            fatalError(
                "Gemma4AssistantDraftModel.bind: target is not a Gemma 4 VLM "
                    + "(got \(type(of: target)))")
        }
        let backbone = g4.languageModel.model
        self.boundInputEmbed = backbone.embedTokens
        self.boundEmbedScale = backbone.embedScale
        self.boundTargetLayerTypes = g4.languageModel.config.layerTypes
    }

    public func draftBlock(
        lastToken: MLXArray,
        lastHidden: MLXArray,
        sharedKV: [String: (MLXArray, MLXArray)],
        positionIds: MLXArray,
        blockSize: Int,
        sampler: any LogitSampler
    ) -> MLXArray {
        precondition(
            boundInputEmbed != nil, "bind(target:) must be called before draftBlock")
        precondition(blockSize >= 2, "blockSize must be >= 2 (K-1 drafted + 1 bonus)")

        // tok shape: [B, 1]
        var tok =
            lastToken.ndim == 1 ? lastToken.reshaped([lastToken.dim(0), 1]) : lastToken
        var hPrev = lastHidden
        var tokens: [MLXArray] = []
        tokens.reserveCapacity(blockSize - 1)

        for _ in 0 ..< (blockSize - 1) {
            let tokEmbed =
                boundInputEmbed!(tok) * MLXArray(boundEmbedScale, dtype: hPrev.dtype)
            let inputsEmbeds = concatenated([tokEmbed, hPrev], axis: -1)
            let (newHidden, logits) = forwardHidden(
                inputsEmbeds: inputsEmbeds,
                sharedKV: sharedKV,
                positionIds: positionIds
            )
            hPrev = newHidden
            // logits: [B, L, vocab]; take last step → [B, vocab]
            let lastStepLogits = logits[0..., -1, 0...]
            let nextTok = sampler.sample(logits: lastStepLogits)
            tok = nextTok.ndim == 1 ? nextTok.reshaped([nextTok.dim(0), 1]) : nextTok
            tokens.append(tok)
        }
        return concatenated(tokens, axis: 1)
    }

    /// One drafter forward — pre-projection, layer loop with shared K/V,
    /// final norm, post-projection, and lm_head application.
    ///
    /// Exposed at `@_spi(Testing)` scope so fixture-parity tests (Rung 2/3)
    /// can compare per-call outputs to Python reference without going through
    /// the autoregressive `draftBlock` loop. Not part of the public API.
    @_spi(Testing) public func forwardHidden(
        inputsEmbeds: MLXArray,
        sharedKV: [String: (MLXArray, MLXArray)],
        positionIds: MLXArray
    ) -> (lastHidden: MLXArray, logits: MLXArray) {
        let textCfg = config.textConfiguration
        var h = preProjection(inputsEmbeds)
        let queryLen = h.dim(1)

        // Per-layer-type masks; KV tensor shape is [B, H, S, D] so axis -2 = seq.
        let fullKvLen = sharedKV["full_attention"].map { $0.0.dim(-2) } ?? 0
        let slidingKvLen = sharedKV["sliding_attention"].map { $0.0.dim(-2) } ?? 0
        let fullMask = createBidirectionalMask(
            queryLen: queryLen, kvLen: fullKvLen, dtype: h.dtype)
        let slidingMask = createBidirectionalSlidingWindowMask(
            queryLen: queryLen, kvLen: slidingKvLen,
            windowSize: textCfg.slidingWindow, dtype: h.dtype)

        let queryOffset = Int(positionIds[0, 0].item(Int32.self))

        for layer in model.layers {
            guard let kvPair = sharedKV[layer.layerType] else {
                fatalError("Gemma4 drafter: missing sharedKV for layer_type \(layer.layerType)")
            }
            let kvState = Gemma4SharedKVState.regular(
                keys: kvPair.0, values: kvPair.1)
            let layerMask: MLXFast.ScaledDotProductAttentionMaskMode =
                layer.layerType == "full_attention"
                ? .array(fullMask)
                : .array(slidingMask)
            let (out, _, _) = layer(
                h, mask: layerMask, cache: nil, perLayerInput: nil,
                sharedKV: kvState, offset: queryOffset)
            h = out
        }

        h = model.norm(h)
        let lastHidden = postProjection(h)

        let logits: MLXArray
        if let maskedEmbedding {
            logits = maskedEmbedding(h, lmHeadWeight: model.embedTokens.weight)
        } else if config.tieWordEmbeddings {
            logits = model.embedTokens.asLinear(h)
        } else {
            guard let lmHead else {
                fatalError("Gemma4 drafter: lm_head missing despite tieWordEmbeddings=false")
            }
            logits = lmHead(h)
        }
        return (lastHidden, logits)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized = weights
        if config.tieWordEmbeddings {
            sanitized.removeValue(forKey: "lm_head.weight")
        }
        if let tokenOrdering = sanitized["masked_embedding.token_ordering"] {
            sanitized["masked_embedding.token_ordering"] = tokenOrdering.asType(.int32)
        }
        return sanitized
    }
}
