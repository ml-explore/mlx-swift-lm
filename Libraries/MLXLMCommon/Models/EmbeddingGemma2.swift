// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN

// MARK: - Configuration

/// Configuration of `google/embeddinggemma-2` (`model_type: embedding_gemma2`).
///
/// Decodes the checkpoint's `config.json` directly, including the nested `text_config`
/// and the optional `vision_config` and `audio_config`.
public struct EmbeddingGemma2Configuration: Codable, Sendable {

    /// Context window from the model card. `max_position_embeddings` is the RoPE table
    /// size, not a usable limit.
    public static let contextLength = 8_192

    /// One attention head layout per layer: global layers use larger heads and one key/value head.
    public struct Attention: Equatable, Sendable {
        public let headDim: Int
        public let kvHeads: Int
        public let ropeBase: Float
        public let isGlobal: Bool
    }

    /// The vision encoder layout and the tokens that mark images and video frames in a
    /// sequence. Video frames share the image begin and end markers.
    public struct Vision: Sendable {
        public let encoder: Gemma4VisionConfiguration
        public let imageTokenID: Int
        public let videoTokenID: Int?
        public let beginImageTokenID: Int
        public let endImageTokenID: Int
    }

    /// The audio encoder layout and the tokens that mark audio in a sequence.
    public struct Audio: Sendable {
        public let encoder: Gemma4AudioConfiguration
        public let audioTokenID: Int
        public let beginAudioTokenID: Int
        public let endAudioTokenID: Int
    }

    private let textConfiguration: TextConfig
    private let modelType: String

    public let hiddenSize: Int
    public let hiddenLayers: Int
    public let attentionHeads: Int
    public let kvHeads: Int
    public let headDim: Int
    public let intermediateSize: Int
    public let perLayerInputSize: Int
    public let embeddingDim: Int
    public let vocabularySize: Int
    public let rmsNormEps: Float
    /// Radius: a token sees the keys within this distance on both sides.
    public let slidingWindow: Int
    public let layerTypes: [String]
    public let attention: [Attention]
    public private(set) var vision: Vision?
    public private(set) var audio: Audio?

    /// The image, video and audio placeholders that soft tokens replace.
    public var softTokenIDs: [Int] {
        [vision?.imageTokenID, vision?.videoTokenID, audio?.audioTokenID].compactMap { $0 }
    }

    private struct LayerOverride: Codable, Sendable {
        let headDim: Int?
        let keyValueHeads: Int?

        enum CodingKeys: String, CodingKey {
            case headDim = "head_dim"
            case keyValueHeads = "num_key_value_heads"
        }
    }

    private struct RopeParameters: Codable, Sendable {
        let ropeTheta: Float
        let ropeType: String?

        enum CodingKeys: String, CodingKey {
            case ropeTheta = "rope_theta"
            case ropeType = "rope_type"
        }
    }

    /// Text fields. A multimodal checkpoint nests them under `text_config`.
    private struct TextConfig: Codable, Sendable {
        let hiddenSize: Int
        let hiddenLayers: Int
        let attentionHeads: Int
        let keyValueHeads: Int?
        let headDim: Int
        let intermediateSize: Int
        let perLayerInputSize: Int
        let embeddingDim: Int
        let vocabularySize: Int
        let rmsNormEps: Float
        let slidingWindow: Int
        let layerTypes: [String]
        let perLayerConfig: [String: LayerOverride]?
        let ropeParameters: [String: RopeParameters]
        let hiddenActivation: String?
        let attentionBias: Bool?

        enum CodingKeys: String, CodingKey {
            case hiddenSize = "hidden_size"
            case hiddenLayers = "num_hidden_layers"
            case attentionHeads = "num_attention_heads"
            case keyValueHeads = "num_key_value_heads"
            case headDim = "head_dim"
            case intermediateSize = "intermediate_size"
            case perLayerInputSize = "hidden_size_per_layer_input"
            case embeddingDim = "embedding_dim"
            case vocabularySize = "vocab_size"
            case rmsNormEps = "rms_norm_eps"
            case slidingWindow = "sliding_window"
            case layerTypes = "layer_types"
            case perLayerConfig = "per_layer_config"
            case ropeParameters = "rope_parameters"
            case hiddenActivation = "hidden_activation"
            case attentionBias = "attention_bias"
        }
    }

    private enum RootKeys: String, CodingKey {
        case modelType = "model_type"
        case textConfig = "text_config"
        case visionConfig = "vision_config"
        case audioConfig = "audio_config"
        case imageTokenId = "image_token_id"
        case videoTokenId = "video_token_id"
        case boiTokenId = "boi_token_id"
        case eoiTokenId = "eoi_token_id"
        case audioTokenId = "audio_token_id"
        case boaTokenId = "boa_token_id"
        case eoaTokenId = "eoa_token_id"
        case eoaTokenIndex = "eoa_token_index"
    }

    public init(from decoder: any Decoder) throws {
        let root = try decoder.container(keyedBy: RootKeys.self)
        if let type = try root.decodeIfPresent(String.self, forKey: .modelType),
            !["embedding_gemma2", "embedding_gemma2_text"].contains(type)
        {
            throw DecodingError.dataCorruptedError(
                forKey: .modelType, in: root,
                debugDescription: "Expected an EmbeddingGemma 2 checkpoint.")
        }
        modelType =
            try root.decodeIfPresent(String.self, forKey: .modelType)
            ?? (root.contains(.textConfig) ? "embedding_gemma2" : "embedding_gemma2_text")
        // Standalone text checkpoints keep the same fields at the top level.
        let text =
            try root.decodeIfPresent(TextConfig.self, forKey: .textConfig)
            ?? TextConfig(from: decoder)

        textConfiguration = text
        hiddenSize = text.hiddenSize
        hiddenLayers = text.hiddenLayers
        attentionHeads = text.attentionHeads
        kvHeads = text.keyValueHeads ?? text.attentionHeads
        headDim = text.headDim
        intermediateSize = text.intermediateSize
        perLayerInputSize = text.perLayerInputSize
        embeddingDim = text.embeddingDim
        vocabularySize = text.vocabularySize
        rmsNormEps = text.rmsNormEps
        slidingWindow = text.slidingWindow
        layerTypes = text.layerTypes

        let overrides = text.perLayerConfig ?? [:]
        var overridesByLayer: [Int: LayerOverride] = [:]
        for (key, value) in overrides {
            guard let index = Int(key), overridesByLayer[index] == nil else {
                throw DecodingError.dataCorruptedError(
                    forKey: RootKeys.textConfig, in: root,
                    debugDescription: "Per-layer keys must be layer indices.")
            }
            overridesByLayer[index] = value
        }
        let hasDefaultRopes = text.ropeParameters.values.allSatisfy {
            ($0.ropeType ?? "default") == "default"
        }
        guard layerTypes.count == hiddenLayers,
            hiddenLayers > 0, slidingWindow > 0, hasDefaultRopes,
            (text.hiddenActivation ?? "gelu_pytorch_tanh") == "gelu_pytorch_tanh",
            !(text.attentionBias ?? false),
            hiddenSize > 0, attentionHeads > 0, kvHeads > 0, headDim > 0,
            intermediateSize > 0, perLayerInputSize > 0, embeddingDim > 0,
            vocabularySize > 0, rmsNormEps.isFinite, rmsNormEps > 0,
            overridesByLayer.keys.allSatisfy({ (0 ..< text.hiddenLayers).contains($0) })
        else {
            throw DecodingError.dataCorruptedError(
                forKey: RootKeys.textConfig, in: root,
                debugDescription: "Unsupported EmbeddingGemma 2 layer layout.")
        }
        var attention: [Attention] = []
        attention.reserveCapacity(hiddenLayers)
        for (index, type) in layerTypes.enumerated() {
            guard ["full_attention", "sliding_attention"].contains(type),
                let rope = text.ropeParameters[type], rope.ropeTheta.isFinite, rope.ropeTheta > 0,
                (overridesByLayer[index]?.headDim ?? headDim) > 0,
                (overridesByLayer[index]?.headDim ?? headDim) % 2 == 0,
                (overridesByLayer[index]?.keyValueHeads ?? kvHeads) > 0,
                attentionHeads % (overridesByLayer[index]?.keyValueHeads ?? kvHeads) == 0
            else {
                throw DecodingError.dataCorruptedError(
                    forKey: RootKeys.textConfig, in: root,
                    debugDescription: "Missing RoPE parameters for \(type).")
            }
            let override = overridesByLayer[index]
            attention.append(
                Attention(
                    headDim: override?.headDim ?? headDim,
                    kvHeads: override?.keyValueHeads ?? kvHeads,
                    ropeBase: rope.ropeTheta,
                    isGlobal: type == "full_attention"))
        }
        self.attention = attention

        // The library covers the encoder layout EmbeddingGemma 2 ships: no activation
        // clipping and no standardization.
        if let encoder = try root.decodeIfPresent(
            Gemma4VisionConfiguration.self, forKey: .visionConfig),
            let image = try root.decodeIfPresent(Int.self, forKey: .imageTokenId),
            let begin = try root.decodeIfPresent(Int.self, forKey: .boiTokenId),
            let end = try root.decodeIfPresent(Int.self, forKey: .eoiTokenId)
        {
            guard !encoder.useClippedLinears, !encoder.standardize else {
                throw DecodingError.dataCorruptedError(
                    forKey: .visionConfig, in: root,
                    debugDescription: "Unsupported EmbeddingGemma 2 vision encoder.")
            }
            vision = Vision(
                encoder: encoder, imageTokenID: image,
                videoTokenID: try root.decodeIfPresent(Int.self, forKey: .videoTokenId),
                beginImageTokenID: begin, endImageTokenID: end)
        }
        // `embedding_gemma2` checkpoints name the end marker `eoa_token_index`.
        if let encoder = try root.decodeIfPresent(
            Gemma4AudioConfiguration.self, forKey: .audioConfig),
            let token = try root.decodeIfPresent(Int.self, forKey: .audioTokenId),
            let begin = try root.decodeIfPresent(Int.self, forKey: .boaTokenId),
            let end = try root.decodeIfPresent(Int.self, forKey: .eoaTokenId)
                ?? root.decodeIfPresent(Int.self, forKey: .eoaTokenIndex)
        {
            audio = Audio(
                encoder: encoder, audioTokenID: token, beginAudioTokenID: begin,
                endAudioTokenID: end)
        }
    }

    public func encode(to encoder: any Encoder) throws {
        var root = encoder.container(keyedBy: RootKeys.self)
        try root.encode(modelType, forKey: .modelType)
        try root.encode(textConfiguration, forKey: .textConfig)
        if let vision {
            try root.encode(vision.encoder, forKey: .visionConfig)
            try root.encode(vision.imageTokenID, forKey: .imageTokenId)
            try root.encodeIfPresent(vision.videoTokenID, forKey: .videoTokenId)
            try root.encode(vision.beginImageTokenID, forKey: .boiTokenId)
            try root.encode(vision.endImageTokenID, forKey: .eoiTokenId)
        }
        if let audio {
            try root.encode(audio.encoder, forKey: .audioConfig)
            try root.encode(audio.audioTokenID, forKey: .audioTokenId)
            try root.encode(audio.beginAudioTokenID, forKey: .boaTokenId)
            try root.encode(audio.endAudioTokenID, forKey: .eoaTokenIndex)
        }
    }

    /// Returns a configuration that omits unused modality encoders before weights load.
    public func selectingEncoders(vision: Bool = true, audio: Bool = true) -> Self {
        var configuration = self
        if !vision { configuration.vision = nil }
        if !audio { configuration.audio = nil }
        return configuration
    }

}

// MARK: - Model

/// EmbeddingGemma 2: a bidirectional text encoder with per-layer embeddings that maps
/// text and the soft tokens of images, video frames and audio into one normalized
/// ``EmbeddingGemma2Configuration/embeddingDim`` space. Load checkpoints with
/// `loadWeights`.
public final class EmbeddingGemma2: Module, BaseLanguageModel {

    public let config: EmbeddingGemma2Configuration

    @ModuleInfo(key: "language_model") private var languageModel: EmbeddingGemma2TextModel
    @ModuleInfo(key: "vision_tower") private var visionTower: Gemma4VisionModel?
    @ModuleInfo(key: "embed_vision") private var embedVision: Gemma4MultimodalEmbedder?
    @ModuleInfo(key: "audio_tower") private var audioTower: Gemma4AudioModel?
    @ModuleInfo(key: "embed_audio") private var embedAudio: Gemma4MultimodalEmbedder?

    private let softTokenIDs: [Int32]

    public init(_ config: EmbeddingGemma2Configuration) {
        self.config = config
        self._languageModel.wrappedValue = EmbeddingGemma2TextModel(config)
        if let vision = config.vision {
            self._visionTower.wrappedValue = Gemma4VisionModel(config: vision.encoder)
            self._embedVision.wrappedValue = Gemma4MultimodalEmbedder(
                embeddingDim: vision.encoder.hiddenSize, textHiddenSize: config.hiddenSize,
                eps: vision.encoder.rmsNormEps)
        }
        if let audio = config.audio {
            self._audioTower.wrappedValue = Gemma4AudioModel(config: audio.encoder)
            self._embedAudio.wrappedValue = Gemma4MultimodalEmbedder(
                embeddingDim: audio.encoder.outputProjectionDimensions,
                textHiddenSize: config.hiddenSize, eps: audio.encoder.rmsNormEps)
        }
        self.softTokenIDs = config.softTokenIDs.map(Int32.init)
        super.init()
    }

    /// - Parameter pixels: `[B, 3, H, W]` images or video frames of one size.
    /// - Returns: `[B, soft tokens, hidden]` features that fill their placeholder tokens,
    ///   or `nil` when this checkpoint has no vision encoder.
    public func imageFeatures(_ pixels: MLXArray) -> MLXArray? {
        guard let visionTower, let embedVision else { return nil }
        return embedVision(visionTower(pixels))
    }

    /// - Parameter features: `[frames, featureSize]` log-mel features of one audio.
    /// - Returns: `[1, soft tokens, hidden]` features that fill its placeholder tokens, or
    ///   `nil` when this checkpoint has no audio encoder.
    public func audioFeatures(_ features: MLXArray) -> MLXArray? {
        guard let audioTower, let embedAudio else { return nil }
        return embedAudio(audioTower(features))
    }

    /// - Parameters:
    ///   - inputIds: `[B, L]` tokens.
    ///   - attentionMask: `[B, L]`, `1` for real tokens. Attention and pooling ignore padding.
    ///   - softTokens: `[soft tokens, hidden]` features of every image, video and audio
    ///     placeholder in `inputIds`, in sequence order; single row only.
    /// - Returns: `[B, embeddingDim]` unit-length float32 embeddings.
    public func embed(
        inputIds: MLXArray, attentionMask: MLXArray?, softTokens: MLXArray? = nil
    ) -> MLXArray {
        precondition(inputIds.ndim == 2 && inputIds.dim(1) > 0)
        if let attentionMask {
            precondition(attentionMask.shape == inputIds.shape)
        }
        var hidden = languageModel.embedTokens(inputIds)
        // The model card forbids float16: activations exceed its range.
        if hidden.dtype == .float16 { hidden = hidden.asType(.float32) }
        // Matches the reference: sqrt(hidden) is rounded to the weight dtype (22.625 in bf16).
        hidden = hidden * MLXArray(Float(config.hiddenSize).squareRoot()).asType(hidden.dtype)
        if let softTokens, let first = softTokenIDs.first {
            precondition(inputIds.dim(0) == 1)
            let isSoftToken = softTokenIDs.dropFirst().reduce(inputIds .== first) {
                $0 .|| (inputIds .== $1)
            }
            let index = maximum(cumsum(isSoftToken.asType(.int32), axis: 1) - 1, 0)
            let rows = softTokens.asType(hidden.dtype).reshaped(-1, hidden.dim(-1))
                .take(index.squeezed(axis: 0), axis: 0).expandedDimensions(axis: 0)
            hidden = MLX.where(isSoftToken.expandedDimensions(axis: -1), rows, hidden)
        }
        hidden = languageModel(hidden, attentionMask: attentionMask)

        // The projection is linear and bias-free, so pooling first is exact and cheaper.
        let pooled = meanPooling(hiddenStates: hidden, attentionMask: attentionMask)
        let projected = languageModel.embeddingProjection(pooled.asType(hidden.dtype))
        let vector = projected.asType(.float32)
        let norm = maximum(
            MLX.sqrt((vector * vector).sum(axis: -1, keepDims: true)), MLXArray(1e-9))
        return vector / norm
    }

    /// Keeps the text model and the encoders this configuration loads. PyTorch checkpoints
    /// store convolution kernels channels-first; they move to the channels-last layout
    /// of MLX.
    public func sanitize(weights: [String: MLXArray]) throws -> [String: MLXArray] {
        var prefixes = ["language_model."]
        if visionTower != nil { prefixes += ["vision_tower.", "embed_vision."] }
        if audioTower != nil { prefixes += ["audio_tower.", "embed_audio."] }
        let shapes = Dictionary(
            uniqueKeysWithValues: parameters().flattened().map { ($0.0, $0.1.shape) })
        var clean: [String: MLXArray] = [:]
        for (originalKey, value) in weights {
            let textPrefixes = [
                "embed_tokens.", "embedding_projection.", "ple.", "layers.", "norm.",
            ]
            let key =
                textPrefixes.contains(where: originalKey.hasPrefix)
                ? "language_model." + originalKey : originalKey
            guard prefixes.contains(where: key.hasPrefix) else { continue }
            if let shape = shapes[key], value.ndim > 2, value.shape != shape {
                let channelsLast = value.movedAxis(source: 1, destination: -1)
                clean[key] = channelsLast.shape == shape ? channelsLast : value
            } else {
                clean[key] = value
            }
        }
        return clean.mapValues { $0.dtype == .float16 ? $0.asType(.float32) : $0 }
    }
}

/// Mean pooling over the attention mask; every token counts when no mask is given.
private func meanPooling(hiddenStates: MLXArray, attentionMask: MLXArray?) -> MLXArray {
    guard let mask = attentionMask else { return hiddenStates.mean(axis: 1) }
    let expanded = mask.expandedDimensions(axes: [2]).asType(.float32)
    let sum = (hiddenStates.asType(.float32) * expanded).sum(axis: 1)
    return sum / maximum(expanded.sum(axis: 1), MLXArray(1e-9))
}

// MARK: - Text Encoder

/// The bidirectional text model with projection-only per-layer embeddings (PLE).
private final class EmbeddingGemma2TextModel: Module {

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "ple") var ple: EmbeddingGemma2PLE
    @ModuleInfo(key: "layers") var layers: [EmbeddingGemma2Layer]
    @ModuleInfo(key: "norm") var norm: EmbeddingGemma2RMSNorm
    @ModuleInfo(key: "embedding_projection") var embeddingProjection: Linear

    private let slidingWindow: Int

    init(_ config: EmbeddingGemma2Configuration) {
        self.slidingWindow = config.slidingWindow
        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
        self._ple.wrappedValue = EmbeddingGemma2PLE(config)
        self._layers.wrappedValue = config.attention.map {
            EmbeddingGemma2Layer(config, attention: $0)
        }
        self._norm.wrappedValue = EmbeddingGemma2RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._embeddingProjection.wrappedValue = Linear(
            config.hiddenSize, config.embeddingDim, bias: false)
    }

    func callAsFunction(_ x: MLXArray, attentionMask: MLXArray?) -> MLXArray {
        let perLayerInputs = ple(x)

        let (batch, length) = (x.dim(0), x.dim(1))
        let global = EmbeddingGemma2Span.full(
            EmbeddingGemma2Masks.bidirectional(
                batch: batch, seqLen: length, paddingMask: attentionMask))
        // The window equals the full span while the sequence fits inside it. Blocks score
        // `3 * slidingWindow` keys per query and pay off once that is less than the square.
        let blocks = (length + slidingWindow - 1) / slidingWindow
        let local: EmbeddingGemma2Span =
            if 3 * blocks * slidingWindow * slidingWindow < length * length {
                .window(radius: slidingWindow, validKeys: attentionMask)
            } else if length - 1 > slidingWindow {
                .full(
                    EmbeddingGemma2Masks.combine(
                        pattern: EmbeddingGemma2Masks.slidingWindowPattern(
                            seqLen: length, radius: slidingWindow),
                        batch: batch, seqLen: length, paddingMask: attentionMask))
            } else {
                global
            }

        var hidden = x
        for (index, layer) in layers.enumerated() {
            hidden = layer(
                hidden,
                perLayerInput: perLayerInputs[0..., 0..., index, 0...],
                span: layer.isGlobal ? global : local)
        }
        return norm(hidden)
    }
}

/// Derives every layer's gating signal from the scaled token embeddings alone.
private final class EmbeddingGemma2PLE: Module {
    let layerCount: Int
    let inputSize: Int
    let scale: Float

    @ModuleInfo(key: "per_layer_model_projection") var perLayerModelProjection: Linear
    @ModuleInfo(key: "per_layer_projection_norm") var perLayerProjectionNorm: EmbeddingGemma2RMSNorm

    init(_ config: EmbeddingGemma2Configuration) {
        self.layerCount = config.hiddenLayers
        self.inputSize = config.perLayerInputSize
        self.scale = 1 / Float(config.hiddenSize).squareRoot()
        self._perLayerModelProjection.wrappedValue = Linear(
            config.hiddenSize, layerCount * inputSize, bias: false)
        self._perLayerProjectionNorm.wrappedValue = EmbeddingGemma2RMSNorm(
            dimensions: inputSize, eps: config.rmsNormEps)
    }

    /// - Returns: `[B, L, layers, inputSize]`.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let projected = perLayerModelProjection(x) * scale
        return perLayerProjectionNorm(
            projected.reshaped(x.dim(0), x.dim(1), layerCount, inputSize))
    }
}

private final class EmbeddingGemma2Layer: Module {
    @ModuleInfo(key: "self_attn") var selfAttention: EmbeddingGemma2Attention
    @ModuleInfo(key: "mlp") var mlp: EmbeddingGemma2MLP
    @ModuleInfo(key: "ple_block") var pleBlock: EmbeddingGemma2PLEBlock
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: EmbeddingGemma2RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: EmbeddingGemma2RMSNorm
    @ModuleInfo(key: "pre_feedforward_layernorm") var preFeedforwardLayerNorm:
        EmbeddingGemma2RMSNorm
    @ModuleInfo(key: "post_feedforward_layernorm") var postFeedforwardLayerNorm:
        EmbeddingGemma2RMSNorm
    @ModuleInfo(key: "layer_scalar") var layerScalar: MLXArray

    let isGlobal: Bool

    init(_ config: EmbeddingGemma2Configuration, attention: EmbeddingGemma2Configuration.Attention)
    {
        let (size, eps) = (config.hiddenSize, config.rmsNormEps)
        self.isGlobal = attention.isGlobal
        self._selfAttention.wrappedValue = EmbeddingGemma2Attention(config, attention: attention)
        self._mlp.wrappedValue = EmbeddingGemma2MLP(config)
        self._pleBlock.wrappedValue = EmbeddingGemma2PLEBlock(config)
        self._inputLayerNorm.wrappedValue = EmbeddingGemma2RMSNorm(dimensions: size, eps: eps)
        self._postAttentionLayerNorm.wrappedValue = EmbeddingGemma2RMSNorm(
            dimensions: size, eps: eps)
        self._preFeedforwardLayerNorm.wrappedValue = EmbeddingGemma2RMSNorm(
            dimensions: size, eps: eps)
        self._postFeedforwardLayerNorm.wrappedValue = EmbeddingGemma2RMSNorm(
            dimensions: size, eps: eps)
        self._layerScalar.wrappedValue = MLXArray.ones([1])
    }

    func callAsFunction(_ x: MLXArray, perLayerInput: MLXArray, span: EmbeddingGemma2Span)
        -> MLXArray
    {
        var hidden = x + postAttentionLayerNorm(selfAttention(inputLayerNorm(x), span: span))
        hidden = hidden + postFeedforwardLayerNorm(mlp(preFeedforwardLayerNorm(hidden)))
        return pleBlock(hidden, perLayerInput: perLayerInput) * layerScalar
    }
}

private final class EmbeddingGemma2Attention: Module {
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: EmbeddingGemma2RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: EmbeddingGemma2RMSNorm

    let rope: MLXNN.RoPE
    let heads: Int
    let kvHeads: Int
    let headDim: Int
    private let eps: Float

    init(_ config: EmbeddingGemma2Configuration, attention: EmbeddingGemma2Configuration.Attention)
    {
        self.heads = config.attentionHeads
        self.kvHeads = attention.kvHeads
        self.headDim = attention.headDim
        self.eps = config.rmsNormEps
        self.rope = MLXNN.RoPE(dimensions: headDim, traditional: false, base: attention.ropeBase)
        self._qProj.wrappedValue = Linear(config.hiddenSize, heads * headDim, bias: false)
        self._kProj.wrappedValue = Linear(config.hiddenSize, kvHeads * headDim, bias: false)
        self._vProj.wrappedValue = Linear(config.hiddenSize, kvHeads * headDim, bias: false)
        self._oProj.wrappedValue = Linear(heads * headDim, config.hiddenSize, bias: false)
        self._qNorm.wrappedValue = EmbeddingGemma2RMSNorm(dimensions: headDim, eps: eps)
        self._kNorm.wrappedValue = EmbeddingGemma2RMSNorm(dimensions: headDim, eps: eps)
    }

    func callAsFunction(_ x: MLXArray, span: EmbeddingGemma2Span) -> MLXArray {
        let (batch, length) = (x.dim(0), x.dim(1))
        let q = rope(qNorm(qProj(x).reshaped(batch, length, heads, headDim)).transposed(0, 2, 1, 3))
        let k = rope(
            kNorm(kProj(x).reshaped(batch, length, kvHeads, headDim)).transposed(0, 2, 1, 3))
        let rawValues = vProj(x).reshaped(batch, length, kvHeads, headDim)
        // Values are normalized without a learned scale; the query and key norms replace
        // the 1/sqrt(d) scale.
        let v = MLXFast.rmsNorm(
            rawValues, weight: MLXArray.ones([headDim], dtype: .float32), eps: eps
        )
        .asType(rawValues.dtype)
        .transposed(0, 2, 1, 3)
        let output = span.attention(queries: q, keys: k, values: v)
        return oProj(output.transposed(0, 2, 1, 3).reshaped(batch, length, -1))
    }
}

private final class EmbeddingGemma2MLP: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(_ config: EmbeddingGemma2Configuration) {
        self._gateProj.wrappedValue = Linear(
            config.hiddenSize, config.intermediateSize, bias: false)
        self._upProj.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        self._downProj.wrappedValue = Linear(
            config.intermediateSize, config.hiddenSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(geluApproximate(gateProj(x)) * upProj(x))
    }
}

/// Gates the residual stream with this layer's slice of the per-layer embeddings.
private final class EmbeddingGemma2PLEBlock: Module {
    @ModuleInfo(key: "per_layer_input_gate") var perLayerInputGate: Linear
    @ModuleInfo(key: "per_layer_projection") var perLayerProjection: Linear
    @ModuleInfo(key: "post_per_layer_input_norm") var postPerLayerInputNorm: EmbeddingGemma2RMSNorm

    init(_ config: EmbeddingGemma2Configuration) {
        self._perLayerInputGate.wrappedValue = Linear(
            config.hiddenSize, config.perLayerInputSize, bias: false)
        self._perLayerProjection.wrappedValue = Linear(
            config.perLayerInputSize, config.hiddenSize, bias: false)
        self._postPerLayerInputNorm.wrappedValue = EmbeddingGemma2RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(_ x: MLXArray, perLayerInput: MLXArray) -> MLXArray {
        let gated = geluApproximate(perLayerInputGate(x)) * perLayerInput
        return x + postPerLayerInputNorm(perLayerProjection(gated))
    }
}

/// Gemma 4 RMSNorm: the weight scales directly (no `1 + weight` offset).
private final class EmbeddingGemma2RMSNorm: Module {
    let weight: MLXArray
    let eps: Float

    init(dimensions: Int, eps: Float) {
        self.weight = MLXArray.ones([dimensions])
        self.eps = eps
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(x, weight: weight.asType(.float32), eps: eps).asType(x.dtype)
    }
}

// MARK: - Spans and Masks

/// The keys each query of a layer reads.
enum EmbeddingGemma2Span {

    /// Every key, under an optional `[B, 1, L, L]` mask.
    case full(MLXArray?)

    /// Keys within `radius` on both sides; `validKeys` is `[B, L]`, `1` for real tokens.
    /// Blocks of `radius` queries read only their own keys and `radius` more on each side,
    /// so the cost grows linearly with the sequence instead of with its square.
    case window(radius: Int, validKeys: MLXArray?)

    /// Bidirectional attention with unit scale over `[B, heads, L, headDim]` inputs.
    func attention(queries: MLXArray, keys: MLXArray, values: MLXArray) -> MLXArray {
        switch self {
        case .full(let mask):
            return MLXFast.scaledDotProductAttention(
                queries: queries, keys: keys, values: values, scale: 1,
                mask: mask.map { .array($0) } ?? .none)
        case .window(let radius, let validKeys):
            let (batch, length) = (queries.dim(0), queries.dim(2))
            let blocks = (length + radius - 1) / radius
            let span = 3 * radius
            // Row `b` of `indices` gathers block `b`'s keys from the sequence padded with
            // `radius` positions on both sides.
            let indices =
                (MLXArray(0 ..< blocks).reshaped(blocks, 1) * radius
                + MLXArray(0 ..< span).reshaped(1, span)).flattened()
            let tail = blocks * radius - length
            let context = IntOrPair((radius, radius + tail))
            func blocked(_ x: MLXArray) -> MLXArray {
                MLX.padded(x, widths: [0, 0, context, 0])
                    .take(indices, axis: 2)
                    .reshaped(batch, x.dim(1), blocks, span, x.dim(3))
                    .transposed(0, 2, 1, 3, 4)
                    .reshaped(batch * blocks, x.dim(1), span, x.dim(3))
            }
            let blockQueries = MLX.padded(queries, widths: [0, 0, .init((0, tail)), 0])
                .reshaped(batch, queries.dim(1), blocks, radius, queries.dim(3))
                .transposed(0, 2, 1, 3, 4)
                .reshaped(batch * blocks, queries.dim(1), radius, queries.dim(3))
            // Key `j` of a block sits `j - radius` positions after its first query.
            let row = MLXArray(0 ..< radius).reshaped(radius, 1)
            let column = MLXArray(0 ..< span).reshaped(1, span)
            let valid = MLX.padded(
                (validKeys ?? MLXArray.ones([batch, length])).asType(.bool),
                widths: [0, context], value: MLXArray(false)
            )
            .take(indices, axis: 1)
            .reshaped(batch * blocks, 1, 1, span)
            // Each query keeps itself, so a padded query never has an empty row.
            let mask =
                ((column .>= row) .&& (column .<= row + 2 * radius) .&& valid)
                .|| (column .== row + radius)
            let output = MLXFast.scaledDotProductAttention(
                queries: blockQueries, keys: blocked(keys), values: blocked(values), scale: 1,
                mask: .array(mask))
            return output.reshaped(batch, blocks, queries.dim(1), radius, values.dim(3))
                .transposed(0, 2, 1, 3, 4)
                .reshaped(batch, queries.dim(1), blocks * radius, values.dim(3))[
                    0..., 0..., ..<length, 0...]
        }
    }
}

enum EmbeddingGemma2Masks {

    /// Full bidirectional mask over real tokens; `nil` when nothing is padded. Entirely
    /// padded rows keep attention defined; pooling discards their states.
    static func bidirectional(batch: Int, seqLen: Int, paddingMask: MLXArray?) -> MLXArray? {
        guard let paddingMask else { return nil }
        let validKeys = paddingMask.asType(.bool).reshaped(batch, 1, 1, seqLen)
        return validKeys .|| logicalNot(validKeys.any(axis: -1, keepDims: true))
    }

    /// Keys within `radius` of the query, both endpoints included.
    static func slidingWindowPattern(seqLen: Int, radius: Int) -> MLXArray {
        let rows = MLXArray(0 ..< seqLen).reshaped(seqLen, 1)
        let columns = MLXArray(0 ..< seqLen).reshaped(1, seqLen)
        return (abs(rows - columns) .<= MLXArray(Int32(radius))).reshaped(1, 1, seqLen, seqLen)
    }

    /// Combines the window pattern with a padding mask, keeping the diagonal so a padded
    /// query never has a fully masked attention row.
    static func combine(pattern: MLXArray, batch: Int, seqLen: Int, paddingMask: MLXArray?)
        -> MLXArray
    {
        guard let paddingMask else { return pattern }
        let validKeys = paddingMask.asType(.bool).reshaped(batch, 1, 1, seqLen)
        let diagonal =
            (MLXArray(0 ..< seqLen).reshaped(seqLen, 1)
            .== MLXArray(0 ..< seqLen).reshaped(1, seqLen))
            .reshaped(1, 1, seqLen, seqLen)
        return (pattern .&& validKeys) .|| diagonal
    }
}
