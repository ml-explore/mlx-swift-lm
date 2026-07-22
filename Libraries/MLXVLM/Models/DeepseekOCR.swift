//
//  DeepseekOCR.swift
//  mlx-swift-lm
//
//  Adapted from https://github.com/mzbac/deepseek-ocr.swift (MIT)
//  and refactored into mlx-swift-lm's MLXVLM model interfaces.
//

import CoreImage
import Foundation
import MLX
import MLXLMCommon
import MLXNN

public struct DeepseekOCRConfiguration: Decodable, Sendable {

    public struct VisionConfiguration: Decodable, Sendable {
        public let hiddenSize: Int
        public let outputChannels: Int
        public let numHiddenLayers: Int
        public let numAttentionHeads: Int
        public let numChannels: Int
        public let imageSize: Int
        public let patchSize: Int
        public let layerNormEps: Float
        public let attentionDropout: Float
        public let qkvBias: Bool
        public let useAbsPos: Bool
        public let useRelPos: Bool
        public let windowSize: Int
        public let globalAttentionIndexes: [Int]
        public let mlpDim: Int

        enum CodingKeys: String, CodingKey {
            case hiddenSize = "hidden_size"
            case outputChannels = "output_channels"
            case numHiddenLayers = "num_hidden_layers"
            case numAttentionHeads = "num_attention_heads"
            case numChannels = "num_channels"
            case imageSize = "image_size"
            case patchSize = "patch_size"
            case layerNormEps = "layer_norm_eps"
            case attentionDropout = "attention_dropout"
            case qkvBias = "qkv_bias"
            case useAbsPos = "use_abs_pos"
            case useRelPos = "use_rel_pos"
            case windowSize = "window_size"
            case globalAttentionIndexes = "global_attn_indexes"
            case mlpDim = "mlp_dim"
        }

        public init(from decoder: any Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            self.hiddenSize = try container.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 768
            self.outputChannels =
                try container.decodeIfPresent(Int.self, forKey: .outputChannels) ?? 256
            self.numHiddenLayers =
                try container.decodeIfPresent(Int.self, forKey: .numHiddenLayers) ?? 12
            self.numAttentionHeads =
                try container.decodeIfPresent(Int.self, forKey: .numAttentionHeads) ?? 12
            self.numChannels = try container.decodeIfPresent(Int.self, forKey: .numChannels) ?? 3
            self.imageSize = try container.decodeIfPresent(Int.self, forKey: .imageSize) ?? 1024
            self.patchSize = try container.decodeIfPresent(Int.self, forKey: .patchSize) ?? 16
            self.layerNormEps =
                try container.decodeIfPresent(Float.self, forKey: .layerNormEps) ?? 1e-6
            self.attentionDropout =
                try container.decodeIfPresent(Float.self, forKey: .attentionDropout) ?? 0.0
            self.qkvBias = try container.decodeIfPresent(Bool.self, forKey: .qkvBias) ?? true
            self.useAbsPos = try container.decodeIfPresent(Bool.self, forKey: .useAbsPos) ?? true
            self.useRelPos = try container.decodeIfPresent(Bool.self, forKey: .useRelPos) ?? true
            self.windowSize = try container.decodeIfPresent(Int.self, forKey: .windowSize) ?? 14
            self.globalAttentionIndexes =
                try container.decodeIfPresent([Int].self, forKey: .globalAttentionIndexes)
                ?? [2, 5, 8, 11]
            self.mlpDim = try container.decodeIfPresent(Int.self, forKey: .mlpDim) ?? 3072
        }
    }

    public struct TextConfiguration: Decodable, Sendable {
        public let vocabSize: Int
        public let hiddenSize: Int
        public let intermediateSize: Int
        public let moeIntermediateSize: Int?
        public let numHiddenLayers: Int
        public let numAttentionHeads: Int
        public let numKeyValueHeads: Int
        public let maxPositionEmbeddings: Int
        public let rmsNormEps: Float
        public let ropeTheta: Float
        public let tieWordEmbeddings: Bool
        public let nRoutedExperts: Int?
        public let nSharedExperts: Int?
        public let numExpertsPerTok: Int?
        public let moeLayerFreq: Int?
        public let firstKDenseReplace: Int?
        public let normTopkProb: Bool?
        public let nGroup: Int?
        public let topkGroup: Int?
        public let routedScalingFactor: Float?

        enum CodingKeys: String, CodingKey {
            case vocabSize = "vocab_size"
            case hiddenSize = "hidden_size"
            case intermediateSize = "intermediate_size"
            case moeIntermediateSize = "moe_intermediate_size"
            case numHiddenLayers = "num_hidden_layers"
            case numAttentionHeads = "num_attention_heads"
            case numKeyValueHeads = "num_key_value_heads"
            case maxPositionEmbeddings = "max_position_embeddings"
            case rmsNormEps = "rms_norm_eps"
            case ropeTheta = "rope_theta"
            case tieWordEmbeddings = "tie_word_embeddings"
            case nRoutedExperts = "n_routed_experts"
            case nSharedExperts = "n_shared_experts"
            case numExpertsPerTok = "num_experts_per_tok"
            case moeLayerFreq = "moe_layer_freq"
            case firstKDenseReplace = "first_k_dense_replace"
            case normTopkProb = "norm_topk_prob"
            case nGroup = "n_group"
            case topkGroup = "topk_group"
            case routedScalingFactor = "routed_scaling_factor"
        }

        public init(from decoder: any Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            self.vocabSize = try container.decodeIfPresent(Int.self, forKey: .vocabSize) ?? 129_280
            self.hiddenSize = try container.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 1280
            self.intermediateSize =
                try container.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 6848
            self.moeIntermediateSize = try container.decodeIfPresent(
                Int.self, forKey: .moeIntermediateSize)
            self.numHiddenLayers =
                try container.decodeIfPresent(Int.self, forKey: .numHiddenLayers) ?? 12
            self.numAttentionHeads =
                try container.decodeIfPresent(Int.self, forKey: .numAttentionHeads) ?? 10
            self.numKeyValueHeads =
                try container.decodeIfPresent(Int.self, forKey: .numKeyValueHeads) ?? 10
            self.maxPositionEmbeddings =
                try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 32_768
            self.rmsNormEps = try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
            self.ropeTheta = try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10_000
            self.tieWordEmbeddings =
                try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? true
            self.nRoutedExperts = try container.decodeIfPresent(Int.self, forKey: .nRoutedExperts)
            self.nSharedExperts = try container.decodeIfPresent(Int.self, forKey: .nSharedExperts)
            self.numExpertsPerTok = try container.decodeIfPresent(
                Int.self, forKey: .numExpertsPerTok)
            self.moeLayerFreq = try container.decodeIfPresent(Int.self, forKey: .moeLayerFreq)
            self.firstKDenseReplace = try container.decodeIfPresent(
                Int.self, forKey: .firstKDenseReplace)
            self.normTopkProb = try container.decodeIfPresent(Bool.self, forKey: .normTopkProb)
            self.nGroup = try container.decodeIfPresent(Int.self, forKey: .nGroup)
            self.topkGroup = try container.decodeIfPresent(Int.self, forKey: .topkGroup)
            self.routedScalingFactor =
                try container.decodeIfPresent(Float.self, forKey: .routedScalingFactor)
        }
    }

    public struct BaseConfiguration: Decodable, Sendable {
        public let modelType: String
        private let _imageTokenId: Int?
        public var imageTokenId: Int { _imageTokenId ?? 128_815 }

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case _imageTokenId = "image_token_id"
        }
    }

    public let visionConfiguration: VisionConfiguration
    public let textConfiguration: TextConfiguration
    public let baseConfiguration: BaseConfiguration
    private let _imageSeqLength: Int?
    public var imageSeqLength: Int { _imageSeqLength ?? 576 }

    enum CodingKeys: String, CodingKey {
        case visionConfiguration = "vision_config"
        case textConfiguration = "text_config"
        case languageConfiguration = "language_config"
        case _imageSeqLength = "image_seq_length"
    }

    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.visionConfiguration = try container.decode(
            VisionConfiguration.self, forKey: .visionConfiguration)
        if let text = try container.decodeIfPresent(
            TextConfiguration.self, forKey: .textConfiguration)
        {
            self.textConfiguration = text
        } else {
            self.textConfiguration = try container.decode(
                TextConfiguration.self, forKey: .languageConfiguration)
        }
        self.baseConfiguration = try BaseConfiguration(from: decoder)
        self._imageSeqLength = try container.decodeIfPresent(Int.self, forKey: ._imageSeqLength)
    }
}

public struct DeepseekOCRProcessorConfiguration: Decodable, Sendable {
    public struct Size: Decodable, Sendable {
        private let _shortestEdge: Int?
        private let _longestEdge: Int?

        public var shortestEdge: Int { _shortestEdge ?? 1024 }
        public var longestEdge: Int { _longestEdge ?? 1024 }

        enum CodingKeys: String, CodingKey {
            case _shortestEdge = "shortest_edge"
            case _longestEdge = "longest_edge"
        }

        public init() {
            self._shortestEdge = nil
            self._longestEdge = nil
        }
    }

    public let imageMean: [CGFloat]
    public let imageStd: [CGFloat]
    public let size: Size
    private let _imageSeqLength: Int?

    public var imageSeqLength: Int { _imageSeqLength ?? 576 }
    public var imageMeanTuple: (CGFloat, CGFloat, CGFloat) {
        (imageMean[0], imageMean[1], imageMean[2])
    }
    public var imageStdTuple: (CGFloat, CGFloat, CGFloat) {
        (imageStd[0], imageStd[1], imageStd[2])
    }

    enum CodingKeys: String, CodingKey {
        case imageMean = "image_mean"
        case imageStd = "image_std"
        case size
        case _imageSeqLength = "image_seq_length"
    }

    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.imageMean =
            try container.decodeIfPresent([CGFloat].self, forKey: .imageMean) ?? [0.5, 0.5, 0.5]
        self.imageStd =
            try container.decodeIfPresent([CGFloat].self, forKey: .imageStd) ?? [0.5, 0.5, 0.5]
        self.size = try container.decodeIfPresent(Size.self, forKey: .size) ?? Size()
        self._imageSeqLength = try container.decodeIfPresent(Int.self, forKey: ._imageSeqLength)
    }

    public init() {
        self.imageMean = [0.5, 0.5, 0.5]
        self.imageStd = [0.5, 0.5, 0.5]
        self.size = Size()
        self._imageSeqLength = nil
    }
}

public struct DeepseekOCRProcessor: UserInputProcessor {
    private let config: DeepseekOCRProcessorConfiguration
    private let tokenizer: any Tokenizer

    public init(_ config: DeepseekOCRProcessorConfiguration, tokenizer: any Tokenizer) {
        self.config = config
        self.tokenizer = tokenizer
    }

    private func preprocess(image: CIImage) -> MLXArray {
        image
            .toSRGB()
            .paddingToSquare(backgroundColor: .init(red: 0.5, green: 0.5, blue: 0.5))
            .resampled(
                to: .init(
                    width: config.size.longestEdge,
                    height: config.size.longestEdge),
                method: .bicubic
            )
            .normalized(mean: config.imageMeanTuple, std: config.imageStdTuple)
            .asMLXArray()
    }

    private func imagePromptTokens() -> [Int] {
        tokenizer.encode(text: Array(repeating: "<image>", count: config.imageSeqLength).joined())
    }

    public func prepare(input: UserInput) async throws -> LMInput {
        let messages = DeepseekOCRMessageGenerator().generate(from: input)
        var promptTokens = try tokenizer.applyChatTemplate(
            messages: messages,
            tools: input.tools,
            additionalContext: input.additionalContext)

        if input.images.isEmpty {
            return LMInput(tokens: MLXArray(promptTokens))
        }

        guard input.images.count == 1 else {
            throw VLMError.singleImageAllowed
        }

        if !promptTokens.contains(where: { $0 == DeepseekOCR.defaultImageTokenId }) {
            promptTokens.append(contentsOf: imagePromptTokens())
        }

        let image = MediaProcessing.apply(
            try input.images[0].asCIImage(), processing: input.processing)
        let pixels = preprocess(image: image)
        let tokens = MLXArray(promptTokens).expandedDimensions(axis: 0)
        let mask = ones(like: tokens).asType(.int8)
        return LMInput(
            text: .init(tokens: tokens, mask: mask),
            image: .init(
                pixels: pixels, frames: [THW(1, config.size.longestEdge, config.size.longestEdge)])
        )
    }
}

public struct DeepseekOCRMessageGenerator: MessageGenerator {
    public init() {}

    public func generate(message: Chat.Message) -> MLXLMCommon.Message {
        var dictionary: MLXLMCommon.Message = [
            "role": message.role.rawValue,
            "content": [["type": "text", "text": message.content]]
                + message.images.map { _ in ["type": "image"] },
        ]
        addToolMetadata(to: &dictionary, for: message)
        return dictionary
    }
}

public final class DeepseekOCR: Module, VLMModel, KVCacheDimensionProvider {
    static let defaultImageTokenId = 128_815

    @ModuleInfo(key: "sam_model") private var samModel: VisionEncoder
    @ModuleInfo(key: "vision_model") private var clipModel: ClipVisionModel
    @ModuleInfo(key: "projector") private var projector: MultiModalProjector
    @ModuleInfo(key: "model") private var languageModel: TextModel
    @ModuleInfo(key: "lm_head") private var lmHead: Linear?

    public let config: DeepseekOCRConfiguration
    public var kvHeads: [Int] { languageModel.kvHeads }
    public var loraLayers: [Module] { languageModel.layers }
    public var vocabularySize: Int { config.textConfiguration.vocabSize }

    private var imageNewline: MLXArray
    private var viewSeparator: MLXArray

    public init(_ config: DeepseekOCRConfiguration) {
        self.config = config
        self._samModel.wrappedValue = VisionEncoder(config.visionConfiguration)
        self._clipModel.wrappedValue = ClipVisionModel()
        self._projector.wrappedValue = MultiModalProjector(config)
        self._languageModel.wrappedValue = TextModel(config.textConfiguration)
        if !config.textConfiguration.tieWordEmbeddings {
            self._lmHead.wrappedValue = Linear(
                config.textConfiguration.hiddenSize,
                config.textConfiguration.vocabSize,
                bias: false)
        }

        let std = Float(1.0 / sqrt(Double(config.textConfiguration.hiddenSize)))
        self.imageNewline = MLXRandom.normal([config.textConfiguration.hiddenSize]) * std
        self.viewSeparator = MLXRandom.normal([config.textConfiguration.hiddenSize]) * std
    }

    public func prepare(
        _ input: LMInput,
        cache: [any KVCache],
        state: LMOutput.State?,
        windowSize _: Int?
    ) throws -> PrepareResult {
        let pixels = input.image?.pixels.asType(samModel.patchEmbed.proj.weight.dtype)
        let embeddings: MLXArray
        if let pixels {
            let imageFeatures = getImageFeatures(pixels)
            embeddings = mergeInputIdsWithImageFeatures(
                inputIds: input.text.tokens, imageFeatures: imageFeatures)
        } else {
            embeddings = languageModel.embedTokens(input.text.tokens)
        }
        let logits = computeLogits(languageModel.forward(embeddings, cache: cache))
        return .logits(LMOutput(logits: logits, state: state ?? .init()))
    }

    public func callAsFunction(
        _ input: LMInput.Text,
        cache: [any KVCache]?,
        state: LMOutput.State?
    ) -> LMOutput {
        let logits = computeLogits(languageModel(input.tokens, cache: cache))
        return LMOutput(logits: logits, state: state ?? .init())
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var newWeights = weights

        for layer in 0 ..< config.textConfiguration.numHiddenLayers {
            let prefix = "model.layers.\(layer)"
            for proj in ["gate_proj", "down_proj", "up_proj"] {
                for key in ["weight", "scales", "biases", "bias"] {
                    let firstKey = "\(prefix).mlp.experts.0.\(proj).\(key)"
                    guard weights[firstKey] != nil else { continue }
                    let expertCount = config.textConfiguration.nRoutedExperts ?? 1
                    let stackedExperts = (0 ..< expertCount).compactMap {
                        weights["\(prefix).mlp.experts.\($0).\(proj).\(key)"]
                    }
                    guard !stackedExperts.isEmpty else { continue }
                    newWeights["\(prefix).mlp.switch_mlp.\(proj).\(key)"] = stacked(stackedExperts)
                }
            }
        }

        var result = [String: MLXArray]()
        for (key, value) in newWeights {
            var newKey: String?
            var adjusted = value

            if key == "model.projector.layers.weight" || key == "projector.layers.weight" {
                newKey = "projector.layers.weight"
            } else if key == "model.projector.layers.bias" || key == "projector.layers.bias" {
                newKey = "projector.layers.bias"
            } else if key.hasPrefix("lm_head.") {
                newKey = key
            } else if key.hasPrefix("model.sam_model.") {
                var suffix = String(key.dropFirst("model.sam_model.".count))
                if suffix.hasPrefix("neck.0.") {
                    suffix = suffix.replacingOccurrences(of: "neck.0.", with: "neck.conv1.")
                } else if suffix.hasPrefix("neck.1.") {
                    suffix = suffix.replacingOccurrences(of: "neck.1.", with: "neck.layer_norm1.")
                } else if suffix.hasPrefix("neck.2.") {
                    suffix = suffix.replacingOccurrences(of: "neck.2.", with: "neck.conv2.")
                } else if suffix.hasPrefix("neck.3.") {
                    suffix = suffix.replacingOccurrences(of: "neck.3.", with: "neck.layer_norm2.")
                }
                suffix = suffix.replacingOccurrences(of: "blocks.", with: "layers.")
                newKey = "sam_model.\(suffix)"
            } else if key.hasPrefix("model.vision_model.") {
                newKey = "clip_model.\(key.dropFirst("model.vision_model.".count))"
            } else if key.hasPrefix("model.") {
                guard !key.contains("projector") else { continue }
                guard !key.contains("sam_model") else { continue }
                guard !key.contains("vision_model") else { continue }
                guard !key.contains("image_newline") else { continue }
                guard !key.contains("view_seperator") else { continue }
                newKey = String(key)
            }

            guard let finalKey = newKey, !finalKey.contains("rotary_emb.inv_freq") else { continue }
            if finalKey.contains("proj.weight") || finalKey.contains("conv")
                || finalKey.contains("patch_embed")
            {
                if value.ndim == 4 {
                    adjusted = value.transposed(0, 2, 3, 1)
                }
            }
            result[finalKey] = adjusted
        }

        if config.textConfiguration.tieWordEmbeddings {
            result["lm_head.weight"] = nil
        }

        return result
    }

    private func computeLogits(_ hiddenStates: MLXArray) -> MLXArray {
        if let lmHead {
            return lmHead(hiddenStates)
        }
        return languageModel.embedTokens.asLinear(hiddenStates)
    }

    private func getImageFeatures(_ pixelValues: MLXArray) -> MLXArray {
        let samFeatures = samModel(pixelValues)
        let batchSize = samFeatures.dim(0)
        let samH = samFeatures.dim(1)
        let samW = samFeatures.dim(2)
        let samChannels = samFeatures.dim(3)
        let samFlat = samFeatures.reshaped(batchSize, samH * samW, samChannels)

        let clipFeatures = clipModel(pixelValues, patchEmbeds: samFeatures)
        let clipWithoutCls = clipFeatures[0..., 1..., 0...]
        var features = projector(concatenated([clipWithoutCls, samFlat], axis: -1))

        let tokenCount = features.dim(1)
        let gridSize = Int(sqrt(Double(tokenCount)))
        let hiddenSize = features.dim(2)
        features = features.reshaped(batchSize, gridSize, gridSize, hiddenSize)

        let newline = imageNewline[.newAxis, .newAxis, .newAxis, 0...]
        let newlineBroadcast = broadcast(newline, to: [batchSize, gridSize, 1, hiddenSize])
        features = concatenated([features, newlineBroadcast], axis: 2)
        features = features.reshaped(batchSize, -1, hiddenSize)

        let separator = viewSeparator[.newAxis, .newAxis, 0...]
        let separatorBroadcast = broadcast(separator, to: [batchSize, 1, hiddenSize])
        return concatenated([features, separatorBroadcast], axis: 1)
    }

    private func mergeInputIdsWithImageFeatures(inputIds: MLXArray, imageFeatures: MLXArray)
        -> MLXArray
    {
        let textEmbeddings = languageModel.embedTokens(inputIds)
        let imageMask = inputIds .== config.baseConfiguration.imageTokenId
        let seqLen = textEmbeddings.dim(1)
        let imageTokenCount = imageFeatures.dim(1)
        let firstImageIndex = argMax(imageMask.asType(.int32), axis: 1)[0].item(Int.self)
        let prePad = firstImageIndex
        let postPad = max(0, seqLen - firstImageIndex - imageTokenCount)

        var paddedImageFeatures = imageFeatures
        if prePad > 0 || postPad > 0 {
            paddedImageFeatures = padded(imageFeatures, widths: [0, .init((prePad, postPad)), 0])
        }
        if paddedImageFeatures.dim(1) > seqLen {
            paddedImageFeatures = paddedImageFeatures[0..., ..<seqLen, 0...]
        }

        return which(imageMask[.ellipsis, .newAxis], paddedImageFeatures, textEmbeddings)
    }

    @_spi(Testing)
    public func samFeaturesForTesting(_ pixelValues: MLXArray) -> MLXArray {
        samModel(pixelValues)
    }
}

private final class TextAttention: Module {
    let config: DeepseekOCRConfiguration.TextConfiguration
    let numHeads: Int
    let numKVHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    let rope: RoPE

    init(_ config: DeepseekOCRConfiguration.TextConfiguration) {
        self.config = config
        self.numHeads = config.numAttentionHeads
        self.numKVHeads = config.numKeyValueHeads
        self.headDim = config.hiddenSize / config.numAttentionHeads
        self.scale = pow(Float(headDim), -0.5)
        self._qProj.wrappedValue = Linear(config.hiddenSize, numHeads * headDim, bias: false)
        self._kProj.wrappedValue = Linear(config.hiddenSize, numKVHeads * headDim, bias: false)
        self._vProj.wrappedValue = Linear(config.hiddenSize, numKVHeads * headDim, bias: false)
        self._oProj.wrappedValue = Linear(numHeads * headDim, config.hiddenSize, bias: false)
        self.rope = RoPE(
            dimensions: headDim, traditional: false, base: config.ropeTheta, scale: 1.0)
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let (batch, length) = (x.dim(0), x.dim(1))
        var queries = qProj(x).reshaped(batch, length, numHeads, headDim).transposed(0, 2, 1, 3)
        var keys = kProj(x).reshaped(batch, length, numKVHeads, headDim).transposed(0, 2, 1, 3)
        let values = vProj(x).reshaped(batch, length, numKVHeads, headDim).transposed(0, 2, 1, 3)

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
        .reshaped(batch, length, -1)

        return oProj(output)
    }
}

private final class TextMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "up_proj") var up: Linear
    @ModuleInfo(key: "down_proj") var down: Linear

    init(hiddenSize: Int, intermediateSize: Int) {
        self._gate.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        self._up.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        self._down.wrappedValue = Linear(intermediateSize, hiddenSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        down(silu(gate(x)) * up(x))
    }
}

private final class MoEGate: Module {
    let topK: Int
    let normTopkProb: Bool
    let routedScalingFactor: Float

    @ModuleInfo(key: "weight") var weight: MLXArray

    init(_ config: DeepseekOCRConfiguration.TextConfiguration) {
        let expertCount = config.nRoutedExperts ?? 1
        self.topK = config.numExpertsPerTok ?? 1
        self.normTopkProb = config.normTopkProb ?? false
        self.routedScalingFactor = config.routedScalingFactor ?? 1.0
        self._weight.wrappedValue = zeros([expertCount, config.hiddenSize])
    }

    func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray) {
        let logits = matmul(x.asType(.float32), weight.T.asType(.float32)).asType(x.dtype)
        let scores = softmax(logits, axis: -1)
        let kth = scores.dim(-1) - topK
        let indices = argPartition(scores, kth: kth, axis: -1)[.ellipsis, kth...]
        var selected = takeAlong(scores, indices, axis: -1)
        if normTopkProb {
            selected = selected / (selected.sum(axis: -1, keepDims: true) + MLXArray(1e-20))
        }
        return (indices, selected * routedScalingFactor)
    }
}

private final class SparseMoE: Module, UnaryLayer {
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
    @ModuleInfo(key: "gate") var gate: MoEGate
    @ModuleInfo(key: "shared_experts") var sharedExperts: TextMLP?

    init(_ config: DeepseekOCRConfiguration.TextConfiguration) {
        let expertCount = config.nRoutedExperts ?? 1
        let hiddenSize = config.hiddenSize
        let intermediate = config.moeIntermediateSize ?? config.intermediateSize
        self._switchMLP.wrappedValue = SwitchGLU(
            inputDims: hiddenSize,
            hiddenDims: intermediate,
            numExperts: expertCount)
        self._gate.wrappedValue = MoEGate(config)
        if let shared = config.nSharedExperts {
            self._sharedExperts.wrappedValue = TextMLP(
                hiddenSize: hiddenSize,
                intermediateSize: intermediate * shared)
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (indices, scores) = gate(x)
        var y = switchMLP(x, indices)
        y = weightedExpertSum(y, scores)
        if let sharedExperts {
            y = y + sharedExperts(x)
        }
        return y
    }
}

private final class TextDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttention: TextAttention
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    let mlp: UnaryLayer

    init(_ config: DeepseekOCRConfiguration.TextConfiguration, layerIndex: Int) {
        self._selfAttention.wrappedValue = TextAttention(config)
        self._inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize,
            eps: config.rmsNormEps)

        let useMoE =
            (config.nRoutedExperts ?? 0) > 0
            && layerIndex >= (config.firstKDenseReplace ?? 0)
            && layerIndex % max(config.moeLayerFreq ?? 1, 1) == 0
        if useMoE {
            self.mlp = SparseMoE(config)
        } else {
            self.mlp = TextMLP(
                hiddenSize: config.hiddenSize, intermediateSize: config.intermediateSize)
        }
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let attentionOut = selfAttention(inputLayerNorm(x), mask: mask, cache: cache)
        let hidden = x + attentionOut
        return hidden + mlp(postAttentionLayerNorm(hidden))
    }
}

private final class TextModel: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    let layers: [TextDecoderLayer]
    let norm: RMSNorm
    let kvHeads: [Int]

    init(_ config: DeepseekOCRConfiguration.TextConfiguration) {
        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabSize,
            dimensions: config.hiddenSize)
        self.layers = (0 ..< config.numHiddenLayers).map {
            TextDecoderLayer(config, layerIndex: $0)
        }
        self.norm = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self.kvHeads = (0 ..< config.numHiddenLayers).map { _ in config.numKeyValueHeads }
    }

    func callAsFunction(_ inputIds: MLXArray, cache: [KVCache]?) -> MLXArray {
        forward(embedTokens(inputIds), cache: cache)
    }

    func forward(_ hidden: MLXArray, cache: [KVCache]?) -> MLXArray {
        var hidden = hidden
        let mask = createAttentionMask(h: hidden, cache: cache?.first)
        for (index, layer) in layers.enumerated() {
            hidden = layer(hidden, mask: mask, cache: cache?[index])
        }
        return norm(hidden)
    }
}

private final class VisionMLP: Module {
    @ModuleInfo(key: "lin1") var lin1: Linear
    @ModuleInfo(key: "lin2") var lin2: Linear

    init(hiddenSize: Int, mlpDim: Int) {
        self._lin1.wrappedValue = Linear(hiddenSize, mlpDim)
        self._lin2.wrappedValue = Linear(mlpDim, hiddenSize)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        lin2(gelu(lin1(x)))
    }
}

private final class VisionAttention: Module {
    let numHeads: Int
    let scale: Float
    let windowSize: Int
    let useRelPos: Bool

    @ModuleInfo(key: "qkv") var qkv: Linear
    @ModuleInfo(key: "proj") var proj: Linear

    var relPosH: MLXArray?
    var relPosW: MLXArray?

    init(_ config: DeepseekOCRConfiguration.VisionConfiguration, windowSize: Int) {
        self.numHeads = config.numAttentionHeads
        self.scale = pow(Float(config.hiddenSize / config.numAttentionHeads), -0.5)
        self.windowSize = windowSize
        self.useRelPos = config.useRelPos
        self._qkv.wrappedValue = Linear(
            config.hiddenSize,
            config.hiddenSize * 3,
            bias: config.qkvBias)
        self._proj.wrappedValue = Linear(config.hiddenSize, config.hiddenSize)
        if useRelPos {
            let inputSize = windowSize > 0 ? windowSize : (config.imageSize / config.patchSize)
            let headDim = config.hiddenSize / config.numAttentionHeads
            self.relPosH = zeros([2 * inputSize - 1, headDim])
            self.relPosW = zeros([2 * inputSize - 1, headDim])
        }
    }

    private func getRelPos(qSize: Int, kSize: Int, relPos: MLXArray) -> MLXArray {
        let maxRelDist = 2 * max(qSize, kSize) - 1
        var relPosResized = relPos
        if relPos.dim(0) != maxRelDist {
            let scale = Float(maxRelDist) / Float(relPos.dim(0))
            relPosResized = Upsample(scaleFactor: [scale], mode: .linear(alignCorners: false))(
                relPos.expandedDimensions(axis: 0)
            ).squeezed(axis: 0)
        }
        let qCoords = MLXArray(
            (0 ..< qSize).map { Float($0) * max(Float(kSize) / Float(qSize), 1.0) })
        let kCoords = MLXArray(
            (0 ..< kSize).map { Float($0) * max(Float(qSize) / Float(kSize), 1.0) })
        let offset = Float(kSize - 1) * max(Float(qSize) / Float(kSize), 1.0)
        return relPosResized[
            (qCoords.reshaped(qSize, 1) - kCoords.reshaped(1, kSize) + offset).asType(.int32)]
    }

    private func relativeBias(query: MLXArray, height: Int, width: Int) -> MLXArray? {
        guard useRelPos, let relPosH, let relPosW else { return nil }
        let batchHeads = query.dim(0)
        let dim = query.dim(-1)
        let q = query.reshaped(batchHeads, height, width, dim)
        let relH = getRelPos(qSize: height, kSize: height, relPos: relPosH)
        let relW = getRelPos(qSize: width, kSize: width, relPos: relPosW)
        let relHBias = matmul(q, relH.swappedAxes(-2, -1)).reshaped(
            batchHeads, height * width, height, 1)
        let relWBias = matmul(
            q.transposed(0, 2, 1, 3),
            relW.swappedAxes(-2, -1)
        ).transposed(0, 2, 1, 3).reshaped(batchHeads, height * width, 1, width)
        return (relHBias + relWBias).reshaped(-1, height * width, height * width)
    }

    func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        let (batch, height, width, _) = (
            hiddenStates.dim(0),
            hiddenStates.dim(1),
            hiddenStates.dim(2),
            hiddenStates.dim(3)
        )
        let headDim = hiddenStates.dim(3) / numHeads
        let qkvOut = qkv(hiddenStates)
            .reshaped(batch, height * width, 3, numHeads, headDim)
            .transposed(2, 0, 3, 1, 4)
        let q = qkvOut[0]
        let k = qkvOut[1]
        let v = qkvOut[2]
        let mask: MLXFast.ScaledDotProductAttentionMaskMode
        if let bias = relativeBias(
            query: q.reshaped(batch * numHeads, height * width, headDim), height: height,
            width: width)
        {
            mask = .array(bias.reshaped(batch, numHeads, height * width, height * width))
        } else {
            mask = .none
        }
        let output = MLXFast.scaledDotProductAttention(
            queries: q,
            keys: k,
            values: v,
            scale: scale,
            mask: mask
        )
        return proj(
            output.reshaped(batch, numHeads, height, width, headDim).transposed(0, 2, 3, 1, 4)
                .reshaped(batch, height, width, -1))
    }
}

private final class VisionLayer: Module {
    @ModuleInfo(key: "norm1") var layerNorm1: LayerNorm
    @ModuleInfo(key: "attn") var attention: VisionAttention
    @ModuleInfo(key: "norm2") var layerNorm2: LayerNorm
    @ModuleInfo(key: "mlp") var mlp: VisionMLP

    let windowSize: Int

    init(_ config: DeepseekOCRConfiguration.VisionConfiguration, windowSize: Int) {
        self.windowSize = windowSize
        self._layerNorm1.wrappedValue = LayerNorm(
            dimensions: config.hiddenSize, eps: config.layerNormEps)
        self._attention.wrappedValue = VisionAttention(config, windowSize: windowSize)
        self._layerNorm2.wrappedValue = LayerNorm(
            dimensions: config.hiddenSize, eps: config.layerNormEps)
        self._mlp.wrappedValue = VisionMLP(hiddenSize: config.hiddenSize, mlpDim: config.mlpDim)
    }

    private func partition(_ hiddenStates: MLXArray) -> (MLXArray, (Int, Int), Int) {
        let (batch, height, width, channels) = (
            hiddenStates.dim(0), hiddenStates.dim(1), hiddenStates.dim(2), hiddenStates.dim(3)
        )
        let padH = (windowSize - height % windowSize) % windowSize
        let padW = (windowSize - width % windowSize) % windowSize
        var paddedStates = hiddenStates
        if padH > 0 || padW > 0 {
            paddedStates = padded(
                hiddenStates,
                widths: [.init((0, 0)), .init((0, padH)), .init((0, padW)), .init((0, 0))])
        }
        let paddedHeight = height + padH
        let paddedWidth = width + padW
        let windows =
            paddedStates
            .reshaped(
                batch, paddedHeight / windowSize, windowSize, paddedWidth / windowSize, windowSize,
                channels
            )
            .transposed(0, 1, 3, 2, 4, 5)
            .reshaped(-1, windowSize, windowSize, channels)
        return (windows, (paddedHeight, paddedWidth), batch)
    }

    private func unpartition(
        _ windows: MLXArray,
        padding: (Int, Int),
        original: (Int, Int),
        batchSize: Int
    ) -> MLXArray {
        let channels = windows.dim(-1)
        var merged =
            windows
            .reshaped(
                batchSize, padding.0 / windowSize, padding.1 / windowSize, windowSize, windowSize,
                channels
            )
            .transposed(0, 1, 3, 2, 4, 5)
            .reshaped(batchSize, padding.0, padding.1, channels)
        if padding != original {
            merged = merged[0..., ..<original.0, ..<original.1, 0...]
        }
        return merged
    }

    func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        let residual = hiddenStates
        var hidden = layerNorm1(hiddenStates)
        let original = (hidden.dim(1), hidden.dim(2))
        var padding: (Int, Int)?
        var batchSize: Int?
        if windowSize > 0 {
            let partitioned = partition(hidden)
            hidden = partitioned.0
            padding = partitioned.1
            batchSize = partitioned.2
        }
        hidden = attention(hidden)
        if let padding, let batchSize {
            hidden = unpartition(hidden, padding: padding, original: original, batchSize: batchSize)
        }
        hidden = residual + hidden
        return hidden + mlp(layerNorm2(hidden))
    }
}

private final class PatchEmbeddings: Module {
    @ModuleInfo(key: "proj") var proj: Conv2d

    init(_ config: DeepseekOCRConfiguration.VisionConfiguration) {
        self._proj.wrappedValue = Conv2d(
            inputChannels: config.numChannels,
            outputChannels: config.hiddenSize,
            kernelSize: IntOrPair(config.patchSize),
            stride: IntOrPair(config.patchSize))
    }

    func callAsFunction(_ pixelValues: MLXArray) -> MLXArray {
        proj(pixelValues)
    }
}

private final class VisionNeck: Module {
    @ModuleInfo(key: "conv1") var conv1: Conv2d
    @ModuleInfo(key: "layer_norm1") var layerNorm1: LayerNorm
    @ModuleInfo(key: "conv2") var conv2: Conv2d
    @ModuleInfo(key: "layer_norm2") var layerNorm2: LayerNorm

    init(_ config: DeepseekOCRConfiguration.VisionConfiguration) {
        self._conv1.wrappedValue = Conv2d(
            inputChannels: config.hiddenSize,
            outputChannels: config.outputChannels,
            kernelSize: IntOrPair(1),
            bias: false)
        self._layerNorm1.wrappedValue = LayerNorm(
            dimensions: config.outputChannels, eps: config.layerNormEps)
        self._conv2.wrappedValue = Conv2d(
            inputChannels: config.outputChannels,
            outputChannels: config.outputChannels,
            kernelSize: IntOrPair(3),
            padding: IntOrPair(1),
            bias: false)
        self._layerNorm2.wrappedValue = LayerNorm(
            dimensions: config.outputChannels, eps: config.layerNormEps)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        layerNorm2(conv2(layerNorm1(conv1(x))))
    }
}

private final class VisionEncoder: Module {
    @ModuleInfo(key: "patch_embed") var patchEmbed: PatchEmbeddings
    @ModuleInfo(key: "layers") var layers: [VisionLayer]
    @ModuleInfo(key: "neck") var neck: VisionNeck
    @ModuleInfo(key: "net_2") var net2: Conv2d
    @ModuleInfo(key: "net_3") var net3: Conv2d

    var posEmbed: MLXArray?
    let patchCount: Int

    init(_ config: DeepseekOCRConfiguration.VisionConfiguration) {
        self.patchCount = config.imageSize / config.patchSize
        self._patchEmbed.wrappedValue = PatchEmbeddings(config)
        self._layers.wrappedValue = (0 ..< config.numHiddenLayers).map { index in
            VisionLayer(
                config,
                windowSize: config.globalAttentionIndexes.contains(index) ? 0 : config.windowSize)
        }
        self._neck.wrappedValue = VisionNeck(config)
        self._net2.wrappedValue = Conv2d(
            inputChannels: config.outputChannels,
            outputChannels: 512,
            kernelSize: IntOrPair(3),
            stride: IntOrPair(2),
            padding: IntOrPair(1),
            bias: false)
        self._net3.wrappedValue = Conv2d(
            inputChannels: 512,
            outputChannels: 1024,
            kernelSize: IntOrPair(3),
            stride: IntOrPair(2),
            padding: IntOrPair(1),
            bias: false)
        if config.useAbsPos {
            self.posEmbed = zeros([1, patchCount, patchCount, config.hiddenSize])
        }
    }

    private func interpolatedPositionalEmbedding(_ posEmbed: MLXArray, height: Int, width: Int)
        -> MLXArray
    {
        if posEmbed.dim(1) == height && posEmbed.dim(2) == width {
            return posEmbed
        }
        let scaleH = Float(height) / Float(posEmbed.dim(1))
        let scaleW = Float(width) / Float(posEmbed.dim(2))
        return Upsample(scaleFactor: [scaleH, scaleW], mode: .cubic(alignCorners: false))(posEmbed)
    }

    func callAsFunction(_ pixelValues: MLXArray) -> MLXArray {
        var hidden = patchEmbed(pixelValues)
        if let posEmbed {
            hidden =
                hidden
                + interpolatedPositionalEmbedding(
                    posEmbed, height: hidden.dim(1), width: hidden.dim(2))
        }
        for layer in layers {
            hidden = layer(hidden)
        }
        hidden = neck(hidden)
        hidden = net2(hidden)
        hidden = net3(hidden)
        return hidden
    }
}

private struct ClipVisionConfig {
    let hiddenSize = 1024
    let numLayers = 24
    let numAttentionHeads = 16
    let ffnHiddenSize = 4096
    let imageSize = 224
    let patchSize = 14
    let layerNormEps: Float = 1e-5
}

private func quickGelu(_ x: MLXArray) -> MLXArray {
    x * sigmoid(1.702 * x)
}

private final class ClipVisionEmbeddings: Module {
    @ModuleInfo(key: "patch_embedding") var patchEmbedding: Conv2d

    var classEmbedding: MLXArray
    var positionEmbedding: MLXArray
    let config = ClipVisionConfig()

    override init() {
        self._patchEmbedding.wrappedValue = Conv2d(
            inputChannels: 3,
            outputChannels: config.hiddenSize,
            kernelSize: IntOrPair(config.patchSize),
            stride: IntOrPair(config.patchSize),
            bias: false)
        self.classEmbedding = MLXRandom.normal([config.hiddenSize])
        self.positionEmbedding = MLXRandom.normal([
            1,
            (config.imageSize / config.patchSize) * (config.imageSize / config.patchSize) + 1,
            config.hiddenSize,
        ])
    }

    private func resizedPositionEmbedding(targetSize: Int) -> MLXArray {
        if positionEmbedding.dim(1) == targetSize {
            return positionEmbedding
        }
        let clsToken = positionEmbedding[0..., ..<1, 0...]
        let patchEmbedding = positionEmbedding[0..., 1..., 0...]
        let sourceSize = Int(sqrt(Double(positionEmbedding.dim(1) - 1)))
        let destinationSize = Int(sqrt(Double(targetSize - 1)))
        if sourceSize == destinationSize {
            return positionEmbedding
        }
        let resized = Upsample(
            scaleFactor: [
                Float(destinationSize) / Float(sourceSize),
                Float(destinationSize) / Float(sourceSize),
            ],
            mode: .cubic(alignCorners: false)
        )(patchEmbedding.reshaped(1, sourceSize, sourceSize, config.hiddenSize))
        return concatenated(
            [clsToken, resized.reshaped(1, destinationSize * destinationSize, config.hiddenSize)],
            axis: 1)
    }

    func callAsFunction(_ pixelValues: MLXArray, patchEmbeds: MLXArray?) -> MLXArray {
        let batchSize = pixelValues.dim(0)
        let patches: MLXArray
        if let patchEmbeds {
            patches = patchEmbeds.reshaped(
                batchSize, patchEmbeds.dim(1) * patchEmbeds.dim(2), patchEmbeds.dim(3))
        } else {
            let projected = patchEmbedding(pixelValues)
            patches = projected.reshaped(
                batchSize, projected.dim(1) * projected.dim(2), projected.dim(3))
        }
        let cls = broadcast(
            classEmbedding[.newAxis, .newAxis, 0...], to: [batchSize, 1, config.hiddenSize])
        let embeddings = concatenated([cls, patches], axis: 1)
        return embeddings + resizedPositionEmbedding(targetSize: embeddings.dim(1))
    }
}

private final class ClipFeedForward: Module {
    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear

    override init() {
        let config = ClipVisionConfig()
        self._fc1.wrappedValue = Linear(config.hiddenSize, config.ffnHiddenSize)
        self._fc2.wrappedValue = Linear(config.ffnHiddenSize, config.hiddenSize)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        fc2(quickGelu(fc1(x)))
    }
}

private final class ClipAttention: Module {
    @ModuleInfo(key: "qkv_proj") var qkvProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear

    let config = ClipVisionConfig()
    let scale: Float

    override init() {
        self.scale = pow(Float(config.hiddenSize / config.numAttentionHeads), -0.5)
        self._qkvProj.wrappedValue = Linear(config.hiddenSize, config.hiddenSize * 3)
        self._outProj.wrappedValue = Linear(config.hiddenSize, config.hiddenSize)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (batch, length) = (x.dim(0), x.dim(1))
        let headDim = config.hiddenSize / config.numAttentionHeads
        let qkv = qkvProj(x).reshaped(batch, length, 3, config.numAttentionHeads, headDim)
        let q = qkv[0..., 0..., 0, 0..., 0...].transposed(0, 2, 1, 3)
        let k = qkv[0..., 0..., 1, 0..., 0...].transposed(0, 2, 1, 3)
        let v = qkv[0..., 0..., 2, 0..., 0...].transposed(0, 2, 1, 3)
        let output = MLXFast.scaledDotProductAttention(
            queries: q,
            keys: k,
            values: v,
            scale: scale,
            mask: .none)
        return outProj(output.transposed(0, 2, 1, 3).reshaped(batch, length, -1))
    }
}

private final class ClipTransformerBlock: Module {
    @ModuleInfo(key: "layer_norm1") var layerNorm1: LayerNorm
    @ModuleInfo(key: "self_attn") var selfAttention: ClipAttention
    @ModuleInfo(key: "layer_norm2") var layerNorm2: LayerNorm
    @ModuleInfo(key: "mlp") var mlp: ClipFeedForward

    override init() {
        let config = ClipVisionConfig()
        self._layerNorm1.wrappedValue = LayerNorm(
            dimensions: config.hiddenSize, eps: config.layerNormEps)
        self._selfAttention.wrappedValue = ClipAttention()
        self._layerNorm2.wrappedValue = LayerNorm(
            dimensions: config.hiddenSize, eps: config.layerNormEps)
        self._mlp.wrappedValue = ClipFeedForward()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let hidden = x + selfAttention(layerNorm1(x))
        return hidden + mlp(layerNorm2(hidden))
    }
}

private final class ClipVisionTransformer: Module {
    @ModuleInfo(key: "layers") var layers: [ClipTransformerBlock]

    override init() {
        self._layers.wrappedValue = (0 ..< ClipVisionConfig().numLayers).map { _ in
            ClipTransformerBlock()
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        layers.reduce(x) { partialResult, layer in layer(partialResult) }
    }
}

private final class ClipVisionModel: Module {
    @ModuleInfo(key: "embeddings") var embeddings: ClipVisionEmbeddings
    @ModuleInfo(key: "pre_layrnorm") var preLayerNorm: LayerNorm
    @ModuleInfo(key: "transformer") var transformer: ClipVisionTransformer

    override init() {
        let config = ClipVisionConfig()
        self._embeddings.wrappedValue = ClipVisionEmbeddings()
        self._preLayerNorm.wrappedValue = LayerNorm(
            dimensions: config.hiddenSize, eps: config.layerNormEps)
        self._transformer.wrappedValue = ClipVisionTransformer()
    }

    func callAsFunction(_ pixelValues: MLXArray, patchEmbeds: MLXArray?) -> MLXArray {
        transformer(preLayerNorm(embeddings(pixelValues, patchEmbeds: patchEmbeds)))
    }
}

private final class MultiModalProjector: Module {
    @ModuleInfo(key: "layers") var layers: Linear

    init(_ config: DeepseekOCRConfiguration) {
        self._layers.wrappedValue = Linear(2048, config.textConfiguration.hiddenSize)
    }

    func callAsFunction(_ features: MLXArray) -> MLXArray {
        layers(features)
    }
}
