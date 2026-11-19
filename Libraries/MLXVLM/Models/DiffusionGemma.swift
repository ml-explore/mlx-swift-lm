// Copyright © 2026 Apple Inc.

import CoreImage
import Foundation
import MLX
import MLXLMCommon
import MLXNN

private enum DiffusionGemmaVLMError: LocalizedError {
    case audioUnsupported
    case videoTokenUnavailable
    case imageTokenCountMismatch(expectedVisionTokens: Int, actualPromptTokens: Int)

    var errorDescription: String? {
        switch self {
        case .audioUnsupported:
            return
                "DiffusionGemma currently supports text, images, and video frames; audio input is not supported."
        case .videoTokenUnavailable:
            return
                "DiffusionGemma video input requires a tokenizer that defines the <|video|> token."
        case .imageTokenCountMismatch(let expectedVisionTokens, let actualPromptTokens):
            return
                "DiffusionGemma image token count mismatch: vision encoder produced \(expectedVisionTokens) soft tokens, but the prompt contains \(actualPromptTokens) image tokens."
        }
    }
}

public struct DiffusionGemmaVLMConfiguration: Codable, Sendable {
    public let modelType: String
    public let textConfig: DiffusionGemmaTextConfiguration
    public let visionConfig: Gemma4VisionConfiguration
    public let generationConfig: DiffusionGemmaGenerationConfiguration
    public let canvasLength: Int
    public let tieWordEmbeddings: Bool
    public let imageTokenId: Int
    public let audioTokenId: Int?
    public let boiTokenId: Int
    public let eoiTokenId: Int?
    public let visionSoftTokensPerImage: Int

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case textConfig = "text_config"
        case visionConfig = "vision_config"
        case generationConfig = "generation_config"
        case canvasLength = "canvas_length"
        case tieWordEmbeddings = "tie_word_embeddings"
        case imageTokenId = "image_token_id"
        case audioTokenId = "audio_token_id"
        case boiTokenId = "boi_token_id"
        case eoiTokenId = "eoi_token_id"
        case visionSoftTokensPerImage = "vision_soft_tokens_per_image"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "diffusion_gemma"
        textConfig = try c.decode(DiffusionGemmaTextConfiguration.self, forKey: .textConfig)
        visionConfig = try c.decode(Gemma4VisionConfiguration.self, forKey: .visionConfig)
        generationConfig =
            try c.decodeIfPresent(
                DiffusionGemmaGenerationConfiguration.self, forKey: .generationConfig)
            ?? .init()
        canvasLength = try c.decodeIfPresent(Int.self, forKey: .canvasLength) ?? 256
        tieWordEmbeddings =
            try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings)
            ?? textConfig.tieWordEmbeddings
        imageTokenId = try c.decodeIfPresent(Int.self, forKey: .imageTokenId) ?? 258_880
        audioTokenId = try c.decodeIfPresent(Int.self, forKey: .audioTokenId)
        boiTokenId = try c.decodeIfPresent(Int.self, forKey: .boiTokenId) ?? 255_999
        eoiTokenId = try c.decodeIfPresent(Int.self, forKey: .eoiTokenId) ?? 258_882
        visionSoftTokensPerImage =
            try c.decodeIfPresent(Int.self, forKey: .visionSoftTokensPerImage)
            ?? visionConfig.defaultOutputLength
    }

    var textModelConfiguration: DiffusionGemmaConfiguration {
        DiffusionGemmaConfiguration(
            textConfig: textConfig,
            generationConfig: generationConfig,
            canvasLength: canvasLength,
            tieWordEmbeddings: tieWordEmbeddings,
            modelType: modelType)
    }
}

public final class DiffusionGemma: Module, VLMModel, BlockDiffusionLanguageModel,
    KVCacheDimensionProvider
{
    @ModuleInfo(key: "diffusion_core") private var diffusionCore: DiffusionGemmaLanguageCore
    @ModuleInfo(key: "vision_tower") private var visionTower: Gemma4VisionModel
    @ModuleInfo(key: "embed_vision") private var embedVision: Gemma4MultimodalEmbedder

    public let config: DiffusionGemmaVLMConfiguration

    public var vocabularySize: Int { diffusionCore.vocabularySize }
    public var diffusionVocabularySize: Int { diffusionCore.diffusionVocabularySize }
    public var diffusionCanvasLength: Int { diffusionCore.diffusionCanvasLength }
    public var diffusionMaxDenoisingSteps: Int { diffusionCore.diffusionMaxDenoisingSteps }
    public var diffusionEntropyBound: Float { diffusionCore.diffusionEntropyBound }
    public var diffusionTemperatureMin: Float { diffusionCore.diffusionTemperatureMin }
    public var diffusionTemperatureMax: Float { diffusionCore.diffusionTemperatureMax }
    public var diffusionStabilityThreshold: Int { diffusionCore.diffusionStabilityThreshold }
    public var diffusionConfidenceThreshold: Float { diffusionCore.diffusionConfidenceThreshold }
    public var diffusionDefaultMaxTokens: Int? { diffusionCore.diffusionDefaultMaxTokens }
    public var diffusionPrefersLogitsSelfConditioning: Bool {
        diffusionCore.diffusionPrefersLogitsSelfConditioning
    }
    public var kvHeads: [Int] { diffusionCore.kvHeads }
    public var loraLayers: [Module] { diffusionCore.loraLayers }

    public init(_ config: DiffusionGemmaVLMConfiguration) {
        self.config = config
        _diffusionCore.wrappedValue = DiffusionGemmaLanguageCore(config.textModelConfiguration)
        _visionTower.wrappedValue = Gemma4VisionModel(config: config.visionConfig)
        _embedVision.wrappedValue = Gemma4MultimodalEmbedder(
            embeddingDim: config.visionConfig.hiddenSize,
            textHiddenSize: config.textConfig.hiddenSize,
            eps: config.visionConfig.rmsNormEps)
        super.init()
    }

    public func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
        try diffusionCore.newCache(parameters: parameters)
    }

    /// Encodes each image or video frame at its own size, as Gemma 4 does: the processor
    /// pads them onto one canvas and records each real size in `frames`. Neighbouring
    /// items of the same size share one tower call. Returns one `[tokens, D]` block per item.
    private func visionFeatureBlocks(pixels: MLXArray?, frames: [THW]?) -> [MLXArray] {
        guard var pixels else { return [] }
        if pixels.ndim == 3 { pixels = expandedDimensions(pixels, axis: 0) }
        let frames =
            frames
            ?? Array(repeating: THW(1, pixels.dim(2), pixels.dim(3)), count: pixels.dim(0))

        var blocks: [MLXArray] = []
        var start = 0
        while start < frames.count {
            let size = frames[start]
            var end = start + 1
            while end < frames.count, frames[end].h == size.h, frames[end].w == size.w {
                end += 1
            }
            let features = embedVision(
                visionTower(pixels[start ..< end, 0..., ..<size.h, ..<size.w]))
            blocks += (0 ..< features.dim(0)).map { features[$0] }
            start = end
        }
        return blocks
    }

    private func visualBlockTypes(_ multimodalTokenTypes: MLXArray?) -> [Int32] {
        guard let multimodalTokenTypes else {
            return []
        }

        let rawTypes =
            multimodalTokenTypes.ndim == 2
            ? multimodalTokenTypes[0, 0...].asArray(Int32.self)
            : multimodalTokenTypes.asArray(Int32.self)
        var result = [Int32]()
        var previousType: Int32 = 0
        for type in rawTypes {
            if type == 0 {
                previousType = 0
                continue
            }
            if type != previousType {
                result.append(type)
            }
            previousType = type
        }
        return result
    }

    private func orderedVisionFeatures(
        imageBlocks: [MLXArray],
        videoBlocks: [MLXArray],
        multimodalTokenTypes: MLXArray?
    ) -> MLXArray? {
        var imageBlocks = imageBlocks
        var videoBlocks = videoBlocks

        var blocks = [MLXArray]()
        let blockTypes = visualBlockTypes(multimodalTokenTypes)
        if blockTypes.isEmpty {
            blocks.append(contentsOf: imageBlocks)
            blocks.append(contentsOf: videoBlocks)
        } else {
            for blockType in blockTypes {
                switch blockType {
                case 1 where !imageBlocks.isEmpty:
                    blocks.append(imageBlocks.removeFirst())
                case 2 where !videoBlocks.isEmpty:
                    blocks.append(videoBlocks.removeFirst())
                default:
                    continue
                }
            }
            blocks.append(contentsOf: imageBlocks)
            blocks.append(contentsOf: videoBlocks)
        }

        guard !blocks.isEmpty else {
            return nil
        }
        return blocks.count == 1 ? blocks[0] : concatenated(blocks, axis: 0)
    }

    private func inputEmbeddings(_ input: LMInput) throws -> MLXArray {
        let inputIds =
            input.text.tokens.ndim == 1
            ? input.text.tokens.expandedDimensions(axis: 0) : input.text.tokens
        let scatterMask =
            if let multimodalTokenTypes = input.multimodalTokenTypes {
                multimodalTokenTypes .!= 0
            } else {
                inputIds .== config.imageTokenId
            }
        let embeddingIds = MLX.where(scatterMask, MLXArray(config.imageTokenId), inputIds)
        var inputsEmbeds = diffusionCore.inputEmbeddings(inputIds: embeddingIds)

        guard
            let visionFeatures = orderedVisionFeatures(
                imageBlocks: visionFeatureBlocks(
                    pixels: input.image?.pixels, frames: input.image?.frames),
                videoBlocks: visionFeatureBlocks(
                    pixels: input.video?.pixels, frames: input.video?.frames),
                multimodalTokenTypes: input.multimodalTokenTypes)
        else {
            return inputsEmbeds
        }
        let expectedImageTokens = scatterMask.asType(.int32).sum().item(Int.self)

        if expectedImageTokens != visionFeatures.dim(0) {
            throw DiffusionGemmaVLMError.imageTokenCountMismatch(
                expectedVisionTokens: visionFeatures.dim(0),
                actualPromptTokens: expectedImageTokens)
        }

        var imageMaskExpanded = expandedDimensions(scatterMask, axis: -1)
        imageMaskExpanded = broadcast(imageMaskExpanded, to: inputsEmbeds.shape)
        inputsEmbeds = gemma4MaskedScatter(
            inputTensor: inputsEmbeds,
            mask: imageMaskExpanded,
            source: visionFeatures.asType(inputsEmbeds.dtype))
        return inputsEmbeds
    }

    public func prepareDiffusion(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws {
        if input.audio != nil {
            throw DiffusionGemmaVLMError.audioUnsupported
        }
        let embeddings = try inputEmbeddings(input)
        try diffusionCore.prepareDiffusion(
            inputEmbeddings: embeddings,
            attentionMask: input.text.mask,
            multimodalTokenTypes: input.multimodalTokenTypes,
            cache: cache,
            windowSize: windowSize)
    }

    public func acceptDiffusionTokens(_ tokens: MLXArray, cache: [KVCache], windowSize: Int?) {
        diffusionCore.acceptDiffusionTokens(tokens, cache: cache, windowSize: windowSize)
    }

    public func diffusionLogits(
        canvasTokens: MLXArray,
        cache: [KVCache],
        selfConditioningLogits: MLXArray?
    ) -> MLXArray {
        diffusionCore.diffusionLogits(
            canvasTokens: canvasTokens,
            cache: cache,
            selfConditioningLogits: selfConditioningLogits)
    }

    public func diffusionLogits(
        canvasTokens: MLXArray,
        cache: [KVCache],
        selfConditioningEmbeddings: MLXArray?
    ) -> MLXArray {
        diffusionCore.diffusionLogits(
            canvasTokens: canvasTokens,
            cache: cache,
            selfConditioningEmbeddings: selfConditioningEmbeddings)
    }

    public func diffusionSelfConditioningWeight() -> MLXArray? {
        diffusionCore.diffusionSelfConditioningWeight()
    }

    public func diffusionSelfConditioningEmbeddings(logits: MLXArray, weight: MLXArray?) -> MLXArray
    {
        diffusionCore.diffusionSelfConditioningEmbeddings(logits: logits, weight: weight)
    }

    public func prepare(
        _ input: LMInput, cache: [KVCache], state: LMOutput.State?, prefill: PrefillParameters
    ) throws -> PrepareResult {
        throw GenerateError.unsupportedAutoregressiveGeneration(String(describing: Self.self))
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        fatalError(
            "DiffusionGemma is a block-diffusion VLM. Use BlockDiffusionTokenIterator through generate instead of autoregressive next-token decoding."
        )
    }

    public func prepareCheckpoint(_ checkpoint: ModelCheckpoint) throws -> ModelCheckpoint {
        var visionRules: [CheckpointNameMapping.Rule] = [
            .replacePrefix("model.encoder.vision_tower", with: "vision_tower"),
            .replacePrefix("model.encoder.embed_vision", with: "embed_vision"),
            .replacePrefix("model.vision_tower", with: "vision_tower"),
            .replacePrefix("model.embed_vision", with: "embed_vision"),
            .excludeModule("rotary_emb"),
        ]
        if !config.visionConfig.useClippedLinears {
            visionRules += [
                .excludeModule("input_min"), .excludeModule("input_max"),
                .excludeModule("output_min"), .excludeModule("output_max"),
            ]
        }
        let visionMapping = CheckpointNameMapping(visionRules)
        return try DiffusionGemmaLanguageCore.normalizeExpertWeights(checkpoint).mapNames { name in
            guard let mapped = visionMapping.mapName(name) else { return nil }
            if mapped == "vision_tower" || mapped.hasPrefix("vision_tower.")
                || mapped == "embed_vision" || mapped.hasPrefix("embed_vision.")
            {
                return mapped
            }
            let textName =
                mapped.hasPrefix("diffusion_core.")
                ? String(mapped.dropFirst("diffusion_core.".count)) : mapped
            guard let textName = DiffusionGemmaLanguageCore.textCheckpointName(textName) else {
                return nil
            }
            if config.tieWordEmbeddings && (textName == "lm_head" || textName.hasPrefix("lm_head."))
            {
                return nil
            }
            return "diffusion_core.\(textName)"
        }
    }

    public func sanitize(weights: [String: MLXArray]) throws -> [String: MLXArray] {
        try prepareCheckpoint(ModelCheckpoint(weights: weights)).weights
    }

}

// MARK: - Chat conventions

extension DiffusionGemma {
    public var toolCallFormat: ToolCallFormat? { .gemma4 }

    /// DiffusionGemma thinks in the labeled channels of Gemma 4 (see `ReasoningConfig.gemma4`).
    public var reasoningConfig: ReasoningConfig? { .gemma4 }
}

public struct DiffusionGemma4Processor: UserInputProcessor {
    private let config: DiffusionGemma4ProcessorConfiguration
    private let tokenizer: any Tokenizer
    private let videoTokenId: Int?

    public init(_ config: DiffusionGemma4ProcessorConfiguration, tokenizer: any Tokenizer) {
        self.config = config
        self.tokenizer = tokenizer
        self.videoTokenId = tokenizer.convertTokenToId("<|video|>")
    }

    /// Resizes an image or video frame for the Gemma 4 vision tower: it keeps its aspect
    /// ratio and fills at most `maxSoftTokens` pooled cells, as in ``Gemma4Processor``.
    public func preprocess(image: CIImage, processing: UserInput.Processing?, maxSoftTokens: Int)
        -> CIImage
    {
        let processedImage = MediaProcessing.apply(image, processing: processing)
        let srgbImage = MediaProcessing.inSRGBToneCurveSpace(processedImage)
        let targetSize = config.targetSize(
            for: srgbImage.extent.size, maxSoftTokens: maxSoftTokens)
        let resizedImage =
            srgbImage.extent.size == targetSize
            ? srgbImage
            : MediaProcessing.resampleBicubic(srgbImage, to: targetSize)
        guard config.doNormalize else { return resizedImage }
        return MediaProcessing.normalize(
            resizedImage, mean: config.imageMeanTuple, std: config.imageStdTuple)
    }

    /// Zero-pads `[1, C, H, W]` items of different sizes onto the largest canvas. The
    /// frames record each real size, so the model encodes every item on its own.
    private func canvas(_ items: [MLXArray]) -> (pixels: MLXArray, frames: [THW]) {
        let frames = items.map { THW(1, $0.dim(2), $0.dim(3)) }
        let height = frames.map(\.h).max() ?? 0
        let width = frames.map(\.w).max() ?? 0
        let padded = items.map { pixels in
            pixels.dim(2) == height && pixels.dim(3) == width
                ? pixels
                : MLX.padded(
                    pixels,
                    widths: [
                        0, 0, .init((0, height - pixels.dim(2))), .init((0, width - pixels.dim(3))),
                    ])
        }
        return (concatenated(padded), frames)
    }

    private struct VisualPlaceholder {
        let tokenId: Int
        let tokenType: Int32
        let blockCount: Int
        let sequenceLength: Int
    }

    private func appendVisualBlock(
        tokenId: Int,
        tokenType: Int32,
        sequenceLength: Int,
        tokens: inout [Int],
        tokenTypes: inout [Int32]
    ) {
        tokens.append(config.boiTokenId)
        tokenTypes.append(0)
        tokens.append(contentsOf: Array(repeating: tokenId, count: sequenceLength))
        tokenTypes.append(contentsOf: Array(repeating: tokenType, count: sequenceLength))
        if let eoiTokenId = config.eoiTokenId {
            tokens.append(eoiTokenId)
            tokenTypes.append(0)
        }
    }

    private func expandPromptTokens(
        _ promptTokens: [Int],
        placeholders: [VisualPlaceholder]
    ) -> (tokens: [Int], tokenTypes: [Int32]) {
        var expandedTokens: [Int] = []
        var tokenTypes: [Int32] = []
        let placeholdersByToken = Dictionary(grouping: placeholders, by: \.tokenId)
        var placeholderOffsets: [Int: Int] = [:]

        for token in promptTokens {
            let offset = placeholderOffsets[token, default: 0]
            if let tokenPlaceholders = placeholdersByToken[token],
                offset < tokenPlaceholders.count
            {
                let placeholder = tokenPlaceholders[offset]
                for _ in 0 ..< placeholder.blockCount {
                    appendVisualBlock(
                        tokenId: placeholder.tokenId,
                        tokenType: placeholder.tokenType,
                        sequenceLength: placeholder.sequenceLength,
                        tokens: &expandedTokens,
                        tokenTypes: &tokenTypes)
                }
                placeholderOffsets[token] = offset + 1
            } else {
                expandedTokens.append(token)
                tokenTypes.append(0)
            }
        }

        return (expandedTokens, tokenTypes)
    }

    public func prepare(input: UserInput) async throws -> LMInput {
        if !input.audios.isEmpty {
            throw DiffusionGemmaVLMError.audioUnsupported
        }
        if !input.videos.isEmpty && videoTokenId == nil {
            throw DiffusionGemmaVLMError.videoTokenUnavailable
        }

        let messages = Gemma4MessageGenerator().generate(from: input)
        let promptTokens = try tokenizer.applyChatTemplate(
            messages: messages, tools: input.tools,
            additionalContext: input.additionalContext)

        if input.images.isEmpty, input.videos.isEmpty {
            // No mask: one unpadded sequence needs none, and a mask stops prompt-cache reuse.
            return LMInput(tokens: MLXArray(promptTokens).expandedDimensions(axis: 0))
        }

        let images = try input.images.map {
            MediaProcessing.asMLXArray(
                preprocess(
                    image: try $0.asCIImage(), processing: input.processing,
                    maxSoftTokens: config.imageSeqLength))
        }

        var videoFrames: [MLXArray] = []
        var videoPlaceholders: [VisualPlaceholder] = []
        for video in input.videos {
            let sequence = try await MediaProcessing.asProcessedSequence(
                video,
                targetFPS: { _ in 1.0 },
                maxFrames: config.videoFrameLimit,
                frameProcessing: { frame in
                    VideoFrame(
                        image: .ciImage(
                            preprocess(
                                image: try frame.image.asCIImage(), processing: input.processing,
                                maxSoftTokens: config.videoSeqLength)),
                        timeStamp: frame.timeStamp)
                })
            videoFrames += sequence.frames
            if let videoTokenId {
                // Frames of one video share its size, so they share a token count.
                videoPlaceholders.append(
                    VisualPlaceholder(
                        tokenId: videoTokenId,
                        tokenType: 2,
                        blockCount: sequence.frames.count,
                        sequenceLength: sequence.frames.first.map {
                            config.softTokenCount(height: $0.dim(2), width: $0.dim(3))
                        } ?? 0))
            }
        }

        let placeholders =
            images.map {
                VisualPlaceholder(
                    tokenId: config.imageTokenId,
                    tokenType: 1,
                    blockCount: 1,
                    sequenceLength: config.softTokenCount(height: $0.dim(2), width: $0.dim(3)))
            } + videoPlaceholders
        let expanded = expandPromptTokens(promptTokens, placeholders: placeholders)

        let promptArray = MLXArray(expanded.tokens).expandedDimensions(axis: 0)
        let mask = ones(like: promptArray).asType(.int8)
        let tokenTypes = MLXArray(expanded.tokenTypes, [1, expanded.tokenTypes.count])

        var imageInput: LMInput.ProcessedImage?
        if !images.isEmpty {
            let (pixels, frames) = canvas(images)
            imageInput = LMInput.ProcessedImage(pixels: pixels, frames: frames)
        }
        var videoInput: LMInput.ProcessedVideo?
        if !videoFrames.isEmpty {
            let (pixels, frames) = canvas(videoFrames)
            videoInput = LMInput.ProcessedVideo(pixels: pixels, frames: frames)
        }

        return LMInput(
            text: .init(tokens: promptArray, mask: mask),
            image: imageInput,
            video: videoInput,
            multimodalTokenTypes: tokenTypes)
    }
}

public struct DiffusionGemma4ProcessorConfiguration: Codable, Sendable {
    static let defaultVideoSeqLength = 70

    public let processorClass: String
    public let doNormalize: Bool
    public let imageMean: [CGFloat]
    public let imageStd: [CGFloat]
    public let imageSeqLength: Int
    public let videoSeqLength: Int
    public let imageTokenId: Int
    public let boiTokenId: Int
    public let eoiTokenId: Int?
    public let patchSize: Int
    public let poolingKernelSize: Int
    public let videoFrameLimit: Int

    enum CodingKeys: String, CodingKey {
        case processorClass = "processor_class"
        case imageProcessor = "image_processor"
        case videoProcessor = "video_processor"
        case doNormalize = "do_normalize"
        case imageMean = "image_mean"
        case imageStd = "image_std"
        case imageSeqLength = "image_seq_length"
        case imageTokenId = "image_token_id"
        case boiTokenId = "boi_token_id"
        case eoiTokenId = "eoi_token_id"
        case patchSize = "patch_size"
        case poolingKernelSize = "pooling_kernel_size"
        case videoFrameLimit = "num_frames"
        case maxSoftTokens = "max_soft_tokens"
    }

    private struct NestedImageProcessor: Codable {
        let doNormalize: Bool?
        let imageMean: [CGFloat]?
        let imageStd: [CGFloat]?
        let imageSeqLength: Int?
        let maxSoftTokens: Int?
        let patchSize: Int?
        let poolingKernelSize: Int?
        let videoFrameLimit: Int?

        enum CodingKeys: String, CodingKey {
            case doNormalize = "do_normalize"
            case imageMean = "image_mean"
            case imageStd = "image_std"
            case imageSeqLength = "image_seq_length"
            case maxSoftTokens = "max_soft_tokens"
            case patchSize = "patch_size"
            case poolingKernelSize = "pooling_kernel_size"
            case videoFrameLimit = "num_frames"
        }
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let imageProcessor = try c.decodeIfPresent(
            NestedImageProcessor.self, forKey: .imageProcessor)
        let videoProcessor = try c.decodeIfPresent(
            NestedImageProcessor.self, forKey: .videoProcessor)
        let topLevelDoNormalize = try c.decodeIfPresent(Bool.self, forKey: .doNormalize)
        let topLevelImageMean = try c.decodeIfPresent([CGFloat].self, forKey: .imageMean)
        let topLevelImageStd = try c.decodeIfPresent([CGFloat].self, forKey: .imageStd)
        let topLevelImageSeqLength = try c.decodeIfPresent(Int.self, forKey: .imageSeqLength)
        let topLevelPatchSize = try c.decodeIfPresent(Int.self, forKey: .patchSize)
        let topLevelPoolingKernelSize = try c.decodeIfPresent(
            Int.self, forKey: .poolingKernelSize)

        processorClass =
            try c.decodeIfPresent(String.self, forKey: .processorClass)
            ?? "DiffusionGemma4Processor"
        doNormalize = imageProcessor?.doNormalize ?? topLevelDoNormalize ?? false
        imageMean = imageProcessor?.imageMean ?? topLevelImageMean ?? [0.0, 0.0, 0.0]
        imageStd = imageProcessor?.imageStd ?? topLevelImageStd ?? [1.0, 1.0, 1.0]
        imageSeqLength =
            imageProcessor?.imageSeqLength ?? imageProcessor?.maxSoftTokens
            ?? topLevelImageSeqLength ?? 280
        videoSeqLength =
            videoProcessor?.maxSoftTokens
            ?? videoProcessor?.imageSeqLength
            ?? Self.defaultVideoSeqLength
        imageTokenId = try c.decodeIfPresent(Int.self, forKey: .imageTokenId) ?? 258_880
        boiTokenId = try c.decodeIfPresent(Int.self, forKey: .boiTokenId) ?? 255_999
        eoiTokenId = try c.decodeIfPresent(Int.self, forKey: .eoiTokenId) ?? 258_882
        patchSize = imageProcessor?.patchSize ?? topLevelPatchSize ?? 16
        poolingKernelSize = imageProcessor?.poolingKernelSize ?? topLevelPoolingKernelSize ?? 3
        videoFrameLimit = videoProcessor?.videoFrameLimit ?? 32
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(processorClass, forKey: .processorClass)
        try c.encode(doNormalize, forKey: .doNormalize)
        try c.encode(imageMean, forKey: .imageMean)
        try c.encode(imageStd, forKey: .imageStd)
        try c.encode(imageSeqLength, forKey: .imageSeqLength)
        try c.encode(imageTokenId, forKey: .imageTokenId)
        try c.encode(boiTokenId, forKey: .boiTokenId)
        try c.encodeIfPresent(eoiTokenId, forKey: .eoiTokenId)
        try c.encode(patchSize, forKey: .patchSize)
        try c.encode(poolingKernelSize, forKey: .poolingKernelSize)
        try c.encode(videoFrameLimit, forKey: .videoFrameLimit)
    }

    public var imageMeanTuple: (CGFloat, CGFloat, CGFloat) {
        (imageMean[0], imageMean[1], imageMean[2])
    }

    public var imageStdTuple: (CGFloat, CGFloat, CGFloat) {
        (imageStd[0], imageStd[1], imageStd[2])
    }

    /// The size an image or video frame is resized to so it fills at most
    /// `maxSoftTokens` pooled cells and keeps its aspect ratio, as in Gemma 4.
    public func targetSize(for imageSize: CGSize, maxSoftTokens: Int) -> CGSize {
        Gemma4ProcessorConfiguration.aspectPreservingTargetSize(
            for: imageSize, maxSoftTokens: maxSoftTokens, patchSize: patchSize,
            poolingKernelSize: poolingKernelSize)
    }

    /// Soft tokens the vision tower produces for a resized image or frame.
    public func softTokenCount(height: Int, width: Int) -> Int {
        ((height / patchSize) * (width / patchSize)) / (poolingKernelSize * poolingKernelSize)
    }
}
