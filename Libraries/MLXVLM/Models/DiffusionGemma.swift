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

    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        diffusionCore.newCache(parameters: parameters)
    }

    private func projectedVisionFeatures(
        pixels: MLXArray?,
        outputLength: Int
    ) -> MLXArray? {
        guard let pixels else {
            return nil
        }
        let features = embedVision(visionTower(pixels, outputLength: outputLength))
        return features.reshaped(-1, features.dim(-1))
    }

    private func splitFeatureBlocks(_ features: MLXArray?, sequenceLength: Int) -> [MLXArray] {
        guard let features, sequenceLength > 0 else {
            return []
        }

        let blockCount = features.dim(0) / sequenceLength
        guard blockCount > 0 else {
            return []
        }

        return (0 ..< blockCount).map { blockIndex in
            let start = blockIndex * sequenceLength
            return features[start ..< (start + sequenceLength), 0...]
        }
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

    private func visualBlockLengths(_ multimodalTokenTypes: MLXArray?, matching tokenType: Int32)
        -> [Int]
    {
        guard let multimodalTokenTypes else {
            return []
        }

        let rawTypes =
            multimodalTokenTypes.ndim == 2
            ? multimodalTokenTypes[0, 0...].asArray(Int32.self)
            : multimodalTokenTypes.asArray(Int32.self)

        var result = [Int]()
        var currentLength = 0
        for type in rawTypes {
            if type == tokenType {
                currentLength += 1
            } else if currentLength > 0 {
                result.append(currentLength)
                currentLength = 0
            }
        }
        if currentLength > 0 {
            result.append(currentLength)
        }
        return result
    }

    private func orderedVisionFeatures(
        imageFeatures: MLXArray?,
        videoFeatures: MLXArray?,
        multimodalTokenTypes: MLXArray?,
        videoSequenceLength: Int
    ) -> MLXArray? {
        var imageBlocks = splitFeatureBlocks(
            imageFeatures, sequenceLength: config.visionSoftTokensPerImage)
        var videoBlocks = splitFeatureBlocks(
            videoFeatures,
            sequenceLength: videoSequenceLength)

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

        let imageFeatures = projectedVisionFeatures(
            pixels: input.image?.pixels,
            outputLength: config.visionSoftTokensPerImage)
        let videoSequenceLength =
            visualBlockLengths(input.multimodalTokenTypes, matching: 2).first
            ?? DiffusionGemma4ProcessorConfiguration.defaultVideoSeqLength
        let videoFeatures = projectedVisionFeatures(
            pixels: input.video?.pixels,
            outputLength: videoSequenceLength)
        guard
            let visionFeatures = orderedVisionFeatures(
                imageFeatures: imageFeatures,
                videoFeatures: videoFeatures,
                multimodalTokenTypes: input.multimodalTokenTypes,
                videoSequenceLength: videoSequenceLength)
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
        diffusionCore.prepareDiffusion(
            inputEmbeddings: embeddings,
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

    public func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws
        -> PrepareResult
    {
        throw GenerateError.unsupportedAutoregressiveGeneration(String(describing: Self.self))
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        fatalError(
            "DiffusionGemma is a block-diffusion VLM. Use BlockDiffusionTokenIterator through generate instead of autoregressive next-token decoding."
        )
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized: [String: MLXArray] = [:]

        for (key, value) in diffusionCore.sanitize(weights: weights) {
            sanitized["diffusion_core.\(key)"] = value
        }

        for (key, value) in weights {
            if key.contains("audio_tower") || key.contains("embed_audio")
                || key.contains("rotary_emb")
                || key.contains("input_min")
                || key.contains("input_max")
                || key.contains("output_min")
                || key.contains("output_max")
            {
                continue
            }

            if key.hasPrefix("model.encoder.vision_tower.") {
                let rest = key.dropFirst("model.encoder.".count)
                sanitized[String(rest)] = value
            } else if key.hasPrefix("model.encoder.embed_vision.") {
                let rest = key.dropFirst("model.encoder.".count)
                sanitized[String(rest)] = value
            } else if key.hasPrefix("model.vision_tower.") || key.hasPrefix("model.embed_vision.") {
                let rest = key.dropFirst("model.".count)
                sanitized[String(rest)] = value
            }
        }

        return sanitized
    }
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

    public func preprocess(images: [CIImage], processing: UserInput.Processing?) throws -> (
        MLXArray, THW
    ) {
        var userProcessing = processing ?? UserInput.Processing()
        userProcessing.resize = config.fixedSize

        let processedImages = images.map { image in
            let processedImage = MediaProcessing.apply(image, processing: userProcessing)
            let srgbImage = MediaProcessing.inSRGBToneCurveSpace(processedImage)
            let resizedImage = MediaProcessing.resampleBicubic(srgbImage, to: config.fixedSize)
            let finalImage =
                if config.doNormalize {
                    MediaProcessing.normalize(
                        resizedImage, mean: config.imageMeanTuple, std: config.imageStdTuple)
                } else {
                    resizedImage
                }
            return MediaProcessing.asMLXArray(finalImage)
        }

        return (
            concatenated(processedImages),
            THW(images.count, Int(config.fixedSize.height), Int(config.fixedSize.width))
        )
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

        var imagePixelsAndFrames: [(MLXArray, THW)] = []
        if !input.images.isEmpty {
            imagePixelsAndFrames = try input.images.map {
                try preprocess(images: [$0.asCIImage()], processing: input.processing)
            }
        }

        var videoPixelsAndFrames: [(MLXArray, THW)] = []
        var videoFrameCounts = [Int]()
        if !input.videos.isEmpty {
            for video in input.videos {
                let sequence = try await MediaProcessing.asProcessedSequence(
                    video,
                    targetFPS: { _ in 1.0 },
                    maxFrames: config.videoFrameLimit
                ) { frame in
                    var userProcessing = input.processing
                    userProcessing.resize = config.fixedSize
                    let processedImage = MediaProcessing.apply(
                        frame.frame, processing: userProcessing)
                    let srgbImage = MediaProcessing.inSRGBToneCurveSpace(processedImage)
                    let resizedImage = MediaProcessing.resampleBicubic(
                        srgbImage, to: config.fixedSize)
                    let finalImage =
                        if config.doNormalize {
                            MediaProcessing.normalize(
                                resizedImage,
                                mean: config.imageMeanTuple,
                                std: config.imageStdTuple)
                        } else {
                            resizedImage
                        }
                    return VideoFrame(frame: finalImage, timeStamp: frame.timeStamp)
                }
                videoFrameCounts.append(sequence.frames.count)
                videoPixelsAndFrames += sequence.frames.map {
                    ($0, THW(1, Int(config.fixedSize.height), Int(config.fixedSize.width)))
                }
            }
        }

        var placeholders = imagePixelsAndFrames.map { _ in
            VisualPlaceholder(
                tokenId: config.imageTokenId,
                tokenType: 1,
                blockCount: 1,
                sequenceLength: config.imageSeqLength)
        }
        if let videoTokenId {
            placeholders.append(
                contentsOf: videoFrameCounts.map {
                    VisualPlaceholder(
                        tokenId: videoTokenId,
                        tokenType: 2,
                        blockCount: $0,
                        sequenceLength: config.videoSeqLength)
                })
        }
        let expanded = expandPromptTokens(promptTokens, placeholders: placeholders)

        let promptArray = MLXArray(expanded.tokens).expandedDimensions(axis: 0)
        let mask = ones(like: promptArray).asType(.int8)
        let tokenTypes = MLXArray(expanded.tokenTypes, [1, expanded.tokenTypes.count])

        let imageInput =
            imagePixelsAndFrames.isEmpty
            ? nil
            : LMInput.ProcessedImage(
                pixels: concatenated(imagePixelsAndFrames.map { $0.0 }),
                frames: imagePixelsAndFrames.map { $0.1 })
        let videoInput =
            videoPixelsAndFrames.isEmpty
            ? nil
            : LMInput.ProcessedVideo(
                pixels: concatenated(videoPixelsAndFrames.map { $0.0 }),
                frames: videoPixelsAndFrames.map { $0.1 })

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
    public let size: Gemma3ProcessorConfiguration.ImageSize?
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
        case size
        case videoFrameLimit = "num_frames"
        case maxSoftTokens = "max_soft_tokens"
    }

    private struct NestedImageProcessor: Codable {
        let doNormalize: Bool?
        let imageMean: [CGFloat]?
        let imageStd: [CGFloat]?
        let imageSeqLength: Int?
        let maxSoftTokens: Int?
        let size: Gemma3ProcessorConfiguration.ImageSize?
        let videoFrameLimit: Int?

        enum CodingKeys: String, CodingKey {
            case doNormalize = "do_normalize"
            case imageMean = "image_mean"
            case imageStd = "image_std"
            case imageSeqLength = "image_seq_length"
            case maxSoftTokens = "max_soft_tokens"
            case size
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
        let topLevelSize = try c.decodeIfPresent(
            Gemma3ProcessorConfiguration.ImageSize.self, forKey: .size)

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
        size = imageProcessor?.size ?? topLevelSize
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
        try c.encodeIfPresent(size, forKey: .size)
        try c.encode(videoFrameLimit, forKey: .videoFrameLimit)
    }

    public var imageMeanTuple: (CGFloat, CGFloat, CGFloat) {
        (imageMean[0], imageMean[1], imageMean[2])
    }

    public var imageStdTuple: (CGFloat, CGFloat, CGFloat) {
        (imageStd[0], imageStd[1], imageStd[2])
    }

    public var fixedSize: CGSize {
        if let size {
            return CGSize(width: size.width, height: size.height)
        }
        return CGSize(width: 224, height: 224)
    }
}
