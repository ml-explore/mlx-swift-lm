//
//  SmolVLM2.swift
//  mlx-swift-lm
//
//  Created by Pedro Cuenca on 20/3/25.
//

import CoreImage
import CoreMedia
import Foundation
import MLX
import MLXLMCommon

// MARK: - Configuration and modeling are Idefics3

typealias SmolVLM2Configuration = Idefics3Configuration
typealias SmolVLM2 = Idefics3

// MARK: - SmolVLMProcessor and configuration

public struct SmolVLMProcessorConfiguration: Codable, Sendable {
    public struct Size: Codable, Sendable {
        public let longestEdge: Int
        enum CodingKeys: String, CodingKey {
            case longestEdge = "longest_edge"
        }
    }

    public struct VideoSampling: Codable, Sendable {
        public let fps: Int
        public let maxFrames: Int
        // Intentionally ignoring videoSize because I believe it's still wrong in the config files
        //        public let videoSize: Size

        enum CodingKeys: String, CodingKey {
            case fps
            case maxFrames = "max_frames"
        }
    }

    public let imageMean: [CGFloat]
    public let imageStd: [CGFloat]
    public let size: Size
    public let maxImageSize: Size
    public let videoSampling: VideoSampling
    private let _imageSequenceLength: Int?
    public var imageSequenceLength: Int { _imageSequenceLength ?? 64 }

    init(
        imageMean: [CGFloat], imageStd: [CGFloat], size: Size, maxImageSize: Size,
        videoSampling: VideoSampling, imageSequenceLength: Int?
    ) {
        self.imageMean = imageMean
        self.imageStd = imageStd
        self.size = size
        self.maxImageSize = maxImageSize
        self.videoSampling = videoSampling
        self._imageSequenceLength = imageSequenceLength
    }

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
        case maxImageSize = "max_image_size"
        case videoSampling = "video_sampling"
        case _imageSequenceLength = "image_seq_len"
        case imageSequenceLengthAlias = "image_sequence_length"
        case sequenceLengthAlias = "image_sequence_len"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.imageMean = try container.decode([CGFloat].self, forKey: .imageMean)
        self.imageStd = try container.decode([CGFloat].self, forKey: .imageStd)
        self.size = try container.decode(Size.self, forKey: .size)
        self.maxImageSize = try container.decode(Size.self, forKey: .maxImageSize)
        self.videoSampling = try container.decode(VideoSampling.self, forKey: .videoSampling)
        self._imageSequenceLength =
            try container.decodeIfPresent(Int.self, forKey: ._imageSequenceLength)
            ?? container.decodeIfPresent(Int.self, forKey: .imageSequenceLengthAlias)
            ?? container.decodeIfPresent(Int.self, forKey: .sequenceLengthAlias)
        if let _imageSequenceLength, _imageSequenceLength <= 0 {
            throw DecodingError.dataCorruptedError(
                forKey: ._imageSequenceLength,
                in: container,
                debugDescription: "image sequence length must be positive"
            )
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(imageMean, forKey: .imageMean)
        try container.encode(imageStd, forKey: .imageStd)
        try container.encode(size, forKey: .size)
        try container.encode(maxImageSize, forKey: .maxImageSize)
        try container.encode(videoSampling, forKey: .videoSampling)
        try container.encodeIfPresent(_imageSequenceLength, forKey: ._imageSequenceLength)
    }
}

public struct SmolVLM2MessageGenerator: MessageGenerator {
    public init() {}

    private func content(text: String, imageCount: Int, videoCount: Int) -> [MLXLMCommon.Message] {
        (0 ..< imageCount).map { _ in ["type": "image"] }
            + (0 ..< videoCount).map { _ in ["type": "video"] }
            + [["type": "text", "text": text]]
    }

    public func generate(from input: UserInput) -> [MLXLMCommon.Message] {
        switch input.prompt {
        case .text(let text):
            [
                [
                    "role": "user",
                    "content": content(
                        text: text, imageCount: input.images.count, videoCount: input.videos.count),
                ]
            ]
        case .messages(let messages):
            messages
        case .chat(let messages):
            generate(messages: messages)
        }
    }

    public func generate(message: Chat.Message) -> MLXLMCommon.Message {
        [
            "role": message.role.rawValue,
            "content": content(
                text: message.content, imageCount: message.images.count,
                videoCount: message.videos.count),
        ]
    }
}

public struct SmolVLMProcessor: UserInputProcessor {
    private let config: SmolVLMProcessorConfiguration
    private let tokenizer: any Tokenizer

    private let fallbackImageTokenId = 49190
    let imageToken = "<image>"
    let fakeImageToken = "<fake_token_around_image>"
    let globalImageToken = "<global-img>"
    var imageTokenId: Int { tokenizer.convertTokenToId(imageToken) ?? fallbackImageTokenId }

    var maxProcessingImageSize: CGFloat { CGFloat(config.size.longestEdge) }  // 2048
    var fixedImageSize: CGFloat { CGFloat(config.maxImageSize.longestEdge) }  // 384 for big models, 512 for small models (200-500M)
    var imageSequenceLength: Int { config.imageSequenceLength }
    var maxVideoFrames: Int { max(config.videoSampling.maxFrames, 1) }
    var targetVideoFPS: Double { Double(config.videoSampling.fps) }

    let defaultVideoSystemMessage =
        "You are a helpful assistant that can understand videos. Describe what type of video this is and what's happening in it."

    public init(
        _ config: SmolVLMProcessorConfiguration,
        tokenizer: any Tokenizer
    ) {
        self.config = config
        self.tokenizer = tokenizer
    }

    func getVideoPromptString(
        frameCount: Int, timeStamps: [String], videoDuration: String, seqLen: Int,
        fakeToken: String, imageToken: String, globalImageToken: String
    ) -> String {
        var textSplitFrames =
            "You are provided the following series of \(frameCount) frames from a \(videoDuration) [H:MM:SS] video.\n"
        for frameIndex in 0 ..< frameCount {
            textSplitFrames += "\nFrame from \(timeStamps[frameIndex]):"
            textSplitFrames +=
                (fakeToken
                    + globalImageToken
                    + String(repeating: imageToken, count: seqLen)
                    + fakeToken)
        }
        textSplitFrames += "\n\n"
        return textSplitFrames
    }

    func getImagePromptString(
        rows: Int, cols: Int, seqLen: Int, fakeToken: String, imageToken: String,
        globalImageToken: String
    ) -> String {
        /// Prompt with expanded image tokens for when the image is split into patches.
        /// This applies to image processing, not video (I think).
        /// This just transliterates this: https://github.com/huggingface/transformers/blob/6a1ab634b6886b6560b0502e7a305c8cd881732e/src/transformers/models/idefics3/processing_idefics3.py#L44
        var textSplitImages = ""
        for h in 0 ..< rows {
            for w in 0 ..< cols {
                textSplitImages +=
                    (fakeToken
                        + "<row_\(h + 1)_col_\(w + 1)>"
                        + String(repeating: imageToken, count: seqLen))
            }
            textSplitImages += "\n"
        }
        textSplitImages +=
            ("\n"
                + fakeToken
                + globalImageToken
                + String(repeating: imageToken, count: seqLen)
                + fakeToken)
        return textSplitImages
    }

    /// Compute the resize size with `longestEdge` for the given size
    /// If `multiple` is not nil, ensures each side is a multiple of that value
    func aspectRatioSize(for size: CGSize, longestEdge: CGFloat, multiple: CGFloat? = nil) -> CGSize
    {
        let targetSize = MediaProcessing.bestFit(
            size, in: CGSize(width: longestEdge, height: longestEdge))
        guard let multiple = multiple else { return targetSize }
        let aspectRatio = targetSize.width / targetSize.height
        if size.width >= size.height {
            let width = ceil(targetSize.width / multiple) * multiple
            var height = width / aspectRatio
            height = ceil(height / multiple) * multiple
            return CGSize(width: width, height: height)
        } else {
            let height = ceil(targetSize.height / multiple) * multiple
            var width = height * aspectRatio
            width = ceil(width / multiple) * multiple
            return CGSize(width: width, height: height)
        }
    }

    /// Compute the resize size with `longestEdge` for the given size
    /// If `multiple` is not nil, ensures each side is a multiple of that value
    func aspectRatioSize(for size: CGSize, longestEdge: Int, multiple: Int? = nil) -> CGSize {
        return aspectRatioSize(
            for: size, longestEdge: CGFloat(longestEdge), multiple: multiple.flatMap(CGFloat.init))
    }

    /// Tile image if it's larger than the maxProcessingImageSize, so the model gets to see more of it.
    /// Video frames are processed through the fixed-size global-frame path instead.
    func tiles(from originalImage: CIImage) -> (tiles: [CIImage], rows: Int, cols: Int) {
        // The original code resizes to maxProcessingImageSize, then resizes again ensuring multiples of fixedImageSize
        // We do both resizes in one go
        let processingSize = aspectRatioSize(
            for: originalImage.extent.size, longestEdge: maxProcessingImageSize,
            multiple: fixedImageSize)
        let image = MediaProcessing.resampleLanczos(originalImage, to: processingSize)

        var tiles: [CIImage] = []

        // Crop nRows x nCols tiles
        let nRows = Int(ceil(image.extent.size.height / CGFloat(fixedImageSize)))
        let nCols = Int(ceil(image.extent.size.width / CGFloat(fixedImageSize)))

        // Warning: in CIImage, y=0 is the bottom side. We reverse the rows to match the transformers processor
        let tileEdge = Int(fixedImageSize)
        for row in (0 ..< nRows).reversed() {
            for col in 0 ..< nCols {
                let x0 = col * tileEdge
                let y0 = row * tileEdge
                let x1 = min(x0 + tileEdge, Int(image.extent.size.width))
                let y1 = min(y0 + tileEdge, Int(image.extent.size.height))

                let tile = image.cropped(to: CGRect(x: x0, y: y0, width: x1 - x0, height: y1 - y0))
                tiles.append(tile)
            }
        }

        return (tiles, nRows, nCols)
    }

    func formatTimestamp(_ time: CMTime) -> String {
        let totalSeconds = Int(ceil(time.seconds))
        let hours = totalSeconds / 3600
        let minutes = (totalSeconds % 3600) / 60
        let seconds = totalSeconds % 60

        return String(format: "%d:%02d:%02d", hours, minutes, seconds)
    }

    public func prepare(input: UserInput) async throws -> LMInput {
        let messages = SmolVLM2MessageGenerator().generate(from: input)

        if input.images.isEmpty && input.videos.isEmpty {
            // No image scenario
            let promptTokens = try tokenizer.applyChatTemplate(
                messages: messages, tools: input.tools,
                additionalContext: input.additionalContext)
            let tokensArray = MLXArray(promptTokens).expandedDimensions(axis: 0)
            let mask = ones(like: tokensArray)
            return LMInput(text: .init(tokens: tokensArray, mask: mask), image: nil)
        } else if input.images.count > 0 && input.videos.isEmpty {
            // Single image scenario
            guard input.images.count == 1 else {
                throw VLMError.singleImageAllowed
            }

            // Unfortunately we don't have a "render" option in Tokenizers yet, so decoding
            let promptTokens = try tokenizer.applyChatTemplate(
                messages: messages, tools: input.tools,
                additionalContext: input.additionalContext)
            let decoded = tokenizer.decode(tokenIds: promptTokens, skipSpecialTokens: false)

            let image = try input.images[0].asCIImage().toSRGB()
            let (tiles, imageRows, imageCols) = tiles(from: image)

            // Append the resized global image
            // Note we are resampling from the original (potentially larger), not the processing size. It shouldn't make much difference.
            let images =
                tiles + [
                    image.resampled(
                        to: CGSize(width: fixedImageSize, height: fixedImageSize), method: .lanczos)
                ]

            let pixelsForImages = images.map {
                $0.normalized(mean: config.imageMeanTuple, std: config.imageStdTuple).asMLXArray()
            }

            // In transformers we have a batch dim plus the number of images per batch, and they get collapsed inside the model.
            // Here we provide the compact version.
            let pixels = concatenated(pixelsForImages, axis: 0).transposed(0, 2, 3, 1)

            let imagePromptString = getImagePromptString(
                rows: imageRows,
                cols: imageCols,
                seqLen: imageSequenceLength,
                fakeToken: fakeImageToken,
                imageToken: imageToken,
                globalImageToken: globalImageToken
            )

            let prompt = decoded.replacingOccurrences(of: imageToken, with: imagePromptString)
            let finalPromptTokens = tokenizer.encode(text: prompt)

            let promptArray = MLXArray(finalPromptTokens).expandedDimensions(axis: 0)
            let mask = ones(like: promptArray)

            return LMInput(
                text: .init(tokens: promptArray, mask: mask),
                image: .init(pixels: pixels)
            )
        } else {
            // Single video scenario
            guard input.images.count == 0 else {
                throw VLMError.singleMediaTypeAllowed
            }
            guard input.videos.count == 1 else {
                throw VLMError.singleVideoAllowed
            }

            // Insert a default system message if the input doesn't have one
            func messagesWithSystem(_ messages: [MLXLMCommon.Message]) -> [MLXLMCommon.Message] {
                guard messages.filter({ $0["role"] as? String == "system" }).isEmpty else {
                    return messages
                }

                var messagesWithSystem = messages
                messagesWithSystem.insert(
                    [
                        "role": "system",
                        "content": [["type": "text", "text": defaultVideoSystemMessage]],
                    ], at: 0)
                return messagesWithSystem
            }

            // Unfortunately we don't have a "render" option in Tokenizers yet, so decoding
            let promptTokens = try tokenizer.applyChatTemplate(
                messages: messagesWithSystem(messages), tools: input.tools,
                additionalContext: input.additionalContext)
            let decoded = tokenizer.decode(tokenIds: promptTokens, skipSpecialTokens: false)

            let video = input.videos[0]

            let processedFrames = try await MediaProcessing.asProcessedSequence(
                video,
                targetFPS: { duration in
                    // 1 fps for duration >= 10s, apply a multiplier if smaller
                    max((10 - 0.9 * duration.seconds) * targetVideoFPS, 1)
                },
                maxFrames: maxVideoFrames
            ) { frame in

                let processedFrame = frame.frame
                    .toSRGB()
                    .resampled(
                        to: CGSize(width: fixedImageSize, height: fixedImageSize),
                        method: CIImage.ResamplingMethod.lanczos
                    )
                    .normalized(mean: config.imageMeanTuple, std: config.imageStdTuple)
                return VideoFrame(frame: processedFrame, timeStamp: frame.timeStamp)
            }

            let thwFrames = (0 ..< processedFrames.frames.count).map {
                THW($0, Int(fixedImageSize), Int(fixedImageSize))
            }

            let stackedFrames = concatenated(processedFrames.frames, axis: 0)
            let transposedFrames = stackedFrames.transposed(0, 2, 3, 1)

            let videoPromptString = getVideoPromptString(
                frameCount: processedFrames.frames.count,
                timeStamps: processedFrames.timestamps.map(formatTimestamp),
                videoDuration: formatTimestamp(processedFrames.totalDuration),
                seqLen: imageSequenceLength,
                fakeToken: fakeImageToken, imageToken: imageToken,
                globalImageToken: globalImageToken)

            let prompt: String
            if let range = decoded.range(of: "User: ") {
                let before = decoded[..<range.upperBound]
                let after = decoded[range.upperBound...]
                prompt = String(before) + videoPromptString + String(after)
            } else {
                // Fallback if the expected marker is not present
                prompt = decoded + "\n" + videoPromptString
            }
            let finalPromptTokens = tokenizer.encode(text: prompt)

            let promptArray = MLXArray(finalPromptTokens).expandedDimensions(axis: 0)
            let mask = ones(like: promptArray)
            return LMInput(
                text: .init(tokens: promptArray, mask: mask),
                image: .init(pixels: transposedFrames, frames: thwFrames)
            )
        }
    }
}
