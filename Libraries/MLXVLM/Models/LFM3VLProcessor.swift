import AVFoundation
import CoreImage
import Foundation
import MLX
import MLXLMCommon

struct LFM3VLProcessorConfiguration: Decodable, Sendable {
    struct Image: Decodable, Sendable {
        struct Size: Decodable, Sendable {
            let shortest: Int
            let longest: Int
            enum CodingKeys: String, CodingKey {
                case shortest = "shortest_edge"
                case longest = "longest_edge"
            }
        }
        let patchSize: Int
        let temporalPatchSize: Int
        let mergeSize: Int
        let mean: [CGFloat]
        let std: [CGFloat]
        let size: Size
        enum CodingKeys: String, CodingKey {
            case patchSize = "patch_size"
            case temporalPatchSize = "temporal_patch_size"
            case mergeSize = "merge_size"
            case mean = "image_mean"
            case std = "image_std"
            case size
        }
    }
    struct Video: Decodable, Sendable {
        let fps: Double
        let minFrames: Int
        let maxFrames: Int
        let size: Image.Size
        enum CodingKeys: String, CodingKey {
            case fps, size
            case minFrames = "min_frames"
            case maxFrames = "max_frames"
        }
    }
    let image: Image
    let video: Video
    enum CodingKeys: String, CodingKey {
        case image = "image_processor"
        case video = "video_processor"
    }
}

struct LFM3VLMessageGenerator: MessageGenerator {
    func generate(message: Chat.Message) -> Message {
        var result: Message = [
            "role": message.role.rawValue,
            "content": contentParts(for: message, layout: .imagesThenVideosThenText),
        ]
        addReasoning(to: &result, for: message)
        addToolMetadata(to: &result, for: message)
        return result
    }
}

struct LFM3VideoFrameGroup {
    let startTime: Double
    let tokens: [Int]
}

struct LFM3VLProcessor: UserInputProcessor {
    let config: LFM3VLProcessorConfiguration
    let tokenizer: any Tokenizer

    init(_ config: LFM3VLProcessorConfiguration, tokenizer: any Tokenizer) {
        self.config = config
        self.tokenizer = tokenizer
    }

    func prepare(input: UserInput) async throws -> LMInput {
        guard input.audios.isEmpty else {
            throw VLMError.processing("LFM3-VL does not support audio input")
        }
        let messages = LFM3VLMessageGenerator().generate(
            from: input.removingSpecialTokenLabels(using: tokenizer))
        let prompt = try tokenizer.applyChatTemplate(
            messages: messages,
            tools: input.tools, additionalContext: input.additionalContext)
        return try await prepare(input: input, prompt: prompt)
    }

    func prepare(
        input: UserInput, prompt: [Int],
        videoExpansion: ((Int, [LFM3VideoFrameGroup]) throws -> [Int])? = nil
    ) async throws -> LMInput {
        let c = config.image
        guard c.patchSize > 0, c.mergeSize > 0, c.temporalPatchSize > 0,
            c.patchSize <= Int.max / c.mergeSize,
            c.mean.count == 3, c.std.count == 3,
            c.mean.allSatisfy(\.isFinite), c.std.allSatisfy({ $0.isFinite && $0 > 0 }),
            c.size.shortest > 0, c.size.longest >= c.size.shortest
        else {
            throw VLMError.processing("Invalid LFM3-VL image processor configuration")
        }
        guard let imageToken = tokenizer.convertTokenToId("<|image|>") else {
            throw VLMError.processing("LFM3-VL tokenizer has no image token")
        }
        var patches = [MLXArray]()
        var frames = [THW]()
        for image in input.images {
            let source = MediaProcessing.apply(try image.asCIImage(), processing: input.processing)
            let size = source.extent.size
            guard size.width.isFinite, size.height.isFinite, size.width > 0, size.height > 0,
                size.width < CGFloat(Int.max), size.height < CGFloat(Int.max)
            else {
                throw VLMError.processing("LFM3-VL image must have finite positive dimensions")
            }
            let (height, width) = try Self.targetSize(
                height: Int(size.height), width: Int(size.width),
                factor: c.patchSize * c.mergeSize,
                minPixels: input.processing.minPixels ?? c.size.shortest,
                maxPixels: input.processing.maxPixels ?? c.size.longest)
            let (pixels, grid) = try patchify([source], height: height, width: width)
            patches.append(pixels)
            frames.append(grid)
        }
        var videoPatches = [MLXArray]()
        var videoGrids = [THW]()
        var videoExpansions = [[Int]]()
        guard let videoToken = tokenizer.convertTokenToId("<|video|>") else {
            throw VLMError.processing("LFM3-VL tokenizer has no video token")
        }
        switch input.processing.video.sampling {
        case .targetFrames(let count) where count <= 0:
            throw VLMError.processing("LFM3-VL video frame count must be positive")
        case .framesPerSecond(let fps) where !fps.isFinite || fps <= 0:
            throw VLMError.processing("LFM3-VL video frame rate must be finite and positive")
        default: break
        }
        for (videoIndex, video) in input.videos.enumerated() {
            var images = [CIImage]()
            let v = config.video
            guard v.fps.isFinite, v.fps > 0, v.minFrames > 0, v.maxFrames >= v.minFrames else {
                throw VLMError.processing("Invalid LFM3-VL video sampling configuration")
            }
            let sampled = try await MediaProcessing.asProcessedSequence(
                video,
                processing: input.processing.video,
                targetFPS: { duration in
                    let seconds = max(duration.seconds, 0.001)
                    return Double(max(Int(seconds * v.fps), v.minFrames)) / seconds
                }, maxFrames: v.maxFrames,
                frameProcessing: { frame in
                    images.append(try frame.image.asCIImage())
                    return frame
                })
            guard let first = images.first else { throw VLMError.processing("Empty LFM3-VL video") }
            let count = max(images.count, c.temporalPatchSize)
            let (height, width) = try Self.targetSize(
                height: Int(first.extent.height), width: Int(first.extent.width),
                factor: c.patchSize * c.mergeSize,
                frames: count, temporalFactor: c.temporalPatchSize,
                minPixels: input.processing.minPixels ?? v.size.shortest,
                maxPixels: input.processing.maxPixels ?? v.size.longest)

            let (pixels, grid) = try patchify(images, height: height, width: width)
            videoPatches.append(pixels)
            videoGrids.append(grid)
            var groups = [LFM3VideoFrameGroup]()
            for group in 0 ..< grid.t {
                let start = min(group * c.temporalPatchSize, sampled.timestamps.count - 1)
                let end = min(start + c.temporalPatchSize - 1, sampled.timestamps.count - 1)
                let time = (sampled.timestamps[start].seconds + sampled.timestamps[end].seconds) / 2
                guard time.isFinite else {
                    throw VLMError.processing("Invalid LFM3-VL video timestamp")
                }
                var expanded = String(
                    format: "<%.1f seconds>", locale: Locale(identifier: "en_US_POSIX"), time)
                expanded +=
                    "<|vision_start|>"
                    + String(
                        repeating: "<|video|>", count: grid.h * grid.w / c.mergeSize / c.mergeSize)
                    + "<|vision_end|>"
                groups.append(
                    .init(
                        startTime: sampled.timestamps[start].seconds,
                        tokens: tokenizer.encode(text: expanded, addSpecialTokens: false)))
            }
            videoExpansions.append(
                try videoExpansion?(videoIndex, groups) ?? groups.flatMap { $0.tokens })
        }
        var videoIndex = 0
        var tokens = [Int]()
        var imageIndex = 0
        for token in prompt {
            if token == imageToken {
                guard imageIndex < frames.count else {
                    throw VLMError.processing("Missing LFM3-VL image")
                }
                tokens.append(
                    contentsOf: repeatElement(
                        imageToken,
                        count: frames[imageIndex].product / c.mergeSize / c.mergeSize))
                imageIndex += 1
            } else if token == videoToken {
                guard videoIndex < videoExpansions.count else {
                    throw VLMError.processing("Missing LFM3-VL video")
                }
                tokens.append(contentsOf: videoExpansions[videoIndex])
                videoIndex += 1
            } else {
                tokens.append(token)
            }
        }
        guard imageIndex == frames.count, videoIndex == videoGrids.count else {
            throw VLMError.processing("Unused LFM3-VL images")
        }
        return LMInput(
            text: .init(tokens: MLXArray(tokens).expandedDimensions(axis: 0)),
            image: patches.isEmpty ? nil : .init(pixels: concatenated(patches), frames: frames),
            video: videoPatches.isEmpty
                ? nil : .init(pixels: concatenated(videoPatches), frames: videoGrids))
    }
    static func targetSize(
        height: Int, width: Int, factor: Int, frames: Int = 1, temporalFactor: Int = 1,
        minPixels: Int, maxPixels: Int
    ) throws -> (Int, Int) {
        guard height > 0, width > 0, factor > 0, temporalFactor > 0, frames >= temporalFactor,
            minPixels > 0, maxPixels >= minPixels,
            Double(max(height, width)) / Double(min(height, width)) <= 200
        else {
            throw VLMError.processing("Invalid LFM3-VL resize dimensions or pixel budget")
        }
        let h = Double(height)
        let w = Double(width)
        let f = Double(factor)
        let t = Double(frames)
        let roundedFrames =
            (t / Double(temporalFactor)).rounded(.toNearestOrEven) * Double(temporalFactor)
        var roundedH = (h / f).rounded(.toNearestOrEven) * f
        var roundedW = (w / f).rounded(.toNearestOrEven) * f
        if roundedFrames * roundedH * roundedW > Double(maxPixels) {
            let scale = sqrt(t * h * w / Double(maxPixels))
            roundedH = max(f, floor(h / scale / f) * f)
            roundedW = max(f, floor(w / scale / f) * f)
        } else if roundedFrames * roundedH * roundedW < Double(minPixels) {
            let scale = sqrt(Double(minPixels) / (t * h * w))
            roundedH = ceil(h * scale / f) * f
            roundedW = ceil(w * scale / f) * f
        }
        guard roundedH >= f, roundedW >= f,
            roundedH < Double(Int.max), roundedW < Double(Int.max)
        else {
            throw VLMError.processing("Invalid LFM3-VL resized dimensions")
        }
        return (Int(roundedH), Int(roundedW))
    }

    private func patchify(_ images: [CIImage], height: Int, width: Int) throws -> (MLXArray, THW) {
        let c = config.image
        let pixels = images.map { image in
            let resized = MediaProcessing.resampleBicubic(
                MediaProcessing.inSRGBToneCurveSpace(image),
                to: CGSize(width: width, height: height))
            return MediaProcessing.asMLXArray(
                MediaProcessing.normalize(
                    resized,
                    mean: (c.mean[0], c.mean[1], c.mean[2]), std: (c.std[0], c.std[1], c.std[2])))
        }
        return try QwenVL.patchify(
            images: pixels, mergeSize: c.mergeSize,
            patchSize: c.patchSize, temporalPatchSize: c.temporalPatchSize)
    }

}
