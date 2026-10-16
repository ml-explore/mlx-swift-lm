// Copyright © 2026 Apple Inc.

import AVFoundation
import CoreImage
import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// The shared EmbeddingGemma 2 checkpoint configuration.
public typealias EmbeddingGemma2Configuration = MLXLMCommon.EmbeddingGemma2Configuration
/// The shared EmbeddingGemma 2 encoder.
public typealias EmbeddingGemma2 = MLXLMCommon.EmbeddingGemma2

// MARK: - Image Preparation

/// Prepares images as the reference `Gemma4ImageProcessor` does: the target size and
/// soft-token budget of ``Gemma4ProcessorConfiguration``, with an antialiased bicubic
/// resize. The Core Image resample moves a resized image's embedding to cosine 0.995
/// from the reference; this one keeps it above 0.9999.
package struct EmbeddingGemma2ImageProcessor: Sendable {

    package let configuration: Gemma4ProcessorConfiguration

    package init(directory: URL) throws {
        self.configuration = try JSONDecoder().decode(
            Gemma4ProcessorConfiguration.self,
            from: try Data(contentsOf: directory.appendingPathComponent("processor_config.json")))
        guard !configuration.doNormalize, configuration.patchSize > 0,
            configuration.poolingKernelSize > 0, configuration.maxSoftTokens > 0
        else {
            throw DecodingError.dataCorrupted(
                .init(
                    codingPath: [],
                    debugDescription: "Unsupported image processor configuration."))
        }
    }

    /// - Returns: `[1, 3, H, W]` pixels in `0...1` resized to the soft-token budget,
    ///   and their soft-token count.
    package func pixels(for image: UserInput.Image) throws -> (pixels: MLXArray, tokens: Int) {
        let image = try Self.prepared(image.asCIImage())
        let target = configuration.aspectPreservingTargetSize(for: image.extent.size)
        let (height, width) = (Int(target.height), Int(target.width))
        return (
            Self.pixels(image, height: height, width: width),
            configuration.softTokenCount(height: height, width: width)
        )
    }

    /// The image turned upright by its EXIF orientation, on the sRGB tone curve, as the
    /// reference reads it. An image decoded with `applyOrientationProperty` carries no
    /// orientation and stays as it is.
    static func prepared(_ image: CIImage) throws -> CIImage {
        let orientation = (image.properties[kCGImagePropertyOrientation as String] as? UInt32)
            .flatMap(CGImagePropertyOrientation.init(rawValue:))
        let oriented = orientation.map(image.oriented) ?? image
        guard !oriented.extent.isEmpty, !oriented.extent.isInfinite else {
            throw EmbeddingGemma2Embedding.Error.invalidMedia
        }
        return MediaProcessing.inSRGBToneCurveSpace(oriented)
    }

    /// `[1, 3, height, width]` pixels in `0...1`.
    static func pixels(_ image: CIImage, height: Int, width: Int) -> MLXArray {
        let pixels = MediaProcessing.asMLXArray(image)
        guard pixels.dim(2) != height || pixels.dim(3) != width else { return pixels }
        return resized(pixels, height: height, width: width)
    }

    /// Separable resize as two matrix products, rounded to 8 bits like the reference's
    /// `uint8` output.
    static func resized(_ pixels: MLXArray, height: Int, width: Int) -> MLXArray {
        let rows = bicubicWeights(input: pixels.dim(2), output: height)
        let columns = bicubicWeights(input: pixels.dim(3), output: width)
        let resized = matmul(matmul(rows, pixels), columns.transposed())
        return round(clip(resized * 255, min: 0, max: 255)) / 255
    }

    /// `[output, input]` interpolation weights of the antialiased bicubic filter
    /// (a = -0.5) of Pillow and torchvision.
    static func bicubicWeights(input: Int, output: Int) -> MLXArray {
        let scale = Double(input) / Double(output)
        let filterScale = max(scale, 1)
        let support = 2 * filterScale
        var weights = [Float](repeating: 0, count: output * input)
        for row in 0 ..< output {
            let center = (Double(row) + 0.5) * scale
            let first = max(Int(center - support + 0.5), 0)
            let last = min(Int(center + support + 0.5), input)
            let taps = (first ..< last).map { cubic((Double($0) - center + 0.5) / filterScale) }
            let total = taps.reduce(0, +)
            for (offset, tap) in taps.enumerated() where total != 0 {
                weights[row * input + first + offset] = Float(tap / total)
            }
        }
        return MLXArray(weights, [output, input])
    }

    private static func cubic(_ x: Double) -> Double {
        let a = -0.5
        let x = abs(x)
        if x < 1 { return ((a + 2) * x - (a + 3)) * x * x + 1 }
        if x < 2 { return (((x - 5) * x + 8) * x - 4) * a }
        return 0
    }
}

// MARK: - Video Preparation

/// Prepares video as the reference `EmbeddingGemma2VideoProcessor` does: one frame per
/// second, spread evenly over at most ``maximumFrames``, each resized like an image to the
/// smaller frame budget. The audio track is not read.
package struct EmbeddingGemma2VideoProcessor: Sendable {

    /// Frames sampled per second of video.
    package let framesPerSecond: Double
    package let maximumFrames: Int
    /// Soft tokens per frame.
    package let budget: Int

    private let images: EmbeddingGemma2ImageProcessor

    private struct Configuration: Decodable {
        struct VideoProcessor: Decodable {
            let fps: Double
            let maxFrames: Int
            let maxSoftTokens: Int
            let overflowStrategy: String
            let addTimestamps: Bool
        }
        let videoProcessor: VideoProcessor
    }

    /// Reads `video_processor` from `processor_config.json`. Throws unless frames spread
    /// evenly and carry no timestamps, the only layout the library builds.
    package init(directory: URL) throws {
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        let video = try decoder.decode(
            Configuration.self,
            from: try Data(contentsOf: directory.appendingPathComponent("processor_config.json"))
        ).videoProcessor
        guard video.overflowStrategy == "uniform", !video.addTimestamps, video.fps > 0,
            video.maxFrames > 0, video.maxSoftTokens > 0, video.fps.isFinite
        else {
            throw DecodingError.dataCorrupted(
                .init(codingPath: [], debugDescription: "Unsupported video sampling."))
        }
        self.framesPerSecond = video.fps
        self.maximumFrames = video.maxFrames
        self.budget = video.maxSoftTokens
        self.images = try EmbeddingGemma2ImageProcessor(directory: directory)
    }

    /// - Returns: `[frames, 3, H, W]` pixels in `0...1`, every frame at the size of the first.
    package nonisolated(nonsending) func frames(for video: UserInput.Video) async throws
        -> MLXArray
    {
        let frames = try await sampledFrames(video).map(EmbeddingGemma2ImageProcessor.prepared)
        guard let first = frames.first else { throw EmbeddingGemma2Embedding.Error.invalidMedia }
        let target = images.configuration.aspectPreservingTargetSize(
            for: first.extent.size, budget: budget)
        let (height, width) = (Int(target.height), Int(target.width))
        let pixels = concatenated(
            frames.map { EmbeddingGemma2ImageProcessor.pixels($0, height: height, width: width) },
            axis: 0)
        try MLX.checkedEval(pixels)
        return pixels
    }

    private nonisolated(nonsending) func sampledFrames(_ video: UserInput.Video) async throws
        -> [CIImage]
    {
        switch video.source {
        case .frames(let frames):
            // Decoded frames carry no frame rate, so all of them count before the cap.
            return try Self.sampledIndices(
                frameCount: frames.count, frameRate: nil, framesPerSecond: framesPerSecond,
                maximumFrames: maximumFrames
            ).map { try frames[$0].image.asCIImage() }
        case .url(let url):
            return try await sampledFrames(AVURLAsset(url: url))
        case .avAsset(let asset):
            return try await sampledFrames(asset)
        }
    }

    private nonisolated(nonsending) func sampledFrames(_ asset: AVAsset) async throws -> [CIImage] {
        guard let track = try await asset.loadTracks(withMediaType: .video).first else {
            throw EmbeddingGemma2Embedding.Error.invalidMedia
        }
        let (range, nominalRate) = try await track.load(.timeRange, .nominalFrameRate)
        let frameRate = Double(nominalRate)
        guard frameRate > 0 else { throw EmbeddingGemma2Embedding.Error.invalidMedia }
        let indices = Self.sampledIndices(
            frameCount: Int((range.duration.seconds * frameRate).rounded()),
            frameRate: frameRate, framesPerSecond: framesPerSecond, maximumFrames: maximumFrames)

        let generator = AVAssetImageGenerator(asset: asset)
        generator.appliesPreferredTrackTransform = true
        generator.requestedTimeToleranceBefore = .zero
        generator.requestedTimeToleranceAfter = .zero
        // The middle of each frame's interval selects it without rounding to a neighbor.
        let times = indices.map {
            CMTime(
                seconds: range.start.seconds + (Double($0) + 0.5) / frameRate,
                preferredTimescale: 90_000)
        }
        var frames: [CIImage] = []
        frames.reserveCapacity(times.count)
        // Like the decoders of the reference, read the decoded RGB values as sRGB.
        let options: [CIImageOption: Any] =
            CGColorSpace(name: CGColorSpace.sRGB).map { [.colorSpace: $0] } ?? [:]
        for await result in generator.images(for: times) {
            if case .success(_, let image, _) = result {
                frames.append(CIImage(cgImage: image, options: options))
            }
        }
        return frames
    }

    /// Indices of the reference `sample_frames`: one frame every `frameRate /
    /// framesPerSecond` frames, then `maximumFrames` spread evenly over them, as
    /// `np.linspace` truncates. Without a frame rate every frame is a candidate.
    static func sampledIndices(
        frameCount: Int, frameRate: Double?, framesPerSecond: Double, maximumFrames: Int
    ) -> [Int] {
        var indices = Array(0 ..< frameCount)
        if let frameRate, frameCount > 0 {
            let step = frameRate / framesPerSecond
            let count = max(1, Int(Double(frameCount) / frameRate * framesPerSecond))
            indices = (0 ..< count).map { min(frameCount - 1, Int(Double($0) * step)) }
        }
        guard indices.count > maximumFrames else { return indices }
        let step = Double(indices.count - 1) / Double(max(maximumFrames - 1, 1))
        return (0 ..< maximumFrames).map { position in
            position > 0 && position == maximumFrames - 1
                ? indices[indices.count - 1] : indices[Int(Double(position) * step)]
        }
    }
}

// MARK: - Audio Preparation

/// Log-mel features as the reference `Gemma4AudioFeatureExtractor` computes them: a
/// semicausal short-time Fourier transform under a periodic Hann window, an HTK mel
/// filter bank, and a log floor. Only frames of real samples are kept, so the audio tower
/// runs without padding, and audio past ``maximumSampleCount`` is dropped.
package struct EmbeddingGemma2AudioProcessor: Sendable {

    /// The reference keeps the first 480,000 samples: 30 seconds at 16 kHz.
    package static let maximumSampleCount = 480_000

    /// Samples per second of the mono audio the features read.
    package let sampleRate: Int
    private let frameLength: Int
    private let hopLength: Int
    private let fftLength: Int
    private let featureSize: Int
    private let melFloor: Float
    /// Periodic Hann window over one frame.
    private let window: [Float]
    /// `[fftLength / 2 + 1, featureSize]` triangular filters, row-major.
    private let melFilters: [Float]

    private struct Configuration: Decodable {
        struct FeatureExtractor: Decodable {
            let featureSize: Int
            let samplingRate: Int
            let frameLength: Int
            let hopLength: Int
            let fftLength: Int
            let minFrequency: Double
            let maxFrequency: Double
            let melFloor: Float
            let preemphasis: Double?
            let inputScaleFactor: Double?
            let perBinMean: [Double]?
            let perBinStddev: [Double]?
        }
        let featureExtractor: FeatureExtractor
    }

    /// Reads `feature_extractor` from `processor_config.json`. Throws when it asks for
    /// pre-emphasis, input scaling or per-bin normalization, which the checkpoint does not use.
    package init(directory: URL) throws {
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        let extractor = try decoder.decode(
            Configuration.self,
            from: try Data(contentsOf: directory.appendingPathComponent("processor_config.json"))
        ).featureExtractor
        guard (extractor.preemphasis ?? 0) == 0, (extractor.inputScaleFactor ?? 1) == 1,
            extractor.perBinMean == nil, extractor.perBinStddev == nil,
            extractor.frameLength > 0, extractor.frameLength <= extractor.fftLength,
            extractor.hopLength > 0, extractor.samplingRate > 0, extractor.featureSize > 0,
            extractor.melFloor.isFinite, extractor.melFloor > 0,
            extractor.minFrequency.isFinite, extractor.maxFrequency.isFinite,
            extractor.minFrequency >= 0, extractor.maxFrequency > extractor.minFrequency,
            extractor.maxFrequency <= Double(extractor.samplingRate) / 2
        else {
            throw DecodingError.dataCorrupted(
                .init(codingPath: [], debugDescription: "Unsupported audio feature extractor."))
        }
        self.sampleRate = extractor.samplingRate
        self.frameLength = extractor.frameLength
        self.hopLength = extractor.hopLength
        self.fftLength = extractor.fftLength
        self.featureSize = extractor.featureSize
        self.melFloor = extractor.melFloor
        self.window = (0 ..< extractor.frameLength).map {
            Float(0.5 - 0.5 * cos(2 * Double.pi * Double($0) / Double(extractor.frameLength)))
        }
        self.melFilters = Self.melFilters(
            bins: extractor.fftLength / 2 + 1, mels: extractor.featureSize,
            minFrequency: extractor.minFrequency, maxFrequency: extractor.maxFrequency,
            sampleRate: extractor.samplingRate)
    }

    /// Decodes mono audio at ``sampleRate``; `.array` sources must already be.
    ///
    /// - Returns: `[frames, featureSize]` log-mel features. Audio shorter than one frame
    ///   has none.
    package nonisolated(nonsending) func features(for audio: UserInput.Audio) async throws
        -> MLXArray
    {
        let samples: MLXArray
        switch audio.source {
        case .array(let array):
            samples = array
        case .url(let url):
            var processing = UserInput.AudioProcessing()
            processing.sampleRate = Double(sampleRate)
            // A new value, not the caller's, crosses into the concurrent decoder.
            samples = try await UserInput.Audio.url(url).asMLXArray(processing: processing)
        }
        guard samples.ndim == 1 else { throw EmbeddingGemma2Embedding.Error.invalidMedia }
        let frames = features(samples: samples)
        // FFT graphs must be evaluated before an async return can switch threads.
        try MLX.checkedEval(frames)
        return frames
    }

    /// `[frames, featureSize]` features of the frames that hold only real samples.
    func features(samples: MLXArray) -> MLXArray {
        let samples = samples[..<min(samples.dim(0), Self.maximumSampleCount)].asType(.float32)
        // Semicausal padding centres the first frame on the first sample.
        let padded = concatenated([MLXArray.zeros([frameLength / 2]), samples])
        // Each reference frame spans one sample more than it transforms.
        let span = frameLength + 1
        guard padded.dim(0) >= span else { return MLXArray.zeros([0, featureSize]) }
        let frames = asStrided(
            padded, [(padded.dim(0) - span) / hopLength + 1, frameLength],
            strides: [hopLength, 1])
        let magnitudes = abs(MLXFFT.rfft(frames * MLXArray(window), n: fftLength, axis: -1))
        let mel = matmul(magnitudes, MLXArray(melFilters, [fftLength / 2 + 1, featureSize]))
        return log(mel + melFloor)
    }

    /// The reference `mel_filter_bank` with HTK mels and no normalization.
    static func melFilters(
        bins: Int, mels: Int, minFrequency: Double, maxFrequency: Double, sampleRate: Int
    ) -> [Float] {
        func mel(_ hertz: Double) -> Double { 2595 * log10(1 + hertz / 700) }
        func hertz(_ mel: Double) -> Double { 700 * (pow(10, mel / 2595) - 1) }
        let (low, high) = (mel(minFrequency), mel(maxFrequency))
        let edges = (0 ... mels + 1).map {
            hertz(low + Double($0) * (high - low) / Double(mels + 1))
        }
        let nyquist = Double(sampleRate / 2)
        var filters = [Float](repeating: 0, count: bins * mels)
        for bin in 0 ..< bins {
            let frequency = nyquist * Double(bin) / Double(bins - 1)
            for filter in 0 ..< mels {
                let rising = (frequency - edges[filter]) / (edges[filter + 1] - edges[filter])
                let falling =
                    (edges[filter + 2] - frequency) / (edges[filter + 2] - edges[filter + 1])
                filters[bin * mels + filter] = Float(max(0, min(rising, falling)))
            }
        }
        return filters
    }
}

// MARK: - Sequence Layout

/// Builds the token sequence of one input, as the reference processor and chat template do.
package enum EmbeddingGemma2Sequence {

    /// One run of an input: text tokens or the soft tokens of one media item.
    package enum Segment: Equatable, Sendable {
        case text([Int])
        case image(tokens: Int)
        case video(frames: Int, tokensPerFrame: Int)
        case audio(tokens: Int)
    }

    /// - Returns: `<bos>`, the segments in order, then `<eos>`. An image and each video
    ///   frame become `<boi> <placeholder>×n <eoi>`, an audio `<boa> <audio>×n <eoa>`.
    ///   Text loses its last tokens to fit `limit`; media is never cut.
    package static func tokens(
        _ segments: [Segment], configuration: EmbeddingGemma2Configuration,
        beginOfSequence: Int, endOfSequence: Int, limit: Int
    ) throws -> [Int] {
        let blocks = try segments.map { try block($0, configuration) }
        var textBudget = limit - 2 - blocks.reduce(0) { $0 + ($1?.count ?? 0) }
        guard textBudget >= 0 else { throw EmbeddingGemma2Embedding.Error.contextExceeded }
        let placeholders = Set(configuration.softTokenIDs)
        var tokens = [beginOfSequence]
        for (segment, block) in zip(segments, blocks) {
            if let block {
                tokens += block
            } else if case .text(let text) = segment {
                // A placeholder in text would take the soft tokens of a media item.
                if blocks.contains(where: { $0 != nil }),
                    text.contains(where: placeholders.contains)
                {
                    throw EmbeddingGemma2Embedding.Error.placeholderInText
                }
                tokens += text.prefix(textBudget)
                textBudget -= min(text.count, textBudget)
            }
        }
        tokens.append(endOfSequence)
        return tokens
    }

    /// Groups sequences into batches of similar length. A batch pads to its longest row and
    /// holds at most `limit` tokens, padding included; a longer sequence forms a batch alone.
    /// - Returns: Indices into `lengths`, shortest first.
    package static func batches(lengths: [Int], limit: Int) -> [[Int]] {
        let order = lengths.indices.sorted { (lengths[$0], $0) < (lengths[$1], $1) }
        var batches: [[Int]] = []
        for index in order {
            // Ascending order makes this sequence the longest row of the batch.
            if let last = batches.last, (last.count + 1) * lengths[index] <= limit {
                batches[batches.count - 1].append(index)
            } else {
                batches.append([index])
            }
        }
        return batches
    }

    /// The placeholder block of a media segment; `nil` for text.
    private static func block(_ segment: Segment, _ configuration: EmbeddingGemma2Configuration)
        throws -> [Int]?
    {
        func marked(_ begin: Int, _ token: Int, _ count: Int, _ end: Int) -> [Int] {
            [begin] + repeatElement(token, count: count) + [end]
        }
        switch segment {
        case .text:
            return nil
        case .image(let count):
            guard let vision = configuration.vision else {
                throw EmbeddingGemma2Embedding.Error.unsupportedMedia
            }
            return marked(
                vision.beginImageTokenID, vision.imageTokenID, count, vision.endImageTokenID)
        case .video(let frames, let count):
            guard let vision = configuration.vision, let token = vision.videoTokenID else {
                throw EmbeddingGemma2Embedding.Error.unsupportedMedia
            }
            let frame = marked(vision.beginImageTokenID, token, count, vision.endImageTokenID)
            return Array(repeatElement(frame, count: frames).joined())
        case .audio(let count):
            guard let audio = configuration.audio else {
                throw EmbeddingGemma2Embedding.Error.unsupportedMedia
            }
            return marked(
                audio.beginAudioTokenID, audio.audioTokenID, count, audio.endAudioTokenID)
        }
    }
}

// MARK: - Embedding

/// Embeddings of text, images, video and audio from an EmbeddingGemma 2 checkpoint, in
/// one shared space. Load the checkpoint once and reuse this actor for every call.
///
/// ```swift
/// let embeddings = try await EmbeddingGemma2Embedding(
///     modelDirectory: directory, tokenizerLoader: loader)
/// let query = try await embeddings.embed(
///     .init(text: "What causes the northern lights?"), task: .searchQuery)
/// let clip = try await embeddings.embed(
///     .init([.text("Aurora over Tromsø: "), .video(.url(movie)), .audio(.url(narration))]),
///     task: .document)
/// ```
public actor EmbeddingGemma2Embedding: MultimodalEmbedding {

    public typealias Modality = EmbeddingModality
    /// Media keeps its own budget: video samples one frame per second, at most 32, without
    /// the audio track; audio is mono and keeps its first 30 seconds.
    public typealias Input = EmbeddingInput
    public typealias Task = EmbeddingTask

    /// Modalities declared by the checkpoint and allowed by the loading policy.
    /// Encoders and processors load on first use.
    public let supportedModalities: Set<Modality>

    public enum Error: Swift.Error, LocalizedError, Sendable {
        case emptyInput
        case unsupportedMedia
        case invalidMedia
        case placeholderInText
        case contextExceeded
        case invalidEmbedding
        case invalidDimension

        public var errorDescription: String? {
            switch self {
            case .emptyInput:
                "An embedding input must contain text or media."
            case .unsupportedMedia:
                "This checkpoint cannot embed this kind of media."
            case .invalidMedia:
                "A media item could not be decoded."
            case .placeholderInText:
                "Text next to media must not contain media placeholder tokens."
            case .contextExceeded:
                "The embedding input exceeds \(EmbeddingGemma2Configuration.contextLength) tokens."
            case .invalidDimension:
                "Embedding dimensions must be 128, 256, 512, or the checkpoint output size."
            case .invalidEmbedding:
                "The model returned an invalid embedding."
            }
        }
    }

    private let model: EmbeddingGemma2
    /// Tokens in one text batch, padding included. Larger batches use more memory and are
    /// not faster: on an M2, 64 document chunks took 2.45 s with 1,024 tokens per batch,
    /// 2.81 s with 8,192, and 4.20 s one at a time.
    private static let batchTokens = 1_024
    private let tokenizer: any Tokenizer
    private let modelDirectory: URL
    private var imageProcessor: EmbeddingGemma2ImageProcessor?
    private var videoProcessor: EmbeddingGemma2VideoProcessor?
    private var audioProcessor: EmbeddingGemma2AudioProcessor?

    /// - Parameters:
    ///   - modelDirectory: A local `embedding_gemma2` checkpoint directory.
    ///   - tokenizerLoader: Loads the checkpoint's tokenizer.
    ///   - loadVision: Allow image/video inputs; false excludes the vision component.
    ///   - loadAudio: Allow audio inputs; false excludes the audio component.
    ///   - excludedComponents: Components that must never load.
    ///   - componentLoading: Load encoders on first use, or immediately with the text model.
    public init(
        modelDirectory: URL, tokenizerLoader: any TokenizerLoader,
        loadVision: Bool = true, loadAudio: Bool = true,
        excludedComponents: Set<ModelComponent> = [],
        componentLoading: ComponentLoading = .onFirstUse
    ) async throws {
        var configuration = ResolvedModelConfiguration(directory: modelDirectory)
        configuration.excludedComponents = excludedComponents
        if !loadVision { configuration.excludedComponents.insert(.vision) }
        if !loadAudio { configuration.excludedComponents.insert(.audio) }
        try await self.init(
            configuration: configuration, tokenizerLoader: tokenizerLoader,
            componentLoading: componentLoading)
    }

    package init(
        configuration: ResolvedModelConfiguration, tokenizerLoader: any TokenizerLoader,
        componentLoading: ComponentLoading = .onFirstUse
    ) async throws {
        let modelDirectory = configuration.modelDirectory
        let configData = try Data(contentsOf: modelDirectory.appendingPathComponent("config.json"))
        let config = try JSONDecoder.json5().decode(
            EmbeddingGemma2Configuration.self, from: configData)
        let model = EmbeddingGemma2(config)
        let base = try JSONDecoder.json5().decode(BaseConfiguration.self, from: configData)
        try await loadWeights(
            modelDirectory: modelDirectory, model: model,
            perLayerQuantization: base.perLayerQuantization,
            weightFileSelection: configuration.weightFileSelection,
            excludedComponents: configuration.excludedComponents, componentLoading: componentLoading
        )
        try Swift.Task.checkCancellation()
        self.model = model
        self.modelDirectory = modelDirectory
        self.tokenizer = try await tokenizerLoader.load(from: configuration.tokenizerDirectory)
        var modalities: Set<Modality> = [.text]
        if model.supportedComponents.contains(.vision) {
            modalities.insert(.image)
            if config.vision?.videoTokenID != nil { modalities.insert(.video) }
        }
        if model.supportedComponents.contains(.audio) { modalities.insert(.audio) }
        self.supportedModalities = modalities
    }

    /// Embeds independent items in input order; returns unit-length float32 vectors.
    /// Retrieval uses `.searchQuery` for queries and `.document` for corpus items.
    ///
    /// Items with media run one at a time. Each media item's soft tokens are evaluated
    /// before the next one is prepared, so only one encoder's activations are alive at a
    /// time. Text runs in padded batches of similar length, at most 1,024 tokens each;
    /// a longer text runs alone.
    public func embed(
        _ inputs: [Input], task: Task, dimensions: Int? = nil
    ) async throws -> [[Float]] {
        let dimension = try outputDimension(dimensions)
        var vectors = [[Float]](repeating: [], count: inputs.count)
        var texts: [(index: Int, tokens: [Int])] = []
        for (index, input) in inputs.enumerated() {
            try Swift.Task.checkCancellation()
            let (tokens, softTokens) = try await sequence(input, task: task)
            guard let softTokens else {
                texts.append((index, tokens))
                continue
            }
            let vector = model.embed(
                inputIds: MLXArray(tokens.map(Int32.init), [1, tokens.count]),
                attentionMask: nil, softTokens: softTokens)
            vectors[index] = try Self.unitRows(vector, dimension: dimension)[0]
        }
        let batches = EmbeddingGemma2Sequence.batches(
            lengths: texts.map(\.tokens.count), limit: Self.batchTokens)
        for batch in batches {
            try Swift.Task.checkCancellation()
            let rows = batch.map { texts[$0].tokens }
            let width = rows.map(\.count).max() ?? 0
            // Padding is masked out of attention and pooling, so its token does not matter.
            let ids = rows.flatMap { $0 + repeatElement(0, count: width - $0.count) }
            let mask = rows.flatMap {
                [Int32](repeating: 1, count: $0.count) + repeatElement(0, count: width - $0.count)
            }
            let output = model.embed(
                inputIds: MLXArray(ids.map(Int32.init), [rows.count, width]),
                attentionMask: rows.allSatisfy { $0.count == width }
                    ? nil : MLXArray(mask, [rows.count, width]))
            for (row, vector) in zip(batch, try Self.unitRows(output, dimension: dimension)) {
                vectors[texts[row].index] = vector
            }
        }
        try Swift.Task.checkCancellation()
        return vectors
    }

    /// Embeds one item; returns a unit-length float32 vector.
    public func embed(
        _ input: Input, task: Task, dimensions: Int? = nil
    ) async throws -> [Float] {
        try await embed([input], task: task, dimensions: dimensions)[0]
    }

    /// Embeds independent items at the checkpoint's full output size.
    public func embed(_ inputs: sending [Input], task: Task) async throws -> [[Float]] {
        try await embed(inputs, task: task, dimensions: nil)
    }

    private func outputDimension(_ requested: Int?) throws -> Int {
        let dimension = requested ?? model.config.embeddingDim
        guard
            dimension == model.config.embeddingDim
                || ([128, 256, 512].contains(dimension) && dimension <= model.config.embeddingDim)
        else { throw Error.invalidDimension }
        return dimension
    }

    /// The token sequence of one input, and the soft tokens that fill its media; `nil`
    /// when it has none.
    private func sequence(_ input: Input, task: Task) async throws -> (
        tokens: [Int], softTokens: MLXArray?
    ) {
        for part in input.parts {
            switch part {
            case .image where !supportedModalities.contains(.image),
                .video where !supportedModalities.contains(.video),
                .audio where !supportedModalities.contains(.audio):
                throw Error.unsupportedMedia
            default: break
            }
        }
        let hasText = input.parts.contains { !($0.text ?? "").allSatisfy(\.isWhitespace) }
        guard hasText || input.parts.contains(where: { $0.text == nil }) else {
            throw Error.emptyInput
        }
        guard let begin = tokenizer.bosToken.flatMap({ tokenizer.convertTokenToId($0) }),
            let end = tokenizer.eosTokenId ?? tokenizer.convertTokenToId("<eos>")
        else { throw Error.invalidEmbedding }

        var segments: [EmbeddingGemma2Sequence.Segment] = []
        var softTokens: [MLXArray] = []
        // Adjacent text joins before tokenization, as the chat template renders it.
        var text = ""
        for part in hasText ? Self.prompted(input, task: task) : input.parts {
            if let value = part.text {
                text += value
                continue
            }
            if !text.isEmpty {
                segments.append(.text(tokenizer.encode(text: text, addSpecialTokens: false)))
                text = ""
            }
            let (segment, features) = try await encode(part)
            if let features {
                try MLX.checkedEval(features)
                softTokens.append(features.reshaped(-1, features.dim(-1)))
            }
            segments.append(segment)
            try Swift.Task.checkCancellation()
        }
        if !text.isEmpty {
            segments.append(.text(tokenizer.encode(text: text, addSpecialTokens: false)))
        }

        let tokens = try EmbeddingGemma2Sequence.tokens(
            segments, configuration: model.config, beginOfSequence: begin, endOfSequence: end,
            limit: EmbeddingGemma2Configuration.contextLength)
        return (tokens, softTokens.isEmpty ? nil : concatenated(softTokens, axis: 0))
    }

    /// The rows of `[B, embeddingDim]` vectors, truncated to `dimension` at unit length.
    private static func unitRows(_ vectors: MLXArray, dimension: Int) throws -> [[Float]] {
        try MLX.checkedEval(vectors)
        let width = vectors.dim(1)
        let values = vectors.asArray(Float.self)
        return try stride(from: 0, to: values.count, by: width).map { start in
            let row = values[start ..< start + width]
            guard row.allSatisfy(\.isFinite), row.contains(where: { $0 != 0 }) else {
                throw Error.invalidEmbedding
            }
            let truncated = row.prefix(dimension)
            let norm = truncated.reduce(0.0) { $0 + Double($1) * Double($1) }.squareRoot()
            guard norm.isFinite, norm > 0 else { throw Error.invalidEmbedding }
            return truncated.map { Float(Double($0) / norm) }
        }
    }

    /// The segment of one media part and the soft tokens that fill it, `nil` when empty.
    private func encode(_ part: Input.Part) async throws -> (
        EmbeddingGemma2Sequence.Segment, MLXArray?
    ) {
        switch part {
        case .text:
            preconditionFailure("Text parts have no soft tokens.")
        case .image(let image):
            let processor =
                try imageProcessor ?? EmbeddingGemma2ImageProcessor(directory: modelDirectory)
            imageProcessor = processor
            let pixels = try processor.pixels(for: image).pixels
            try MLX.checkedEval(pixels)
            try Swift.Task.checkCancellation()
            try await model.loadMediaComponents([.vision])
            try Swift.Task.checkCancellation()
            guard let features = try model.imageFeatures(pixels) else {
                throw Error.unsupportedMedia
            }
            return (.image(tokens: features.dim(1)), features)
        case .video(let video):
            let processor =
                try videoProcessor ?? EmbeddingGemma2VideoProcessor(directory: modelDirectory)
            videoProcessor = processor
            let pixels = try await processor.frames(for: video)
            try Swift.Task.checkCancellation()
            try await model.loadMediaComponents([.vision])
            try Swift.Task.checkCancellation()
            guard let features = try model.imageFeatures(pixels) else {
                throw Error.unsupportedMedia
            }
            return (.video(frames: features.dim(0), tokensPerFrame: features.dim(1)), features)
        case .audio(let audio):
            let processor =
                try audioProcessor ?? EmbeddingGemma2AudioProcessor(directory: modelDirectory)
            audioProcessor = processor
            let frames = try await processor.features(for: audio)
            // Audio shorter than one frame keeps its markers and no soft tokens.
            guard frames.dim(0) > 0 else { return (.audio(tokens: 0), nil) }
            try Swift.Task.checkCancellation()
            try await model.loadMediaComponents([.audio])
            try Swift.Task.checkCancellation()
            guard let features = try model.audioFeatures(frames) else {
                throw Error.unsupportedMedia
            }
            return (.audio(tokens: features.dim(1)), features)
        }
    }

    /// The parts with the task's instruction prefix joined to the leading text.
    private static func prompted(_ input: Input, task: Task) -> [Input.Part] {
        if let first = input.parts.first?.text {
            return [.text(prompt(text: first, title: input.title, task: task))]
                + input.parts.dropFirst()
        }
        return [.text(prompt(text: "", title: input.title, task: task))] + input.parts
    }

    /// The task instruction prefix from the model card. Inputs that already carry one
    /// pass through unchanged.
    public static func prompt(text: String, title: String?, task: Task) -> String {
        if text.hasPrefix("task:") || text.hasPrefix("title:") { return text }
        switch task {
        case .none:
            return text
        case .questionAnswering:
            return "task: question answering | query: \(text)"
        case .factChecking:
            return "task: fact checking | query: \(text)"
        case .codeRetrieval:
            return "task: code retrieval | query: \(text)"
        case .sentenceSimilarity:
            return "task: sentence similarity | query: \(text)"
        case .searchQuery:
            return "task: search result | query: \(text)"
        case .clustering:
            return "task: clustering | query: \(text)"
        case .classification:
            return "task: classification | query: \(text)"
        case .document:
            let line = title?
                .split(whereSeparator: \.isWhitespace).joined(separator: " ")
            if let line, !line.isEmpty {
                return "title: \(line) | text: \(text)"
            }
            return "title: none | text: \(text)"
        }
    }
}

extension EmbeddingGemma2Embedding.Input.Part {
    fileprivate var text: String? {
        if case .text(let text) = self { text } else { nil }
    }
}
