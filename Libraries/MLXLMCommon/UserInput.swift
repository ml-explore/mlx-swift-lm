// Copyright © 2024 Apple Inc.

import Foundation
import MLX

#if canImport(AVFoundation)
@preconcurrency import AVFoundation
#endif
#if canImport(CoreImage)
import CoreImage
#endif

public typealias Message = [String: any Sendable]

/// Container for raw user input.
///
/// A ``UserInputProcessor`` can convert this to ``LMInput``.
/// See also ``ModelContext``.
public struct UserInput {

    /// Representation of a prompt or series of messages (conversation).
    ///
    /// This may be a single string with a user prompt or a series of back
    /// and forth responses representing a conversation.
    public enum Prompt: CustomStringConvertible {
        /// A single string
        case text(String)

        /// Model-specific array of dictionaries
        case messages([Message])

        /// Model-agnostic structured chat (series of messages)
        case chat([Chat.Message])

        public var description: String {
            switch self {
            case .text(let text):
                return text
            case .messages(let messages):
                return messages.map { $0.description }.joined(separator: "\n")
            case .chat(let messages):
                return messages.map(\.content).joined(separator: "\n")
            }
        }
    }

    public struct VideoFrame {
        public let image: Image
        public let timeStamp: CMTime

        public init(image: Image, timeStamp: CMTime) {
            self.image = image
            self.timeStamp = timeStamp
        }

        #if canImport(CoreImage)

        @available(
            *, deprecated,
            message: "Use init(image:, timeStamp:) instead"
        )
        public init(frame: CIImage, timeStamp: CMTime) {
            self.image = .ciImage(frame)
            self.timeStamp = timeStamp
        }

        @available(
            *, deprecated,
            message: "Use image.asCIImage()"
        )
        public var frame: CIImage {
            return try! image.asCIImage()
        }

        #endif
    }

    /// Representation of a video resource.
    public struct Video {

        public enum Source {
            #if canImport(AVFoundation)
            case avAsset(AVAsset)
            #endif
            case url(URL)
            /// Useful for decoded frames held in memory
            case frames([VideoFrame])
        }

        public var source: Source

        public init(source: Source) {
            self.source = source
        }

        #if canImport(AVFoundation)
        public static func avAsset(_ asset: AVAsset) -> Self {
            Self(source: .avAsset(asset))
        }
        #endif

        public static func url(_ url: URL) -> Self {
            Self(source: .url(url))
        }

        public static func frames(_ frames: [VideoFrame]) -> Self {
            Self(source: .frames(frames))
        }

        #if canImport(AVFoundation)
        @available(
            *, deprecated,
            message: "Use MediaProcessing.asProcessedSequence() with the Video directly"
        )
        public func asAVAsset() -> AVAsset {
            switch source {
            case .avAsset(let asset):
                return asset
            case .url(let url):
                return AVAsset(url: url)
            case .frames:
                fatalError(
                    "calling asAVAsset() on Video Input with VideoFames provided is unsupported and deprecated - please use MediaProcessing.asProcessedSequence() instead"
                )
            }
        }
        #endif
    }

    /// Representation of an image resource.
    public struct Image {

        public enum Source {
            #if canImport(CoreImage)
            case ciImage(CIImage)
            #endif
            case url(URL)
            case array(MLXArray)
        }

        public var source: Source

        /// Text that a vision message generator writes into the prompt as `[label]`,
        /// immediately before this image. The generator leaves out and logs a label that
        /// contains `<`, `>`, `|`, `[` or `]`. A vision processor also leaves out a label
        /// whose `[label]` form contains a special token, such as `IMG` on Mistral3.
        public var label: String?

        /// The characters that delimit image placeholders, such as `<|image_pad|>` and `[IMG]`.
        package static let markerCharacters: Set<Character> = ["<", ">", "|", "[", "]"]

        package var labelMarkerCharacter: Character? {
            label?.first(where: Self.markerCharacters.contains)
        }

        package static func promptText(forLabel label: String) -> String {
            "[\(label)]"
        }

        public init(source: Source, label: String? = nil) {
            self.source = source
            self.label = label
        }

        #if canImport(CoreImage)
        public static func ciImage(_ image: CIImage, label: String? = nil) -> Self {
            Self(source: .ciImage(image), label: label)
        }

        /// Makes `images.map(UserInput.Image.ciImage)` compile, because a function value
        /// cannot use the default `label`.
        public static func ciImage(_ image: CIImage) -> Self {
            Self(source: .ciImage(image))
        }
        #endif

        public static func url(_ url: URL, label: String? = nil) -> Self {
            Self(source: .url(url), label: label)
        }

        /// Makes `urls.map(UserInput.Image.url)` compile, because a function value cannot
        /// use the default `label`.
        public static func url(_ url: URL) -> Self {
            Self(source: .url(url))
        }

        public static func array(_ array: MLXArray, label: String? = nil) -> Self {
            Self(source: .array(array), label: label)
        }

        /// Makes `arrays.map(UserInput.Image.array)` compile, because a function value
        /// cannot use the default `label`.
        public static func array(_ array: MLXArray) -> Self {
            Self(source: .array(array))
        }

        #if canImport(CoreImage)
        public func asCIImage() throws -> CIImage {
            switch source {
            case .ciImage(let image):
                return image

            case .url(let url):
                if let image = CIImage(contentsOf: url) {
                    return image
                }
                throw UserInputError.unableToLoad(url)

            case .array(let array):
                guard array.ndim == 3 else {
                    throw UserInputError.arrayError(
                        "array must have 3 dimensions: \(array.ndim)")
                }

                var array = array

                // convert to 0 .. 255
                if array.max().item(Float.self) <= 1.0 {
                    array = array * 255
                }

                // planar -> pixels
                switch array.dim(0) {
                case 3, 4:
                    // channels first (planar)
                    array = array.transposed(1, 2, 0)
                default:
                    break
                }

                // 4 components per pixel
                switch array.dim(-1) {
                case 3:
                    // pad to 4 bytes per pixel
                    array = padded(array, widths: [0, 0, [0, 1]], value: MLXArray(255))
                case 4:
                    // good
                    break
                default:
                    throw UserInputError.arrayError(
                        "channel dimension must be last and 3/4: \(array.shape)")
                }

                let arrayData = array.asData()
                let (H, W, _) = array.shape3
                let cs = CGColorSpace(name: CGColorSpace.sRGB)!

                return CIImage(
                    bitmapData: arrayData.data, bytesPerRow: W * 4,
                    size: .init(width: W, height: H),
                    format: .RGBA8, colorSpace: cs)
            }
        }
        #endif
    }

    /// Representation of an audio resource.
    public struct Audio {

        public enum Source {
            case url(URL)
            case array(MLXArray)
        }

        public var source: Source

        public init(source: Source) {
            self.source = source
        }

        public static func url(_ url: URL) -> Self {
            Self(source: .url(url))
        }

        public static func array(_ array: MLXArray) -> Self {
            Self(source: .array(array))
        }

        // See also UserInput+Audio
    }

    /// Representation of the audio format.
    public enum AudioFormat: Sendable {
        case linearPCM
    }

    /// Representation of processing to apply to media.
    public struct Processing: Sendable {
        public var resize: CGSize?

        public var video = VideoProcessing()
        public var audio = AudioProcessing()

        /// Optional per-call overrides for the image resize budget. When set,
        /// they replace the model's configured `min_pixels` / `max_pixels` for
        /// this request; when `nil` the model configuration is used. This lets
        /// a caller request the resolution a model was tuned for without
        /// hard-coding pixel counts in the processor.
        public var minPixels: Int?
        public var maxPixels: Int?

        public init(
            resize: CGSize? = nil,
            video: VideoProcessing = VideoProcessing(),
            audio: AudioProcessing = AudioProcessing(),
            minPixels: Int? = nil,
            maxPixels: Int? = nil
        ) {
            self.resize = resize
            self.video = video
            self.audio = audio
            self.minPixels = minPixels
            self.maxPixels = maxPixels
        }
    }

    /// Representation of video processing options.
    public struct VideoProcessing: Sendable, Equatable {
        /// Strategy for temporal frame sampling.
        public enum SamplingMethod: Sendable, Equatable {
            /// Sample a fixed total count of frames distributed evenly across the video duration.
            case targetFrames(Int)

            /// Derive a target frame count from frames per second, then distribute those frames
            /// evenly across the video duration.
            case framesPerSecond(Double)
        }

        /// The sampling strategy to use, or `nil` to use model default behavior.
        public var sampling: SamplingMethod?

        public init(sampling: SamplingMethod? = nil) {
            self.sampling = sampling
        }

        /// Convenience initializer to sample a fixed number of frames across the video.
        public init(targetFrames: Int) {
            self.sampling = .targetFrames(targetFrames)
        }

        /// Convenience initializer to target a sampling density in frames per second.
        public init(targetFramesPerSecond: Double) {
            self.sampling = .framesPerSecond(targetFramesPerSecond)
        }

        /// Target number of frames to sample from the video, regardless of duration.
        public var targetFrames: Int? {
            get {
                if case .targetFrames(let count) = sampling { return count }
                return nil
            }
            set {
                if let newValue {
                    sampling = .targetFrames(newValue)
                } else if case .targetFrames = sampling {
                    sampling = nil
                }
            }
        }

        /// Target frame rate (frames per second) to sample from the video.
        public var targetFramesPerSecond: Double? {
            get {
                if case .framesPerSecond(let fps) = sampling { return fps }
                return nil
            }
            set {
                if let newValue {
                    sampling = .framesPerSecond(newValue)
                } else if case .framesPerSecond = sampling {
                    sampling = nil
                }
            }
        }
    }

    /// Representation of audio processing
    public struct AudioProcessing: Sendable {
        /// Sample rate
        public var sampleRate = 48_000.0

        /// Number of channels of audio.  If 1, convert to mono
        public var channels = 1

        /// Audio format
        public var audioFormat: AudioFormat = .linearPCM

        public init() {
        }
    }

    /// The prompt to evaluate.
    public var prompt: Prompt {
        didSet {
            switch prompt {
            case .text, .messages:
                // no action
                break
            case .chat(let messages):
                // rebuild images & videos
                self.images = messages.reduce(into: []) { result, message in
                    result.append(contentsOf: message.images)
                }
                self.videos = messages.reduce(into: []) { result, message in
                    result.append(contentsOf: message.videos)
                }
                self.audios = messages.reduce(into: []) { result, message in
                    result.append(contentsOf: message.audios)
                }
            }
        }
    }

    /// The images associated with the `UserInput`.
    ///
    /// If the ``prompt-swift.property`` is a ``Prompt-swift.enum/chat(_:)`` this will
    /// collect the images from the chat messages, otherwise these are the stored images with the ``UserInput``.
    public var images = [Image]()

    /// The images associated with the `UserInput`.
    ///
    /// If the ``prompt-swift.property`` is a ``Prompt-swift.enum/chat(_:)`` this will
    /// collect the videos from the chat messages, otherwise these are the stored videos with the ``UserInput``.
    public var videos = [Video]()

    /// The audios associated with the `UserInput`.
    ///
    /// If the ``prompt-swift.property`` is a ``Prompt-swift.enum/chat(_:)`` this will
    /// collect the audios from the chat messages, otherwise these are the stored audios with the ``UserInput``.
    public var audios = [Audio]()

    public var tools: [ToolSpec]?

    /// Additional values provided for the chat template rendering context
    public var additionalContext: [String: any Sendable]?
    public var processing: Processing = .init()

    /// Initialize the `UserInput` with a single text prompt.
    ///
    /// - Parameters:
    ///   - prompt: text prompt
    ///   - images: optional images
    ///   - videos: optional videos
    ///   - audios: optional audios
    ///   - tools: optional tool specifications
    ///   - additionalContext: optional context (model specific)
    /// ### See Also
    /// - ``Prompt-swift.enum/text(_:)``
    /// - ``init(chat:processing:tools:additionalContext:)``
    public init(
        prompt: String,
        images: [Image] = [Image](),
        videos: [Video] = [Video](),
        audios: [Audio] = [Audio](),
        tools: [ToolSpec]? = nil,
        additionalContext: [String: any Sendable]? = nil
    ) {
        self.prompt = .chat([
            .user(prompt, images: images, videos: videos, audios: audios)
        ])
        // note: prompt.didSet is not triggered in init
        self.images = images
        self.videos = videos
        self.audios = audios
        self.tools = tools
        self.additionalContext = additionalContext
    }

    /// Initialize the `UserInput` with model specific mesage structures.
    ///
    /// For example, the Qwen2VL model wants input in this format:
    ///
    /// ```
    /// [
    ///     [
    ///         "role": "user",
    ///         "content": [
    ///             [
    ///                 "type": "text",
    ///                 "text": "What is this?"
    ///             ],
    ///             [
    ///                 "type": "image",
    ///             ],
    ///         ]
    ///     ]
    /// ]
    /// ```
    ///
    /// Typically the ``init(chat:processing:tools:additionalContext:)``
    /// should be used instead along with a model specific
    /// ``MessageGenerator`` (supplied by the ``UserInputProcessor``).
    ///
    /// - Parameters:
    ///   - messages: array of dictionaries representing the prompt in a model specific format
    ///   - images: optional images
    ///   - videos: optional videos
    ///   - audios: optional audios
    ///   - tools: optional tool specifications
    ///   - additionalContext: optional context (model specific)
    /// ### See Also
    /// - ``Prompt-swift.enum/text(_:)``
    /// - ``init(chat:processing:tools:additionalContext:)``
    public init(
        messages: [Message],
        images: [Image] = [Image](),
        videos: [Video] = [Video](),
        audios: [Audio] = [Audio](),
        tools: [ToolSpec]? = nil,
        additionalContext: [String: any Sendable]? = nil
    ) {
        self.prompt = .messages(messages)
        self.images = images
        self.videos = videos
        self.audios = audios
        self.tools = tools
        self.additionalContext = additionalContext
    }

    /// Initialize the `UserInput` with a model agnostic structured context.
    ///
    /// For example:
    ///
    /// ```
    /// let chat: [Chat.Message] = [
    ///     .system("You are a helpful photographic assistant."),
    ///     .user("Please describe the photo.", images: [image1]),
    /// ]
    /// let userInput = UserInput(chat: chat)
    /// ```
    ///
    /// A model specific ``MessageGenerator`` (supplied by the ``UserInputProcessor``)
    /// is used to convert this into a model specific format.
    ///
    /// - Parameters:
    ///   - chat: structured content
    ///   - tools: optional tool specifications
    ///   - processing: optional processing to be applied to media
    ///   - additionalContext: optional context (model specific)
    /// ### See Also
    /// - ``Prompt-swift.enum/text(_:)``
    /// - ``init(chat:processing:tools:additionalContext:)``
    public init(
        chat: [Chat.Message],
        processing: Processing = .init(),
        tools: [ToolSpec]? = nil,
        additionalContext: [String: any Sendable]? = nil
    ) {
        self.prompt = .chat(chat)

        // note: prompt.didSet is not triggered in init
        self.images = chat.reduce(into: []) { result, message in
            result.append(contentsOf: message.images)
        }
        self.videos = chat.reduce(into: []) { result, message in
            result.append(contentsOf: message.videos)
        }
        self.audios = chat.reduce(into: []) { result, message in
            result.append(contentsOf: message.audios)
        }

        self.processing = processing
        self.tools = tools
        self.additionalContext = additionalContext
    }

    /// Initialize the `UserInput` with a preconfigured ``Prompt-swift.enum``.
    ///
    /// ``init(chat:processing:tools:additionalContext:)`` is
    /// the preferred mechanism.
    ///
    /// - Parameters:
    ///   - prompt: the prompt
    ///   - images: optional images
    ///   - videos: optional videos
    ///   - audios: optional audios
    ///   - tools: optional tool specifications
    ///   - processing: optional processing to be applied to media
    ///   - additionalContext: optional context (model specific)
    /// ### See Also
    /// - ``Prompt-swift.enum/text(_:)``
    /// - ``init(chat:processing:tools:additionalContext:)``
    public init(
        prompt: Prompt,
        images: [Image] = [Image](),
        videos: [Video] = [Video](),
        audios: [Audio] = [Audio](),
        processing: Processing = .init(),
        tools: [ToolSpec]? = nil, additionalContext: [String: any Sendable]? = nil
    ) {
        self.prompt = prompt
        // note: prompt.didSet is not triggered in init
        switch prompt {
        case .text, .messages:
            self.images = images
            self.videos = videos
            self.audios = audios
        case .chat(let messages):
            self.images = messages.reduce(into: []) { result, message in
                result.append(contentsOf: message.images)
            }
            self.videos = messages.reduce(into: []) { result, message in
                result.append(contentsOf: message.videos)
            }
            self.audios = messages.reduce(into: []) { result, message in
                result.append(contentsOf: message.audios)
            }
        }
        self.processing = processing
        self.tools = tools
        self.additionalContext = additionalContext
    }
}

/// Protocol for a type that can convert ``UserInput`` to ``LMInput``.
///
/// See also ``ModelContext``.
public protocol UserInputProcessor: Sendable {
    func prepare(input: UserInput) async throws -> LMInput
}

/// Applies a configured message generator before delegating input processing.
///
/// This lets a model configuration override a VLM processor's built-in generator without
/// changing the processor implementation.
///
/// - Important: the override fully replaces the model's own generator, including any
///   image/video placeholder content that generator would emit. A text-only generator on a
///   VLM configuration therefore loses media conditioning: Qwen-family processors throw on
///   the placeholder-count mismatch, others silently ignore the attached media.
public struct MessageGeneratorUserInputProcessor: UserInputProcessor {
    private let processor: any UserInputProcessor
    private let messageGenerator: any MessageGenerator
    private let tokenizer: (any Tokenizer)?

    public init(
        processor: any UserInputProcessor,
        messageGenerator: any MessageGenerator
    ) {
        self.processor = processor
        self.messageGenerator = messageGenerator
        self.tokenizer = nil
    }

    package init(
        processor: any UserInputProcessor,
        messageGenerator: any MessageGenerator,
        tokenizer: any Tokenizer
    ) {
        self.processor = processor
        self.messageGenerator = messageGenerator
        self.tokenizer = tokenizer
    }

    public func prepare(input: UserInput) async throws -> LMInput {
        var input = tokenizer.map { input.removingSpecialTokenLabels(using: $0) } ?? input
        input.prompt = .messages(messageGenerator.generate(from: input))
        return try await processor.prepare(input: input)
    }
}

internal enum UserInputError: LocalizedError {
    case notImplemented
    case unableToLoad(URL)
    case arrayError(String)
    case noAudioData(URL)

    var errorDescription: String? {
        switch self {
        case .notImplemented:
            return String(localized: "This functionality is not implemented.")
        case .unableToLoad(let url):
            return String(localized: "Unable to load image from URL: \(url.path).")
        case .arrayError(let message):
            return String(localized: "Error processing image array: \(message).")
        case .noAudioData(let url):
            return String(localized: "No audio data in file: \(url.path)")
        }
    }
}

/// A do-nothing ``UserInputProcessor``.
public struct StandInUserInputProcessor: UserInputProcessor {
    public init() {}

    public func prepare(input: UserInput) throws -> LMInput {
        throw UserInputError.notImplemented
    }
}

extension Tokenizer {

    /// The special tokens that `[label]` encodes to, or `nil` if it encodes to none.
    /// An empty array still means that `[label]` contains a special token.
    package func specialTokenNames(inImageLabel label: String) -> [String]? {
        // Special tokens that `encode` adds, such as BOS, would otherwise flag every label.
        let ids = encode(
            text: UserInput.Image.promptText(forLabel: label), addSpecialTokens: false)
        guard containsSpecialToken(ids) else { return nil }
        return specialTokenNames(inIDs: ids)
    }

    /// The `Tokenizer` protocol cannot list special tokens, so this function compares a
    /// decode with and without `skipSpecialTokens`. An added token without the special
    /// flag passes, so callers must refuse the marker characters too.
    private func containsSpecialToken(_ ids: [Int]) -> Bool {
        decode(tokenIds: ids, skipSpecialTokens: false)
            != decode(tokenIds: ids, skipSpecialTokens: true)
    }

    /// Each name comes from `convertIdToToken`, because decoding one id can change the
    /// token's text.
    private func specialTokenNames(inIDs ids: [Int]) -> [String] {
        var names: [String] = []
        var seen = Set<Int>()
        for id in ids where containsSpecialToken([id]) {
            guard seen.insert(id).inserted else { continue }
            names.append(convertIdToToken(id) ?? "token \(id)")
        }
        return names
    }
}
