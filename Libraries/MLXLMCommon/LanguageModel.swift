// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXNN

/// Abstract form of a model that processes language.
public protocol BaseLanguageModel: Module {
    /// Optionally preprocess the weights and modify / remove values as needed.
    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray]

    /// Optionally preprocess the weights with access to safetensor metadata.
    ///
    /// The default implementation forwards to ``sanitize(weights:)``.
    /// Models can override this to inspect metadata (e.g. check `metadata["format"] == "mlx"`)
    /// and skip or customize sanitization accordingly.
    func sanitize(weights: [String: MLXArray], metadata: [String: String]) -> [String: MLXArray]

    /// Translate a runtime module path to the corresponding path in a checkpoint's
    /// per-layer quantization configuration.
    ///
    /// Composite models that namespace sanitized weights can override this hook so
    /// mixed-precision overrides continue to address the original checkpoint paths.
    func quantizationConfigurationPath(for modulePath: String) -> String
}

/// Optional metadata a model wants written into converted safetensors.
///
/// Model-specific metadata lets future loaders distinguish transformed MLX-native
/// checkpoints from original upstream checkpoints without relying only on tensor shapes.
public protocol ModelConversionMetadataProvider {
    var modelConversionMetadata: [String: String] { get }
}

extension BaseLanguageModel {
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        weights
    }

    public func sanitize(weights: [String: MLXArray], metadata: [String: String]) -> [String:
        MLXArray]
    {
        sanitize(weights: weights)
    }

    public func quantizationConfigurationPath(for modulePath: String) -> String {
        modulePath
    }
}

/// Time/Height/Width struct to represent information about input images.
public struct THW: Sendable {

    public let t: Int
    public let h: Int
    public let w: Int

    public init(_ t: Int, _ h: Int, _ w: Int) {
        self.t = t
        self.h = h
        self.w = w
    }

    public var values: (Int, Int, Int) {
        (t, h, w)
    }

    public var product: Int { t * h * w }
}

/// Representation of ``LanguageModel`` input.
///
/// This can contain text (tokens), prepared images (`MLXArray`), or other media as
/// needed. ``LMInput`` is produced by ``UserInputProcessor`` in response
/// to ``UserInput``.
///
/// The ``ModelContext`` holds the ``UserInputProcessor`` associated with a
/// ``LanguageModel``.
public struct LMInput {
    public let text: Text
    public let image: ProcessedImage?
    public let video: ProcessedVideo?
    public let audio: ProcessedAudio?
    public let multimodalTokenTypes: MLXArray?

    /// Representation of tokenized input text.
    public struct Text {

        /// input token array
        public let tokens: MLXArray

        /// optional mask array
        public let mask: MLXArray?

        public init(tokens: MLXArray, mask: MLXArray? = nil) {
            self.tokens = tokens
            self.mask = mask
        }

        public subscript(
            indices: MLXArrayIndex..., stream stream: StreamOrDevice = .default
        ) -> Text {
            Text(tokens: tokens[indices, stream: stream], mask: mask?[indices, stream: stream])
        }

        public subscript(
            text indices: MLXArrayIndex..., stream stream: StreamOrDevice = .default
        ) -> Text {
            Text(tokens: tokens[indices, stream: stream], mask: mask)
        }

        /// Per-batch sequence lengths derived from the optional attention mask.
        public var sequenceLengths: [Int]? {
            if let mask {
                return mask.asType(.int32).sum(axis: -1).asArray(Int.self)
            }
            guard tokens.ndim == 2 else { return nil }
            return Array(repeating: tokens.dim(1), count: tokens.dim(0))
        }

        /// Number of logical sequence positions consumed by one model call.
        /// Batch dimensions do not duplicate the shared cache timeline.
        @inline(__always)
        package var cacheSequenceLength: Int {
            tokens.ndim == 0 ? 0 : tokens.dim(-1)
        }
    }

    /// Representation of prepared input image(s).
    public struct ProcessedImage {

        /// Concatenated pixels from one or more images
        public let pixels: MLXArray
        /// Optional per-patch position ids for encoder-free vision embedders.
        public let positionIds: MLXArray?
        /// Time, height, and width of the images
        public let frames: [THW]?

        public init(
            pixels: MLXArray, positionIds: MLXArray? = nil, frames: [THW]? = nil
        ) {
            self.pixels = pixels
            self.positionIds = positionIds
            self.frames = frames
        }
    }

    /// Representation of prepared input video(s).
    /// For now, this is virtually identical to ProcessedImage.
    public struct ProcessedVideo {

        public let pixels: MLXArray
        public let positionIds: MLXArray?
        public let frames: [THW]?

        public init(
            pixels: MLXArray, positionIds: MLXArray? = nil, frames: [THW]? = nil
        ) {
            self.pixels = pixels
            self.positionIds = positionIds
            self.frames = frames
        }
    }

    /// Representation of prepared audio features.
    public struct ProcessedAudio {
        public let features: MLXArray
        public let mask: MLXArray?

        public init(features: MLXArray, mask: MLXArray? = nil) {
            self.features = features
            self.mask = mask
        }

        public init(samples: MLXArray) {
            self.init(features: samples)
        }
    }

    public init(tokens: MLXArray, mask: MLXArray? = nil) {
        self.init(text: .init(tokens: tokens, mask: mask))
    }

    public init(
        text: LMInput.Text,
        image: LMInput.ProcessedImage? = nil,
        video: LMInput.ProcessedVideo? = nil,
        audio: LMInput.ProcessedAudio? = nil
    ) {
        self.init(
            text: text,
            image: image,
            video: video,
            audio: audio,
            multimodalTokenTypes: nil)
    }

    public init(
        text: LMInput.Text,
        image: LMInput.ProcessedImage? = nil,
        video: LMInput.ProcessedVideo? = nil,
        audio: LMInput.ProcessedAudio? = nil,
        multimodalTokenTypes: MLXArray?
    ) {
        self.text = text
        self.image = image
        self.video = video
        self.audio = audio
        self.multimodalTokenTypes = multimodalTokenTypes
    }
}

/// Validate the shape shared by block-diffusion streaming and return the logical
/// prompt positions selected by an optional attention mask. A `nil` result means
/// every prompt position is valid and no gather is needed.
package func blockDiffusionPromptIndices(
    mask: MLXArray?, sequenceLength: Int, batchSize: Int, modelName: String
) throws -> MLXArray? {
    guard batchSize == 1 else {
        throw GenerateError.unsupportedBatchSize(modelName: modelName, batchSize: batchSize)
    }
    guard let mask else { return nil }

    let values = mask.asType(.bool).flattened().asArray(Bool.self)
    guard values.count == sequenceLength else {
        throw GenerateError.invalidAttentionMask(
            "expected \(sequenceLength) values for a batch-one prompt, got \(values.count).")
    }
    guard values.contains(true) else {
        throw GenerateError.invalidAttentionMask("the prompt cannot be entirely masked.")
    }
    guard values.contains(false) else { return nil }

    return MLXArray(
        values.enumerated().compactMap { index, isValid in
            isValid ? Int32(index) : nil
        })
}

/// ``LanguageModel`` step output. This is consumed internally
/// by the ``TokenIterator``.
public struct LMOutput {

    /// logits (one hot vector of probabilities for tokens)
    public let logits: MLXArray

    /// optional ``State`` to carry forward into the next step
    public let state: State?

    /// typed key for use in ``State``
    public struct Key<T>: Identifiable, Sendable {
        public let id: String

        public init(_ id: String) {
            self.id = id
        }
    }

    /// Dictionary of typed ``Key`` to carry state between steps.
    public struct State {
        private var contents: [String: Any]

        public init() {
            self.contents = [:]
        }

        public subscript<T>(_ key: Key<T>) -> T? {
            get {
                contents[key.id] as? T
            }
            set {
                contents[key.id] = newValue
            }
        }
    }

    public init(logits: MLXArray, state: LMOutput.State? = nil) {
        self.logits = logits
        self.state = state
    }
}

/// The result of the call to ``LanguageModel/prepare(_:cache:state:prefill:)``
public enum PrepareResult {
    /// tokens to process by the ``TokenIterator``
    case tokens(LMInput.Text)

    /// logits representing the next token
    case logits(LMOutput)
}

/// Feature flags that describe generation behavior supported by a language model.
public struct LanguageModelCapabilities: OptionSet, Sendable {
    public let rawValue: Int

    public init(rawValue: Int) {
        self.rawValue = rawValue
    }

    public static let blockDiffusion = Self(rawValue: 1 << 0)
}

/// Interface for all Language Models (e.g. LLM, VLM).
///
/// The language model is typically called by the ``TokenIterator`` and it:
///
/// - consumes the ``LMInput``
/// - calls ``prepare(_:cache:state:prefill:)`` to initialize the KVCache and consume the prompt
/// - calls ``callAsFunction(_:cache:state:)-9kuvf`` for each token, producing an ``LMOutput``
/// - the ``TokenIterator`` accumulates this information into a ``GenerateResult``
public protocol LanguageModel: BaseLanguageModel, ChatConventionsProviding {

    /// Feature flags that describe generation behavior supported by the model.
    var capabilities: LanguageModelCapabilities { get }

    /// Prepare the cache state and consume the ``LMInput``.
    ///
    /// `state` is the ``LMOutput/state`` a caller carried over from earlier
    /// evaluation against the same `cache` — present when `cache` is already
    /// warm (a multi-turn chat, a tool-call restart, a restored prompt
    /// cache). Models that keep per-call positional state (e.g. the M-RoPE
    /// `ropeDeltas` of the Qwen VLM families) use it to anchor the new
    /// tokens' positions at the cache offset; models without such state can
    /// ignore it. In the typical cold call it is `nil`.
    ///
    /// This can return:
    /// - ``PrepareResult/tokens(_:)`` if the caller should evaluate the (remaining) tokens normally
    /// - ``PrepareResult/logits(_:)`` to produce the next token from the prompt
    ///
    /// Implementations that chunk the prompt should drive the loop with
    /// ``PrefillParameters/forEachChunk(total:reserving:defaultStepSize:maximumStepSize:_:)``,
    /// which owns cancellation, pooling, and per-chunk progress. An
    /// implementation returning `.logits` owns its whole
    /// ``PrefillParameters/progress`` sequence, including the terminal
    /// `(total, total)`; one returning `.tokens` reports only its own chunks —
    /// the iterator that evaluates the remainder completes the sequence.
    func prepare(
        _ input: LMInput, cache: [KVCache], state: LMOutput.State?, prefill: PrefillParameters
    )
        throws -> PrepareResult

    /// Primary entry point to produce a step (single token) from the model
    func callAsFunction(_ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?)
        -> LMOutput

    /// Models may implement this simplified interface if they do not produce any ``LMOutput/State``
    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray

    /// create a new array of ``KVCache``: automatic implementation if self
    /// implements ``KVCacheDimensionProvider``
    func newCache(parameters: GenerateParameters?) -> [KVCache]
}

/// Optional interface for text diffusion models that generate a block of tokens
/// at a time instead of predicting one next-token logit.
public protocol BlockDiffusionLanguageModel: LanguageModel {
    var diffusionCanvasLength: Int { get }
    var diffusionMinimumCanvasLength: Int { get }
    var diffusionMaxDenoisingSteps: Int { get }
    var diffusionEntropyBound: Float { get }
    var diffusionTemperatureMin: Float { get }
    var diffusionTemperatureMax: Float { get }
    var diffusionStabilityThreshold: Int { get }
    var diffusionConfidenceThreshold: Float { get }
    var diffusionVocabularySize: Int { get }
    var diffusionDefaultMaxTokens: Int? { get }
    var diffusionPrefersLogitsSelfConditioning: Bool { get }

    func prepareDiffusion(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws
    func acceptDiffusionTokens(_ tokens: MLXArray, cache: [KVCache], windowSize: Int?)
    func diffusionLogits(
        canvasTokens: MLXArray,
        cache: [KVCache],
        selfConditioningLogits: MLXArray?
    ) -> MLXArray
    func diffusionLogits(
        canvasTokens: MLXArray,
        cache: [KVCache],
        selfConditioningEmbeddings: MLXArray?
    ) -> MLXArray
    func diffusionSelfConditioningWeight() -> MLXArray?
    func diffusionSelfConditioningEmbeddings(logits: MLXArray, weight: MLXArray?) -> MLXArray
}

extension BlockDiffusionLanguageModel {
    public var diffusionMinimumCanvasLength: Int { 64 }
    public var diffusionTemperatureMin: Float { 0.4 }
    public var diffusionTemperatureMax: Float { 0.8 }
    public var diffusionStabilityThreshold: Int { 1 }
    public var diffusionConfidenceThreshold: Float { 0.005 }
    public var diffusionDefaultMaxTokens: Int? { nil }
    public var diffusionPrefersLogitsSelfConditioning: Bool { false }

    public func diffusionLogits(
        canvasTokens: MLXArray,
        cache: [KVCache],
        selfConditioningEmbeddings: MLXArray?
    ) -> MLXArray {
        diffusionLogits(
            canvasTokens: canvasTokens,
            cache: cache,
            selfConditioningLogits: selfConditioningEmbeddings)
    }

    public func diffusionSelfConditioningWeight() -> MLXArray? {
        nil
    }

    public func diffusionSelfConditioningEmbeddings(logits: MLXArray, weight: MLXArray?) -> MLXArray
    {
        logits
    }
}

extension LanguageModel {
    public var capabilities: LanguageModelCapabilities {
        var capabilities: LanguageModelCapabilities = []
        if self is any BlockDiffusionLanguageModel {
            capabilities.insert(.blockDiffusion)
        }
        return capabilities
    }

    @available(
        *, deprecated, renamed: "prepare(_:cache:state:prefill:)",
        message:
            "prefill now defaults to balanced chunking; use prefill.chunking = .remainder for the legacy chunk boundaries"
    )
    public func prepare(
        _ input: LMInput, cache: [KVCache], state: LMOutput.State?, windowSize: Int?
    ) throws -> PrepareResult {
        try prepare(input, cache: cache, state: state, prefill: .init(stepSize: windowSize))
    }

    public func callAsFunction(_ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?)
        -> LMOutput
    {
        let logits = callAsFunction(input.tokens, cache: cache)
        return .init(logits: logits)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        fatalError("callAsFunction(inputs:cache:) not implemented for \(Self.self)")
    }
}

/// Optional protocol that can be implemented by ``LanguageModel`` and will
/// provide an automatic implementation of ``LanguageModel/newCache(parameters:)``
public protocol KVCacheDimensionProvider {
    var kvHeads: [Int] { get }
}

extension LanguageModel where Self: KVCacheDimensionProvider {
    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        // Create one cache per layer (kvHeads.count = number of layers)
        // The number of heads per layer (kvHeads[i]) is not used for cache creation
        let numLayers = kvHeads.count

        // Follow Python logic: use RotatingKVCache if a capacity is provided.
        if let capacity = parameters?.effectiveKVCacheCapacity {
            return (0 ..< numLayers).map { _ in
                capacity.makeRotatingCache()
            }
        } else {
            return (0 ..< numLayers).map { _ in KVCacheSimple() }
        }
    }
}
