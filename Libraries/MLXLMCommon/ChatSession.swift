// Copyright © 2025 Apple Inc.

import CoreGraphics
import Foundation
import MLX

/// Configuration for speculative decoding in a `ChatSession`.
///
/// Speculative decoding uses a small draft model to propose candidate tokens
/// that the main model then verifies in a single forward pass, providing a
/// speedup with no quality degradation when both models fit comfortably in memory.
///
/// Both models must share the same tokenizer vocabulary.
///
/// Example usage:
/// ```swift
/// let main  = try await LLMModelFactory.shared.loadContainer(configuration: mainConfig)
/// let draft = try await LLMModelFactory.shared.loadContainer(configuration: draftConfig)
///
/// let session = ChatSession(
///     main,
///     speculativeDecoding: SpeculativeDecodingConfig(draftModel: draft, numDraftTokens: 5)
/// )
/// ```
///
/// To avoid loading a draft model that would exceed a memory policy, pass a
/// byte estimate and a loader closure:
///
/// ```swift
/// let session = ChatSession(
///     main,
///     speculativeDecoding: SpeculativeDecodingConfig(
///         draftModelBytes: estimatedDraftBytes,
///         memoryPolicy: .recommendedWorkingSet
///     ) {
///         try await LLMModelFactory.shared.loadContainer(configuration: draftConfig)
///     }
/// )
/// ```
public struct SpeculativeDecodingConfig: Sendable {

    package enum DraftModelSource: Sendable {
        case loaded(ModelContainer)
        case deferred(bytes: Int, @Sendable () async throws -> ModelContainer)
    }

    package let draftModelSource: DraftModelSource

    /// The lightweight model used to propose candidate tokens, when it was provided eagerly.
    ///
    /// Configurations initialized with a loader closure return `nil` because the
    /// draft model is loaded asynchronously by ``ChatSession`` only when speculation
    /// is admitted by the memory policy.
    public var draftModel: ModelContainer? {
        if case .loaded(let draftModel) = draftModelSource {
            return draftModel
        }
        return nil
    }

    /// Number of tokens proposed by the draft model per verification cycle.
    /// The default value of 5 offers a good balance between speed and accuracy.
    public let numDraftTokens: Int

    /// Optional memory policy used to decide whether auxiliary-model speculation should run.
    /// Pass `.recommendedWorkingSet` to fall back to regular generation when
    /// the combined main and draft model parameters exceed the recommended
    /// working set.
    public let memoryPolicy: SpeculativeDecodingMemoryPolicy?

    public init(
        draftModel: ModelContainer,
        numDraftTokens: Int = 5,
        memoryPolicy: SpeculativeDecodingMemoryPolicy? = nil
    ) {
        self.draftModelSource = .loaded(draftModel)
        self.numDraftTokens = numDraftTokens
        self.memoryPolicy = memoryPolicy
    }

    /// Initialize speculative decoding with a draft model loader.
    ///
    /// When a memory policy is present, `draftModelBytes` lets `ChatSession`
    /// decide whether to use speculative decoding before it loads the draft
    /// model. This is the preferred initializer when the draft model may not fit
    /// comfortably beside the main model.
    ///
    /// - Parameters:
    ///   - draftModelBytes: estimated resident parameter bytes for the draft model
    ///   - numDraftTokens: number of tokens proposed by the draft model per verification cycle
    ///   - memoryPolicy: optional memory policy used before loading the draft model
    ///   - loadDraftModel: closure that loads the draft model only if speculation is admitted
    public init(
        draftModelBytes: Int,
        numDraftTokens: Int = 5,
        memoryPolicy: SpeculativeDecodingMemoryPolicy? = nil,
        loadDraftModel: @escaping @Sendable () async throws -> ModelContainer
    ) {
        self.draftModelSource = .deferred(bytes: max(0, draftModelBytes), loadDraftModel)
        self.numDraftTokens = numDraftTokens
        self.memoryPolicy = memoryPolicy
    }

    package var estimatedDraftModelBytes: Int? {
        guard case .deferred(let bytes, _) = draftModelSource else {
            return nil
        }
        return bytes
    }

    package func loadDraftModel() async throws -> ModelContainer {
        switch draftModelSource {
        case .loaded(let draftModel):
            draftModel
        case .deferred(_, let load):
            try await load()
        }
    }
}

/// Protocol for processors that can render a reusable static-only prompt prefix.
///
/// Do not conform a processor to this protocol unless ``preparePrefix`` produces
/// a complete semantic prefix boundary for later full prompt renders.
/// Generic ``UserInputProcessor`` implementations are not assumed to have that
/// property.
public protocol PrefixUserInputProcessor: UserInputProcessor {
    func preparePrefix(
        input: UserInput,
        fullInput: LMInput
    ) async throws -> LMInput?
}

/// Simplified API for multi-turn conversations with LLMs and VLMs.
///
/// For example:
///
/// ```swift
/// let modelContainer = try await loadModelContainer(id: "mlx-community/Qwen3-4B-4bit")
/// let session = ChatSession(modelContainer)
/// print(try await session.respond(to: "What are two things to see in San Francisco?"))
/// print(try await session.respond(to: "How about a great place to eat?"))
/// ```
///
/// To enable speculative decoding for faster generation, pass a `SpeculativeDecodingConfig`:
///
/// ```swift
/// let draft = try await LLMModelFactory.shared.loadContainer(configuration: draftConfig)
/// let session = ChatSession(
///     modelContainer,
///     speculativeDecoding: SpeculativeDecodingConfig(draftModel: draft)
/// )
/// ```
///
/// - Note: `ChatSession` is not thread-safe. Each session should be used from a single
///   task/thread at a time. The underlying `ModelContainer` handles thread safety for
///   model operations.
public final class ChatSession {

    private enum Cache {
        case empty
        case kvcache([KVCache], draftKVCache: [KVCache]?, prompt: Prompt.State)
        case history([Chat.Message])
    }

    /// Tool schemas used by a chat session.
    public struct ToolConfiguration: Sendable {
        /// Tool schemas rendered into the model prompt.
        public var prompt: [ToolSpec]?
        /// Tool schemas used for parsing generated tool calls.
        public var parser: [ToolSpec]?

        public init(prompt: [ToolSpec]? = nil, parser: [ToolSpec]? = nil) {
            self.prompt = prompt
            self.parser = parser
        }

        public init(_ tools: [ToolSpec]?) {
            self.init(prompt: tools, parser: tools)
        }
    }

    private let model: ModelContainer
    /// System instructions rendered into the session prompt.
    public var instructions: String? {
        didSet {
            promptConfigurationRevision += 1
        }
    }
    private let cache: SerialAccessContainer<Cache>
    private let loadedDraftModel: SerialAccessContainer<ModelContainer?>
    public var processing: UserInput.Processing
    public var generateParameters: GenerateParameters
    public var additionalContext: [String: any Sendable]? {
        didSet {
            promptConfigurationRevision += 1
        }
    }
    /// Tool schemas for prompt rendering and tool-call parsing.
    public var toolConfiguration: ToolConfiguration {
        didSet {
            if !Self.toolSpecsEqual(oldValue.prompt, toolConfiguration.prompt) {
                promptConfigurationRevision += 1
            }
        }
    }
    /// Tool schemas used both for prompt rendering and tool-call parsing.
    ///
    /// This property preserves the original ``ChatSession`` behavior for callers
    /// that use a single tool schema list. Assigning it updates
    /// ``toolConfiguration`` for both prompt rendering and parsing.
    public var tools: [ToolSpec]? {
        get { toolConfiguration.prompt }
        set {
            toolConfiguration = .init(newValue)
        }
    }
    public var toolDispatch: (@Sendable (ToolCall) async throws -> String)?
    private var promptConfigurationRevision = 0

    private static func toolSpecsEqual(_ lhs: [ToolSpec]?, _ rhs: [ToolSpec]?) -> Bool {
        switch (lhs, rhs) {
        case (nil, nil):
            return true
        case (let lhs?, let rhs?):
            guard
                let lhsData = try? JSONSerialization.data(
                    withJSONObject: lhs, options: [.sortedKeys]),
                let rhsData = try? JSONSerialization.data(
                    withJSONObject: rhs, options: [.sortedKeys])
            else {
                return false
            }
            return lhsData == rhsData
        default:
            return false
        }
    }

    /// Speculative decoding configuration, nil if disabled.
    public let speculativeDecoding: SpeculativeDecodingConfig?

    /// Initialize the `ChatSession`.
    ///
    /// - Parameters:
    ///   - model: the ``ModelContainer``
    ///   - instructions: optional system instructions for the session
    ///   - speculativeDecoding: optional speculative decoding configuration for faster generation
    ///   - generateParameters: parameters that control generation
    ///   - processing: media processing configuration for images/videos
    ///   - tools: optional tool specifications
    ///   - toolDispatch: optional tool dispatch -- required for toolcalls if streaming strings rather than details
    ///   - additionalContext: optional model-specific context
    public init(
        _ model: ModelContainer,
        instructions: String? = nil,
        speculativeDecoding: SpeculativeDecodingConfig? = nil,
        generateParameters: GenerateParameters = .init(),
        processing: UserInput.Processing = .init(resize: CGSize(width: 512, height: 512)),
        additionalContext: [String: any Sendable]? = nil,
        tools: [ToolSpec]? = nil,
        toolDispatch: (@Sendable (ToolCall) async throws -> String)? = nil
    ) {
        self.model = model
        self.instructions = instructions
        self.cache = .init(.empty)
        self.loadedDraftModel = .init(speculativeDecoding?.draftModel)
        self.processing = processing
        self.generateParameters = generateParameters
        self.toolConfiguration = .init(tools)
        self.toolDispatch = toolDispatch
        self.additionalContext = additionalContext
        self.speculativeDecoding = speculativeDecoding
    }

    /// Initialize the `ChatSession`.
    ///
    /// - Parameters:
    ///   - model: the ``ModelContext``
    ///   - instructions: optional system instructions for the session
    ///   - speculativeDecoding: optional speculative decoding configuration for faster generation
    ///   - generateParameters: parameters that control generation
    ///   - processing: media processing configuration for images/videos
    ///   - tools: optional tool specifications
    ///   - toolDispatch: optional tool dispatch -- required for toolcalls if streaming strings rather than details
    ///   - additionalContext: optional model-specific context
    public init(
        _ model: ModelContext,
        instructions: String? = nil,
        speculativeDecoding: SpeculativeDecodingConfig? = nil,
        generateParameters: GenerateParameters = .init(),
        processing: UserInput.Processing = .init(resize: CGSize(width: 512, height: 512)),
        additionalContext: [String: any Sendable]? = nil,
        tools: [ToolSpec]? = nil,
        toolDispatch: (@Sendable (ToolCall) async throws -> String)? = nil
    ) {
        self.model = ModelContainer(context: model)
        self.instructions = instructions
        self.cache = .init(.empty)
        self.loadedDraftModel = .init(speculativeDecoding?.draftModel)
        self.processing = processing
        self.generateParameters = generateParameters
        self.toolConfiguration = .init(tools)
        self.toolDispatch = toolDispatch
        self.additionalContext = additionalContext
        self.speculativeDecoding = speculativeDecoding
    }

    /// Initialize the `ChatSession` with an existing message history.
    ///
    /// This enables "Prompt Re-hydration" for persistent chat applications.
    ///
    /// - Parameters:
    ///   - model: the ``ModelContainer``
    ///   - instructions: optional system instructions for the session
    ///   - history: The full array of messages to restore (including system prompt)
    ///   - speculativeDecoding: optional speculative decoding configuration for faster generation
    ///   - generateParameters: parameters that control generation
    ///   - processing: media processing configuration for images/videos
    ///   - tools: optional tool specifications
    ///   - toolDispatch: optional tool dispatch -- required for toolcalls if streaming strings rather than details
    ///   - additionalContext: optional model-specific context
    public init(
        _ model: ModelContainer,
        instructions: String? = nil,
        history: consuming [Chat.Message],
        speculativeDecoding: SpeculativeDecodingConfig? = nil,
        generateParameters: GenerateParameters = .init(),
        processing: UserInput.Processing = .init(resize: CGSize(width: 512, height: 512)),
        additionalContext: [String: any Sendable]? = nil,
        tools: [ToolSpec]? = nil,
        toolDispatch: (@Sendable (ToolCall) async throws -> String)? = nil
    ) {
        self.model = model
        self.instructions = instructions
        self.cache = .init(.history(history))
        self.loadedDraftModel = .init(speculativeDecoding?.draftModel)
        self.processing = processing
        self.generateParameters = generateParameters
        self.toolConfiguration = .init(tools)
        self.toolDispatch = toolDispatch
        self.additionalContext = additionalContext
        self.speculativeDecoding = speculativeDecoding
    }

    /// Initialize the `ChatSession` with an existing message history.
    ///
    /// This enables "Prompt Re-hydration" for persistent chat applications.
    ///
    /// - Parameters:
    ///   - model: the ``ModelContext``
    ///   - instructions: optional system instructions for the session
    ///   - history: The full array of messages to restore (including system prompt)
    ///   - speculativeDecoding: optional speculative decoding configuration for faster generation
    ///   - generateParameters: parameters that control generation
    ///   - processing: media processing configuration for images/videos
    ///   - tools: optional tool specifications
    ///   - toolDispatch: optional tool dispatch -- required for toolcalls if streaming strings rather than details
    ///   - additionalContext: optional model-specific context
    public init(
        _ model: ModelContext,
        instructions: String? = nil,
        history: [Chat.Message],
        speculativeDecoding: SpeculativeDecodingConfig? = nil,
        generateParameters: GenerateParameters = .init(),
        processing: UserInput.Processing = .init(resize: CGSize(width: 512, height: 512)),
        additionalContext: [String: any Sendable]? = nil,
        tools: [ToolSpec]? = nil,
        toolDispatch: (@Sendable (ToolCall) async throws -> String)? = nil
    ) {
        self.model = ModelContainer(context: model)
        self.instructions = instructions
        self.cache = .init(.history(history))
        self.loadedDraftModel = .init(speculativeDecoding?.draftModel)
        self.processing = processing
        self.generateParameters = generateParameters
        self.toolConfiguration = .init(tools)
        self.toolDispatch = toolDispatch
        self.additionalContext = additionalContext
        self.speculativeDecoding = speculativeDecoding
    }

    /// Initialize the `ChatSession` with a pre-built KV cache.
    ///
    /// This enables prefix caching: build a KV cache from a long shared context (e.g. a
    /// system prompt and document) once, save it via ``saveCache(to:)``, and restore it
    /// across multiple sessions to avoid re-prefilling the same tokens each time.
    ///
    /// > Important: If the cache was built from a session that already included system
    /// > instructions or prompt-rendered tools, do not pass the same values here; they
    /// > would be re-tokenized on each call to ``respond(to:role:images:videos:audios:)``
    /// > without matching KV state, producing incoherent output.
    ///
    /// - Parameters:
    ///   - model: the ``ModelContainer``
    ///   - instructions: optional system instructions for the session; leave `nil` if the
    ///     cache already encodes a system prompt
    ///   - cache: a non-empty `[KVCache]` previously loaded with ``loadPromptCache(url:)``,
    ///     matching the given model
    ///   - speculativeDecoding: optional speculative decoding configuration for faster generation
    ///   - generateParameters: parameters that control generation
    ///   - processing: media processing configuration for images/videos
    ///   - tools: optional tool specifications
    ///   - toolDispatch: optional tool dispatch -- required for toolcalls if streaming strings rather than details
    ///   - additionalContext: optional model-specific context
    public init(
        _ model: ModelContainer,
        instructions: String? = nil,
        cache: consuming [KVCache],
        speculativeDecoding: SpeculativeDecodingConfig? = nil,
        generateParameters: GenerateParameters = .init(),
        processing: UserInput.Processing = .init(resize: CGSize(width: 512, height: 512)),
        additionalContext: [String: any Sendable]? = nil,
        tools: [ToolSpec]? = nil,
        toolDispatch: (@Sendable (ToolCall) async throws -> String)? = nil
    ) {
        self.model = model
        self.instructions = instructions
        self.cache = .init(.kvcache(cache, draftKVCache: nil, prompt: .restored))
        self.loadedDraftModel = .init(speculativeDecoding?.draftModel)
        self.processing = processing
        self.generateParameters = generateParameters
        self.toolConfiguration = .init(tools)
        self.toolDispatch = toolDispatch
        self.additionalContext = additionalContext
        self.speculativeDecoding = speculativeDecoding
    }

    /// Initialize the `ChatSession` with a pre-built KV cache.
    ///
    /// This enables prefix caching: build a KV cache from a long shared context (e.g. a
    /// system prompt and document) once, save it via ``saveCache(to:)``, and restore it
    /// across multiple sessions to avoid re-prefilling the same tokens each time.
    ///
    /// > Important: If the cache was built from a session that already included system
    /// > instructions or prompt-rendered tools, do not pass the same values here; they
    /// > would be re-tokenized on each call to ``respond(to:role:images:videos:audios:)``
    /// > without matching KV state, producing incoherent output.
    ///
    /// - Parameters:
    ///   - model: the ``ModelContext``
    ///   - instructions: optional system instructions for the session; leave `nil` if the
    ///     cache already encodes a system prompt
    ///   - cache: a non-empty `[KVCache]` previously loaded with ``loadPromptCache(url:)``,
    ///     matching the given model
    ///   - speculativeDecoding: optional speculative decoding configuration for faster generation
    ///   - generateParameters: parameters that control generation
    ///   - processing: media processing configuration for images/videos
    ///   - tools: optional tool specifications
    ///   - toolDispatch: optional tool dispatch -- required for toolcalls if streaming strings rather than details
    ///   - additionalContext: optional model-specific context
    public init(
        _ model: ModelContext,
        instructions: String? = nil,
        cache: consuming [KVCache],
        speculativeDecoding: SpeculativeDecodingConfig? = nil,
        generateParameters: GenerateParameters = .init(),
        processing: UserInput.Processing = .init(resize: CGSize(width: 512, height: 512)),
        additionalContext: [String: any Sendable]? = nil,
        tools: [ToolSpec]? = nil,
        toolDispatch: (@Sendable (ToolCall) async throws -> String)? = nil
    ) {
        self.model = ModelContainer(context: model)
        self.instructions = instructions
        self.cache = .init(.kvcache(cache, draftKVCache: nil, prompt: .restored))
        self.loadedDraftModel = .init(speculativeDecoding?.draftModel)
        self.processing = processing
        self.generateParameters = generateParameters
        self.toolConfiguration = .init(tools)
        self.toolDispatch = toolDispatch
        self.additionalContext = additionalContext
        self.speculativeDecoding = speculativeDecoding
    }

    /// Produces a response to a prompt.
    ///
    /// - Parameters:
    ///   - prompt: the user prompt
    ///   - role: the message role (defaults to `.user`)
    ///   - images: list of images (for use with VLMs)
    ///   - videos: list of videos (for use with VLMs)
    ///   - audios: list of audios (for use with VLMs)
    /// - Returns: the model's response
    public func respond(
        to prompt: String,
        role: Chat.Message.Role = .user,
        images: consuming [UserInput.Image],
        videos: consuming [UserInput.Video],
        audios: consuming [UserInput.Audio]
    ) async throws -> String {
        var output = ""
        for try await chunk in streamResponse(
            to: prompt, role: role, images: images, videos: videos, audios: audios
        ) {
            output += chunk
        }
        return output
    }

    /// Produces a response to a prompt.
    ///
    /// - Parameters:
    ///   - prompt: the user prompt
    ///   - role: the message role (defaults to `.user`)
    ///   - image: optional image (for use with VLMs)
    ///   - video: optional video (for use with VLMs)
    ///   - audio: optional audio (for use with VLMs)
    /// - Returns: the model's response
    public func respond(
        to prompt: String,
        role: Chat.Message.Role = .user,
        image: consuming UserInput.Image? = nil,
        video: consuming UserInput.Video? = nil,
        audio: consuming UserInput.Audio? = nil
    ) async throws -> String {
        try await respond(
            to: prompt,
            role: role,
            images: image.map { [$0] } ?? [],
            videos: video.map { [$0] } ?? [],
            audios: audio.map { [$0] } ?? []
        )
    }

    /// Produces a response after appending a batch of structured chat messages.
    ///
    /// Use this to continue an existing session with non-user roles, such as one
    /// or more tool results, while preserving the session's KV cache.
    ///
    /// - Important: Initializing a new session from history must prefill that
    ///   history once. Reuse the same session with this method for subsequent
    ///   tool or agent turns to avoid repeatedly pre-filling the accumulated
    ///   transcript.
    ///
    /// - Parameter messages: chat messages to append before generation
    /// - Returns: the model's response
    public func respond(
        to messages: consuming [Chat.Message]
    ) async throws -> String {
        var output = ""
        for try await chunk in streamResponse(to: messages) {
            output += chunk
        }
        return output
    }

    /// Produces a streaming response to a prompt as Strings.
    ///
    /// - Parameters:
    ///   - prompt: the user prompt
    ///   - role: the message role (defaults to `.user`)
    ///   - images: list of images (for use with VLMs)
    ///   - videos: list of videos (for use with VLMs)
    ///   - audios: list of audios (for use with VLMs)
    /// - Returns: a stream of string chunks from the model
    public func streamResponse(
        to prompt: String,
        role: Chat.Message.Role = .user,
        images: consuming [UserInput.Image] = [],
        videos: consuming [UserInput.Video] = [],
        audios: consuming [UserInput.Audio] = []
    ) -> AsyncThrowingStream<String, Error> {
        streamMap(to: prompt, role: role, images: images, videos: videos, audios: audios) {
            $0.chunk
        }
    }

    /// Produces a streaming response after appending a batch of structured chat messages.
    ///
    /// Use this to continue an existing session with non-user roles, such as one
    /// or more tool results, while preserving the session's KV cache.
    ///
    /// - Parameter messages: chat messages to append before generation
    /// - Returns: a stream of string chunks from the model
    public func streamResponse(
        to messages: consuming [Chat.Message]
    ) -> AsyncThrowingStream<String, Error> {
        streamMap(messages: messages) {
            $0.chunk
        }
    }

    /// Produces a streaming response to a prompt as `Generation`.
    ///
    /// - Parameters:
    ///   - prompt: the user prompt
    ///   - role: the message role (defaults to `.user`)
    ///   - images: list of images (for use with VLMs)
    ///   - videos: list of videos (for use with VLMs)
    ///   - audios: list of audios (for use with VLMs)
    /// - Returns: a stream of `Generation` from the model
    public func streamDetails(
        to prompt: String,
        role: Chat.Message.Role = .user,
        images: consuming [UserInput.Image] = [],
        videos: consuming [UserInput.Video] = [],
        audios: consuming [UserInput.Audio] = [],
    ) -> AsyncThrowingStream<Generation, Error> {
        streamMap(to: prompt, role: role, images: images, videos: videos, audios: audios) {
            $0
        }
    }

    /// Produces a streaming response after appending a batch of structured chat messages as `Generation`.
    ///
    /// Use this to continue an existing session with non-user roles, such as one
    /// or more tool results, while preserving the session's KV cache.
    ///
    /// - Parameter messages: chat messages to append before generation
    /// - Returns: a stream of `Generation` from the model
    public func streamDetails(
        to messages: consuming [Chat.Message]
    ) -> AsyncThrowingStream<Generation, Error> {
        streamMap(messages: messages) {
            $0
        }
    }

    /// Groups prompt rendering, exact-prefix validation, and prompt-cache state transitions.
    ///
    /// The canonical prompt is rendered each turn. When the processor and cache support
    /// safe prefix reuse, an exact already-cached static prefix is sliced before model
    /// prefill.
    private enum Prompt {
        enum State: Equatable, Sendable {
            case pending
            case applied(staticPrefix: [Int], promptConfigurationRevision: Int)
            case restored
        }

        struct Configuration: Sendable {
            var instructions: String?
            var processing: UserInput.Processing
            var toolConfiguration: ToolConfiguration
            var additionalContext: [String: any Sendable]?
            var revision: Int
            var allowsPrefixReuse: Bool
        }

        struct PreparedInput {
            var input: LMInput
            var state: State
        }

        static func fullMessages(
            turnMessages: [Chat.Message],
            instructions: String?
        ) -> [Chat.Message] {
            guard let instructions else {
                return turnMessages
            }

            return [.system(instructions)] + turnMessages
        }

        static func staticPrefixTokens(
            staticInput: LMInput,
            fullInput: LMInput
        ) -> [Int]? {
            guard staticInput.text.tokens.ndim == 1,
                fullInput.text.tokens.ndim == 1,
                staticInput.text.mask == nil,
                fullInput.text.mask == nil
            else {
                return nil
            }

            let staticTokens = staticInput.text.tokens.asArray(Int.self)
            guard !staticTokens.isEmpty else {
                return nil
            }

            let fullTokens = fullInput.text.tokens.asArray(Int.self)
            guard fullTokens.starts(with: staticTokens) else {
                return nil
            }

            return staticTokens
        }

        static func suffixInput(fullInput: LMInput, dropping prefixTokens: [Int]) -> LMInput? {
            guard fullInput.text.tokens.ndim == 1, fullInput.text.mask == nil else {
                return nil
            }

            let fullTokens = fullInput.text.tokens.asArray(Int.self)
            guard fullTokens.starts(with: prefixTokens) else {
                return nil
            }

            return LMInput(
                text: .init(tokens: fullInput.text.tokens[prefixTokens.count...]),
                image: fullInput.image,
                video: fullInput.video,
                audio: fullInput.audio)
        }

        static func prepareInput(
            processor: any UserInputProcessor,
            turnMessages: [Chat.Message],
            kvCache: [KVCache],
            promptState: State,
            configuration: Configuration
        ) async throws -> PreparedInput {
            switch promptState {
            case .restored:
                let userInput = UserInput(
                    chat: fullMessages(
                        turnMessages: turnMessages,
                        instructions: configuration.instructions),
                    processing: configuration.processing,
                    tools: configuration.toolConfiguration.prompt,
                    additionalContext: configuration.additionalContext)
                return PreparedInput(
                    input: try await processor.prepare(input: userInput),
                    state: promptState)

            case .applied(_, let storedRevision) where storedRevision != configuration.revision:
                throw ChatSessionError.promptCacheMismatch

            case .pending, .applied:
                if configuration.instructions == nil
                    && configuration.toolConfiguration.prompt == nil
                {
                    let userInput = UserInput(
                        chat: turnMessages,
                        processing: configuration.processing,
                        tools: nil,
                        additionalContext: configuration.additionalContext)
                    let preparedState =
                        if promptState == .pending {
                            State.applied(
                                staticPrefix: [],
                                promptConfigurationRevision: configuration.revision)
                        } else {
                            promptState
                        }
                    return PreparedInput(
                        input: try await processor.prepare(input: userInput),
                        state: preparedState)
                }

                let fullUserInput = UserInput(
                    chat: fullMessages(
                        turnMessages: turnMessages,
                        instructions: configuration.instructions),
                    processing: configuration.processing,
                    tools: configuration.toolConfiguration.prompt,
                    additionalContext: configuration.additionalContext)
                let fullInput = try await processor.prepare(input: fullUserInput)
                let prefixProcessor = processor as? any PrefixUserInputProcessor
                let canReuseStaticPrefix =
                    configuration.allowsPrefixReuse
                    && prefixProcessor != nil
                    && !kvCache.isEmpty
                    && kvCache.allSatisfy(\.supportsStaticPrefixReuse)

                switch promptState {
                case .pending:
                    let staticPrefix: [Int]?
                    if canReuseStaticPrefix {
                        let staticInput: LMInput?
                        if let prefixProcessor {
                            staticInput = try? await prefixProcessor.preparePrefix(
                                input: fullUserInput,
                                fullInput: fullInput)
                        } else {
                            staticInput = nil
                        }
                        staticPrefix = staticInput.flatMap {
                            staticPrefixTokens(staticInput: $0, fullInput: fullInput)
                        }
                    } else {
                        staticPrefix = nil
                    }
                    return PreparedInput(
                        input: fullInput,
                        state: .applied(
                            staticPrefix: staticPrefix ?? [],
                            promptConfigurationRevision: configuration.revision))

                case .applied(let staticPrefix, _):
                    guard canReuseStaticPrefix, !staticPrefix.isEmpty else {
                        return PreparedInput(input: fullInput, state: promptState)
                    }
                    guard
                        let suffixInput = suffixInput(fullInput: fullInput, dropping: staticPrefix)
                    else {
                        throw ChatSessionError.promptCacheMismatch
                    }
                    return PreparedInput(input: suffixInput, state: promptState)

                case .restored:
                    preconditionFailure("handled above")
                }
            }
        }
    }

    /// Produces a streaming response to a prompt by transforming the
    /// raw `Generation` values.
    ///
    /// - Parameters:
    ///   - prompt: the user prompt
    ///   - images: list of images (for use with VLMs)
    ///   - videos: list of videos (for use with VLMs)
    ///   - audios: list of audios (for use with VLMs)
    /// - Returns: a stream of transformed values from the model
    private func streamMap<R: Sendable>(
        to prompt: String,
        role: Chat.Message.Role,
        images: consuming [UserInput.Image] = [],
        videos: consuming [UserInput.Video] = [],
        audios: consuming [UserInput.Audio] = [],
        transform: @Sendable @escaping (Generation) -> R?
    ) -> AsyncThrowingStream<R, Error> {
        streamMap(
            messages: [
                .init(role: role, content: prompt, images: images, videos: videos, audios: audios)
            ],
            transform: transform
        )
    }

    private func streamMap<R: Sendable>(
        messages: consuming [Chat.Message],
        transform: @Sendable @escaping (Generation) -> R?
    ) -> AsyncThrowingStream<R, Error> {
        let (stream, continuation) = AsyncThrowingStream<R, Error>.makeStream()

        // images and videos are not Sendable (MLXArray) but they are consumed
        // and are only being sent to the inner async
        let inputMessages = SendableBox<[Chat.Message]>(messages)
        let model = self.model
        let instructions = self.instructions
        let processing = self.processing
        let toolConfiguration = self.toolConfiguration
        let toolDispatch = self.toolDispatch
        let additionalContext = self.additionalContext
        let promptInputConfiguration = Prompt.Configuration(
            instructions: instructions,
            processing: processing,
            toolConfiguration: toolConfiguration,
            additionalContext: additionalContext,
            revision: self.promptConfigurationRevision,
            allowsPrefixReuse: generateParameters.processor() == nil)
        let cache = self.cache
        let loadedDraftModel = self.loadedDraftModel
        let generateParameters = self.generateParameters
        let speculativeDecoding = self.speculativeDecoding

        let task = Task { @Sendable in
            do {
                try await cache.update { cache in

                    // these are all Sendable
                    let processor = await model.processor
                    let tokenizer = await model.tokenizer
                    let modelConfiguration = await model.configuration

                    var messages: [Chat.Message] = []

                    // prepare the cache, if needed.  note:
                    // this is using the LanguageModel (not Sendable) outside
                    // the protective lock.  Assuming the weights are not
                    // being mutated behind the scenes, this will obey the MLXArray
                    // contract that they be evaluated if used across threads.
                    // This is internal to the implementation and this technique
                    // should not be used in calling code.
                    //
                    // The benefit is that callers can be running multiple
                    // ChatSessions in parallel, as long as the instances
                    // are distinct.  In particular the KVCache cannot
                    // be shared and that is the lock that is held here.

                    let model = await model.perform { context in
                        SendableBox(context.model)
                    }.consume()

                    var kvCache: [KVCache]
                    var draftKVCache: [KVCache]?
                    var promptState: Prompt.State
                    switch cache {
                    case .empty:
                        kvCache = model.newCache(parameters: generateParameters)
                        promptState = .pending
                        cache = .kvcache(kvCache, draftKVCache: nil, prompt: promptState)

                    case .kvcache(let array, let storedDraftCache, let storedPromptState):
                        kvCache = array
                        draftKVCache = storedDraftCache
                        promptState = storedPromptState

                    case .history(let history):
                        // the KVCache is represented by a chat history
                        kvCache = model.newCache(parameters: generateParameters)
                        promptState = .pending
                        cache = .kvcache(kvCache, draftKVCache: nil, prompt: promptState)
                        messages.append(contentsOf: history)
                    }

                    // prepare the input
                    messages.append(contentsOf: inputMessages.consume())

                    // loop can restart on tool calls
                    restart: while !messages.isEmpty {
                        let turnMessages = messages
                        messages.removeAll()

                        let preparedPrompt = try await Prompt.prepareInput(
                            processor: processor,
                            turnMessages: turnMessages,
                            kvCache: kvCache,
                            promptState: promptState,
                            configuration: promptInputConfiguration)
                        let input = preparedPrompt.input
                        let preparedPromptState = preparedPrompt.state

                        // Select the token iterator based on speculative decoding configuration.
                        let (genStream, genTask): (AsyncStream<Generation>, Task<Void, Never>)
                        func defaultGeneration() throws -> (
                            AsyncStream<Generation>, Task<Void, Never>
                        ) {
                            let iterator = try TokenIterator(
                                input: input, model: model, cache: kvCache,
                                parameters: generateParameters)

                            return MLXLMCommon.generateTask(
                                promptTokenCount: input.text.tokens.size,
                                modelConfiguration: modelConfiguration,
                                tokenizer: tokenizer,
                                iterator: iterator,
                                tools: toolConfiguration.parser
                            )
                        }

                        if let speculativeDecoding {
                            var shouldFallBackBeforeLoadingDraft = false
                            if let memoryPolicy = speculativeDecoding.memoryPolicy,
                                let draftModelBytes =
                                    speculativeDecoding.estimatedDraftModelBytes
                            {
                                let memoryEvaluation = memoryPolicy.evaluate(
                                    mainModelBytes:
                                        SpeculativeDecodingMemoryPolicy
                                        .modelWeightBytes(model),
                                    draftModelBytes: draftModelBytes
                                )
                                if !memoryEvaluation.shouldUseSpeculativeDecoding {
                                    if memoryEvaluation.action == .fail {
                                        throw SpeculativeDecodingMemoryError(
                                            evaluation: memoryEvaluation)
                                    }

                                    shouldFallBackBeforeLoadingDraft = true
                                }
                            }

                            if shouldFallBackBeforeLoadingDraft {
                                (genStream, genTask) = try defaultGeneration()
                            } else {
                                let cachedDraftContainer = await loadedDraftModel.read { $0 }
                                let draftContainer: ModelContainer
                                if let cachedDraftContainer {
                                    draftContainer = cachedDraftContainer
                                } else {
                                    draftContainer = try await speculativeDecoding.loadDraftModel()
                                }

                                // Extract the draft model from its container (same pattern as the main model).
                                let draftModel = await draftContainer.perform { context in
                                    SendableBox(context.model)
                                }.consume()

                                let memoryEvaluation = speculativeDecoding.memoryPolicy?.evaluate(
                                    mainModel: model,
                                    draftModel: draftModel
                                )
                                if let memoryEvaluation,
                                    !memoryEvaluation.shouldUseSpeculativeDecoding
                                {
                                    if memoryEvaluation.action == .fail {
                                        throw SpeculativeDecodingMemoryError(
                                            evaluation: memoryEvaluation)
                                    }

                                    (genStream, genTask) = try defaultGeneration()
                                } else {
                                    if cachedDraftContainer == nil {
                                        await loadedDraftModel.update { storedDraftModel in
                                            if storedDraftModel == nil {
                                                storedDraftModel = draftContainer
                                            }
                                        }
                                    }

                                    // Allocate the draft KV cache once and reuse it across turns,
                                    // exactly like the main model's KV cache.
                                    if draftKVCache == nil {
                                        draftKVCache = draftModel.newCache(
                                            parameters: generateParameters)
                                        cache = .kvcache(
                                            kvCache,
                                            draftKVCache: draftKVCache,
                                            prompt: promptState)
                                    }
                                    let draftCache = draftKVCache!

                                    let iterator = try SpeculativeTokenIterator(
                                        input: input,
                                        mainModel: model,
                                        draftModel: draftModel,
                                        mainCache: kvCache,
                                        draftCache: draftCache,
                                        parameters: generateParameters,
                                        numDraftTokens: speculativeDecoding.numDraftTokens
                                    )

                                    (genStream, genTask) = MLXLMCommon.generateTask(
                                        promptTokenCount: input.text.tokens.size,
                                        modelConfiguration: modelConfiguration,
                                        tokenizer: tokenizer,
                                        iterator: iterator,
                                        tools: toolConfiguration.parser
                                    )
                                }
                            }
                        } else {
                            // Standard path with no speculative decoding.
                            (genStream, genTask) = try defaultGeneration()
                        }

                        if preparedPromptState != promptState {
                            promptState = preparedPromptState
                            cache = .kvcache(
                                kvCache, draftKVCache: draftKVCache, prompt: promptState)
                        }

                        var pendingToolCalls: [ToolCall] = []

                        for await item in genStream {
                            // collect tool calls for dispatch; if no
                            // toolDispatch the caller handles them via
                            // the transform (streamDetails path)
                            if let toolCall = item.toolCall, toolDispatch != nil {
                                pendingToolCalls.append(toolCall)
                            } else if let value = transform(item) {
                                if case .terminated = continuation.yield(value) {
                                    break
                                }
                            }
                        }

                        // wait for the task to complete -- this is important in
                        // the case where we broke the loop early as the generation
                        // work may continue (briefly) and use the KVCache
                        await genTask.value

                        // dispatch all tool calls from this generation pass
                        if let toolDispatch, !pendingToolCalls.isEmpty,
                            !Task.isCancelled
                        {
                            for toolCall in pendingToolCalls {
                                let toolResult = try await toolDispatch(toolCall)
                                messages.append(.tool(toolResult))
                            }
                            continue restart
                        }
                    }

                    continuation.finish()
                }
            } catch {
                continuation.finish(throwing: error)
            }
        }

        continuation.onTermination = { _ in
            task.cancel()
        }

        return stream
    }

    /// Produces a streaming response to a prompt.
    ///
    /// - Parameters:
    ///   - prompt: the user prompt
    ///   - image: optional image (for use with VLMs)
    ///   - video: optional video (for use with VLMs)
    ///   - audio: optional audio (for use with VLMs)
    /// - Returns: a stream of string chunks from the model
    public func streamResponse(
        to prompt: String,
        image: consuming UserInput.Image? = nil,
        video: consuming UserInput.Video? = nil,
        audio: consuming UserInput.Audio? = nil
    ) -> AsyncThrowingStream<String, Error> {
        streamResponse(
            to: prompt,
            images: image.map { [$0] } ?? [],
            videos: video.map { [$0] } ?? [],
            audios: audio.map { [$0] } ?? [],
        )
    }

    /// Clear the session history and cache, preserving system instructions.
    public func clear() async {
        await cache.update { cache in
            cache = .empty
        }
    }

    /// Wait for exclusive access to the KVCache.
    ///
    /// This is useful for cases where a program is terminating and wants to ensure that any
    /// async operations are complete.
    public func synchronize() async {
        await cache.read { _ in }
    }

    /// Visit the current cache value, if realized as a `[KVCache]`.
    ///
    /// This method is meant for test support.
    func withCache<R: Sendable>(_ body: @Sendable ([KVCache]?) async throws -> R) async rethrows
        -> R?
    {
        try await cache.read { cache in
            switch cache {
            case .kvcache(let cache, _, _):
                return try await body(cache)
            default:
                return try await body(nil)
            }
        }
    }

    /// Saves the current KV cache to disk.
    ///
    /// Use one of the initializers that accept a `cache` parameter together with
    /// ``loadPromptCache(url:)`` to restore the saved cache in a future session.
    ///
    /// - Parameter url: the file URL to write the cache to
    /// - Throws: ``ChatSessionError/noCacheAvailable`` if no generation has occurred yet,
    ///   or any error thrown by the underlying file write
    public func saveCache(to url: URL) async throws {
        try await cache.read { cache in
            switch cache {
            case .kvcache(let cache, _, _):
                try savePromptCache(url: url, cache: cache)
            default:
                throw ChatSessionError.noCacheAvailable
            }
        }
    }
}

/// Errors thrown by ``ChatSession``.
public enum ChatSessionError: LocalizedError {
    /// ``ChatSession/saveCache(to:)`` was called before any generation occurred.
    case noCacheAvailable
    /// The stored prompt prefix no longer matches the rendered prompt.
    case promptCacheMismatch

    public var errorDescription: String? {
        switch self {
        case .noCacheAvailable:
            "No KV cache is available. Call respond() or streamResponse() before saveCache(to:)."
        case .promptCacheMismatch:
            "The cached prompt prefix no longer matches the rendered prompt. Call clear() after changing instructions, tools, or additionalContext."
        }
    }
}
