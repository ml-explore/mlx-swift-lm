// Copyright © 2025 Apple Inc.

import Foundation
import MLX

#if canImport(CoreGraphics)
import CoreGraphics
#endif

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
/// - Note: A session retains its structured transcript, including referenced media,
///   until ``clear()`` is called. Long-running VLM sessions should clear or replace
///   the session when that retained context is no longer needed.
public final class ChatSession {

    private struct Conversation {
        var messages: [Chat.Message]
        /// Exact tokens represented by the main KV cache when its count
        /// matches the cache offset. An empty value paired with a nonzero
        /// cache offset is an invalidated ledger and forces a rebuild.
        var cachedTokens: [Int]

        @discardableResult
        mutating func record(
            _ assistant: AssistantGeneration,
            generatedTokens: [Int],
            cacheOffset: Int?
        ) -> Bool {
            guard assistant.shouldRecord else {
                // A cancelled or semantically empty generation has no
                // assistant turn to replay. Its cache may nevertheless
                // contain generated or lookahead tokens, so invalidate
                // the ledger and rebuild from the retained messages.
                cachedTokens.removeAll()
                return false
            }

            let cachedTokenCount = cachedTokens.count
            if let cacheOffset,
                cacheOffset >= cachedTokenCount,
                cacheOffset - cachedTokenCount <= generatedTokens.count
            {
                cachedTokens.append(
                    contentsOf: generatedTokens.prefix(cacheOffset - cachedTokenCount))
            } else {
                // The iterator/cache relationship could not be proven
                // (for example, a speculative iterator retained
                // un-emitted lookahead).
                cachedTokens.removeAll()
            }

            messages.append(
                .assistant(
                    assistant.content,
                    toolCalls: assistant.toolCalls.isEmpty ? nil : assistant.toolCalls))
            return true
        }
    }

    private struct GenerationRun {
        let stream: AsyncStream<Generation>
        let task: Task<[Int], Never>

        init(_ pair: (AsyncStream<Generation>, Task<[Int], Never>)) {
            (stream, task) = pair
        }
    }

    private struct AssistantGeneration {
        var content = ""
        var toolCalls: [ToolCall] = []
        var stopReason: GenerateStopReason?
        var wasTerminatedByConsumer = false

        var shouldRecord: Bool {
            (!content.isEmpty || !toolCalls.isEmpty)
                && !wasTerminatedByConsumer
                && stopReason != .cancelled
        }

        mutating func consume(_ item: Generation) {
            if let chunk = item.chunk {
                content += chunk
            }
            if let toolCall = item.toolCall {
                toolCalls.append(toolCall)
            }
            if let info = item.info {
                stopReason = info.stopReason
            }
        }
    }

    private enum Cache {
        /// `state` is the per-call model state (e.g. M-RoPE rope deltas)
        /// from the last prefill against this cache. It must survive across
        /// turns: without it, a model that anchors positions on carried
        /// state re-derives them from a cold start on the next turn.
        case empty
        case kvcache(
            [KVCache], draftKVCache: [KVCache]?, state: LMOutput.State?,
            conversation: Conversation?
        )
        case history([Chat.Message])
    }

    private let model: ModelContainer
    public var instructions: String?
    private let cache: SerialAccessContainer<Cache>
    private let loadedDraftModel: SerialAccessContainer<ModelContainer?>
    public var processing: UserInput.Processing
    public var generateParameters: GenerateParameters
    public var additionalContext: [String: any Sendable]?
    public var tools: [ToolSpec]?
    public var toolDispatch: (@Sendable (ToolCall) async throws -> String)?

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
        self.tools = tools
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
        self.tools = tools
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
        self.tools = tools
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
        self.tools = tools
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
    /// > instructions, do not pass the same `instructions` here — they would be
    /// > re-tokenized on each call to ``respond(to:role:images:videos:audios:)`` without matching
    /// > KV state, producing incoherent output.
    ///
    /// > Important: Prefer the initializer that takes a `promptCache`. A cache and the model state
    /// > that belongs with it must travel together: passing `cache: snapshot.cache` while dropping
    /// > `snapshot.state` compiles, but positions later tokens as if the cached prefix contained no
    /// > images. Models that carry a position anchor throw ``ContinuationStateError`` rather than
    /// > continue without it.
    ///
    /// > Important: A raw cache does not carry the structured chat transcript.
    /// > This initializer therefore preserves fragment-based continuation behavior.
    /// > Use a history initializer when later turns must re-render the full conversation.
    ///
    /// - Parameters:
    ///   - model: the ``ModelContainer``
    ///   - instructions: optional system instructions for the session — leave `nil` if the
    ///     cache already encodes a system prompt
    ///   - cache: a non-empty `[KVCache]`, matching the given model — from
    ///     ``loadPromptCacheSnapshot(url:)`` if it was saved to disk
    ///   - state: the model state captured with that cache; pass `snapshot.state` whenever the
    ///     cache came from a snapshot
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
        state: LMOutput.State? = nil,
        speculativeDecoding: SpeculativeDecodingConfig? = nil,
        generateParameters: GenerateParameters = .init(),
        processing: UserInput.Processing = .init(resize: CGSize(width: 512, height: 512)),
        additionalContext: [String: any Sendable]? = nil,
        tools: [ToolSpec]? = nil,
        toolDispatch: (@Sendable (ToolCall) async throws -> String)? = nil
    ) {
        self.model = model
        self.instructions = instructions
        self.cache = .init(
            .kvcache(cache, draftKVCache: nil, state: state, conversation: nil))
        self.loadedDraftModel = .init(speculativeDecoding?.draftModel)
        self.processing = processing
        self.generateParameters = generateParameters
        self.tools = tools
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
    /// > instructions, do not pass the same `instructions` here — they would be
    /// > re-tokenized on each call to ``respond(to:role:images:videos:audios:)`` without matching
    /// > KV state, producing incoherent output.
    ///
    /// > Important: Prefer the initializer that takes a `promptCache`. A cache and the model state
    /// > that belongs with it must travel together: passing `cache: snapshot.cache` while dropping
    /// > `snapshot.state` compiles, but positions later tokens as if the cached prefix contained no
    /// > images. Models that carry a position anchor throw ``ContinuationStateError`` rather than
    /// > continue without it.
    ///
    /// > Important: A raw cache does not carry the structured chat transcript.
    /// > This initializer therefore preserves fragment-based continuation behavior.
    /// > Use a history initializer when later turns must re-render the full conversation.
    ///
    /// - Parameters:
    ///   - model: the ``ModelContext``
    ///   - instructions: optional system instructions for the session — leave `nil` if the
    ///     cache already encodes a system prompt
    ///   - cache: a non-empty `[KVCache]`, matching the given model — from
    ///     ``loadPromptCacheSnapshot(url:)`` if it was saved to disk
    ///   - state: the model state captured with that cache; pass `snapshot.state` whenever the
    ///     cache came from a snapshot
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
        state: LMOutput.State? = nil,
        speculativeDecoding: SpeculativeDecodingConfig? = nil,
        generateParameters: GenerateParameters = .init(),
        processing: UserInput.Processing = .init(resize: CGSize(width: 512, height: 512)),
        additionalContext: [String: any Sendable]? = nil,
        tools: [ToolSpec]? = nil,
        toolDispatch: (@Sendable (ToolCall) async throws -> String)? = nil
    ) {
        self.model = ModelContainer(context: model)
        self.instructions = instructions
        self.cache = .init(
            .kvcache(cache, draftKVCache: nil, state: state, conversation: nil))
        self.loadedDraftModel = .init(speculativeDecoding?.draftModel)
        self.processing = processing
        self.generateParameters = generateParameters
        self.tools = tools
        self.toolDispatch = toolDispatch
        self.additionalContext = additionalContext
        self.speculativeDecoding = speculativeDecoding
    }

    /// Initialize the `ChatSession` with a prompt cache and the model state that belongs with it.
    ///
    /// This is the preferred way to start a session from a saved cache: the snapshot keeps the KV
    /// arrays and the model state together, so neither can be dropped. Read one from disk with
    /// ``loadPromptCacheSnapshot(url:)``, or build one in process with
    /// ``PromptCacheSnapshot/init(cache:metadata:state:)``.
    ///
    /// Models that do not carry model state simply ignore it, so this works for any model.
    ///
    /// > Important: A snapshot carries the cache and model state, not the structured chat
    /// > transcript, so this initializer uses fragment-based continuation. A later
    /// > image-bearing turn continues warm from the cached prefix rather than rebuilding and
    /// > re-rendering earlier messages. Positions stay correct; the prompt differs.
    /// > Use a history initializer when later turns must re-render the full conversation.
    public convenience init(
        _ model: ModelContainer,
        instructions: String? = nil,
        promptCache: consuming PromptCacheSnapshot,
        speculativeDecoding: SpeculativeDecodingConfig? = nil,
        generateParameters: GenerateParameters = .init(),
        processing: UserInput.Processing = .init(resize: CGSize(width: 512, height: 512)),
        additionalContext: [String: any Sendable]? = nil,
        tools: [ToolSpec]? = nil,
        toolDispatch: (@Sendable (ToolCall) async throws -> String)? = nil
    ) {
        self.init(
            model,
            instructions: instructions,
            cache: promptCache.cache,
            state: promptCache.state,
            speculativeDecoding: speculativeDecoding,
            generateParameters: generateParameters,
            processing: processing,
            additionalContext: additionalContext,
            tools: tools,
            toolDispatch: toolDispatch)
    }

    /// Initialize the `ChatSession` with a prompt cache and the model state that belongs with it.
    ///
    /// This is the preferred way to start a session from a saved cache: the snapshot keeps the KV
    /// arrays and the model state together, so neither can be dropped. Read one from disk with
    /// ``loadPromptCacheSnapshot(url:)``, or build one in process with
    /// ``PromptCacheSnapshot/init(cache:metadata:state:)``.
    ///
    /// Models that do not carry model state simply ignore it, so this works for any model.
    ///
    /// > Important: A snapshot carries the cache and model state, not the structured chat
    /// > transcript, so this initializer uses fragment-based continuation. A later
    /// > image-bearing turn continues warm from the cached prefix rather than rebuilding and
    /// > re-rendering earlier messages. Positions stay correct; the prompt differs.
    /// > Use a history initializer when later turns must re-render the full conversation.
    public convenience init(
        _ model: ModelContext,
        instructions: String? = nil,
        promptCache: consuming PromptCacheSnapshot,
        speculativeDecoding: SpeculativeDecodingConfig? = nil,
        generateParameters: GenerateParameters = .init(),
        processing: UserInput.Processing = .init(resize: CGSize(width: 512, height: 512)),
        additionalContext: [String: any Sendable]? = nil,
        tools: [ToolSpec]? = nil,
        toolDispatch: (@Sendable (ToolCall) async throws -> String)? = nil
    ) {
        self.init(
            ModelContainer(context: model),
            instructions: instructions,
            promptCache: promptCache,
            speculativeDecoding: speculativeDecoding,
            generateParameters: generateParameters,
            processing: processing,
            additionalContext: additionalContext,
            tools: tools,
            toolDispatch: toolDispatch)
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

        let task = Task {
            [
                model,
                instructions, processing, tools, toolDispatch,
                additionalContext, cache, loadedDraftModel, generateParameters, speculativeDecoding
            ] in
            do {
                try await cache.update { cache in

                    // these are all Sendable
                    let processor = await model.processor
                    let tokenizer = await model.tokenizer
                    let modelConfiguration = await model.configuration

                    var leadingMessages: [Chat.Message] = []
                    if let instructions {
                        leadingMessages.append(.system(instructions))
                    }

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
                    // Per-call model state (e.g. M-RoPE rope deltas) carried
                    // across turns alongside the KV cache; updated after each
                    // prefill and stored back at the end of the turn.
                    var lmState: LMOutput.State?
                    var conversation: Conversation?
                    switch cache {
                    case .empty:
                        kvCache = model.newCache(parameters: generateParameters)
                        conversation = Conversation(messages: [], cachedTokens: [])
                        cache = .kvcache(
                            kvCache, draftKVCache: nil, state: nil, conversation: conversation)

                    case .kvcache(
                        let storedCache, let storedDraftCache, let storedState,
                        let storedConversation
                    ):
                        kvCache = storedCache
                        draftKVCache = storedDraftCache
                        lmState = storedState
                        conversation = storedConversation

                    case .history(let history):
                        // the KVCache is represented by a chat history
                        kvCache = model.newCache(parameters: generateParameters)
                        conversation = Conversation(messages: history, cachedTokens: [])
                        cache = .kvcache(
                            kvCache, draftKVCache: nil, state: nil, conversation: conversation)
                    }

                    var pendingMessages = inputMessages.consume()

                    // loop can restart on tool calls
                    restart: while !pendingMessages.isEmpty {
                        let templateMessages: [Chat.Message]
                        let conversationMessageCountBeforePending: Int?
                        if var currentConversation = conversation {
                            conversationMessageCountBeforePending =
                                currentConversation.messages.count
                            currentConversation.messages.append(contentsOf: pendingMessages)
                            templateMessages = leadingMessages + currentConversation.messages
                            conversation = currentConversation
                        } else {
                            conversationMessageCountBeforePending = nil
                            // A cache restored without its transcript cannot be reconciled
                            // against a complete chat template. Preserve the legacy prefix
                            // continuation behavior for this explicitly low-level API.
                            templateMessages = leadingMessages + pendingMessages
                        }
                        let containsNewMedia = pendingMessages.contains {
                            !$0.images.isEmpty || !$0.videos.isEmpty || !$0.audios.isEmpty
                        }

                        let userInput = UserInput(
                            chat: templateMessages,
                            processing: processing,
                            tools: tools, additionalContext: additionalContext)
                        let preparedInput = try await processor.prepare(input: userInput)
                        var input = preparedInput
                        pendingMessages.removeAll()

                        let speculativeMemoryEvaluation: SpeculativeDecodingMemoryEvaluation?
                        if let speculativeDecoding,
                            let memoryPolicy = speculativeDecoding.memoryPolicy,
                            let draftModelBytes = speculativeDecoding.estimatedDraftModelBytes
                        {
                            speculativeMemoryEvaluation = memoryPolicy.evaluate(
                                mainModelBytes:
                                    SpeculativeDecodingMemoryPolicy.modelWeightBytes(model),
                                draftModelBytes: draftModelBytes)
                        } else {
                            speculativeMemoryEvaluation = nil
                        }
                        let willFallBackBeforeLoadingDraft =
                            speculativeMemoryEvaluation.map {
                                !$0.shouldUseSpeculativeDecoding && $0.action != .fail
                            } ?? false

                        var reusedMainCacheWithoutDraft = false
                        if var currentConversation = conversation {
                            let fullTokenIds = input.text.tokens.asArray(Int.self)
                            let cachedTokenIds = currentConversation.cachedTokens
                            let cacheOffset = kvCache.first?.offset
                            let allMainCachesAreAligned =
                                !kvCache.isEmpty
                                && kvCache.allSatisfy { $0.offset == cachedTokenIds.count }
                            let draftCacheIsAligned: Bool
                            let allDraftCachesAreAligned: Bool
                            if let draftKVCache {
                                draftCacheIsAligned =
                                    draftKVCache.first?.offset == cachedTokenIds.count
                                allDraftCachesAreAligned =
                                    !draftKVCache.isEmpty
                                    && draftKVCache.allSatisfy {
                                        $0.offset == cachedTokenIds.count
                                    }
                            } else {
                                // Tentatively reuse the main cache. If speculative
                                // decoding is admitted below, both caches are rebuilt
                                // from `preparedInput`; a fallback can keep this suffix.
                                draftCacheIsAligned = true
                                allDraftCachesAreAligned = true
                            }
                            let extendsCachedPrefix =
                                cacheOffset == cachedTokenIds.count
                                && draftCacheIsAligned
                                && fullTokenIds.starts(with: cachedTokenIds)

                            if extendsCachedPrefix && !containsNewMedia
                                && fullTokenIds.count > cachedTokenIds.count
                                && input.text.mask == nil
                            {
                                let suffix = Array(fullTokenIds.dropFirst(cachedTokenIds.count))
                                input = LMInput(tokens: MLXArray(suffix))
                                reusedMainCacheWithoutDraft =
                                    speculativeDecoding != nil
                                    && !willFallBackBeforeLoadingDraft
                                    && draftKVCache == nil
                                    && !cachedTokenIds.isEmpty
                            } else if cacheOffset != 0 {
                                let commonPrefixCount = zip(fullTokenIds, cachedTokenIds)
                                    .prefix { $0 == $1 }
                                    .count
                                let trimCount = cachedTokenIds.count - commonPrefixCount
                                let hasPreparedMedia =
                                    input.image != nil || input.video != nil || input.audio != nil
                                let canReuseCommonPrefix =
                                    commonPrefixCount > 0
                                    && commonPrefixCount < fullTokenIds.count
                                    && trimCount > 0
                                    && allMainCachesAreAligned
                                    && allDraftCachesAreAligned
                                    && !containsNewMedia
                                    && !hasPreparedMedia
                                    && input.text.mask == nil
                                    && lmState == nil
                                    && canTrimPromptCache(kvCache)
                                    && (draftKVCache.map(canTrimPromptCache) ?? true)

                                if canReuseCommonPrefix {
                                    let mainTrimmed = trimPromptCache(
                                        kvCache, numTokens: trimCount)
                                    let draftTrimmed = draftKVCache.map {
                                        trimPromptCache($0, numTokens: trimCount)
                                    }
                                    let mainTrimIsAligned =
                                        mainTrimmed == trimCount
                                        && kvCache.allSatisfy {
                                            $0.offset == commonPrefixCount
                                        }
                                    let draftTrimIsAligned: Bool
                                    if let draftKVCache {
                                        draftTrimIsAligned =
                                            draftTrimmed == trimCount
                                            && draftKVCache.allSatisfy {
                                                $0.offset == commonPrefixCount
                                            }
                                    } else {
                                        draftTrimIsAligned = true
                                    }

                                    if mainTrimIsAligned && draftTrimIsAligned {
                                        input = LMInput(
                                            tokens: MLXArray(
                                                Array(
                                                    fullTokenIds.dropFirst(
                                                        commonPrefixCount))))
                                        reusedMainCacheWithoutDraft =
                                            speculativeDecoding != nil
                                            && !willFallBackBeforeLoadingDraft
                                            && draftKVCache == nil
                                    } else {
                                        kvCache = model.newCache(
                                            parameters: generateParameters)
                                        draftKVCache = nil
                                        lmState = nil
                                    }
                                } else {
                                    // The template changed an already-cached portion of the
                                    // transcript, or this input carries state/media that cannot
                                    // be rewound safely. Rebuild rather than combining a
                                    // mismatched prompt with stale model state.
                                    kvCache = model.newCache(parameters: generateParameters)
                                    draftKVCache = nil
                                    lmState = nil
                                }
                            }

                            currentConversation.cachedTokens = fullTokenIds
                            conversation = currentConversation
                        }

                        guard input.text.tokens.size > 0 else {
                            throw ChatSessionError.emptyPreparedInput
                        }

                        // Select the token iterator based on speculative decoding configuration.
                        let generation: GenerationRun
                        /// Generate with the normal iterator, abandoning any draft
                        /// cache: every caller below reaches this because
                        /// speculation was refused, and a retained draft cache
                        /// would then sit at an offset the main cache never visits.
                        func defaultGenerationWithoutDraft() throws -> GenerationRun {
                            draftKVCache = nil
                            // Seed the iterator with the carried state; read
                            // back the post-prefill state (prefill runs in the
                            // iterator's init, and the rope delta does not
                            // change during decode) so the next turn — or the
                            // next tool restart — anchors correctly.
                            let iterator = try TokenIterator(
                                input: input, model: model, cache: kvCache,
                                state: lmState,
                                parameters: generateParameters)
                            lmState = iterator.state

                            return GenerationRun(
                                MLXLMCommon.generateTaskRecordingTokens(
                                    promptTokenCount: input.text.tokens.size,
                                    modelConfiguration: modelConfiguration,
                                    tokenizer: tokenizer,
                                    iterator: iterator,
                                    tools: tools)
                            )
                        }

                        // Carried model state is fine here: the iterator seeds it
                        // and reads it back, and the anchors the vision models
                        // carry position from the cache offset, which a rejected
                        // proposal rewinds along with the KV rows. Media is not —
                        // the draft model would have to prefill the same images.
                        if let speculativeDecoding,
                            // `input` may have been narrowed to a token-only suffix
                            // above; the prepared input still shows whether this turn
                            // carries media.
                            preparedInput.image == nil,
                            preparedInput.video == nil,
                            preparedInput.audio == nil,
                            draftKVCache != nil || kvCache.allSatisfy({ $0.offset == 0 })
                        {
                            var shouldFallBackBeforeLoadingDraft = false
                            if let memoryEvaluation = speculativeMemoryEvaluation {
                                if !memoryEvaluation.shouldUseSpeculativeDecoding {
                                    if memoryEvaluation.action == .fail {
                                        throw SpeculativeDecodingMemoryError(
                                            evaluation: memoryEvaluation)
                                    }

                                    shouldFallBackBeforeLoadingDraft = true
                                }
                            }

                            if shouldFallBackBeforeLoadingDraft {
                                generation = try defaultGenerationWithoutDraft()
                            } else {
                                let cachedDraftContainer = await loadedDraftModel.read { $0 }
                                let draftContainer: ModelContainer
                                if let cachedDraftContainer {
                                    draftContainer = cachedDraftContainer
                                } else {
                                    draftContainer = try await speculativeDecoding.loadDraftModel()
                                }

                                let draftModel = await draftContainer.perform { context in
                                    SendableBox(context.model)
                                }.consume()
                                let memoryEvaluation = speculativeDecoding.memoryPolicy?.evaluate(
                                    mainModel: model,
                                    draftModel: draftModel)

                                if let memoryEvaluation,
                                    !memoryEvaluation.shouldUseSpeculativeDecoding
                                {
                                    if memoryEvaluation.action == .fail {
                                        throw SpeculativeDecodingMemoryError(
                                            evaluation: memoryEvaluation)
                                    }

                                    generation = try defaultGenerationWithoutDraft()
                                } else {
                                    if cachedDraftContainer == nil {
                                        await loadedDraftModel.update { storedDraftModel in
                                            if storedDraftModel == nil {
                                                storedDraftModel = draftContainer
                                            }
                                        }
                                    }

                                    if reusedMainCacheWithoutDraft {
                                        // The main cache took a suffix-only path, but
                                        // speculation was admitted without a matching
                                        // draft cache. Rebuild both from the full input.
                                        kvCache = model.newCache(
                                            parameters: generateParameters)
                                        draftKVCache = nil
                                        lmState = nil
                                        input = preparedInput
                                    }

                                    // Allocate the draft KV cache once and reuse it across turns,
                                    // exactly like the main model's KV cache.
                                    let draftCache: [KVCache]
                                    if let existingDraftCache = draftKVCache {
                                        draftCache = existingDraftCache
                                    } else {
                                        let newDraftCache = draftModel.newCache(
                                            parameters: generateParameters)
                                        draftKVCache = newDraftCache
                                        cache = .kvcache(
                                            kvCache, draftKVCache: newDraftCache, state: lmState,
                                            conversation: conversation)
                                        draftCache = newDraftCache
                                    }

                                    let iterator = try SpeculativeTokenIterator(
                                        input: input,
                                        mainModel: model,
                                        draftModel: draftModel,
                                        mainCache: kvCache,
                                        draftCache: draftCache,
                                        mainState: lmState,
                                        parameters: generateParameters,
                                        numDraftTokens: speculativeDecoding.numDraftTokens
                                    )
                                    // Carry the main model's post-prefill state,
                                    // as the standard path does. Without this a
                                    // model that positions from a rope delta
                                    // would lose its anchor after a speculative
                                    // turn and mis-position the next one.
                                    lmState = iterator.state

                                    generation = GenerationRun(
                                        MLXLMCommon.generateTaskRecordingTokens(
                                            promptTokenCount: input.text.tokens.size,
                                            modelConfiguration: modelConfiguration,
                                            tokenizer: tokenizer,
                                            iterator: iterator,
                                            tools: tools))
                                }
                            }
                        } else {
                            // Standard path for stateful, media, or unsynchronized cache input.
                            generation = try defaultGenerationWithoutDraft()
                        }

                        var pendingToolCalls: [ToolCall] = []
                        var assistant = AssistantGeneration()

                        for await item in generation.stream {
                            assistant.consume(item)

                            // collect tool calls for dispatch; if no
                            // toolDispatch the caller handles them via
                            // the transform (streamDetails path)
                            if let toolCall = item.toolCall, toolDispatch != nil {
                                pendingToolCalls.append(toolCall)
                            } else if let value = transform(item) {
                                if case .terminated = continuation.yield(value) {
                                    assistant.wasTerminatedByConsumer = true
                                    generation.task.cancel()
                                    break
                                }
                            }
                        }

                        // The generation task is unstructured, so cancellation of
                        // this task (stream onTermination) does not propagate to
                        // it. Without an explicit cancel, awaiting its value
                        // would wait for the FULL generation while holding the
                        // cache lock — deadlocking the session's next call (e.g.
                        // a caller that cancels mid-stream and immediately asks
                        // again). The generate loop checks Task.isCancelled per
                        // token, so this stops it promptly.
                        if Task.isCancelled {
                            assistant.wasTerminatedByConsumer = true
                            generation.task.cancel()
                        }

                        // wait for the task to complete -- this is important in
                        // the case where we broke the loop early as the generation
                        // work may continue (briefly) and use the KVCache
                        let generatedTokens = await generation.task.value

                        if var currentConversation = conversation {
                            let recordedAssistant = currentConversation.record(
                                assistant,
                                generatedTokens: generatedTokens,
                                cacheOffset: kvCache.first?.offset)
                            if !recordedAssistant,
                                let conversationMessageCountBeforePending
                            {
                                // A cancelled or empty generation did not commit an
                                // assistant turn. Roll back this restart's pending input
                                // so a later request cannot produce invalid role sequences
                                // such as user/user on strict chat templates.
                                currentConversation.messages.removeSubrange(
                                    conversationMessageCountBeforePending...)
                            }
                            conversation = currentConversation
                        }

                        // dispatch all tool calls from this generation pass
                        if let toolDispatch, !pendingToolCalls.isEmpty,
                            !Task.isCancelled
                        {
                            if conversation == nil {
                                pendingMessages.append(
                                    .assistant("", toolCalls: pendingToolCalls))
                            }
                            for toolCall in pendingToolCalls {
                                let toolResult = try await toolDispatch(toolCall)
                                pendingMessages.append(.tool(toolResult, id: toolCall.id))
                            }
                            continue restart
                        }
                    }

                    // Store the carried state back alongside the KV cache so
                    // the next turn resumes with correct position anchoring.
                    cache = .kvcache(
                        kvCache, draftKVCache: draftKVCache, state: lmState,
                        conversation: conversation)

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
            case .kvcache(let cache, _, _, _):
                return try await body(cache)
            default:
                return try await body(nil)
            }
        }
    }

    /// Saves the current KV cache to disk.
    ///
    /// Use ``loadPromptCacheSnapshot(url:)`` and an initializer that accepts a
    /// `promptCache` to restore the saved cache and model state in a future session.
    ///
    /// > Important: This saves raw KV state only. It does not save the structured
    /// > transcript retained by `ChatSession`. A restored raw cache therefore uses
    /// > fragment-based continuation and is not suitable for conversation-aware
    /// > structured tool continuations. Persist the messages separately and restore
    /// > them with a history initializer when the transcript is required.
    ///
    /// - Parameter url: the file URL to write the cache to
    /// - Throws: ``ChatSessionError/noCacheAvailable`` if no generation has occurred yet,
    ///   or any error thrown by the underlying file write
    public func saveCache(to url: URL) async throws {
        try await cache.read { cache in
            switch cache {
            case .kvcache(let cache, _, let state, _):
                try savePromptCache(url: url, cache: cache, state: state)
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
    /// The processor produced no tokens for generation.
    case emptyPreparedInput

    public var errorDescription: String? {
        switch self {
        case .noCacheAvailable:
            "No KV cache is available. Call respond() or streamResponse() before saveCache(to:)."
        case .emptyPreparedInput:
            "The chat template produced no uncached tokens for generation."
        }
    }
}
