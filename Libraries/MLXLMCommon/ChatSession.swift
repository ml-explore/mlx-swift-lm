// Copyright © 2025 Apple Inc.

import CoreGraphics
import Foundation
import MLX

/// Configuration for speculative decoding in a `ChatSession`.
///
/// Speculative decoding uses a small draft model to propose candidate tokens
/// that the main model then verifies in a single forward pass, providing a
/// ~2–3× generation speedup with no quality degradation.
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
public struct SpeculativeDecodingConfig: Sendable {

    /// The lightweight model used to propose candidate tokens.
    public let draftModel: ModelContainer

    /// Number of tokens proposed by the draft model per verification cycle.
    /// The default value of 5 offers a good balance between speed and accuracy.
    public let numDraftTokens: Int

    public init(draftModel: ModelContainer, numDraftTokens: Int = 5) {
        self.draftModel = draftModel
        self.numDraftTokens = numDraftTokens
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
public final class ChatSession {

    enum Cache {
        case empty
        case kvcache([KVCache], draftKVCache: [KVCache]?)
        case history([Chat.Message])
    }

    private let model: ModelContainer
    public var instructions: String?
    private let cache: SerialAccessContainer<Cache>
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
    /// > re-tokenized on each call to ``respond(to:role:images:videos:)`` without matching
    /// > KV state, producing incoherent output.
    ///
    /// - Parameters:
    ///   - model: the ``ModelContainer``
    ///   - instructions: optional system instructions for the session — leave `nil` if the
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
        self.cache = .init(.kvcache(cache, draftKVCache: nil))
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
    /// > re-tokenized on each call to ``respond(to:role:images:videos:)`` without matching
    /// > KV state, producing incoherent output.
    ///
    /// - Parameters:
    ///   - model: the ``ModelContext``
    ///   - instructions: optional system instructions for the session — leave `nil` if the
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
        self.cache = .init(.kvcache(cache, draftKVCache: nil))
        self.processing = processing
        self.generateParameters = generateParameters
        self.tools = tools
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
    /// - Returns: the model's response
    public func respond(
        to prompt: String,
        role: Chat.Message.Role = .user,
        images: consuming [UserInput.Image],
        videos: consuming [UserInput.Video]
    ) async throws -> String {
        var output = ""
        for try await chunk in streamResponse(
            to: prompt, role: role, images: images, videos: videos
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
    /// - Returns: the model's response
    public func respond(
        to prompt: String,
        role: Chat.Message.Role = .user,
        image: UserInput.Image? = nil,
        video: UserInput.Video? = nil
    ) async throws -> String {
        try await respond(
            to: prompt,
            role: role,
            images: image.map { [$0] } ?? [],
            videos: video.map { [$0] } ?? []
        )
    }

    /// Produces a streaming response to a prompt as Strings.
    ///
    /// - Parameters:
    ///   - prompt: the user prompt
    ///   - role: the message role (defaults to `.user`)
    ///   - images: list of images (for use with VLMs)
    ///   - videos: list of videos (for use with VLMs)
    /// - Returns: a stream of string chunks from the model
    public func streamResponse(
        to prompt: String,
        role: Chat.Message.Role = .user,
        images: consuming [UserInput.Image],
        videos: consuming [UserInput.Video]
    ) -> AsyncThrowingStream<String, Error> {
        streamMap(to: prompt, role: role, images: images, videos: videos) {
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
    /// - Returns: a stream of `Generation` from the model
    public func streamDetails(
        to prompt: String,
        role: Chat.Message.Role = .user,
        images: consuming [UserInput.Image],
        videos: consuming [UserInput.Video]
    ) -> AsyncThrowingStream<Generation, Error> {
        streamMap(to: prompt, role: role, images: images, videos: videos) {
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
    /// - Returns: a stream of transformed values from the model
    private func streamMap<R: Sendable>(
        to prompt: String,
        role: Chat.Message.Role,
        images: consuming [UserInput.Image],
        videos: consuming [UserInput.Video],
        transform: @Sendable @escaping (Generation) -> R?
    ) -> AsyncThrowingStream<R, Error> {
        let (stream, continuation) = AsyncThrowingStream<R, Error>.makeStream()

        // images and videos are not Sendable (MLXArray) but they are consumed
        // and are only being sent to the inner async
        let message = SendableBox<Chat.Message>(
            .init(role: role, content: prompt, images: images, videos: videos)
        )

        let task = Task {
            [
                model,
                instructions, processing, tools, toolDispatch,
                additionalContext, cache, generateParameters, speculativeDecoding
            ] in
            do {
                try await cache.update { cache in

                    // these are all Sendable
                    let processor = await model.processor
                    let tokenizer = await model.tokenizer
                    let modelConfiguration = await model.configuration

                    var messages: [Chat.Message] = []
                    if let instructions {
                        messages.append(.system(instructions))
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
                    switch cache {
                    case .empty:
                        kvCache = model.newCache(parameters: generateParameters)
                        cache = .kvcache(kvCache, draftKVCache: nil)

                    case .kvcache(let array, let storedDraftCache):
                        kvCache = array
                        draftKVCache = storedDraftCache

                    case .history(let history):
                        // the KVCache is represented by a chat history
                        kvCache = model.newCache(parameters: generateParameters)
                        cache = .kvcache(kvCache, draftKVCache: nil)
                        messages.append(contentsOf: history)
                    }

                    // prepare the input
                    messages.append(message.consume())

                    // loop can restart on tool calls
                    restart: while !messages.isEmpty {
                        let userInput = UserInput(
                            chat: messages, processing: processing,
                            tools: tools, additionalContext: additionalContext)
                        let input = try await processor.prepare(input: userInput)
                        messages.removeAll()

                        // Select the token iterator based on speculative decoding configuration.
                        let (genStream, genTask): (AsyncStream<Generation>, Task<Void, Never>)

                        if let speculativeDecoding {
                            // Extract the draft model from its container (same pattern as the main model).
                            let draftModel = await speculativeDecoding.draftModel.perform { context in
                                SendableBox(context.model)
                            }.consume()

                            // Allocate the draft KV cache once and reuse it across turns,
                            // exactly like the main model's KV cache.
                            if draftKVCache == nil {
                                draftKVCache = draftModel.newCache(parameters: generateParameters)
                                cache = .kvcache(kvCache, draftKVCache: draftKVCache)
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
                                iterator: iterator
                            )
                        } else {
                            // Standard path with no speculative decoding.
                            let iterator = try TokenIterator(
                                input: input, model: model, cache: kvCache,
                                parameters: generateParameters)

                            (genStream, genTask) = MLXLMCommon.generateTask(
                                promptTokenCount: input.text.tokens.size,
                                modelConfiguration: modelConfiguration,
                                tokenizer: tokenizer,
                                iterator: iterator
                            )
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
    /// - Returns: a stream of string chunks from the model
    public func streamResponse(
        to prompt: String,
        image: UserInput.Image? = nil,
        video: UserInput.Video? = nil
    ) -> AsyncThrowingStream<String, Error> {
        streamResponse(
            to: prompt,
            images: image.map { [$0] } ?? [],
            videos: video.map { [$0] } ?? []
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
            case .kvcache(let cache, _):
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
            case .kvcache(let cache, _):
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

    public var errorDescription: String? {
        "No KV cache is available. Call respond() or streamResponse() before saveCache(to:)."
    }
}
