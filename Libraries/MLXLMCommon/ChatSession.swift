// Copyright © 2025 Apple Inc.

import Foundation
import MLX

#if canImport(CoreGraphics)
import CoreGraphics
#endif

/// Validation failures for speculative-decoding configuration.
public enum SpeculativeDecodingConfigurationError: Error, Sendable, Equatable, LocalizedError {
    case invalidMTPBlockSize(Int)

    public var errorDescription: String? {
        switch self {
        case .invalidMTPBlockSize(let value):
            "MTP verification block size must be at least 2; received \(value)."
        }
    }
}

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

    package enum Strategy: Sendable {
        case draftModel(DraftModelSource)
        case mtp(MTPDrafterContainer, blockSize: Int)
    }

    package let strategy: Strategy

    /// The lightweight model used to propose candidate tokens, when it was provided eagerly.
    ///
    /// Configurations initialized with a loader closure return `nil` because the
    /// draft model is loaded asynchronously by ``ChatSession`` only when speculation
    /// is admitted by the memory policy. MTP configurations also return `nil`
    /// because their drafter uses ``MTPDrafterContainer`` instead.
    public var draftModel: ModelContainer? {
        if case .draftModel(.loaded(let draftModel)) = strategy {
            return draftModel
        }
        return nil
    }

    /// Number of candidate tokens proposed per verification cycle.
    ///
    /// The draft-model initializers default to 5. For MTP configurations this
    /// is `blockSize - 1`, because the verification block also includes the
    /// target model's current token.
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
        self.strategy = .draftModel(.loaded(draftModel))
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
        self.strategy = .draftModel(
            .deferred(bytes: max(0, draftModelBytes), loadDraftModel))
        self.numDraftTokens = numDraftTokens
        self.memoryPolicy = memoryPolicy
    }

    /// Initialize speculative decoding with an MTP drafter.
    ///
    /// - Parameters:
    ///   - mtpDrafter: drafter that proposes `blockSize - 1` tokens per verification cycle
    ///   - blockSize: total number of tokens in each verification block
    /// - Throws: ``SpeculativeDecodingConfigurationError/invalidMTPBlockSize(_:)``
    ///   when `blockSize` is smaller than 2.
    public init(
        mtpDrafter: MTPDrafterContainer,
        blockSize: Int = 2
    ) throws {
        guard blockSize >= 2 else {
            throw SpeculativeDecodingConfigurationError.invalidMTPBlockSize(blockSize)
        }

        self.strategy = .mtp(mtpDrafter, blockSize: blockSize)
        self.numDraftTokens = blockSize - 1
        self.memoryPolicy = nil
    }

    package var estimatedDraftModelBytes: Int? {
        guard case .draftModel(.deferred(let bytes, _)) = strategy else {
            return nil
        }
        return bytes
    }

    package func loadDraftModel() async throws -> ModelContainer {
        guard case .draftModel(let source) = strategy else {
            preconditionFailure("MTP speculation does not use a draft ModelContainer")
        }

        switch source {
        case .loaded(let draftModel):
            return draftModel
        case .deferred(_, let load):
            return try await load()
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
        /// matches the model-cache progress. An empty value paired with
        /// nonzero progress is an invalidated ledger and forces a rebuild.
        var cachedTokens: [Int]

        /// Tokens returned by the previous generation but not yet represented
        /// by `cachedTokens`. This is normally empty; a speculative round can
        /// leave its final verifier sample here.
        var uncommittedTokens: [Int] = []

        @discardableResult
        mutating func record(
            _ assistant: AssistantGeneration,
            generatedTokens: [Int],
            processedTokenCount: Int
        ) -> Bool {
            guard assistant.shouldRecord else {
                // A cancelled or semantically empty generation has no
                // assistant turn to replay. Its cache may nevertheless
                // contain generated or lookahead tokens, so invalidate
                // the ledger and rebuild from the retained messages.
                cachedTokens.removeAll()
                uncommittedTokens.removeAll()
                return false
            }

            let cachedTokenCount = cachedTokens.count
            if processedTokenCount >= cachedTokenCount,
                processedTokenCount - cachedTokenCount <= generatedTokens.count
            {
                let committedGeneratedTokenCount = processedTokenCount - cachedTokenCount
                cachedTokens.append(
                    contentsOf: generatedTokens.prefix(committedGeneratedTokenCount))
                uncommittedTokens = Array(
                    generatedTokens.dropFirst(committedGeneratedTokenCount))
            } else {
                // The iterator/cache relationship could not be proven
                // (for example, a speculative iterator retained
                // un-emitted lookahead).
                cachedTokens.removeAll()
                uncommittedTokens.removeAll()
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
        let task: Task<SendableBox<RecordedGenerationResult>, Never>

        init(
            _ pair: (
                AsyncStream<Generation>, Task<SendableBox<RecordedGenerationResult>, Never>
            )
        ) {
            (stream, task) = pair
        }
    }

    private struct AssistantGeneration {
        var content = ""
        var toolCalls: [ToolCall] = []
        var rejectedToolCalls: [RejectedToolCall] = []
        var stopReason: GenerateStopReason?
        var wasTerminatedByConsumer = false

        var shouldRecord: Bool {
            (!content.isEmpty || !toolCalls.isEmpty)
                && rejectedToolCalls.isEmpty
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
            if let rejection = item.rejectedToolCall {
                rejectedToolCalls.append(rejection)
            }
            if let info = item.info {
                stopReason = info.stopReason
            }
        }
    }

    /// All mutable continuation state for one realized model cache.
    ///
    /// `KVCacheStorage` provides shared ownership of the cache array because
    /// dynamic compression replaces array elements. Conversation state lives
    /// at the same boundary so cache progress and its exact token ledger cannot
    /// be accidentally separated when the session is carried across turns.
    private struct RealizedCache {
        let main: KVCacheStorage
        var draft: KVCacheStorage?
        var state: LMOutput.State?
        var mtpDrafterContinuation: MTPDrafterContinuation?
        var conversation: Conversation?

        init(
            cache: consuming [KVCache],
            draft: consuming [KVCache]? = nil,
            state: LMOutput.State? = nil,
            mtpDrafterContinuation: MTPDrafterContinuation? = nil,
            conversation: Conversation? = nil,
            plan: KVCachePlan
        ) {
            self.main = KVCacheStorage(cache, plan: plan)
            self.draft = draft.map { KVCacheStorage($0, plan: plan) }
            self.state = state
            self.mtpDrafterContinuation = mtpDrafterContinuation
            self.conversation = conversation
        }

        init(
            main: KVCacheStorage,
            draft: KVCacheStorage? = nil,
            state: LMOutput.State? = nil,
            mtpDrafterContinuation: MTPDrafterContinuation? = nil,
            conversation: Conversation? = nil
        ) {
            self.main = main
            self.draft = draft
            self.state = state
            self.mtpDrafterContinuation = mtpDrafterContinuation
            self.conversation = conversation
        }

        func requirePlan(_ requested: KVCachePlan) throws {
            guard main.plan == requested,
                draft.map({ $0.plan == requested }) ?? true
            else {
                throw ChatSessionError.kvCacheConfigurationChanged(
                    previous: main.plan.configuration,
                    requested: requested.configuration)
            }
        }
    }

    private enum Cache {
        /// `state` is the per-call model state (e.g. M-RoPE rope deltas)
        /// from the last prefill against this cache. It must survive across
        /// turns: without it, a model that anchors positions on carried
        /// state re-derives them from a cold start on the next turn.
        case empty
        case kvcache(RealizedCache)
        case history([Chat.Message])
    }

    private let model: ModelContainer
    public var instructions: String?
    private let cache: SerialAccessContainer<Cache>
    private let loadedDraftModel: SerialAccessContainer<ModelContainer?>
    public var processing: UserInput.Processing
    public var generateParameters: GenerateParameters

    /// Optional behavioral components (e.g. a custom ``LogitProcessor``) applied
    /// on each generation. The `logitProcessorFactory` is invoked fresh for
    /// every ``respond(to:role:images:videos:audios:)`` so stateful processors
    /// do not leak state across turns.
    public var components: GenerationComponents

    public var additionalContext: [String: any Sendable]?
    public var tools: [ToolSpec]?
    public var toolDispatch: (@Sendable (ToolCall) async throws -> String)?

    /// Speculative decoding configuration, nil if disabled.
    public let speculativeDecoding: SpeculativeDecodingConfig?

    /// Unified KV / recurrent-cache status for this session.
    ///
    /// If the session owns a realized cache, this reports its actual request,
    /// topology, strategy state, and progress. Otherwise it reports the cache
    /// planned by the current ``generateParameters``.
    public func cacheStatus() async throws -> KVCacheStatus {
        let parameters = generateParameters
        if let realized = await cache.read({ cache -> KVCacheStatus? in
            guard case .kvcache(let stored) = cache else { return nil }
            return KVCacheStatus(
                cache: stored.main.cache,
                plan: stored.main.plan,
                phase: .realized,
                processedTokenCount: stored.main.processedTokenCount)
        }) {
            return realized
        }
        return try await model.cacheStatus(parameters: parameters)
    }

    /// Initialize the `ChatSession`.
    ///
    /// - Parameters:
    ///   - model: the ``ModelContainer``
    ///   - instructions: optional system instructions for the session
    ///   - speculativeDecoding: optional speculative decoding configuration for faster generation
    ///   - generateParameters: parameters that control generation
    ///   - components: optional behavioral components, e.g. a custom ``LogitProcessor``
    ///   - processing: media processing configuration for images/videos
    ///   - tools: optional tool specifications
    ///   - toolDispatch: optional tool dispatch -- required for toolcalls if streaming strings rather than details
    ///   - additionalContext: optional model-specific context
    public init(
        _ model: ModelContainer,
        instructions: String? = nil,
        speculativeDecoding: SpeculativeDecodingConfig? = nil,
        generateParameters: GenerateParameters = .init(),
        components: GenerationComponents = .init(),
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
        self.components = components
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
    ///   - components: optional behavioral components, e.g. a custom ``LogitProcessor``
    ///   - processing: media processing configuration for images/videos
    ///   - tools: optional tool specifications
    ///   - toolDispatch: optional tool dispatch -- required for toolcalls if streaming strings rather than details
    ///   - additionalContext: optional model-specific context
    public init(
        _ model: ModelContext,
        instructions: String? = nil,
        speculativeDecoding: SpeculativeDecodingConfig? = nil,
        generateParameters: GenerateParameters = .init(),
        components: GenerationComponents = .init(),
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
        self.components = components
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
    ///   - components: optional behavioral components, e.g. a custom ``LogitProcessor``
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
        components: GenerationComponents = .init(),
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
        self.components = components
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
    ///   - components: optional behavioral components, e.g. a custom ``LogitProcessor``
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
        components: GenerationComponents = .init(),
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
        self.components = components
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
    ///   - components: optional behavioral components, e.g. a custom ``LogitProcessor``
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
        components: GenerationComponents = .init(),
        processing: UserInput.Processing = .init(resize: CGSize(width: 512, height: 512)),
        additionalContext: [String: any Sendable]? = nil,
        tools: [ToolSpec]? = nil,
        toolDispatch: (@Sendable (ToolCall) async throws -> String)? = nil
    ) {
        self.model = model
        self.instructions = instructions
        self.cache = .init(
            .kvcache(
                .init(
                    cache: cache,
                    state: state,
                    plan: (try? generateParameters.kvCachePlan()) ?? .disabled)))
        self.loadedDraftModel = .init(speculativeDecoding?.draftModel)
        self.processing = processing
        self.generateParameters = generateParameters
        self.components = components
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
    ///   - components: optional behavioral components, e.g. a custom ``LogitProcessor``
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
        components: GenerationComponents = .init(),
        processing: UserInput.Processing = .init(resize: CGSize(width: 512, height: 512)),
        additionalContext: [String: any Sendable]? = nil,
        tools: [ToolSpec]? = nil,
        toolDispatch: (@Sendable (ToolCall) async throws -> String)? = nil
    ) {
        self.model = ModelContainer(context: model)
        self.instructions = instructions
        self.cache = .init(
            .kvcache(
                .init(
                    cache: cache,
                    state: state,
                    plan: (try? generateParameters.kvCachePlan()) ?? .disabled)))
        self.loadedDraftModel = .init(speculativeDecoding?.draftModel)
        self.processing = processing
        self.generateParameters = generateParameters
        self.components = components
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
    /// > re-rendering earlier messages. Positions stay correct, but the prompt differs — and on
    /// > vision encoders that attend across image boundaries (Qwen2-VL today) so do the new
    /// > image's features, since it is encoded alone rather than beside the cached one.
    /// > Use a history initializer when later turns must re-render the full conversation.
    public convenience init(
        _ model: ModelContainer,
        instructions: String? = nil,
        promptCache: consuming PromptCacheSnapshot,
        speculativeDecoding: SpeculativeDecodingConfig? = nil,
        generateParameters: GenerateParameters = .init(),
        components: GenerationComponents = .init(),
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
            components: components,
            processing: processing,
            additionalContext: additionalContext,
            tools: tools,
            toolDispatch: toolDispatch)
    }

    /// Initialize the `ChatSession` with a prompt cache and the model state that belongs with it.
    ///
    /// See ``init(_:instructions:promptCache:speculativeDecoding:generateParameters:components:processing:additionalContext:tools:toolDispatch:)-(ModelContainer,_,_,_,_,_,_,_,_,_)``
    /// for the continuation behavior a snapshot implies.
    public convenience init(
        _ model: ModelContext,
        instructions: String? = nil,
        promptCache: consuming PromptCacheSnapshot,
        speculativeDecoding: SpeculativeDecodingConfig? = nil,
        generateParameters: GenerateParameters = .init(),
        components: GenerationComponents = .init(),
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
            components: components,
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
    /// - Throws: ``RejectedToolCallError`` if the model emits a rejected tool
    ///   call, or an error produced during generation.
    public func streamResponse(
        to prompt: String,
        role: Chat.Message.Role = .user,
        images: consuming [UserInput.Image] = [],
        videos: consuming [UserInput.Video] = [],
        audios: consuming [UserInput.Audio] = []
    ) -> AsyncThrowingStream<String, Error> {
        streamMap(
            to: prompt, role: role, images: images, videos: videos, audios: audios,
            failOnRejectedToolCall: true
        ) { $0.chunk }
    }

    /// Produces a streaming response after appending a batch of structured chat messages.
    ///
    /// Use this to continue an existing session with non-user roles, such as one
    /// or more tool results, while preserving the session's KV cache.
    ///
    /// - Parameter messages: chat messages to append before generation
    /// - Returns: a stream of string chunks from the model
    /// - Throws: ``RejectedToolCallError`` if the model emits a rejected tool
    ///   call, or an error produced during generation.
    public func streamResponse(
        to messages: consuming [Chat.Message]
    ) -> AsyncThrowingStream<String, Error> {
        streamMap(messages: messages, failOnRejectedToolCall: true) { $0.chunk }
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
    ///
    /// Rejected calls are emitted as ``Generation/rejectedToolCall(_:)``. When
    /// automatic tool dispatch is configured, the stream instead throws
    /// ``RejectedToolCallError`` before dispatching any call from that turn.
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
    ///
    /// Rejected calls are emitted as ``Generation/rejectedToolCall(_:)``. When
    /// automatic tool dispatch is configured, the stream instead throws
    /// ``RejectedToolCallError`` before dispatching any call from that turn.
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
        failOnRejectedToolCall: Bool = false,
        transform: @Sendable @escaping (Generation) -> R?
    ) -> AsyncThrowingStream<R, Error> {
        streamMap(
            messages: [
                .init(role: role, content: prompt, images: images, videos: videos, audios: audios)
            ],
            failOnRejectedToolCall: failOnRejectedToolCall,
            transform: transform
        )
    }

    private func streamMap<R: Sendable>(
        messages: consuming [Chat.Message],
        failOnRejectedToolCall: Bool = false,
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
                additionalContext, cache, loadedDraftModel, generateParameters, components,
                speculativeDecoding
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
                    let kvCachePlan = try generateParameters.kvCachePlan()

                    var kvCache: KVCacheStorage
                    var draftKVCache: KVCacheStorage?
                    // Per-call model state (e.g. M-RoPE rope deltas) carried
                    // across turns alongside the KV cache; updated after each
                    // prefill and stored back at the end of the turn.
                    var lmState: LMOutput.State?
                    var mtpDrafterContinuation: MTPDrafterContinuation?
                    var conversation: Conversation?
                    switch cache {
                    case .empty:
                        kvCache = KVCacheStorage(
                            try model.newCache(parameters: generateParameters), plan: kvCachePlan)
                        conversation = Conversation(messages: [], cachedTokens: [])
                        cache = .kvcache(
                            .init(main: kvCache, conversation: conversation))

                    case .kvcache(let stored):
                        let realizedStatus = KVCacheStatus(
                            cache: stored.main.cache,
                            plan: stored.main.plan,
                            phase: .realized,
                            processedTokenCount: stored.main.processedTokenCount)
                        let desiredStatus = try model.cacheStatus(
                            parameters: generateParameters)

                        // The plan captures the typed/legacy strategy and capacity; the
                        // realized layer kinds additionally catch layout changes the plan
                        // alone cannot see.
                        if var restored = stored.conversation,
                            stored.main.plan != kvCachePlan
                                || realizedStatus.layers.map(\.kind)
                                    != desiredStatus.layers.map(\.kind)
                        {
                            // Cache-affecting generation parameters changed. Because a
                            // structured transcript is retained, rebuild the cache from it
                            // rather than silently continuing with the old cache policy.
                            restored.cachedTokens.removeAll()
                            restored.uncommittedTokens.removeAll()
                            kvCache = KVCacheStorage(
                                try model.newCache(parameters: generateParameters),
                                plan: kvCachePlan)
                            draftKVCache = nil
                            lmState = nil
                            mtpDrafterContinuation = nil
                            conversation = restored
                        } else {
                            // Either the realized policy is unchanged, or this is a
                            // restored raw cache with no transcript to rebuild from.
                            // In the latter case, requirePlan rejects a changed policy.
                            try stored.requirePlan(kvCachePlan)
                            kvCache = stored.main
                            draftKVCache = stored.draft
                            lmState = stored.state
                            mtpDrafterContinuation = stored.mtpDrafterContinuation
                            conversation = stored.conversation
                        }

                    case .history(let history):
                        // the KVCache is represented by a chat history
                        kvCache = KVCacheStorage(
                            try model.newCache(parameters: generateParameters), plan: kvCachePlan)
                        conversation = Conversation(messages: history, cachedTokens: [])
                        cache = .kvcache(
                            .init(main: kvCache, conversation: conversation))
                    }

                    var pendingMessages = inputMessages.consume()

                    // How this model's generated token stream relates to a chat
                    // template re-render is a property of its response protocol,
                    // not of the session.
                    let promptCachePolicy = PromptCacheReusePolicy(
                        protocolRules: modelConfiguration.toolCallFormat?
                            .promptCacheReuseRules(tokenizer: tokenizer) ?? [])

                    // loop can restart on tool calls
                    restart: while !pendingMessages.isEmpty {
                        let isToolResultContinuation =
                            pendingMessages.contains { $0.role == .tool }
                            && conversation?.messages.last?.tool?.calls?.isEmpty == false
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
                        let structuredToolCallCount = modelConfiguration.toolCallFormat?
                            .promptCacheStructuredToolCallCount(in: templateMessages)

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

                        let mtpDrafter: (any MTPDrafterModel)?
                        if let speculativeDecoding,
                            case .mtp(let drafterContainer, _) = speculativeDecoding.strategy
                        {
                            mtpDrafter = await drafterContainer.perform { context in
                                SendableBox(context.model)
                            }.consume()
                        } else {
                            mtpDrafter = nil
                        }
                        let usesMTP = mtpDrafter != nil
                        let speculationIsEligibleForParameters: Bool
                        if let mtpDrafter {
                            speculationIsEligibleForParameters =
                                !mtpDrafter.requiresGreedySampling
                                || generateParameters.temperature == 0
                        } else {
                            speculationIsEligibleForParameters = speculativeDecoding != nil
                        }
                        if let mtpDrafter, speculationIsEligibleForParameters {
                            try mtpDrafter.validateCompatibility(with: model)
                        }

                        let mtpIsStateful = mtpDrafter is any StatefulMTPDrafterModel
                        let mtpSupportsResumption =
                            mtpDrafter is any ResumableMTPDrafterModel
                        let mtpContinuationIsAligned =
                            mtpDrafterContinuation?.targetProcessedTokenCount
                            == kvCache.processedTokenCount
                        let mtpSpeculativeStateIsReady =
                            !mtpIsStateful
                            || (mtpSupportsResumption && mtpContinuationIsAligned)

                        var reusedMainCacheWithoutDraft = false
                        // A raw prompt cache has no transcript from which a stateful MTP drafter
                        // can rebuild private state. Preserve the authoritative target cache and
                        // continue target-only unless an aligned continuation exists.
                        var requiresMainOnlyContinuation =
                            usesMTP && mtpIsStateful && conversation == nil
                            && kvCache.processedTokenCount > 0
                            && !mtpSpeculativeStateIsReady
                        // Prompt tokens this turn does not prefill because the cache
                        // already represents them. Reported to the caller on `.info`.
                        var cachedPromptTokenCount = 0
                        // Read off the prepared input, not `input`: the latter may be narrowed to
                        // a token-only suffix below, which would hide media the model still sees.
                        let carriesPreparedMedia =
                            preparedInput.image != nil || preparedInput.video != nil
                            || preparedInput.audio != nil
                        if var currentConversation = conversation {
                            let promptTokenIds = input.text.tokens.asArray(Int.self)
                            let cachedTokenIds = currentConversation.cachedTokens
                            assert(
                                kvCache.nativeAttentionOffsetsAreAligned,
                                "Main attention cache offsets diverged from model-cache progress")
                            let mainCacheIsAligned =
                                kvCache.processedTokenCount == cachedTokenIds.count
                            let ordinaryDraftCacheIsAligned: Bool
                            if let draftKVCache {
                                assert(
                                    draftKVCache.nativeAttentionOffsetsAreAligned,
                                    "Draft attention cache offsets diverged from model-cache progress"
                                )
                                ordinaryDraftCacheIsAligned =
                                    draftKVCache.processedTokenCount == cachedTokenIds.count
                            } else {
                                // Tentatively reuse the main cache. If speculative
                                // decoding is admitted below, both caches are rebuilt
                                // from `preparedInput`; a fallback can keep this suffix.
                                ordinaryDraftCacheIsAligned = true
                            }

                            let hasSpeculativeState =
                                usesMTP ? mtpSpeculativeStateIsReady : draftKVCache != nil
                            let speculativeStateIsAligned =
                                usesMTP
                                ? mtpSpeculativeStateIsReady : ordinaryDraftCacheIsAligned
                            let speculativeStateIsRewindable =
                                usesMTP ? !mtpIsStateful : ordinaryDraftCacheIsAligned

                            let turn = PromptCacheTurn(
                                promptTokens: promptTokenIds,
                                carriesNewMedia: containsNewMedia,
                                carriesPreparedMedia: carriesPreparedMedia,
                                carriesAttentionMask: input.text.mask != nil,
                                carriesModelState: lmState != nil,
                                isToolResultContinuation: isToolResultContinuation,
                                previousGenerationUncommittedTokens:
                                    currentConversation.uncommittedTokens,
                                structuredToolCallCount: structuredToolCallCount,
                                usesSpeculativeDecoding: speculationIsEligibleForParameters)
                            let cacheState = PromptCacheState(
                                cachedTokens: cachedTokenIds,
                                processedTokenCount: kvCache.processedTokenCount,
                                mainCacheIsAligned: mainCacheIsAligned,
                                hasSpeculativeState: hasSpeculativeState,
                                speculativeStateIsAligned: speculativeStateIsAligned,
                                speculativeStateIsRewindable: speculativeStateIsRewindable,
                                canRebuildSpeculativeStateFromPrompt:
                                    !usesMTP && draftKVCache == nil,
                                allowsMainOnlyFallback: usesMTP,
                                isTrimmable: canTrimPromptCache(kvCache.cache)
                                    && (draftKVCache.map { canTrimPromptCache($0.cache) } ?? true))

                            var decision = promptCachePolicy.decide(
                                turn: turn, cache: cacheState)

                            // Rewinding is the one decision that can fail while being
                            // applied: a cache may trim fewer tokens than requested.
                            // Verify and downgrade to a rebuild before prefilling.
                            if case .trimToCommonPrefix(let commonPrefixLength, let trimCount) =
                                decision
                            {
                                let mainTrimmed = kvCache.trim(trimCount)
                                let draftTrimmed = draftKVCache.map { $0.trim(trimCount) }
                                let mainTrimIsAligned =
                                    mainTrimmed == trimCount
                                    && kvCache.processedTokenCount == commonPrefixLength
                                let draftTrimIsAligned =
                                    draftKVCache.map { draftCache in
                                        draftTrimmed == trimCount
                                            && draftCache.processedTokenCount == commonPrefixLength
                                    } ?? true

                                if !(mainTrimIsAligned && draftTrimIsAligned) {
                                    decision = .rebuild
                                }
                            } else if case .trimToCommonPrefixMainOnly(
                                let commonPrefixLength, let trimCount) = decision
                            {
                                let mainTrimmed = kvCache.trim(trimCount)
                                if mainTrimmed != trimCount
                                    || kvCache.processedTokenCount != commonPrefixLength
                                {
                                    decision = .rebuild
                                }
                            }

                            switch decision {
                            case .prefillAll:
                                break

                            case .appendSuffix(let suffixStart, _):
                                input = LMInput(
                                    tokens: MLXArray(Array(promptTokenIds[suffixStart...])))
                                cachedPromptTokenCount = suffixStart

                            case .appendSuffixToMain(let suffixStart, _):
                                input = LMInput(
                                    tokens: MLXArray(Array(promptTokenIds[suffixStart...])))
                                cachedPromptTokenCount = suffixStart
                                // The speculative state does not represent the
                                // same continuation path. Preserve the authoritative
                                // target cache and use it alone for this turn.
                                draftKVCache = nil
                                mtpDrafterContinuation = nil
                                requiresMainOnlyContinuation = true

                            case .trimToCommonPrefix(let commonPrefixLength, _):
                                input = LMInput(
                                    tokens: MLXArray(
                                        Array(promptTokenIds.dropFirst(commonPrefixLength))))
                                cachedPromptTokenCount = commonPrefixLength

                            case .trimToCommonPrefixMainOnly(let commonPrefixLength, _):
                                input = LMInput(
                                    tokens: MLXArray(
                                        Array(promptTokenIds.dropFirst(commonPrefixLength))))
                                cachedPromptTokenCount = commonPrefixLength
                                draftKVCache = nil
                                mtpDrafterContinuation = nil
                                requiresMainOnlyContinuation = true

                            case .rebuild:
                                kvCache = KVCacheStorage(
                                    try model.newCache(parameters: generateParameters),
                                    plan: kvCachePlan)
                                draftKVCache = nil
                                lmState = nil
                                mtpDrafterContinuation = nil
                            }

                            reusedMainCacheWithoutDraft =
                                !usesMTP && decision.reusesCachedPrefix
                                && speculationIsEligibleForParameters
                                && !willFallBackBeforeLoadingDraft
                                && draftKVCache == nil

                            // Usually the rendered prompt, but a protocol rule may
                            // keep generated tokens the cold render cannot reproduce.
                            switch decision {
                            case .appendSuffix(_, let representedTokens),
                                .appendSuffixToMain(_, let representedTokens):
                                currentConversation.cachedTokens = representedTokens
                            case .prefillAll, .trimToCommonPrefix,
                                .trimToCommonPrefixMainOnly, .rebuild:
                                currentConversation.cachedTokens = promptTokenIds
                            }
                            currentConversation.uncommittedTokens.removeAll()
                            conversation = currentConversation
                        }

                        guard input.text.tokens.size > 0 else {
                            throw ChatSessionError.emptyPreparedInput
                        }

                        // Select the token iterator based on speculative decoding configuration.
                        let generation: GenerationRun
                        func defaultGeneration() throws -> GenerationRun {
                            // Seed the carried state and read back the post-prefill state so the
                            // next turn, or the next tool restart, anchors correctly.
                            let iterator = try TokenIterator(
                                input: input, model: model, cacheStorage: kvCache,
                                state: lmState,
                                parameters: generateParameters, components: components)
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

                        // Carried model state is safe to speculate on: the anchors position from
                        // the cache offset, which a rejected proposal rewinds along with the KV
                        // rows. Media is not — the draft would have to prefill the same images.
                        //
                        // A warm main cache with no draft cache is recoverable only when
                        // `reusedMainCacheWithoutDraft` says both can be rebuilt from the full
                        // input below. Without that ledger — a session restored from a snapshot —
                        // there is nothing to re-prefill the draft from, and handing mismatched
                        // caches to the iterator would throw rather than fall back.
                        let mtpStateIsRecoverable =
                            !mtpIsStateful || kvCache.processedTokenCount == 0
                            || (mtpDrafterContinuation?.targetProcessedTokenCount
                                == kvCache.processedTokenCount)
                        let draftCacheIsRecoverable =
                            usesMTP
                            ? mtpStateIsRecoverable
                            : draftKVCache != nil || kvCache.processedTokenCount == 0
                                || reusedMainCacheWithoutDraft
                        if carriesPreparedMedia || !draftCacheIsRecoverable {
                            // Drop the draft cache as `.appendSuffixToMain` does: retaining it
                            // would leave it at an offset the main cache never visits.
                            draftKVCache = nil
                            mtpDrafterContinuation = nil
                            requiresMainOnlyContinuation = true
                        }

                        if !speculationIsEligibleForParameters || requiresMainOnlyContinuation {
                            generation = try defaultGeneration()
                        } else if let speculativeDecoding,
                            case .mtp(_, let blockSize) = speculativeDecoding.strategy,
                            let drafter = mtpDrafter
                        {
                            let iterator = try MTPSpeculativeTokenIterator(
                                input: input,
                                mainModel: model,
                                drafter: drafter,
                                mainCacheStorage: kvCache,
                                mainState: lmState,
                                mtpContinuation: mtpDrafterContinuation,
                                parameters: generateParameters,
                                blockSize: blockSize,
                                components: components
                            )

                            generation = GenerationRun(
                                MLXLMCommon.generateTaskRecordingTokens(
                                    promptTokenCount: input.text.tokens.size,
                                    modelConfiguration: modelConfiguration,
                                    tokenizer: tokenizer,
                                    iterator: iterator,
                                    tools: tools))
                        } else if let speculativeDecoding {
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
                                generation = try defaultGeneration()
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

                                    generation = try defaultGeneration()
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
                                        kvCache = KVCacheStorage(
                                            try model.newCache(parameters: generateParameters),
                                            plan: kvCachePlan)
                                        draftKVCache = nil
                                        lmState = nil
                                        input = preparedInput
                                        cachedPromptTokenCount = 0
                                    }

                                    // Allocate the draft KV cache once and reuse it across turns,
                                    // exactly like the main model's KV cache.
                                    if draftKVCache == nil {
                                        draftKVCache = KVCacheStorage(
                                            try draftModel.newCache(
                                                parameters: generateParameters),
                                            plan: kvCachePlan)
                                        cache = .kvcache(
                                            .init(
                                                main: kvCache,
                                                draft: draftKVCache,
                                                state: lmState,
                                                mtpDrafterContinuation: mtpDrafterContinuation,
                                                conversation: conversation))
                                    }

                                    let iterator = try SpeculativeTokenIterator(
                                        input: input,
                                        mainModel: model,
                                        draftModel: draftModel,
                                        mainCacheStorage: kvCache,
                                        draftCacheStorage: draftKVCache!,
                                        mainState: lmState,
                                        parameters: generateParameters,
                                        numDraftTokens: speculativeDecoding.numDraftTokens,
                                        components: components
                                    )
                                    // Carry the post-prefill state, as the standard path does,
                                    // so the next turn keeps its anchor.
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
                            generation = try defaultGeneration()
                        }

                        var pendingToolCalls: [ToolCall] = []
                        var assistant = AssistantGeneration()

                        for await item in generation.stream {
                            let item = item.attributingCachedPromptTokens(cachedPromptTokenCount)
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
                        let generationResult = await generation.task.value
                        let completedGeneration = generationResult.consume()
                        let generatedTokens = completedGeneration.generatedTokens
                        lmState = completedGeneration.state
                        mtpDrafterContinuation =
                            completedGeneration.mtpDrafterContinuation

                        if var currentConversation = conversation {
                            let recordedAssistant = currentConversation.record(
                                assistant,
                                generatedTokens: generatedTokens,
                                processedTokenCount: kvCache.processedTokenCount)
                            if !recordedAssistant {
                                // The target ledger is invalidated below, so an MTP witness tied
                                // to that boundary must not escape this generation pass.
                                mtpDrafterContinuation = nil
                                if let conversationMessageCountBeforePending {
                                    // A cancelled or empty generation did not commit an
                                    // assistant turn. Roll back this restart's pending input
                                    // so a later request cannot produce invalid role sequences
                                    // such as user/user on strict chat templates.
                                    currentConversation.messages.removeSubrange(
                                        conversationMessageCountBeforePending...)
                                }
                            }
                            conversation = currentConversation
                        }

                        // Commit the whole continuation aggregate before any tool dispatch,
                        // restart, or error path can leave this generation pass.
                        cache = .kvcache(
                            .init(
                                main: kvCache,
                                draft: draftKVCache,
                                state: lmState,
                                mtpDrafterContinuation: mtpDrafterContinuation,
                                conversation: conversation))

                        if let rejection = assistant.rejectedToolCalls.first,
                            failOnRejectedToolCall || toolDispatch != nil
                        {
                            throw RejectedToolCallError(rejection)
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
                                pendingMessages.append(
                                    .tool(
                                        toolResult, id: toolCall.id,
                                        name: toolCall.function.name))
                            }
                            continue restart
                        }
                    }

                    // Store the carried state back alongside the KV cache so
                    // the next turn resumes with correct position anchoring.
                    cache = .kvcache(
                        .init(
                            main: kvCache,
                            draft: draftKVCache,
                            state: lmState,
                            mtpDrafterContinuation: mtpDrafterContinuation,
                            conversation: conversation))

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

    /// Return the legacy runtime-only view of the configured KV-cache strategy.
    ///
    /// The report is `nil` until a typed or legacy cache configuration exists,
    /// or when the session currently stores history rather than a realized cache.
    @available(*, deprecated, message: "Use cacheStatus() for unified cache diagnostics.")
    public func kvCacheRuntimeReport() async throws -> KVCacheRuntimeReport? {
        let kvCachePlan = try generateParameters.kvCachePlan()
        return try await cache.read { cache in
            guard case .kvcache(let stored) = cache else { return nil }
            try stored.requirePlan(kvCachePlan)
            _ = try stored.main.plan.validated(stored.main.cache)
            return stored.main.plan.report(for: stored.main)
        }
    }

    /// Visit the current cache value, if realized as a `[KVCache]`.
    ///
    /// This method is meant for test support.
    func withCache<R: Sendable>(_ body: @Sendable ([KVCache]?) async throws -> R) async rethrows
        -> R?
    {
        try await cache.read { cache in
            switch cache {
            case .kvcache(let stored):
                return try await body(stored.main.cache)
            default:
                return try await body(nil)
            }
        }
    }

    /// Return model-wide cache progress for test support.
    func cacheProgress() async -> (main: Int, draft: Int?)? {
        await cache.read { cache in
            guard case .kvcache(let stored) = cache else { return nil }
            return (
                main: stored.main.processedTokenCount,
                draft: stored.draft?.processedTokenCount
            )
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
            case .kvcache(let stored):
                try savePromptCache(url: url, cache: stored.main.cache, state: stored.state)
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
    /// The cache was realized under a different KV-cache configuration.
    case kvCacheConfigurationChanged(
        previous: KVCacheConfiguration?,
        requested: KVCacheConfiguration?
    )

    public var errorDescription: String? {
        switch self {
        case .noCacheAvailable:
            "No KV cache is available. Call respond() or streamResponse() before saveCache(to:)."
        case .emptyPreparedInput:
            "The chat template produced no uncached tokens for generation."
        case .kvCacheConfigurationChanged:
            "KV-cache configuration changed after the session cache was realized. Clear the session or respond() to rebuild the cache from its retained transcript."
        }
    }
}
