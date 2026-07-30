// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXNN

/// Container for models that guarantees single threaded access.
///
/// * Important: `ModelContext` is now `Sendable` that can be used directly.  `ModelContainer`
/// is now deprecated.
///
/// Wrap models used by e.g. the UI in a ModelContainer. Callers can access
/// the model and/or tokenizer (any values from the ``ModelContext``):
///
/// ```swift
/// let messages = [["role": "user", "content": prompt]]
/// let promptTokens = try await modelContainer.perform { context in
///     try context.tokenizer.applyChatTemplate(messages: messages)
/// }
/// ```
///
/// or:
///
/// ```swift
/// let userInput: UserInput
/// let result = await modelContainer.perform { context in
///     let input = try await context.processor.prepare(input: userInput)
///     return generate(
///         input: input, parameters: generateParameters, context: context
///     ) { tokens in
///     ...
///     }
/// }
/// ```
///
/// ## Source Compatibility
///
/// All callers _should_ migrate to using ``ModelContext`` or ``TrainableModelContext`` directly.
///
/// Previously this held and provided a `ModelContext`.  That type still exists but is now `Sendable`
/// and immutable.  The container now holds a `TrainableModelContext`, which is equivalent
/// to the old mutable context.
///
/// Methods, such as ``perform(_:)->_``, call the closure with a `TrainableModelContext`.
/// Typical calls to this will still compile -- the type inference finds the new type and the methods are
/// the same.
///
/// From example code, there are two cases that must be updated.  The first is when your closure calls
/// a function that will mutate the context:
///
/// ```swift
/// // OLD
/// func prepare(_ context: inout ModelContext) { ... }
///
/// // NEW
/// func prepare(_ context: inout TrainableModelContext) { ... }
/// ```
///
/// The second is when the closure calls a method that will do inference (or related) and
/// just needs access to the tokenizer and `LanguageModel`:
///
/// ```swift
/// // OLD
/// func generate(input: LMInput, context: ModelContext) async throws -> ... { ... }
///
/// // NEW
/// func generate(input: LMInput, context: ModelContextProviding) async throws -> ... { ... }
/// ```
///
/// Code that relies on type inference should build without change.
///
/// ## Migration
///
/// `ModelContainer` and methods that create it are all deprecated.  There are two main
/// patterns for migration off of it and onto ``ModelContext`` and ``TrainableModelContext``.
/// For inference-only uses, migrate onto `ModelContext`.
///
/// * Note: Both original examples still work, but they now have deprecation warnings.
///
/// For example, if you had code along these lines:
///
/// ```swift
/// let modelContainer = try await LLMModelFactory.shared.loadContainer(
///     from: self.downloader,
///     using: #huggingFaceTokenizerLoader(),
///     configuration: modelConfiguration)
///
/// let session = ChatSession(
///     modelContainer,
///     instructions: "You are a helpful assistant."
/// )
///
/// let response = try await session.respond(to: "Tell me a story.")
/// print(response)
/// ```
///
/// You could switch it to `ModelContext` by calling `loadModel()` instead
/// of `loadContainer()`:
///
/// ```swift
/// let modelContext = try await LLMModelFactory.shared.load(
///     from: self.downloader,
///     using: #huggingFaceTokenizerLoader(),
///     configuration: modelConfiguration)
///
/// let session = ChatSession(
///     modelContext,
///     instructions: "You are a helpful assistant."
/// )
///
/// let response = try await session.respond(to: "Tell me a story.")
/// print(response)
/// ```
///
/// For cases where the goal is model mutation, e.g. training or LoRA fine-tuning, you will need
/// a ``TrainableModelContext``.  If the previous code looked like this:
///
/// ```swift
/// let modelContainer = try await LLMModelFactory.shared.loadContainer(
///     from: self.downloader,
///     using: #huggingFaceTokenizerLoader(),
///     configuration: modelConfiguration)
///
/// modelAdapter = try await modelContainer.perform { context in
///     return try LoRAContainer.from(
///         model: context.model, configuration: LoRAConfiguration(numLayers: loraLayers))
/// }
/// ```
///
/// The migrated code would call `loadTrainable()` instead:
///
/// ```swift
/// // load a mutable language model
/// let modelContext = try await LLMModelFactory.shared.loadTrainable(
///     from: self.downloader,
///     using: #huggingFaceTokenizerLoader(),
///     configuration: modelConfiguration)
///
/// // augment the model with LoRA adaptors
/// modelAdapter = try! LoRAContainer.from(
///     model: modelContext.model, configuration: LoRAConfiguration(numLayers: loraLayers))
/// ```
///
/// ## Mutation and Thread Safety
///
/// In earlier versions, `ModelContainer` stored a plain `Module`.  If you had reference to it,
/// you could mutate it.  The ``update(_:)`` method was meant for this purpose, though in
/// practice that was only used to mutate the `ModelConfiguration`,
///
/// However, there are cases where callers also mutated through the context's `model` reference:
///
/// ```swift
/// modelAdapter = try await modelContainer.perform { context in
///     return try LoRAContainer.from(
///         model: context.model, configuration: LoRAConfiguration(numLayers: loraLayers))
/// }
/// ```
///
/// This appears to be thread safe -- the `ModelContainer` guarantees exclusive access to the
/// context during calls to ``perform(_:)->_``, but in practice callers would carefully _borrow_
/// the model in order to allow concurrent evaluation.  As long as LoRA (or other training) was not
/// done at the same time as inference, this worked, but was unsafe.  See:
///
/// - ``generate(input:parameters:wiredMemoryTicket:)``
/// - ``ChatSession``
///
/// Now, the container stores a ``TrainableModelContext`` and a private `MaterializedState`.
/// If a caller initializes a ``ChatSession`` with a `ModelContainer`, this will materialize the
/// model and produce a `ModelContext`.  Callers who attempt to access the
/// ``TrainableModelContext`` after this occurs will be met with a `fatalError`.
///
/// * Important: there is a race when initializing `ChatSession` and using `perform()`.
/// This implements a best effort check, but cannot guarantee it doesn't happen.  This race is
/// probably insignificant when compared to callers that escape the `model` in calls to
/// `perform()` (including this type's own ``generate(input:parameters:wiredMemoryTicket:)``).
/// These are done on purpose, but were never safe when combined with mutation
///
/// Please migrate to ``ModelContext`` and ``TrainableModelContext`` directly as
/// soon as possible.
@available(*, deprecated, message: "use ModelContext instead")
public final class ModelContainer: @unchecked (Sendable) {

    private let context: SerialAccessContainer<TrainableModelContext>

    /// Internal state to provide `ChatSession` with synchronous access to the ModelContext
    private enum MaterializedState {
        case editable(TrainableModelContext)
        case materialized(ModelContext)
    }

    private var _state: MaterializedState
    private let lock = NSLock()

    public var configuration: ModelConfiguration {
        get async {
            await context.read { $0.configuration }
        }
    }

    public var processor: UserInputProcessor {
        get async {
            await context.read { $0.processor }
        }
    }

    public var tokenizer: Tokenizer {
        get async {
            await context.read { $0.tokenizer }
        }
    }

    public init(context: TrainableModelContext) {
        self.context = .init(context)
        self._state = .editable(context)
    }

    /// Verify that the model is still mutable (has not been materialized).
    ///
    /// Note: there is a race in calling this and using `context`.  This is
    /// best effort on top of the fact that many callers allowed the `model`
    /// to escape calls to `perform()`.
    ///
    /// Migrate off `ModelContainer` at your earliest convenience.
    private func check() {
        lock.withLock {
            switch _state {
            case .editable:
                break
            case .materialized:
                fatalError("context has been converted to ModelContext (immutable)")
            }
        }
    }

    /// Allow ChatSession to obtain a materialized version of the ModelContext.
    ///
    /// Note: this permanently marks the main context as unusable as it is not longer
    /// mutable.
    func materialize() -> ModelContext {
        lock.withLock {
            switch _state {
            case .editable(let context):
                let materialized = ModelContext.init(context)
                _state = .materialized(materialized)
                return materialized
            case .materialized(let context):
                return context
            }
        }
    }

    /// Perform an action on the model and/or tokenizer. Callers _must_ eval any `MLXArray` before returning as
    /// `MLXArray` is not `Sendable`.
    @available(*, deprecated, message: "prefer perform(_:) that uses a ModelContext")
    public func perform<R: Sendable>(
        _ action: @Sendable (any LanguageModel, Tokenizer) throws -> sending R
    )
        async rethrows
        -> sending R
    {
        check()
        return try await context.read {
            try action($0.model, $0.tokenizer)
        }
    }

    /// Perform an action on the model and/or tokenizer with additional context values.
    /// Callers _must_ eval any `MLXArray` before returning as
    /// `MLXArray` is not `Sendable`.
    @available(*, deprecated, message: "prefer perform(values:_:) that uses a ModelContext")
    public func perform<V: Sendable, R: Sendable>(
        values: V, _ action: @Sendable (any LanguageModel, Tokenizer, V) throws -> sending R
    ) async rethrows -> sending R {
        check()
        return try await context.read {
            try action($0.model, $0.tokenizer, values)
        }
    }

    /// Perform an action on the ``ModelContext``. Callers _must_ eval any `MLXArray` before returning as
    /// `MLXArray` is not `Sendable`.
    ///
    /// - Note: The closure receives `ModelContext` which is not `Sendable`. This is intentional -
    ///   the closure runs within the actor's isolation, ensuring thread-safe access to the model.
    /// - Note: The `sending` keyword indicates the return value is transferred (not shared) across
    ///   isolation boundaries, allowing non-Sendable types to be safely returned.
    public func perform<R: Sendable>(
        _ action: @Sendable (TrainableModelContext) async throws -> sending R
    ) async rethrows -> sending R {
        check()
        return try await context.read {
            try await action($0)
        }
    }

    /// Perform an action on the ``ModelContext`` with additional context values.
    /// Callers _must_ eval any `MLXArray` before returning as
    /// `MLXArray` is not `Sendable`.
    public func perform<V: Sendable, R: Sendable>(
        values: V, _ action: @Sendable (TrainableModelContext, V) async throws -> R
    ) async rethrows -> sending R {
        check()
        return try await context.read {
            try await action($0, values)
        }
    }

    /// Perform an action on the ``ModelContext`` with additional (non `Sendable`) context values.
    /// Callers _must_ eval any `MLXArray` before returning as
    /// `MLXArray` is not `Sendable`.
    public func perform<V, R: Sendable>(
        nonSendable values: consuming V,
        _ action: @Sendable (TrainableModelContext, V) async throws -> R
    ) async rethrows -> sending R {
        check()
        let values = SendableBox(values)
        return try await context.read {
            try await action($0, values.consume())
        }
    }

    /// Update the owned `ModelContext`.
    /// - Parameter action: update action
    public func update(_ action: @Sendable (inout TrainableModelContext) -> Void) async {
        check()
        return await context.update {
            action(&$0)
        }
    }

    // MARK: - Thread-safe convenience methods

    /// The resolved local model directory for the loaded container.
    public var modelDirectory: URL {
        get async throws {
            check()
            return try (await configuration).modelDirectory
        }
    }

    /// The resolved local tokenizer directory for the loaded container.
    public var tokenizerDirectory: URL {
        get async throws {
            check()
            return try (await configuration).tokenizerDirectory
        }
    }

    /// Prepare user input for generation.
    ///
    /// This method safely prepares input within the actor's isolation,
    /// avoiding the need for closure-based `perform` calls.
    ///
    /// - Parameter input: The user input to prepare
    /// - Returns: Prepared language model input (transferred via `sending`)
    /// - Note: The `sending` keyword indicates the return value is transferred (not shared),
    ///   allowing non-Sendable types like `LMInput` to safely cross isolation boundaries.
    public func prepare(input: consuming sending UserInput) async throws -> sending LMInput {
        check()
        let processor = await self.processor
        return try await processor.prepare(input: input)
    }

    /// Generate tokens from prepared input, returning an AsyncStream.
    ///
    /// This method provides a thread-safe way to generate tokens without
    /// needing to use closure-based `perform` calls.
    ///
    /// Example:
    /// ```swift
    /// let input = try await modelContainer.prepare(input: userInput)
    /// let stream = try modelContainer.generate(input: input, parameters: parameters)
    /// for await generation in stream {
    ///     switch generation {
    ///     case .chunk(let text): print(text)
    ///     case .info(let info): print(info.tokensPerSecond)
    ///     case .toolCall(let call): handleToolCall(call)
    ///     }
    /// }
    /// ```
    ///
    /// - Parameters:
    ///   - input: Prepared language model input (transferred via `sending`)
    ///   - parameters: Generation parameters
    ///   - wiredMemoryTicket: Optional wired memory ticket for policy-based coordination
    /// - Returns: An AsyncStream of generation events
    /// - Note: The `sending` parameter indicates the input is transferred (not shared),
    ///   allowing non-Sendable types like `LMInput` to safely cross isolation boundaries.
    public func generate(
        input: consuming sending LMInput,
        parameters: GenerateParameters,
        wiredMemoryTicket: WiredMemoryTicket? = nil
    ) async throws -> AsyncStream<Generation> {
        // handle a model that has been materialized
        let materializedStream: AsyncStream<Generation>? = try lock.withLock {
            switch _state {
            case .editable:
                return nil
            case .materialized(let context):
                return try MLXLMCommon.generate(
                    input: input,
                    parameters: parameters,
                    context: context,
                    wiredMemoryTicket: wiredMemoryTicket
                )
            }
        }

        if let materializedStream {
            return materializedStream
        }

        // else, generate using the non-materialized model
        let input = SendableBox(input)

        // Note: this is only visiting the model exclusively
        // for the pre-fill time.  Beyond that there is no
        // shared mutable state.
        //
        // This means that there may be concurrent access to the
        // model weights themselves (but they are already evaluated).

        return try await context.read { context in
            try MLXLMCommon.generate(
                input: input.consume(),
                parameters: parameters,
                context: context,
                wiredMemoryTicket: wiredMemoryTicket
            )
        }
    }

    /// Decode token IDs to a string.
    ///
    /// - Parameter tokenIds: Array of token IDs
    /// - Returns: Decoded string
    public func decode(tokenIds: [Int]) async -> String {
        check()
        let tokenizer = await self.tokenizer
        return tokenizer.decode(tokenIds: tokenIds)
    }

    @available(*, deprecated, renamed: "decode(tokenIds:)")
    public func decode(tokens: [Int]) async -> String {
        await decode(tokenIds: tokens)
    }

    /// Encode a string to token IDs.
    ///
    /// - Parameter text: Text to encode
    /// - Returns: Array of token IDs
    public func encode(_ text: String) async -> [Int] {
        check()
        let tokenizer = await self.tokenizer
        return tokenizer.encode(text: text)
    }

    /// Apply chat template to messages and return token IDs.
    ///
    /// - Parameter messages: Array of message dictionaries with "role" and "content" keys
    /// - Returns: Array of token IDs
    @available(*, deprecated, message: "Use applyChatTemplate directly on tokenizer")
    public func applyChatTemplate(messages: [[String: String]]) async throws -> [Int] {
        check()
        let tokenizer = await self.tokenizer
        return try tokenizer.applyChatTemplate(messages: messages)
    }
}
