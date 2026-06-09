// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXNN

/// Container for models that guarantees single threaded access.
///
/// * Important: `ModelContext` is now `Sendable` that can be used directly.
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
/// ## Mutable Models
///
/// In earlier versions, `ModelContainer` stored a plain `Module`.  If you had reference to it,
/// you could mutate it.  The ``update(_:)`` method was meant for this purpose, though in
/// practice that was only used to mutate the `ModelConfiguration`.
///
/// The `ModelContext` stored in the container is no longer a plain `Module` -- in order to
/// be `Sendable`, it is also immutable.  The `update()` path cannot mutate it (though it could
/// replace it).  The ``perform(_:)-((ModelContext)->R)`` calls also can't mutate
/// the model.  For example, in the LoRA example code:
///
/// ```swift
/// modelAdapter = try await modelContainer.perform { context in
///     return try LoRAContainer.from(
///         model: context.model, configuration: LoRAConfiguration(numLayers: loraLayers))
/// }
/// ```
///
/// That mutates the `context.model` as a side effect (since the `Module` is a reference type
/// there was nothing to prevent it).
///
/// This code will no longer compile and has to be done this way:
///
/// ```swift
/// // load a mutable language model
/// let modelContext = try await args.loadTrainable(
///     defaultModel: defaultModel, modelFactory: modelFactory)
///
/// // augment the model with LoRA adaptors
/// modelAdapter = try! LoRAContainer.from(
///     model: modelContext.model, configuration: LoRAConfiguration(numLayers: loraLayers))
/// ```
///
/// ## Implementation Note
///
/// Previously the `ModelContainer` held the `ModelContext` in a `SerialAccessContainer` -- an internal type
/// that provided a lock-like exclusive access for ``perform(_:)-((ModelContext)->R)`` and ``update(_:)``.
/// The `ModelContext` was not `Sendable` and this provided the `@unchecked Sendable` protection needed.
/// In practice, some code like `ChatSession` would use ``perform(_:)-((ModelContext)->R)`` to
/// _borrow_ the model.  The code was carefully constructed to allow thread-safe access to the shared model
/// state (the weights) so that multiple sessions could be run concurrently.  This wouldn't have been safe if
/// another thread was doing LoRA style mutations, of course.
///
/// The new code uses an `NSLock` to guard access to the `ModelContext`.  The context is now `Sendable`
/// and the model itself is immutable.  This now allows concurrent _use_ of the context -- all reads of the struct
/// itself are done under the lock.  Callers to ``update(_:)`` can modify the context (to a lesser extent than before)
/// and this is done with the lock held.
///
/// Ideally, all use cases will move to use `ModelContext` directly, but in the meantime be aware of this
/// change in implementation.
@available(*, deprecated, message: "use ModelContext instead")
public final class ModelContainer: @unchecked (Sendable) {

    private var _context: ModelContext
    private let lock = NSLock()
    private var context: ModelContext {
        get {
            lock.lock()
            defer { lock.unlock() }
            return _context
        }
        set {
            lock.lock()
            defer { lock.unlock() }
            _context = newValue
        }
    }

    public var modelContext: ModelContext { context }

    public var configuration: ModelConfiguration { context.configuration }

    public var model: any LanguageModel & Sendable { context.model }

    public var processor: UserInputProcessor { context.processor }

    public var tokenizer: Tokenizer { context.tokenizer }

    public init(context: ModelContext) {
        self._context = context
    }

    /// Perform an action on the model and/or tokenizer. Callers _must_ eval any `MLXArray` before returning as
    /// `MLXArray` is not `Sendable`.
    @available(*, deprecated, message: "prefer perform(_:) that uses a ModelContext")
    public func perform<R: Sendable>(
        _ action: @Sendable (any LanguageModel, Tokenizer) throws -> sending R
    ) rethrows -> sending R {
        try action(context.model, context.tokenizer)
    }

    /// Perform an action on the model and/or tokenizer with additional context values.
    @available(*, deprecated, message: "prefer perform(values:_:) that uses a ModelContext")
    public func perform<V: Sendable, R: Sendable>(
        values: V, _ action: @Sendable (any LanguageModel, Tokenizer, V) throws -> sending R
    ) rethrows -> sending R {
        try action(context.model, context.tokenizer, values)
    }

    /// Perform an action on the ``ModelContext``. Callers _must_ eval any `MLXArray` before returning as
    /// `MLXArray` is not `Sendable`.
    ///
    /// - Note: The closure receives `ModelContext` which is not `Sendable`. This is intentional -
    ///   the closure runs within the actor's isolation, ensuring thread-safe access to the model.
    /// - Note: The `sending` keyword indicates the return value is transferred (not shared) across
    ///   isolation boundaries, allowing non-Sendable types to be safely returned.
    public func perform<R: Sendable>(
        _ action: @Sendable (ModelContext) async throws -> sending R
    ) async rethrows -> sending R {
        try await action(context)
    }

    /// Perform an action on the ``ModelContext`` with additional context values.
    /// Callers _must_ eval any `MLXArray` before returning as
    /// `MLXArray` is not `Sendable`.
    public func perform<V: Sendable, R: Sendable>(
        values: V, _ action: @Sendable (ModelContext, V) async throws -> R
    ) async rethrows -> sending R {
        try await action(context, values)
    }

    /// Perform an action on the ``ModelContext`` with additional (non `Sendable`) context values.
    /// Callers _must_ eval any `MLXArray` before returning as
    /// `MLXArray` is not `Sendable`.
    public func perform<V, R: Sendable>(
        nonSendable values: consuming V, _ action: @Sendable (ModelContext, V) async throws -> R
    ) async rethrows -> sending R {
        try await action(context, values)
    }

    /// Update the owned `ModelContext`.
    /// - Parameter action: update action
    @available(
        *, deprecated, message: "ModelContext is now Sendable -- hold that and mutate as needed"
    )
    public func update(_ action: @Sendable (inout ModelContext) -> Void) async {
        lock.withLock {
            action(&_context)
        }
    }

    // MARK: - Thread-safe convenience methods

    /// The resolved local model directory for the loaded container.
    public var modelDirectory: URL {
        get throws { try context.configuration.modelDirectory }
    }

    /// The resolved local tokenizer directory for the loaded container.
    public var tokenizerDirectory: URL {
        get throws { try context.configuration.tokenizerDirectory }
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
        let processor = self.processor
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
        try MLXLMCommon.generate(
            input: input,
            parameters: parameters,
            context: context,
            wiredMemoryTicket: wiredMemoryTicket
        )
    }

    /// Decode token IDs to a string.
    ///
    /// - Parameter tokenIds: Array of token IDs
    /// - Returns: Decoded string
    public func decode(tokenIds: [Int]) -> String {
        let tokenizer = self.tokenizer
        return tokenizer.decode(tokenIds: tokenIds)
    }

    @available(*, deprecated, renamed: "decode(tokenIds:)")
    public func decode(tokens: [Int]) -> String {
        decode(tokenIds: tokens)
    }

    /// Encode a string to token IDs.
    ///
    /// - Parameter text: Text to encode
    /// - Returns: Array of token IDs
    public func encode(_ text: String) -> [Int] {
        let tokenizer = self.tokenizer
        return tokenizer.encode(text: text)
    }

    /// Apply chat template to messages and return token IDs.
    ///
    /// - Parameter messages: Array of message dictionaries with "role" and "content" keys
    /// - Returns: Array of token IDs
    @available(*, deprecated, message: "Use applyChatTemplate directly on tokenizer")
    public func applyChatTemplate(messages: [[String: String]]) throws -> [Int] {
        let tokenizer = self.tokenizer
        return try tokenizer.applyChatTemplate(messages: messages)
    }
}

/// For internal implementation we declare a non-deprecated typealias that can be used e.g. for type parameters
public typealias ModelContainerConstraint = ModelContainer
