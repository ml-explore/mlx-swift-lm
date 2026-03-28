// Copyright © 2024 Apple Inc.

import Foundation
import Hub
import MLX
import MLXNN
import Tokenizers

/// Container for models that guarantees single threaded access.
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
public final class ModelContainer: Sendable {
    private let context: SerialAccessContainer<ModelContext>

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

    public init(context: consuming ModelContext) {
        self.context = .init(context)
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
        try await context.read {
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
        try await context.read {
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
        _ action: @Sendable (ModelContext) async throws -> sending R
    ) async rethrows -> sending R {
        try await context.read {
            try await action($0)
        }
    }

    /// Perform an action on the ``ModelContext`` with additional context values.
    /// Callers _must_ eval any `MLXArray` before returning as
    /// `MLXArray` is not `Sendable`.
    public func perform<V: Sendable, R: Sendable>(
        values: V, _ action: @Sendable (ModelContext, V) async throws -> R
    ) async rethrows -> sending R {
        try await context.read {
            try await action($0, values)
        }
    }

    /// Perform an action on the ``ModelContext`` with additional (non `Sendable`) context values.
    /// Callers _must_ eval any `MLXArray` before returning as
    /// `MLXArray` is not `Sendable`.
    public func perform<V, R: Sendable>(
        nonSendable values: consuming V, _ action: @Sendable (ModelContext, V) async throws -> R
    ) async rethrows -> sending R {
        let values = SendableBox(values)
        return try await context.read {
            try await action($0, values.consume())
        }
    }

    /// Update the owned `ModelContext`.
    /// - Parameter action: update action
    public func update(_ action: @Sendable (inout ModelContext) -> Void) async {
        await context.update {
            action(&$0)
        }
    }

    // MARK: - Thread-safe convenience methods

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
    /// - Parameter tokens: Array of token IDs
    /// - Returns: Decoded string
    public func decode(tokens: [Int]) async -> String {
        let tokenizer = await self.tokenizer
        return tokenizer.decode(tokens: tokens)
    }

    /// Encode a string to token IDs.
    ///
    /// - Parameter text: Text to encode
    /// - Returns: Array of token IDs
    public func encode(_ text: String) async -> [Int] {
        let tokenizer = await self.tokenizer
        return tokenizer.encode(text: text)
    }

    /// Apply chat template to messages and return token IDs.
    ///
    /// - Parameter messages: Array of message dictionaries with "role" and "content" keys
    /// - Returns: Array of token IDs
    public func applyChatTemplate(messages: [[String: String]]) async throws -> [Int] {
        let tokenizer = await self.tokenizer
        return try tokenizer.applyChatTemplate(messages: messages)
    }

    // MARK: - Layer Partitioning

    /// Configure GPU/CPU layer partitioning for the loaded model.
    ///
    /// When set, the first `gpuLayers` transformer layers run on GPU
    /// and the rest run on CPU. This enables inference of models larger
    /// than available GPU memory on Apple Silicon with UMA.
    ///
    /// - Parameter gpuLayers: Number of layers to run on GPU, or nil for all-GPU.
    /// - Returns: The actual number of GPU layers set (after clamping), or nil.
    @discardableResult
    public func setGPULayers(_ gpuLayers: Int?) async -> Int? {
        await context.read { ctx in
            // Walk the module tree to find a LayerPartitionable child
            if let partitionable = Self.findPartitionable(in: ctx.model) {
                partitionable.setGPULayers(gpuLayers)
                return partitionable.gpuLayerCount
            }
            return nil
        }
    }

    /// Enable or disable SSD Expert Streaming for MoE models.
    ///
    /// When enabled, the model evaluates intermediate states and clears the MLX cache
    /// layer-by-layer. This allows the OS Page Cache to trivially load active experts
    /// from SSD and discard them immediately, preventing OOM on memory-constrained devices.
    ///
    /// - Parameter stream: Whether expert streaming should be enabled.
    /// - Returns: True if the model supports streaming and it was set, false otherwise.
    @discardableResult
    public func setStreamExperts(_ stream: Bool) async -> Bool {
        await context.read { ctx in
            if let streamable = Self.findStreamable(in: ctx.model) {
                streamable.streamExperts = stream
                return true
            }
            return false
        }
    }

    /// Recursively search for a `LayerPartitionable` in the module tree.
    private static func findPartitionable(in module: Module) -> (any LayerPartitionable)? {
        // Check if the module itself conforms
        if let p = module as? (any LayerPartitionable) {
            return p
        }
        // Check named children — common pattern: model has a `model` property
        // that is the ModelInner
        let mirror = Mirror(reflecting: module)
        for child in mirror.children {
            if let childModule = child.value as? Module {
                if let found = findPartitionable(in: childModule) {
                    return found
                }
            }
        }
        return nil
    }

    /// Recursively search for a `StreamableMoE` in the module tree.
    private static func findStreamable(in module: Module) -> (any StreamableMoE)? {
        // Check if the module itself conforms
        if let p = module as? (any StreamableMoE) {
            return p
        }
        // Check named children
        let mirror = Mirror(reflecting: module)
        for child in mirror.children {
            if let childModule = child.value as? Module {
                if let found = findStreamable(in: childModule) {
                    return found
                }
            }
        }
        return nil
    }
}
