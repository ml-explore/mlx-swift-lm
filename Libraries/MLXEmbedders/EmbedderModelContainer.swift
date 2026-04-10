// Copyright © 2026 Apple Inc.

import Foundation
import MLXLMCommon

/// Container for embedder models that guarantees single threaded access.
///
/// Wrap models used by e.g. the UI in a ModelContainer. Callers can access
/// the model and/or tokenizer (any values from the ``EmbedderModelContext``):
///
/// ```swift
/// let resultEmbeddings = await modelContainer.perform { context in
///     let tokenizer = context.tokenizer
///     let encoded = inputs.map {
///         tokenizer.encode(text: $0, addSpecialTokens: true)
///     }
///     ...
///     let modelOutput = context.model(
///         padded, positionIds: nil, tokenTypeIds: tokenTypes, attentionMask: mask)
///
///     let result = context.pooling(
///         modelOutput,
///         normalize: true, applyLayerNorm: true
///     )
///     result.eval()
///     return result.map { $0.asArray(Float.self) }
/// }
/// ```
public final class EmbedderModelContainer: Sendable {
    private let context: SerialAccessContainer<EmbedderModelContext>

    public var configuration: ModelConfiguration {
        get async {
            await context.read { $0.configuration }
        }
    }

    public var tokenizer: Tokenizer {
        get async {
            await context.read { $0.tokenizer }
        }
    }

    public var poolingStrategy: Pooling.Strategy {
        get async {
            await context.read { $0.pooling.strategy }
        }
    }

    public init(context: consuming EmbedderModelContext) {
        self.context = .init(context)
    }

    /// Perform an action on the ``EmbedderModelContext``.
    /// Callers _must_ eval any `MLXArray` before returning as `MLXArray` is not `Sendable`.
    ///
    /// - Note: The closure receives `EmbedderModelContext` which is not `Sendable`. This is intentional -
    ///   the closure runs within the actor's isolation, ensuring thread-safe access to the model.
    /// - Note: The `sending` keyword indicates the return value is transferred (not shared) across
    ///   isolation boundaries, allowing non-Sendable types to be safely returned.
    public func perform<R: Sendable>(
        _ action: @Sendable (EmbedderModelContext) async throws -> sending R
    ) async rethrows -> sending R {
        try await context.read {
            try await action($0)
        }
    }

    @available(*, deprecated, message: "use perform(_: (EmbedderModelContext) -> R) instead")
    public func perform<R: Sendable>(
        _ action: @Sendable (EmbeddingModel, Tokenizer, Pooling) async throws -> sending R
    ) async rethrows -> sending R {
        try await context.read {
            try await action($0.model, $0.tokenizer, $0.pooling)
        }
    }

    /// Perform an action on the ``EmbedderModelContext`` with additional (non `Sendable`) context values.
    /// Callers _must_ eval any `MLXArray` before returning as
    /// `MLXArray` is not `Sendable`.
    public func perform<V, R: Sendable>(
        nonSendable values: consuming V,
        _ action: @Sendable (EmbedderModelContext, V) async throws -> R
    ) async rethrows -> sending R {
        let values = SendableBox(values)
        return try await context.read {
            try await action($0, values.consume())
        }
    }

    /// Update the owned `EmbedderModelContext`.
    /// - Parameter action: update action
    public func update(_ action: @Sendable (inout EmbedderModelContext) -> Void) async {
        await context.update {
            action(&$0)
        }
    }

    // MARK: - Thread-safe convenience methods

    /// The resolved local model directory for the loaded container.
    public var modelDirectory: URL {
        get async throws {
            try (await configuration).modelDirectory
        }
    }

    /// The resolved local tokenizer directory for the loaded container.
    public var tokenizerDirectory: URL {
        get async throws {
            try (await configuration).tokenizerDirectory
        }
    }
}
