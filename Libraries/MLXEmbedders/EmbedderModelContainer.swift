// Copyright © 2026 Apple Inc.

import Foundation
import MLXLMCommon

/// Container for embedder models that guarantees single threaded access.
///
/// * Important: `EmbedderModelContext` is now `Sendable` and can be used directly.
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
///
/// ## Implementation Note
///
/// Previously the `EmbedderModelContainer` held the `EmbedderModelContext` in a `SerialAccessContainer` -- an internal type
/// that provided a lock-like exclusive access for ``perform(_:)-((EmbedderModelContext)->R)`` and ``update(_:)``.
/// The `EmbedderModelContext` was not `Sendable` and this provided the `@unchecked Sendable` protection needed.
/// In practice, some code would use ``perform(_:)-((EmbedderModelContext)->R)`` to
/// _borrow_ the model.  This wouldn't have been safe if
/// another thread was mutating through the reference, of course.
///
/// The new code uses an `NSLock` to guard access to the `EmbedderModelContext`.  The context is now `Sendable`
/// and the model itself is immutable.  This now allows concurrent _use_ of the context -- all reads of the struct
/// itself are done under the lock.  Callers to ``update(_:)`` can modify the context (to a lesser extent than before)
/// and this is done with the lock held.
///
/// Ideally, all use cases will move to use `EmbedderModelContext` directly, but in the meantime be aware of this
/// change in implementation.
@available(*, deprecated, message: "use EmbedderModelContext instead")
public final class EmbedderModelContainer: @unchecked (Sendable) {
    private var _context: EmbedderModelContext
    private let lock = NSLock()
    private var context: EmbedderModelContext {
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

    public var configuration: ModelConfiguration {
        context.configuration
    }

    public var tokenizer: Tokenizer {
        context.tokenizer
    }

    public var poolingStrategy: Pooling.Strategy {
        context.pooling.strategy
    }

    public init(context: consuming EmbedderModelContext) {
        self._context = context
    }

    /// Perform an action on the ``EmbedderModelContext``.
    /// Callers _must_ eval any `MLXArray` before returning as `MLXArray` is not `Sendable`.
    ///
    /// - Note: The `sending` keyword indicates the return value is transferred (not shared) across
    ///   isolation boundaries, allowing non-Sendable types to be safely returned.
    public func perform<R: Sendable>(
        _ action: @Sendable (EmbedderModelContext) async throws -> sending R
    ) async rethrows -> sending R {
        try await action(context)
    }

    /// Perform an action on the ``EmbedderModelContext``.
    ///
    /// This is the synchronous form of ``perform(_:)`` and has
    /// fewer restrictions.
    public func perform<R>(
        _ action: @Sendable (EmbedderModelContext) throws -> R
    ) rethrows -> R {
        try action(context)
    }

    @available(*, deprecated, message: "use perform(_: (EmbedderModelContext) -> R) instead")
    public func perform<R: Sendable>(
        _ action: @Sendable (EmbeddingModel, Tokenizer, Pooling) async throws -> sending R
    ) async rethrows -> sending R {
        try await action(
            context.model, context.tokenizer, context.pooling
        )
    }

    /// Perform an action on the ``EmbedderModelContext`` with additional (non `Sendable`) context values.
    /// Callers _must_ eval any `MLXArray` before returning as
    /// `MLXArray` is not `Sendable`.
    public func perform<V, R: Sendable>(
        nonSendable values: consuming V,
        _ action: @Sendable (EmbedderModelContext, V) async throws -> R
    ) async rethrows -> sending R {
        try await action(context, values)
    }

    /// Update the owned `EmbedderModelContext`.
    /// - Parameter action: update action
    @available(
        *, deprecated,
        message: "mutate EmbedderModelContext before passing to EmbedderModelContainer"
    )
    public func update(_ action: @Sendable (inout EmbedderModelContext) -> Void) async {
        lock.withLock {
            action(&_context)
        }
    }

    // MARK: - Thread-safe convenience methods

    /// The resolved local model directory for the loaded container.
    public var modelDirectory: URL {
        get throws {
            try configuration.modelDirectory
        }
    }

    /// The resolved local tokenizer directory for the loaded container.
    public var tokenizerDirectory: URL {
        get throws {
            try configuration.tokenizerDirectory
        }
    }
}
