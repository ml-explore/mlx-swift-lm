// Copyright © 2025 Apple Inc.

import Foundation
import MLX

/// Optional behavioral components that augment ``GenerateParameters`` for a generation.
///
/// ``GenerateParameters`` is a plain, `Sendable` value describing *what* to
/// generate (sampling temperature, penalties, token limits, …).
/// `GenerationComponents` carries the *behavioral* hooks that shape generation
/// but are not themselves simple values -- for example a custom
/// ``LogitProcessor``. Keeping these out of ``GenerateParameters`` lets that
/// type stay a pure value while still letting callers inject custom behavior
/// through the same parameters-driven APIs (``TokenIterator``, ``ChatSession``,
/// the speculative iterators and the `generate(...)` free functions).
///
/// Every field is optional and defaulted, so an empty `GenerationComponents()`
/// reproduces the default behavior and can be threaded through any entry point
/// without changing existing results.
///
/// ```swift
/// var components = GenerationComponents()
/// components.logitProcessorFactory = { GrammarProcessor(grammar: grammar) }
/// let session = ChatSession(model, components: components)
/// ```
public struct GenerationComponents: Sendable {

    /// Optional factory producing a custom ``LogitProcessor`` that is composed
    /// after the built-in penalty processor, see ``logitProcessor(parameters:)``.
    ///
    /// This lets custom logit processing -- for example grammar-constrained or
    /// otherwise structured decoding -- ride any parameters-driven API,
    /// including ``ChatSession`` and the parameters-based ``TokenIterator``
    /// initializers.
    ///
    /// This is a factory rather than a stored ``LogitProcessor`` because
    /// processors are stateful (see ``LogitProcessor/didSample(token:)``): it is
    /// invoked once per generation so each generation gets a fresh instance and
    /// state cannot leak across generations. A `@Sendable` closure also keeps
    /// `GenerationComponents` `Sendable`.
    public var logitProcessorFactory: (@Sendable () -> any LogitProcessor)?

    public init(
        logitProcessorFactory: (@Sendable () -> any LogitProcessor)? = nil
    ) {
        self.logitProcessorFactory = logitProcessorFactory
    }

    /// Build the ``LogitProcessor`` for a single generation.
    ///
    /// Composes the built-in penalty processor from
    /// ``GenerateParameters/processor()`` with the custom processor produced by
    /// ``logitProcessorFactory`` (if any). The custom processor runs *after* the
    /// penalty processor, matching Python mlx-lm where user `logits_processors`
    /// append after `make_logits_processors`.
    ///
    /// The factory is invoked here, so a fresh custom processor is produced on
    /// every call and stateful processors do not leak state across generations.
    public func logitProcessor(parameters: GenerateParameters) -> LogitProcessor? {
        switch (parameters.processor(), logitProcessorFactory?()) {
        case (nil, nil):
            return nil
        case (let penalty?, nil):
            return penalty
        case (nil, let custom?):
            return custom
        case (let penalty?, let custom?):
            return ChainedLogitProcessor(processors: [penalty, custom])
        }
    }
}
