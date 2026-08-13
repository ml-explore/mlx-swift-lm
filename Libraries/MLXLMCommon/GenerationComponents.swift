// Copyright © 2025 Apple Inc.

import Foundation
import MLX

/// Optional behavioral components that augment ``GenerateParameters`` for a generation.
///
/// ``GenerateParameters`` is a plain, `Sendable` value describing *what* to
/// generate (sampling temperature, penalties, token limits, …).
/// `GenerationComponents` carries the *behavioral* hooks that shape generation
/// but are not themselves simple values -- for example a custom
/// ``LogitProcessor`` or parameter validation. Keeping these out of
/// ``GenerateParameters`` lets that
/// type stay a pure value while still letting callers inject custom behavior
/// through the same parameters-driven APIs (``TokenIterator``, ``ChatSession``,
/// the speculative iterators and the `generate(...)` free functions).
///
/// Every field is optional and defaulted, so an empty `GenerationComponents()`
/// reproduces the default behavior and can be threaded through any entry point
/// without changing existing results.
///
/// ```swift
/// let components = GenerationComponents().appendingLogitProcessor {
///     GrammarProcessor(grammar: grammar)
/// }
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

    /// Optional validation for constraints that depend on generation parameters.
    ///
    /// Validation runs before a token iterator performs prefill, so an invalid
    /// configuration fails without spending model work or mutating a cache.
    public var parameterValidator: (@Sendable (GenerateParameters) throws -> Void)?

    public init(
        logitProcessorFactory: (@Sendable () -> any LogitProcessor)? = nil,
        parameterValidator: (@Sendable (GenerateParameters) throws -> Void)? = nil
    ) {
        self.logitProcessorFactory = logitProcessorFactory
        self.parameterValidator = parameterValidator
    }

    /// Return a copy with another custom ``LogitProcessor`` appended.
    ///
    /// Processors are applied in insertion order. The returned component keeps
    /// factories—not processor instances—so every generation receives a fresh
    /// stateful chain. This is the generic composition primitive used by
    /// higher-level features such as
    /// ``applyingThinkingBudget(_:reasoning:tokenizer:diagnosticHandler:)``.
    public func appendingLogitProcessor(
        _ factory: @escaping @Sendable () -> any LogitProcessor
    ) -> Self {
        let existingFactory = logitProcessorFactory
        return Self(
            logitProcessorFactory: {
                guard let existing = existingFactory?() else {
                    return factory()
                }
                return ChainedLogitProcessor(processors: [existing, factory()])
            },
            parameterValidator: parameterValidator)
    }

    /// Return a copy with an additional generation-parameter validator.
    ///
    /// Validators run in insertion order. This composes independently from the
    /// logit-processor chain so behavioral components can reject incompatible
    /// token limits before generation starts.
    public func appendingParameterValidator(
        _ validator: @escaping @Sendable (GenerateParameters) throws -> Void
    ) -> Self {
        let existingValidator = parameterValidator
        return Self(
            logitProcessorFactory: logitProcessorFactory,
            parameterValidator: { parameters in
                try existingValidator?(parameters)
                try validator(parameters)
            })
    }

    /// Validate generation parameters for every configured component.
    public func validate(parameters: GenerateParameters) throws {
        try parameterValidator?(parameters)
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
