// Copyright © 2026 Apple Inc.

/// How much a model that steers its own reasoning should think.
///
/// Names are model-specific. Discover them through ``LanguageModel/supportedReasoningEfforts``.
/// The named constants are conveniences, not guarantees of support or equivalent behavior.
/// Native steering does not change chat-template controls or the reasoning-token budget.
///
/// ## See Also
/// - ``GenerateParameters/reasoningEffort``
public struct ReasoningEffort: Hashable, Sendable, CustomStringConvertible {
    public let name: String

    public init(_ name: String) {
        self.name = name
    }

    public var description: String { name }

    /// Native off steering (`none`). This does not disable thinking in the chat template.
    public static let off = ReasoningEffort("none")
    public static let low = ReasoningEffort("low")
    public static let medium = ReasoningEffort("medium")
    public static let high = ReasoningEffort("high")
    public static let xhigh = ReasoningEffort("xhigh")
    public static let max = ReasoningEffort("max")
}

/// The ``ReasoningEffort`` of the request a forward pass belongs to.
///
/// Token iterators set it from ``GenerateParameters/reasoningEffort`` before the first forward
/// pass, and remove it when the parameters have none. It applies to the tokens of that request
/// only, so it is never persisted with a cache.
public let reasoningEffortKey = LMOutput.Key<ReasoningEffort>("reasoning.effort")

extension Optional where Wrapped == LMOutput.State {
    /// Applies the current request's forward-pass options.
    package func applying(
        _ effort: ReasoningEffort?, logitsAccuracy: LogitsAccuracy = .exact
    ) -> LMOutput.State? {
        var state = self ?? LMOutput.State()
        state[reasoningEffortKey] = effort
        state[logitsAccuracyKey] = logitsAccuracy == .exact ? nil : logitsAccuracy
        return state.isEmpty ? nil : state
    }
}
