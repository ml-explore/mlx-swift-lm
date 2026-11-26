// Copyright © 2026 Apple Inc.

/// How much a model that steers its own reasoning should think.
///
/// The names follow the `reasoning_effort` of OpenAI-compatible servers. A model that steers its
/// reasoning maps each name to its own setting; other models ignore it. The chat template still
/// controls whether the model thinks: set ``off`` when the template's thinking is disabled, so a
/// model steered for that case gets its setting.
///
/// ## See Also
/// - ``GenerateParameters/reasoningEffort``
public struct ReasoningEffort: Hashable, Sendable, CustomStringConvertible {
    public let name: String

    public init(_ name: String) {
        self.name = name
    }

    public var description: String { name }

    /// Thinking off, `reasoning_effort` `none`.
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
    /// This state with ``reasoningEffortKey`` set to `effort`, or removed when it is `nil`.
    package func applying(_ effort: ReasoningEffort?) -> LMOutput.State? {
        var state = self ?? LMOutput.State()
        state[reasoningEffortKey] = effort
        return state.isEmpty ? nil : state
    }
}
