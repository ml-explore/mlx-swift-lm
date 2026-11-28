// Copyright © 2026 Apple Inc.

/// Rules for recovering and validating generated tool calls.
/// Native parser requirements and declared-tool authorization always apply.
public struct ToolCallPolicy: Hashable, Sendable {
    /// Bounded recovery of calls emitted in a different dialect.
    /// Applies to text formats; framed token protocols use their native parser.
    public var recovery: ToolCallRecoveryPolicy

    /// Schema enforcement after unambiguous argument normalization.
    public var validation: ToolCallValidationPolicy

    /// When and how ``ChatSession`` runs accepted calls through
    /// ``ChatSession/toolDispatch``. See <doc:tool-dispatch>.
    public var dispatch: ToolDispatchOptions

    public init(
        recovery: ToolCallRecoveryPolicy = .conservative,
        validation: ToolCallValidationPolicy = .permissive,
        dispatch: ToolDispatchOptions = []
    ) {
        self.recovery = recovery
        self.validation = validation
        self.dispatch = dispatch
    }
}

/// How ``ChatSession`` runs the tool calls of one model step.
///
/// With no options, the calls run one at a time after the step ends. Whatever
/// the options, results enter the conversation after the step, in call order, so
/// the KV cache only grows at its end.
public struct ToolDispatchOptions: OptionSet, Hashable, Sendable {
    public let rawValue: Int

    public init(rawValue: Int) {
        self.rawValue = rawValue
    }

    /// Start each call as soon as the model finishes writing it, while the step
    /// keeps generating. A rejected call later in the step cancels the calls
    /// already running, and the response throws as it does without this option.
    public static let whenWritten = Self(rawValue: 1 << 0)

    /// Run the calls of a step at the same time instead of in call order. Use it
    /// when no call depends on the effects of another.
    public static let concurrent = Self(rawValue: 1 << 1)

    /// Prefill each result as soon as it returns, while later calls still run, so
    /// the next step prefills only the last result. The KV cache must be able to
    /// rewind what this prefills; otherwise the results prefill after the step.
    public static let prefillResult = Self(rawValue: 1 << 2)
}

/// Controls schema enforcement independently of syntax recovery and tool-name
/// authorization. Both modes normalize unambiguous, schema-declared values.
public enum ToolCallValidationPolicy: String, Hashable, Sendable, CaseIterable {
    /// Reject proven schema violations. Unsupported schema assertions remain
    /// unknown. Enable this to check arguments before automatic tool dispatch.
    case strict
    /// Forward parsed arguments after normalization, even if they violate the
    /// schema. This is the default; applications validate arguments themselves.
    /// Native parser requirements and declared-tool authorization still apply.
    case permissive
}
