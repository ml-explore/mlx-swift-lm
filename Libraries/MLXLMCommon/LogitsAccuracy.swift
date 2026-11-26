// Copyright © 2026 Apple Inc.

/// Controls whether generation can use a model's approximate output head.
public enum LogitsAccuracy: Sendable {
    /// Use the full output head.
    case exact
    /// Allow model-specific shortcuts that can change token rankings or probabilities.
    case allowApproximate
}

/// Request-scoped accuracy policy. An absent value means ``LogitsAccuracy/exact``.
public let logitsAccuracyKey = LMOutput.Key<LogitsAccuracy>("logits.accuracy")
