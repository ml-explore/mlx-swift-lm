// Copyright © 2025 Apple Inc.

import Foundation

// MARK: - ChatConventionsProviding

/// A model's chat conventions: how it encodes tool calls and how it reasons.
///
/// This knowledge lives with the model definition rather than in a centralized
/// `model_type` table. ``LanguageModel`` conforms with `nil` defaults, so the model
/// factories read `model.toolCallFormat` / `model.reasoningConfig` directly. A model
/// opts in by overriding either property in an extension.
///
/// For conventions that are not model-intrinsic (anything keyed on the repo id, such
/// as DeepSeek-R1-Distill), register a ``ChatConventionsResolving`` with
/// ``ChatConventionsRegistry`` instead. Resolvers take precedence over a model's own
/// declaration; see ``ChatConventionsRegistry`` for the full resolution order.
public protocol ChatConventionsProviding {
    /// The tool-call format this model emits, or `nil` for the JSON default.
    var toolCallFormat: ToolCallFormat? { get }

    /// The model's reasoning protocol, or `nil` for non-reasoning models.
    var reasoningConfig: ReasoningConfig? { get }
}

extension ChatConventionsProviding {
    public var toolCallFormat: ToolCallFormat? { nil }
    public var reasoningConfig: ReasoningConfig? { nil }
}
