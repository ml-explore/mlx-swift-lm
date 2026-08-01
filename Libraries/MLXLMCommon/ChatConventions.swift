// Copyright © 2025 Apple Inc.

import Foundation

// MARK: - ChatConventionsProviding

/// A model's chat conventions: how it encodes tool calls and how it reasons.
///
/// This knowledge lives with the model definition rather than in centralized
/// `model_type` string tables (``ToolCallFormat/infer(from:configData:)`` and
/// ``ReasoningConfig/infer(from:modelId:configData:)``). ``LanguageModel``
/// conforms with `nil` defaults, so the model factories read `model.toolCallFormat`
/// / `model.reasoningConfig` directly and fall back to the inference chains when a
/// model declares nothing. A model opts in by overriding either property in an
/// extension.
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
