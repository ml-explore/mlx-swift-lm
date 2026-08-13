// Copyright © 2025 Apple Inc.

import Foundation

// MARK: - ReasoningError

/// Errors raised while resolving or applying a model's reasoning configuration.
public enum ReasoningError: Error, Equatable {
    /// The caller asked to disable reasoning on a model whose reasoning cannot
    /// be turned off (e.g. DeepSeek-R1).
    ///
    /// This is a package-internal error. The `MLXFoundationModels` layer
    /// translates it into the framework's `LanguageModelError.unsupportedCapability`
    /// so app developers see a first-party error type.
    case cannotDisableReasoning
}

// MARK: - ReasoningPromptStrategy

/// How a model's "thinking on / off" preference is expressed to its chat template.
///
/// `MLXLMCommon` deliberately does not depend on `FoundationModels`, so this
/// takes a plain `Bool?` (think on / off / unspecified) rather than a
/// `FoundationModels` reasoning level. The level → `Bool?` mapping lives in the
/// `MLXFoundationModels` layer, mirroring how ``ToolCallFormat`` carries no
/// `FoundationModels`-typed mirror.
public enum ReasoningPromptStrategy: Sendable, Equatable {
    /// Toggleable via a chat-template keyword argument (e.g. Qwen3's
    /// `enable_thinking`). The `key` is the kwarg name; `defaultOn` is the
    /// value used when the caller expresses no preference, matching the
    /// model's own template default.
    case templateFlag(key: String, defaultOn: Bool)

    /// The model always reasons and cannot be turned off (e.g. DeepSeek-R1).
    case alwaysOn

    /// The model has no prompt-level thinking control.
    case none

    /// Maps a "thinking enabled" preference to the chat-template
    /// `additionalContext` it implies.
    ///
    /// - Parameter thinkingEnabled: `true` / `false` to force thinking on / off,
    ///   `nil` when the caller expressed no preference.
    /// - Returns: the `additionalContext` to merge into the rendered prompt, or
    ///   `nil` when no context needs to be injected.
    /// - Throws: ``ReasoningError/cannotDisableReasoning`` when `false` is
    ///   requested on a non-suppressible strategy (``alwaysOn`` or ``none``).
    public func additionalContext(
        forThinkingEnabled thinkingEnabled: Bool?
    ) throws -> [String: any Sendable]? {
        switch self {
        case .templateFlag(let key, let defaultOn):
            return [key: thinkingEnabled ?? defaultOn]
        case .alwaysOn:
            if thinkingEnabled == false {
                throw ReasoningError.cannotDisableReasoning
            }
            return nil
        case .none:
            // .none is non-suppressible: there is no prompt-level knob to
            // turn thinking off. Asking to disable it is identical in
            // outcome to asking .alwaysOn to disable, so it raises the
            // same typed error. The capability gate at MLXLanguageModel
            // routes this to LanguageModelError.unsupportedCapability.
            if thinkingEnabled == false {
                throw ReasoningError.cannotDisableReasoning
            }
            return nil
        }
    }
}

// MARK: - ReasoningConfig

/// Describes a model's reasoning (chain-of-thought) protocol: the delimiters
/// that bracket its thinking in the decoded generation stream, and how thinking
/// is toggled at prompt time.
///
/// Rides on ``ModelConfiguration`` (and therefore ``ResolvedModelConfiguration``)
/// so it reaches generation-time code via `ModelContext.configuration`, exactly
/// like ``ToolCallFormat``.
public struct ReasoningConfig: Sendable, Equatable {

    /// The marker that opens a reasoning span (e.g. `<think>`).
    public var startDelimiter: String

    /// The marker that closes a reasoning span (e.g. `</think>`).
    public var endDelimiter: String

    /// How a thinking on / off preference is expressed to the chat template.
    public var promptStrategy: ReasoningPromptStrategy

    /// Markers that implicitly leave reasoning without emitting ``endDelimiter``.
    ///
    /// These markers remain part of the generated stream. They are boundaries,
    /// not replacements for the canonical closing delimiter. For example,
    /// Qwen3.5 may begin `<tool_call>` directly from its thinking block.
    public var implicitEndDelimiters: [String]

    /// How this model safely transitions to its answer when a reasoning budget
    /// is exhausted, or `nil` when hard budget enforcement is unsupported.
    public var budgetTransition: ReasoningBudgetTransition?

    /// Diagnostic only: whether ``startDelimiter`` is a registered special token
    /// for this model's tokenizer. Budget enforcement compiles every boundary
    /// with the active tokenizer and does not rely on this hint.
    public var isSpecialToken: Bool

    public init(
        startDelimiter: String,
        endDelimiter: String,
        promptStrategy: ReasoningPromptStrategy,
        isSpecialToken: Bool = false,
        implicitEndDelimiters: [String] = [],
        budgetTransition: ReasoningBudgetTransition? = nil
    ) {
        self.startDelimiter = startDelimiter
        self.endDelimiter = endDelimiter
        self.promptStrategy = promptStrategy
        self.isSpecialToken = isSpecialToken
        self.implicitEndDelimiters = implicitEndDelimiters
        self.budgetTransition = budgetTransition
    }

    // MARK: - Presets

    /// Generic `<think>`/`</think>` protocol toggled by the `enable_thinking`
    /// chat-template flag (default on).
    ///
    /// This deliberately declares no budget transition. Sharing delimiters and
    /// a template flag does not imply that two model families were trained to
    /// respond to the same forced early-stop sequence.
    public static let thinkTagsWithEnableThinking = ReasoningConfig(
        startDelimiter: "<think>", endDelimiter: "</think>",
        promptStrategy: .templateFlag(key: "enable_thinking", defaultOn: true),
        isSpecialToken: true)

    /// Always-on `<think>`/`</think>` with no prompt-level off switch
    /// (DeepSeek-R1 and its distills). The protocol does not claim a known-safe
    /// hard-budget transition; applications may opt into one explicitly after
    /// validating it for their exact model and tokenizer.
    public static let alwaysOnThinking = ReasoningConfig(
        startDelimiter: "<think>", endDelimiter: "</think>",
        promptStrategy: .alwaysOn)
}
