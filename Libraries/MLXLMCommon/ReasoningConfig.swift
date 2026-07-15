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

    /// Diagnostic only: whether ``startDelimiter`` is a registered special token
    /// for this model's tokenizer.
    ///
    /// Not load-bearing in v1 — detection is always string-scan based (the
    /// decoded stream renders the delimiter as literal text whether or not it is
    /// a special token, because `decode(tokenIds:)` defaults
    /// `skipSpecialTokens: false`). Reserved for a future token-ID-stream
    /// optimization.
    public var isSpecialToken: Bool

    public init(
        startDelimiter: String,
        endDelimiter: String,
        promptStrategy: ReasoningPromptStrategy,
        isSpecialToken: Bool = false
    ) {
        self.startDelimiter = startDelimiter
        self.endDelimiter = endDelimiter
        self.promptStrategy = promptStrategy
        self.isSpecialToken = isSpecialToken
    }

    // MARK: - Inference

    /// Infer a reasoning configuration from a model's `model_type` and repo id.
    ///
    /// Unlike ``ToolCallFormat/infer(from:configData:)``, `modelId` is
    /// load-bearing: DeepSeek-R1-Distill models report `model_type == "qwen2"`
    /// (or `"llama"`), indistinguishable from plain Qwen2.5/Llama by type alone,
    /// and must be recognized by their repo id.
    ///
    /// - Parameters:
    ///   - modelType: the `model_type` value from config.json.
    ///   - modelId: the Hugging Face repo id (e.g. `mlx-community/Qwen3-4B-4bit`).
    ///   - configData: raw config.json data for secondary signals (reserved; unused in v1).
    /// - Returns: the inferred ``ReasoningConfig``, or `nil` for non-reasoning models.
    public static func infer(
        from modelType: String,
        modelId: String? = nil,
        configData: Data? = nil
    ) -> ReasoningConfig? {
        let type = modelType.lowercased()
        let id = (modelId ?? "").lowercased()

        // Qwen3 family: <think>/</think>, thinking toggled via `enable_thinking`.
        //
        // Keyed on the model_type prefix, so a non-thinking Qwen3 variant (e.g.
        // a future Qwen3-Coder) could match. This is accepted today; on-device
        // verification and registry overrides refine specific models.
        if type.hasPrefix("qwen3") {
            return ReasoningConfig(
                startDelimiter: "<think>", endDelimiter: "</think>",
                promptStrategy: .templateFlag(key: "enable_thinking", defaultOn: true),
                isSpecialToken: true)
        }

        // DeepSeek-R1 (and R1-Distill): always-on <think>/</think>.
        //
        // R1-Distill reports its *base* model_type ("qwen2"/"llama"), so it must
        // be recognized by repo id. (Plain DeepSeek-V3 shares R1's "deepseek_v3"
        // model_type; this type is treated as reasoning, refined by registry overrides.)
        if type == "deepseek_v3" || type == "deepseek_r1"
            || id.contains("deepseek-r1") || id.contains("r1-distill")
        {
            return ReasoningConfig(
                startDelimiter: "<think>", endDelimiter: "</think>",
                promptStrategy: .alwaysOn)
        }

        return nil
    }
}
