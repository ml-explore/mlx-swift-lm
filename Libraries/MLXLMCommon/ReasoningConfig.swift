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

// MARK: - ReasoningChannel

/// The labeled-channel shape of a reasoning protocol: the start delimiter opens a
/// *channel* whose role is named by a short label that follows it, and one shared
/// end delimiter closes every channel whatever its role.
///
/// Gemma 4 is the family that needs this. Its delimiters are real special tokens
/// (`soc_token` id 100 `<|channel>`, `eoc_token` id 101 `<channel|>`) and the
/// reasoning span is named by the label:
///
/// ```
/// <|channel>thought\n ... <channel|>the answer
/// ```
///
/// A plain ``ReasoningConfig`` (start/end delimiter pair) cannot express that.
/// A pair anchored on `<|channel>` emits `thought` as the first word of the
/// thinking; a pair anchored on `<|channel>thought\n` stops matching the moment
/// the label is anything else, leaking the raw opener into the answer.
public struct ReasoningChannel: Sendable, Equatable {

    /// Ends the role label that follows the start delimiter (Gemma 4: a newline).
    public var labelTerminator: String

    /// Labels whose channel carries the answer rather than reasoning.
    ///
    /// Every label NOT listed here routes to reasoning. Hiding an unrecognized
    /// channel in the reasoning stream is the safe direction: nothing is lost (the
    /// text stays readable) and no scratchpad text or raw marker can reach the
    /// answer.
    public var responseLabels: Set<String>

    /// How far past the start delimiter a label may run before the opener is
    /// treated as malformed and its text is shown as reasoning rather than
    /// consumed as metadata.
    ///
    /// Bounding it matters in both directions: without the bound, an opener whose
    /// label never terminates either buffers the whole generation or silently
    /// swallows everything up to a distant end delimiter.
    public var maxLabelLength: Int

    public init(
        labelTerminator: String = "\n",
        responseLabels: Set<String>,
        maxLabelLength: Int = 32
    ) {
        self.labelTerminator = labelTerminator
        self.responseLabels = responseLabels
        self.maxLabelLength = maxLabelLength
    }

    /// Gemma 4: `<|channel>thought\n ... <channel|>` carries reasoning.
    ///
    /// `content` is listed defensively rather than observed. Every shipped Gemma 4
    /// template emits only the `thought` label, and Google's own `x-regex` response
    /// schema names the answer group `content` while leaving it OUTSIDE any channel
    /// - so an answer arriving channel-wrapped is not expected. Listing the label
    /// costs nothing, and means a channel-wrapped answer would not be hidden in the
    /// thought stream if one ever did appear.
    public static let gemma4 = ReasoningChannel(responseLabels: ["content"])
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

    /// When non-nil, ``startDelimiter`` opens a *labeled channel* rather than a
    /// reasoning span directly: a role label follows it and decides whether the
    /// span is reasoning or the answer. `nil` (the default) is the plain
    /// delimiter-pair protocol, e.g. `<think>` ... `</think>`.
    public var channel: ReasoningChannel?

    public init(
        startDelimiter: String,
        endDelimiter: String,
        promptStrategy: ReasoningPromptStrategy,
        isSpecialToken: Bool = false,
        channel: ReasoningChannel? = nil
    ) {
        self.startDelimiter = startDelimiter
        self.endDelimiter = endDelimiter
        self.promptStrategy = promptStrategy
        self.isSpecialToken = isSpecialToken
        self.channel = channel
    }

    // MARK: - Presets

    /// `<think>`/`</think>`, toggled by the `enable_thinking` chat-template flag
    /// (default on). The contract shared by the Qwen3 family and Nanbeige4.2+.
    public static let thinkTagsWithEnableThinking = ReasoningConfig(
        startDelimiter: "<think>", endDelimiter: "</think>",
        promptStrategy: .templateFlag(key: "enable_thinking", defaultOn: true),
        isSpecialToken: true)

    /// Always-on `<think>`/`</think>` with no prompt-level off switch
    /// (DeepSeek-R1 and its distills).
    public static let alwaysOnThinking = ReasoningConfig(
        startDelimiter: "<think>", endDelimiter: "</think>",
        promptStrategy: .alwaysOn)

    /// Gemma 4's protocol: labeled channels (see ``ReasoningChannel/gemma4``), toggled
    /// via `enable_thinking`, whose template default is thinking OFF.
    ///
    /// Shared here rather than spelled per model because four types (text, unified,
    /// and the LLM and VLM entry points) declare the same protocol.
    ///
    /// The markers are not hypothetical. Gemma 4's chat templates vary in how much they
    /// prefill: the 31B template writes a closed, empty thought block into the generation
    /// prompt when thinking is off, while the E2B template prefills nothing at all. And
    /// the 31B template suppresses the whole generation prompt - `<|turn>model` and the
    /// prefill together - when the previous message was a tool call or tool response,
    /// which is precisely the turn an agentic client reads. In each of those cases the
    /// model is expected to open the channel itself, and without this the delimiters
    /// reach the caller as literal text.
    public static let gemma4 = ReasoningConfig(
        startDelimiter: "<|channel>", endDelimiter: "<channel|>",
        promptStrategy: .templateFlag(key: "enable_thinking", defaultOn: false),
        isSpecialToken: true,
        channel: .gemma4)
}
