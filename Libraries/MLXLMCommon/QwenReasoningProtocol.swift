// Copyright © 2026 Apple Inc.

/// Qwen-family reasoning protocol declarations.
///
/// Model-specific wire behavior lives here rather than in ``ReasoningConfig``.
/// The generic configuration only describes a protocol; this adapter decides
/// which Qwen families have a validated hard-budget transition. Each of those
/// is derived from ``tagged`` through
/// ``ReasoningConfig/declaringBudgetTransition(_:)``, so the delimiters have a
/// single definition and a family can only differ in what was measured for it.
public enum QwenReasoningProtocol {
    /// Qwen-compatible `<think>` tags and tool-call boundary, without claiming
    /// that a hard-budget transition is safe for the model.
    public static let tagged = ReasoningConfig(
        startDelimiter: "<think>", endDelimiter: "</think>",
        promptStrategy: .templateFlag(key: "enable_thinking", defaultOn: true),
        isSpecialToken: true,
        implicitEndDelimiters: ["<tool_call>"])

    /// The original hybrid Qwen3 protocol and its published budget transition.
    ///
    /// The exact leading space after the two newlines is part of Qwen's
    /// `early_stopping_text`. Keeping this adapter family-specific prevents a
    /// later Qwen-derived model from inheriting the transition merely because it
    /// happens to use the same delimiters.
    public static let qwen3 = tagged.declaringBudgetTransition(
        ReasoningBudgetTransition(
            beforeEndDelimiter:
                "\n\n Considering the limited time by the user, I have to give the solution based on the thinking directly now.\n",
            afterEndDelimiter: "\n\n"))

    /// Qwen3.5, which answers from a span that was only closed.
    ///
    /// Qwen publishes no `early_stopping_text` for this generation, and forcing
    /// Qwen3's changed nothing: over 20 forced cuts on `Qwen3.5-9B-8bit` both
    /// closing the span alone and adding Qwen3's text produced a well-formed
    /// answer every time, while unbounded reasoning ran past the generation
    /// limit and answered almost never.
    public static let qwen35 = tagged.declaringBudgetTransition(.immediate)
}
