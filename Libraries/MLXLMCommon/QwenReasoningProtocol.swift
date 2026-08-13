// Copyright © 2026 Apple Inc.

/// Qwen-family reasoning protocol declarations.
///
/// Model-specific wire behavior lives here rather than in ``ReasoningConfig``.
/// The generic configuration only describes a protocol; this adapter decides
/// which Qwen families have a validated hard-budget transition.
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
    public static let qwen3 = ReasoningConfig(
        startDelimiter: "<think>", endDelimiter: "</think>",
        promptStrategy: .templateFlag(key: "enable_thinking", defaultOn: true),
        isSpecialToken: true,
        implicitEndDelimiters: ["<tool_call>"],
        budgetTransition: ReasoningBudgetTransition(
            beforeEndDelimiter:
                "\n\n Considering the limited time by the user, I have to give the solution based on the thinking directly now.\n",
            afterEndDelimiter: "\n\n"))

}
