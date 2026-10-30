// Copyright © 2026 Apple Inc.

/// Onyx-family reasoning protocol declarations.
///
/// Onyx models reason inside a frame addressed to themselves and leave it at a
/// frame boundary rather than at an end-of-turn token. As in
/// ``QwenReasoningProtocol``, the frame syntax is declared once and each family
/// adds only the transition it was measured with, so a later Onyx model cannot
/// inherit a budget transition by sharing the control vocabulary.
public enum OnyxReasoningProtocol {
    /// The `to=self` reasoning frame, without claiming that a hard-budget
    /// transition is safe for the model.
    public static let frames = ReasoningConfig(
        startDelimiter: "to=self<|message|>", endDelimiter: "<|eom|>",
        promptStrategy: .none,
        isSpecialToken: true)

    /// Muse Glimmer, which answers from a frame that was only closed.
    ///
    /// `<|eom|>` ends the frame without ending the turn, so the model opens its
    /// own `assistant to=user` frame next and closing the frame is the whole
    /// transition. Measured on `Muse-Glimmer-30B-4bit`: over 20 forced cuts the
    /// model always answered the user and never resumed reasoning, while also
    /// forcing an answer header pushed scratch work into the user-visible
    /// answer.
    public static let glimmer = frames.declaringBudgetTransition(.immediate)
}
