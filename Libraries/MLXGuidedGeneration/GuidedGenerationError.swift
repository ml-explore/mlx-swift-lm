// Copyright © 2026 Apple Inc.

/// Errors from grammar-constrained generation.
///
/// These indicate structural failures where the grammar could not reach
/// an accepting state, meaning the output is syntactically incomplete.
public enum GuidedGenerationError: Error, Hashable {
    /// Generation exhausted `maxTokens` before the grammar reached a stop state.
    /// The output is incomplete (e.g., truncated JSON missing closing braces).
    case incompleteOutput

    /// The model emitted EOS before the grammar reached a stop state.
    /// The output is incomplete despite the model thinking it was done.
    case prematureEOS

    /// The grammar allowed no token the model can produce, so generation
    /// could not go on.
    ///
    /// A possible cause is a stop token that the grammar needs as text: the
    /// grammar never allows a stop token as text. `stopTokenIds` holds the
    /// grammar's stop token ids, sorted. When it is empty, no stop token is in
    /// the grammar's vocabulary, so even a complete output cannot end.
    case noAllowedToken(stopTokenIds: [Int])
}
