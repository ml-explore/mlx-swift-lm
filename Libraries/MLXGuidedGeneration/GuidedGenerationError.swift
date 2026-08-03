// Copyright © 2026 Apple Inc.

/// Errors from grammar-constrained generation.
///
/// These indicate structural failures where the grammar could not reach
/// an accepting state, meaning the output is syntactically incomplete.
public enum GuidedGenerationError: Error {
    /// Generation exhausted `maxTokens` before the grammar reached a stop state.
    /// The output is incomplete (e.g., truncated JSON missing closing braces).
    case incompleteOutput

    /// The model emitted EOS before the grammar reached a stop state.
    /// The output is incomplete despite the model thinking it was done.
    case prematureEOS
}
