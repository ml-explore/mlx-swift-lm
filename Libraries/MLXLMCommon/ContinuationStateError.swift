// Copyright © 2024 Apple Inc.

import Foundation

/// Errors thrown when a warm KV cache is continued without the model state it requires.
///
/// Models that place tokens with M-RoPE (the Qwen vision families, GLM-OCR) keep a continuation
/// anchor in ``LMOutput/State`` alongside the KV cache. Continuing a warm cache without that
/// anchor positions new tokens as if the cached prefix contained no images, silently changing
/// the output rather than failing.
///
/// Save and restore the pair with `savePromptCache(url:cache:metadata:state:)` and
/// `loadPromptCacheSnapshot(url:)`.
public enum ContinuationStateError: LocalizedError, Equatable {
    /// A cache warmed to a non-zero offset was supplied without the state key the model needs.
    case missingState(model: String, key: String)

    /// A warm cache was supplied for a batched continuation, but this model carries a single
    /// position anchor.
    case unsupportedBatchContinuation(model: String)

    public var errorDescription: String? {
        switch self {
        case .missingState(let model, let key):
            """
            \(model) cannot continue a warm prompt cache without the model state key '\(key)'. \
            Restore the cache with loadPromptCacheSnapshot(url:) and pass the snapshot to \
            ChatSession(_:promptCache:); a file saved without model state must be rebuilt.
            """
        case .unsupportedBatchContinuation(let model):
            """
            \(model) cannot continue a warm prompt cache for more than one batch row. \
            Continue each row with its own cache and state.
            """
        }
    }
}
