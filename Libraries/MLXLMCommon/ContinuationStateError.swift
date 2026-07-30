// Copyright © 2024 Apple Inc.

import Foundation

/// Errors thrown when a warm KV cache is continued without the model state it requires.
///
/// Models that place tokens with M-RoPE (the Qwen vision families, GLM-OCR) keep a
/// continuation anchor in ``LMOutput/State`` alongside the KV cache. Continuing a cache that
/// already holds tokens without that anchor positions the new tokens as if the cached prefix
/// contained no images, which silently changes the model's output rather than failing.
///
/// Save the anchor with the cache using `savePromptCache(url:cache:metadata:state:)`, restore
/// both with `loadPromptCacheSnapshot(url:)`, and hand the snapshot to a `ChatSession`
/// initializer that accepts a `promptCache` so the pair cannot be separated.
public enum ContinuationStateError: LocalizedError, Equatable {
    /// A cache warmed to a non-zero offset was supplied without the state key the model needs.
    ///
    /// - Parameters:
    ///   - model: the model that requires the state
    ///   - key: the missing ``LMOutput/Key`` identifier
    case missingState(model: String, key: String)

    /// A warm cache was supplied for a batched continuation, but this model only
    /// supports a single cache/state anchor.
    case unsupportedBatchContinuation(model: String)

    public var errorDescription: String? {
        switch self {
        case .missingState(let model, let key):
            """
            \(model) cannot continue a warm prompt cache without the model state key '\(key)'. \
            Continuing would position new tokens as if the cached prefix contained no images, \
            silently changing the output. Restore the cache with loadPromptCacheSnapshot(url:) \
            and pass the snapshot to ChatSession(_:promptCache:), or pass its state to \
            TokenIterator(input:model:cache:state:parameters:). A cache file saved without \
            model state cannot be continued — rebuild it with \
            savePromptCache(url:cache:metadata:state:).
            """
        case .unsupportedBatchContinuation(let model):
            """
            \(model) cannot continue a warm prompt cache for more than one batch row. \
            Its cached continuation state has one position anchor, so a batched resume would \
            silently use the wrong positions. Continue each row with its own cache and state.
            """
        }
    }
}
