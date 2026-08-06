// Copyright © 2026 Apple Inc.

import Foundation

/// A model that can split an already-prepared ``LMInput`` at a token boundary.
///
/// ``ChatSession`` reuses a warm KV cache when a new turn only *appends* to the
/// transcript it already holds: it prefills the uncached suffix instead of the
/// whole prompt. For a text turn the suffix is just an array slice, so
/// ``ChatSession`` builds it directly.
///
/// A turn that carries media cannot be sliced that way. The prepared input holds
/// one media payload for the *entire* transcript — for Qwen VL that is the
/// concatenated patches and per-image grids of every image in the conversation —
/// so keeping the payload while dropping the cached tokens would hand the model
/// more images than the suffix has placeholders for. The result is a silent
/// mis-merge of vision features and, because the position math walks the grids in
/// token order, incorrect M-RoPE positions.
///
/// A model that knows its own media layout can do the split correctly: drop the
/// media that belongs to the cached prefix, keep the media whose placeholders lie
/// in the suffix. Conforming to this protocol opts a model into warm-cache reuse
/// for append-only media turns. Models that do not conform keep the previous
/// behavior — the cache is rebuilt.
///
/// Implementations must be conservative. Returning `nil` is always safe: it means
/// "I cannot prove this split is equivalent to a full prefill", and ``ChatSession``
/// falls back to rebuilding the cache.
///
/// ## Conformance requirement
///
/// A model may conform only if its media encoder computes each item's features
/// **independently of the other items in the payload**. If the encoder lets media
/// attend across item boundaries, then removing the cached prefix's items changes
/// the features computed for the items that remain, and the continuation no longer
/// matches a cold prefill even when the token positions line up exactly.
///
/// This is a real distinction and not a hypothetical one: ``Qwen25VL`` builds a
/// per-frame vision attention mask and so conforms, while `Qwen2VL` runs its vision
/// attention unmasked over the whole concatenated buffer and so does not.
public protocol PreparedInputSplitting {

    /// Return the suffix of `input` that begins at `prefixTokenCount`, carrying only
    /// the media whose placeholder tokens lie inside that suffix.
    ///
    /// Returning `nil` means the input cannot be split safely at this boundary and
    /// the caller must fall back to a full prefill. Implementations should return
    /// `nil` rather than guess — in particular when the boundary falls inside a
    /// media block, when a payload cannot be attributed to individual media items,
    /// or when the input carries state the implementation does not know how to slice.
    ///
    /// - Parameters:
    ///   - input: the prepared input for the full prompt
    ///   - prefixTokenCount: number of leading tokens already represented in the cache
    /// - Returns: an `LMInput` whose tokens are `input`'s tokens after the first
    ///   `prefixTokenCount`, with a media payload consistent with those tokens, or
    ///   `nil` if no such input can be produced.
    func splitPreparedInput(_ input: LMInput, droppingFirst prefixTokenCount: Int) -> LMInput?
}
