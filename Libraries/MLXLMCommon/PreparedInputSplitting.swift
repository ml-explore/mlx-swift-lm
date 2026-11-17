// Copyright © 2026 Apple Inc.

import Foundation

/// A model that can split its own prepared input at a token boundary, keeping
/// only the media whose placeholders fall in the suffix.
///
/// Conform only if the media encoder computes each item's features independently
/// of the others; otherwise dropping cached items changes the features of the
/// items that remain. `Qwen25VL`, for example, masks each frame to itself and conforms.
public protocol PreparedInputSplitting {

    /// Return the suffix of `input` that begins at `prefixTokenCount`, carrying only
    /// the media whose placeholder tokens lie inside that suffix.
    ///
    /// Returning `nil` means the input cannot be split safely at this boundary and
    /// the caller must fall back to a full prefill. Implementations should return
    /// `nil` rather than guess -- in particular when the boundary falls inside a
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
