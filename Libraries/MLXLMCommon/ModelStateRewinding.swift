// Copyright © 2026 Apple Inc.

import Foundation

/// A model whose carried state can follow its cache back to a shorter prefix.
///
/// Some models hand back state with every prefill, such as the M-RoPE position
/// delta of the Qwen VL family, and a session carries it across turns with the
/// cache. Rewinding the cache to a common prefix leaves that state describing
/// tokens the cache no longer holds, so without this conformance a session
/// rebuilds the cache instead.
///
/// Conform only if the state a cache resumes from is a function of the tokens
/// it holds: the rewound state must be what a cold prefill of `prefix` would
/// hand back.
public protocol ModelStateRewinding {

    /// Return the state a cache holding exactly `prefix` resumes from.
    ///
    /// Returning `nil` means the state cannot be derived for this rewind and the
    /// caller must rebuild the cache. Implementations should return `nil` rather
    /// than guess -- in particular when the state depends on media the tokens
    /// alone do not describe.
    ///
    /// - Parameters:
    ///   - state: the state carried with the cache before the rewind
    ///   - prefix: the tokens the cache holds after the rewind
    ///   - dropped: the tokens the rewind removes from the end of the cache
    /// - Returns: the state to resume from, or `nil` if it cannot be derived.
    func rewoundState(
        _ state: LMOutput.State, keeping prefix: [Int], dropping dropped: [Int]
    ) -> LMOutput.State?
}
