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
    /// Returning `nil` means the state cannot be derived for this prefix and the
    /// caller must rebuild the cache. Implementations should return `nil` rather
    /// than guess -- in particular when the state depends on media in `prefix`
    /// that the tokens alone do not describe.
    ///
    /// - Parameter prefix: the tokens the cache holds after the rewind
    /// - Returns: the state to resume from, or `nil` if it cannot be derived.
    func rewoundState(forPrefix prefix: [Int]) -> LMOutput.State?
}
