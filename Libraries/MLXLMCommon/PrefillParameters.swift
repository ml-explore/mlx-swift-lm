// Copyright © 2026 Apple Inc.

import Foundation

/// Parameters controlling how a prompt is prefilled into the ``KVCache``.
///
/// Set via ``GenerateParameters/prefill`` and passed to
/// ``LanguageModel/prepare(_:cache:state:prefill:)``:
///
/// ```swift
/// var parameters = GenerateParameters()
/// parameters.prefill.stepSize = 1024
/// parameters.prefill.progress = { processed, total in ... }
/// ```
public struct PrefillParameters: Sendable {

    /// Ceiling on tokens evaluated per prefill forward. `nil` lets each model pick
    /// its own default (512 for the generic path; the Gemma 3 text path uses a
    /// smaller, tuned chunk).
    public var stepSize: Int?

    /// How the prompt is divided into forwards. See ``Chunking``.
    public var chunking: Chunking

    /// Called after each prefill chunk with `(processedTokens, totalTokens)`.
    ///
    /// A terminal `(total, total)` is delivered once the prompt is fully consumed,
    /// for every model — including those that prefill in a single forward. Because
    /// chunks are pipelined with `asyncEval`, intermediate calls report graph
    /// submission, slightly ahead of GPU completion.
    public var progress: (@Sendable (_ processed: Int, _ total: Int) -> Void)?

    /// Strategy dividing the prompt into prefill forwards.
    public enum Chunking: Sendable {
        /// The fewest equal chunks that respect the step-size ceiling, so no
        /// forward is disproportionately small at large KV length (the default).
        case balanced

        /// Legacy stride: chunks of exactly the step size, with the remainder
        /// riding along with the final forward. An escape hatch back to the
        /// pre-``balanced`` chunk boundaries and their exact outputs.
        case remainder

        /// The whole prompt in one forward — the reference computation that any
        /// chunking approximates. Attention scores are quadratic in the prompt
        /// length, so this is for validation at short lengths, not production.
        case unchunked
    }

    public init(
        stepSize: Int? = nil,
        chunking: Chunking = .balanced,
        progress: (@Sendable (_ processed: Int, _ total: Int) -> Void)? = nil
    ) {
        self.stepSize = stepSize
        self.chunking = chunking
        self.progress = progress
    }

    /// The per-forward chunk length for prefilling `count` positions, or `nil`
    /// when the prompt should go through in a single forward (``Chunking/unchunked``).
    ///
    /// `count` is the number of positions the caller intends to chunk — excluding
    /// any tail it reserves for the first sampled forward. Models pass their own
    /// `defaultStepSize` (used when ``stepSize`` is `nil`) and, if they have one,
    /// a `maximumStepSize` cap such as a sliding-window size.
    public func chunkLength(
        forChunking count: Int, defaultStepSize: Int = 512, maximumStepSize: Int? = nil
    ) -> Int? {
        let step = max(1, min(stepSize ?? defaultStepSize, maximumStepSize ?? Int.max))
        switch chunking {
        case .unchunked:
            return nil
        case .remainder:
            return step
        case .balanced:
            guard count > step else { return max(count, 1) }
            let chunkCount = (count + step - 1) / step
            return (count + chunkCount - 1) / chunkCount
        }
    }
}
