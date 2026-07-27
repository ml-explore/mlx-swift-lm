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

    /// Called after each prefill chunk with `(processedPositions, totalPositions)`.
    ///
    /// Positions are the units the model prefills: prompt tokens for text
    /// models, merged embedding positions (after image expansion) for VLMs. A
    /// terminal `(total, total)` is delivered once the prompt is fully
    /// consumed, for every model — including those that prefill in a single
    /// forward. Because chunks are pipelined with `asyncEval`, intermediate
    /// calls report graph submission, slightly ahead of GPU completion.
    ///
    /// The closure is retained for as long as the containing parameters (e.g.
    /// by a stored `GenerateParameters`) — capture weakly anything long-lived.
    public var progress: (@Sendable (_ processed: Int, _ total: Int) -> Void)?

    /// Strategy dividing the prompt into prefill forwards.
    public enum Chunking: Sendable {
        /// The fewest equal chunks that respect the step-size ceiling, so no
        /// forward is disproportionately small at large KV length (the default).
        case balanced

        /// Legacy stride: chunks of exactly the step size. An escape hatch back
        /// to each model's pre-``balanced`` chunk boundaries and their exact
        /// outputs — on the default `LLMModel` path the remainder rides along
        /// with the `TokenIterator`'s first forward; models that reserve one
        /// position for the logits keep that shape and stride the rest.
        case remainder

        /// The whole prompt in one forward — the reference computation that any
        /// chunking approximates. Attention scores are quadratic in the prompt
        /// length and the logits materialize at `[1, L, vocab]`, so this is for
        /// validation at short lengths, not production.
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

    /// The step size used when neither ``stepSize`` nor a model default applies.
    public static let defaultStepSize = 512

    /// The effective step-size ceiling: ``stepSize`` if set, else the model's
    /// `defaultStepSize`, clamped to `maximumStepSize` and floored at 1.
    public func resolvedStepSize(
        defaultStepSize: Int = PrefillParameters.defaultStepSize, maximumStepSize: Int? = nil
    ) -> Int {
        max(1, min(stepSize ?? defaultStepSize, maximumStepSize ?? Int.max))
    }

    /// The per-forward chunk length for prefilling `count` positions, or `nil`
    /// when the prompt should go through in a single forward (``Chunking/unchunked``).
    ///
    /// `count` is the number of positions the caller intends to chunk — excluding
    /// any tail it reserves for the first sampled forward. Models pass their own
    /// `defaultStepSize` (used when ``stepSize`` is `nil`) and, if they have one,
    /// a `maximumStepSize` cap such as a sliding-window size.
    public func chunkLength(
        forChunking count: Int,
        defaultStepSize: Int = PrefillParameters.defaultStepSize,
        maximumStepSize: Int? = nil
    ) -> Int? {
        let step = resolvedStepSize(
            defaultStepSize: defaultStepSize, maximumStepSize: maximumStepSize)
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

    /// Drives a chunked prefill over `total` positions and returns the number
    /// of positions processed.
    ///
    /// Computes the chunk length once, then calls `body` with each chunk's
    /// range. The driver owns the loop's shared obligations: cooperative
    /// cancellation between chunks (a long prompt must be interruptible — see
    /// the note in `LLMModel.prepare`), an autorelease pool per chunk (long
    /// prompts run hundreds of forwards before any autorelease boundary), and
    /// the per-chunk ``progress`` report. `body` performs the model's forward
    /// and its own cache fencing (`asyncEval`/`eval`).
    ///
    /// `reserving` positions are left unprocessed for the caller's final
    /// forward — the one that produces logits (1 for every current caller).
    /// Under ``Chunking/unchunked`` no chunking happens and 0 is returned:
    /// the caller's final forward takes the whole prompt.
    public func forEachChunk(
        total: Int,
        reserving: Int = 1,
        defaultStepSize: Int = PrefillParameters.defaultStepSize,
        maximumStepSize: Int? = nil,
        _ body: (Range<Int>) -> Void
    ) throws -> Int {
        guard
            let chunkLength = chunkLength(
                forChunking: total - reserving,
                defaultStepSize: defaultStepSize, maximumStepSize: maximumStepSize)
        else {
            return 0
        }
        let slack = min(reserving, 1)
        var processed = 0
        while total - processed > reserving {
            try Task.checkCancellation()
            autoreleasepool {
                let n = min(chunkLength, total - processed - slack)
                body(processed ..< processed + n)
                processed += n
                progress?(processed, total)
            }
        }
        return processed
    }
}
