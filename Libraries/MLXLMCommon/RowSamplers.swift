// Per-row samplers for `BatchGenerator`. Mirrors `mlx_lm.sample_utils`:
// temperature scaling, top-K truncation, top-P (nucleus) truncation, and
// optional seeded categorical sampling. Each constructed sampler is
// `@Sendable` so it can be stored alongside an admitted batch row.

import Foundation
import MLX
import MLXRandom

/// Build a `RowSampler` from OpenAI-style request parameters.
///
/// Behavior, mirroring upstream `mlx_lm.sample_utils.make_sampler`:
///   - `temperature <= 0`: return `greedySampler` (no RNG, deterministic).
///   - `temperature > 0`: scale logits by `1/temperature`, optionally
///     mask to top-K and/or top-P, then sample categorically.
///
/// `seed` makes the resulting stream deterministic. Each call to the
/// returned sampler advances the per-sampler PRNG key, so successive
/// decode steps produce different draws even with the same input
/// distribution. Two requests sharing the same seed but called in
/// different orders/contexts will produce **different** sequences --
/// this matches `mlx_lm` behavior and OpenAI's documented "best-effort"
/// determinism.
///
/// - Parameters:
///   - temperature: 0 = greedy, otherwise the logits divisor.
///   - topP: nucleus probability mass (1.0 = disabled).
///   - topK: number of top tokens to keep (0 = disabled).
///   - seed: optional RNG seed; nil uses MLX's global PRNG.
public func makeRowSampler(
    temperature: Float = 0.0,
    topP: Float = 1.0,
    topK: Int = 0,
    seed: UInt64? = nil
) -> RowSampler {
    if temperature <= 0 {
        return greedySampler
    }

    let keyHolder = SamplerKeyHolder(seed: seed)
    let temp = temperature
    let p = topP
    let k = topK

    return { @Sendable logprobs in
        var lp = logprobs * (1.0 / temp)
        if k > 0 { lp = applyTopK(lp, k: k) }
        if p > 0, p < 1 { lp = applyTopP(lp, p: p) }
        let key = keyHolder.next()
        return MLXRandom.categorical(lp, axis: -1, key: key)
    }
}

// MARK: - Top-K

/// Mask all but the top-K logits along the last axis to `-inf`. If
/// `k >= vocab` the input is returned unchanged.
@usableFromInline
func applyTopK(_ logprobs: MLXArray, k: Int) -> MLXArray {
    let vocab = logprobs.shape.last ?? 0
    if k <= 0 || k >= vocab { return logprobs }

    let sortedIdx = argSort(-logprobs, axis: -1)
    let topIdx = sortedIdx[.ellipsis, 0 ..< k]
    let topVals = takeAlong(logprobs, topIdx, axis: -1)
    let threshold = topVals.min(axes: [-1], keepDims: true)
    let negInf = MLXArray(-Float.infinity)
    return which(logprobs .>= threshold, logprobs, negInf)
}

// MARK: - Top-P (nucleus)

/// Keep the smallest set of tokens whose cumulative softmax mass is at
/// least `p`. Tokens outside that nucleus are masked to `-inf`. The
/// "first token over threshold" stays in (mlx_lm semantics) so a
/// degenerate distribution still yields one valid pick.
@usableFromInline
func applyTopP(_ logprobs: MLXArray, p: Float) -> MLXArray {
    let sortedIdxAsc = argSort(logprobs, axis: -1)
    let sortedLogits = takeAlong(logprobs, sortedIdxAsc, axis: -1)
    let sortedProbs = softmax(sortedLogits, axis: -1)
    let cumProbs = sortedProbs.cumsum(axis: -1)

    // Keep tokens whose suffix-mass (tail starting here) >= 1 - p.
    // i.e. mask sorted positions where cumProbs <= 1 - p, but keep the
    // boundary token (first one above threshold).
    let keepThreshold = MLXArray(1.0 - p)
    let keepMask = cumProbs .> keepThreshold
    let negInf = MLXArray(-Float.infinity)
    let maskedSorted = which(keepMask, sortedLogits, negInf)

    // Scatter back to original token order.
    let inverseIdx = argSort(sortedIdxAsc, axis: -1)
    return takeAlong(maskedSorted, inverseIdx, axis: -1)
}

// MARK: - Per-sampler PRNG state

/// Holds a per-sampler PRNG key and advances it on every draw.
///
/// `RowSampler` is `@Sendable`, but each row's sampler is invoked from a
/// single context (`GenerationBatch.step` runs serially on the
/// `ModelContainer` actor), so a class-backed mutable holder is sound;
/// we mark it `@unchecked Sendable` because MLX `MLXArray` keys are not
/// Sendable but our access pattern is single-threaded per row.
final class SamplerKeyHolder: @unchecked Sendable {
    private var key: MLXArray?

    init(seed: UInt64?) {
        self.key = seed.map { MLXRandom.key($0) }
    }

    func next() -> MLXArray? {
        guard let current = key else { return nil }
        let (subkey, nextKey) = MLXRandom.split(key: current)
        key = nextKey
        return subkey
    }
}
