// Copyright © 2026 Apple Inc.

import MLX

/// Speculative sampling (Leviathan et al. 2023, Chen et al. 2023) over `N` draft rows.
///
/// Draft `i` is kept with probability `min(1, p_i(x_i) / q_i(x_i))`. The correction of row `i`
/// is drawn from the residual `max(p_i - q_i, 0)`: it is the token to emit when draft `i` is the
/// first draft that is not kept. Kept drafts and the correction follow `p` exactly, at any
/// temperature.
///
/// - Parameters:
///   - target: float32 log probabilities `[N, V]` of the target's sampling distribution.
///   - proposal: float32 log probabilities `[N, V']`, `V' <= V`, that the drafts were drawn
///     from. The drafter never proposes a token past `V'`.
///   - drafts: the `N` draft tokens.
///   - randomState: the source of the acceptance and correction draws.
/// - Returns: whether each draft is kept, and the correction of each row.
func speculativeSamplingDecisions(
    target: MLXArray, proposal: MLXArray, drafts: MLXArray, randomState: MLXRandom.RandomState
) -> (kept: MLXArray, corrections: MLXArray) {
    let (rows, width) = (target.dim(0), target.dim(-1))
    var proposal = proposal
    if proposal.dim(-1) < width {
        let unreachable = MLXArray.full(
            [rows, width - proposal.dim(-1)], values: MLXArray(-Float.infinity))
        proposal = concatenated([proposal, unreachable], axis: -1)
    }
    let tokens = drafts.reshaped(rows, 1).asType(.int32)
    let logRatio =
        takeAlong(target, tokens, axis: -1) - takeAlong(proposal, tokens, axis: -1)
    let kept = log(uniform(0 ..< 1, [rows, 1], key: randomState)) .< logRatio

    // A row whose residual is empty has `p == q`, so its draft is always kept.
    let residual = maximum(exp(target) - exp(proposal), 0)
    let hasResidual = residual.sum(axis: -1, keepDims: true) .> 0
    let corrections = categorical(
        MLX.where(hasResidual, log(residual), target), key: randomState)
    return (kept.reshaped(rows), corrections)
}

/// The sampler a drafter draws its proposals with when the target samples at a nonzero
/// temperature.
///
/// It draws from the target sampler's distribution, with the suppressed tokens masked, and keeps
/// the distribution of each draw so the verify pass can apply ``speculativeSamplingDecisions``.
final class DraftProposalSampler: LogitSampler {
    let distribution: any DistributionLogitSampler
    let suppressor: SuppressTokensProcessor?
    let randomState: MLXRandom.RandomState
    private var proposals = [MLXArray]()

    init(
        distribution: any DistributionLogitSampler, suppressor: SuppressTokensProcessor?,
        randomState: MLXRandom.RandomState
    ) {
        self.distribution = distribution
        self.suppressor = suppressor
        self.randomState = randomState
    }

    func sample(logits: MLXArray) -> MLXArray {
        let proposal = distribution.logProbabilities(
            logits: suppressor?.process(logits: logits) ?? logits)
        proposals.append(proposal.reshaped(-1, proposal.dim(-1)))
        return categorical(proposal, key: randomState)
    }

    /// The proposals `[N, V']` of the draws since the last call, in draw order, and forgets them.
    /// `nil` when nothing was drawn.
    func takeProposals() -> MLXArray? {
        defer { proposals.removeAll() }
        return proposals.isEmpty ? nil : concatenated(proposals)
    }

    /// Forgets the draws whose drafts will never be verified.
    func discardProposals() {
        proposals.removeAll()
    }
}

/// The proposal `[N, width]` of a drafter that drafts without drawing: all of each row's mass
/// is on its draft.
func deterministicProposal(drafts: MLXArray, width: Int) -> MLXArray {
    MLX.where(
        drafts.reshaped(-1, 1).asType(.int32) .== MLXArray.arange(width).asType(.int32),
        MLXArray(Float(0)), MLXArray(-Float.infinity))
}
