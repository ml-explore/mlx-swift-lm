// Copyright © 2026 Apple Inc.

import Foundation
import MLX

/// Generator of tokens using MTP (Multi-Token Prediction) speculative
/// decoding.
///
/// Parallels ``SpeculativeTokenIterator`` but for Gemma 4 - style drafters
/// that share K/V with the target model and produce K - 1 candidate tokens
/// per round in a single ``MTPDrafterModel/draftBlock(target:lastToken:lastHidden:sharedKV:queryOffset:blockSize:sampler:)`` call (rather
/// than K sequential single-token calls). The drafter has no own KV cache:
/// every per-round input — `lastToken`, `lastHidden`, `sharedKV`,
/// `positionIds` — is threaded as a method argument, with the target's last
/// hidden state and per-`layer_type` shared K/V extracted from the
/// ``LMOutput/State`` emitted by the target on the previous main-model call.
///
/// The iterator pre-populates each main-model call's incoming `state` with
/// ``mtpEmitFlagKey`` set to `true`, opting the target into populating
/// ``mtpLastHiddenStatesKey`` and ``mtpSharedKVStatesKey`` on its returned
/// ``LMOutput/state``. If the target ever returns nil or partial state
/// (e.g. once the KV cache quantizes and the regular K/V tuples are no
/// longer available), the iterator transparently switches into a
/// single-token "passthrough" mode for the remainder of generation —
/// covering R8 (no quantized MTP for this PR) and R13 (mid-generation
/// quantization onset must not crash).
///
/// Port of `_speculative_walk` from mlx-vlm/generate.py at SHA `d49d428`,
/// with no-mutation-during-eval idioms (state is threaded through method
/// args; drafter holds no target-derived state — the target is passed as
/// a parameter to `draftBlock(...)` so drafter instances are safe to share
/// across iterators).
public struct MTPSpeculativeTokenIterator: TokenIteratorProtocol {

    var y: LMInput.Text

    let mainModel: any LanguageModel
    let drafter: any MTPDrafterModel

    var mainState: LMOutput.State?
    var mainCache: [KVCache]
    let quantizeKVCache: (inout [KVCache]) -> Void

    var processor: LogitProcessor?
    let sampler: LogitSampler

    /// Distribution surface for non-greedy speculative sampling. Built-in
    /// temperature/top-p samplers conform; argmax intentionally does not and
    /// keeps the cheaper exact-token greedy path.
    let distributionSampler: (any DistributionLogitSampler)?

    /// Independent RNG used for speculative acceptance and residual/bonus
    /// draws. Draft proposals use `draftSampler`'s own RNG; separating them is
    /// required by the rejection-sampling proof.
    let speculativeRandomState: MLXRandom.RandomState

    /// Independent sampler handed to the drafter. Distribution-capable
    /// drafters receive a copy of the full target processor as well, so their
    /// proposal distribution matches the target's (penalties included).
    let draftSampler: LogitSampler

    /// Sampler for older greedy-only drafters whose API cannot receive a
    /// processor.
    let legacyDraftSampler: LogitSampler

    public var tokenCount: Int { telemetry.emittedTokenCount }
    public let maxTokens: Int?
    /// Total tokens proposed per round (`blockSize - 1` drafted, plus the
    /// bonus token from the previous verify). Mirrors mlx-vlm's
    /// `draft_block_size` parameter.
    public let blockSize: Int

    private var pendingTokens = [Int]()
    private var pendingIndex = 0

    /// Set to `true` when the iterator detects that the target can no
    /// longer emit drafter state (typically due to KV cache quantization
    /// converting `Gemma4SharedKVState.regular` to `.quantized`). Once set,
    /// `next()` runs single-token generation against the main model only —
    /// no further `speculateRound` calls. Sticky: never reverts to `false`.
    private var passthrough = false
    private var passthroughLoggedOnce = false

    /// Verify-position index in the prior round's emitted hidden that
    /// produced the newly-accepted bonus's logit prediction. Set at the end
    /// of each `speculateRound()`. `nil` on the first round means slice the
    /// last position (round 1's `lastHidden` has shape `[B, 1, hidden]`, so
    /// last-position == only-position == the correct slot). Round 2+ slices
    /// at this index, mirroring mlx-lm's `verify.hidden[:, accepted : accepted + 1, :]`.
    /// Mismatch (e.g. unconditional last-position) is silent: drafter still
    /// produces tokens, but they're conditioned on the wrong slot → less
    /// coherent drafts → lower acceptance, especially at higher blockSize.
    private var lastRoundAccepted: Int? = nil

    public var promptPrefillTime: TimeInterval = 0.0
    private var telemetry = SpeculativeDecodingTelemetry()
    public var speculativeDecodingTelemetry: SpeculativeDecodingTelemetry? {
        telemetry.roundCount > 0 ? telemetry : nil
    }

    public mutating func discardGeneratedToken() {
        telemetry.discardGeneratedToken()
    }

    // Optional instrumentation used by acceptance-rate floor tests.
    // Public read-only so test cases can compute `acceptedCount /
    // proposedCount` after the stream drains.
    public private(set) var acceptedCount: Int = 0
    public private(set) var proposedCount: Int = 0

    // Reason recorded the first time sticky-passthrough engaged, or nil if
    // the iterator stayed speculative for the full stream. Surfaced through
    // ``MTPStatsCollecting`` so `generateLoopTask` can include it on the
    // emitted `.info` event.
    public private(set) var passthroughReason: String?

    public init(
        input: LMInput,
        mainModel: any LanguageModel,
        drafter: any MTPDrafterModel,
        mainCache: [KVCache]? = nil,
        parameters: GenerateParameters,
        blockSize: Int
    ) throws {
        precondition(
            blockSize >= 2,
            "MTPSpeculativeTokenIterator requires blockSize >= 2 (1 bonus + K-1 drafted)")

        self.y = input.text
        self.mainModel = mainModel
        self.drafter = drafter

        self.mainCache = mainCache ?? mainModel.newCache(parameters: parameters)
        guard canTrimPromptCache(self.mainCache) else {
            throw KVCacheError(
                message: "MTP speculative decoding requires a trimmable main KV cache.")
        }

        self.sampler = parameters.sampler()
        self.distributionSampler = self.sampler as? any DistributionLogitSampler
        self.speculativeRandomState =
            parameters.seed.map {
                MLXRandom.RandomState(seed: $0 &+ 0x9E37_79B9_7F4A_7C15)
            } ?? MLXRandom.RandomState()
        self.processor = parameters.processor()

        var draftParameters = parameters
        draftParameters.seed = parameters.seed.map { $0 &+ 0xD1B5_4A32_D192_ED03 }
        let drafterSampler = draftParameters.sampler()
        self.draftSampler = drafterSampler
        self.legacyDraftSampler = drafterSampler

        self.maxTokens = parameters.maxTokens
        self.blockSize = blockSize

        self.quantizeKVCache = { cache in
            maybeQuantizeKVCache(
                cache: &cache,
                kvBits: parameters.kvBits,
                kvGroupSize: parameters.kvGroupSize,
                quantizedKVStart: parameters.quantizedKVStart
            )
        }

        let prefillStart = Date.timeIntervalSinceReferenceDate
        try prepare(input: input, windowSize: parameters.prefillStepSize)
        self.promptPrefillTime = Date.timeIntervalSinceReferenceDate - prefillStart
    }

    /// Prefill the main model with the prompt. The drafter has no cache to
    /// prime; its first-round inputs come from the prefill's `LMOutput.state`.
    mutating func prepare(input: LMInput, windowSize: Int? = nil) throws {
        processor?.prompt(input.text.tokens)

        var prefillState = LMOutput.State()
        prefillState[mtpEmitFlagKey] = true
        // Note: the drafter is primed via an explicit follow-up forward call
        // after prefill (one position, the bonus token) rather than by
        // passing `prefillState` into `prepare` — the emit flag is meant for
        // exactly one position, not the whole prompt.

        switch try mainModel.prepare(input, cache: mainCache, state: nil, windowSize: windowSize)
        {
        case .tokens(let tokens):
            y = tokens
            // Final prompt position not yet evaluated -- run one forward to
            // produce the bonus token AND prime drafter state.
            let result = mainModel(y[text: .newAxis], cache: mainCache, state: prefillState)
            var logits = result.logits[0..., -1, 0...]
            logits = processor?.process(logits: logits) ?? logits
            let token = sampler.sample(logits: logits)
            processor?.didSample(token: token)
            y = .init(tokens: token)
            mainState = result.state
            // Yield the bonus to the iterator's consumer. Without this,
            // the iterator silently starts 1 position ahead of an
            // equivalent autoregressive run, violating speculative
            // decoding's bit-exact-equivalence-to-greedy guarantee.
            pendingTokens.append(token.item(Int.self))
        case .logits(let prefillResult):
            // Some `prepare` implementations evaluate the final position
            // themselves and return logits directly; their `state` here may
            // or may not carry drafter state depending on whether the model
            // override threads it.
            var logits = prefillResult.logits[0..., -1, 0...]
            logits = processor?.process(logits: logits) ?? logits
            let token = sampler.sample(logits: logits)
            processor?.didSample(token: token)
            y = .init(tokens: token)
            mainState = prefillResult.state

            // If prefill didn't emit drafter state, do one more forward call
            // with the just-sampled bonus token to prime the state. The cost
            // is one extra token's forward pass; acceptable.
            if mainState?[mtpLastHiddenStatesKey] == nil
                || mainState?[mtpSharedKVStatesKey] == nil
            {
                let primed = mainModel(y[text: .newAxis], cache: mainCache, state: prefillState)
                mainState = primed.state
                // Resample bonus from this forward's logits so the chain stays
                // coherent at this position (the cache offset moves by 1, so
                // we must re-pick the bonus from the new step's logits).
                var newLogits = primed.logits[0..., -1, 0...]
                newLogits = processor?.process(logits: newLogits) ?? newLogits
                let newToken = sampler.sample(logits: newLogits)
                processor?.didSample(token: newToken)
                y = .init(tokens: newToken)
                // Yield BOTH bonuses to the consumer, in sample order.
                // `token` is the prefill-position-N sample (consumed by the
                // re-prime forward, now committed in cache); `newToken` is
                // the prefill-position-N+1 sample that becomes the input
                // to the first speculateRound.
                pendingTokens.append(token.item(Int.self))
                pendingTokens.append(newToken.item(Int.self))
            } else {
                // Prefill state already carried drafter keys; the single
                // bonus is the input to the first speculateRound.
                pendingTokens.append(token.item(Int.self))
            }
        }
    }

    /// Single round: draft `blockSize - 1` tokens, verify with main, accept
    /// the longest matching prefix, emit the bonus correction.
    mutating func speculateRound() {
        guard !passthrough else { return }

        // A speculative round can emit up to `numDraft + 1` tokens: the
        // accepted draft prefix plus the verifier's correction/bonus token.
        // Keep the whole pending buffer within the remaining output budget.
        let numDraft: Int
        if let maxTokens {
            let remaining = maxTokens - tokenCount
            guard remaining > 0 else { return }

            let draftBudget = Swift.min(remaining - 1, blockSize - 1)
            guard draftBudget > 0 else {
                if let token = passthroughStep() {
                    pendingTokens.append(token)
                }
                return
            }
            numDraft = draftBudget
        } else {
            numDraft = blockSize - 1
        }

        guard
            let state = mainState,
            let lastHidden = state[mtpLastHiddenStatesKey],
            let sharedKV = state[mtpSharedKVStatesKey]
        else {
            switchToPassthrough(reason: "main model did not emit drafter state")
            return
        }

        // Slice the hidden at the slot that produced the newly-accepted
        // bonus's prediction. Round 1: last (and only) position. Round 2+:
        // index `lastRoundAccepted`, matching mlx-lm's
        // `verify.hidden[:, accepted : accepted + 1, :]` semantic.
        let bonusSlotHidden: MLXArray
        if let idx = lastRoundAccepted {
            bonusSlotHidden = lastHidden[0..., idx ..< (idx + 1), 0...]
        } else {
            bonusSlotHidden = lastHidden[0..., (-1)..., 0...]
        }

        let cacheOffset = mainCache.first?.offset ?? 0

        // Invariant: the span the drafter attends over describes exactly the
        // true sequence — the rewind site trims the emitted snapshot in
        // lockstep with the cache. `dim()` is shape metadata (no eval, no GPU
        // sync). The check stands down if the cache ever leaves the trimmable
        // regime (post-wrap sliding window), where the rewind machinery
        // itself no-ops.
        assert(
            sharedKV.allSatisfy { $0.value.0.dim(-2) == cacheOffset }
                || !canTrimPromptCache(mainCache),
            "stale sharedKV: spans \(sharedKV.mapValues { $0.0.dim(-2) }) != main cache offset \(cacheOffset)"
        )

        let bonusToken = y.tokens
        let draftOutput: MTPDraftBlockOutput?
        let draftTokens: MLXArray
        if let distributionDrafter = drafter as? any MTPDistributionDrafterModel {
            let output = distributionDrafter.draftBlockWithLogits(
                target: mainModel,
                lastToken: bonusToken,
                lastHidden: bonusSlotHidden,
                sharedKV: sharedKV,
                queryOffset: cacheOffset,
                blockSize: numDraft + 1,  // total round size: bonus + numDraft
                processor: processor,
                sampler: draftSampler
            )
            draftOutput = output
            draftTokens = output.tokens
        } else {
            // Greedy matching needs tokens only and remains compatible with
            // the original protocol. Sampling cannot preserve the target
            // distribution without q logits, so fail safe to target-only
            // generation instead of silently changing semantics.
            guard distributionSampler == nil else {
                switchToPassthrough(
                    reason: "MTP drafter does not expose logits required for sampling")
                return
            }
            draftOutput = nil
            draftTokens = drafter.draftBlock(
                target: mainModel,
                lastToken: bonusToken,
                lastHidden: bonusSlotHidden,
                sharedKV: sharedKV,
                queryOffset: cacheOffset,
                blockSize: numDraft + 1,
                sampler: legacyDraftSampler
            )
        }
        // draftTokens shape [B, numDraft] -> flatten to [numDraft].
        let flatDraftTokens = draftTokens.flattened()

        // Verify pass: main model evaluates [bonus, draft_1, ..., draft_numDraft]
        // in one forward call, emitting state for next round.
        var verifyState = LMOutput.State()
        verifyState[mtpEmitFlagKey] = true
        let verifyTokens = concatenated([bonusToken, flatDraftTokens])
        let verifyInput = LMInput.Text(tokens: verifyTokens)
        let verifyStart = verifyInput.tokens.dim(0) - (numDraft + 1)
        let mainResult = mainModel(
            verifyInput[text: .newAxis], cache: mainCache, state: verifyState)
        let mainLogits = mainResult.logits
        mainState = mainResult.state

        let selection: (accepted: Int, emitted: [MLXArray])
        if let distributionSampler, let draftOutput {
            selection = selectDistributionPreservingTokens(
                targetLogits: mainLogits,
                targetStart: verifyStart,
                draftTokens: flatDraftTokens,
                draftProcessedLogits: draftOutput.processedLogits,
                numDraft: numDraft,
                sampler: distributionSampler
            )
        } else {
            selection = selectGreedyTokens(
                targetLogits: mainLogits,
                targetStart: verifyStart,
                draftTokens: flatDraftTokens,
                numDraft: numDraft
            )
        }

        let emittedTokens = concatenated(selection.emitted)
        eval(emittedTokens, flatDraftTokens)
        let emittedTokenList = emittedTokens.asArray(Int.self)
        for (index, token) in selection.emitted.enumerated() {
            processor?.didSample(token: token)
            pendingTokens.append(emittedTokenList[index])
        }

        let accepted = selection.accepted
        let finalToken = selection.emitted.last!

        proposedCount += numDraft
        acceptedCount += accepted
        lastRoundAccepted = accepted
        telemetry.recordRound(
            drafted: numDraft,
            accepted: accepted,
            targetVerified: numDraft + 1,
            draftModelCalls: 1
        )

        // Rewind the main cache and the emitted sharedKV snapshot by the
        // rejected count, in lockstep. The drafter has no cache of its own,
        // but the verify pass's state emission spans the full verify chunk —
        // stale tail rows must not survive into the next round's draftBlock.
        // Trimming the snapshot by the amount the cache actually trimmed
        // keeps the two consistent even if the cache ever reports itself
        // untrimmable (post-wrap sliding window), where trimPromptCache
        // no-ops and returns 0.
        let rejected = numDraft - accepted
        let trimmed = trimPromptCache(mainCache, numTokens: rejected)
        trimSharedKVState(&mainState, numTokens: trimmed)

        // Dynamic cache quantization may convert `.regular` K/V to `.quantized`,
        // at which point the target's emit-hook returns sharedKV: nil and the
        // next round transitions to passthrough.
        quantizeKVCache(&mainCache)

        y = .init(tokens: finalToken)
    }

    /// Greedy verifier walk. The target is evaluated once for the whole
    /// verify block, then the longest exact candidate prefix is committed and
    /// followed by the target correction/bonus token.
    private func selectGreedyTokens(
        targetLogits: MLXArray,
        targetStart: Int,
        draftTokens: MLXArray,
        numDraft: Int
    ) -> (accepted: Int, emitted: [MLXArray]) {
        // Local processor state follows target samples only for this tentative
        // walk. Canonical state is updated by the caller from emitted tokens.
        var verificationProcessor = processor
        var targetTokens = [MLXArray]()
        targetTokens.reserveCapacity(numDraft + 1)
        for i in 0 ..< (numDraft + 1) {
            var logits = targetLogits[0..., targetStart + i, 0...]
            logits = verificationProcessor?.process(logits: logits) ?? logits
            let token = sampler.sample(logits: logits)
            verificationProcessor?.didSample(token: token)
            targetTokens.append(token)
        }

        let targetTokenArray = concatenated(targetTokens)
        eval(targetTokenArray, draftTokens)
        let targetTokenList = targetTokenArray.asArray(Int.self)
        let draftTokenList = draftTokens.asArray(Int.self)

        var accepted = 0
        while accepted < numDraft
            && targetTokenList[accepted] == draftTokenList[accepted]
        {
            accepted += 1
        }
        return (accepted, Array(targetTokens.prefix(accepted + 1)))
    }

    /// Distribution-preserving speculative sampling (Leviathan et al.).
    ///
    /// Candidate `x ~ q` is accepted with `min(1, p(x) / q(x))`. On the first
    /// rejection, the correction is sampled from normalized `max(p - q, 0)`;
    /// if all candidates are accepted, the target bonus is sampled from `p`.
    /// This preserves the target distribution while allowing the target to
    /// verify the entire candidate block in one forward pass.
    private func selectDistributionPreservingTokens(
        targetLogits: MLXArray,
        targetStart: Int,
        draftTokens: MLXArray,
        draftProcessedLogits: MLXArray,
        numDraft: Int,
        sampler distributionSampler: any DistributionLogitSampler
    ) -> (accepted: Int, emitted: [MLXArray]) {
        var verificationProcessor = processor
        let draftTokenList = draftTokens.asArray(Int.self)
        var emitted = [MLXArray]()
        emitted.reserveCapacity(numDraft + 1)

        for i in 0 ..< numDraft {
            let candidateID = draftTokenList[i]
            let candidate = draftTokens[i ..< (i + 1)]

            var pLogits = targetLogits[0..., targetStart + i, 0...]
            pLogits = verificationProcessor?.process(logits: pLogits) ?? pLogits
            let pLogProbabilities = distributionSampler.logProbabilities(logits: pLogits)
            let qLogits = draftProcessedLogits[0..., i, 0...]
            let qLogProbabilities = distributionSampler.logProbabilities(logits: qLogits)

            let decision = sampleSpeculativeDecision(
                candidateID: candidateID,
                targetLogProbabilities: pLogProbabilities,
                draftLogProbabilities: qLogProbabilities)

            if decision.accepted {
                emitted.append(candidate)
                verificationProcessor?.didSample(token: candidate)
                continue
            }

            emitted.append(decision.correction)
            return (i, emitted)
        }

        // Every draft was accepted. The final verifier position predicts one
        // extra target-only bonus token.
        var bonusLogits = targetLogits[0..., targetStart + numDraft, 0...]
        bonusLogits = verificationProcessor?.process(logits: bonusLogits) ?? bonusLogits
        let bonusLogProbabilities = distributionSampler.logProbabilities(logits: bonusLogits)
        emitted.append(sampleSpeculative(logProbabilities: bonusLogProbabilities))
        return (numDraft, emitted)
    }

    /// Draw acceptance and the possible residual correction in one GPU
    /// evaluation. Sampling the correction eagerly (and discarding it on
    /// acceptance) preserves the distribution while avoiding a second or
    /// third CPU/GPU synchronization per candidate.
    private func sampleSpeculativeDecision(
        candidateID: Int,
        targetLogProbabilities p: MLXArray,
        draftLogProbabilities q: MLXArray
    ) -> (accepted: Bool, correction: MLXArray) {
        let logAcceptance = minimum(
            p[0, candidateID] - q[0, candidateID],
            MLXArray(Float(0)))
        let correctionLogProbabilities = speculativeResidualLogProbabilities(
            target: p,
            draft: q)
        let draws = withRandomState(speculativeRandomState) {
            (
                MLXRandom.uniform(Float(0) ..< Float(1), [1]),
                categorical(correctionLogProbabilities)
            )
        }
        let accepted = log(draws.0) .< logAcceptance
        eval(accepted, draws.1)
        return (accepted.item(Bool.self), draws.1)
    }

    private func sampleSpeculative(logProbabilities: MLXArray) -> MLXArray {
        withRandomState(speculativeRandomState) {
            categorical(logProbabilities)
        }
    }

    /// Switch to single-token generation for the remainder of the stream.
    /// Sticky — once flipped, `next()` never returns to speculation.
    private mutating func switchToPassthrough(reason: String) {
        if !passthroughLoggedOnce {
            // Log one-time only so a quantization-onset round doesn't spam.
            // The Swift stdlib `print` is intentional here: the iterator is
            // a low-level component without access to a logger.
            print("[MTPSpeculativeTokenIterator] passthrough mode: \(reason)")
            passthroughLoggedOnce = true
        }
        passthroughReason = reason
        passthrough = true
    }

    /// One single-token forward step against the main model, used in
    /// passthrough mode. The drafter is not invoked.
    private mutating func passthroughStep() -> Int? {
        if let maxTokens, tokenCount >= maxTokens { return nil }

        let result = mainModel(y[text: .newAxis], cache: mainCache, state: nil)
        var logits = result.logits[0..., -1, 0...]
        logits = processor?.process(logits: logits) ?? logits
        let token = sampler.sample(logits: logits)
        processor?.didSample(token: token)
        eval(token)
        let tokenInt = token.item(Int.self)
        y = .init(tokens: token)
        quantizeKVCache(&mainCache)
        return tokenInt
    }

    public mutating func next() -> Int? {
        if let maxTokens, tokenCount >= maxTokens {
            return nil
        }

        if passthrough {
            if let token = passthroughStep() {
                telemetry.recordGeneratedToken()
                return token
            }
            return nil
        }

        // Drain the pending buffer first.
        if pendingIndex < pendingTokens.count {
            let token = pendingTokens[pendingIndex]
            pendingIndex += 1
            telemetry.recordGeneratedToken()
            return token
        }

        // Run a new speculation round (may transition to passthrough).
        pendingTokens.removeAll(keepingCapacity: true)
        pendingIndex = 0
        autoreleasepool { speculateRound() }

        if pendingTokens.isEmpty {
            // speculateRound chose passthrough -- fall through.
            if passthrough {
                if let token = passthroughStep() {
                    telemetry.recordGeneratedToken()
                    return token
                }
            }
            return nil
        }

        let token = pendingTokens[pendingIndex]
        pendingIndex += 1
        telemetry.recordGeneratedToken()
        return token
    }
}

/// Acceptance probability for one candidate in exact speculative sampling.
/// Kept as a scalar helper so boundary cases (`p=0`, `q=0`, and very small
/// ratios) can be tested without requiring a Metal device.
func speculativeAcceptanceProbability(
    targetLogProbability p: Float,
    draftLogProbability q: Float
) -> Double {
    guard q.isFinite else { return 0 }
    guard !p.isNaN else { return 0 }
    if p >= q { return 1 }
    return Foundation.exp(Double(p - q))
}

/// Normalized residual distribution `max(p - q, 0)` used after a rejected
/// speculative candidate. When finite-precision arithmetic collapses the
/// residual mass, `p` is the stable limiting fallback.
func speculativeResidualLogProbabilities(
    target p: MLXArray,
    draft q: MLXArray
) -> MLXArray {
    let residual = maximum(exp(p) - exp(q), MLXArray(Float(0)))
    let residualMass = residual.sum()
    let hasResidual = residualMass .> Float(1e-7)
    let safeMass = maximum(residualMass, MLXArray(Float(1e-7)))
    return MLX.where(hasResidual, log(residual / safeMass), p)
}

extension MTPSpeculativeTokenIterator: MTPStatsCollecting {
    public var proposedDraftTokens: Int { proposedCount }
    public var acceptedDraftTokens: Int { acceptedCount }
}

extension MTPSpeculativeTokenIterator {
    /// Test-only setter for the canonical `LogitProcessor`. Lets regression
    /// tests install a recording probe AFTER `init` (which calls `prepare`
    /// and would otherwise consume the prepare-time bonus before the probe
    /// is observable). Used by the emit-only invariant regression tests in
    /// `MTPSpeculativeTokenIteratorTests` (CI-scoped) and
    /// `MTPIteratorEndToEndDiagnosticTests` (31B end-to-end).
    @_spi(Testing) public mutating func _setProcessorForTesting(
        _ processor: LogitProcessor?
    ) {
        self.processor = processor
    }

    /// Test-only getter for the canonical `LogitProcessor` so regression
    /// tests can inspect its post-drain state (e.g., a recording probe's
    /// accumulated didSample log).
    @_spi(Testing) public var _processorForTesting: LogitProcessor? {
        processor
    }
}

/// Rewinds the emitted MTP shared-K/V snapshot by `numTokens` trailing
/// sequence positions, mirroring `trimPromptCache` on the main cache.
///
/// The verify pass emits K/V spanning the full `[bonus, d_1 ... d_numDraft]`
/// chunk — materialized before acceptance is known. After a partial
/// acceptance, the rejected tail rows describe tokens that are not part of
/// the sequence; without this trim, the next round's `draftBlock` would
/// cross-attend over them. (PR #308 review: discussion_r3391133046,
/// discussion_r3391147261.)
///
/// No-op when `numTokens <= 0`, when `state` is nil, or when the key is
/// absent (e.g. the quantization-onset round, whose fresh verify state
/// carries no sharedKV). Cost is metadata-only: the slices are lazy views
/// consumed by the next `draftBlock` like the rest of the round's inputs;
/// no `eval`. Iterator-internal — `trimPromptCache` is public because
/// caches are a public surface, but this snapshot is the iterator's own
/// cross-round state.
func trimSharedKVState(_ state: inout LMOutput.State?, numTokens: Int) {
    guard numTokens > 0,
        let sharedKV = state?[mtpSharedKVStatesKey]
    else { return }
    state?[mtpSharedKVStatesKey] = sharedKV.mapValues { kv in
        let newLen = kv.0.dim(-2) - numTokens
        return (
            kv.0[.ellipsis, ..<newLen, 0...],
            kv.1[.ellipsis, ..<newLen, 0...]
        )
    }
}
