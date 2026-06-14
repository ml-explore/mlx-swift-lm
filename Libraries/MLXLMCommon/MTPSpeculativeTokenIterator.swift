// Copyright © 2026 Apple Inc.

import Foundation
import MLX

/// Generator of tokens using MTP (Multi-Token Prediction) speculative
/// decoding.
///
/// Parallels ``SpeculativeTokenIterator`` but for Gemma 4 - style drafters
/// that share K/V with the target model and produce K - 1 candidate tokens
/// per round in a single ``MTPDrafterModel/draftBlock(target:lastToken:lastHidden:sharedKV:positionDeltas:queryOffset:blockSize:sampler:)`` call (rather
/// than K sequential single-token calls). Every per-round input —
/// `lastToken`, `lastHidden`, `sharedKV`, `positionIds` — is threaded as a
/// method argument, with the target's last hidden state and per-`layer_type`
/// shared K/V extracted from the ``LMOutput/State`` emitted by the target on
/// the previous main-model call.
/// If the drafter needs its own KV cache (Qwen MTP), that cache is owned by
/// this iterator and trimmed after rejected speculative steps; it is never
/// stored on the shared drafter model.
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
/// args; drafter holds no target-derived state — the target and optional
/// per-stream drafter cache are passed as parameters to `draftBlock(...)` so
/// drafter instances are safe to share across iterators).
public struct MTPSpeculativeTokenIterator: TokenIteratorProtocol {

    var y: LMInput.Text

    let mainModel: any LanguageModel
    let drafter: any MTPDrafterModel

    var mainState: LMOutput.State?
    var mainCache: [KVCache]
    var drafterState: MTPDrafterState?
    let quantizeKVCache: (inout [KVCache]) -> Void

    var processor: LogitProcessor?
    let sampler: LogitSampler

    public var tokenCount = 0
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
        self.drafterState = (drafter as? any StatefulMTPDrafterModel)?
            .makeState(parameters: parameters)

        self.sampler = parameters.sampler()
        self.processor = parameters.processor()

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

    /// Prefill the main model with the prompt. The drafter's own state starts
    /// empty; its first-round conditioning inputs come from the prefill's
    /// `LMOutput.state`.
    mutating func prepare(input: LMInput, windowSize: Int? = nil) throws {
        processor?.prompt(input.text.tokens)

        var prefillState = LMOutput.State()
        prefillState[mtpEmitFlagKey] = true
        // Note: `prepare(_:cache:windowSize:)` does not currently thread
        // state through. To prime drafter state we run an explicit follow-up
        // forward call after prefill (one position, the bonus token).

        switch try mainModel.prepare(input, cache: mainCache, windowSize: windowSize) {
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
                var primeState = mainState ?? prefillState
                primeState[mtpEmitFlagKey] = true
                let primed = mainModel(y[text: .newAxis], cache: mainCache, state: primeState)
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
            let sharedKV = state[mtpSharedKVStatesKey],
            let fullAttentionKV = sharedKV["full_attention"]
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

        let sharedKVSpan = fullAttentionKV.0.dim(-2)
        let queryOffset =
            state[mtpSharedKVOffsetsKey]?["full_attention"] ?? sharedKVSpan

        // Invariant: the span the drafter attends over describes exactly the
        // true sequence — the rewind site trims the emitted snapshot in
        // lockstep with the cache. `dim()` is shape metadata (no eval, no GPU
        // sync). The check stands down if the cache ever leaves the trimmable
        // regime (post-wrap sliding window), where the rewind machinery
        // itself no-ops.
        assert(
            sharedKV.allSatisfy { $0.value.0.dim(-2) == sharedKVSpan }
                || !canTrimPromptCache(mainCache),
            "stale sharedKV: spans \(sharedKV.mapValues { $0.0.dim(-2) }) != full-attention span \(sharedKVSpan)"
        )

        let bonusToken = y.tokens
        let draftTokens: MLXArray
        if let statefulDrafter = drafter as? any StatefulMTPDrafterModel,
            var currentDrafterState = drafterState
        {
            draftTokens = statefulDrafter.draftBlock(
                target: mainModel,
                lastToken: bonusToken,
                lastHidden: bonusSlotHidden,
                sharedKV: sharedKV,
                positionDeltas: state[mtpPositionDeltasKey],
                queryOffset: queryOffset,
                blockSize: numDraft + 1,  // total round size: bonus + numDraft
                state: &currentDrafterState,
                sampler: sampler
            )
            drafterState = currentDrafterState
        } else {
            draftTokens = drafter.draftBlock(
                target: mainModel,
                lastToken: bonusToken,
                lastHidden: bonusSlotHidden,
                sharedKV: sharedKV,
                positionDeltas: state[mtpPositionDeltasKey],
                queryOffset: queryOffset,
                blockSize: numDraft + 1,  // total round size: bonus + numDraft
                sampler: sampler
            )
        }
        // draftTokens shape [B, numDraft] -> flatten to [numDraft].
        let flatDraftTokens = draftTokens.flattened()

        // Verify pass: main model evaluates [bonus, draft_1, ..., draft_numDraft]
        // in one forward call, emitting state for next round.
        var verifyState = state
        verifyState[mtpEmitFlagKey] = true
        let verifyTokens = concatenated([bonusToken, flatDraftTokens])
        let verifyInput = LMInput.Text(tokens: verifyTokens)
        let verifyStart = verifyInput.tokens.dim(0) - (numDraft + 1)
        let verifyOnMainCache = canTrimPromptCache(mainCache)
        let verifyCache = verifyOnMainCache ? mainCache : copyKVCache(mainCache)
        let mainResult = mainModel(
            verifyInput[text: .newAxis], cache: verifyCache, state: verifyState)
        let mainLogits = mainResult.logits
        mainState = mainResult.state

        // Sample one main-model token per verify position.
        let mainTokens: MLXArray
        // Local copy: process() may mutate processor state via Swift struct
        // value semantics, but those mutations stay scoped to this verify
        // loop. Canonical processor state at `self.processor` is only
        // updated by the accept loop below, which evolves it across the
        // actually-emitted tokens. This keeps rejected-draft sampling from
        // polluting cross-round state.
        if var verifyProcessorCopy = processor {
            var sampled = [MLXArray]()
            for i in 0 ..< (numDraft + 1) {
                var logits = mainLogits[0..., verifyStart + i, 0...]
                logits = verifyProcessorCopy.process(logits: logits)
                let token = sampler.sample(logits: logits)
                verifyProcessorCopy.didSample(token: token)
                sampled.append(token)
            }
            mainTokens = concatenated(sampled)
        } else {
            let verifyLogits = mainLogits[0..., verifyStart..., 0...].squeezed(axis: 0)
            mainTokens = sampler.sample(logits: verifyLogits)
        }

        eval(mainTokens, flatDraftTokens)
        let mainTokensList = mainTokens.asArray(Int.self)
        let draftTokensList = flatDraftTokens.asArray(Int.self)

        var accepted = 0
        for i in 0 ..< numDraft {
            guard mainTokensList[i] == draftTokensList[i] else { break }
            // Re-feed accepted draft positions to the processor so its
            // history matches the accepted-prefix view.
            let drafted = flatDraftTokens[i ..< (i + 1)]
            processor?.didSample(token: drafted)
            pendingTokens.append(mainTokensList[i])
            accepted += 1
        }

        // Always emit the main model's token at position `accepted` (either
        // a correction or the bonus token if all drafts matched).
        let finalToken = mainTokens[accepted ... accepted]
        processor?.didSample(token: finalToken)
        pendingTokens.append(mainTokensList[accepted])

        proposedCount += numDraft
        acceptedCount += accepted
        lastRoundAccepted = accepted

        let rejected = numDraft - accepted
        if let state = drafterState, rejected > 0 {
            trimPromptCache(state.cache, numTokens: rejected)
        }

        if verifyOnMainCache {
            if rejected > 0 {
                // Rewind the main cache and the emitted sharedKV snapshot by
                // the rejected count, in lockstep. The verify pass's state
                // emission spans the full verify chunk — stale tail rows must
                // not survive into the next round's draftBlock.
                let trimmed = trimPromptCache(mainCache, numTokens: rejected)
                trimSharedKVState(&mainState, numTokens: trimmed)
            }
        } else {
            // Hybrid models such as Qwen3.5/Qwen3.6 carry Mamba-style caches
            // that cannot be rewound after a rejected speculative tail. Verify
            // on a copied cache. If every draft is accepted, the verified
            // cache already represents the committed sequence and can be
            // adopted directly. Otherwise replay only the real prefix into
            // the canonical cache: bonus token plus the accepted draft
            // tokens. The finalToken is the next token to feed, so it is
            // intentionally not replayed here.
            if rejected == 0 {
                mainCache = verifyCache
            } else {
                var commitState = mainState ?? state
                commitState[mtpEmitFlagKey] = true
                let committedTokens = verifyTokens[0 ..< (accepted + 1)]
                let committed = mainModel(
                    LMInput.Text(tokens: committedTokens)[text: .newAxis],
                    cache: mainCache,
                    state: commitState
                )
                mainState = committed.state
            }
        }

        // Dynamic cache quantization may convert regular K/V to quantized K/V,
        // at which point the target's emit-hook cannot provide full_attention
        // shared K/V and the next round transitions to passthrough.
        quantizeKVCache(&mainCache)

        y = .init(tokens: finalToken)
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

        let result = mainModel(y[text: .newAxis], cache: mainCache, state: mainState)
        mainState = result.state
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
                tokenCount += 1
                return token
            }
            return nil
        }

        // Drain the pending buffer first.
        if pendingIndex < pendingTokens.count {
            let token = pendingTokens[pendingIndex]
            pendingIndex += 1
            tokenCount += 1
            return token
        }

        // Run a new speculation round (may transition to passthrough).
        pendingTokens.removeAll(keepingCapacity: true)
        pendingIndex = 0
        speculateRound()

        if pendingTokens.isEmpty {
            // speculateRound chose passthrough -- fall through.
            if passthrough {
                if let token = passthroughStep() {
                    tokenCount += 1
                    return token
                }
            }
            return nil
        }

        let token = pendingTokens[pendingIndex]
        pendingIndex += 1
        tokenCount += 1
        return token
    }
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
    if let offsets = state?[mtpSharedKVOffsetsKey] {
        state?[mtpSharedKVOffsetsKey] = offsets.mapValues {
            Swift.max(0, $0 - numTokens)
        }
    }
}

private func copyKVCache(_ cache: [KVCache]) -> [KVCache] {
    cache.map { $0.copy() }
}
