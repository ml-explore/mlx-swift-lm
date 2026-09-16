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
/// this iterator, prefilled over the shifted prompt, and reconciled against
/// accepted target tokens after every verify pass; it is never stored on the
/// shared drafter model.
///
/// The iterator pre-populates each main-model call's incoming `state` with
/// ``mtpEmitFlagKey`` set to `true`, opting the target into populating
/// ``mtpLastHiddenStatesKey`` and ``mtpSharedKVStatesKey`` on its returned
/// ``LMOutput/state``. If the target ever returns nil or partial state
/// (for example when a shared-KV drafter can no longer read regular K/V), the
/// iterator transparently switches into a
/// single-token "passthrough" mode for the remainder of generation —
/// a mid-generation capability loss never crashes or corrupts the stream.
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
    let mainCacheStorage: KVCacheStorage
    var mainCache: [KVCache] { mainCacheStorage.cache }
    var kvCachePlan: KVCachePlan { mainCacheStorage.plan }
    var drafterState: MTPDrafterState?

    var processor: LogitProcessor?
    let sampler: LogitSampler

    public var tokenCount: Int { telemetry.emittedTokenCount }
    public let maxTokens: Int?
    /// Total tokens proposed per round (`blockSize - 1` drafted, plus the
    /// bonus token from the previous verify). Mirrors mlx-vlm's
    /// `draft_block_size` parameter.
    public let blockSize: Int

    private var pendingTokens = [Int]()
    private var pendingIndex = 0
    /// Number of pending tokens already represented by `mainCache`. The last
    /// verifier sample is not committed until a later forward pass.
    private var committedPendingTokenCount = 0

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
        blockSize: Int,
        components: GenerationComponents = .init()
    ) throws {
        precondition(
            blockSize >= 2,
            "MTPSpeculativeTokenIterator requires blockSize >= 2 (1 bonus + K-1 drafted)")
        guard drafter.isCompatible(with: mainModel) else {
            throw ModelFactoryError.invalidConfiguration(
                "MTP drafter \(type(of: drafter)) is incompatible with target "
                    + "\(type(of: mainModel))")
        }

        let kvCachePlan = try parameters.kvCachePlan()
        let mainCache = try kvCachePlan.validated(
            mainCache ?? (try mainModel.newCache(parameters: parameters)))
        self.y = input.text
        self.mainModel = mainModel
        self.drafter = drafter

        self.mainCacheStorage = KVCacheStorage(mainCache, plan: kvCachePlan)
        self.drafterState = (drafter as? any StatefulMTPDrafterModel)?
            .makeState(parameters: parameters)

        self.sampler = parameters.sampler()
        try components.validate(parameters: parameters)
        self.processor = components.logitProcessor(parameters: parameters)

        self.maxTokens = parameters.maxTokens
        // A round presents `blockSize` positions at once, and a sliding layer can only show a
        // query the `maxSize` entries before it. Past that the extra drafts still decode
        // correctly -- masks are position-relative and the staged view is clamped -- but the
        // deepest ones attend over less context than the verifier gave them, so acceptance
        // stops meaning what the stats say it means. Clamp rather than trap: the block size is
        // a tuning knob, not a correctness input.
        let drafterBlockSize = Swift.min(blockSize, drafter.maximumBlockSize ?? blockSize)
        let effectiveBlockSize = Swift.min(
            drafterBlockSize, Self.maximumBlockSize(for: mainCache))
        self.blockSize = effectiveBlockSize

        // Probe by opening a round at the width rounds will actually use and discarding it,
        // rather than duplicating the leaf classification as a predicate that could drift from
        // it. Qwen's hybrid cache is the one typed exception: its target advertises a bounded
        // recurrent checkpoint and performs the round in place.
        let nativeRewindDepth =
            (mainModel as? any SpeculativeCacheRewindModel)?
            .maximumNativeTargetCacheRewind ?? 0
        let usesNativeHybridRewind =
            nativeRewindDepth >= effectiveBlockSize - 1
            && mainCache.contains { $0 is MambaCache }
            && mainCache.allSatisfy { $0.isTrimmable || $0 is MambaCache }
        if !usesNativeHybridRewind {
            guard
                let probe = self.mainCacheStorage.beginRound(
                    maximumPositions: effectiveBlockSize)
            else {
                throw KVCacheError(
                    message: "MTP speculative decoding requires a stageable main KV cache.")
            }
            self.mainCacheStorage.rollback(probe)
        }

        let prefillStart = Date.timeIntervalSinceReferenceDate
        var mtpPrefill = parameters.prefill
        if drafter.requiresPromptPrefill {
            // The target must expose one hidden row per prompt token so a
            // private Qwen MTP cache can be filled with the shifted prompt.
            // Until model-specific chunk aggregation is available, use the
            // reference single-forward computation for this architecture.
            mtpPrefill.stepSize = Int.max
            mtpPrefill.chunking = .unchunked
        }
        try prepare(input: input, prefill: mtpPrefill)
        self.promptPrefillTime = Date.timeIntervalSinceReferenceDate - prefillStart

        if drafter.requiresGreedySampling, parameters.temperature != 0 {
            switchToPassthrough(
                reason:
                    "Qwen MTP currently requires temperature == 0; generating without speculation"
            )
        }
    }

    static let missingSharedKVSourcesReason =
        "main model did not report which cache entries its shared K/V came from; continuing "
        + "without speculation"

    /// The tightest sliding window among the cache's layers, or `nil` when no layer slides.
    static func narrowestSlidingWindow(in cache: [KVCache]) -> Int? {
        KVCacheTree.leaves(in: cache)
            .compactMap { leaf -> Int? in
                guard case .rotating(let rotating) = leaf.kind else { return nil }
                return rotating.maxSize
            }
            .min()
    }

    /// The widest round the narrowest sliding layer can present coherently: one bonus plus a
    /// window's worth of drafts. Unbounded when no layer slides.
    static func maximumBlockSize(for cache: [KVCache]) -> Int {
        guard let narrowest = narrowestSlidingWindow(in: cache) else { return .max }
        return Swift.max(2, narrowest + 1)
    }

    /// Prefill the main model with the prompt. The drafter's own state starts
    /// empty; its first-round conditioning inputs come from the prefill's
    /// `LMOutput.state`.
    mutating func prepare(input: LMInput, prefill: PrefillParameters = .init()) throws {
        processor?.prompt(input.text.tokens)
        let inputLength = input.text.cacheSequenceLength

        var prefillState = LMOutput.State()
        prefillState[mtpEmitFlagKey] = true
        // Note: the drafter is primed via an explicit follow-up forward call
        // after prefill (one position, the bonus token) rather than by
        // passing `prefillState` into `prepare` — the emit flag is meant for
        // exactly one position, not the whole prompt.

        let incomingPrefillState: LMOutput.State? =
            drafter.requiresPromptPrefill ? prefillState : nil
        switch try mainModel.prepare(
            input, cache: mainCache, state: incomingPrefillState, prefill: prefill)
        {
        case .tokens(let tokens):
            let remainingLength = tokens.cacheSequenceLength
            precondition(
                remainingLength <= inputLength,
                "Main model prepare returned more tokens than it received")
            mainCacheStorage.commitProcessedTokens(inputLength - remainingLength)
            y = tokens
            // Final prompt position not yet evaluated -- run one forward to
            // produce the bonus token AND prime drafter state.
            let result = mainModel(y[text: .newAxis], cache: mainCache, state: prefillState)
            mainCacheStorage.commitProcessedTokens(y.cacheSequenceLength)
            var logits = result.logits[0..., -1, 0...]
            logits = processor?.process(logits: logits) ?? logits
            let token = sampler.sample(logits: logits)
            processor?.didSample(token: token)
            y = .init(tokens: token)
            mainState = result.state
            // Prefill emits too, and a multi-token prefill presents a sliding layer more than its
            // window just as a verify pass does. Nothing was rejected here, so only the head
            // clamp applies -- but a snapshot that cannot be placed against the entries at all is
            // refused here for the same reason it is mid-stream, and before anything reads it.
            if !reconcileSharedKVState(
                &mainState, discarding: 0,
                lengths: mainCacheStorage.emittedLength(forLeaf:))
            {
                switchToPassthrough(reason: Self.missingSharedKVSourcesReason)
                mainState = nil
            }
            // Yield the bonus to the iterator's consumer. Without this,
            // the iterator silently starts 1 position ahead of an
            // equivalent autoregressive run, violating speculative
            // decoding's bit-exact-equivalence-to-greedy guarantee.
            pendingTokens.append(token.item(Int.self))

            // the model reported per-chunk progress; the bonus forward above
            // consumed the remainder of the prompt
            let total = input.text.tokens.size
            prefill.progress?(total, total)
        case .logits(let prefillResult):
            mainCacheStorage.commitProcessedTokens(inputLength)
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
            let placeable = reconcileSharedKVState(
                &mainState, discarding: 0,
                lengths: mainCacheStorage.emittedLength(forLeaf:))
            if !placeable {
                switchToPassthrough(reason: Self.missingSharedKVSourcesReason)
                mainState = nil
            }

            // If prefill didn't emit drafter state, do one more forward call
            // with the just-sampled bonus token to prime the state. The cost
            // is one extra token's forward pass; acceptable. Skipped once the
            // snapshot has been refused: it exists only to obtain drafter
            // state, and this stream has stopped having a use for any.
            if placeable,
                mainState?[mtpLastHiddenStatesKey] == nil
                    || mainState?[mtpSharedKVStatesKey] == nil
            {
                var primeState = mainState ?? prefillState
                primeState[mtpEmitFlagKey] = true
                let primed = mainModel(y[text: .newAxis], cache: mainCache, state: primeState)
                mainCacheStorage.commitProcessedTokens(y.cacheSequenceLength)
                mainState = primed.state
                if !reconcileSharedKVState(
                    &mainState, discarding: 0,
                    lengths: mainCacheStorage.emittedLength(forLeaf:))
                {
                    switchToPassthrough(reason: Self.missingSharedKVSourcesReason)
                    mainState = nil
                }
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
                committedPendingTokenCount = 1
            } else {
                // Prefill state already carried drafter keys; the single
                // bonus is the input to the first speculateRound.
                pendingTokens.append(token.item(Int.self))
            }
        }

        if drafter.requiresPromptPrefill,
            let statefulDrafter = drafter as? any StatefulMTPDrafterModel,
            var currentDrafterState = drafterState,
            let targetHidden = mainState?[mtpLastHiddenStatesKey]
        {
            let promptLength = input.text.tokens.dim(-1)
            guard targetHidden.dim(1) >= promptLength else {
                switchToPassthrough(
                    reason:
                        "target did not emit full prompt hidden states for Qwen MTP prefill"
                )
                return
            }
            statefulDrafter.prepareDrafterState(
                target: mainModel,
                promptTokens: input.text.tokens,
                targetHidden: targetHidden,
                firstBonus: y.tokens,
                positionDeltas: mainState?[mtpPositionDeltasKey],
                state: &currentDrafterState,
                sampler: sampler)
            drafterState = currentDrafterState
        } else if drafter.requiresPromptPrefill {
            switchToPassthrough(
                reason: "target did not emit drafter state for Qwen MTP prompt prefill")
        }

        try kvCachePlan.applyAndValidate(to: mainCacheStorage)
    }

    /// Single round: draft `blockSize - 1` tokens, verify with main, accept
    /// the longest matching prefix, emit the bonus correction.
    mutating func speculateRound() {
        guard !passthrough else { return }
        // A prior all-accepted round may keep one recurrent checkpoint until
        // its pending output is drained so early finalization can rewind it.
        discardSpeculativePromptCacheCheckpoints(mainCache)

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

        guard let state = mainState,
            let lastHidden = state[mtpLastHiddenStatesKey]
        else {
            switchToPassthrough(reason: "main model did not emit drafter state")
            return
        }
        let sharedKV = state[mtpSharedKVStatesKey] ?? [:]
        if drafter.requiresSharedTargetKV, sharedKV["full_attention"] == nil {
            switchToPassthrough(reason: "main model did not emit shared target K/V")
            return
        }

        // Attention-only targets verify through a staged round. Qwen's hybrid target instead
        // checkpoints its recurrent entries and rewinds them in place; the typed capability and
        // cache topology keep that exception fail-closed.
        let nativeRewindDepth =
            (mainModel as? any SpeculativeCacheRewindModel)?
            .maximumNativeTargetCacheRewind ?? 0
        let nativeHybridRewind =
            nativeRewindDepth >= numDraft
            && mainCache.contains { $0 is MambaCache }
            && mainCache.allSatisfy { $0.isTrimmable || $0 is MambaCache }
        let round =
            nativeHybridRewind
            ? nil : mainCacheStorage.beginRound(maximumPositions: numDraft + 1)
        guard nativeHybridRewind || round != nil else {
            switchToPassthrough(
                reason: "main KV cache cannot stage a speculative round; continuing without "
                    + "speculation")
            mainState = nil
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

        let queryOffset =
            state[mtpSharedKVOffsetsKey]?["full_attention"]
            ?? mainCacheStorage.processedTokenCount

        // Invariant: the snapshot the drafter attends over describes the emitted sequence and
        // nothing else. A global layer's entry spans the whole stream; a sliding layer's spans
        // its window. Both are exact, because the commit that decided what the cache kept also
        // reconciled the snapshot. `dim()` is shape metadata (no eval, no GPU sync).
        assert(
            {
                guard let sources = state[mtpSharedKVSourceIndicesKey] else { return false }
                return sharedKV.allSatisfy { key, entry in
                    guard let leaf = sources[key] else { return false }
                    return entry.0.dim(-2) == mainCacheStorage.emittedLength(forLeaf: leaf)
                }
            }(),
            """
            stale sharedKV: spans \(sharedKV.mapValues { $0.0.dim(-2) }) do not match the \
            emitted lengths implied by a timeline of \(mainCacheStorage.processedTokenCount)
            """
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
        verifyState[mtpCacheCheckpointIndexKey] = nativeHybridRewind ? 1 : nil
        let verifyCache = nativeHybridRewind ? mainCache : round!.caches
        let mainResult = mainModel(
            verifyInput[text: .newAxis], cache: verifyCache, state: verifyState)
        let mainLogits = mainResult.logits
        mainState = mainResult.state

        eval(flatDraftTokens)
        let draftTokensList = flatDraftTokens.asArray(Int.self)

        var accepted = 0
        var finalToken: MLXArray?
        for i in 0 ..< numDraft {
            var logits = mainLogits[0..., verifyStart + i, 0...]
            logits = processor?.process(logits: logits) ?? logits
            let targetToken = sampler.sample(logits: logits)
            eval(targetToken)
            let targetTokenValue = targetToken.item(Int.self)
            processor?.didSample(token: targetToken)
            pendingTokens.append(targetTokenValue)
            guard targetTokenValue == draftTokensList[i] else {
                finalToken = targetToken
                break
            }
            accepted += 1
        }

        // Only the all-accepted path samples the bonus row. On rejection the
        // mismatching target sample above is already the emitted correction.
        if finalToken == nil {
            var logits = mainLogits[0..., verifyStart + accepted, 0...]
            logits = processor?.process(logits: logits) ?? logits
            let bonus = sampler.sample(logits: logits)
            eval(bonus)
            processor?.didSample(token: bonus)
            pendingTokens.append(bonus.item(Int.self))
            finalToken = bonus
        }
        let emittedFinalToken = finalToken!
        committedPendingTokenCount = accepted

        proposedCount += numDraft
        acceptedCount += accepted
        lastRoundAccepted = accepted
        telemetry.recordRound(
            drafted: numDraft,
            accepted: accepted,
            targetVerified: numDraft + 1,
            draftModelCalls: 1
        )

        if let statefulDrafter = drafter as? any StatefulMTPDrafterModel,
            var currentDrafterState = drafterState
        {
            statefulDrafter.commitDrafterState(
                target: mainModel,
                targetHidden: mainResult.state?[mtpLastHiddenStatesKey] ?? lastHidden,
                draftTokens: draftTokens,
                acceptedCount: accepted,
                finalToken: emittedFinalToken,
                positionDeltas: mainResult.state?[mtpPositionDeltasKey],
                state: &currentDrafterState,
                sampler: sampler)
            drafterState = currentDrafterState
        }

        let rejected = numDraft - accepted
        let snapshotPlaced: Bool
        if nativeHybridRewind {
            if rejected == 0 {
                mainCacheStorage.commitProcessedTokens(verifyInput.cacheSequenceLength)
            } else {
                let rewound = rewindSpeculativePromptCache(
                    mainCache, numTokens: rejected)
                precondition(
                    rewound == rejected,
                    "Target advertised native speculative rewind depth \(nativeRewindDepth), but rewound \(rewound) of \(rejected) positions"
                )
                // Qwen MTP-1: attention KV trims one token while every GDN
                // cache restores the state captured after the committed bonus.
                // The target's 9B weights are not replayed on rejection.
                mainCacheStorage.commitProcessedTokens(accepted + 1)
            }
            snapshotPlaced = reconcileSharedKVState(
                &mainState, discarding: rejected,
                lengths: mainCacheStorage.emittedLength(forLeaf:))
        } else {
            // Staged layers never held rejected rows; write-through layers clamp back over them.
            let commit = mainCacheStorage.commit(round!, retaining: accepted + 1)
            snapshotPlaced = reconcileSharedKVState(
                &mainState, discarding: rejected,
                lengths: { commit.emittedLengths[$0] })
        }

        guard snapshotPlaced else {
            switchToPassthrough(reason: Self.missingSharedKVSourcesReason)
            mainState = nil
            y = .init(tokens: emittedFinalToken)
            return
        }

        // Dynamic cache quantization may convert regular K/V to quantized K/V,
        // at which point the target's emit-hook cannot provide full_attention
        // shared K/V and the next round transitions to passthrough.
        kvCachePlan.apply(to: mainCacheStorage)

        y = .init(tokens: emittedFinalToken)
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
        discardSpeculativePromptCacheCheckpoints(mainCache)
        mainState?[mtpEmitFlagKey] = false
        mainState?[mtpCacheCheckpointIndexKey] = nil
        mainState?[mtpLastHiddenStatesKey] = nil
        mainState?[mtpSharedKVStatesKey] = nil
        mainState?[mtpSharedKVSourceIndicesKey] = nil
        mainState?[mtpSharedKVOffsetsKey] = nil
        mainState?[mtpPositionDeltasKey] = nil
    }

    /// One single-token forward step against the main model, used in
    /// passthrough mode. The drafter is not invoked.
    private mutating func passthroughStep() -> Int? {
        if let maxTokens, tokenCount >= maxTokens { return nil }

        let result = mainModel(y[text: .newAxis], cache: mainCache, state: mainState)
        mainState = result.state
        mainCacheStorage.commitProcessedTokens(y.cacheSequenceLength)
        var logits = result.logits[0..., -1, 0...]
        logits = processor?.process(logits: logits) ?? logits
        let token = sampler.sample(logits: logits)
        processor?.didSample(token: token)
        eval(token)
        let tokenInt = token.item(Int.self)
        y = .init(tokens: token)
        kvCachePlan.apply(to: mainCacheStorage)
        return tokenInt
    }

    public mutating func next() -> Int? {
        if let maxTokens, tokenCount >= maxTokens {
            return nil
        }

        // Drain the pending buffer first — before the passthrough branch. A
        // stand-down at the end of `init` leaves the prepare-time bonus
        // sitting here, already committed to the cache (`y` is the token
        // after it). Short-circuiting to `passthroughStep()` would drop it and
        // start the stream one position ahead of an equivalent autoregressive
        // run. Rounds that flip passthrough mid-stream always clear the buffer
        // first, so this reordering is a no-op for them.
        if pendingIndex < pendingTokens.count {
            let token = pendingTokens[pendingIndex]
            pendingIndex += 1
            telemetry.recordGeneratedToken()
            return token
        }

        if passthrough {
            if let token = passthroughStep() {
                telemetry.recordGeneratedToken()
                return token
            }
            return nil
        }

        // Run a new speculation round (may transition to passthrough).
        pendingTokens.removeAll(keepingCapacity: true)
        pendingIndex = 0
        committedPendingTokenCount = 0
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

extension MTPSpeculativeTokenIterator: GenerationFinalizingTokenIterator {
    mutating func finalizeGeneration() {
        // A fully consumed all-accepted round can still retain the recurrent
        // checkpoint used for early-finalization rollback. Release it even
        // when no committed lookahead remains.
        defer { discardSpeculativePromptCacheCheckpoints(mainCache) }
        let consumed = Swift.min(pendingIndex, committedPendingTokenCount)
        let lookahead = committedPendingTokenCount - consumed
        guard lookahead > 0 else { return }

        let usesNativeHybridRewind =
            ((mainModel as? any SpeculativeCacheRewindModel)?
                .maximumNativeTargetCacheRewind ?? 0) >= blockSize - 1
            && mainCache.contains { $0 is MambaCache }
            && mainCache.allSatisfy { $0.isTrimmable || $0 is MambaCache }
        let rewound: Int
        if usesNativeHybridRewind {
            let trimmed = mainCacheStorage.trim(lookahead)
            rewound = trimmed > 0 ? trimmed : mainCacheStorage.rewindSpeculative(lookahead)
        } else {
            // A staged round keeps what a wrapped ring needs to undo committed lookahead exactly.
            rewound = mainCacheStorage.rewindLastRound(lookahead)
        }
        // The only site that ignores the result, and the only one that can: the stream is over,
        // so nothing reads the snapshot after this. Everywhere else a refusal stops speculation.
        reconcileSharedKVState(
            &mainState, discarding: rewound,
            lengths: mainCacheStorage.emittedLength(forLeaf:))
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

/// Bring the emitted MTP shared-K/V snapshot into line with what the cache actually kept.
///
/// The verify pass emits K/V spanning its whole chunk, materialized before acceptance is known,
/// and a sliding layer is presented more than its window on purpose so every query in the chunk
/// still sees a full one. The drafter has no cache of its own and attends over this snapshot
/// bidirectionally, so both excesses have to go: the rejected tail, then the head beyond what the
/// layer can still describe.
///
/// Order matters. Dropping the tail first and clamping the head second leaves a sliding entry
/// with a full window; clamping first would leave it `discarding` rows short, every round.
///
/// - Parameters:
///   - discarding: trailing positions the commit did not keep.
///   - lengths: for the cache entry behind a given key, how much of the emitted sequence it still
///     describes — the whole stream for a global layer, the trailing window for a sliding one.
///
/// No-op when `state` is nil or carries no snapshot (the quantization-onset round emits none).
/// Returns `false` when the snapshot cannot be placed against cache entries at all, which is a
/// signal to stop speculating rather than reconcile against a guess. Cost is metadata-only: every
/// slice is a lazy view consumed by the next `draftBlock` like the rest of the round's inputs.
@discardableResult
func reconcileSharedKVState(
    _ state: inout LMOutput.State?, discarding: Int, lengths: (Int) -> Int
) -> Bool {
    guard let sharedKV = state?[mtpSharedKVStatesKey] else { return true }
    guard let sources = state?[mtpSharedKVSourceIndicesKey],
        sharedKV.keys.allSatisfy({ sources[$0] != nil })
    else { return false }

    state?[mtpSharedKVStatesKey] = sharedKV.reduce(into: [:]) { result, entry in
        let (key, kv) = entry
        var length = kv.0.dim(-2)
        if discarding > 0 {
            length = Swift.max(0, length - discarding)
        }
        let bound = lengths(sources[key]!)
        let start = Swift.max(0, length - bound)
        result[key] = (
            kv.0[.ellipsis, start ..< length, 0...],
            kv.1[.ellipsis, start ..< length, 0...]
        )
    }
    if discarding > 0, let offsets = state?[mtpSharedKVOffsetsKey] {
        state?[mtpSharedKVOffsetsKey] = offsets.mapValues {
            Swift.max(0, $0 - discarding)
        }
    }
    return true
}
