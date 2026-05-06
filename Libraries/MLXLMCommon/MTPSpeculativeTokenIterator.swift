//
//  MTPSpeculativeTokenIterator.swift
//  mlx-swift-lm
//
//  Speculative-decoding iterator for Multi-Token Prediction (MTP) drafters.
//  Companion to ``SpeculativeTokenIterator``: that one drives a *generic*
//  draft model with its own KV cache; this one drives a tightly-coupled MTP
//  drafter that shares the target's K/V and last-hidden state.
//
//  Reference: https://ai.google.dev/gemma/docs/mtp/overview
//

import Foundation
import MLX
import MLXNN

/// Protocol any MTP-compatible target must implement to expose drafter hooks.
///
/// The drafter borrows the target's input embedding table and scales, reads
/// K/V from the target's last full- and sliding-attention layers, and consumes
/// the target's last-layer hidden state. This protocol exposes only the bits
/// the drafter / iterator need so the wider model API stays untouched.
public protocol MTPTargetModel {
    /// Backbone hidden size (drafter's pre/post projection target dim).
    var backboneHiddenSize: Int { get }

    /// Sliding-window size for sliding-attention layers.
    var slidingWindow: Int { get }

    /// Index in ``newCache(parameters:)`` of the last full-attention layer
    /// whose K/V the drafter should read.
    var lastFullAttentionLayerIndex: Int? { get }

    /// Index in ``newCache(parameters:)`` of the last sliding-attention layer.
    var lastSlidingAttentionLayerIndex: Int? { get }

    /// Target's input embedding table (the drafter borrows it for token embeds).
    var inputEmbeddings: Embedding { get }

    /// Target's embedding scale (Gemma scales input embeddings by sqrt(hidden)).
    var inputEmbedScale: Float { get }

    /// Forward pass returning both logits and the post-norm last-layer hidden
    /// state. The hidden state is required by the MTP drafter.
    func forwardWithHidden(_ inputs: MLXArray, cache: [KVCache]?) -> (
        logits: MLXArray, hidden: MLXArray
    )

    /// Allocate a new KV cache (per-layer). Inherited from ``LanguageModel``.
    func newCache(parameters: GenerateParameters?) -> [any KVCache]
}

/// Protocol any MTP drafter must implement so the iterator can drive it
/// without depending on a specific drafter family.
public protocol MTPDrafterModel {
    /// Wire target's input embeddings + scale into the drafter. Must be called
    /// before ``draftBlock(...)``.
    func bindMTP(target: any MTPTargetModel)

    /// Generate `blockSize - 1` candidate tokens (greedy).
    ///
    /// - Parameters:
    ///   - bonusToken: most recently accepted target token (verify-input
    ///     position 0; not drafted here).
    ///   - targetLastHidden: target's hidden state at the position before the
    ///     bonus token, shape `[1, 1, backboneHiddenSize]`.
    ///   - sharedKV: target's last full-/sliding-attention K/V keyed by layer
    ///     type ("full_attention" / "sliding_attention").
    ///   - positionOffset: bonus token's absolute position; held constant
    ///     across all draft steps within the block.
    ///   - blockSize: total verify-block size; produces `blockSize - 1`
    ///     candidates.
    func draftBlock(
        bonusToken: Int,
        targetLastHidden: MLXArray,
        sharedKV: [String: (MLXArray, MLXArray)],
        positionOffset: Int,
        blockSize: Int
    ) -> [Int]
}

/// Generator of tokens using Multi-Token Prediction (MTP) speculative decoding.
///
/// Single-batch (B=1), greedy (temperature 0) MVP. Output is byte-identical
/// to the no-drafter target generation per Google's MTP guarantees.
///
/// Round structure (per Python reference):
/// 1. Read target's last full-/sliding-attention K/V from cache.
/// 2. Drafter generates `blockSize - 1` candidates from `(bonusToken,
///    lastHidden)`, with RoPE position held constant at the bonus token's
///    absolute position.
/// 3. Target verifies `[bonus, c0, c1, ..., c_{K-1}]` in a single forward,
///    yielding fresh logits + hidden states at each of the `blockSize`
///    positions.
/// 4. Greedy compare: accept candidates up to first mismatch; emit the
///    target's correction (or last-position prediction if all accepted).
/// 5. Trim cache for rejected verify tokens; advance bonus / hidden /
///    position for the next round.
public struct MTPSpeculativeTokenIterator: TokenIteratorProtocol {

    let target: any MTPTargetModel
    let drafter: any MTPDrafterModel
    var cache: [KVCache]
    let blockSize: Int
    let lastFullIdx: Int
    let lastSlidingIdx: Int

    var bonusToken: Int
    var lastHidden: MLXArray
    var positionOffset: Int

    let maxTokens: Int?
    var tokenCount: Int = 0

    private var pendingTokens: [Int] = []
    private var pendingIndex: Int = 0

    /// Internal metric: prompt prefill time (seconds).
    var promptPrefillTime: TimeInterval = 0.0

    /// Initialize the iterator: prefill the target with the prompt and prime
    /// the drafter binding.
    ///
    /// - Parameters:
    ///   - input: prompt input.
    ///   - target: ``MTPTargetModel`` (e.g. ``Gemma4Model``).
    ///   - drafter: matching ``MTPDrafterModel`` (e.g.
    ///     ``Gemma4AssistantDraftModel``).
    ///   - cache: optional target KV cache (allocated fresh if nil).
    ///   - parameters: generation parameters (only `maxTokens` is consumed).
    ///   - blockSize: number of tokens verified per round (drafter generates
    ///     `blockSize - 1` candidates).
    public init(
        input: LMInput,
        target: any MTPTargetModel,
        drafter: any MTPDrafterModel,
        cache: [KVCache]? = nil,
        parameters: GenerateParameters,
        blockSize: Int = 4
    ) throws {
        self.target = target
        self.drafter = drafter
        self.blockSize = blockSize
        self.maxTokens = parameters.maxTokens

        guard let lastFull = target.lastFullAttentionLayerIndex,
            let lastSliding = target.lastSlidingAttentionLayerIndex
        else {
            throw KVCacheError(
                message:
                    "MTP requires target with both full-attention and sliding-attention layers"
            )
        }
        self.lastFullIdx = lastFull
        self.lastSlidingIdx = lastSliding

        drafter.bindMTP(target: target)

        let allocatedCache = cache ?? target.newCache(parameters: parameters)
        guard canTrimPromptCache(allocatedCache) else {
            throw KVCacheError(message: "MTP requires trimmable KV caches.")
        }
        self.cache = allocatedCache

        let start = Date()
        var promptTokens = input.text.tokens
        if promptTokens.ndim == 1 {
            promptTokens = promptTokens.expandedDimensions(axis: 0)
        }
        let (logits, hidden) = target.forwardWithHidden(promptTokens, cache: self.cache)
        let lastIdx = promptTokens.dim(-1) - 1
        let firstSampleLogits = logits[0..., lastIdx, 0...]
        let firstSample = argMax(firstSampleLogits, axis: -1).reshaped([1])
        asyncEval(firstSample)
        let firstSampleId = firstSample.item(Int.self)

        self.bonusToken = firstSampleId
        self.lastHidden = hidden[0..., lastIdx ... lastIdx, 0...]
        self.positionOffset = self.cache[0].offset

        self.promptPrefillTime = -start.timeIntervalSinceNow

        // First emitted token is the bonus sampled from the prompt.
        self.pendingTokens.append(firstSampleId)
    }

    /// Pull target's last full-/sliding-attention K/V from cache.
    private func extractSharedKV() -> [String: (MLXArray, MLXArray)] {
        var dict: [String: (MLXArray, MLXArray)] = [:]
        let fullState = cache[lastFullIdx].state
        if fullState.count >= 2 {
            dict["full_attention"] = (fullState[0], fullState[1])
        }
        let slidingState = cache[lastSlidingIdx].state
        if slidingState.count >= 2 {
            dict["sliding_attention"] = (slidingState[0], slidingState[1])
        }
        return dict
    }

    mutating private func speculateRound() {
        let remaining = maxTokens.map { $0 - tokenCount } ?? blockSize
        guard remaining > 0 else { return }
        let effectiveBlock = Swift.min(remaining + 1, blockSize)
        if effectiveBlock < 2 {
            // No drafting possible; fall back to a single target step.
            let stepArr = MLXArray([Int32(bonusToken)]).reshaped([1, 1])
            let (sLogits, _) = target.forwardWithHidden(stepArr, cache: cache)
            let nextTok = argMax(sLogits[0..., -1, 0...], axis: -1).reshaped([1])
            asyncEval(nextTok)
            let nextId = nextTok.item(Int.self)
            pendingTokens.append(nextId)
            // After this single-step, target cache has consumed bonus; new
            // bonus = nextId, but its hidden is unknown without a 2nd forward.
            // Subsequent round won't run (remaining hits 0), so we don't need
            // to update lastHidden.
            bonusToken = nextId
            positionOffset = cache[0].offset
            return
        }

        let sharedKV = extractSharedKV()

        let candidates = drafter.draftBlock(
            bonusToken: bonusToken,
            targetLastHidden: lastHidden,
            sharedKV: sharedKV,
            positionOffset: positionOffset,
            blockSize: effectiveBlock
        )

        // Verify input: [bonus, c0, ..., c_{K-2}] of length effectiveBlock
        var verifyInts: [Int32] = [Int32(bonusToken)]
        verifyInts.append(contentsOf: candidates.map { Int32($0) })
        let verifyArr = MLXArray(verifyInts).reshaped([1, effectiveBlock])

        // Run the verify forward on a dedicated stream — mirrors mlx-vlm's
        // `with mx.stream(generation_stream):` around `lm(verify_input, ...)`.
        // Without this, MLX-Swift's default-stream evaluation interleaves verify
        // ops with drafter ops in a way that produces ε-different SDPA / matmul
        // reductions vs the no-drafter baseline (which runs single-token forwards
        // on the default stream alone), causing argmax flips on narrow-margin
        // tokens.
        let (vLogits, vHidden) = Stream.withNewDefaultStream {
            target.forwardWithHidden(verifyArr, cache: cache)
        }
        // vLogits: [1, effectiveBlock, V], vHidden: [1, effectiveBlock, H]

        // Greedy: argmax over vocab dim.
        let mainTokensArr = argMax(vLogits, axis: -1)  // [1, effectiveBlock]
        asyncEval(mainTokensArr)
        let mainTokens = mainTokensArr.reshaped([effectiveBlock]).asArray(Int.self)

        var accepted = 0
        for i in 0 ..< (effectiveBlock - 1) {
            if mainTokens[i] == candidates[i] {
                pendingTokens.append(mainTokens[i])
                accepted += 1
            } else {
                break
            }
        }
        let newBonus = mainTokens[accepted]
        pendingTokens.append(newBonus)

        let toTrim = (effectiveBlock - 1) - accepted
        if toTrim > 0 {
            trimPromptCache(cache, numTokens: toTrim)
        }

        bonusToken = newBonus
        lastHidden = vHidden[0..., accepted ... accepted, 0...]
        positionOffset = cache[0].offset
    }

    mutating public func next() -> Int? {
        if let maxTokens, tokenCount >= maxTokens {
            return nil
        }

        if pendingIndex < pendingTokens.count {
            let token = pendingTokens[pendingIndex]
            pendingIndex += 1
            tokenCount += 1
            return token
        }

        pendingTokens.removeAll(keepingCapacity: true)
        pendingIndex = 0
        speculateRound()

        if pendingTokens.isEmpty { return nil }
        let token = pendingTokens[pendingIndex]
        pendingIndex += 1
        tokenCount += 1
        return token
    }
}
