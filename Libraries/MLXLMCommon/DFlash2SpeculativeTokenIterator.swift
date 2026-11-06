// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN

/// Why a ``DFlash2SpeculativeTokenIterator`` could not be built.
public enum DFlash2SpeculationError: Error, Equatable {
    /// The target model does not conform to ``DFlash2TargetModel``.
    case unsupportedTarget
    /// The cache or the requested KV cache configuration has entries the verify
    /// pass cannot write, such as quantized or rotating attention caches.
    case unsupportedCache
    /// The drafter was distilled for a target of another shape.
    case geometryMismatch(drafter: DFlash2Geometry, target: DFlash2Geometry)
    /// DFlash2 speculation covers text-only prompts.
    case textOnly
    /// `prefilledPrefixTokens` must leave at least one prompt token.
    case promptTooShort
}

/// Token iterator for DFlash2 block speculative decoding.
///
/// Each round verifies `[anchor, draft_1, ..., draft_{S-1}]` in one target pass.
/// The next round is built while that pass runs: the accept count, the bonus
/// token and the next position stay lazy, so the host syncs once per round.
///
/// Greedy rounds accept the longest prefix that matches the target's argmax.
/// Sampled rounds use rejection sampling against the drafter's candidate
/// distribution, so emitted tokens follow the processed target distribution.
/// A verify pass sums in a different order than a one-token step, so greedy
/// output can differ from ``TokenIterator`` where the top two logits nearly tie.
public struct DFlash2SpeculativeTokenIterator: TokenIteratorProtocol {
    let target: any DFlash2TargetModel
    let drafter: any DFlash2DrafterModel

    var cache: [KVCache]
    var drafterState: DFlash2DrafterState

    var processor: LogitProcessor?
    let sampler: LogitSampler
    /// Masks suppressed ids in the draft logits; the processor masks them in the target's.
    let draftSuppressor: SuppressTokensProcessor?
    let temperature: Float
    let topP: Float
    let topK: Int
    let minP: Float
    let randomState: MLXRandom.RandomState

    public let maxTokens: Int?
    /// Tokens per verify pass: one anchor plus `blockSize - 1` drafts.
    public let blockSize: Int

    public private(set) var promptPrefillTime: TimeInterval = 0

    private var pendingTokens = [Int]()
    private var pendingIndex = 0
    /// Drafts the last round accepted, for ``finalizeGeneration()``.
    private var committedAcceptedCount = 0
    /// Tokens produced so far, drained or pending.
    private var producedTokens = 0

    /// A round whose verify pass is scheduled but whose accept is unknown.
    private struct Round {
        var drafts: MLXArray
        var candidates: MLXArray
        var probabilities: MLXArray?
        var width: Int
        var verify: DFlash2VerifyResult
    }

    private var inFlight: Round?
    /// Absolute position of the in-flight round's anchor.
    private var anchorPosition = 0
    /// Recurrent captures of the last committed round, for ``finalizeGeneration()``.
    private var committedCaptures: [GatedDeltaCapture] = []

    private var telemetry = SpeculativeDecodingTelemetry()
    public var speculativeDecodingTelemetry: SpeculativeDecodingTelemetry? {
        telemetry.roundCount > 0 ? telemetry : nil
    }
    public private(set) var acceptedCount = 0
    public private(set) var proposedCount = 0

    private var greedy: Bool { temperature <= 0 }

    /// - Parameters:
    ///   - input: the prompt, text only.
    ///   - mainModel: the target; must conform to ``DFlash2TargetModel``.
    ///   - drafter: a DFlash2 drafter distilled for `mainModel`'s shape.
    ///   - mainCache: a target cache to decode over.
    ///   - prefilledPrefixTokens: leading positions of `input` that `mainCache`
    ///     already holds; only the rest is prefilled.
    ///   - parameters: sampling and generation limits.
    ///   - blockSize: tokens per verify pass; `nil` uses the drafter's.
    ///   - components: generation hooks such as a custom logit processor.
    public init(
        input: LMInput,
        mainModel: any LanguageModel,
        drafter: any DFlash2DrafterModel,
        mainCache: [KVCache]? = nil,
        prefilledPrefixTokens: Int = 0,
        parameters: GenerateParameters,
        blockSize: Int? = nil,
        components: GenerationComponents = .init()
    ) throws {
        guard let target = mainModel as? DFlash2TargetModel else {
            throw DFlash2SpeculationError.unsupportedTarget
        }
        guard
            drafter.targetGeometry.accepts(target.dflash2Geometry, layerIds: drafter.targetLayerIds)
        else {
            throw DFlash2SpeculationError.geometryMismatch(
                drafter: drafter.targetGeometry, target: target.dflash2Geometry)
        }
        guard input.image == nil, input.video == nil, input.audio == nil else {
            throw DFlash2SpeculationError.textOnly
        }
        let promptLength = input.text.tokens.dim(-1)
        guard prefilledPrefixTokens >= 0, prefilledPrefixTokens < promptLength else {
            throw DFlash2SpeculationError.promptTooShort
        }
        // The verify pass writes plain attention rows: no compression or capacity.
        if let configuration = try parameters.kvCachePlan().configuration,
            configuration.capacity != nil || configuration.strategy != .fullPrecision
        {
            throw DFlash2SpeculationError.unsupportedCache
        }
        let cache = try mainCache ?? target.newCache(parameters: parameters)
        guard target.dflash2SupportsCache(cache) else {
            throw DFlash2SpeculationError.unsupportedCache
        }
        try components.validate(parameters: parameters)

        self.target = target
        self.drafter = drafter
        self.cache = cache
        self.drafterState = drafter.makeState()
        self.processor = components.logitProcessor(parameters: parameters, model: mainModel)
        self.draftSuppressor = makeSuppressTokensProcessor(model: mainModel)
        self.sampler = parameters.sampler()
        self.temperature = parameters.temperature
        self.topP = parameters.topP
        self.topK = parameters.topK
        self.minP = parameters.minP
        self.randomState =
            parameters.seed.map { MLXRandom.RandomState(seed: $0) } ?? MLXRandom.RandomState()
        self.maxTokens = parameters.maxTokens
        self.blockSize = Swift.max(2, blockSize ?? drafter.blockSize)

        let prefillStart = Date.timeIntervalSinceReferenceDate
        try prepare(
            input: input, prefilledPrefixTokens: prefilledPrefixTokens,
            prefill: parameters.prefill)
        self.promptPrefillTime = Date.timeIntervalSinceReferenceDate - prefillStart
    }

    // MARK: Prefill

    /// Chunked prefill that keeps the last `contextWindow` captured rows as the
    /// drafter's first context, then samples the first token and builds round 0.
    private mutating func prepare(
        input: LMInput, prefilledPrefixTokens: Int, prefill: PrefillParameters
    ) throws {
        processor?.prompt(input.text.tokens)
        prefill.prefix?(prefilledPrefixTokens)
        let promptTokens = input.text.tokens
        let promptLength = promptTokens.dim(0)
        if prefilledPrefixTokens > 0,
            let attention = cache.first(where: { type(of: $0) == KVCacheSimple.self })
        {
            precondition(
                attention.offset == prefilledPrefixTokens,
                "cache holds \(attention.offset) positions, prefilledPrefixTokens is \(prefilledPrefixTokens)"
            )
        }

        let window = drafter.contextWindow
        let captureLayers = drafter.targetLayerIds
        var hiddenWindow: MLXArray? = nil
        var hiddenOffset = prefilledPrefixTokens
        func forward(_ range: Range<Int>) -> MLXArray {
            let chunk = promptTokens[range].expandedDimensions(axis: 0)
            let result = target.dflash2Prefill(chunk, cache: cache, captureLayers: captureLayers)
            var hidden = concatenatedHidden(result.hidden)
            if let existing = hiddenWindow {
                hidden = concatenated([existing, hidden], axis: 1)
            }
            let rows = hidden.dim(1)
            if rows > window {
                hiddenOffset += rows - window
                hidden = hidden[0..., (rows - window)..., 0...]
            }
            hiddenWindow = hidden
            return result.logits
        }

        let total = promptLength - prefilledPrefixTokens
        let processed = try prefill.forEachChunk(total: total) { range in
            _ = forward(range.shifted(by: prefilledPrefixTokens))
            asyncEval(cache.flatMap { $0.innerState() } + [hiddenWindow!])
        }
        let logits = forward((prefilledPrefixTokens + processed) ..< promptLength)
        prefill.progress?(total, total)

        var last = logits[0..., -1, 0...]
        last = processor?.process(logits: last) ?? last
        let token = sampler.sample(logits: last)
        processor?.didSample(token: token)
        eval(token)
        pendingTokens.append(token.item(Int.self))
        producedTokens = 1
        anchorPosition = promptLength

        // The prompt rows are committed context as soon as round 0 appends them.
        let context = hiddenWindow!
        let contextRows = buildRound(
            anchor: token.reshaped([1]),
            validRows: MLXArray(Int32(context.dim(1))),
            targetHidden: context,
            contextPosition: hiddenOffset,
            position: MLXArray([Int32(promptLength)]),
            positionUpperBound: promptLength,
            producedAtLeast: 1)
        for contextCache in drafterState.contextCaches {
            contextCache.resolve(newest: contextRows, valid: contextRows)
        }
    }

    private func concatenatedHidden(_ layers: [MLXArray]) -> MLXArray {
        layers.count == 1 ? layers[0] : concatenated(layers, axis: -1)
    }

    // MARK: Rounds

    /// Drafts in the next round, fewer near `maxTokens`. `producedAtLeast` is
    /// the fewest tokens produced before that round runs.
    private func draftCount(producedAtLeast: Int) -> Int? {
        guard let maxTokens else { return blockSize - 1 }
        let remaining = maxTokens - producedAtLeast
        return remaining > 0 ? Swift.min(blockSize - 1, remaining) : nil
    }

    /// Schedule one round from possibly lazy inputs: the proposal, then the
    /// verify pass over it. Returns the context rows appended to the drafter.
    @discardableResult
    private mutating func buildRound(
        anchor: MLXArray, validRows: MLXArray, targetHidden: MLXArray,
        contextPosition: Int, position: MLXArray, positionUpperBound: Int,
        producedAtLeast: Int
    ) -> Int {
        guard let drafts = draftCount(producedAtLeast: producedAtLeast) else {
            inFlight = nil
            return 0
        }
        let masks = MLXArray(Array(repeating: Int32(drafter.maskTokenId), count: drafts))
        let block = concatenated([anchor.asType(.int32), masks]).expandedDimensions(axis: 0)
        let proposal = drafter.propose(
            DFlash2ProposalRequest(
                block: block, targetHidden: targetHidden, contextPosition: contextPosition,
                validRows: validRows, temperature: temperature, suppressedTokens: draftSuppressor),
            target: target, state: &drafterState)
        let draftTokens = proposal.tokens.flattened().asType(.int32)
        let request = DFlash2VerifyRequest(
            tokens: concatenated([anchor.asType(.int32), draftTokens]).expandedDimensions(axis: 0),
            position: position,
            positionUpperBound: positionUpperBound,
            captureLayers: drafter.targetLayerIds)
        let verify = target.dflash2Verify(request, cache: cache)
        asyncEval(verify.logits)
        inFlight = Round(
            drafts: draftTokens, candidates: proposal.candidates,
            probabilities: proposal.probabilities, width: drafts + 1, verify: verify)
        return targetHidden.dim(1)
    }

    /// Accept the in-flight round, build the next one while its verify pass
    /// still runs, then sync once and commit.
    private mutating func speculateRound() {
        guard let round = inFlight else { return }
        let gamma = round.width - 1

        // Accept, lazily. Drafts feed a copy of the processor row by row.
        let logits = processedRows(round.verify.logits[0], drafts: round.drafts)
        let acceptance: Acceptance
        if greedy {
            acceptance = Self.greedyAcceptance(drafts: round.drafts, logits: logits)
        } else {
            acceptance = sampledAcceptance(round: round, logits: logits)
        }
        asyncEval(acceptance.packed)

        let validCount = acceptance.accepted + 1
        commitRecurrentState(round.verify.recurrentCaptures, validCount: validCount)

        let appendedRows = buildRound(
            anchor: acceptance.bonus,
            validRows: validCount,
            targetHidden: concatenatedHidden(round.verify.hidden),
            contextPosition: anchorPosition,
            position: MLXArray([Int32(anchorPosition)]) + validCount.asType(.int32),
            positionUpperBound: anchorPosition + round.width,
            producedAtLeast: producedTokens + 1)

        // The round's one host sync.
        let packed = acceptance.packed.asArray(Int32.self)
        let drafts = packed[0 ..< gamma].map(Int.init)
        let accepted = Int(packed[gamma])
        let bonus = Int(packed[gamma + 1])

        let committed = anchorPosition + accepted + 1
        for entry in cache {
            if let recurrent = entry as? MambaCache {
                recurrent.offset = committed
            } else if let attention = entry as? KVCacheSimple {
                attention.commitRows(count: committed)
            }
        }
        for contextCache in drafterState.contextCaches {
            contextCache.resolve(newest: appendedRows, valid: accepted + 1)
        }
        for token in drafts.prefix(accepted) {
            processor?.didSample(token: MLXArray(token))
        }
        processor?.didSample(token: MLXArray(bonus))

        var newTokens = Array(drafts.prefix(accepted))
        newTokens.append(bonus)
        if let maxTokens {
            newTokens = Array(newTokens.prefix(Swift.max(0, maxTokens - tokenCount)))
        }
        pendingTokens.append(contentsOf: newTokens)
        committedAcceptedCount = accepted
        producedTokens += accepted + 1
        anchorPosition = committed
        committedCaptures = round.verify.recurrentCaptures

        proposedCount += gamma
        acceptedCount += accepted
        telemetry.recordRound(
            drafted: gamma, accepted: accepted, targetVerified: gamma + 1, draftModelCalls: 1)
    }

    /// `[S, vocab]` verify logits through the processor, row by row on a copy
    /// that has seen the drafts before each row. Row `i` matters only when
    /// drafts `0 ..< i` were accepted, so that is the history it sees.
    private func processedRows(_ logits: MLXArray, drafts: MLXArray) -> MLXArray {
        guard let processor else { return logits }
        var scratch = processor.copy()
        var rows: [MLXArray] = []
        for i in 0 ..< logits.dim(0) {
            rows.append(scratch.process(logits: logits[i ..< (i + 1)]))
            if i < drafts.dim(0) {
                scratch.didSample(token: drafts[i ..< (i + 1)])
            }
        }
        return concatenated(rows, axis: 0)
    }

    /// Give every gated-delta layer the state it would hold after the first
    /// `validCount` positions of the pass. Exact for any count, so it is built
    /// before the count is known.
    private func commitRecurrentState(_ captures: [GatedDeltaCapture], validCount: MLXArray) {
        let recurrentCaches = cache.compactMap { $0 as? MambaCache }
        precondition(
            recurrentCaches.count == captures.count,
            "\(captures.count) recurrent captures for \(recurrentCaches.count) recurrent caches")
        for (recurrent, capture) in zip(recurrentCaches, captures) {
            let replayed = capture.replay(validCount: validCount)
            recurrent[0] = replayed.conv
            recurrent[1] = replayed.recurrent
        }
    }

    // MARK: Acceptance

    /// The accept outcome as lazy arrays, plus `[drafts..., accepted, bonus]`
    /// packed for the round's one sync.
    struct Acceptance {
        var accepted: MLXArray
        var bonus: MLXArray
        var packed: MLXArray
    }

    /// Accept the longest prefix that matches the target's argmax; the bonus is
    /// the argmax at the first mismatch.
    private static func greedyAcceptance(drafts: MLXArray, logits: MLXArray) -> Acceptance {
        let gamma = drafts.dim(0)
        let targets = argMax(logits, axis: -1).asType(.int32)
        let matches = (drafts .== targets[0 ..< gamma]).asType(.int32)
        let accepted = cumprod(matches).sum()
        let bonus = takeAlong(targets, accepted.reshaped([1]), axis: 0)
        return Acceptance(
            accepted: accepted, bonus: bonus,
            packed: concatenated([drafts, accepted.reshaped([1]), bonus]))
    }

    /// Rejection sampling against the drafter's candidate distribution, as the
    /// reference `_rejection_sample` does.
    private func sampledAcceptance(round: Round, logits: MLXArray) -> Acceptance {
        guard let probabilities = round.probabilities else {
            preconditionFailure("sampled proposal without selector probabilities")
        }
        let targetProbs = speculativeSamplingProbabilities(
            logits, temperature: temperature, topP: topP, minP: minP, topK: topK)
        return withRandomState(randomState) {
            Self.rejectionSample(
                drafts: round.drafts, targetProbs: targetProbs,
                draftProbs: probabilities[0], candidates: round.candidates[0])
        }
    }

    static func rejectionSample(
        drafts: MLXArray, targetProbs: MLXArray, draftProbs: MLXArray, candidates: MLXArray
    ) -> Acceptance {
        let gamma = drafts.dim(0)
        // p: the target's probability of each draft; q: the drafter's.
        let p = takeAlong(targetProbs[0 ..< gamma], drafts[0..., .newAxis], axis: -1)[0..., 0]
        let drafted = (candidates .== drafts[0..., .newAxis]).asType(.float32)
        let q = (draftProbs * drafted).sum(axis: -1)
        let accept = (uniform(0 ..< 1, [gamma]) * q) .< p
        let accepted = cumprod(accept.asType(.int32)).sum()

        // Past a fully accepted block the bonus is the target's own sample. At a
        // rejection it is drawn from `max(p - q, 0)`, or from `p` if that is empty.
        let targetRow = takeAlong(targetProbs, accepted.reshaped([1, 1]), axis: 0)[0]
        let rejectedRow = minimum(accepted, MLXArray(Int32(gamma - 1))).reshaped([1, 1])
        let candidateRow = takeAlong(candidates, rejectedRow, axis: 0)[0]
        let draftRow = takeAlong(draftProbs, rejectedRow, axis: 0)[0]
        var residual = putAlong(
            targetRow, candidateRow,
            values: takeAlong(targetRow, candidateRow, axis: -1) - draftRow, axis: -1)
        residual = maximum(residual, MLXArray(Float(0)))
        let total = residual.sum()
        residual = MLX.where(
            total .> 0, residual / maximum(total, MLXArray(Float(1e-30))), targetRow)
        let bonusDistribution = MLX.where(accepted .== Int32(gamma), targetRow, residual)
        let bonus = categorical(log(bonusDistribution)).asType(.int32).reshaped([1])
        return Acceptance(
            accepted: accepted, bonus: bonus,
            packed: concatenated([drafts, accepted.reshaped([1]), bonus]))
    }

    // MARK: TokenIteratorProtocol

    public var tokenCount: Int { telemetry.emittedTokenCount }

    public var reasoningTokenCount: Int? {
        (processor as? any GenerationReasoningTokenCounting)?.generationReasoningTokenCount
    }

    public mutating func recordEmittedToken(_ token: Int) {
        processor?.didEmit(token: token)
    }

    public mutating func discardGeneratedToken() {
        telemetry.discardGeneratedToken()
    }

    public mutating func next() -> Int? {
        if let maxTokens, tokenCount >= maxTokens {
            return nil
        }
        if pendingIndex == pendingTokens.count {
            pendingTokens.removeAll(keepingCapacity: true)
            pendingIndex = 0
            committedAcceptedCount = 0
            autoreleasepool { speculateRound() }
            if pendingTokens.isEmpty {
                return nil
            }
        }
        // Return freed buffers after prefill and every 256 tokens, as TokenIterator does.
        if tokenCount % 256 == 0 {
            Memory.clearCache()
        }
        let token = pendingTokens[pendingIndex]
        pendingIndex += 1
        telemetry.recordGeneratedToken()
        return token
    }
}

extension DFlash2SpeculativeTokenIterator: GenerationFinalizingTokenIterator {
    /// Rewind the cache to the tokens the consumer drained. Pending draft `j`
    /// sits at verify position `j + 1`; the bonus has no cache row yet.
    public mutating func finalizeGeneration() {
        defer { processor?.finalizeGeneration() }
        let kept = Swift.min(pendingIndex, committedAcceptedCount)
        let rewind = committedAcceptedCount - kept
        guard rewind > 0 else { return }
        for entry in cache {
            if let recurrent = entry as? MambaCache {
                recurrent.offset -= rewind
            } else if entry.isTrimmable {
                entry.trim(rewind)
            }
        }
        commitRecurrentState(committedCaptures, validCount: MLXArray(Int32(kept + 1)))
        anchorPosition -= rewind
        committedAcceptedCount = kept
    }
}

extension DFlash2SpeculativeTokenIterator: MTPStatsCollecting {
    public var proposedDraftTokens: Int { proposedCount }
    public var acceptedDraftTokens: Int { acceptedCount }
    public var passthroughReason: String? { nil }
}

// MARK: - Sampling distribution

/// The distribution a sampled round accepts against: ``TopPSampler``'s filters
/// in its order (top-p, min-p, top-k on log-probabilities, then temperature).
func speculativeSamplingProbabilities(
    _ logits: MLXArray, temperature: Float, topP: Float, minP: Float, topK: Int
) -> MLXArray {
    var logprobs = logSoftmax(logits.asType(.float32), axis: -1)
    let negInf = MLXArray(-Float.infinity)
    if topP > 0, topP < 1 {
        let sortedIndices = argSort(logprobs, axis: -1)
        let sortedLogprobs = takeAlong(logprobs, sortedIndices, axis: -1)
        let cumulative = cumsum(exp(sortedLogprobs), axis: -1)
        let filtered = MLX.where(cumulative .> (1 - topP), sortedLogprobs, negInf)
        logprobs = putAlong(logprobs, sortedIndices, values: filtered, axis: -1)
    }
    if minP > 0 {
        let threshold = logprobs.max(axis: -1, keepDims: true) + log(MLXArray(minP))
        logprobs = MLX.where(logprobs .>= threshold, logprobs, negInf)
    }
    if topK > 0, topK < logprobs.dim(-1) {
        let masked = argPartition(-logprobs, kth: topK - 1, axis: -1)[0..., topK...]
        logprobs = putAlong(logprobs, masked, values: negInf, axis: -1)
    }
    return softmax(logprobs * (1 / temperature), axis: -1)
}

extension Range where Bound == Int {
    fileprivate func shifted(by offset: Int) -> Range<Int> {
        (lowerBound + offset) ..< (upperBound + offset)
    }
}
