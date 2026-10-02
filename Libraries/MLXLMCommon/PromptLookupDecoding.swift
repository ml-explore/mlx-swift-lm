// Copyright © 2026 Apple Inc.

import Foundation
import MLX

/// Bounded, model-free speculation from token frequencies in the recent context.
///
/// The longest matching n-gram proposes its most frequent continuation when it meets both
/// thresholds. Each proposed token extends the context for the next one, so a draft chains
/// frequent continuations and need not appear verbatim in the history. The target model
/// verifies every proposal. No auxiliary weights are loaded.
public struct PromptLookupConfiguration: Sendable, Equatable {
    /// Maximum proposed tokens per verification, excluding the carried target token.
    public var maxDraftTokens: Int
    /// Longest context key, from one through four tokens.
    public var maxNGramLength: Int
    /// Shortest context key to consider.
    public var minNGramLength: Int
    /// Maximum number of recent tokens indexed, independently of the model's KV capacity.
    ///
    /// The index allocates about 100 bytes per token and n-gram order up front: about 6 MiB
    /// with the defaults and about 390 MiB at the largest size and order.
    public var contextSize: Int
    /// Minimum observed repetitions of the proposed context and continuation together.
    public var minimumOccurrences: Int
    /// Minimum fraction of a context's occurrences followed by the proposed token.
    public var minimumConfidence: Float
    /// Adjust the draft length and periodically skip lookups after repeated misses or rejections.
    public var adaptive: Bool

    public init(
        maxDraftTokens: Int = 4, maxNGramLength: Int = 4, minNGramLength: Int = 1,
        contextSize: Int = 16_384, minimumOccurrences: Int = 1,
        minimumConfidence: Float = 0.5, adaptive: Bool = true
    ) {
        self.maxDraftTokens = maxDraftTokens
        self.maxNGramLength = maxNGramLength
        self.minNGramLength = minNGramLength
        self.contextSize = contextSize
        self.minimumOccurrences = minimumOccurrences
        self.minimumConfidence = minimumConfidence
        self.adaptive = adaptive
    }

    func validate() throws {
        guard (1 ... 256).contains(maxDraftTokens),
            (1 ... 4).contains(maxNGramLength),
            (1 ... maxNGramLength).contains(minNGramLength),
            (maxNGramLength + 1 ... 1_048_576).contains(contextSize),
            minimumOccurrences > 0, minimumOccurrences <= contextSize,
            minimumConfidence.isFinite, (0 ... 1).contains(minimumConfidence)
        else { throw PromptLookupError.invalidConfiguration }
    }
}

/// Invalid input to prompt-lookup generation. Validation precedes cache mutation.
public enum PromptLookupError: Error, LocalizedError {
    case invalidConfiguration
    case invalidInput

    public var errorDescription: String? {
        switch self {
        case .invalidConfiguration:
            "Prompt lookup requires 1...256 draft tokens, n-gram lengths in 1...4, "
                + "a context of maxNGramLength + 1...1048576 tokens, a positive occurrence "
                + "threshold within that context, and confidence in 0...1."
        case .invalidInput:
            "Prompt lookup requires one nonempty token sequence."
        }
    }
}

/// Generates tokens using a bounded n-gram index and staged target verification.
///
/// Sampling always uses the target distribution. Greedy verification without processors
/// performs one batched argmax and readback; other modes apply processors and sample in order.
/// Batched model kernels can differ numerically from single-token kernels near tied logits.
///
/// Attention caches use staged rounds, including wrapped sliding windows. Unsupported recurrent
/// caches and media inputs continue through ordinary decoding. Rounds stop at a dynamic
/// compression threshold, so each position sees the cache representation ordinary decoding
/// would use. As with ``TokenIterator``, do not advance copies of an iterator concurrently:
/// they share the model and cache storage. If you stop iterating before `next()` returns
/// `nil` and keep the cache, call ``finish()`` first.
///
/// > Important: Verification passes the model state through every proposed token, and a
/// > rejection rewinds only the cache. Use this iterator only with models whose
/// > ``LMOutput/State`` a decode step does not rewrite, as with ``SpeculativeTokenIterator``.
/// > Positional anchors resolved against the cache offset, such as M-RoPE deltas, qualify.
public struct PromptLookupTokenIterator: TokenIteratorProtocol {
    private var base: TokenIterator
    private let index: PromptLookupIndex?
    private let configuration: PromptLookupConfiguration
    private let batchGreedy: Bool
    private var proposal: [Int] = []
    private var pending: [Int] = []
    private var pendingIndex = 0
    private var committedPending = 0
    private var stagedPending = false
    private var speculativePending = false
    private var knownToken: Int?
    private var draftLimit: Int
    private var misses = 0
    private var rejections = 0
    private var cooldown = 0
    private var lastMemoryClear = 0
    private var finished = false
    private var telemetry = SpeculativeDecodingTelemetry()

    public private(set) var tokenCount = 0
    public var maxTokens: Int? { base.maxTokens }
    public var state: LMOutput.State? { base.state }
    public var promptPrefillTime: TimeInterval { base.promptPrefillTime }
    public var speculativeDecodingTelemetry: SpeculativeDecodingTelemetry? { telemetry }
    public var evictedTokenCount: Int { base.evictedTokenCount }
    public var reasoningTokenCount: Int? { base.reasoningTokenCount }
    /// Ordinary decode steps, including lookup misses and unsupported cache topologies.
    public private(set) var fallbackTokenCount = 0
    /// Lookups that produced no usable continuation.
    public private(set) var lookupMissCount = 0
    /// Steps for which the adaptive policy skipped a lookup.
    public private(set) var suppressedLookupCount = 0

    var cacheStorage: KVCacheStorage { base.cacheStorage }
    var cache: [KVCache] { base.cache }

    /// - Parameters:
    ///   - input: Input still needing prefill, which can be a suffix of a cached prompt.
    ///   - model: Target language model.
    ///   - cache: Optional cache to continue.
    ///   - state: Positional state associated with the cache.
    ///   - parameters: Sampling, prefill, token limit, and cache configuration.
    ///   - components: Optional logit processors and validation.
    ///   - configuration: Lookup capacity, confidence, and draft limits.
    ///   - history: Complete logical prompt when `input` contains only an uncached suffix.
    public init(
        input: LMInput, model: any LanguageModel, cache: [KVCache]? = nil,
        state: LMOutput.State? = nil, parameters: GenerateParameters,
        components: GenerationComponents = .init(),
        configuration: PromptLookupConfiguration = .init(), history: [Int]? = nil
    ) throws {
        try configuration.validate()
        let plan = try parameters.kvCachePlan()
        try self.init(
            input: input, model: model,
            cacheStorage: KVCacheStorage(
                try cache ?? model.newCache(parameters: parameters), plan: plan),
            state: state, parameters: parameters, components: components,
            configuration: configuration, history: history)
    }

    package init(
        input: LMInput, model: any LanguageModel, cacheStorage: KVCacheStorage,
        state: LMOutput.State? = nil, parameters: GenerateParameters,
        components: GenerationComponents = .init(),
        configuration: PromptLookupConfiguration = .init(), history: [Int]? = nil
    ) throws {
        try configuration.validate()
        guard input.text.tokens.ndim == 1, input.text.tokens.size > 0 else {
            throw PromptLookupError.invalidInput
        }
        self.configuration = configuration
        self.draftLimit = configuration.maxDraftTokens
        let base = try TokenIterator(
            input: input, model: model, cacheStorage: cacheStorage, state: state,
            parameters: parameters, components: components)
        self.base = base
        self.batchGreedy = parameters.temperature == 0 && base.processor == nil
        let leaves = KVCacheTree.leaves(in: base.cache)
        let supportsRounds =
            !leaves.isEmpty && leaves.count == base.cache.count
            && leaves.allSatisfy { $0.path.count == 1 && $0.isAttentionCache }
            && input.image == nil && input.video == nil && input.audio == nil

        let start = Date.timeIntervalSinceReferenceDate
        self.index =
            supportsRounds
            ? PromptLookupIndex(
                capacity: configuration.contextSize, maximumOrder: configuration.maxNGramLength)
            : nil
        index?.append(contentsOf: history ?? input.text.tokens.asArray(Int.self))
        proposal.reserveCapacity(configuration.maxDraftTokens + 1)
        pending.reserveCapacity(configuration.maxDraftTokens + 1)
        self.base.promptPrefillTime += Date.timeIntervalSinceReferenceDate - start
    }

    public mutating func next() -> Int? {
        guard !finished else { return nil }
        guard maxTokens.map({ tokenCount < $0 }) ?? true else {
            finish()
            return nil
        }
        if pendingIndex == pending.count {
            // Conversion may replace cache leaves and invalidate their restore points.
            if stagedPending { base.kvCachePlan.apply(to: cacheStorage) }
            pending.removeAll(keepingCapacity: true)
            pendingIndex = 0
            committedPending = 0
            stagedPending = false
            speculativePending = false
            autoreleasepool { advance() }
        }
        guard pendingIndex < pending.count else { return nil }
        let token = pending[pendingIndex]
        pendingIndex += 1
        tokenCount += 1
        if speculativePending { telemetry.recordGeneratedToken() }
        return token
    }

    private mutating func advance() {
        let indexedCurrent = knownToken != nil
        if let knownToken { index?.append(knownToken) }
        let remaining = maxTokens.map { $0 - tokenCount } ?? (configuration.maxDraftTokens + 1)
        let count = Swift.min(draftLimit, remaining - 1, positionsBeforeCompression - 1)
        guard let index, count > 0 else {
            singleStep(indexedCurrent: indexedCurrent)
            return
        }
        if cooldown > 0 {
            cooldown -= 1
            suppressedLookupCount += 1
            singleStep(indexedCurrent: indexedCurrent)
            return
        }

        index.draft(
            maximumTokens: count + (indexedCurrent ? 0 : 1),
            minimumOrder: configuration.minNGramLength,
            minimumOccurrences: configuration.minimumOccurrences,
            minimumConfidence: configuration.minimumConfidence, into: &proposal)
        // The async pipeline's carried token is not yet on the CPU. Skip its guess and
        // verify the remaining guesses against the actual token, without stalling the GPU.
        if !indexedCurrent && !proposal.isEmpty { proposal.removeFirst() }
        guard !proposal.isEmpty else {
            lookupMissCount += 1
            misses += 1
            if configuration.adaptive && misses >= 8 {
                cooldown = 8
                misses = 0
            }
            singleStep(indexedCurrent: indexedCurrent)
            return
        }
        misses = 0

        var round: KVCacheRound?
        while !proposal.isEmpty {
            round = cacheStorage.beginRound(maximumPositions: proposal.count + 1)
            if round != nil { break }
            proposal.removeLast()
        }
        guard let round else {
            singleStep(indexedCurrent: indexedCurrent)
            return
        }
        verify(round, indexedCurrent: indexedCurrent)
    }

    /// Positions a round can verify before ordinary decoding would compress the cache.
    ///
    /// Ordinary decoding compresses once an uncompressed leaf's offset passes the start, so a
    /// round must not run later positions against the uncompressed entries.
    private var positionsBeforeCompression: Int {
        guard !cacheStorage.isApplicationTerminal,
            let start = base.kvCachePlan.configuration?.strategy.compressionStart
        else { return .max }
        let offsets = KVCacheTree.leaves(in: cache).compactMap { leaf -> Int? in
            guard case .simple(let simple) = leaf.kind, simple.offset <= start else { return nil }
            return simple.offset
        }
        return offsets.min().map { start + 1 - $0 } ?? .max
    }

    private mutating func verify(_ round: KVCacheRound, indexedCurrent: Bool) {
        let current = base.y.tokens
        let input = LMInput.Text(tokens: concatenated([current, MLXArray(proposal)]))
        let result = base.model(input[text: .newAxis], cache: round.caches, state: base.state)
        base.state = result.state
        let accepted: Int
        let correction: Int
        if batchGreedy {
            let tokens = argMax(result.logits.squeezed(axis: 0), axis: -1)
            eval(tokens, current)
            let values = tokens.asArray(Int.self)
            pending.append(current.item(Int.self))
            var count = 0
            while count < proposal.count && values[count] == proposal[count] {
                pending.append(values[count])
                count += 1
            }
            accepted = count
            correction = values[count]
        } else {
            eval(result.logits, current)
            pending.append(current.item(Int.self))
            var count = 0
            var final = 0
            for position in 0 ... proposal.count {
                var logits = result.logits[0..., position, 0...]
                logits = base.processor?.process(logits: logits) ?? logits
                let sampled = base.sampler.sample(logits: logits)
                eval(sampled)
                final = sampled.item(Int.self)
                base.processor?.didSample(token: sampled)
                guard position < proposal.count, final == proposal[position] else { break }
                pending.append(final)
                count += 1
            }
            accepted = count
            correction = final
        }

        if !indexedCurrent { index?.append(pending[0]) }
        for token in pending.dropFirst() { index?.append(token) }
        cacheStorage.commit(round, retaining: accepted + 1)
        committedPending = accepted + 1
        stagedPending = true
        speculativePending = true
        telemetry.recordRound(
            drafted: proposal.count, accepted: accepted, targetVerified: proposal.count + 1,
            draftModelCalls: 0)
        base.y = .init(tokens: MLXArray([correction]))
        knownToken = correction
        asyncEval(cache.flatMap { $0.state })
        if tokenCount == 0 || tokenCount - lastMemoryClear >= 256 {
            MLX.Memory.clearCache()
            lastMemoryClear = tokenCount
        }

        if configuration.adaptive {
            if accepted == 0 {
                draftLimit = Swift.max(1, draftLimit / 2)
                rejections += 1
                if rejections >= 3 {
                    cooldown = 16
                    rejections = 0
                }
            } else {
                rejections = 0
                if accepted == proposal.count {
                    draftLimit = Swift.min(configuration.maxDraftTokens, draftLimit + 2)
                }
            }
        }
    }

    private mutating func singleStep(indexedCurrent: Bool) {
        base.tokenCount = tokenCount
        guard let token = base.next() else { return }
        if tokenCount % 256 == 0 { lastMemoryClear = tokenCount }
        if !indexedCurrent { index?.append(token) }
        knownToken = nil
        pending.append(token)
        committedPending = 1
        fallbackTokenCount += 1
    }

    /// Ends generation and removes verified tokens that ``next()`` has not returned from the
    /// cache, so the cache holds exactly the prompt and the returned tokens.
    ///
    /// Call it on the iterator value you advanced before you reuse its cache after stopping
    /// early. Later calls to ``next()`` return `nil`, and further calls have no effect.
    /// Streaming generation and ``ChatSession`` call it for you.
    public mutating func finish() {
        guard !finished else { return }
        finished = true
        let lookahead = committedPending - Swift.min(pendingIndex, committedPending)
        if lookahead > 0 {
            if stagedPending {
                cacheStorage.rewindLastRound(lookahead)
            } else {
                cacheStorage.trim(lookahead)
            }
            committedPending -= lookahead
        }
        // Nothing rewinds after this, so do not keep sliding-window snapshots alive.
        cacheStorage.discardLastRound()
        base.kvCachePlan.apply(to: cacheStorage)
        base.finalizeGeneration()
    }

    public mutating func recordEmittedToken(_ token: Int) {
        base.recordEmittedToken(token)
    }

    public mutating func discardGeneratedToken() {
        // The token collector retains stop tokens even when the text stream suppresses them.
        if speculativePending { telemetry.discardGeneratedToken() }
    }
}

extension PromptLookupTokenIterator: GenerationFinalizingTokenIterator {
    mutating func finalizeGeneration() {
        finish()
    }
}

func makeTokenIterator(
    input: LMInput, model: any LanguageModel, cache: [KVCache]? = nil,
    state: LMOutput.State? = nil, parameters: GenerateParameters,
    components: GenerationComponents = .init()
) throws -> any TokenIteratorProtocol {
    let plan = try parameters.kvCachePlan()
    return try makeTokenIterator(
        input: input, model: model,
        cacheStorage: KVCacheStorage(
            try cache ?? model.newCache(parameters: parameters), plan: plan),
        state: state, parameters: parameters, components: components)
}

func makeTokenIterator(
    input: LMInput, model: any LanguageModel, cacheStorage: KVCacheStorage,
    state: LMOutput.State? = nil, parameters: GenerateParameters,
    components: GenerationComponents = .init(), history: [Int]? = nil
) throws -> any TokenIteratorProtocol {
    if let configuration = parameters.promptLookup {
        return try PromptLookupTokenIterator(
            input: input, model: model, cacheStorage: cacheStorage, state: state,
            parameters: parameters, components: components, configuration: configuration,
            history: history)
    }
    return try TokenIterator(
        input: input, model: model, cacheStorage: cacheStorage, state: state,
        parameters: parameters, components: components)
}
