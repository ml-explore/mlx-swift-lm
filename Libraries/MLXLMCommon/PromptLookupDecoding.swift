// Copyright © 2026 Apple Inc.

import Foundation
import MLX

/// Find a draft continuation for the current context by n-gram lookup.
///
/// Prompt-lookup decoding (PLD) proposes draft tokens by matching the
/// trailing n-gram of the generated-so-far token history against an earlier
/// occurrence in that same history, and proposing the tokens that followed
/// it. No draft model is involved; the proposal is free. It pays off
/// whenever the output re-uses spans of the input verbatim — summarization,
/// code editing, retrieval-grounded answers, chat with recurring phrasing —
/// and costs one wasted verify slot per round otherwise.
///
/// Longer patterns are preferred (tried from `maxNGramLength` down to
/// `minNGramLength`), and among equal-length matches the most recent
/// occurrence wins.
///
/// Linear scan; O(history × maxNGramLength) per round. Fine for on-device
/// context sizes — an n-gram hash index (as in llama.cpp) is a follow-up if
/// profiling ever shows this to matter.
///
/// - Returns: up to `maxTokens` proposed continuation tokens; empty when no
///   pattern of at least `minNGramLength` recurs.
func promptLookupDraft(
    history: [Int], maxNGramLength: Int, minNGramLength: Int, maxTokens: Int
) -> [Int] {
    guard maxTokens > 0, history.count > minNGramLength else { return [] }

    let upperLength = min(maxNGramLength, history.count - 1)
    guard upperLength >= minNGramLength else { return [] }

    for length in stride(from: upperLength, through: max(minNGramLength, 1), by: -1) {
        let pattern = history.suffix(length)
        // Scan candidate start positions from most recent to oldest. The
        // match may not be the trailing occurrence itself, hence the
        // `history.count - length - 1` upper bound.
        var start = history.count - length - 1
        while start >= 0 {
            if history[start ..< start + length].elementsEqual(pattern) {
                let continuationStart = start + length
                let continuationEnd = min(continuationStart + maxTokens, history.count)
                if continuationEnd > continuationStart {
                    return Array(history[continuationStart ..< continuationEnd])
                }
                break  // trailing match with no room to continue; try shorter
            }
            start -= 1
        }
    }
    return []
}

/// Generator of tokens using prompt-lookup (n-gram) speculative decoding.
///
/// Structurally a ``SpeculativeTokenIterator`` whose draft source is
/// `promptLookupDraft(history:maxNGramLength:minNGramLength:maxTokens:)`
/// over the prompt plus the accepted tokens, instead of a second model:
/// proposals are verified by the main model in a single batched forward
/// pass, accepted while they match the main model's own samples, and the
/// cache is rewound past the first mismatch.
///
/// Because the proposal is deterministic (no draft distribution), comparing
/// against tokens *sampled from the main model's distribution* is unbiased
/// at any temperature — the output distribution is exactly the main
/// model's, so this needs none of the residual-sampling machinery a model
/// drafter requires.
///
/// Rounds only run while the cache can rewind rejected tokens. When it no
/// longer can — e.g. a sliding-window layer wrapped mid-generation and
/// became untrimmable — the iterator permanently falls back to single-token
/// generation instead of corrupting the cache.
public struct PromptLookupTokenIterator: TokenIteratorProtocol {

    var y: LMInput.Text
    let model: any LanguageModel

    var state: LMOutput.State?
    var cache: [KVCache]
    let quantizeKVCache: (inout [KVCache]) -> Void

    var processor: LogitProcessor?
    let sampler: LogitSampler

    public var tokenCount: Int { telemetry.emittedTokenCount }
    public let maxTokens: Int?
    let numDraftTokens: Int
    let maxNGramLength: Int
    let minNGramLength: Int

    /// Prompt plus accepted tokens — the corpus the n-gram lookup draws
    /// proposals from.
    private var history: [Int]

    // Buffer of accepted tokens from the current speculation round
    private var pendingTokens = [Int]()
    private var pendingIndex = 0

    /// Set once the cache can no longer rewind rejected draft tokens
    /// (untrimmable — e.g. a wrapped sliding-window layer). Sticky: the
    /// remainder of the stream is plain single-token generation.
    private var passthrough = false

    // Internal metrics
    public var promptPrefillTime: TimeInterval = 0.0
    private var telemetry = SpeculativeDecodingTelemetry()
    public var speculativeDecodingTelemetry: SpeculativeDecodingTelemetry? {
        telemetry.roundCount > 0 ? telemetry : nil
    }

    public mutating func discardGeneratedToken() {
        telemetry.discardGeneratedToken()
    }

    /// Initialize a `PromptLookupTokenIterator` with the given input.
    ///
    /// - Parameters:
    ///   - input: language model input
    ///   - model: the ``LanguageModel``
    ///   - cache: optional ``KVCache``
    ///   - parameters: the generation parameters
    ///   - numDraftTokens: maximum tokens proposed per lookup round
    ///   - maxNGramLength: longest trailing pattern to match (tried first)
    ///   - minNGramLength: shortest trailing pattern worth matching
    public init(
        input: LMInput,
        model: any LanguageModel,
        cache: [KVCache]? = nil,
        parameters: GenerateParameters,
        numDraftTokens: Int = 10,
        maxNGramLength: Int = 3,
        minNGramLength: Int = 1
    ) throws {
        precondition(minNGramLength >= 1, "minNGramLength must be >= 1")
        precondition(
            maxNGramLength >= minNGramLength, "maxNGramLength must be >= minNGramLength")
        precondition(numDraftTokens >= 1, "numDraftTokens must be >= 1")

        self.y = input.text
        self.model = model

        self.cache = cache ?? model.newCache(parameters: parameters)
        guard canTrimPromptCache(self.cache) else {
            throw KVCacheError(
                message: "Prompt-lookup decoding requires a trimmable KV cache.")
        }

        self.sampler = parameters.sampler()
        self.processor = parameters.processor()

        self.maxTokens = parameters.maxTokens
        self.numDraftTokens = numDraftTokens
        self.maxNGramLength = maxNGramLength
        self.minNGramLength = minNGramLength

        self.history = input.text.tokens.asArray(Int.self)

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

    /// Prefill the model with the prompt, priming the cache for generation.
    mutating func prepare(input: LMInput, windowSize: Int? = nil) throws {
        processor?.prompt(input.text.tokens)

        switch try model.prepare(input, cache: cache, state: state, windowSize: windowSize) {
        case .tokens(let tokens):
            y = tokens
        case .logits(let result):
            var logits = result.logits[0..., -1, 0...]
            logits = processor?.process(logits: logits) ?? logits
            let token = sampler.sample(logits: logits)
            processor?.didSample(token: token)
            y = .init(tokens: token)
            state = result.state
        }
    }

    /// Emit one token without speculation: forward `y`, sample, buffer.
    private mutating func singleStep() {
        let result = model(y[text: .newAxis], cache: cache, state: state)
        state = result.state
        var logits = result.logits[0..., -1, 0...]
        logits = processor?.process(logits: logits) ?? logits
        let token = sampler.sample(logits: logits)
        processor?.didSample(token: token)
        asyncEval(token)

        quantizeKVCache(&cache)

        let tokenValue = token.item(Int.self)
        pendingTokens.append(tokenValue)
        history.append(tokenValue)
        y = .init(tokens: token)
    }

    /// Whether every cache layer can still rewind after `tokensToAdd` more
    /// tokens: trimmable now, and — for window-bounded caches — with enough
    /// headroom that the verify pass cannot wrap the window mid-round
    /// (rewinding must never be attempted on a freshly wrapped buffer).
    private func canRewind(after tokensToAdd: Int) -> Bool {
        cache.allSatisfy { layer in
            guard layer.isTrimmable else { return false }
            if let maxSize = layer.maxSize {
                return layer.offset + tokensToAdd < maxSize
            }
            return true
        }
    }

    /// Run one round: look up a draft continuation, verify it in a single
    /// batched forward pass, accept the matching prefix, rewind the rest.
    private mutating func speculateRound() {
        let remaining = maxTokens.map { $0 - tokenCount } ?? numDraftTokens
        let numDraft = Swift.min(remaining, numDraftTokens)
        guard numDraft > 0 else { return }

        // The rollback below must be able to rewind rejected tokens.
        // Checked per round with the round's own size as headroom, not just
        // at init: a sliding-window layer wraps (and becomes untrimmable)
        // mid-generation, and a silent no-op rollback would leave rejected
        // tokens in the cache — corrupting every subsequent step. Sticky:
        // offsets only grow, so once the headroom is gone it never returns.
        guard !passthrough, canRewind(after: numDraft + 1) else {
            passthrough = true
            singleStep()
            return
        }

        let proposal = promptLookupDraft(
            history: history,
            maxNGramLength: maxNGramLength,
            minNGramLength: minNGramLength,
            maxTokens: numDraft)
        // No recurring pattern: a verify pass would cost a forward with
        // nothing to accept, so just generate one token.
        guard !proposal.isEmpty else {
            singleStep()
            return
        }

        let draftCount = proposal.count
        let draftTokens = MLXArray(proposal.map { Int32($0) })

        // Verification: the model processes bonus + proposal in one pass.
        let verifyInput = LMInput.Text(tokens: concatenated([y.tokens, draftTokens]))
        let verifyStart = verifyInput.tokens.dim(0) - (draftCount + 1)
        let result = model(verifyInput[text: .newAxis], cache: cache, state: state)
        state = result.state
        let verifyLogits = result.logits

        let mainTokens: MLXArray
        if var verifyProcessor = processor {
            // Process each position sequentially so the processor sees the
            // tokens sampled at earlier positions.
            var sampled = [MLXArray]()
            for i in 0 ..< (draftCount + 1) {
                var logits = verifyLogits[0..., verifyStart + i, 0...]
                logits = verifyProcessor.process(logits: logits)
                let token = sampler.sample(logits: logits)
                verifyProcessor.didSample(token: token)
                sampled.append(token)
            }
            mainTokens = concatenated(sampled)
        } else {
            let logits = verifyLogits[0..., verifyStart..., 0...].squeezed(axis: 0)
            mainTokens = sampler.sample(logits: logits)
        }

        // Accept the proposal prefix that matches the model's own samples.
        // The proposal is deterministic (no draft distribution), so
        // comparing against sampled target tokens keeps the output
        // distribution exactly the model's at any temperature.
        eval(mainTokens)
        let mainTokensList = mainTokens.asArray(Int.self)
        var accepted = 0
        for i in 0 ..< draftCount {
            guard mainTokensList[i] == proposal[i] else { break }
            processor?.didSample(token: mainTokens[i ... i])
            pendingTokens.append(mainTokensList[i])
            history.append(mainTokensList[i])
            accepted += 1
        }

        // Always emit the model's token at position `accepted` (the
        // correction token, or the bonus token when everything matched).
        let finalToken = mainTokens[accepted ... accepted]
        processor?.didSample(token: finalToken)
        pendingTokens.append(mainTokensList[accepted])
        history.append(mainTokensList[accepted])

        telemetry.recordRound(
            drafted: draftCount,
            accepted: accepted,
            targetVerified: draftCount + 1
        )

        // Rewind the cache past the first mismatch. Guaranteed to succeed:
        // the pre-round headroom check ensures no layer wrapped during the
        // verify pass, and nothing else has touched the cache since.
        let rejected = draftCount - accepted
        let trimmed = trimPromptCache(cache, numTokens: rejected)
        assert(trimmed == rejected, "rollback failed despite pre-round headroom check")

        quantizeKVCache(&cache)

        y = .init(tokens: finalToken)
    }

    mutating public func next() -> Int? {
        if let maxTokens, tokenCount >= maxTokens {
            return nil
        }

        // Drain the pending buffer first
        if pendingIndex < pendingTokens.count {
            let token = pendingTokens[pendingIndex]
            pendingIndex += 1
            telemetry.recordGeneratedToken()
            return token
        }

        pendingTokens.removeAll(keepingCapacity: true)
        pendingIndex = 0
        speculateRound()

        if pendingTokens.isEmpty {
            return nil
        }

        let token = pendingTokens[pendingIndex]
        pendingIndex += 1
        telemetry.recordGeneratedToken()
        return token
    }
}
