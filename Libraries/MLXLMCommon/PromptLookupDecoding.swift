// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN

/// Prompt-lookup (n-gram) speculative decoding.
///
/// A coding/agentic workload constantly regenerates spans that already exist in
/// the context — identifiers, signatures, edited code, tool-call scaffolding.
/// Decode throughput of a large dense model is memory-bandwidth-bound (every
/// generated token reads all weights once), so the way past that roofline is
/// emitting more than one token per weight pass. Prompt-lookup decoding drafts
/// continuation candidates by longest-suffix n-gram match against the tokens
/// already seen, and the main model verifies `numDraftTokens + 1` positions in
/// a single forward — the same weight reads as one decode step. No second
/// model, no additional weight memory.
///
/// This is the technique behind vLLM's `ngram` speculator and llama.cpp's
/// lookup decoding. As there, a draft is accepted only when it matches the
/// verifier's own sampled token, so the emitted stream is exactly what the
/// main model would have produced on its own — prompt lookup changes how fast
/// tokens come out, never which tokens.
///
/// The drafter plugs into ``SpeculativeTokenIterator`` as a synthetic
/// ``LanguageModel``: its "KV cache" is the raw token history (so the
/// iterator's rewind-on-reject trims it like a real cache) and its logits are
/// a one-hot row over the vocabulary.

/// Token-history "KV cache" for the prompt-lookup drafter.
///
/// `offset`/`trim` mirror a real cache so ``SpeculativeTokenIterator``'s
/// bookkeeping (rewind of rejected drafts, processed-token ledger) applies to
/// the drafter's view of history exactly as it does to the main model's KV.
package final class PromptLookupTokenCache: KVCache {
    package var tokens: [Int32] = []

    package init() {}

    package var offset: Int { tokens.count }
    package var maxSize: Int? { nil }

    package func append(_ newTokens: [Int32]) {
        tokens.append(contentsOf: newTokens)
    }

    package var isTrimmable: Bool { true }

    @discardableResult
    package func trim(_ n: Int) -> Int {
        let trimmed = min(max(n, 0), tokens.count)
        tokens.removeLast(trimmed)
        return trimmed
    }

    package func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        fatalError("PromptLookupTokenCache stores tokens, not K/V tensors")
    }

    package var state: [MLXArray] {
        get { [MLXArray(tokens)] }
        set { tokens = newValue.first?.asArray(Int32.self) ?? [] }
    }

    package var metaState: [String] {
        get { [""] }
        set {}
    }

    package func makeMask(
        n: Int, windowSize: Int?, returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        .none
    }

    package func copy() -> any KVCache {
        let copy = PromptLookupTokenCache()
        copy.tokens = tokens
        return copy
    }

    package func innerState() -> [MLXArray] { [] }
}

/// Draft model that proposes continuations by n-gram lookup over its own
/// token history (prompt plus accepted generation).
///
/// Matching follows vLLM's ngram proposer and llama.cpp's lookup decoding:
/// the longest current suffix of length `maxNGram ... minNGram` is matched
/// against the most recent earlier occurrence, and subsequent draft positions
/// block-copy the tokens that followed the match. When the copied stream stops
/// being accepted, the drafter re-matches from scratch; when no match exists
/// it proposes a repeat of the last token, which the verifier then rejects at
/// the ordinary cost of one round.
///
/// Use it anywhere a draft ``LanguageModel`` is accepted:
///
/// ```swift
/// let drafter = PromptLookupDraftModel(
///     configuration: .init(vocabularySize: 202_048))
/// let stream = try generate(
///     input: input, parameters: parameters, context: context,
///     draftModel: drafter, numDraftTokens: 3)
/// ```
///
/// or through ``ChatSession`` with
/// ``SpeculativeDecodingConfig/init(lookup:numDraftTokens:)``.
public final class PromptLookupDraftModel: Module, LanguageModel {

    public struct Configuration: Sendable, Hashable {
        /// Width of the one-hot draft logits. Must cover every token id the
        /// model can produce (the language model's vocabulary size, not the
        /// tokenizer's entry count when they differ).
        public var vocabularySize: Int

        /// Longest suffix the drafter tries to match (vLLM `prompt_lookup_max`).
        public var maxNGram: Int

        /// Shortest suffix worth matching (vLLM `prompt_lookup_min`). With
        /// exact-match acceptance a short match cannot corrupt output — it can
        /// only lower the acceptance rate — so 1 is a safe default.
        public var minNGram: Int

        /// Cap on how far back the fresh-match scan walks, bounding host-side
        /// cost on very long histories. `nil` scans the whole history.
        public var maxLookback: Int?

        public init(
            vocabularySize: Int,
            maxNGram: Int = 4,
            minNGram: Int = 1,
            maxLookback: Int? = nil
        ) {
            precondition(vocabularySize > 0, "vocabularySize must be positive")
            precondition(
                maxNGram >= minNGram && minNGram >= 1,
                "require maxNGram >= minNGram >= 1")
            self.vocabularySize = vocabularySize
            self.maxNGram = maxNGram
            self.minNGram = minNGram
            self.maxLookback = maxLookback
        }
    }

    public let configuration: Configuration

    /// Index into the history of the next token to block-copy, valid while the
    /// previous prediction keeps landing in the history unchanged.
    private var continuationSource: Int?
    private var lastPrediction: Int32?

    /// A logit large enough that softmax at any practical temperature puts all
    /// probability on the predicted token (e^(100/T) dwarfs a vocabulary of
    /// e^0 entries for T ≲ 5), yet small enough to survive logit processors.
    private static let oneHotLogit: Float = 100

    public init(configuration: Configuration) {
        self.configuration = configuration
        super.init()
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] { weights }

    public func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
        [PromptLookupTokenCache()]
    }

    public func prepare(
        _ input: LMInput, cache: [KVCache], state: LMOutput.State?, prefill: PrefillParameters
    ) throws -> PrepareResult {
        guard let history = cache.first as? PromptLookupTokenCache else {
            throw KVCacheError(
                message: "PromptLookupDraftModel requires its own token-history cache")
        }
        var tokens = input.text.tokens
        if tokens.ndim == 2 { tokens = tokens[0] }
        history.append(tokens.asArray(Int32.self))
        continuationSource = nil
        lastPrediction = nil
        let total = input.text.tokens.size
        prefill.progress?(total, total)
        return .logits(LMOutput(logits: oneHotLogits(predict(history.tokens))))
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        guard let history = cache?.first as? PromptLookupTokenCache else {
            fatalError("PromptLookupDraftModel requires its own token-history cache")
        }
        history.append(inputs.reshaped(-1).asArray(Int32.self))
        return oneHotLogits(predict(history.tokens))
    }

    /// Next-token proposal: continue the current block copy when the previous
    /// prediction landed, otherwise longest-suffix match, most recent
    /// occurrence wins.
    func predict(_ history: [Int32]) -> Int32 {
        guard let last = history.last else {
            continuationSource = nil
            lastPrediction = nil
            return 0
        }

        // Block copy: the verifier accepted (or the draft loop appended) our
        // previous prediction, so keep copying the same source span.
        if let source = continuationSource, lastPrediction == last, source < history.count - 1 {
            let prediction = history[source]
            continuationSource = source + 1
            lastPrediction = prediction
            return prediction
        }

        let count = history.count
        let floor = configuration.maxLookback.map { Swift.max(0, count - $0) } ?? 0
        var n = Swift.min(configuration.maxNGram, count - 1)
        while n >= configuration.minNGram {
            let suffixStart = count - n
            var i = count - n - 1
            while i >= floor {
                if history[i] == history[suffixStart] {
                    var matches = true
                    var j = 1
                    while j < n {
                        if history[i + j] != history[suffixStart + j] {
                            matches = false
                            break
                        }
                        j += 1
                    }
                    if matches {
                        let prediction = history[i + n]
                        continuationSource = i + n + 1
                        lastPrediction = prediction
                        return prediction
                    }
                }
                i -= 1
            }
            n -= 1
        }

        // No match: propose a repeat of the last token. The verifier rejects a
        // wrong guess at the same weight-read cost as one plain decode step.
        continuationSource = nil
        lastPrediction = last
        return last
    }

    /// `[1, 1, vocabularySize]` one-hot row; any sampler recovers the
    /// prediction because the hot logit dominates the softmax.
    private func oneHotLogits(_ token: Int32) -> MLXArray {
        precondition(
            Int(token) < configuration.vocabularySize,
            "predicted token \(token) exceeds vocabularySize \(configuration.vocabularySize)")
        let logits = MLXArray.zeros([1, 1, configuration.vocabularySize], type: Float.self)
        logits[0, 0, Int(token)] = MLXArray(Self.oneHotLogit)
        return logits
    }
}
