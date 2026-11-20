// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXLMCommon

/// Marker protocol for LLMModels
public protocol LLMModel: LanguageModel, LoRAModel {

    /// Models can implement this is they need a custom `MessageGenerator`.
    ///
    /// The default implementation returns `DefaultMessageGenerator`.
    func messageGenerator(tokenizer: Tokenizer) -> MessageGenerator
}

extension LLMModel {

    /// Default prepare step for ``LLMModel``.
    ///
    /// Evaluates the prompt into the cache in chunks of at most
    /// `PrefillParameters.stepSize` (default 512), leaving one token for the
    /// `TokenIterator`'s first forward. With `PrefillParameters.Chunking.balanced`
    /// (the default) the chunks are equal-sized, so no forward is a small
    /// remainder paying full attention cost against the whole prompt.
    ///
    /// For an `MTPChunkedPrefillModel` given `mtpEmitFlagKey`, it also runs the last token
    /// and returns `.logits` with every chunk's hidden rows joined.
    public func prepare(
        _ input: LMInput, cache: [KVCache], state: LMOutput.State?, prefill: PrefillParameters
    ) throws
        -> PrepareResult
    {
        let stepSize = prefill.resolvedStepSize()
        let y = input.text
        let total = y.tokens.size

        // A prompt that fits in one chunk is handed to the iterator whole,
        // keeping short prompts bitwise-identical to the pre-chunking path.
        // `.unchunked` (forEachChunk processes nothing) takes the same route
        // at any prompt length.
        guard total > stepSize else { return .tokens(y) }

        let emitsHidden = self is any MTPChunkedPrefillModel && (state?[mtpEmitFlagKey] ?? false)
        var processed = 0
        var result: PrepareResult?
        try withPreparedCache(cache, lengths: y.sequenceLengths) {
            // asyncEval lets the CPU build chunk N+1's graph while the GPU evaluates
            // chunk N. Under .remainder the reserved tail is the legacy leftover
            // (up to a full step) rather than a single token.
            var state: LMOutput.State? = state
            var hidden: [MLXArray] = []
            processed = try prefill.forEachChunk(
                total: total,
                reserving: prefill.chunking == .remainder && !emitsHidden ? stepSize : 1
            ) { range in
                let input = y[.newAxis, range]
                let output = self(input, cache: cache.isEmpty ? nil : cache, state: state)
                state = output.state
                if emitsHidden, let rows = output.state?[mtpLastHiddenStatesKey] {
                    hidden.append(rows)
                    asyncEval(cache, rows)
                } else {
                    asyncEval(cache)
                }
            }

            // Single sync after the loop to flush any remaining async work.
            if processed > 0 {
                eval(cache)
            }

            // The last token runs alone, so only its row reaches the vocabulary projection.
            guard emitsHidden else { return }
            let output = self(
                y[.newAxis, processed...], cache: cache.isEmpty ? nil : cache, state: state)
            guard var outState = output.state, let tail = outState[mtpLastHiddenStatesKey]
            else {
                result = .logits(output)
                return
            }
            outState[mtpLastHiddenStatesKey] = concatenated(hidden + [tail], axis: 1)
            result = .logits(LMOutput(logits: output.logits, state: outState))
            prefill.progress?(total, total)
        }

        return result ?? .tokens(y[processed...])
    }

    public func messageGenerator(tokenizer: Tokenizer) -> MessageGenerator {
        DefaultMessageGenerator()
    }
}
