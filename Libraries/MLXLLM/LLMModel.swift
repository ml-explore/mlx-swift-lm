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
    /// ``PrefillParameters/stepSize`` (default 512), leaving one token for the
    /// `TokenIterator`'s first forward. With ``PrefillParameters/Chunking/balanced``
    /// (the default) the chunks are equal-sized, so no forward is a small
    /// remainder paying full attention cost against the whole prompt.
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

        var processed = 0
        try withPreparedCache(cache, lengths: y.sequenceLengths) {
            // asyncEval lets the CPU build chunk N+1's graph while the GPU evaluates
            // chunk N. Under .remainder the reserved tail is the legacy leftover
            // (up to a full step) rather than a single token.
            var state: LMOutput.State? = state
            processed = try prefill.forEachChunk(
                total: total, reserving: prefill.chunking == .remainder ? stepSize : 1
            ) { range in
                let input = y[.newAxis, range]
                let output = self(input, cache: cache.isEmpty ? nil : cache, state: state)
                state = output.state
                asyncEval(cache)
            }

            // Single sync after the loop to flush any remaining async work.
            if processed > 0 {
                eval(cache)
            }
        }

        return .tokens(y[processed...])
    }

    public func messageGenerator(tokenizer: Tokenizer) -> MessageGenerator {
        DefaultMessageGenerator()
    }
}
