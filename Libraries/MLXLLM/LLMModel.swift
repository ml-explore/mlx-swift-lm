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
    public func prepare(
        _ input: LMInput, cache: [KVCache], state: LMOutput.State?, prefill: PrefillParameters
    ) throws
        -> PrepareResult
    {
        let y = input.text
        // A prompt that fits in one chunk is handed to the iterator whole,
        // keeping short prompts bitwise-identical to the pre-chunking path.
        // `.unchunked` (forEachChunk processes nothing) takes the same route
        // at any prompt length.
        guard y.tokens.size > prefill.resolvedStepSize() else { return .tokens(y) }

        var processed = 0
        try withPreparedCache(cache, lengths: y.sequenceLengths) {
            var state: LMOutput.State? = state
            processed = try prefill.forEachChunk(
                total: y.tokens.size, reserving: prefillReservation(prefill)
            ) { range in
                state = prefillChunk(y, range: range, cache: cache, state: state)
            }

            // Single sync after the loop to flush any remaining async work.
            if processed > 0 {
                eval(cache)
            }
        }

        return .tokens(y[processed...])
    }

    /// The chunks ``prepare(_:cache:state:prefill:)`` feeds; see ``ResumablePrefillModel``.
    public func prefillChunks(for prompt: LMInput.Text, prefill: PrefillParameters)
        -> [Range<Int>]
    {
        guard prompt.tokens.size > prefill.resolvedStepSize() else { return [] }
        return prefill.chunkRanges(
            total: prompt.tokens.size, reserving: prefillReservation(prefill))
    }

    /// One chunk of ``prepare(_:cache:state:prefill:)``; see ``ResumablePrefillModel``.
    public func prefillChunk(
        _ prompt: LMInput.Text, range: Range<Int>, cache: [KVCache], state: LMOutput.State?
    ) -> LMOutput.State? {
        let output = self(prompt[.newAxis, range], cache: cache.isEmpty ? nil : cache, state: state)
        // asyncEval lets the CPU build chunk N+1's graph while the GPU evaluates chunk N.
        asyncEval(cache)
        return output.state
    }

    /// Positions left for the first sampled forward. Under `.remainder` the reserved tail is
    /// the legacy leftover (up to a full step) rather than a single token.
    private func prefillReservation(_ prefill: PrefillParameters) -> Int {
        prefill.chunking == .remainder ? prefill.resolvedStepSize() : 1
    }

    public func messageGenerator(tokenizer: Tokenizer) -> MessageGenerator {
        DefaultMessageGenerator()
    }
}
