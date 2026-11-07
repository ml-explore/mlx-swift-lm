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
                state = feedPrefillChunk(y, range: range, cache: cache, state: state)
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

    /// One chunk of ``prepare(_:cache:state:prefill:)``, for a model whose chunks carry no
    /// model state; see ``ResumablePrefillModel``.
    public func prefillChunk(_ prompt: LMInput.Text, range: Range<Int>, cache: [KVCache]) {
        let state = feedPrefillChunk(prompt, range: range, cache: cache, state: nil)
        precondition(
            state?.isEmpty != false,
            "\(Self.self) returns model state from its prefill chunks, so it cannot resume them")
    }

    /// Feeds one chunk of the prompt and returns the model state for the next chunk.
    private func feedPrefillChunk(
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

// MARK: - Resumable prefill

// Models that use the default `LLMModel` prefill and return no model state from their forward
// pass, so their prefill chunks can be fed one per call. Each is checked against
// `TokenIterator` in `BatchTokenIteratorTests.resumablePrefillFamilies`.
extension LlamaModel: ResumablePrefillModel {}
extension Qwen2Model: ResumablePrefillModel {}
extension Qwen3Model: ResumablePrefillModel {}
extension Qwen3MoEModel: ResumablePrefillModel {}
extension PhiModel: ResumablePrefillModel {}
extension Phi3Model: ResumablePrefillModel {}
extension GemmaModel: ResumablePrefillModel {}
extension Gemma2Model: ResumablePrefillModel {}
extension Starcoder2Model: ResumablePrefillModel {}
extension CohereModel: ResumablePrefillModel {}
extension InternLM2Model: ResumablePrefillModel {}
extension GraniteModel: ResumablePrefillModel {}
extension GLM4Model: ResumablePrefillModel {}
extension SmolLM3Model: ResumablePrefillModel {}
extension Olmo2Model: ResumablePrefillModel {}
extension MixtralModel: ResumablePrefillModel {}
