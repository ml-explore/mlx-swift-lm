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
        let stepSize = max(1, prefill.stepSize ?? 512)
        var y = input.text
        let total = y.tokens.size

        // A prompt that fits in one chunk is handed to the iterator whole:
        // chunking it would only add a second forward. `.unchunked` (a nil
        // chunk length) takes the same path at any prompt length.
        guard total > stepSize,
            let chunkSize = prefill.chunkLength(forChunking: total - 1, defaultStepSize: stepSize)
        else {
            return .tokens(y)
        }
        let tail = prefill.chunking == .remainder ? stepSize : 1

        try withPreparedCache(cache, lengths: y.sequenceLengths) {
            // asyncEval lets the CPU build chunk N+1's graph while the GPU evaluates
            // chunk N.
            var state: LMOutput.State? = state
            while y.tokens.size > tail {
                // Cooperative cancellation between prefill windows. On iOS, GPU work
                // submitted after the app moves to the background is rejected by the
                // system ("Insufficient Permission"), and the resulting command-buffer
                // error is thrown from a Metal completion handler where it cannot be
                // caught, aborting the process. Without this check a long prompt's
                // prefill cannot be interrupted, so apps cannot stop GPU submissions
                // in time when entering the background. See ml-explore/mlx-swift-examples#230.
                try Task.checkCancellation()
                // Pool per chunk: long prompts run hundreds of chunk forwards
                // before returning to any autorelease boundary.
                autoreleasepool {
                    let n = min(chunkSize, y.tokens.size - 1)
                    let input = y[.newAxis, ..<n]
                    let output = self(input, cache: cache.isEmpty ? nil : cache, state: state)
                    state = output.state
                    asyncEval(cache)
                    y = y[n...]
                    prefill.progress?(total - y.tokens.size, total)
                }
            }

            // Single sync after the loop to flush any remaining async work.
            eval(cache)
        }

        return .tokens(y)
    }

    public func messageGenerator(tokenizer: Tokenizer) -> MessageGenerator {
        DefaultMessageGenerator()
    }
}
