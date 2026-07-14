// Copyright © 2024 Apple Inc.

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
    /// This will evaluate the prompt in chunks until there is a small number of
    /// tokens left to feed into the `TokenIterator`.
    public func prepare(
        _ input: LMInput, cache: [KVCache], state: LMOutput.State?, windowSize: Int?
    ) throws
        -> PrepareResult
    {
        let prefillStepSize = windowSize ?? 512
        var y = input.text

        try withPreparedCache(cache, lengths: y.sequenceLengths) {
            // Prepare the prompt in chunks if larger than the prefill size.
            // asyncEval lets the CPU build chunk N+1's graph while the GPU evaluates
            // chunk N.
            var state: LMOutput.State? = state
            while y.tokens.size > prefillStepSize {
                // Cooperative cancellation between prefill windows. On iOS, GPU work
                // submitted after the app moves to the background is rejected by the
                // system ("Insufficient Permission"), and the resulting command-buffer
                // error is thrown from a Metal completion handler where it cannot be
                // caught, aborting the process. Without this check a long prompt's
                // prefill cannot be interrupted, so apps cannot stop GPU submissions
                // in time when entering the background. See ml-explore/mlx-swift-examples#230.
                try Task.checkCancellation()
                let input = y[.newAxis, ..<prefillStepSize]
                let output = self(input, cache: cache.isEmpty ? nil : cache, state: state)
                state = output.state
                asyncEval(cache)
                y = y[prefillStepSize...]
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
