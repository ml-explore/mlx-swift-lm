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
    public func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws
        -> PrepareResult
    {
        let prefillStepSize = windowSize ?? 512
        var y = input.text

        // Prepare the prompt in chunks if larger than the prefill size.
        // asyncEval lets the CPU build chunk N+1's graph while the GPU evaluates
        // chunk N. Python mlx-lm gets this pipelining for free because its bindings
        // defer eval until a value is read. The previous `eval(cache)` call was a
        // blocking sync that drained the GPU pipeline between chunks.
        while y.tokens.size > prefillStepSize {
            let input = y[.newAxis, ..<prefillStepSize]
            _ = self(input, cache: cache.isEmpty ? nil : cache, state: nil)

            var cacheArrays: [MLXArray] = []
            for c in cache { cacheArrays.append(contentsOf: c.innerState()) }
            asyncEval(cacheArrays)

            y = y[prefillStepSize...]
        }

        // Single sync after the loop — flush any remaining async work.
        eval(cache)

        return .tokens(y)
    }

    public func messageGenerator(tokenizer: Tokenizer) -> MessageGenerator {
        DefaultMessageGenerator()
    }
}
