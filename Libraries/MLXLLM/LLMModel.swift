// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXLMCommon

/// Runtime switch over ``LLMModel/prepare(_:cache:state:windowSize:)``'s chunking strategy.
///
/// Exists so an A/B can alternate strategies **inside one process**, sharing the
/// loaded weights and the machine's thermal state. Separate-binary A/Bs of this
/// change were confounded: the M3 Max throttles hard under sustained 35B prefill,
/// and whichever arm ran second always measured on a hotter machine.
///
/// Production leaves this at ``balanced``. Not thread-safe by construction — set it
/// before generation starts, from the benchmark harness only.
public enum PrefillChunking: Sendable {
    /// Equal chunks of at most `prefillStepSize`; the iterator gets one token.
    case balanced
    /// Upstream behaviour: chunks of exactly `prefillStepSize`, and the iterator
    /// swallows `promptTokens mod prefillStepSize` in one forward.
    case remainder
    /// No chunking at all: the whole prompt in a single forward.
    ///
    /// This is the *reference* computation — the one chunking approximates. It is
    /// not usable in production (the `[1, H, Lq, Lk]` score matrix is quadratic in
    /// the prompt), but at a few thousand tokens it fits, and it is the only way to
    /// ask whether a chunking scheme moves logits *toward or away from* the true
    /// answer. Comparing two chunkings to each other cannot answer that: neither is
    /// a ground truth.
    case unchunked

    nonisolated(unsafe) public static var strategy: PrefillChunking = .balanced
}

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
    /// Evaluates the prompt in equal chunks of at most `prefillStepSize`, leaving
    /// exactly ``tokenIteratorTail`` tokens for the `TokenIterator`'s first forward.
    ///
    /// **Why the chunks are balanced.** The obvious loop —
    /// `while y.tokens.size > prefillStepSize` — hands the iterator a remainder of
    /// `promptTokens mod prefillStepSize`, which it swallows in one un-pipelined
    /// forward at the largest `Lk` of the whole prefill. That forward is
    /// disproportionately slow: it pays full attention cost against the entire
    /// prompt while giving each MoE expert only `Lq × topK / numExperts` rows of
    /// GEMM. Measured on Qwen3.6-35B-A3B at 32K, a 141-token
    /// remainder cost **2.89 s** — three times a full 1024-token chunk — and the
    /// remainder *grows with the step size* (141 → 1,165 → 3,213 tokens at
    /// 1024 / 2048 / 4096), so raising `prefillStepSize` to speed the loop up made
    /// the remainder worse by more than the loop gained.
    ///
    /// Balancing removes the remainder without splitting it into a second small
    /// forward: 31,884 tokens at a 1024 ceiling become 32 chunks of ~997, not 31 of
    /// 1024 plus a straggler. `prefillStepSize` becomes a pure loop-efficiency knob,
    /// bounded only by the peak memory of one chunk.
    public func prepare(
        _ input: LMInput, cache: [KVCache], state: LMOutput.State?, windowSize: Int?
    ) throws
        -> PrepareResult
    {
        let prefillStepSize = windowSize ?? 512
        var y = input.text

        // A prompt that fits in one chunk is handed to the iterator whole, exactly
        // as before: chunking it would only add a second forward.
        let chunkSize: Int
        let tail: Int
        switch PrefillChunking.strategy {
        case .unchunked:
            return .tokens(y)
        case .balanced:
            guard
                let balanced = Self.balancedChunkSize(
                    promptTokens: y.tokens.size, prefillStepSize: prefillStepSize)
            else {
                return .tokens(y)
            }
            (chunkSize, tail) = (balanced, Self.tokenIteratorTail)
        case .remainder:
            (chunkSize, tail) = (prefillStepSize, prefillStepSize)
        }

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
                    let n = min(chunkSize, y.tokens.size - Self.tokenIteratorTail)
                    let input = y[.newAxis, ..<n]
                    let output = self(input, cache: cache.isEmpty ? nil : cache, state: state)
                    state = output.state
                    asyncEval(cache)
                    y = y[n...]
                }
            }

            // Single sync after the loop to flush any remaining async work.
            eval(cache)
        }

        return .tokens(y)
    }

    /// Tokens left for the `TokenIterator`'s first forward, which is the step that
    /// produces the first sampled logits. One token makes that step decode-shaped.
    public static var tokenIteratorTail: Int { 1 }

    /// The largest chunk size ≤ `prefillStepSize` that splits the prompt into equal
    /// chunks leaving only ``tokenIteratorTail`` tokens over, or `nil` when the
    /// prompt is short enough that the iterator should take all of it.
    public static func balancedChunkSize(promptTokens: Int, prefillStepSize: Int) -> Int? {
        guard prefillStepSize > 0, promptTokens > prefillStepSize else { return nil }
        let toChunk = promptTokens - tokenIteratorTail
        let chunkCount = (toChunk + prefillStepSize - 1) / prefillStepSize
        return (toChunk + chunkCount - 1) / chunkCount
    }

    public func messageGenerator(tokenizer: Tokenizer) -> MessageGenerator {
        DefaultMessageGenerator()
    }
}
