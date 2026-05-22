// Port of mlx_lm.generate.GenerationBatch.
// https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/generate.py

import Foundation
import MLX
import MLXNN

/// Picks one token per row from a `[B, vocab]` logits tensor.
public typealias RowSampler = @Sendable (MLXArray) -> MLXArray

/// Deterministic greedy sampler.
@Sendable public func greedySampler(_ logprobs: MLXArray) -> MLXArray {
    argMax(logprobs, axis: -1)
}

/// Per-row response from a single decode step.
///
/// Marked `@unchecked Sendable` because `promptCache` carries non-Sendable
/// `KVCacheSimple` references for cross-request prefix caching; cross-actor
/// transfer is the caller's responsibility.
public struct GenerationBatchResponse: @unchecked Sendable {
    public let uid: Int
    public let token: Int

    /// `"length"`, `"stop"`, or nil if the row is still generating.
    public let finishReason: String?

    /// The matched stop sequence if a multi-token stop completed on this token.
    public let matchedSequence: [Int]?

    /// State machine state name after this token's transition (nil = terminated).
    public let currentState: String?

    /// All produced tokens for this row. Set only on the final response.
    public let allTokens: [Int]?

    /// Single-row prompt cache for prefix caching across requests.
    /// Set only on the final response.
    public let promptCache: [any KVCache]?
}

/// Decode-phase batch over a shared `[any BatchedCache]` (one per layer).
/// Each layer's cache is the appropriate batched type for that layer:
/// `BatchKVCache` for full attention, `ArraysCache`/`MambaCache` for SSM.
/// Construct after prefill has populated the caches; call `next()` to
/// drive generation one step at a time.
public final class GenerationBatch: @unchecked Sendable {

    public let model: any LanguageModel
    public private(set) var uids: [Int]
    public private(set) var promptCache: [any BatchedCache]
    public private(set) var tokens: [[Int]]
    public private(set) var maxTokens: [Int]

    public private(set) var samplers: [RowSampler?]
    public let fallbackSampler: RowSampler
    public private(set) var stateMachines: [SequenceStateMachine]

    /// Tokens queued for the next model call. At construction this is the
    /// final prompt token for each row. After priming, and after every
    /// decode step, it holds the sampled token that should be returned on
    /// the next `next()` call. `[B]`.
    private var nextTokens: MLXArray
    private var numTokens: [Int]
    private var matcherStates: [SequenceStateMachineState]

    public init(
        model: any LanguageModel,
        uids: [Int],
        seedTokens: MLXArray,
        promptCache: [any BatchedCache],
        tokens: [[Int]],
        maxTokens: [Int],
        samplers: [RowSampler?]? = nil,
        fallbackSampler: @escaping RowSampler = greedySampler,
        stateMachines: [SequenceStateMachine]? = nil
    ) {
        precondition(uids.count == tokens.count, "uids/tokens count mismatch")
        precondition(uids.count == maxTokens.count, "uids/max_tokens count mismatch")
        self.model = model
        self.uids = uids
        self.promptCache = promptCache
        self.tokens = tokens
        self.maxTokens = maxTokens
        self.samplers = samplers ?? Array(repeating: nil, count: uids.count)
        self.fallbackSampler = fallbackSampler
        let machines = stateMachines ?? Array(repeating: SequenceStateMachine(), count: uids.count)
        self.stateMachines = machines
        self.matcherStates = machines.map { $0.makeState() }
        self.numTokens = Array(repeating: 0, count: uids.count)
        self.nextTokens = seedTokens

        // Match upstream mlx_lm.GenerationBatch: immediately run one
        // decode step in the constructor so the first call to `next()`
        // returns an already-computed token while scheduling the following
        // token. This double-buffer keeps the GPU queue ahead of the CPU
        // token extraction path.
        if !uids.isEmpty {
            _ = step()
        }
    }

    /// Run one decode step. Finished rows (length / stop) are filtered out
    /// of the active set after this call; their final responses appear with
    /// non-nil `finishReason`.
    public func next() -> [GenerationBatchResponse] {
        if uids.isEmpty { return [] }

        let stepTokens = step()

        var keep: [Int] = []
        var responses: [GenerationBatchResponse] = []
        responses.reserveCapacity(uids.count)

        for i in 0 ..< uids.count {
            numTokens[i] += 1

            var finishReason: String? = nil
            if numTokens[i] >= maxTokens[i] {
                finishReason = "length"
            }

            let machine = stateMachines[i]
            let (nextState, matchedSequence, currentState) =
                machine.match(matcherStates[i], stepTokens[i])
            matcherStates[i] = nextState
            if matchedSequence != nil, currentState == nil {
                finishReason = "stop"
            }

            if finishReason != nil {
                let extracted: [any KVCache] = promptCache.map { $0.extractBatched(i) }
                responses.append(
                    GenerationBatchResponse(
                        uid: uids[i],
                        token: stepTokens[i],
                        finishReason: finishReason,
                        matchedSequence: matchedSequence,
                        currentState: currentState,
                        allTokens: tokens[i],
                        promptCache: extracted
                    ))
            } else {
                keep.append(i)
                responses.append(
                    GenerationBatchResponse(
                        uid: uids[i],
                        token: stepTokens[i],
                        finishReason: nil,
                        matchedSequence: matchedSequence,
                        currentState: currentState,
                        allTokens: nil,
                        promptCache: nil
                    ))
            }
        }

        if keep.count < uids.count {
            filter(keep: keep)
        }

        return responses
    }

    /// In-place keep only the rows at the given indices.
    public func filter(keep: [Int]) {
        let keepArr = MLXArray(keep.map { Int32($0) })

        if keep.isEmpty {
            promptCache.removeAll()
        } else {
            for cache in promptCache {
                cache.filterBatched(batchIndices: keepArr)
            }
        }

        uids = keep.map { uids[$0] }
        tokens = keep.map { tokens[$0] }
        samplers = keep.map { samplers[$0] }
        maxTokens = keep.map { maxTokens[$0] }
        stateMachines = keep.map { stateMachines[$0] }
        matcherStates = keep.map { matcherStates[$0] }
        numTokens = keep.map { numTokens[$0] }
        if !keep.isEmpty {
            nextTokens = take(nextTokens, keepArr, axis: 0)
        }
    }

    /// In-place: append `other`'s rows to this batch. Per-layer caches are
    /// concatenated via `BatchedCache.extendBatched`.
    public func extend(_ other: GenerationBatch) {
        precondition(
            promptCache.count == other.promptCache.count,
            "Cannot extend with a batch that has a different layer count"
        )
        for (a, b) in zip(promptCache, other.promptCache) {
            a.extendBatched(b)
        }
        uids.append(contentsOf: other.uids)
        tokens.append(contentsOf: other.tokens)
        samplers.append(contentsOf: other.samplers)
        maxTokens.append(contentsOf: other.maxTokens)
        stateMachines.append(contentsOf: other.stateMachines)
        matcherStates.append(contentsOf: other.matcherStates)
        numTokens.append(contentsOf: other.numTokens)
        nextTokens = concatenated([nextTokens, other.nextTokens], axis: 0)
    }

    public var isEmpty: Bool { uids.isEmpty }
    public var batchSize: Int { uids.count }

    /// One forward pass + per-row sample, double-buffered like upstream
    /// `mlx_lm.generate.GenerationBatch._step`.
    ///
    /// `nextTokens` is treated as the *current* token batch to return from
    /// this call. We immediately feed it back through the model, sample the
    /// following token batch, and `asyncEval` that future batch before
    /// synchronously materializing the current tokens for CPU-side stop
    /// detection / response dispatch.
    private func step() -> [Int] {
        let currentTokens = nextTokens
        let inputs = currentTokens[0..., .newAxis]

        let logits = model.callAsFunction(inputs, cache: promptCache.map { $0 as any KVCache })

        // [B, 1, vocab] -> [B, vocab]
        let stepLogits = logits[.ellipsis, -1, 0...]

        let sampledTokens: MLXArray
        if samplers.contains(where: { $0 != nil }) {
            let logprobs = stepLogits - logSumExp(stepLogits, axis: -1, keepDims: true)
            var samples: [MLXArray] = []
            samples.reserveCapacity(uids.count)
            for i in 0 ..< uids.count {
                let rowLogprobs = logprobs[i ..< (i + 1), 0...]
                let sampler = samplers[i] ?? fallbackSampler
                samples.append(sampler(rowLogprobs))
            }
            sampledTokens = concatenated(samples, axis: 0)
        } else {
            // Greedy fast path. Avoid the full-vocabulary logSumExp when
            // all rows are greedy: argMax(logits) == argMax(logprobs),
            // and Swift does not currently expose logprobs downstream.
            // This removes one expensive reduction kernel per decode step.
            sampledTokens = argMax(stepLogits, axis: -1)
        }

        // Start computing the next token before forcing the current token
        // values back to the CPU. This overlaps GPU work with the CPU
        // extraction / response-building path.
        nextTokens = sampledTokens
        asyncEval(sampledTokens)

        eval(currentTokens)
        let stepTokens = currentTokens.asArray(UInt32.self).map { Int($0) }

        for (i, t) in stepTokens.enumerated() {
            tokens[i].append(t)
        }

        return stepTokens
    }
}
