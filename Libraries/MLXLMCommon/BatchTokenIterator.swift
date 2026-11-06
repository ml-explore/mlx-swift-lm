// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN

/// Tokens for a batch of prompts, decoded together, with rows joining and leaving at any step.
///
/// The batched form of ``TokenIterator``: port of mlx-vlm's `BatchGenerator` and
/// `GenerationBatch` (`mlx_vlm/generate/ar.py` at mlx-vlm dcea787,
/// https://github.com/Blaizzy/mlx-vlm/blob/dcea787/mlx_vlm/generate/ar.py#L1089) and
/// mlx-lm's (`mlx_lm/generate.py`), without their prompt scheduling. Prompts queued with
/// ``insert(prompts:maxTokens:caches:samplers:processors:)`` are prefilled on the next
/// call to ``next()``, one at a time through the model's own
/// prefill, then their caches are merged into the decoding batch. Every ``next()`` decodes
/// one token for every row; a row leaves when it emits a stop token or reaches its limit, or
/// when it is removed.
///
/// One sampler draws the whole `[B, V]` batch when every row shares one; otherwise each row
/// is sampled on its own, so a row with a seeded sampler draws the tokens it would draw
/// alone. Logit processors run per row.
///
/// Model-owned state is carried for a single row. Multiple rows require nil or empty state:
/// arbitrary per-request state cannot be merged or filtered without a model-specific contract.
public struct BatchTokenIterator {

    /// One decoded token of one row.
    public struct Response: Sendable, Equatable {
        public let uid: Int
        public let token: Int
        /// Set on the row's last token: the stop token itself, or the last one within the limit.
        public let finishReason: FinishReason?
    }

    public enum FinishReason: Sendable, Equatable {
        case stop
        case length
    }

    public enum StateError: Error, LocalizedError, Equatable {
        case modelStateRequiresSingleRow
        case invalidatedByModelState

        public var errorDescription: String? {
            switch self {
            case .modelStateRequiresSingleRow:
                "Model-owned state requires a single row; remove the queued rows before continuing."
            case .invalidatedByModelState:
                "The model returned state while decoding multiple rows. Remove all rows before continuing."
            }
        }
    }

    private let model: any LanguageModel
    private let sampler: LogitSampler
    private let processor: LogitProcessor?
    private let stopTokens: Set<Int>
    private let maxTokens: Int
    private let prefill: PrefillParameters

    private struct Row {
        let uid: Int
        var sampler: LogitSampler
        var processor: LogitProcessor?
        let maxTokens: Int
        var generated = 0
    }

    private struct Pending {
        let row: Row
        let prompt: [Int]
        let cache: [KVCache]?
    }

    private var pending: [Pending] = []
    private var uidCount = 0

    /// The rows being decoded.
    private var rows: [Row] = []
    /// Their caches, one batched cache per layer.
    private var cache: [KVCache] = []
    /// The tokens the next step feeds: each row's sampled token.
    private var nextTokens: MLXArray?
    private var state: LMOutput.State?
    private var invalidatedByModelState = false

    /// - Parameters:
    ///   - model: the model
    ///   - sampler: the sampler for rows inserted without one
    ///   - processor: the processor rows inserted without one get a copy of
    ///   - stopTokens: tokens that end a row
    ///   - maxTokens: the token limit for rows inserted without one
    ///   - prefill: how prompts are prefilled
    public init(
        model: any LanguageModel, sampler: LogitSampler = ArgMaxSampler(),
        processor: LogitProcessor? = nil, stopTokens: Set<Int> = [], maxTokens: Int = 128,
        prefill: PrefillParameters = .init()
    ) {
        self.model = model
        self.sampler = sampler
        self.processor = processor
        self.stopTokens = stopTokens
        self.maxTokens = maxTokens
        self.prefill = prefill
    }

    /// The uids of every row decoding or queued, in insertion order.
    public var uids: [Int] { rows.map(\.uid) + pending.map(\.row.uid) }

    /// Queues prompts; they are prefilled on the next ``next()``.
    ///
    /// - Parameters:
    ///   - prompts: the token prompts, each at least one token
    ///   - maxTokens: a token limit per prompt, or the iterator's for all
    ///   - caches: a prompt cache per prompt to continue from, as ``LanguageModel/newCache(parameters:)``
    ///     makes it, or `nil` for a fresh one. The batch keeps them; do not use them elsewhere.
    ///   - samplers: a sampler per prompt, or the iterator's
    ///   - processors: a processor per prompt, or a copy of the iterator's
    /// - Returns: one uid per prompt, in order
    public mutating func insert(
        prompts: [[Int]], maxTokens: [Int]? = nil, caches: [[KVCache]?]? = nil,
        samplers: [LogitSampler?]? = nil, processors: [LogitProcessor?]? = nil
    ) -> [Int] {
        let count = prompts.count
        precondition(maxTokens.map { $0.count == count } ?? true, "one limit per prompt")
        precondition(caches.map { $0.count == count } ?? true, "one cache per prompt")
        precondition(samplers.map { $0.count == count } ?? true, "one sampler per prompt")
        precondition(processors.map { $0.count == count } ?? true, "one processor per prompt")
        var uids: [Int] = []
        for (index, prompt) in prompts.enumerated() {
            precondition(!prompt.isEmpty, "a prompt needs at least one token")
            let limit = maxTokens?[index] ?? self.maxTokens
            precondition(limit > 0, "a row's token limit must be positive")
            var processor = processors?[index] ?? self.processor?.copy()
            processor?.prompt(MLXArray(prompt.map { Int32($0) }))
            let row = Row(
                uid: uidCount, sampler: samplers?[index] ?? sampler, processor: processor,
                maxTokens: limit)
            pending.append(Pending(row: row, prompt: prompt, cache: caches?[index]))
            uids.append(uidCount)
            uidCount += 1
        }
        return uids
    }

    /// Drops rows, queued or decoding.
    public mutating func remove(_ uids: [Int]) {
        let removed = Set(uids)
        pending.removeAll { removed.contains($0.row.uid) }
        let keep = rows.indices.filter { !removed.contains(rows[$0].uid) }
        if keep.count < rows.count {
            filter(keeping: keep)
        }
    }

    /// Prefills the queued prompts and decodes one token for every row.
    ///
    /// - Returns: a token per decoding row, in batch order; empty once nothing is left
    /// - Throws: `KVCacheError` when caches cannot be batched or extended compatibly, or ``StateError`` when
    ///   multiple rows need model-owned state. An error after a batched forward invalidates
    ///   the batch because its cache writes cannot be undone; remove all rows before resuming.
    public mutating func next() throws -> [Response] {
        guard !invalidatedByModelState else { throw StateError.invalidatedByModelState }
        if !pending.isEmpty {
            try admit()
        }
        guard !rows.isEmpty else { return [] }

        // Drain the forward's autoreleased temporaries every step, as TokenIterator does.
        let tokens = try autoreleasepool { try step() }
        var responses: [Response] = []
        var keep: [Int] = []
        for (index, token) in tokens.enumerated() {
            rows[index].generated += 1
            let reason: FinishReason? =
                stopTokens.contains(token)
                ? .stop : rows[index].generated >= rows[index].maxTokens ? .length : nil
            responses.append(Response(uid: rows[index].uid, token: token, finishReason: reason))
            if reason == nil {
                keep.append(index)
            }
        }
        if keep.count < rows.count {
            filter(keeping: keep)
        }
        return responses
    }

    /// Prefills the queued prompts separately and joins them to the rows.
    private mutating func admit() throws {
        guard state?.isEmpty != false else { throw StateError.modelStateRequiresSingleRow }
        let admitted = pending
        // Check topology before a model can index a supplied cache during prefill.
        func preservesTypes(_ original: KVCache, _ copy: KVCache) -> Bool {
            guard type(of: original) == type(of: copy) else { return false }
            if let original = original as? CacheList, let copy = copy as? CacheList {
                return original.children.count == copy.children.count
                    && zip(original.children, copy.children).allSatisfy(preservesTypes)
            }
            return true
        }
        let caches = try admitted.map { pending in
            try pending.cache?.map { original in
                let copy = original.copy()
                guard preservesTypes(original, copy) else {
                    throw KVCacheError(message: "A prompt cache copy must preserve its exact types")
                }
                return copy
            } ?? model.newCache(parameters: nil)
        }
        let layerCount = rows.isEmpty ? caches[0].count : cache.count
        guard caches.allSatisfy({ $0.count == layerCount }) else {
            throw KVCacheError(message: "Every row must share the active cache's layer count")
        }

        // Each prompt goes into its own cache first, as TokenIterator feeds it: the model's
        // prefill (chunked) and then the remainder it leaves, whose last position's logits
        // give the row's first sample. mlx-vlm's chunked prefill likewise stops short of the
        // last token and feeds it in the step that samples:
        // https://github.com/Blaizzy/mlx-vlm/blob/dcea787/mlx_vlm/generate/ar.py#L2358-L2364 The rows merge after, every one prefilled, so a
        // one-token prompt joins a longer one on any cache type. Rows are not prefilled
        // together right-padded: the recurrent models' scans do not honour the prepared
        // lengths yet.
        var firstLogits: [MLXArray] = []
        var admittedState: LMOutput.State?
        for (pending, cache) in zip(admitted, caches) {
            let input = LMInput(tokens: MLXArray(pending.prompt.map { Int32($0) }))
            let output: LMOutput
            if let prefix = prefill.prefix {
                prefix(KVCacheStorage(cache, plan: .disabled).processedTokenCount)
            }
            switch try model.prepare(input, cache: cache, state: nil, prefill: prefill) {
            case .tokens(let remainder):
                output = model.nextTokenLogits(remainder[text: .newAxis], cache: cache, state: nil)
                // A `.tokens` prefill reports its own chunks only; the remainder completes
                // the prompt (a `.logits` prefill reports its own terminal).
                let total = pending.prompt.count
                prefill.progress?(total, total)
            case .logits(let prepared):
                output = prepared
            }
            if output.state?.isEmpty == false {
                guard rows.count + admitted.count == 1 else {
                    throw StateError.modelStateRequiresSingleRow
                }
                admittedState = output.state
            }
            let logits = output.logits[0..., -1, 0...]
            eval(cache)
            eval(logits)
            firstLogits.append(logits)
        }
        // Validate views of individual rows before bulk merge can concatenate incompatible
        // arrays. Compare every pair: an empty first row may not describe later rows' slots.
        let rowBatches = try caches.map { row in
            try row.map { try mergeCaches([$0]) }
        }
        for index in rowBatches.indices {
            for layer in 0 ..< layerCount {
                if !rows.isEmpty {
                    try cache[layer].validateExtension(other: rowBatches[index][layer])
                }
                for previous in 0 ..< index {
                    try rowBatches[previous][layer].validateExtension(
                        other: rowBatches[index][layer])
                }
            }
        }
        let merged: [KVCache]
        if rowBatches.count == 1 {
            merged = rowBatches[0]
        } else {
            merged = try (0 ..< layerCount).map { layer in
                try mergeCaches(caches.map { $0[layer] })
            }
        }
        if !rows.isEmpty {
            for layer in 0 ..< layerCount {
                try cache[layer].validateExtension(other: merged[layer])
            }
        }

        // The new rows sample their first tokens alone: the decoding rows' next tokens are
        // sampled already.
        var batch = BatchTokenIterator(
            model: model, sampler: sampler, processor: processor, stopTokens: stopTokens,
            maxTokens: maxTokens, prefill: prefill)
        batch.rows = admitted.map(\.row)
        batch.cache = merged
        batch.state = admittedState
        batch.nextTokens = batch.sampleNext(concatenated(firstLogits))

        if rows.isEmpty {
            rows = batch.rows
            cache = batch.cache
            nextTokens = batch.nextTokens
            state = batch.state
        } else {
            for (layer, added) in zip(cache, batch.cache) {
                layer.extend(other: added)
            }
            rows += batch.rows
            nextTokens = concatenated([nextTokens!, batch.nextTokens!])
        }
        pending = []
    }

    /// Feeds the next tokens, samples their successors and returns the fed tokens.
    private mutating func step() throws -> [Int] {
        let current = nextTokens!
        let output = model.nextTokenLogits(
            .init(tokens: current[0..., .newAxis]), cache: cache, state: state)
        if output.state?.isEmpty == false, rows.count > 1 {
            invalidatedByModelState = true
            throw StateError.invalidatedByModelState
        }
        state = output.state
        nextTokens = sampleNext(output.logits[0..., -1, 0...])
        return current.asArray(Int32.self).map { Int($0) }
    }

    /// Every row's next token from its `[B, V]` logits, through the rows' processors.
    private mutating func sampleNext(_ logits: MLXArray) -> MLXArray {
        var logits = logits
        if rows.contains(where: { $0.processor != nil }) {
            logits = concatenated(
                rows.indices.map { index in
                    let row = logits[index ..< index + 1]
                    return rows[index].processor?.process(logits: row) ?? row
                })
        }
        // Int32 in the graph, so reading the tokens back next step converts nothing.
        let sampled = sample(logits).asType(.int32)
        for index in rows.indices {
            rows[index].processor?.didSample(token: sampled[index ..< index + 1])
        }
        // The cache state settles with the token, as TokenIterator does, so an unevaluated
        // chain of updates never keeps earlier steps alive.
        asyncEval([sampled] + cache.flatMap { $0.state })
        return sampled
    }

    /// One batched draw when every row shares one of the library's samplers, which take a
    /// `[B, V]` batch; otherwise one draw per row. A row's `TopPSampler` draws from
    /// log-probabilities normalized once for the batch; any other sampler gets the row's
    /// raw logits, as ``LogitSampler/sample(logits:)`` promises.
    private func sample(_ logits: MLXArray) -> MLXArray {
        let first = rows[0].sampler
        if rows.allSatisfy({ $0.sampler.sharesState(with: first) }) {
            return first.sample(logits: logits)
        }
        var logProbabilities: MLXArray? = nil
        return concatenated(
            rows.indices.map { index -> MLXArray in
                if let sampler = rows[index].sampler as? TopPSampler {
                    let normalized =
                        logProbabilities ?? TopPSampler.normalizedLogProbabilities(logits)
                    logProbabilities = normalized
                    return sampler.sample(logProbabilities: normalized[index ..< index + 1])
                }
                return rows[index].sampler.sample(logits: logits[index ..< index + 1])
                    .reshaped([-1])
            })
    }

    private mutating func filter(keeping keep: [Int]) {
        rows = keep.map { rows[$0] }
        guard !keep.isEmpty else {
            cache = []
            nextTokens = nil
            state = nil
            invalidatedByModelState = false
            return
        }
        let indices = MLXArray(keep.map { Int32($0) })
        for layer in cache {
            layer.filter(batchIndices: indices)
        }
        nextTokens = nextTokens?[indices]
    }
}

extension LogitSampler {
    /// Whether `other` is one of the library's samplers drawing from this sampler's random
    /// state, so one call can sample a batch. Never true for a sampler from outside, whose
    /// handling of a `[B, V]` batch is unknown.
    fileprivate func sharesState(with other: LogitSampler) -> Bool {
        switch (self, other) {
        case (is ArgMaxSampler, is ArgMaxSampler):
            return true
        case (let a as TopPSampler, let b as TopPSampler):
            return a.randomState === b.randomState
        case (let a as CategoricalSampler, let b as CategoricalSampler):
            return a.randomState === b.randomState
        default:
            return false
        }
    }
}

/// Generates up to `maxTokens` tokens for every prompt as one batch. Port of mlx-lm's
/// `batch_generate`: the thin loop over ``BatchTokenIterator``.
///
/// - Parameters:
///   - model: the model
///   - prompts: the token prompts
///   - caches: a prompt cache per prompt to continue from, or `nil`
///   - maxTokens: the token limit per prompt
///   - stopTokens: tokens that end a prompt's generation; a stop token is not returned
///   - sampler: the sampler for every prompt
///   - processor: the processor every prompt gets a copy of
///   - prefill: how prompts are prefilled
/// - Returns: the generated tokens per prompt, in order
public func batchGenerate(
    model: any LanguageModel, prompts: [[Int]], caches: [[KVCache]?]? = nil,
    maxTokens: Int = 128, stopTokens: Set<Int> = [], sampler: LogitSampler = ArgMaxSampler(),
    processor: LogitProcessor? = nil, prefill: PrefillParameters = .init()
) throws -> [[Int]] {
    var iterator = BatchTokenIterator(
        model: model, sampler: sampler, processor: processor, stopTokens: stopTokens,
        maxTokens: maxTokens, prefill: prefill)
    let uids = iterator.insert(prompts: prompts, caches: caches)
    var results = [Int: [Int]](uniqueKeysWithValues: uids.map { ($0, []) })
    while true {
        let responses = try iterator.next()
        if responses.isEmpty { break }
        for response in responses where response.finishReason != .stop {
            results[response.uid, default: []].append(response.token)
        }
    }
    return uids.map { results[$0] ?? [] }
}
