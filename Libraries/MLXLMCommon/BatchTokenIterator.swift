// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN

/// What one ``BatchTokenIterator/next()`` call reports for one row.
///
/// Every row ends with exactly one terminal event, `.finished` or `.failed`, placed after its
/// last `.token` in the same array; its id never appears again. A row dropped with
/// ``BatchTokenIterator/remove(_:)`` gets no terminal event.
public enum BatchEvent: Sendable {
    /// Prefill progress for the queued row that got a prefill chunk this step, in prompt
    /// tokens. Emitted for a ``ResumablePrefillModel`` only, once per chunk; `processed ==
    /// total` marks the final chunk, after which the row joins the batch and its first `.token`
    /// comes with the next step. An earlier `.prefill` means the chunk was submitted to the
    /// GPU, not that it finished; the final one means the row's cache is ready. Other models
    /// prefill a queued prompt whole within one
    /// ``BatchTokenIterator/next()`` and report no `.prefill`.
    case prefill(id: Int, processed: Int, total: Int)
    /// A generated token. Stop tokens are never reported here.
    case token(id: Int, token: Int)
    /// The row ended normally: `.stop` on a stop token, carried in `stopToken`, or `.length`
    /// after the row's last `.token` within its limit.
    case finished(id: Int, reason: GenerateStopReason, stopToken: Int?)
    /// The row could not join the batch and produced no tokens: its supplied cache does not
    /// match the batch, its prefill failed, or the model's state allows only one row.
    case failed(id: Int, error: any Error)

    /// The row the event is about.
    public var id: Int {
        switch self {
        case .prefill(let id, _, _), .token(let id, _), .finished(let id, _, _),
            .failed(let id, _):
            id
        }
    }
}

/// Tokens for a batch of prompts, decoded together, with rows joining and leaving at any step.
///
/// The batched form of ``TokenIterator``: port of mlx-vlm's `BatchGenerator` and
/// `GenerationBatch` (`mlx_vlm/generate/ar.py` at mlx-vlm dcea787,
/// https://github.com/Blaizzy/mlx-vlm/blob/dcea787/mlx_vlm/generate/ar.py#L1089) and
/// mlx-lm's (`mlx_lm/generate.py`), without their prompt scheduling. Prompts queued with
/// ``insert(prompts:maxTokens:caches:samplers:processors:)`` are prefilled on the next
/// call to ``next()``, one at a time through the model's own
/// prefill, then their caches are merged into the decoding batch. Every ``next()`` decodes
/// one token for every row; a row leaves when it samples a stop token or reaches its limit,
/// or when it is removed.
///
/// For a ``ResumablePrefillModel``, each ``next()`` instead decodes first and then feeds at
/// most one prefill chunk of the first queued prompt (first in, first out), reporting it as
/// `.prefill`. A long prompt therefore delays the decoding rows by one chunk per step rather
/// than by its whole prefill. A step that only prefills returns `[.prefill(…)]`, so it is
/// never mistaken for idle. `PrefillParameters.progress` still reports each chunk.
///
/// The iterator clears MLX's buffer cache after each prefill chunk, as single-row prefill
/// does, but never while decoding: the cache is process-wide, so memory policy belongs to the
/// process owner, which can set `Memory.cacheLimit` or clear it on its own schedule.
///
/// ``next()`` returns the step's ``BatchEvent``s, or `nil` when the iterator is idle: no row
/// is decoding or queued. Unlike `IteratorProtocol`, `nil` does not mean exhausted; inserting
/// more prompts makes the next call return events again. Drive it with
/// `while let events = try iterator.next() { … }`.
///
/// The caller decides which tokens reach its consumer and reports each one with
/// ``recordEmittedToken(id:token:)``. A row's processor is finalized once: at the start of the
/// ``next()`` call after the row's terminal event, or when the row is removed.
///
/// One sampler draws the whole `[B, V]` batch when every row shares one; otherwise each row
/// is sampled on its own, so a row with a seeded sampler draws the tokens it would draw
/// alone. Logit processors run per row.
///
/// Model-owned state is carried for a single row. Multiple rows require nil or empty state:
/// arbitrary per-request state cannot be merged or filtered without a model-specific contract.
public struct BatchTokenIterator {

    public enum StateError: Error, LocalizedError, Equatable {
        case modelStateRequiresSingleRow
        case invalidatedByModelState

        public var errorDescription: String? {
            switch self {
            case .modelStateRequiresSingleRow:
                "Model-owned state requires a single row; this row cannot join the batch."
            case .invalidatedByModelState:
                "The model returned state while decoding multiple rows. Remove all rows before continuing."
            }
        }
    }

    /// A ``GenerateParameters`` setting that batched generation does not support.
    public enum ParameterError: Error, LocalizedError, Equatable {
        /// `maxKVSize` keeps a pinned prefix in its rotating caches.
        case maxKVSize
        /// A `kvCache` capacity with `preservedPrefixTokens` above zero.
        case pinnedPrefix(Int)
        /// A `kvCache` rewind reserve.
        case rewindReserve
        /// A cache strategy other than full precision.
        case strategy(KVCacheStrategyIdentifier)
        /// `logProbabilities`.
        case logProbabilities

        public var errorDescription: String? {
            switch self {
            case .maxKVSize:
                "maxKVSize pins a prefix of up to 4 tokens, which batched rotating caches do not support. Use kvCache with a capacity whose preservedPrefixTokens is 0."
            case .pinnedPrefix(let count):
                "Batched rotating caches do not support a pinned prefix; this capacity preserves \(count) tokens. Use preservedPrefixTokens: 0."
            case .rewindReserve:
                "Batched rotating caches do not support a rewind reserve."
            case .strategy(let strategy):
                "Batched generation does not support the \(strategy) KV-cache strategy."
            case .logProbabilities:
                "Batched generation does not report log probabilities."
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
        let id: Int
        var sampler: LogitSampler
        var processor: LogitProcessor?
        let maxTokens: Int
        var generated = 0
    }

    private struct Pending {
        let row: Row
        let prompt: [Int]
        let cache: [KVCache]?
        /// The row's prefill so far, for a ``ResumablePrefillModel``.
        var resume: Resume?
    }

    /// A queued row partway through a resumable prefill: its own cache and the chunks left.
    private struct Resume {
        let cache: [KVCache]
        let prompt: LMInput.Text
        let chunks: [Range<Int>]
        var next = 0
        /// The last chunk's marker, waited on after the next chunk is submitted, so at most two
        /// chunks are in flight. Not the cache arrays themselves: a cache that writes in place
        /// rebinds those objects to the next chunk's result. See `chunkMarker`.
        var submitted: MLXArray?
    }

    /// A queued row prefilled into its own cache, ready to join the batch.
    private struct Prefilled {
        let row: Row
        let cache: [KVCache]
        /// The row's cache, layer by layer, as a batch of one.
        let batches: [any BatchableKVCache]
        let logits: MLXArray
    }

    private var pending: [Pending] = []
    private var nextID = 0

    /// The rows being decoded.
    private var rows: [Row] = []
    /// Their caches, one batched cache per layer. Admission merges each layer into one and
    /// throws `KVCacheError` for a layer without a batched form.
    private var cache: [any BatchableKVCache] = []
    /// The tokens the next step feeds: each row's sampled token.
    private var nextTokens: MLXArray?
    private var state: LMOutput.State?
    private var invalidatedByModelState = false
    /// `.failed` events from a ``next()`` call that then threw, returned by the next call.
    private var undelivered: [BatchEvent] = []

    /// A row's processor kept after the row ended, until it is finalized.
    private typealias Ended = (id: Int, processor: LogitProcessor)
    /// The processors of the rows whose terminal events the last ``next()`` returned. They are
    /// finalized at the start of the next call, so the caller can first record the tokens it
    /// delivered from that step.
    private var ended: [Ended] = []
    /// The processors of the rows ending in the current ``next()`` call.
    private var ending: [Ended] = []
    /// The processors of the rows in `undelivered`.
    private var undeliveredEnded: [Ended] = []
    /// What ``LanguageModel/newCache(parameters:)`` builds fresh rows' caches from.
    private var cacheParameters: GenerateParameters?

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

    /// An iterator configured from `parameters`, as ``TokenIterator`` is for one sequence.
    ///
    /// The rows inserted without their own get `parameters`' sampler, a copy of its processor
    /// and its token limit; `maxTokens == nil` means no limit, so a row runs until it samples
    /// a stop token or is removed. Fresh rows' caches come from
    /// ``LanguageModel/newCache(parameters:)``, so a `kvCache` capacity gives every attention
    /// layer a rotating cache of that size. Prompts are prefilled with `parameters.prefill`.
    ///
    /// - Throws: ``ParameterError`` for a setting batched generation does not support:
    ///   `maxKVSize`, a capacity with a pinned prefix, a rewind reserve, a cache strategy other
    ///   than full precision, or `logProbabilities`; `KVCacheConfigurationError` for an
    ///   invalid cache configuration.
    public init(
        model: any LanguageModel, parameters: GenerateParameters, stopTokens: Set<Int> = []
    ) throws {
        guard parameters.maxKVSize == nil else { throw ParameterError.maxKVSize }
        guard parameters.logProbabilities == nil else { throw ParameterError.logProbabilities }
        let configuration = try parameters.resolvedKVCacheConfiguration()
        if let configuration {
            if let capacity = configuration.capacity, capacity.preservedPrefixTokens > 0 {
                throw ParameterError.pinnedPrefix(capacity.preservedPrefixTokens)
            }
            guard configuration.rewind == nil else { throw ParameterError.rewindReserve }
            guard configuration.strategy == .fullPrecision else {
                throw ParameterError.strategy(configuration.strategy.identifier)
            }
        }
        self.init(
            model: model, sampler: parameters.sampler(), processor: parameters.processor(),
            stopTokens: stopTokens, maxTokens: parameters.maxTokens ?? .max,
            prefill: parameters.prefill)
        cacheParameters = configuration == nil ? nil : parameters
    }

    /// The ids of every row decoding or queued, in insertion order.
    public var ids: [Int] { rows.map(\.id) + pending.map(\.row.id) }

    /// The ids of the queued rows, in the order they are prefilled.
    public var pendingIDs: [Int] { pending.map(\.row.id) }

    /// Queues prompts; they are prefilled on the next ``next()``.
    ///
    /// - Parameters:
    ///   - prompts: the token prompts, each at least one token
    ///   - maxTokens: a token limit per prompt, or the iterator's for all
    ///   - caches: a prompt cache per prompt to continue from, as ``LanguageModel/newCache(parameters:)``
    ///     makes it, or `nil` for a fresh one. The batch works on a copy; the supplied cache
    ///     is not changed.
    ///   - samplers: a sampler per prompt, or the iterator's
    ///   - processors: a processor per prompt, or a copy of the iterator's
    /// - Returns: one id per prompt, in order
    public mutating func insert(
        prompts: [[Int]], maxTokens: [Int]? = nil, caches: [[KVCache]?]? = nil,
        samplers: [LogitSampler?]? = nil, processors: [LogitProcessor?]? = nil
    ) -> [Int] {
        let count = prompts.count
        precondition(maxTokens.map { $0.count == count } ?? true, "one limit per prompt")
        precondition(caches.map { $0.count == count } ?? true, "one cache per prompt")
        precondition(samplers.map { $0.count == count } ?? true, "one sampler per prompt")
        precondition(processors.map { $0.count == count } ?? true, "one processor per prompt")
        var ids: [Int] = []
        for (index, prompt) in prompts.enumerated() {
            precondition(!prompt.isEmpty, "a prompt needs at least one token")
            let limit = maxTokens?[index] ?? self.maxTokens
            precondition(limit > 0, "a row's token limit must be positive")
            var processor = processors?[index] ?? self.processor?.copy()
            processor?.prompt(MLXArray(prompt.map { Int32($0) }))
            let row = Row(
                id: nextID, sampler: samplers?[index] ?? sampler, processor: processor,
                maxTokens: limit)
            pending.append(Pending(row: row, prompt: prompt, cache: caches?[index], resume: nil))
            ids.append(nextID)
            nextID += 1
        }
        return ids
    }

    /// Queues one prompt; it is prefilled on the next ``next()``. The single-row form of
    /// ``insert(prompts:maxTokens:caches:samplers:processors:)``.
    ///
    /// - Returns: the row's id
    @discardableResult
    public mutating func insert(
        prompt: [Int], maxTokens: Int? = nil, cache: [KVCache]? = nil,
        sampler: LogitSampler? = nil, processor: LogitProcessor? = nil
    ) -> Int {
        insert(
            prompts: [prompt], maxTokens: maxTokens.map { [$0] }, caches: [cache],
            samplers: [sampler], processors: [processor])[0]
    }

    /// Drops rows, queued or decoding, and finalizes their processors now
    /// (`LogitProcessor.finalizeGeneration()`). A removed row gets no terminal event.
    ///
    /// Removing a row that ended in the last step finalizes its processor now instead of at the
    /// next ``next()``. Ids that were never inserted or are already finalized are ignored.
    public mutating func remove(_ ids: [Int]) {
        let removed = Set(ids)
        for item in pending where removed.contains(item.row.id) {
            var processor = item.row.processor
            processor?.finalizeGeneration()
        }
        pending.removeAll { removed.contains($0.row.id) }
        for index in rows.indices where removed.contains(rows[index].id) {
            rows[index].processor?.finalizeGeneration()
        }
        let keep = rows.indices.filter { !removed.contains(rows[$0].id) }
        if keep.count < rows.count {
            filter(keeping: keep)
        }
        func finalizeRemoved(_ list: inout [Ended]) {
            for var entry in list where removed.contains(entry.id) {
                entry.processor.finalizeGeneration()
            }
            list.removeAll { removed.contains($0.id) }
        }
        finalizeRemoved(&ended)
        finalizeRemoved(&undeliveredEnded)
    }

    /// Records that the caller delivered `token` from row `id`: forwards it to the row's
    /// processor (`LogitProcessor.didEmit(token:)`), as ``TokenIterator/recordEmittedToken(_:)``
    /// does for one sequence.
    ///
    /// Call it for every token you deliver, including a stop token you choose to deliver, and
    /// only for those. A row that ended keeps its processor until the next ``next()`` call,
    /// which finalizes it, so record the tokens from one step before asking for the next.
    /// Ids that are neither decoding nor ended in the last step are ignored.
    public mutating func recordEmittedToken(id: Int, token: Int) {
        if let index = rows.firstIndex(where: { $0.id == id }) {
            rows[index].processor?.didEmit(token: token)
        } else if let index = ended.firstIndex(where: { $0.id == id }) {
            ended[index].processor.didEmit(token: token)
        }
    }

    /// Finalizes the processors of the rows that ended in the last step.
    private mutating func finalizeEnded() {
        for var entry in ended {
            entry.processor.finalizeGeneration()
        }
        ended = []
    }

    /// Keeps `row`'s processor, if it has one, until the step reporting its end is handled.
    private mutating func end(_ row: Row) {
        if let processor = row.processor {
            ending.append((row.id, processor))
        }
    }

    /// Prefills the queued prompts and decodes one token for every row.
    ///
    /// A queued row that cannot join the batch ends with a `.failed` event and the other rows
    /// carry on: its supplied cache does not match the batch (`KVCacheError`), its prefill
    /// throws, or the model's state allows one row only (``StateError/modelStateRequiresSingleRow``).
    ///
    /// The processors of rows that ended in the previous step are finalized first
    /// (`LogitProcessor.finalizeGeneration()`), once each; see ``recordEmittedToken(id:token:)``.
    ///
    /// - Returns: the step's events, in batch order and never empty, or `nil` when no row is
    ///   decoding or queued
    /// - Throws: ``StateError/invalidatedByModelState`` when the model returns state while
    ///   decoding several rows. An error from a batched forward invalidates the batch because
    ///   its cache writes cannot be undone; remove all rows before resuming.
    ///   `CancellationError` when a prefill stops because the current task was cancelled:
    ///   admission stops before any sampling or decoding, and the queued rows that have not
    ///   failed stay queued. `.failed` events from a throwing call are returned by the next one.
    public mutating func next() throws -> [BatchEvent]? {
        finalizeEnded()
        // Drain admission and decode temporaries, including rejected admissions.
        return try autoreleasepool {
            if !undelivered.isEmpty {
                let events = undelivered
                ended = undeliveredEnded
                undelivered = []
                undeliveredEnded = []
                return events
            }
            guard !invalidatedByModelState else { throw StateError.invalidatedByModelState }
            guard !rows.isEmpty || !pending.isEmpty else { return nil }

            if let resumable = model as? any ResumablePrefillModel {
                return try resumableStep(resumable)
            }
            // Every queued row either fails here or joins the rows.
            var events = pending.isEmpty ? [] : try admit()
            if !rows.isEmpty {
                do {
                    events += try decode()
                } catch {
                    undelivered = events
                    undeliveredEnded = ending
                    ending = []
                    throw error
                }
            }
            ended = ending
            ending = []
            return events
        }
    }

    /// One step for a ``ResumablePrefillModel``: the decode step first, then at most one chunk
    /// of the first queued row's prefill. The final chunk is the forward whose logits give the
    /// row's first sample; the row then joins the rows, and its first token comes with the
    /// next step.
    private mutating func resumableStep(_ model: any ResumablePrefillModel) throws
        -> [BatchEvent]
    {
        var events: [BatchEvent] = []
        if !rows.isEmpty {
            do {
                events = try decode()
            } catch {
                undelivered = []
                undeliveredEnded = ending
                ending = []
                throw error
            }
        }
        if !pending.isEmpty {
            if Task.isCancelled {
                // Submit no prefill work for a cancelled task; the queued rows stay queued.
                guard !events.isEmpty else { throw CancellationError() }
            } else {
                events += prefillStep(model)
            }
        }
        ended = ending
        ending = []
        return events
    }

    /// Feeds one chunk of the first queued row's prefill. Returns its `.prefill` progress, and
    /// on the final chunk the row joins the rows; or a `.failed` event when it cannot.
    private mutating func prefillStep(_ model: any ResumablePrefillModel) -> [BatchEvent] {
        var item = pending[0]
        let id = item.row.id
        let total = item.prompt.count
        do {
            var resume: Resume
            if let started = item.resume {
                resume = started
            } else {
                let rowCache = try rowCache(for: item, joining: [], admittedState: nil)
                if let prefix = prefill.prefix {
                    prefix(KVCacheStorage(rowCache, plan: .disabled).processedTokenCount)
                }
                let prompt = LMInput.Text(tokens: MLXArray(item.prompt.map { Int32($0) }))
                resume = Resume(
                    cache: rowCache, prompt: prompt,
                    chunks: model.prefillChunks(for: prompt, prefill: prefill))
            }
            if resume.next < resume.chunks.count {
                let range = resume.chunks[resume.next]
                autoreleasepool {
                    model.prefillChunk(resume.prompt, range: range, cache: resume.cache)
                    // Submit this chunk, even for a model that leaves its cache writes lazy,
                    // then wait for the previous one: the CPU builds the next chunk's graph
                    // while the GPU evaluates this one, with at most two chunks in flight.
                    let state = resume.cache.flatMap { $0.innerState() }
                    let marker = Self.chunkMarker(state)
                    asyncEval(state + [marker])
                    if let previous = resume.submitted { eval(previous) }
                    resume.submitted = marker
                    // As PrefillParameters.forEachChunk does after each chunk.
                    Memory.clearCache()
                }
                resume.next += 1
                prefill.progress?(range.upperBound, total)
                item.resume = resume
                pending[0] = item
                return [.prefill(id: id, processed: range.upperBound, total: total)]
            }
            // The remainder the chunks leave, as the model's `prepare` leaves it.
            let remainder = resume.prompt[(resume.chunks.last?.upperBound ?? 0)...]
            let output = model.nextTokenLogits(
                remainder[text: .newAxis], cache: resume.cache, state: nil)
            prefill.progress?(total, total)
            let (row, rowState) = try joinable(
                item, cache: resume.cache, output: output, joining: [])
            pending.removeFirst()
            return [.prefill(id: id, processed: total, total: total)]
                + join([row], state: rowState)
        } catch {
            pending.removeFirst()
            end(item.row)
            return [.failed(id: id, error: error)]
        }
    }

    /// A small array of its own that is ready once the chunk's cache writes, `state`, are: the
    /// first element of each, copied together. It must not be a view of a cache buffer, which
    /// would keep the buffer alive and stop the next chunk from writing it in place (MLX
    /// returns its input from a reduction over one element, and from a concatenation of one
    /// array, so a constant element makes the concatenation copy).
    private static func chunkMarker(_ state: [MLXArray]) -> MLXArray {
        concatenated(
            state.map { $0.reshaped([-1])[..<1].asType(.float32) }
                + [MLXArray.zeros([1], dtype: .float32)])
    }

    /// Feeds every row its next token and reports the tokens, finishing the rows that end.
    private mutating func decode() throws -> [BatchEvent] {
        let tokens = try step()
        var events: [BatchEvent] = []
        var keep: [Int] = []
        for (index, token) in tokens.enumerated() {
            let id = rows[index].id
            if stopTokens.contains(token) {
                events.append(.finished(id: id, reason: .stop, stopToken: token))
                end(rows[index])
                continue
            }
            rows[index].generated += 1
            events.append(.token(id: id, token: token))
            if rows[index].generated >= rows[index].maxTokens {
                events.append(.finished(id: id, reason: .length, stopToken: nil))
                end(rows[index])
            } else {
                keep.append(index)
            }
        }
        if keep.count < rows.count {
            filter(keeping: keep)
        }
        return events
    }

    /// Prefills the queued prompts separately, in queue order, and joins the ones that can
    /// to the rows. Returns a `.failed` event for each one that cannot; the active rows and
    /// the other queued rows are unaffected by it.
    private mutating func admit() throws -> [BatchEvent] {
        let queued = pending
        pending = []
        var events: [BatchEvent] = []
        var admitted: [Prefilled] = []
        var admittedState: LMOutput.State?
        for item in queued {
            do {
                let (row, rowState) = try prefillRow(
                    item, joining: admitted, admittedState: admittedState)
                if rowState != nil { admittedState = rowState }
                admitted.append(row)
            } catch is CancellationError where Task.isCancelled {
                // The task was cancelled: stop before sampling or decoding anything. Every row
                // that has not failed goes back to the queue; the failures are returned by the
                // next call.
                let failed = Set(events.map(\.id))
                pending = queued.filter { !failed.contains($0.row.id) }
                undelivered = events
                undeliveredEnded = ending
                ending = []
                throw CancellationError()
            } catch {
                events.append(.failed(id: item.row.id, error: error))
                end(item.row)
            }
        }
        guard !admitted.isEmpty else { return events }
        return events + join(admitted, state: admittedState)
    }

    /// Merges prefilled rows into the batch and samples their first tokens. Returns a
    /// `.failed` event for each when they cannot merge, leaving the batch unchanged.
    private mutating func join(_ admitted: [Prefilled], state admittedState: LMOutput.State?)
        -> [BatchEvent]
    {
        // Each row was checked against the active rows and the rows before it; one bulk merge
        // avoids copying KV buffers by extending each prefix repeatedly. No sampling or
        // active-cache changes happen until every layer merges.
        let merged: [any BatchableKVCache]
        do {
            merged = try admitted[0].cache.indices.map { layer in
                let joined =
                    admitted.count == 1
                    ? admitted[0].batches[layer] : try mergeCaches(admitted.map { $0.cache[layer] })
                if !rows.isEmpty {
                    try cache[layer].validateExtension(other: joined)
                }
                return joined
            }
        } catch {
            for row in admitted { end(row.row) }
            return admitted.map { .failed(id: $0.row.id, error: error) }
        }

        // The new rows sample their first tokens alone: the decoding rows' next tokens are
        // sampled already.
        var batch = BatchTokenIterator(
            model: model, sampler: sampler, processor: processor, stopTokens: stopTokens,
            maxTokens: maxTokens, prefill: prefill)
        batch.rows = admitted.map(\.row)
        batch.cache = merged
        batch.state = admittedState
        batch.nextTokens = batch.sampleNext(concatenated(admitted.map(\.logits)))

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
        return []
    }

    /// Prefills one queued prompt into its own cache and checks that it can join the active
    /// rows and `admitted`, the rows already prefilled this step.
    ///
    /// - Returns: the prefilled row, and the model state its prefill returned, if any
    private func prefillRow(
        _ item: Pending, joining admitted: [Prefilled], admittedState: LMOutput.State?
    ) throws -> (Prefilled, LMOutput.State?) {
        let rowCache = try rowCache(for: item, joining: admitted, admittedState: admittedState)

        // The prompt goes into its own cache first, as TokenIterator feeds it: the model's
        // prefill (chunked) and then the remainder it leaves, whose last position's logits
        // give the row's first sample. mlx-vlm's chunked prefill likewise stops short of the
        // last token and feeds it in the step that samples:
        // https://github.com/Blaizzy/mlx-vlm/blob/dcea787/mlx_vlm/generate/ar.py#L2358-L2364 The rows merge after, every one prefilled, so a
        // one-token prompt joins a longer one on any cache type. Rows are not prefilled
        // together right-padded: the recurrent models' scans do not honour the prepared
        // lengths yet.
        let input = LMInput(tokens: MLXArray(item.prompt.map { Int32($0) }))
        let output: LMOutput
        if let prefix = prefill.prefix {
            prefix(KVCacheStorage(rowCache, plan: .disabled).processedTokenCount)
        }
        switch try model.prepare(input, cache: rowCache, state: nil, prefill: prefill) {
        case .tokens(let remainder):
            output = model.nextTokenLogits(remainder[text: .newAxis], cache: rowCache, state: nil)
            // A `.tokens` prefill reports its own chunks only; the remainder completes
            // the prompt (a `.logits` prefill reports its own terminal).
            let total = item.prompt.count
            prefill.progress?(total, total)
        case .logits(let prepared):
            output = prepared
        }
        return try joinable(item, cache: rowCache, output: output, joining: admitted)
    }

    /// The cache a queued row is prefilled into: a copy of its supplied cache, or a fresh one,
    /// checked against the active rows and `admitted`, the rows already prefilled this step.
    private func rowCache(
        for item: Pending, joining admitted: [Prefilled], admittedState: LMOutput.State?
    ) throws -> [KVCache] {
        // Model state is carried for one row only.
        guard state?.isEmpty != false, admittedState == nil else {
            throw StateError.modelStateRequiresSingleRow
        }
        // Check topology before a model can index a supplied cache during prefill.
        func preservesTypes(_ original: KVCache, _ copy: KVCache) -> Bool {
            guard type(of: original) == type(of: copy) else { return false }
            if let original = original as? CacheList, let copy = copy as? CacheList {
                return original.children.count == copy.children.count
                    && zip(original.children, copy.children).allSatisfy(preservesTypes)
            }
            return true
        }
        let rowCache =
            try item.cache?.map { original in
                let copy = original.copy()
                guard preservesTypes(original, copy) else {
                    throw KVCacheError(message: "A prompt cache copy must preserve its exact types")
                }
                return copy
            } ?? model.newCache(parameters: cacheParameters)
        // A supplied cache with fewer layers than the model would have the model index layers
        // it lacks, so the first row's is checked against a fresh cache; later rows match it.
        if item.cache != nil, rows.isEmpty, admitted.isEmpty,
            rowCache.count < (try model.newCache(parameters: cacheParameters).count)
        {
            throw KVCacheError(message: "A supplied cache needs a layer for every model layer")
        }
        let layerCount = rows.isEmpty ? admitted.first?.cache.count ?? rowCache.count : cache.count
        guard rowCache.count == layerCount else {
            throw KVCacheError(message: "Every row must share the active cache's layer count")
        }
        return rowCache
    }

    /// A prefilled row ready to join: `output` is its final prefill forward. Checks every
    /// layer against the active rows and `admitted`, so a row that cannot join fails alone.
    ///
    /// - Returns: the row, and the model state its prefill returned, if any
    private func joinable(
        _ item: Pending, cache rowCache: [KVCache], output: LMOutput, joining admitted: [Prefilled]
    ) throws -> (Prefilled, LMOutput.State?) {
        let rowState = output.state?.isEmpty == false ? output.state : nil
        if rowState != nil, !rows.isEmpty || !admitted.isEmpty {
            throw StateError.modelStateRequiresSingleRow
        }
        let logits = output.logits[0..., -1, 0...]
        eval(rowCache)
        eval(logits)

        // Check every layer against the active rows and the rows admitted before it, so a
        // row that cannot join fails alone.
        let batches = try rowCache.map { try mergeCaches([$0]) }
        for layer in batches.indices {
            for previous in admitted {
                try previous.batches[layer].validateExtension(other: batches[layer])
            }
            if !rows.isEmpty {
                try cache[layer].validateExtension(other: batches[layer])
            }
        }
        return (
            Prefilled(row: item.row, cache: rowCache, batches: batches, logits: logits), rowState
        )
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
/// - Throws: the error of the first prompt that cannot join the batch, or an error from a
///   batched forward
public func batchGenerate(
    model: any LanguageModel, prompts: [[Int]], caches: [[KVCache]?]? = nil,
    maxTokens: Int = 128, stopTokens: Set<Int> = [], sampler: LogitSampler = ArgMaxSampler(),
    processor: LogitProcessor? = nil, prefill: PrefillParameters = .init()
) throws -> [[Int]] {
    var iterator = BatchTokenIterator(
        model: model, sampler: sampler, processor: processor, stopTokens: stopTokens,
        maxTokens: maxTokens, prefill: prefill)
    let ids = iterator.insert(prompts: prompts, caches: caches)
    // On every exit, including a throw from next(): every row, including the ones that ended
    // in the last step, so each processor is finalized once.
    defer { iterator.remove(ids) }
    var results = [Int: [Int]](uniqueKeysWithValues: ids.map { ($0, []) })
    while let events = try iterator.next() {
        for event in events {
            switch event {
            case .token(let id, let token):
                // Every returned token reaches the row's processor, as emitted.
                iterator.recordEmittedToken(id: id, token: token)
                results[id, default: []].append(token)
            case .failed(_, let error):
                throw error
            case .prefill, .finished:
                break
            }
        }
    }
    return ids.map { results[$0] ?? [] }
}
