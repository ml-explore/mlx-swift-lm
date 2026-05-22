// Port of mlx_lm.generate.BatchGenerator.
// https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/generate.py

import Foundation
import MLX

/// Continuous-batching engine.
///
///   1. `insert(prompts:)` queues new requests and returns their UIDs.
///   2. `next()` runs one engine step: drains the queue into prefill,
///      runs one decode step, and emits per-row responses. Finished rows
///      are filtered out and their slots become available for new
///      admissions on the next call.
///   3. `close()` releases resources.
public final class BatchGenerator: @unchecked Sendable {

    public let model: any LanguageModel
    public let prefillStepSize: Int
    public let prefillBatchSize: Int
    public let completionBatchSize: Int
    public let defaultMaxTokens: Int

    public let defaultEosTokens: [[Int]]
    public let defaultSampler: RowSampler
    public let defaultStateMachine: SequenceStateMachine

    private var uidCounter: Int = 0
    private var unprocessed: [QueuedRequest] = []
    private var promptBatch: PromptProcessingBatch
    private var generationBatch: GenerationBatch?

    public private(set) var promptTokensProcessed: Int = 0
    public private(set) var generatedTokens: Int = 0

    public init(
        model: any LanguageModel,
        eosTokens: [[Int]] = [],
        defaultMaxTokens: Int = 128,
        prefillStepSize: Int = 2048,
        prefillBatchSize: Int = 8,
        completionBatchSize: Int = 32
    ) {
        self.model = model
        self.prefillStepSize = prefillStepSize
        self.prefillBatchSize = prefillBatchSize
        self.completionBatchSize = max(completionBatchSize, prefillBatchSize)
        self.defaultMaxTokens = defaultMaxTokens
        self.defaultEosTokens = eosTokens
        self.defaultSampler = greedySampler

        if eosTokens.isEmpty {
            self.defaultStateMachine = SequenceStateMachine()
        } else {
            self.defaultStateMachine = SequenceStateMachine(
                states: ["normal": eosTokens.map { (sequence: $0, next: nil) }],
                initial: "normal"
            )
        }
        self.promptBatch = PromptProcessingBatch.empty(
            model: model,
            prefillStepSize: prefillStepSize
        )
    }

    /// Append a batch of prompts. Returns the assigned UIDs in input order.
    @discardableResult
    public func insert(
        prompts: [[Int]],
        maxTokens: [Int]? = nil,
        samplers: [RowSampler?]? = nil,
        stateMachines: [SequenceStateMachine]? = nil
    ) -> [Int] {
        precondition(
            maxTokens == nil || maxTokens?.count == prompts.count,
            "maxTokens.count must equal prompts.count"
        )

        var assignedUids: [Int] = []
        assignedUids.reserveCapacity(prompts.count)

        for i in 0 ..< prompts.count {
            let uid = uidCounter
            uidCounter += 1
            assignedUids.append(uid)
            unprocessed.append(
                QueuedRequest(
                    uid: uid,
                    tokens: prompts[i],
                    maxTokens: maxTokens?[i] ?? defaultMaxTokens,
                    sampler: samplers?[i],
                    stateMachine: stateMachines?[i] ?? defaultStateMachine
                ))
        }
        return assignedUids
    }

    /// Run one engine step. Returns per-uid responses for the active rows;
    /// rows that finished on this step have a non-nil `finishReason`.
    ///
    /// Looped to drain the queue:
    /// `while gen.hasWork { for r in gen.next() { ... } }`
    public func next() -> [GenerationBatchResponse] {
        admitFromQueue()

        if let gen = generationBatch, !gen.isEmpty {
            let responses = gen.next()
            generatedTokens += responses.count
            if gen.isEmpty {
                generationBatch = nil
            }
            return responses
        }

        if unprocessed.isEmpty {
            return []
        }

        admitFromQueue(forceTransition: true)
        return next()
    }

    public var hasWork: Bool {
        !unprocessed.isEmpty
            || (generationBatch?.isEmpty == false)
            || !promptBatch.isEmpty
    }

    public var queuedCount: Int { unprocessed.count }
    public var activeCount: Int { generationBatch?.batchSize ?? 0 }

    /// Remove a queued or active request from the generator.
    @discardableResult
    public func cancel(uid: Int) -> Bool {
        if let queuedIndex = unprocessed.firstIndex(where: { $0.uid == uid }) {
            unprocessed.remove(at: queuedIndex)
            return true
        }

        if let active = generationBatch, let row = active.uids.firstIndex(of: uid) {
            let keep = active.uids.indices.filter { $0 != row }
            active.filter(keep: keep)
            if active.isEmpty {
                generationBatch = nil
            }
            return true
        }

        if let row = promptBatch.uids.firstIndex(of: uid) {
            let keep = promptBatch.uids.indices.filter { $0 != row }
            promptBatch.filter(keep: keep)
            return true
        }

        return false
    }

    public func close() {
        unprocessed.removeAll()
        generationBatch = nil
        promptBatch = PromptProcessingBatch.empty(
            model: model,
            prefillStepSize: prefillStepSize
        )
    }

    /// Admit up to `min(prefillBatchSize, free completion slots)` queued
    /// requests, prefill them as a sub-batch, and merge them into the
    /// running `GenerationBatch`.
    private func admitFromQueue(forceTransition: Bool = false) {
        let activeRunning = generationBatch?.batchSize ?? 0
        var capacity = max(0, completionBatchSize - activeRunning)

        if generationBatch == nil {
            capacity = max(capacity, 1)
        }
        if capacity == 0 || unprocessed.isEmpty { return }

        let admitCount = min(capacity, prefillBatchSize, unprocessed.count)
        let batchSlice = Array(unprocessed.prefix(admitCount))
        unprocessed.removeFirst(admitCount)

        let promptCache = makeBatchedCache(batchSize: batchSlice.count)
        let prompt = PromptProcessingBatch(
            model: model,
            uids: batchSlice.map { $0.uid },
            promptCache: promptCache,
            tokens: Array(repeating: [], count: batchSlice.count),
            maxTokens: batchSlice.map { $0.maxTokens },
            prefillStepSize: prefillStepSize,
            samplers: batchSlice.map { $0.sampler },
            fallbackSampler: defaultSampler,
            stateMachines: batchSlice.map { $0.stateMachine }
        )

        let admittedGen = prompt.generate(
            lastTokensOf: batchSlice.map { $0.tokens }
        )
        promptTokensProcessed += batchSlice.reduce(into: 0) { $0 += $1.tokens.count }

        if let existing = generationBatch {
            existing.extend(admittedGen)
        } else if !admittedGen.isEmpty || forceTransition {
            generationBatch = admittedGen
        }
    }

    /// Allocate one batched cache per layer. The model's
    /// `newCache(parameters:)` describes the per-layer cache topology
    /// (`KVCacheSimple` / `RotatingKVCache` for full attention, `MambaCache`
    /// or other `ArraysCache` subclasses for SSM-style layers); we build a
    /// batched analog of each.
    private func makeBatchedCache(batchSize B: Int) -> [any BatchedCache] {
        let probe = model.newCache(parameters: nil)
        let zeroLeftPadding = Array(repeating: 0, count: B)
        return probe.map { layer -> any BatchedCache in
            if layer is MambaCache {
                return MambaCache(leftPadding: zeroLeftPadding)
            }
            if let arrays = layer as? ArraysCache {
                return ArraysCache(size: arrays.slotCount, leftPadding: zeroLeftPadding)
            }
            if let rotating = layer as? RotatingKVCache, let maxSize = rotating.maxSize {
                let keep = Int(rotating.metaState.first ?? "0") ?? 0
                precondition(keep == 0, "RotatingKVCache with keep tokens is not supported")
                return BatchRotatingKVCache(maxSize: maxSize, leftPadding: zeroLeftPadding)
            }
            return BatchKVCache(leftPadding: zeroLeftPadding)
        }
    }

    private struct QueuedRequest: Sendable {
        let uid: Int
        let tokens: [Int]
        let maxTokens: Int
        let sampler: RowSampler?
        let stateMachine: SequenceStateMachine
    }
}
