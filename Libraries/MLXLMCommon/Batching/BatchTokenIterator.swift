// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXNN

// MARK: - Supporting Types

/// A queued prompt waiting to be prefilled and added to the active batch.
///
/// Ported from the Python mlx-lm `BatchGenerator.unprocessed_prompts` tuple.
public struct PendingPrompt: @unchecked Sendable {
    /// Unique identifier for this request.
    public let uid: Int

    /// Token IDs for the prompt.
    public let tokens: [Int]

    /// Maximum number of tokens to generate for this request.
    public let maxTokens: Int

    /// Per-request sampler (nil uses the default).
    public let sampler: (any LogitSampler)?

    /// Per-request logit processor (nil means no processing).
    public let processor: LogitProcessor?

    /// Total effective length for sorting (prompt tokens).
    public var effectiveLength: Int { tokens.count }
}

/// Holds the state of all active sequences being decoded in the batch.
///
/// Ported from Python mlx-lm's `Batch` dataclass.
public class ActiveBatch {
    /// Unique IDs for each sequence in the batch.
    public var uids: [Int]

    /// Current token for each sequence, shape `[B]`.
    public var y: MLXArray

    /// Per-layer batch KV caches.
    public var cache: [KVCache]

    /// Per-request samplers (nil entries use the default sampler).
    public var samplers: [LogitSampler?]

    /// Per-request logit processors.
    public var processors: [LogitProcessor?]

    /// Maximum tokens per request.
    public var maxTokens: [Int]

    /// Number of tokens generated so far per request.
    public var numTokens: [Int]

    /// Accumulated tokens per request (for logit processors).
    public var tokens: [MLXArray]

    /// The number of active sequences.
    public var count: Int { uids.count }

    public init(
        uids: [Int],
        y: MLXArray,
        cache: [KVCache],
        samplers: [LogitSampler?],
        processors: [LogitProcessor?],
        maxTokens: [Int],
        numTokens: [Int],
        tokens: [MLXArray]
    ) {
        self.uids = uids
        self.y = y
        self.cache = cache
        self.samplers = samplers
        self.processors = processors
        self.maxTokens = maxTokens
        self.numTokens = numTokens
        self.tokens = tokens
    }

    /// Filter the batch to keep only the sequences at the given indices.
    public func filter(keepIndices: [Int]) {
        uids = keepIndices.map { uids[$0] }
        samplers = keepIndices.map { samplers[$0] }
        processors = keepIndices.map { processors[$0] }
        maxTokens = keepIndices.map { maxTokens[$0] }
        numTokens = keepIndices.map { numTokens[$0] }
        tokens = keepIndices.map { tokens[$0] }

        let indices = MLXArray(keepIndices.map { Int32($0) })
        y = y[indices]
        for c in cache {
            if let batchCache = c as? BatchKVCache {
                batchCache.filter(batchIndices: keepIndices)
            }
        }
    }

    /// Extend this batch with sequences from another batch.
    public func extend(other: ActiveBatch) {
        uids.append(contentsOf: other.uids)
        y = concatenated([y, other.y], axis: 0)
        samplers.append(contentsOf: other.samplers)
        processors.append(contentsOf: other.processors)
        maxTokens.append(contentsOf: other.maxTokens)
        numTokens.append(contentsOf: other.numTokens)
        tokens.append(contentsOf: other.tokens)

        for (selfCache, otherCache) in zip(cache, other.cache) {
            if let selfBatch = selfCache as? BatchKVCache,
                let otherBatch = otherCache as? BatchKVCache
            {
                selfBatch.extend(other: otherBatch)
            }
        }
    }
}

// MARK: - BatchTokenIterator

/// The core batch generation engine, managing prefill and decode phases
/// for multiple concurrent sequences.
///
/// Ported from Python mlx-lm's `BatchGenerator`. This handles:
/// - Inserting new prompts (queued as pending)
/// - Prefilling pending prompts (sorted by length, left-padded, chunked)
/// - Decoding active sequences (one token per step)
/// - Detecting finished sequences (stop tokens or maxTokens)
/// - Removing sequences mid-generation
///
/// Usage:
/// ```swift
/// let iterator = BatchTokenIterator(model: model, stopTokens: stopTokenIDs)
/// let uids = iterator.insert(prompts: [[1,2,3], [4,5]], maxTokens: [100, 100])
/// while let responses = iterator.next(), !responses.isEmpty {
///     for r in responses {
///         // process r.uid, r.token, r.finishReason
///     }
/// }
/// iterator.close()
/// ```
public class BatchTokenIterator: @unchecked Sendable {

    /// A single token response from one sequence in the batch.
    public struct Response: Sendable {
        /// The unique request ID.
        public let uid: Int

        /// The generated token.
        public let token: Int

        /// Why this sequence finished, or `nil` if it's still generating.
        public let finishReason: GenerateStopReason?
    }

    // MARK: - Configuration

    /// The language model used for generation.
    public let model: any LanguageModel

    /// Tokens that signal end-of-sequence.
    public let stopTokens: Set<Int>

    /// Default sampler when per-request sampler is nil.
    public let defaultSampler: any LogitSampler

    /// Maximum number of sequences in the decode batch.
    public let completionBatchSize: Int

    /// Maximum number of prompts to prefill at once.
    public let prefillBatchSize: Int

    /// Maximum tokens to process per prefill chunk.
    public let prefillStepSize: Int

    // MARK: - Synchronization

    /// Lock protecting all mutable state below.
    private let lock = NSLock()

    // MARK: - State (protected by `lock`)

    /// Prompts waiting to be prefilled.
    internal var pendingPrompts: [PendingPrompt] = []

    /// The currently active decode batch, or nil if none.
    internal var activeBatch: ActiveBatch?

    /// Monotonically increasing UID counter.
    private var uidCounter: Int = 0

    /// Whether the iterator has been closed.
    private var isClosed: Bool = false

    /// Internal step counter for periodic cache clearing.
    private var stepCount: Int = 0

    // MARK: - Init

    /// Create a new BatchTokenIterator.
    ///
    /// - Parameters:
    ///   - model: The language model to use for generation.
    ///   - stopTokens: Set of token IDs that signal end-of-sequence.
    ///   - defaultSampler: Default sampler (used when per-request sampler is nil).
    ///   - completionBatchSize: Maximum concurrent decode sequences. Default: 32.
    ///   - prefillBatchSize: Maximum prompts to prefill at once. Default: 8.
    ///   - prefillStepSize: Maximum tokens per prefill chunk. Default: 2048.
    public init(
        model: any LanguageModel,
        stopTokens: Set<Int> = [],
        defaultSampler: any LogitSampler = ArgMaxSampler(),
        completionBatchSize: Int = 32,
        prefillBatchSize: Int = 8,
        prefillStepSize: Int = 2048
    ) {
        self.model = model
        self.stopTokens = stopTokens
        self.defaultSampler = defaultSampler
        self.completionBatchSize = completionBatchSize
        self.prefillBatchSize = prefillBatchSize
        self.prefillStepSize = prefillStepSize
    }

    // MARK: - Public API

    /// Insert new prompts for generation.
    ///
    /// Prompts are queued as pending and will be prefilled on the next `next()` call
    /// when there are free slots in the completion batch.
    ///
    /// - Parameters:
    ///   - prompts: Array of token ID arrays, one per prompt.
    ///   - maxTokens: Maximum tokens to generate per prompt (one per prompt).
    ///   - samplers: Optional per-request samplers. Nil entries use the default.
    ///   - processors: Optional per-request logit processors.
    /// - Returns: Array of unique IDs, one per inserted prompt.
    @discardableResult
    public func insert(
        prompts: [[Int]],
        maxTokens: [Int],
        samplers: [LogitSampler?]? = nil,
        processors: [LogitProcessor?]? = nil
    ) -> [Int] {
        lock.lock()
        defer { lock.unlock() }

        precondition(!isClosed, "Cannot insert into a closed BatchTokenIterator")
        precondition(
            prompts.count == maxTokens.count,
            "prompts and maxTokens must have the same count"
        )

        let samplerArray = samplers ?? Array(repeating: nil, count: prompts.count)
        let processorArray = processors ?? Array(repeating: nil, count: prompts.count)

        var uids = [Int]()
        for i in 0 ..< prompts.count {
            let uid = uidCounter
            uidCounter += 1
            pendingPrompts.append(
                PendingPrompt(
                    uid: uid,
                    tokens: prompts[i],
                    maxTokens: maxTokens[i],
                    sampler: samplerArray[i],
                    processor: processorArray[i]
                )
            )
            uids.append(uid)
        }

        // Sort pending by ascending length for efficient padding during prefill
        pendingPrompts.sort { $0.effectiveLength < $1.effectiveLength }

        return uids
    }

    /// Perform one generation step: prefill pending prompts if slots are available,
    /// then decode one token for all active sequences.
    ///
    /// - Returns: Array of `Response` for each active sequence. Returns an empty array
    ///   when all generation is complete (no pending and no active sequences).
    ///   Returns `nil` if the iterator is closed.
    public func next() -> [Response]? {
        lock.lock()
        defer { lock.unlock() }

        guard !isClosed else { return nil }

        // Check for free slots and prefill pending prompts.
        // Admit min(freeSlots, prefillBatchSize, pendingCount) prompts per
        // iteration so that free decode capacity is filled even when fewer
        // than prefillBatchSize slots are available.
        let numActive = activeBatch?.count ?? 0
        var freeSlots = completionBatchSize - numActive

        while freeSlots > 0 && !pendingPrompts.isEmpty {
            let numToAdmit = min(freeSlots, prefillBatchSize, pendingPrompts.count)
            let promptsToProcess = Array(pendingPrompts.prefix(numToAdmit))

            // Prefill this batch of prompts
            let newBatch = processPrompts(promptsToProcess)
            pendingPrompts.removeFirst(promptsToProcess.count)

            if activeBatch == nil {
                activeBatch = newBatch
            } else {
                activeBatch!.extend(other: newBatch)
            }

            freeSlots -= newBatch.count
        }

        guard let batch = activeBatch else {
            // No pending and no active: generation complete
            return []
        }

        // Append current tokens to per-sequence token history (before decode)
        for i in 0 ..< batch.count {
            batch.tokens[i] = concatenated([batch.tokens[i], batch.y[i ..< (i + 1)]], axis: 0)
        }

        // Decode step: run the model on current tokens and sample next tokens
        let (sampled, _) = step(
            inputTokens: batch.y[0..., .newAxis],
            cache: batch.cache,
            samplers: batch.samplers,
            processors: &batch.processors,
            tokens: batch.tokens
        )

        // Store previous y for response generation, update batch with new tokens
        let previousY = batch.y
        batch.y = sampled

        asyncEval(batch.y)

        // Build responses and determine finished sequences
        let yValues = previousY.asArray(Int.self)
        var keepIndices = [Int]()
        var responses = [Response]()

        for (e, (token, uid)) in zip(yValues, batch.uids).enumerated() {
            batch.numTokens[e] += 1

            let finishReason: GenerateStopReason?
            if stopTokens.contains(token) {
                finishReason = .stop
            } else if batch.numTokens[e] >= batch.maxTokens[e] {
                finishReason = .length
            } else {
                finishReason = nil
                keepIndices.append(e)
            }

            responses.append(Response(uid: uid, token: token, finishReason: finishReason))
        }

        // Remove finished sequences
        if keepIndices.count < batch.count {
            if keepIndices.isEmpty {
                activeBatch = nil
            } else {
                batch.filter(keepIndices: keepIndices)
            }
        }

        stepCount += 1

        return responses
    }

    /// Remove sequences from the active batch or pending queue.
    ///
    /// - Parameter uids: The UIDs of the sequences to remove.
    public func remove(uids: Set<Int>) {
        lock.lock()
        defer { lock.unlock() }

        // Remove from active batch
        if let batch = activeBatch {
            let keepIndices = batch.uids.enumerated()
                .filter { !uids.contains($0.element) }
                .map(\.offset)

            if keepIndices.isEmpty {
                activeBatch = nil
            } else if keepIndices.count < batch.count {
                batch.filter(keepIndices: keepIndices)
            }
        }

        // Remove from pending queue
        pendingPrompts.removeAll { uids.contains($0.uid) }
    }

    /// Stop all generation. After calling close, `next()` returns nil.
    public func close() {
        lock.lock()
        defer { lock.unlock() }

        isClosed = true
        activeBatch = nil
        pendingPrompts.removeAll()
    }

    // MARK: - Internal

    /// Process a batch of pending prompts: left-pad, run prefill in chunks,
    /// then sample the first decode token.
    internal func processPrompts(_ prompts: [PendingPrompt]) -> ActiveBatch {
        let inputs = prompts.map(\.tokens)
        let lengths = inputs.map(\.count)
        let maxLength = lengths.max() ?? 0
        let padding = lengths.map { maxLength - $0 }

        // Left-pad the inputs
        let paddedInputs = leftPadPrompts(inputs, maxLength: maxLength)

        // Create batch KV cache with one BatchKVCache per layer
        let promptCache = makeBatchCache(leftPadding: padding)

        // Initialize per-request processors with their prompt tokens.
        // This mirrors TokenIterator.prepare() calling processor?.prompt(tokens).
        var processors = prompts.map(\.processor)
        for i in 0 ..< prompts.count {
            let promptArray = MLXArray(prompts[i].tokens.map { Int32($0) })
            processors[i]?.prompt(promptArray)
        }

        // Process prompt in chunks of prefillStepSize.
        // We leave the last token for the sampling step below.
        var remainingInputs = paddedInputs
        while remainingInputs.dim(1) > 1 {
            let nToProcess = min(prefillStepSize, remainingInputs.dim(1) - 1)
            let chunk = remainingInputs[0..., ..<nToProcess]
            let _ = model(
                LMInput.Text(tokens: chunk),
                cache: promptCache.isEmpty ? nil : promptCache,
                state: nil
            )
            eval(promptCache.flatMap { $0.innerState() })
            remainingInputs = remainingInputs[0..., nToProcess...]
        }

        // Final step: process last token and sample the first decode token
        let tokenArrays = prompts.map { MLXArray($0.tokens) }
        let (sampled, _) = step(
            inputTokens: remainingInputs,
            cache: promptCache,
            samplers: prompts.map(\.sampler),
            processors: &processors,
            tokens: tokenArrays
        )

        asyncEval(sampled)

        return ActiveBatch(
            uids: prompts.map(\.uid),
            y: sampled,
            cache: promptCache,
            samplers: prompts.map(\.sampler),
            processors: processors,
            maxTokens: prompts.map(\.maxTokens),
            numTokens: Array(repeating: 0, count: prompts.count),
            tokens: tokenArrays
        )
    }

    /// Run one model step: forward pass, process logits, sample, update processor state.
    private func step(
        inputTokens: MLXArray,
        cache: [KVCache],
        samplers: [LogitSampler?],
        processors: inout [LogitProcessor?],
        tokens: [MLXArray]
    ) -> (MLXArray, [MLXArray]) {
        let batchSize = inputTokens.dim(0)

        let result = model(
            LMInput.Text(tokens: inputTokens),
            cache: cache.isEmpty ? nil : cache,
            state: nil
        )
        // Take last token logits: [B, S, V] -> [B, V]
        var logits = result.logits[0..., (-1)..., 0...]
        logits = logits.squeezed(axis: 1)

        // Apply per-request logit processors if any exist
        if processors.contains(where: { $0 != nil }) {
            var processedLogits = [MLXArray]()
            for e in 0 ..< batchSize {
                var sampleLogits = logits[e ..< (e + 1)]
                if processors[e] != nil {
                    sampleLogits = processors[e]!.process(logits: sampleLogits)
                }
                processedLogits.append(sampleLogits)
            }
            logits = concatenated(processedLogits, axis: 0)
        }

        let logprobs = logits - logSumExp(logits, axis: -1, keepDims: true)

        // Per-request sampling if any non-nil samplers exist
        let sampled: MLXArray
        if samplers.contains(where: { $0 != nil }) {
            var allSamples = [MLXArray]()
            for e in 0 ..< batchSize {
                let sampleSampler = samplers[e] ?? defaultSampler
                let sampleLogprobs = logprobs[e ..< (e + 1)]
                let s = sampleSampler.sample(logits: sampleLogprobs)
                allSamples.append(s)
            }
            sampled = concatenated(allSamples, axis: 0)
        } else {
            sampled = defaultSampler.sample(logits: logprobs)
        }

        // Notify processors of the sampled tokens so penalty state stays current.
        // This mirrors TokenIterator's processor?.didSample(token: y) pattern.
        if processors.contains(where: { $0 != nil }) {
            for e in 0 ..< batchSize {
                if processors[e] != nil {
                    processors[e]!.didSample(token: sampled[e])
                }
            }
        }

        let logprobsList = (0 ..< batchSize).map { logprobs[$0] }
        return (sampled, logprobsList)
    }

    /// Left-pad token arrays to the given max length, returning shape `[B, maxLength]`.
    private func leftPadPrompts(_ prompts: [[Int]], maxLength: Int) -> MLXArray {
        let flat = prompts.flatMap { prompt -> [Int32] in
            let paddingCount = maxLength - prompt.count
            return Array(repeating: Int32(0), count: paddingCount) + prompt.map { Int32($0) }
        }
        return MLXArray(flat, [prompts.count, maxLength])
    }

    /// Create a per-layer batch KV cache with the given left-padding.
    private func makeBatchCache(leftPadding: [Int]) -> [KVCache] {
        let templateCache = model.newCache(parameters: nil)
        return templateCache.map { _ in
            BatchKVCache(leftPadding: leftPadding)
        }
    }
}
