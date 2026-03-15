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

    /// Pre-existing per-layer KV cache from prompt cache (nil means no cached prefix).
    ///
    /// When non-nil, these caches cover a prefix of `tokens` and only the
    /// uncached suffix needs to go through model prefill. The number of
    /// cached tokens equals the cache's offset.
    public let cachedKVState: [KVCache]?

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
            } else if let batchRotCache = c as? BatchRotatingKVCache {
                batchRotCache.filter(batchIndices: keepIndices)
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
            } else if let selfBatchRot = selfCache as? BatchRotatingKVCache,
                let otherBatchRot = otherCache as? BatchRotatingKVCache
            {
                selfBatchRot.extend(other: otherBatchRot)
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
    public struct Response: @unchecked Sendable {
        /// The unique request ID.
        public let uid: Int

        /// The generated token.
        public let token: Int

        /// Why this sequence finished, or `nil` if it's still generating.
        public let finishReason: GenerateStopReason?

        /// The extracted per-layer KV cache for this sequence, available only when
        /// `finishReason` is non-nil. Used for prompt cache write-back after
        /// generation completes. Extracted before the batch is filtered.
        public let finalCache: [KVCache]?
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

    /// Allocate a unique ID without inserting a prompt.
    ///
    /// Used by the scheduler's upgrade path to reserve a UID for a request
    /// that will be injected directly via `setActiveBatch()`.
    ///
    /// - Returns: A unique request ID.
    public func allocateUID() -> Int {
        lock.lock()
        defer { lock.unlock() }
        let uid = uidCounter
        uidCounter += 1
        return uid
    }

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
    ///   - cachedKVStates: Optional per-prompt cached KV state from prompt cache.
    ///     When non-nil for a prompt, only the uncached suffix tokens go through
    ///     model prefill — the cached prefix is loaded directly into the batch cache.
    /// - Returns: Array of unique IDs, one per inserted prompt.
    @discardableResult
    public func insert(
        prompts: [[Int]],
        maxTokens: [Int],
        samplers: [LogitSampler?]? = nil,
        processors: [LogitProcessor?]? = nil,
        cachedKVStates: [[KVCache]?]? = nil
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
        let cachedArray = cachedKVStates ?? Array(repeating: nil, count: prompts.count)

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
                    processor: processorArray[i],
                    cachedKVState: cachedArray[i]
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

            // Extract per-layer KV cache for finished sequences BEFORE filtering.
            // This allows the caller to write-back the final cache to LRUPromptCache.
            var extractedCache: [KVCache]?
            if finishReason != nil {
                var layers = [KVCache]()
                for layerCache in batch.cache {
                    if let batchCache = layerCache as? BatchKVCache {
                        layers.append(batchCache.extract(idx: e))
                    } else if let batchRotCache = layerCache as? BatchRotatingKVCache {
                        layers.append(batchRotCache.extract(idx: e))
                    }
                }
                if !layers.isEmpty {
                    extractedCache = layers
                }
            }

            responses.append(
                Response(
                    uid: uid, token: token, finishReason: finishReason,
                    finalCache: extractedCache))
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

    /// Set a pre-existing active batch directly, bypassing the normal
    /// insert → prefill pipeline.
    ///
    /// This is used by the scheduler's single-to-batch upgrade path to
    /// migrate an in-flight request (with its already-filled KV cache)
    /// into the batch without re-prefilling.
    ///
    /// - Parameter batch: A fully constructed `ActiveBatch` with pre-filled
    ///   cache and current decode state.
    public func setActiveBatch(_ batch: ActiveBatch) {
        lock.lock()
        defer { lock.unlock() }

        precondition(!isClosed, "Cannot set active batch on a closed BatchTokenIterator")

        if let existing = activeBatch {
            existing.extend(other: batch)
        } else {
            activeBatch = batch
        }
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
    ///
    /// If any prompt has a `cachedKVState`, the cached and uncached prompts
    /// are processed separately and the resulting batches are merged. Cached
    /// prompts skip model prefill for the cached prefix tokens, running only
    /// the uncached suffix through the model.
    internal func processPrompts(_ prompts: [PendingPrompt]) -> ActiveBatch {
        // Partition into cached and uncached prompts
        let cachedPrompts = prompts.filter { $0.cachedKVState != nil }
        let uncachedPrompts = prompts.filter { $0.cachedKVState == nil }

        if cachedPrompts.isEmpty {
            // Fast path: no cached prompts, use standard prefill
            return processUncachedPrompts(uncachedPrompts)
        }

        if uncachedPrompts.isEmpty {
            // All prompts have cached KV state
            return processCachedPrompts(cachedPrompts)
        }

        // Mixed: process both groups and merge
        let cachedBatch = processCachedPrompts(cachedPrompts)
        let uncachedBatch = processUncachedPrompts(uncachedPrompts)
        cachedBatch.extend(other: uncachedBatch)
        return cachedBatch
    }

    /// Process prompts without cached KV state (standard left-pad + full prefill).
    private func processUncachedPrompts(_ prompts: [PendingPrompt]) -> ActiveBatch {
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

    /// Process prompts that have cached KV state from the prompt cache.
    ///
    /// For each prompt, the cached prefix tokens are loaded directly into the
    /// batch cache, and only the uncached suffix tokens go through model
    /// prefill. This significantly reduces computation when a large portion
    /// of the prompt is already cached.
    ///
    /// Left-padding alignment: When merging cached KV states of different
    /// depths alongside variable-length suffixes, all left-padding (from both
    /// cache-depth differences and suffix-length differences) must be
    /// contiguous at the start of the buffer. This is achieved by computing
    /// total left-padding upfront and building the merged buffer with the
    /// correct alignment, rather than mutating leftPadding after merge (which
    /// would desynchronise padding from the stored KV tensors).
    ///
    /// Exact cache hits: When the cache covers the entire prompt, prefill is
    /// skipped entirely and generation begins immediately — the last prompt
    /// token is NOT replayed through the model.
    private func processCachedPrompts(_ prompts: [PendingPrompt]) -> ActiveBatch {
        precondition(!prompts.isEmpty)
        precondition(prompts.allSatisfy { $0.cachedKVState != nil })

        // Each prompt has a cachedKVState covering some prefix.
        // The suffix tokens (after the cached prefix) still need prefilling.
        let cachedStates = prompts.map { $0.cachedKVState! }
        let numLayers = cachedStates[0].count

        // Compute suffix tokens for each prompt.
        // The cached prefix length = cache offset (number of tokens already in cache).
        let cachedLengths = cachedStates.map { layers -> Int in
            layers.first?.offset ?? 0
        }

        // Separate exact cache hits (entire prompt cached) from partial hits.
        // Exact hits skip prefill entirely; partial hits need suffix prefill.
        var exactHitIndices = [Int]()
        var partialHitIndices = [Int]()
        for (i, cachedLen) in cachedLengths.enumerated() {
            if cachedLen >= prompts[i].tokens.count {
                exactHitIndices.append(i)
            } else {
                partialHitIndices.append(i)
            }
        }

        // Handle exact cache hits: skip prefill, sample directly from cached state.
        let exactBatch: ActiveBatch? = processExactCacheHits(
            prompts: prompts, indices: exactHitIndices, cachedStates: cachedStates,
            numLayers: numLayers
        )

        // Handle partial cache hits: merge cached KV + prefill suffix tokens.
        let partialBatch: ActiveBatch? = processPartialCacheHits(
            prompts: prompts, indices: partialHitIndices, cachedStates: cachedStates,
            cachedLengths: cachedLengths, numLayers: numLayers
        )

        // Combine results
        if let exact = exactBatch, let partial = partialBatch {
            exact.extend(other: partial)
            return exact
        }
        return exactBatch ?? partialBatch!
    }

    /// Handle prompts where the cache covers the entire prompt (exact hit).
    /// No prefill is needed — we sample the first decode token directly from
    /// the cached KV state without replaying any prompt tokens.
    private func processExactCacheHits(
        prompts: [PendingPrompt], indices: [Int], cachedStates: [[KVCache]],
        numLayers: Int
    ) -> ActiveBatch? {
        guard !indices.isEmpty else { return nil }

        let selectedPrompts = indices.map { prompts[$0] }
        let selectedStates = indices.map { cachedStates[$0] }

        // Build per-layer batch caches by merging the individual cached caches.
        // Dispatches to the correct batch cache type based on the layer cache type.
        var batchCaches = [KVCache]()
        for l in 0 ..< numLayers {
            let layerCaches = selectedStates.map { $0[l] }
            batchCaches.append(mergeLayerCaches(layerCaches))
        }

        // Initialize per-request processors with their prompt tokens.
        var processors = selectedPrompts.map(\.processor)
        for i in 0 ..< selectedPrompts.count {
            let promptArray = MLXArray(selectedPrompts[i].tokens.map { Int32($0) })
            processors[i]?.prompt(promptArray)
        }

        // For exact hits, the last prompt token is already in the KV cache.
        // We need a single model call with no new input to get logits for
        // the next token. Feed the last prompt token as a query-only input
        // so we can extract logits, but the KV cache already contains it.
        //
        // Since the cache already has all tokens, we run a single forward
        // pass with the last cached token to produce logits for sampling.
        // We must first trim the last token from the cache so re-processing
        // it doesn't duplicate the KV entry.
        for cache in batchCaches {
            cache.trim(1)
        }

        // Build input: last prompt token for each sequence, shape [B, 1]
        let lastTokens = selectedPrompts.map { Int32($0.tokens.last ?? 0) }
        let inputTokens = MLXArray(lastTokens, [selectedPrompts.count, 1])

        let tokenArrays = selectedPrompts.map { MLXArray($0.tokens) }
        let (sampled, _) = step(
            inputTokens: inputTokens,
            cache: batchCaches,
            samplers: selectedPrompts.map(\.sampler),
            processors: &processors,
            tokens: tokenArrays
        )

        asyncEval(sampled)

        return ActiveBatch(
            uids: selectedPrompts.map(\.uid),
            y: sampled,
            cache: batchCaches,
            samplers: selectedPrompts.map(\.sampler),
            processors: processors,
            maxTokens: selectedPrompts.map(\.maxTokens),
            numTokens: Array(repeating: 0, count: selectedPrompts.count),
            tokens: tokenArrays
        )
    }

    /// Handle prompts where only a prefix is cached (partial hit).
    /// Merges cached KV states with correct left-padding, RIGHT-pads
    /// the uncached suffix tokens, prefills through the model, then
    /// calls `finalize()` to roll right-padding zeros to the left.
    ///
    /// **Prepare/finalize lifecycle** (ported from Python mlx-lm):
    /// 1. Merge cached KV into batch caches (right-aligned by cache depth)
    /// 2. RIGHT-pad suffix tokens (shorter suffixes padded on the right)
    /// 3. Call `prepare(rightPadding:)` on each cache layer
    /// 4. Prefill ALL right-padded suffix tokens through the model
    /// 5. Call `finalize()` on each cache layer — this rolls the
    ///    right-padding zeros to the LEFT side, adjusting `leftPadding`
    ///    and `batchOffsets` so the causal mask correctly excludes them
    /// 6. Trim the last token from cache, then re-process it via `step()`
    ///    to get logits for sampling the first decode token
    ///
    /// This eliminates the mixed-depth hole problem: after finalize,
    /// all padding is contiguous on the left and every position in
    /// `leftPadding[i] ..< _idx` is valid cached or prefilled data.
    private func processPartialCacheHits(
        prompts: [PendingPrompt], indices: [Int], cachedStates: [[KVCache]],
        cachedLengths: [Int], numLayers: Int
    ) -> ActiveBatch? {
        guard !indices.isEmpty else { return nil }

        let selectedPrompts = indices.map { prompts[$0] }
        let selectedStates = indices.map { cachedStates[$0] }
        let selectedCacheLengths = indices.map { cachedLengths[$0] }

        // Compute suffix tokens for each prompt.
        let suffixTokens = zip(selectedPrompts, selectedCacheLengths).map {
            prompt, cachedLen -> [Int] in
            Array(prompt.tokens[cachedLen...])
        }

        let suffixLengths = suffixTokens.map(\.count)
        let maxSuffixLength = suffixLengths.max() ?? 0
        let maxCacheLen = selectedCacheLengths.max() ?? 0

        // Buffer size = maxCacheLen (just enough for the longest cached prefix).
        // Each sequence's cached data is right-aligned to end at bufferLen,
        // so leftPadding[i] = bufferLen - cachedLen[i].
        let bufferLen = maxCacheLen
        let B = selectedPrompts.count
        let rightAlignedPadding = (0 ..< B).map { i in
            bufferLen - selectedCacheLengths[i]
        }

        // Compute per-sequence right-padding for suffix alignment.
        // Shorter suffixes are right-padded to match the longest suffix.
        let suffixRightPadding = suffixLengths.map { maxSuffixLength - $0 }

        var batchCaches = [KVCache]()
        for l in 0 ..< numLayers {
            let layerCaches = selectedStates.map { $0[l] }

            // Per-layer type check: mixed-layer models (e.g. Gemma3) have
            // KVCacheSimple for global layers and RotatingKVCache for
            // sliding-window layers. Checking each layer individually
            // ensures neither type's cached data is silently dropped.
            let layerIsRotating = layerCaches[0] is RotatingKVCache

            if layerIsRotating {
                // Rotating cache path: use BatchRotatingKVCache.merge then
                // prepare/finalize lifecycle for right-padding alignment.
                let merged = BatchRotatingKVCache.merge(layerCaches)
                merged.prepare(
                    lengths: suffixLengths,
                    rightPadding: suffixRightPadding
                )
                batchCaches.append(merged)
            } else {
                // KVCacheSimple path: build right-aligned buffer manually.
                let batchCache = buildRightAlignedBatchCache(
                    layerCaches: layerCaches,
                    rightAlignedPadding: rightAlignedPadding,
                    cachedLengths: selectedCacheLengths,
                    bufferLen: bufferLen,
                    batchSize: B
                )
                // Prepare for right-padded suffix prefill
                let rpArray = MLXArray(suffixRightPadding.map { Int32($0) })
                batchCache.prepare(rightPadding: rpArray)
                batchCaches.append(batchCache)
            }
        }

        // Initialize per-request processors with their full prompt tokens.
        var processors = selectedPrompts.map(\.processor)
        for i in 0 ..< selectedPrompts.count {
            let promptArray = MLXArray(selectedPrompts[i].tokens.map { Int32($0) })
            processors[i]?.prompt(promptArray)
        }

        // RIGHT-pad the suffix tokens for prefill (instead of left-padding).
        // After prefill, finalize() will roll the right-padding zeros to the left.
        let paddedSuffix = rightPadPrompts(suffixTokens, maxLength: maxSuffixLength)

        // Prefill ALL right-padded suffix tokens through the model.
        // Unlike the uncached path which holds back the last token for
        // step(), here we process everything so that finalize() can
        // operate on the complete KV state including all suffix tokens.
        var remainingInputs = paddedSuffix
        while remainingInputs.dim(1) > 0 {
            let nToProcess = min(prefillStepSize, remainingInputs.dim(1))
            let chunk = remainingInputs[0..., ..<nToProcess]
            let _ = model(
                LMInput.Text(tokens: chunk),
                cache: batchCaches.isEmpty ? nil : batchCaches,
                state: nil
            )
            eval(batchCaches.flatMap { $0.innerState() })
            if nToProcess < remainingInputs.dim(1) {
                remainingInputs = remainingInputs[0..., nToProcess...]
            } else {
                break
            }
        }

        // Finalize: roll right-padding zeros to the left.
        // After this, leftPadding is adjusted and all padding is
        // contiguous on the left side of the buffer.
        for cache in batchCaches {
            if let batchCache = cache as? BatchKVCache {
                batchCache.finalize()
            } else if let batchRotCache = cache as? BatchRotatingKVCache {
                batchRotCache.finalize()
            }
        }

        // Trim the last token from cache and re-process it to get
        // logits for sampling the first decode token. This mirrors
        // the exact-hit path's trim+replay approach and ensures
        // sampling sees the correct cache state after finalize.
        for cache in batchCaches {
            cache.trim(1)
        }

        // Build input: last real prompt token for each sequence
        let lastTokens = selectedPrompts.map { Int32($0.tokens.last ?? 0) }
        let lastTokenInput = MLXArray(lastTokens, [B, 1])

        let tokenArrays = selectedPrompts.map { MLXArray($0.tokens) }
        let (sampled, _) = step(
            inputTokens: lastTokenInput,
            cache: batchCaches,
            samplers: selectedPrompts.map(\.sampler),
            processors: &processors,
            tokens: tokenArrays
        )

        asyncEval(sampled)

        return ActiveBatch(
            uids: selectedPrompts.map(\.uid),
            y: sampled,
            cache: batchCaches,
            samplers: selectedPrompts.map(\.sampler),
            processors: processors,
            maxTokens: selectedPrompts.map(\.maxTokens),
            numTokens: Array(repeating: 0, count: selectedPrompts.count),
            tokens: tokenArrays
        )
    }

    /// Build a right-aligned `BatchKVCache` for the partial-hit path.
    ///
    /// Each sequence's cached KV data is placed so it ends exactly at
    /// `bufferLen` (which becomes `_idx`), ensuring no unwritten holes
    /// in the `leftPadding[i] ..< _idx` region.
    private func buildRightAlignedBatchCache(
        layerCaches: [KVCache],
        rightAlignedPadding: [Int],
        cachedLengths: [Int],
        bufferLen: Int,
        batchSize B: Int
    ) -> BatchKVCache {
        // Find dimensions from first non-empty cache (KVCacheSimple or RotatingKVCache)
        var H = 0
        var Dk = 0
        var Dv = 0
        var dt: DType = .float16
        for c in layerCaches {
            if let simple = c as? KVCacheSimple, let k = simple.keys {
                H = k.dim(1)
                Dk = k.dim(3)
                Dv = simple.values!.dim(3)
                dt = k.dtype
                break
            }
        }

        guard H > 0 && bufferLen > 0 else {
            return BatchKVCache(leftPadding: rightAlignedPadding)
        }

        // Build the merged buffer with right-aligned cached data.
        let keysArr = MLXArray.zeros([B, H, bufferLen, Dk], dtype: dt)
        let valuesArr = MLXArray.zeros([B, H, bufferLen, Dv], dtype: dt)

        for (i, cache) in layerCaches.enumerated() {
            let pad = rightAlignedPadding[i]
            if let simple = cache as? KVCacheSimple, let k = simple.keys,
                let v = simple.values
            {
                let seqLen = cache.offset
                // Right-align: data fills pad ..< bufferLen
                keysArr[i ..< (i + 1), 0..., pad ..< (pad + seqLen), 0...] =
                    k[.ellipsis, ..<seqLen, 0...]
                valuesArr[i ..< (i + 1), 0..., pad ..< (pad + seqLen), 0...] =
                    v[.ellipsis, ..<seqLen, 0...]
            }
        }

        let batchCache = BatchKVCache(leftPadding: rightAlignedPadding)
        batchCache.keys = keysArr
        batchCache.values = valuesArr
        batchCache._idx = bufferLen
        // Set batchOffsets to reflect each sequence's cached position.
        batchCache.batchOffsets = MLXArray(cachedLengths.map { Int32($0) })
        return batchCache
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
                var s = sampleSampler.sample(logits: sampleLogprobs)
                // Normalize scalar (0-dim) results to 1-D so concatenation works.
                // Some samplers (e.g. FixedTokenSampler, categorical) may return a
                // 0-dimensional MLXArray, but concatenate requires at least 1 dimension.
                if s.ndim == 0 {
                    s = s.reshaped([1])
                }
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

    /// Right-pad token arrays to the given max length, returning shape `[B, maxLength]`.
    ///
    /// Mirrors `leftPadPrompts` but places padding zeros after the real tokens.
    /// Used by the prepare/finalize lifecycle for mixed-depth cached-prompt prefill.
    private func rightPadPrompts(_ prompts: [[Int]], maxLength: Int) -> MLXArray {
        let flat = prompts.flatMap { prompt -> [Int32] in
            let paddingCount = maxLength - prompt.count
            return prompt.map { Int32($0) } + Array(repeating: Int32(0), count: paddingCount)
        }
        return MLXArray(flat, [prompts.count, maxLength])
    }

    /// Create a per-layer batch KV cache with the given left-padding.
    ///
    /// Inspects the template cache from `model.newCache(parameters: nil)` to determine
    /// whether each layer uses a standard or rotating (sliding-window) cache, and creates
    /// the corresponding batch cache type. This ensures models with sliding-window
    /// attention (Gemma3, Mistral3, etc.) get `BatchRotatingKVCache` for the appropriate
    /// layers instead of silently losing window semantics.
    private func makeBatchCache(leftPadding: [Int]) -> [KVCache] {
        let templateCache = model.newCache(parameters: nil)
        return templateCache.map { layer in
            if let rotatingCache = layer as? RotatingKVCache {
                return BatchRotatingKVCache(
                    maxSize: rotatingCache.maxSize ?? 0,
                    leftPadding: leftPadding,
                    keep: rotatingCache.keep
                )
            } else {
                return BatchKVCache(leftPadding: leftPadding)
            }
        }
    }

    /// Merge individual per-layer caches into the appropriate batch cache type.
    ///
    /// Dispatches to `BatchRotatingKVCache.merge()` for `RotatingKVCache` layers
    /// and `BatchKVCache.merge()` for `KVCacheSimple` layers. This ensures that
    /// cached RotatingKVCache entries survive the cached-prefill path instead of
    /// being silently dropped.
    private func mergeLayerCaches(_ caches: [KVCache]) -> KVCache {
        guard !caches.isEmpty else {
            return BatchKVCache(leftPadding: [])
        }

        // Check if the first non-empty cache is a RotatingKVCache
        if caches.first is RotatingKVCache {
            return BatchRotatingKVCache.merge(caches)
        } else {
            return BatchKVCache.merge(caches)
        }
    }
}
