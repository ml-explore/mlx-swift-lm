// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXNN
import Tokenizers

// MARK: - InferenceScheduler

/// Actor that manages the lifecycle of concurrent inference requests with a
/// single-first upgrade strategy.
///
/// Ported from Python mlx-lm's `ResponseGenerator`. The scheduler routes
/// requests through two paths:
///
/// - **Single path:** The first request (or incompatible requests) uses
///   `TokenIterator` directly — the existing fast path with zero batch overhead.
/// - **Batch path:** When a second concurrent request arrives while the first
///   is still generating, the scheduler upgrades to `BatchTokenIterator` by
///   migrating the first request's KV cache into a `BatchKVCache`.
///
/// State machine: `.idle` → `.single` → `.batched`
///
/// Usage:
/// ```swift
/// let scheduler = InferenceScheduler()
/// let stream = scheduler.submit(
///     input: lmInput,
///     parameters: params,
///     model: model,
///     cache: nil,
///     tokenizer: tokenizer,
///     configuration: config
/// )
/// for await generation in stream {
///     // handle generation events
/// }
/// ```
public actor InferenceScheduler {

    // MARK: - State Machine

    /// The internal state of the scheduler.
    enum SchedulerState {
        /// No active generation.
        case idle

        /// A single request is active via `TokenIterator`.
        case single(SingleRequestState)

        /// A single-to-batch upgrade is in progress. The scheduler has
        /// suspended to await live state from the single-request task.
        /// Additional requests during this phase run independently on
        /// the single path.
        case upgrading

        /// Multiple requests are active via `BatchTokenIterator`.
        case batched(BatchedState)
    }

    /// Snapshot of the live `TokenIterator` decode state, captured by the
    /// running single-request task and handed to the scheduler during upgrade.
    struct LiveIteratorState: @unchecked Sendable {
        /// The per-layer KV caches with the latest decode state.
        let cache: [KVCache]

        /// The current decode token (`y`) — input for the next step.
        let y: LMInput.Text

        /// Tokens generated so far.
        let tokenCount: Int

        /// Maximum tokens allowed.
        let maxTokens: Int?

        /// The logit sampler.
        let sampler: LogitSampler

        /// The logit processor.
        let processor: LogitProcessor?

        /// The number of tokens in the original prompt input.
        let promptTokenCount: Int

        /// The time taken for prompt processing (prefill) on the single path.
        let promptTime: TimeInterval
    }

    /// Shared mutable flag used to signal that a single request should be
    /// upgraded to batch mode. When the scheduler sets `upgradeRequested`,
    /// the running single-request task captures its live `TokenIterator`
    /// state, deposits it via `depositLiveState(_:)`, and exits its loop.
    /// The scheduler's `upgradeToBatch()` awaits the live state before
    /// building the batch.
    class UpgradeFlag: @unchecked Sendable {
        /// Lock protecting all mutable state in this class.
        private let lock = NSLock()

        /// Set to `true` once the live state has been deposited and the
        /// batch loop owns the continuation.
        private var _upgraded = false

        /// Set to `true` by `upgradeToBatch()` to request the task to
        /// capture its live state and stop iterating.
        private var _upgradeRequested = false

        /// Set to `true` when the single-request task has finished its
        /// decode loop (naturally or via stop/cancel). Used to detect
        /// that the task can no longer respond to an upgrade request.
        private var _taskFinished = false

        /// Continuation that `upgradeToBatch()` awaits. Resumed by the
        /// task when it deposits live state.
        private var liveContinuation: CheckedContinuation<LiveIteratorState?, Never>?

        /// Thread-safe getter for `upgraded`.
        var upgraded: Bool {
            lock.lock()
            defer { lock.unlock() }
            return _upgraded
        }

        /// Thread-safe setter for `upgraded`.
        func setUpgraded(_ value: Bool) {
            lock.lock()
            _upgraded = value
            lock.unlock()
        }

        /// Thread-safe getter for `upgradeRequested`.
        var upgradeRequested: Bool {
            lock.lock()
            defer { lock.unlock() }
            return _upgradeRequested
        }

        /// Called by the scheduler to provide the continuation and
        /// atomically request the upgrade. If the task has already
        /// finished, resumes the continuation immediately with `nil`
        /// so the scheduler does not hang.
        func requestUpgrade(
            continuation: CheckedContinuation<LiveIteratorState?, Never>
        ) {
            lock.lock()
            if _taskFinished {
                // Task already exited its loop — it will never deposit
                // state. Resume immediately so the scheduler can fall back.
                lock.unlock()
                continuation.resume(returning: nil)
                return
            }
            liveContinuation = continuation
            _upgradeRequested = true
            lock.unlock()
        }

        /// Called by the single-request task to deposit live state and
        /// resume the scheduler's continuation.
        func depositLiveState(_ state: LiveIteratorState) {
            lock.lock()
            let cont = liveContinuation
            liveContinuation = nil
            lock.unlock()
            cont?.resume(returning: state)
        }

        /// Called by the single-request task when it exits the decode
        /// loop (either naturally or via stop/cancel). If an upgrade
        /// was requested but we already finished, resumes the
        /// scheduler's continuation with `nil`.
        func markTaskFinished() {
            lock.lock()
            _taskFinished = true
            let cont = liveContinuation
            liveContinuation = nil
            lock.unlock()
            // If the scheduler set a continuation before we could
            // respond, resume it with nil to avoid hanging.
            cont?.resume(returning: nil)
        }
    }

    /// State for a single active request.
    struct SingleRequestState {
        /// The token iterator for the active request.
        let iterator: TokenIterator

        /// The per-layer KV caches being used (extracted from iterator).
        let cache: [KVCache]

        /// The generation task driving the stream.
        let task: Task<Void, Never>

        /// Unique ID for this request (for tracking).
        let requestID: Int

        /// Tokens generated so far for this request.
        var tokensGenerated: Int

        /// The model being used.
        let model: any LanguageModel

        /// The tokenizer for this request.
        let tokenizer: Tokenizer

        /// The model configuration.
        let configuration: ModelConfiguration

        /// The AsyncStream continuation for the first request's stream.
        /// Stored so it can be reused during upgrade to batch mode.
        let continuation: AsyncStream<Generation>.Continuation

        /// Shared flag signaling that this request was upgraded to batch.
        /// When set, the single-request task must not finish the continuation.
        let upgradeFlag: UpgradeFlag

        /// The number of tokens in the original prompt input.
        let promptTokenCount: Int

        /// The input token sequence for prompt cache write-back.
        let inputTokens: [Int]?

        /// Optional prompt cache for write-back after generation.
        let promptCache: LRUPromptCache?

        /// Model name for prompt cache operations.
        let promptCacheModelName: String?
    }

    /// State for batched generation.
    struct BatchedState {
        /// The batch token iterator managing all active sequences.
        let batchIterator: BatchTokenIterator

        /// The driving task that runs the batch generation loop.
        let task: Task<Void, Never>

        /// Mapping from UID -> AsyncStream continuation for routing tokens.
        var continuations: [Int: AsyncStream<Generation>.Continuation]

        /// Mapping from UID -> prompt token count for each request.
        /// Used by the batch loop to report correct promptTokenCount in .info.
        var promptTokenCounts: [Int: Int]

        /// Mapping from UID -> submit timestamp for each request.
        /// Used by the batch loop to compute accurate promptTime for requests
        /// that join the batch after upgrade (3rd+ requests via joinExistingBatch).
        var submitTimes: [Int: Date]

        /// Mapping from UID -> input token sequence for prompt cache write-back.
        var inputTokens: [Int: [Int]]

        /// The model being used.
        let model: any LanguageModel

        /// The tokenizer.
        let tokenizer: Tokenizer

        /// The model configuration.
        let configuration: ModelConfiguration

        /// Stop token IDs.
        let stopTokenIDs: Set<Int>

        /// Optional prompt cache for write-back after generation.
        let promptCache: LRUPromptCache?

        /// Model name for prompt cache operations.
        let promptCacheModelName: String?
    }

    // MARK: - Properties

    /// Current scheduler state.
    private var state: SchedulerState = .idle

    /// Monotonically increasing request ID counter.
    private var requestCounter: Int = 0

    // MARK: - Init

    public init() {}

    // MARK: - Public API

    /// Submit an inference request, returning an `AsyncStream<Generation>` of results.
    ///
    /// - Parameters:
    ///   - input: The prepared language model input.
    ///   - parameters: Generation parameters.
    ///   - model: The language model.
    ///   - cache: Optional pre-existing KV cache.
    ///   - tokenizer: The tokenizer for detokenization and EOS detection.
    ///   - configuration: The model configuration (EOS tokens, tool call format, etc.).
    ///   - cachedKVState: Optional cached KV state from `LRUPromptCache`. When provided,
    ///     the cached prefix is loaded directly into the batch cache and only the uncached
    ///     suffix tokens go through model prefill.
    ///   - promptCache: Optional `LRUPromptCache` for writing back final KV state after
    ///     generation completes. When provided, the final per-request KV cache is stored
    ///     so future requests with the same prefix can skip prefill.
    ///   - promptCacheModelName: Model name used as key for prompt cache operations.
    ///   - inputTokens: The full token sequence for this request, used as key for prompt
    ///     cache write-back.
    /// - Returns: An `AsyncStream<Generation>` yielding generation events for this request.
    public func submit(
        input: LMInput,
        parameters: GenerateParameters,
        model: any LanguageModel,
        cache: [KVCache]?,
        tokenizer: Tokenizer,
        configuration: ModelConfiguration,
        cachedKVState: [KVCache]? = nil,
        promptCache: LRUPromptCache? = nil,
        promptCacheModelName: String? = nil,
        inputTokens: [Int]? = nil
    ) async throws -> AsyncStream<Generation> {
        // Check if this request is batch-compatible
        let compatible = Self.isBatchCompatible(
            input: input,
            parameters: parameters,
            cache: cache,
            model: model
        )

        if !compatible {
            // Incompatible request: always use single path
            return try createSingleStream(
                input: input,
                parameters: parameters,
                model: model,
                cache: cache,
                tokenizer: tokenizer,
                configuration: configuration
            )
        }

        switch state {
        case .idle:
            // First request: use single path (TokenIterator).
            // When cachedKVState is provided (from LRUPromptCache), use it
            // as the initial cache so the TokenIterator skips prefill for
            // the cached prefix tokens.
            return try startSingleRequest(
                input: input,
                parameters: parameters,
                model: model,
                cache: cachedKVState ?? cache,
                tokenizer: tokenizer,
                configuration: configuration,
                promptCache: promptCache,
                promptCacheModelName: promptCacheModelName,
                inputTokens: inputTokens
            )

        case .single(let singleState):
            // Second request while first is active: upgrade to batch
            return try await upgradeToBatch(
                existingSingle: singleState,
                newInput: input,
                newParameters: parameters,
                model: model,
                cache: cache,
                tokenizer: tokenizer,
                configuration: configuration,
                cachedKVState: cachedKVState,
                promptCache: promptCache,
                promptCacheModelName: promptCacheModelName,
                inputTokens: inputTokens
            )

        case .upgrading:
            // Upgrade is in progress — run this request independently on
            // the single path so it doesn't interfere with the ongoing
            // handoff. It will complete on its own without joining the batch.
            // Use cachedKVState if available.
            return try createSingleStream(
                input: input,
                parameters: parameters,
                model: model,
                cache: cachedKVState ?? cache,
                tokenizer: tokenizer,
                configuration: configuration
            )

        case .batched(var batchedState):
            // Third+ request: join existing batch
            return try joinExistingBatch(
                batchedState: &batchedState,
                input: input,
                parameters: parameters,
                tokenizer: tokenizer,
                cachedKVState: cachedKVState
            )
        }
    }

    // MARK: - Batch Compatibility Check

    /// Check if a request is compatible with batch generation.
    ///
    /// Returns `false` for:
    /// - VLMs (input contains images or video)
    /// - Hybrid SSM models (cache contains `MambaCache` or `CacheList`)
    /// - Requests with `kvBits` set (QuantizedKVCache incompatible)
    /// - Caches containing `QuantizedKVCache`
    ///
    /// Returns `true` for:
    /// - Standard LLMs with `KVCacheSimple` and default parameters
    public static func isBatchCompatible(
        input: LMInput,
        parameters: GenerateParameters,
        cache: [KVCache]?,
        model: any LanguageModel
    ) -> Bool {
        // VLM check: images or video present
        if input.image != nil || input.video != nil {
            return false
        }

        // kvBits check: quantized KV cache requested
        if parameters.kvBits != nil {
            return false
        }

        // Cache type check: use existing isBatchCompatible for cache arrays
        if let cache = cache, !cache.isEmpty {
            if !MLXLMCommon.isBatchCompatible(cache) {
                return false
            }
        }

        // Check what cache types the model creates by default
        let templateCache = model.newCache(parameters: parameters)
        if !templateCache.isEmpty && !MLXLMCommon.isBatchCompatible(templateCache) {
            return false
        }

        return true
    }

    // MARK: - Single Request Path

    /// Start a single request using `TokenIterator` — the existing fast path.
    private func startSingleRequest(
        input: LMInput,
        parameters: GenerateParameters,
        model: any LanguageModel,
        cache: [KVCache]?,
        tokenizer: Tokenizer,
        configuration: ModelConfiguration,
        promptCache: LRUPromptCache? = nil,
        promptCacheModelName: String? = nil,
        inputTokens: [Int]? = nil
    ) throws -> AsyncStream<Generation> {
        let iterator = try TokenIterator(
            input: input,
            model: model,
            cache: cache,
            parameters: parameters
        )

        let requestID = requestCounter
        requestCounter += 1

        let (stream, continuation) = AsyncStream<Generation>.makeStream()

        // Store the cache reference from the iterator for potential migration
        let iteratorCache = iterator.cache

        // Pre-compute values needed by the Task closure to avoid capturing
        // non-Sendable types (tokenizer, configuration) across isolation boundaries.
        let stopTokenIDs = Self.buildStopTokenIDs(
            configuration: configuration,
            tokenizer: tokenizer
        )
        let unknownTokenId = tokenizer.unknownTokenId
        let promptTokenCount = input.text.tokens.size
        let toolCallFormat = configuration.toolCallFormat ?? .json
        let tokenizerBox = SendableBox(tokenizer as AnyObject)

        // Shared flag: when set by upgradeToBatch(), the task must not
        // finish the continuation — the batch loop now owns it.
        let upgradeFlag = UpgradeFlag()

        let iteratorBox = SendableBox(iterator)
        let task = Task { [weak self] in
            var iter = iteratorBox.consume()
            let tok = tokenizerBox.consume() as! Tokenizer

            var detokenizer = NaiveStreamingDetokenizer(tokenizer: tok)
            let toolCallProcessor = ToolCallProcessor(format: toolCallFormat)

            var start = Date.timeIntervalSinceReferenceDate
            var promptTime: TimeInterval = 0
            var tokenCount = 0
            var stopReason: GenerateStopReason?

            while let token = iter.next() {
                if Task.isCancelled {
                    stopReason = .cancelled
                    break
                }

                if promptTime == 0 {
                    let now = Date.timeIntervalSinceReferenceDate
                    promptTime = now - start
                    start = now
                }

                if token == unknownTokenId || stopTokenIDs.contains(token) {
                    stopReason = .stop
                    break
                }

                tokenCount += 1

                // Detokenize and emit the token BEFORE checking the upgrade
                // flag. This ensures the boundary token produced by this
                // iteration is not dropped during handoff.
                detokenizer.append(token: token)
                if let chunk = detokenizer.next() {
                    if let textToYield = toolCallProcessor.processChunk(chunk) {
                        if case .terminated = continuation.yield(.chunk(textToYield)) {
                            stopReason = .cancelled
                            break
                        }
                    }
                    if let toolCall = toolCallProcessor.toolCalls.popLast() {
                        if case .terminated = continuation.yield(.toolCall(toolCall)) {
                            stopReason = .cancelled
                            break
                        }
                    }
                }

                // Check for upgrade request AFTER yielding the token.
                // When upgradeRequested is set, deposit the live iterator
                // state for the scheduler and exit the loop.
                if upgradeFlag.upgradeRequested {
                    let liveState = LiveIteratorState(
                        cache: iter.cache,
                        y: iter.y,
                        tokenCount: iter.tokenCount,
                        maxTokens: iter.maxTokens,
                        sampler: iter.sampler,
                        processor: iter.processor,
                        promptTokenCount: promptTokenCount,
                        promptTime: promptTime + iter.promptPrefillTime
                    )
                    upgradeFlag.depositLiveState(liveState)
                    // The batch loop now owns the continuation. Exit without
                    // finishing it — the upgraded flag will be set by the
                    // scheduler after it receives the live state.
                    return
                }
            }

            // Mark the task as finished so any future upgrade request
            // knows it can no longer obtain live state from this task.
            // If an upgrade request arrived but we already exited the
            // loop, this also resumes the scheduler's continuation
            // with nil to prevent hanging.
            upgradeFlag.markTaskFinished()

            // If we were upgraded to batch mode, the batch loop now owns the
            // continuation. Do not emit completion info or finish it.
            if upgradeFlag.upgraded {
                return
            }

            if stopReason == nil {
                if Task.isCancelled {
                    stopReason = .cancelled
                } else if let maxTokens = iter.maxTokens, iter.tokenCount >= maxTokens {
                    stopReason = .length
                } else {
                    stopReason = .cancelled
                }
            }

            // Emit any remaining tool calls
            toolCallProcessor.processEOS()
            for toolCall in toolCallProcessor.toolCalls {
                if case .terminated = continuation.yield(.toolCall(toolCall)) {
                    break
                }
            }

            let now = Date.timeIntervalSinceReferenceDate
            let generateTime = now - start

            let info = GenerateCompletionInfo(
                promptTokenCount: promptTokenCount,
                generationTokenCount: tokenCount,
                promptTime: promptTime + iter.promptPrefillTime,
                generationTime: generateTime,
                stopReason: stopReason ?? .cancelled
            )
            _ = continuation.yield(.info(info))

            // Write back final KV cache to prompt cache for future reuse.
            if let promptCache, let modelName = promptCacheModelName,
                let tokens = inputTokens, !tokens.isEmpty
            {
                promptCache.insertCache(
                    model: modelName,
                    tokens: tokens,
                    promptCache: iter.cache
                )
            }

            Stream().synchronize()
            continuation.finish()

            // Clean up state when single request finishes
            await self?.handleSingleRequestFinished(requestID: requestID)
        }

        continuation.onTermination = { termination in
            if case .cancelled = termination {
                task.cancel()
            }
        }

        state = .single(
            SingleRequestState(
                iterator: iterator,
                cache: iteratorCache,
                task: task,
                requestID: requestID,
                tokensGenerated: 0,
                model: model,
                tokenizer: tokenizer,
                configuration: configuration,
                continuation: continuation,
                upgradeFlag: upgradeFlag,
                promptTokenCount: promptTokenCount,
                inputTokens: inputTokens,
                promptCache: promptCache,
                promptCacheModelName: promptCacheModelName
            ))

        return stream
    }

    /// Create a single-path stream for incompatible requests (doesn't modify scheduler state).
    private func createSingleStream(
        input: LMInput,
        parameters: GenerateParameters,
        model: any LanguageModel,
        cache: [KVCache]?,
        tokenizer: Tokenizer,
        configuration: ModelConfiguration
    ) throws -> AsyncStream<Generation> {
        let iterator = try TokenIterator(
            input: input,
            model: model,
            cache: cache,
            parameters: parameters
        )

        let (stream, _) = generateTask(
            promptTokenCount: input.text.tokens.size,
            modelConfiguration: configuration,
            tokenizer: tokenizer,
            iterator: iterator
        )
        return stream
    }

    // MARK: - Upgrade to Batch

    /// Upgrade from single to batched mode when a second request arrives.
    ///
    /// Key invariants maintained during upgrade:
    /// 1. The first request's original `AsyncStream` continuation is preserved.
    ///    Tokens continue to flow to the same stream the caller received from `submit()`.
    /// 2. The first request's **live** KV cache is used — the running single-request
    ///    task detects the upgrade flag, captures its current `TokenIterator` state
    ///    (which includes the up-to-date cache), and deposits it back to the scheduler.
    /// 3. The second request goes through the normal insert → prefill pipeline.
    /// 4. The first request's cancellation handler is rebound so that cancellation
    ///    after upgrade removes its UID from the `BatchTokenIterator` rather than
    ///    cancelling the defunct single-request task.
    private func upgradeToBatch(
        existingSingle: SingleRequestState,
        newInput: LMInput,
        newParameters: GenerateParameters,
        model: any LanguageModel,
        cache: [KVCache]?,
        tokenizer: Tokenizer,
        configuration: ModelConfiguration,
        cachedKVState: [KVCache]? = nil,
        promptCache: LRUPromptCache? = nil,
        promptCacheModelName: String? = nil,
        inputTokens: [Int]? = nil
    ) async throws -> AsyncStream<Generation> {
        // --- Phase 1: Request live state from the single-request task ---
        // Set state to .upgrading BEFORE the await so that additional
        // requests arriving during the suspension run independently
        // rather than triggering a duplicate upgrade on the same flag.
        state = .upgrading

        // Atomically set the upgradeRequested flag and provide the
        // continuation. If the task has already finished, the
        // continuation is resumed immediately with nil.
        let liveState: LiveIteratorState? = await withCheckedContinuation { continuation in
            existingSingle.upgradeFlag.requestUpgrade(continuation: continuation)
        }

        // If the task already finished before we could capture its state,
        // fall back: the new request runs as an independent single stream
        // and the scheduler remains in idle (the old single already cleaned
        // up).
        guard let liveState else {
            state = .idle
            return try startSingleRequest(
                input: newInput,
                parameters: newParameters,
                model: model,
                cache: cache,
                tokenizer: tokenizer,
                configuration: configuration
            )
        }

        // Mark the upgrade as complete so any late checks in the task see it.
        existingSingle.upgradeFlag.setUpgraded(true)

        // --- Phase 2: Build the batch using live state ---
        let stopTokenIDs = Self.buildStopTokenIDs(
            configuration: configuration,
            tokenizer: tokenizer
        )

        // Create the BatchTokenIterator
        let batchIterator = BatchTokenIterator(
            model: model,
            stopTokens: stopTokenIDs,
            defaultSampler: ArgMaxSampler()
        )

        // Convert each layer's live cache into the appropriate batch cache type.
        // RotatingKVCache must be checked BEFORE KVCacheSimple since both inherit
        // from BaseKVCache, and we need to preserve sliding-window semantics.
        var batchCaches = [KVCache]()
        for layerCache in liveState.cache {
            if let rotatingCache = layerCache as? RotatingKVCache {
                batchCaches.append(BatchRotatingKVCache.fromSingle(rotatingCache))
            } else if let simpleCache = layerCache as? KVCacheSimple {
                batchCaches.append(BatchKVCache.fromSingle(simpleCache))
            } else {
                batchCaches.append(BatchKVCache(leftPadding: [0]))
            }
        }

        // The live `y` is the current decode token — input for the next step.
        let firstLastToken = liveState.y.tokens
        let firstMaxTokens = (liveState.maxTokens ?? 1000) - liveState.tokenCount
        let firstSampler = liveState.sampler
        let firstProcessor = liveState.processor

        // If the first request has exhausted its token budget, finish it
        // immediately and start the second request as a fresh single request.
        // This avoids reinserting a zero-budget entry into the batch engine
        // which would overrun maxTokens by 1.
        if firstMaxTokens <= 0 {
            let firstContinuation = existingSingle.continuation
            let info = GenerateCompletionInfo(
                promptTokenCount: liveState.promptTokenCount,
                generationTokenCount: liveState.tokenCount,
                promptTime: liveState.promptTime,
                generationTime: 0,
                stopReason: .length
            )
            _ = firstContinuation.yield(.info(info))
            firstContinuation.finish()

            state = .idle
            return try startSingleRequest(
                input: newInput,
                parameters: newParameters,
                model: model,
                cache: cache,
                tokenizer: tokenizer,
                configuration: configuration
            )
        }

        // Allocate a UID for the first request inside the batch.
        let firstUID = batchIterator.allocateUID()

        let firstBatch = ActiveBatch(
            uids: [firstUID],
            y: firstLastToken.reshaped([1]).asType(Int32.self),
            cache: batchCaches,
            samplers: [firstSampler],
            processors: [firstProcessor],
            maxTokens: [firstMaxTokens],
            numTokens: [0],
            tokens: [MLXArray]([MLXArray([Int32]())])
        )

        // Inject the pre-filled batch so the first request resumes from its
        // existing KV state — no re-prefill needed.
        batchIterator.setActiveBatch(firstBatch)

        // --- Insert the second (new) request via normal pipeline ---
        let newPromptTokens = newInput.text.tokens.asArray(Int.self)
        let newMaxTokens = newParameters.maxTokens ?? 1000
        let newSampler = newParameters.sampler()
        let newProcessor = newParameters.processor()

        let secondUIDs = batchIterator.insert(
            prompts: [newPromptTokens],
            maxTokens: [newMaxTokens],
            samplers: [newSampler],
            processors: [newProcessor],
            cachedKVStates: [cachedKVState]
        )
        let secondUID = secondUIDs[0]

        // --- Phase 3: Set up continuations and cancellation ---
        // Reuse the original first-request continuation (preserving stream continuity).
        let firstContinuation = existingSingle.continuation
        let (secondStream, secondContinuation) = AsyncStream<Generation>.makeStream()

        let continuations: [Int: AsyncStream<Generation>.Continuation] = [
            firstUID: firstContinuation,
            secondUID: secondContinuation,
        ]

        requestCounter += 1

        // Rebind the first request's cancellation handler so it removes the
        // UID from the BatchTokenIterator instead of cancelling the old task.
        firstContinuation.onTermination = {
            [weak batchIterator] termination in
            if case .cancelled = termination {
                batchIterator?.remove(uids: [firstUID])
            }
        }

        // Capture per-UID prompt token counts and first request's prompt time
        // for use inside the batch loop Task.
        let firstPromptTokenCount = liveState.promptTokenCount
        let firstPromptTime = liveState.promptTime
        let secondPromptTokenCount = newInput.text.tokens.size

        // Start the batch generation loop
        let task = Task { [weak self] in
            var detokenizers: [Int: NaiveStreamingDetokenizer] = [:]
            var toolCallProcessors: [Int: ToolCallProcessor] = [:]
            let format = configuration.toolCallFormat ?? .json

            var starts: [Int: Date] = [:]
            var promptTimes: [Int: TimeInterval] = [:]
            var promptTokenCounts: [Int: Int] = [:]
            var tokenCounts: [Int: Int] = [:]

            let now = Date.timeIntervalSinceReferenceDate
            for uid in [firstUID, secondUID] {
                detokenizers[uid] = NaiveStreamingDetokenizer(tokenizer: tokenizer)
                toolCallProcessors[uid] = ToolCallProcessor(format: format)
                starts[uid] = Date(timeIntervalSinceReferenceDate: now)
                promptTimes[uid] = 0
                tokenCounts[uid] = 0
            }

            // Store per-UID prompt token counts.
            promptTokenCounts[firstUID] = firstPromptTokenCount
            promptTokenCounts[secondUID] = secondPromptTokenCount

            // Preserve the first request's prompt time from the single path.
            // It was already measured before the upgrade — don't reset to 0.
            promptTimes[firstUID] = firstPromptTime

            while let responses = batchIterator.next(), !responses.isEmpty {
                if Task.isCancelled { break }

                for response in responses {
                    let uid = response.uid
                    guard let cont = await self?.getContinuation(uid: uid) else { continue }

                    // Lazy-initialize streaming state for UIDs that joined
                    // the batch after upgrade (3rd+ requests via
                    // joinExistingBatch).
                    if detokenizers[uid] == nil {
                        detokenizers[uid] = NaiveStreamingDetokenizer(tokenizer: tokenizer)
                        toolCallProcessors[uid] = ToolCallProcessor(format: format)
                        // Use the submit timestamp stored by joinExistingBatch
                        // so promptTime reflects submission-to-first-token, not
                        // first-decode-to-first-token.
                        starts[uid] =
                            await self?.getSubmitTime(uid: uid) ?? Date()
                        promptTimes[uid] = 0
                        tokenCounts[uid] = 0
                        // Fetch the prompt token count stored by joinExistingBatch.
                        if promptTokenCounts[uid] == nil {
                            promptTokenCounts[uid] =
                                await self?.getPromptTokenCount(uid: uid) ?? 0
                        }
                    }

                    let token = response.token

                    // Track timing
                    if promptTimes[uid] == 0 {
                        let start = starts[uid]?.timeIntervalSinceReferenceDate ?? now
                        promptTimes[uid] = Date.timeIntervalSinceReferenceDate - start
                        starts[uid] = Date(
                            timeIntervalSinceReferenceDate:
                                Date.timeIntervalSinceReferenceDate)
                    }

                    // Check for stop tokens
                    if stopTokenIDs.contains(token)
                        || token == tokenizer.unknownTokenId
                    {
                        // Don't emit stop tokens as chunks
                    } else {
                        tokenCounts[uid, default: 0] += 1

                        // Detokenize and emit
                        detokenizers[uid]?.append(token: token)
                        if let chunk = detokenizers[uid]?.next() {
                            if let textToYield = toolCallProcessors[uid]?.processChunk(chunk) {
                                _ = cont.yield(.chunk(textToYield))
                            }
                            if let toolCall = toolCallProcessors[uid]?.toolCalls.popLast() {
                                _ = cont.yield(.toolCall(toolCall))
                            }
                        }
                    }

                    if response.finishReason != nil {
                        // Emit final info
                        toolCallProcessors[uid]?.processEOS()
                        if let toolCalls = toolCallProcessors[uid]?.toolCalls {
                            for toolCall in toolCalls {
                                _ = cont.yield(.toolCall(toolCall))
                            }
                        }

                        let generateTime =
                            Date.timeIntervalSinceReferenceDate
                            - (starts[uid]?.timeIntervalSinceReferenceDate ?? now)
                        let info = GenerateCompletionInfo(
                            promptTokenCount: promptTokenCounts[uid] ?? 0,
                            generationTokenCount: tokenCounts[uid] ?? 0,
                            promptTime: promptTimes[uid] ?? 0,
                            generationTime: generateTime,
                            stopReason: response.finishReason ?? .stop
                        )
                        _ = cont.yield(.info(info))
                        cont.finish()

                        // Write back final KV cache for this request to prompt cache.
                        // The cache was extracted by BatchTokenIterator.next() before
                        // the batch was filtered, so it's always available for finished
                        // sequences regardless of post-filter batch state.
                        if let finalCache = response.finalCache,
                            let tokens = await self?.getInputTokens(uid: uid),
                            !tokens.isEmpty
                        {
                            let (pCache, modelName) =
                                await self?.getPromptCacheInfo() ?? (nil, nil)
                            if let pCache, let modelName {
                                pCache.insertCache(
                                    model: modelName,
                                    tokens: tokens,
                                    promptCache: finalCache
                                )
                            }
                        }

                        await self?.removeContinuation(uid: uid)
                    }
                }
            }

            // If we get here, all sequences are done or iterator was closed
            await self?.finishAllContinuations()
            await self?.handleBatchFinished()
        }

        // Wire up second request's cancellation
        secondContinuation.onTermination = {
            [weak batchIterator] termination in
            if case .cancelled = termination {
                batchIterator?.remove(uids: [secondUID])
            }
        }

        // Capture input tokens for prompt cache write-back.
        // First request's tokens come from the SingleRequestState.
        // Second request's tokens come from the submit() call.
        var batchInputTokens: [Int: [Int]] = [:]
        if let firstTokens = existingSingle.inputTokens {
            batchInputTokens[firstUID] = firstTokens
        }
        if let secondTokens = inputTokens {
            batchInputTokens[secondUID] = secondTokens
        }

        state = .batched(
            BatchedState(
                batchIterator: batchIterator,
                task: task,
                continuations: continuations,
                promptTokenCounts: [
                    firstUID: firstPromptTokenCount,
                    secondUID: secondPromptTokenCount,
                ],
                submitTimes: [:],
                inputTokens: batchInputTokens,
                model: model,
                tokenizer: tokenizer,
                configuration: configuration,
                stopTokenIDs: stopTokenIDs,
                promptCache: promptCache ?? existingSingle.promptCache,
                promptCacheModelName: promptCacheModelName ?? existingSingle.promptCacheModelName
            ))

        return secondStream
    }

    // MARK: - Join Existing Batch

    /// Add a new request to the existing batch.
    private func joinExistingBatch(
        batchedState: inout BatchedState,
        input: LMInput,
        parameters: GenerateParameters,
        tokenizer: Tokenizer,
        cachedKVState: [KVCache]? = nil
    ) throws -> AsyncStream<Generation> {
        let promptTokens = input.text.tokens.asArray(Int.self)
        let maxTokens = parameters.maxTokens ?? 1000
        let sampler = parameters.sampler()
        let processor = parameters.processor()

        let uids = batchedState.batchIterator.insert(
            prompts: [promptTokens],
            maxTokens: [maxTokens],
            samplers: [sampler],
            processors: [processor],
            cachedKVStates: [cachedKVState]
        )

        let uid = uids[0]
        let (stream, continuation) = AsyncStream<Generation>.makeStream()

        continuation.onTermination = {
            [weak batchIterator = batchedState.batchIterator]
            termination in
            if case .cancelled = termination {
                batchIterator?.remove(uids: [uid])
            }
        }

        batchedState.continuations[uid] = continuation
        batchedState.promptTokenCounts[uid] = input.text.tokens.size
        batchedState.submitTimes[uid] = Date()
        batchedState.inputTokens[uid] = promptTokens

        // Update state
        state = .batched(batchedState)

        return stream
    }

    // MARK: - State Management Helpers

    /// Called when a single request finishes naturally.
    private func handleSingleRequestFinished(requestID: Int) {
        if case .single(let s) = state, s.requestID == requestID {
            state = .idle
        }
    }

    /// Called when the batch generation loop finishes.
    private func handleBatchFinished() {
        if case .batched = state {
            state = .idle
        }
    }

    /// Get a continuation for a UID from the batched state.
    private func getContinuation(uid: Int) -> AsyncStream<Generation>.Continuation? {
        if case .batched(let batchedState) = state {
            return batchedState.continuations[uid]
        }
        return nil
    }

    /// Remove a continuation for a finished UID.
    private func removeContinuation(uid: Int) {
        if case .batched(var batchedState) = state {
            batchedState.continuations.removeValue(forKey: uid)
            state = .batched(batchedState)
        }
    }

    /// Get the prompt token count for a UID from the batched state.
    private func getPromptTokenCount(uid: Int) -> Int? {
        if case .batched(let batchedState) = state {
            return batchedState.promptTokenCounts[uid]
        }
        return nil
    }

    /// Get the submit timestamp for a UID from the batched state.
    private func getSubmitTime(uid: Int) -> Date? {
        if case .batched(let batchedState) = state {
            return batchedState.submitTimes[uid]
        }
        return nil
    }

    /// Get the input tokens for a UID from the batched state (for prompt cache write-back).
    private func getInputTokens(uid: Int) -> [Int]? {
        if case .batched(let batchedState) = state {
            return batchedState.inputTokens[uid]
        }
        return nil
    }

    /// Get the prompt cache and model name from the batched state (for write-back).
    private func getPromptCacheInfo() -> (LRUPromptCache?, String?) {
        if case .batched(let batchedState) = state {
            return (batchedState.promptCache, batchedState.promptCacheModelName)
        }
        return (nil, nil)
    }

    /// Finish all remaining continuations (e.g., on batch loop exit).
    private func finishAllContinuations() {
        if case .batched(let batchedState) = state {
            for (_, continuation) in batchedState.continuations {
                continuation.finish()
            }
        }
    }

    // MARK: - Utility

    /// Build the set of stop token IDs from configuration and tokenizer.
    private static func buildStopTokenIDs(
        configuration: ModelConfiguration,
        tokenizer: Tokenizer
    ) -> Set<Int> {
        var stopTokenIDs = configuration.eosTokenIds
        if let tokenizerEOS = tokenizer.eosTokenId {
            stopTokenIDs.insert(tokenizerEOS)
        }
        for token in configuration.extraEOSTokens {
            if let id = tokenizer.convertTokenToId(token) {
                stopTokenIDs.insert(id)
            }
        }
        return stopTokenIDs
    }

    /// The current state for testing/inspection.
    public var currentState: String {
        switch state {
        case .idle: return "idle"
        case .single: return "single"
        case .upgrading: return "upgrading"
        case .batched: return "batched"
        }
    }

    /// The batch cache layers from the active batch, for testing/inspection.
    ///
    /// Returns the per-layer `[KVCache]` array from the batch iterator's active
    /// batch when in batched state, or `nil` otherwise.
    public var batchCacheLayers: [KVCache]? {
        if case .batched(let batchedState) = state {
            return batchedState.batchIterator.activeBatch?.cache
        }
        return nil
    }
}
