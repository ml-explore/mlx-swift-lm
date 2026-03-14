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

        /// Multiple requests are active via `BatchTokenIterator`.
        case batched(BatchedState)
    }

    /// Shared mutable flag used to signal that a single request has been
    /// upgraded to batch mode. The single-request task checks this flag
    /// before finishing its continuation — if set, the continuation is
    /// now owned by the batch loop and must not be finished by the
    /// single-request task.
    class UpgradeFlag: @unchecked Sendable {
        var upgraded = false
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
    }

    /// State for batched generation.
    struct BatchedState {
        /// The batch token iterator managing all active sequences.
        let batchIterator: BatchTokenIterator

        /// The driving task that runs the batch generation loop.
        let task: Task<Void, Never>

        /// Mapping from UID -> AsyncStream continuation for routing tokens.
        var continuations: [Int: AsyncStream<Generation>.Continuation]

        /// The model being used.
        let model: any LanguageModel

        /// The tokenizer.
        let tokenizer: Tokenizer

        /// The model configuration.
        let configuration: ModelConfiguration

        /// Stop token IDs.
        let stopTokenIDs: Set<Int>
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
    /// - Returns: An `AsyncStream<Generation>` yielding generation events for this request.
    public func submit(
        input: LMInput,
        parameters: GenerateParameters,
        model: any LanguageModel,
        cache: [KVCache]?,
        tokenizer: Tokenizer,
        configuration: ModelConfiguration
    ) throws -> AsyncStream<Generation> {
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
            // First request: use single path (TokenIterator)
            return try startSingleRequest(
                input: input,
                parameters: parameters,
                model: model,
                cache: cache,
                tokenizer: tokenizer,
                configuration: configuration
            )

        case .single(let singleState):
            // Second request while first is active: upgrade to batch
            return try upgradeToBatch(
                existingSingle: singleState,
                newInput: input,
                newParameters: parameters,
                model: model,
                cache: cache,
                tokenizer: tokenizer,
                configuration: configuration
            )

        case .batched(var batchedState):
            // Third+ request: join existing batch
            return try joinExistingBatch(
                batchedState: &batchedState,
                input: input,
                parameters: parameters,
                tokenizer: tokenizer
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
        configuration: ModelConfiguration
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
            let iter = iteratorBox.consume()
            let tok = tokenizerBox.consume() as! Tokenizer

            var detokenizer = NaiveStreamingDetokenizer(tokenizer: tok)
            let toolCallProcessor = ToolCallProcessor(format: toolCallFormat)

            var start = Date.timeIntervalSinceReferenceDate
            var promptTime: TimeInterval = 0
            var tokenCount = 0
            var stopReason: GenerateStopReason?

            for token in iter {
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

                // Detokenize and emit
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
            }

            // If we were upgraded to batch mode, the batch loop now owns the
            // continuation. Do not emit completion info or finish it.
            if upgradeFlag.upgraded {
                await self?.handleSingleRequestFinished(requestID: requestID)
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
                upgradeFlag: upgradeFlag
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
    /// 2. The first request's KV cache is migrated into `BatchKVCache` via `fromSingle()`,
    ///    then injected into the `BatchTokenIterator` through `setActiveBatch()`.
    /// 3. The second request goes through the normal insert → prefill pipeline.
    private func upgradeToBatch(
        existingSingle: SingleRequestState,
        newInput: LMInput,
        newParameters: GenerateParameters,
        model: any LanguageModel,
        cache: [KVCache]?,
        tokenizer: Tokenizer,
        configuration: ModelConfiguration
    ) throws -> AsyncStream<Generation> {
        // Signal upgrade before cancelling so the single-request task knows
        // not to finish the continuation — the batch loop now owns it.
        existingSingle.upgradeFlag.upgraded = true
        existingSingle.task.cancel()

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

        // --- Migrate the first request's KV cache into a batch cache ---
        let firstCache = existingSingle.cache
        let firstIterator = existingSingle.iterator

        // Convert each layer's KVCacheSimple into a batch-1 BatchKVCache.
        var batchCaches = [KVCache]()
        for layerCache in firstCache {
            if let simpleCache = layerCache as? KVCacheSimple {
                batchCaches.append(BatchKVCache.fromSingle(simpleCache))
            } else {
                batchCaches.append(BatchKVCache(leftPadding: [0]))
            }
        }

        // Build an ActiveBatch for the first request with its migrated cache.
        // The last token produced by the TokenIterator is the current decode
        // token (`y`); it will be the "input" for the next decode step.
        let firstLastToken = firstIterator.y.tokens
        let firstMaxTokens = (firstIterator.maxTokens ?? 1000) - firstIterator.tokenCount
        let firstSampler = firstIterator.sampler
        let firstProcessor = firstIterator.processor

        // Allocate a UID for the first request inside the batch.
        let firstUID = batchIterator.allocateUID()

        let firstBatch = ActiveBatch(
            uids: [firstUID],
            y: firstLastToken.reshaped([1]).asType(Int32.self).squeezed(),
            cache: batchCaches,
            samplers: [firstSampler],
            processors: [firstProcessor],
            maxTokens: [max(firstMaxTokens, 1)],
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
            processors: [newProcessor]
        )
        let secondUID = secondUIDs[0]

        // --- Set up continuations ---
        // Reuse the original first-request continuation (preserving stream continuity).
        let firstContinuation = existingSingle.continuation
        let (secondStream, secondContinuation) = AsyncStream<Generation>.makeStream()

        let continuations: [Int: AsyncStream<Generation>.Continuation] = [
            firstUID: firstContinuation,
            secondUID: secondContinuation,
        ]

        requestCounter += 1

        // Start the batch generation loop
        let task = Task { [weak self] in
            var detokenizers: [Int: NaiveStreamingDetokenizer] = [:]
            var toolCallProcessors: [Int: ToolCallProcessor] = [:]
            let format = configuration.toolCallFormat ?? .json

            var starts: [Int: Date] = [:]
            var promptTimes: [Int: TimeInterval] = [:]
            var tokenCounts: [Int: Int] = [:]

            let now = Date.timeIntervalSinceReferenceDate
            for uid in [firstUID, secondUID] {
                detokenizers[uid] = NaiveStreamingDetokenizer(tokenizer: tokenizer)
                toolCallProcessors[uid] = ToolCallProcessor(format: format)
                starts[uid] = Date(timeIntervalSinceReferenceDate: now)
                promptTimes[uid] = 0
                tokenCounts[uid] = 0
            }

            while let responses = batchIterator.next(), !responses.isEmpty {
                if Task.isCancelled { break }

                for response in responses {
                    let uid = response.uid
                    guard let cont = await self?.getContinuation(uid: uid) else { continue }

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
                            promptTokenCount: 0,
                            generationTokenCount: tokenCounts[uid] ?? 0,
                            promptTime: promptTimes[uid] ?? 0,
                            generationTime: generateTime,
                            stopReason: response.finishReason ?? .stop
                        )
                        _ = cont.yield(.info(info))
                        cont.finish()

                        await self?.removeContinuation(uid: uid)
                    }
                }
            }

            // If we get here, all sequences are done or iterator was closed
            await self?.finishAllContinuations()
            await self?.handleBatchFinished()
        }

        // Wire up cancellation
        secondContinuation.onTermination = { termination in
            if case .cancelled = termination {
                batchIterator.remove(uids: [secondUID])
            }
        }

        state = .batched(
            BatchedState(
                batchIterator: batchIterator,
                task: task,
                continuations: continuations,
                model: model,
                tokenizer: tokenizer,
                configuration: configuration,
                stopTokenIDs: stopTokenIDs
            ))

        return secondStream
    }

    // MARK: - Join Existing Batch

    /// Add a new request to the existing batch.
    private func joinExistingBatch(
        batchedState: inout BatchedState,
        input: LMInput,
        parameters: GenerateParameters,
        tokenizer: Tokenizer
    ) throws -> AsyncStream<Generation> {
        let promptTokens = input.text.tokens.asArray(Int.self)
        let maxTokens = parameters.maxTokens ?? 1000
        let sampler = parameters.sampler()
        let processor = parameters.processor()

        let uids = batchedState.batchIterator.insert(
            prompts: [promptTokens],
            maxTokens: [maxTokens],
            samplers: [sampler],
            processors: [processor]
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
        case .batched: return "batched"
        }
    }
}
