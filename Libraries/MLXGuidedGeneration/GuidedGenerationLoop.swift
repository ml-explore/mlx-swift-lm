// Copyright © 2026 Apple Inc.

import MLX
import MLXLMCommon
import os

/// Runs grammar-constrained generation with fast-forward token support.
///
/// When the grammar forces deterministic tokens (e.g. JSON structural
/// characters `{`, `}`, `,`, `:`), they're fed through the model one at
/// a time to update the KV cache. Each pass uses the optimized T_q=1
/// Metal kernel.
///
/// The loop overlaps grammar mask computation (CPU) with the model forward
/// pass (GPU). After committing a token, the grammar state is ready for the
/// next mask computation. We compute it while the GPU processes the forward
/// pass, hiding the ~50us CPU cost behind the 10-100ms GPU latency.
///
/// This is a self-contained loop that has direct access to the grammar state
/// and can inject fast-forwarded tokens into both the output stream and the
/// KV cache.
public enum GuidedGenerationLoop {

    private static let logger = Logger(
        subsystem: "com.apple.FoundationModels-MLX",
        category: "GuidedGenerationLoop"
    )

    /// Result of a single generation step.
    enum StepResult {
        /// A sampled token (normal generation).
        case token(Int)
        /// A batch of tokens: the sampled token followed by fast-forward tokens.
        case tokenBatch([Int])
        /// Generation should stop (grammar accepted or error).
        case stop
    }

    /// Runs the guided generation loop, yielding text deltas through `emit`.
    ///
    /// Overlaps grammar mask computation with GPU forward passes: after
    /// committing a token, the next mask is computed on the CPU while the
    /// model forward pass runs on the GPU.
    ///
    /// - Parameters:
    ///   - input: Prepared model input (prompt already tokenized)
    ///   - context: Model context (model, tokenizer, configuration)
    ///   - constraint: The xgrammar constraint (must have `fastForward: true`)
    ///   - maxTokens: Maximum tokens to generate
    ///   - completionReserve: Number of tokens before maxTokens at which closing
    ///     bias activates. When `tokenCount >= maxTokens - completionReserve`,
    ///     the bias nudges sampling toward JSON-closing tokens.
    ///   - hardReserve: Number of tokens before maxTokens at which a *hard*
    ///     closing zone activates: closing tokens are forced and all others
    ///     suppressed (plus an EOS penalty), more aggressively than
    ///     `completionReserve`'s soft nudge. Defaults to 0, which disables it.
    ///   - vocabSize: Number of tokens in the grammar's vocabulary. May differ
    ///     from the model's logit dimension (e.g. added special tokens beyond
    ///     the embedding size). Used to correctly interpret the grammar bitmask.
    ///   - kvBits: Bit width for KV-cache quantization, or nil to disable. When
    ///     set, the KV cache is quantized after each forward pass to reduce
    ///     memory use, mirroring the unconstrained `TokenIterator`. nil is a
    ///     no-op, so models that don't quantize are unaffected.
    ///   - kvGroupSize: Group size for KV-cache quantization (default 64, matching
    ///     `GenerateParameters`). Only used when `kvBits` is non-nil.
    ///   - quantizedKVStart: Token offset at which quantization begins (default 0).
    ///     Only used when `kvBits` is non-nil.
    ///   - closingBias: Pre-computed logit bias array favoring closing tokens
    ///     (from `ClosingTokenBias.compute`). Nil disables forced completion.
    ///   - whitespaceBias: Pre-computed negative logit bias array penalizing
    ///     whitespace-only tokens (from `WhitespaceTokenBias.compute`). Nil
    ///     disables whitespace suppression.
    ///   - whitespaceTokenIDs: Set of token IDs classified as whitespace-only.
    ///     Used by the run tracker to detect consecutive whitespace runs.
    ///   - diagnosticLog: When true, flush the grammar constraint's diagnostic
    ///     logs after the run completes. Defaults to false.
    ///   - emit: Callback for each text delta. Return `false` to stop.
    /// - Returns: Total number of tokens generated (including FF tokens).
    /// - Throws: `GuidedGenerationError.incompleteOutput` if maxTokens is
    ///   exhausted before the grammar reaches a stop state.
    ///   `GuidedGenerationError.prematureEOS` if the model emits EOS
    ///   before the grammar accepts.
    @discardableResult
    public static func run(
        input: LMInput,
        context: ModelContext,
        constraint: GrammarConstraint,
        maxTokens: Int,
        vocabSize: Int,
        kvBits: Int? = nil,
        kvGroupSize: Int = 64,
        quantizedKVStart: Int = 0,
        completionReserve: Int = 64,
        hardReserve: Int = 0,
        closingBias: MLXArray? = nil,
        whitespaceBias: MLXArray? = nil,
        whitespaceTokenIDs: Set<Int> = [],
        diagnosticLog: Bool = false,
        emit: (String) -> Bool
    ) throws -> Int {
        let model = context.model
        var cache = model.newCache(parameters: nil)
        var modelState: LMOutput.State?

        // Build EOS token set
        let stopTokenIDs = Self.buildStopTokenIDs(
            tokenizer: context.tokenizer,
            configuration: context.configuration
        )

        // Prefill prompt and get first set of logits
        var logits: MLXArray
        switch try model.prepare(input, cache: cache, windowSize: 512) {
        case .tokens(let tokens):
            let result = model(tokens[text: .newAxis], cache: cache, state: nil)
            modelState = result.state
            logits = result.logits

        case .logits(let result):
            modelState = result.state
            logits = result.logits
        }

        var detokenizer = NaiveStreamingDetokenizer(tokenizer: context.tokenizer)
        var tokenCount = 0
        var grammarStopped = false
        var whitespaceTracker = WhitespaceRunTracker(whitespaceTokenIDs: whitespaceTokenIDs)

        // Pre-compute bias arrays used in the zone policy.
        //
        // eosPenalty: -10000 at each EOS/stop position. Used in the normal
        // zone to prevent premature EOS at structurally incomplete states,
        // and in the hard zone alongside closing token penalties.
        //
        // The EOS penalty is NOT applied in the soft zone. The grammar mask
        // ensures structural validity (EOS only appears when JSON is
        // structurally complete). Removing the penalty lets the model stop
        // when output is structurally valid but semantically short, which
        // is acceptable near the budget limit.
        let eosPenalty: MLXArray? =
            if let bias = closingBias {
                {
                    let biasLen = bias.shape[0]
                    var penalty = [Float32](repeating: 0.0, count: biasLen)
                    for eos in stopTokenIDs where eos >= 0 && eos < biasLen {
                        penalty[eos] = -10000.0
                    }
                    return MLXArray(penalty)
                }()
            } else {
                nil
            }

        let clock = ContinuousClock()
        let startInstant = clock.now
        var accumulatedText = ""

        // Logit dimension is constant across the generation; capture once so the
        // grammar-mask array can be built outside applyMaskAndSample.
        let logitDim = logits.shape[logits.ndim - 1]

        // Pre-compute the first mask + its sample array (no overlap possible for
        // the first iteration). Subsequent arrays are built in the overlap window.
        var mask = try constraint.computeMask()
        var maskArray = buildMaskArray(for: mask, vocabSize: vocabSize, logitDim: logitDim)

        while tokenCount < maxTokens {
            // Cooperative cancellation: exit promptly when the enclosing Task
            // is cancelled (e.g. test timeout or user-initiated cancellation).
            try Task.checkCancellation()

            // Diagnostic: capture mask state before sampling
            if diagnosticLog {
                let snapshot = mask.mask.withUnsafeBufferPointer { buffer -> MaskSnapshot in
                    let ptr: UnsafePointer<UInt32>? =
                        mask.needsApply
                        ? UnsafeRawPointer(buffer.baseAddress!).assumingMemoryBound(
                            to: UInt32.self)
                        : nil
                    return MaskSnapshot.capture(
                        sampleMask: ptr,
                        vocabSize: vocabSize,
                        tokenIndex: tokenCount,
                        isStop: mask.isTerminated
                    )
                }
                logger.info("\(snapshot.summary())")
            }

            // Check stop from the pre-computed mask
            if mask.isTerminated {
                if diagnosticLog {
                    logger.info(
                        "[GuidedGen] Stop reason: mask.isTerminated at token \(tokenCount)")
                }
                grammarStopped = true
                break
            }

            // Zone policy for budget management:
            //
            //   Normal zone (tokenCount < maxTokens - completionReserve):
            //     No bias. The grammar mask already gates EOS on
            //     structural validity, so primitive schemas (e.g.
            //     `{"type": "integer"}`, where the grammar allows EOS
            //     after one digit) can stop naturally after one token,
            //     without a bias layer on top.
            //
            //   Soft zone (completionReserve .. hardReserve tokens left):
            //     Closing bias only (+200 EOS, +100 closing tokens). No EOS
            //     penalty. The grammar mask ensures EOS only appears when JSON
            //     is structurally valid, so removing the penalty lets the model
            //     stop naturally. May produce shorter output for unbounded
            //     schemas, which is acceptable this close to the budget.
            //
            //   Hard zone (hardReserve tokens left):
            //     Penalize all non-closing tokens (-10000) AND EOS (-10000).
            //     Forces the model to select closing tokens (}, ], ", digits)
            //     that build up JSON structure. The grammar reaches a natural
            //     stop state when JSON is complete. EOS is penalized because
            //     the grammar may allow it at intermediate valid states before
            //     all required fields are present.
            //
            // Only applied when the grammar's mask carries exclusions
            // (`needsApply == true`). When false, the grammar is in an
            // unconditional splice (all tokens forced by FF). Applying
            // bias without a grammar mask can cause EOS selection before
            // the grammar has accepted the output.
            var activeBias: MLXArray? = nil
            if mask.needsApply {
                if let bias = closingBias {
                    if hardReserve > 0 && tokenCount >= maxTokens - hardReserve {
                        // Hard zone: force closing tokens, suppress everything else.
                        var hardBias = which(bias .> 0, Float32(0.0), Float32(-10000.0))
                        if let eosPenalty {
                            hardBias = hardBias + eosPenalty
                        }
                        activeBias = hardBias
                    } else if tokenCount >= maxTokens - completionReserve {
                        // Soft zone: nudge toward closing tokens, no EOS penalty.
                        activeBias = bias
                    }
                    // Normal zone: no bias. Grammar mask + natural EOS handle
                    // termination. Intentionally leaves `activeBias == nil`.
                }
                if let wsBias = whitespaceBias, whitespaceTracker.isActive {
                    activeBias = activeBias.map { $0 + wsBias } ?? wsBias
                }
            }
            let token = applyMaskAndSample(
                logits: logits,
                maskArray: maskArray,
                closingBias: activeBias
            )
            let tokenId = Int(token)

            // Track the sampled token for whitespace run detection.
            // Fast-forward tokens are NOT tracked (they are grammar-forced).
            if whitespaceBias != nil {
                _ = whitespaceTracker.record(tokenID: tokenId)
            }

            // Check EOS only when the grammar exposed a real mask
            // (`needsApply == true`). When `false` the grammar is in an
            // unconditional splice: the sampled value is irrelevant
            // because commitToken will surface the forced tokens.
            // Checking for EOS here would cause a spurious stop -- the
            // model's raw logits might have EOS as the highest value
            // even though the grammar has NOT accepted the output.
            //
            // When `needsApply` IS true: if the grammar mask allowed
            // EOS (bit = 1), the grammar considers the output
            // acceptable. If the mask did NOT allow EOS,
            // `applyMaskAndSample` set it to -inf, so argmax would not
            // have selected it.
            if mask.needsApply {
                if tokenId == context.tokenizer.unknownTokenId || stopTokenIDs.contains(tokenId) {
                    if diagnosticLog {
                        logger.info(
                            "[GuidedGen] Stop reason: EOS/unk tokenId=\(tokenId) at token \(tokenCount)"
                        )
                    }
                    grammarStopped = true
                    break
                }
            }

            // Commit to grammar
            let commitResult = try constraint.commitToken(Int32(token))

            // Yield the sampled token
            detokenizer.append(token: tokenId)
            if let text = detokenizer.next() {
                accumulatedText += text
                if !emit(text) { break }
            }
            tokenCount += 1

            // Periodic progress logging (once per main loop iteration, not per FF token)
            if tokenCount % 50 == 0 {
                let elapsed = clock.now - startInstant
                let ms =
                    elapsed.components.seconds * 1000 + elapsed.components.attoseconds
                    / 1_000_000_000_000_000
                let prefix = String(accumulatedText.prefix(200))
                logger.info("[GuidedGen] token=\(tokenCount) elapsed=\(ms)ms text=\(prefix)")
            }

            if commitResult.isTerminated {
                if diagnosticLog {
                    logger.info(
                        "[GuidedGen] Stop reason: commitResult.isTerminated at token \(tokenCount)"
                    )
                }
                grammarStopped = true
                break
            }

            // Handle fast-forward tokens. CommitResult.tokens carries
            // ONLY the jump-forward ids (the sampled token is not echoed
            // back by xgrammar), so use the array directly.
            let ffTokens: [Int32] = commitResult.tokens

            if !ffTokens.isEmpty {
                // Yield FF tokens to output. The caller's `emit`
                // stop signal (`emit(text) == false`) must halt
                // generation immediately, just like on the sampled-
                // token path above. A bare `break` here would only
                // exit the inner `for`, leaving the outer `while` to
                // run another full iteration — wasting GPU work and
                // violating the caller's stop contract. Propagate
                // through `shouldStopAfterFF` and break the outer
                // `while` after the FF block.
                var shouldStopAfterFF = false
                for ffToken in ffTokens {
                    if tokenCount >= maxTokens {
                        shouldStopAfterFF = true
                        break
                    }
                    detokenizer.append(token: Int(ffToken))
                    if let text = detokenizer.next() {
                        accumulatedText += text
                        if !emit(text) {
                            shouldStopAfterFF = true
                            break
                        }
                    }
                    tokenCount += 1
                }

                if shouldStopAfterFF { break }

                // Process FF tokens one at a time to update KV cache.
                // Batching (T_q > 1 with populated cache) triggers an MLX
                // bug: scaledDotProductAttention in .causal mode creates a
                // mask of shape (T_q, T_q) instead of (T_q, T_kv), causing
                // a broadcast failure on models with global attention layers
                // (e.g., Gemma 3). Single-token passes (T_q=1) use the
                // optimized Metal kernel and skip the mask entirely.
                for (i, ffToken) in ffTokens.enumerated() {
                    let tokenInput = LMInput.Text(tokens: MLXArray([ffToken]))
                    let result = model(
                        tokenInput[text: .newAxis],
                        cache: cache.isEmpty ? nil : cache,
                        state: modelState
                    )
                    modelState = result.state
                    // Only need logits from the last FF token
                    if i == ffTokens.count - 1 {
                        logits = result.logits
                    }
                }

                // Quantize the KV cache after the forward pass(es), matching the
                // unconstrained TokenIterator. No-op unless `kvBits` is set.
                maybeQuantizeKVCache(
                    cache: &cache, kvBits: kvBits, kvGroupSize: kvGroupSize,
                    quantizedKVStart: quantizedKVStart)

                // Kick off GPU computation asynchronously
                asyncEval(logits)

                // Overlap: compute the next mask AND build its sample array on the
                // CPU while the GPU runs the forward pass.
                mask = try constraint.computeMask()
                maskArray = buildMaskArray(for: mask, vocabSize: vocabSize, logitDim: logitDim)

                // Wait for GPU to finish (may already be done)
                eval(logits)
            } else {
                // Normal single-token forward pass (lazy)
                let nextInput = LMInput.Text(tokens: MLXArray([Int32(token)]))
                let result = model(
                    nextInput[text: .newAxis],
                    cache: cache.isEmpty ? nil : cache,
                    state: modelState
                )
                modelState = result.state
                logits = result.logits

                // Quantize the KV cache after the forward pass, matching the
                // unconstrained TokenIterator. No-op unless `kvBits` is set.
                maybeQuantizeKVCache(
                    cache: &cache, kvBits: kvBits, kvGroupSize: kvGroupSize,
                    quantizedKVStart: quantizedKVStart)

                // Kick off GPU computation asynchronously
                asyncEval(logits)

                // Overlap: compute the next mask AND build its sample array on the
                // CPU while the GPU runs the forward pass.
                mask = try constraint.computeMask()
                maskArray = buildMaskArray(for: mask, vocabSize: vocabSize, logitDim: logitDim)

                // Wait for GPU to finish (may already be done)
                eval(logits)
            }
        }

        // Log final generation stats
        let totalElapsed = clock.now - startInstant
        let totalMs =
            totalElapsed.components.seconds * 1000 + totalElapsed.components.attoseconds
            / 1_000_000_000_000_000
        logger.info("[GuidedGen] done tokens=\(tokenCount) elapsed=\(totalMs)ms")

        // Flush any xgrammar warnings (limit exceedances, parser state)
        if diagnosticLog, let logs = constraint.flushLogs() {
            logger.warning("[GuidedGen] xgrammar logs:\n\(logs)")
        }

        // If we exhausted maxTokens without the grammar reaching a stop state,
        // the output is structurally incomplete (e.g., truncated JSON).
        if !grammarStopped && tokenCount >= maxTokens {
            throw GuidedGenerationError.incompleteOutput
        }

        return tokenCount
    }

    // MARK: - Internal (visible for testing)

    /// Build the set of token ids that terminate generation.
    ///
    /// Pulls from three sources, all carried by the `ModelConfiguration` and
    /// tokenizer (all required for chat-tuned models to stop correctly):
    ///
    /// 1. `configuration.eosTokenIds` — loaded from `config.json` /
    ///    `generation_config.json` at model-load time. Chat models like
    ///    Gemma 3 ship `eos_token_id` as an array (e.g. `[1, 106]` for
    ///    `<eos>` + `<end_of_turn>`); this source is the only way to pick
    ///    up the turn-ender when the tokenizer's primary EOS is the
    ///    completion EOS.
    /// 2. `tokenizer.eosTokenId` — the tokenizer's single primary EOS.
    /// 3. `configuration.extraEOSTokens` — hardcoded-by-token-string
    ///    additions from registry entries (e.g. `["<end_of_turn>"]` on
    ///    some Gemma variants in `LLMModelFactory`). Callers needing extra
    ///    stop tokens add them here (via the model configuration), not as a
    ///    per-call argument.
    static func buildStopTokenIDs(
        tokenizer: any Tokenizer,
        configuration: ModelConfiguration
    ) -> Set<Int> {
        var stopTokenIDs = Set(configuration.eosTokenIds)
        if let eos = tokenizer.eosTokenId {
            stopTokenIDs.insert(eos)
        }
        for token in configuration.extraEOSTokens {
            if let id = tokenizer.convertTokenToId(token) {
                stopTokenIDs.insert(id)
            }
        }
        return stopTokenIDs
    }

    /// Apply a *prebuilt* grammar mask and optional bias to logits, then argmax.
    ///
    /// The mask is built once per token in the eval loop's overlap window (see
    /// `buildMaskArray`) and passed in here, so this runs no `bitmaskToMLXArray`
    /// work on the sampling critical path.
    ///
    /// - Parameters:
    ///   - logits: Raw model output logits (shape: [batch, seq, vocab]).
    ///   - maskArray: Prebuilt additive grammar mask (length == logit dim;
    ///     0.0 allowed, -inf disallowed), or nil when the grammar forces all
    ///     tokens (no mask to apply).
    ///   - closingBias: Optional logit bias favoring closing tokens. Applied
    ///     after the grammar mask so masked-out tokens remain at -inf.
    /// - Returns: The sampled token ID.
    static func applyMaskAndSample(
        logits rawLogits: MLXArray,
        maskArray: MLXArray?,
        closingBias: MLXArray? = nil
    ) -> UInt32 {
        // Extract last-position logits: [batch, seq, vocab] -> [vocab]
        var logits = rawLogits[0..., -1, 0...]

        if let maskArray {
            logits = logits + maskArray
        }

        if let bias = closingBias {
            let logitDim = logits.shape[logits.ndim - 1]
            let biasDim = bias.shape[0]
            if biasDim < logitDim {
                // Model logit dimension can exceed tokenizer vocab (padding/special tokens).
                // Pad with zeros so the bias has no effect on extra positions.
                let padding = MLXArray.zeros([logitDim - biasDim])
                logits = logits + concatenated([bias, padding])
            } else if biasDim > logitDim {
                // Tokenizer vocab can exceed model logit dimension (added special tokens
                // beyond the embedding size). Truncate to match.
                logits = logits + bias[0 ..< logitDim]
            } else {
                logits = logits + bias
            }
        }

        // Grammar-constrained generation samples greedily by construction.
        let sampled = argMax(logits, axis: -1)
        return sampled.item(UInt32.self)
    }

    /// Build the additive grammar-mask array for a freshly computed `MaskResult`,
    /// or nil when the grammar needs no mask applied (unconditional FF splice).
    ///
    /// Hoisted out of `applyMaskAndSample` so the loop can build it in the
    /// CPU/GPU overlap window (alongside `computeMask`) instead of on the
    /// critical path at sample time. `logitDim` is the model's logit dimension
    /// (constant across the generation); `vocabSize` is the grammar bitmask's
    /// valid-bit count.
    static func buildMaskArray(for mask: MaskResult, vocabSize: Int, logitDim: Int) -> MLXArray? {
        guard mask.needsApply else { return nil }
        return mask.mask.withUnsafeBufferPointer { buffer in
            let ptr = UnsafeRawPointer(buffer.baseAddress!).assumingMemoryBound(to: UInt32.self)
            return bitmaskToMLXArray(ptr, maskBitCount: vocabSize, totalCount: logitDim)
        }
    }

    /// Convert a packed bitmask (1 bit per token) to an MLXArray of floats.
    /// Allowed tokens get 0.0, disallowed tokens get -inf.
    ///
    /// `maskBitCount` is the number of valid bits in the mask (= tokenizer vocab
    /// size). `totalCount` is the model's logit dimension. When the tokenizer
    /// has more tokens than the model has logits (e.g. added special tokens
    /// beyond the embedding dimension), we only read `min(maskBitCount, totalCount)`
    /// bits. Positions beyond the mask are left at -inf.
    ///
    /// Internal (not private) so the mask-build microbenchmark can time it in
    /// isolation.
    static func bitmaskToMLXArray(
        _ maskPtr: UnsafePointer<UInt32>,
        maskBitCount: Int,
        totalCount: Int
    ) -> MLXArray {
        var floats = [Float](repeating: -Float.infinity, count: totalCount)
        let readCount = min(maskBitCount, totalCount)
        for i in 0 ..< readCount {
            let word = maskPtr[i / 32]
            let bit = (word >> (UInt32(i) % 32)) & 1
            if bit == 1 {
                floats[i] = 0.0
            }
        }
        return MLXArray(floats)
    }
}
