//
//  Evaluate+Token.swift
//  mlx-swift-lm
//
//  Created by Ronald Mannak on 1/31/26.
//

import Foundation
import MLX
import Tokenizers

/// Represents the different stages or outputs of raw-token generation.
///
/// This mirrors `Generation`, but yields raw token IDs instead of decoded text/tool calls.
public enum TokenGeneration: Sendable {
    /// A generated token ID.
    case token(Int)

    /// Completion information summarizing token counts and performance metrics.
    case info(GenerateCompletionInfo)

    /// Token ID or nil
    public var token: Int? {
        switch self {
        case .token(let token): token
        case .info: nil
        }
    }

    /// Completion info or nil
    public var info: GenerateCompletionInfo? {
        switch self {
        case .token: nil
        case .info(let info): info
        }
    }

    /// Reducer that can be used with `throttle()` to gather elements into a batch
    @Sendable
    public static func collect(_ batch: [TokenGeneration]?, _ element: TokenGeneration) -> [TokenGeneration] {
        (batch ?? []) + [element]
    }
}

/// Generates raw token IDs asynchronously using the provided language model input, parameters, and context.
///
/// This is similar to `generate(input:cache:parameters:context:)`, but yields raw token IDs instead of decoded text/tool calls.
/// This is useful for downstream parsers that need access to token IDs directly (e.g. Harmony parsing).
///
/// - Parameters:
///   - input: The input for the language model.
///   - cache: optional ``KVCache``
///   - parameters: The configuration options for token generation.
///   - context: The model context, including the model itself and associated tokenizer.
///   - includeStopToken: when true, the terminating EOS/unknown token is yielded before finishing
/// - Returns: An `AsyncStream` that emits `TokenGeneration` values.
public func generateTokens(
    input: LMInput,
    cache: [KVCache]? = nil,
    parameters: GenerateParameters,
    context: ModelContext,
    includeStopToken: Bool = false
) throws -> AsyncStream<TokenGeneration> {
    let iterator = try TokenIterator(
        input: input, model: context.model, cache: cache, parameters: parameters)
    let (stream, _) = generateTokenTask(
        promptTokenCount: input.text.tokens.size,
        modelConfiguration: context.configuration,
        tokenizer: context.tokenizer,
        iterator: iterator,
        includeStopToken: includeStopToken
    )
    return stream
}

/// Generates raw token IDs asynchronously and returns the stream plus a `Task`.
///
/// Prefer this overload if you want to be able to observe when the underlying generation work is finished
/// (especially if the consumer terminates the stream early).
///
/// - Returns: An `AsyncStream` that emits `TokenGeneration` values and a `Task`.
public func generateTokensTask(
    input: LMInput,
    cache: [KVCache]? = nil,
    parameters: GenerateParameters,
    context: ModelContext,
    includeStopToken: Bool = false
) throws -> (AsyncStream<TokenGeneration>, Task<Void, Never>) {
    let iterator = try TokenIterator(
        input: input, model: context.model, cache: cache, parameters: parameters)
    return generateTokenTask(
        promptTokenCount: input.text.tokens.size,
        modelConfiguration: context.configuration,
        tokenizer: context.tokenizer,
        iterator: iterator,
        includeStopToken: includeStopToken
    )
}

/// Low-level raw token generation using a `TokenIterator`, returning an
/// `AsyncStream<TokenGeneration>` and a `Task`.
///
/// This is useful for parsers that need access to the token IDs directly (e.g. Harmony parsing)
/// without detokenization or tool-call parsing.
///
/// - Parameters:
///   - promptTokenCount: number of tokens in the prompt
///   - modelConfiguration: model configuration (for EOS/extra EOS tokens)
///   - tokenizer: tokenizer (for EOS id and unknown token id)
///   - iterator: token iterator
///   - includeStopToken: when true, the terminating EOS/unknown token is yielded before finishing
/// - Returns: An `AsyncStream` that emits token IDs and a final `.info`, plus a `Task`.
public func generateTokenTask(
    promptTokenCount: Int,
    modelConfiguration: ModelConfiguration,
    tokenizer: Tokenizer,
    iterator: consuming TokenIterator,
    includeStopToken: Bool = false
) -> (AsyncStream<TokenGeneration>, Task<Void, Never>) {

    let (stream, continuation) = AsyncStream<TokenGeneration>.makeStream()

    let iterator = SendableBox(iterator)

    // Launch a Task to perform iteration asynchronously.
    let task = Task {
        let iterator = iterator.consume()

        var start = Date.timeIntervalSinceReferenceDate
        var promptTime: TimeInterval = 0

        // Build complete EOS token set from all sources
        var eosTokenIds = modelConfiguration.eosTokenIds
        if let tokenizerEos = tokenizer.eosTokenId {
            eosTokenIds.insert(tokenizerEos)
        }
        for token in modelConfiguration.extraEOSTokens {
            if let id = tokenizer.convertTokenToId(token) {
                eosTokenIds.insert(id)
            }
        }

        var tokenCount = 0

        for token in iterator {
            // Check for cancellation on every loop iteration.
            if Task.isCancelled {
                break
            }

            if promptTime == 0 {
                let now = Date.timeIntervalSinceReferenceDate
                promptTime = now - start
                start = now
            }

            // Check for end-of-sequence tokens
            if token == tokenizer.unknownTokenId || eosTokenIds.contains(token) {
                if includeStopToken {
                    tokenCount += 1
                    if case .terminated = continuation.yield(.token(token)) {
                        break
                    }
                }
                break
            }

            tokenCount += 1
            if case .terminated = continuation.yield(.token(token)) {
                break
            }
        }

        let now = Date.timeIntervalSinceReferenceDate
        let generateTime = now - start

        let info = GenerateCompletionInfo(
            promptTokenCount: promptTokenCount,
            generationTokenCount: tokenCount,
            promptTime: promptTime + iterator.promptPrefillTime,
            generationTime: generateTime
        )
        continuation.yield(.info(info))

        // Synchronize with the stream to ensure tasks are completed
        Stream().synchronize()

        // Finalize the stream
        continuation.finish()
    }

    // When the consumer cancels (or ends) the stream, cancel our underlying task.
    continuation.onTermination = { _ in
        task.cancel()
    }

    return (stream, task)
}
