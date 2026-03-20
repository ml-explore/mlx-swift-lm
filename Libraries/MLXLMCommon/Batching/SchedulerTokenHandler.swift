// Copyright © 2024 Apple Inc.

import Foundation
import Tokenizers

// MARK: - SchedulerTokenHandler

/// Type-erased handler that encapsulates output-mode-specific token processing.
///
/// The scheduler calls `handler.processToken(token)` without knowing whether the
/// consumer wants decoded text (`AsyncStream<Generation>`) or raw token IDs
/// (`AsyncStream<TokenGeneration>`). Two factory methods produce handlers for each mode.
struct SchedulerTokenHandler: @unchecked Sendable {

    /// The output mode this handler was created for.
    enum OutputMode {
        case decoded
        case rawTokens(includeStopToken: Bool)
    }

    /// Which output mode this handler serves.
    let mode: OutputMode

    /// Process a generated token. Returns `false` if the consumer cancelled.
    let processToken: @Sendable (Int) -> Bool

    /// Process a stop token. Only meaningful for `.rawTokens(includeStopToken: true)`.
    /// Returns `false` if the consumer cancelled.
    let processStopToken: @Sendable (Int) -> Bool

    /// Flush buffered state at end-of-sequence (e.g. pending tool calls for text mode).
    let processEndOfSequence: @Sendable () -> Void

    /// Yield completion info.
    let yieldInfo: @Sendable (GenerateCompletionInfo) -> Void

    /// Close the stream.
    let finish: @Sendable () -> Void

    /// Register a cancellation callback on the stream's continuation.
    let onCancellation: @Sendable (@Sendable @escaping () -> Void) -> Void
}

// MARK: - Factory: Text Mode

extension SchedulerTokenHandler {

    /// Mutable state box for the text-mode handler.
    /// Captures detokenizer + tool-call processor + continuation so the handler
    /// closures can mutate streaming state. Access is single-threaded by design
    /// (one Task drives the decode loop per request).
    private final class TextState: @unchecked Sendable {
        var detokenizer: NaiveStreamingDetokenizer
        let toolCallProcessor: ToolCallProcessor
        let continuation: AsyncStream<Generation>.Continuation

        init(
            tokenizer: Tokenizer,
            toolCallFormat: ToolCallFormat,
            continuation: AsyncStream<Generation>.Continuation
        ) {
            self.detokenizer = NaiveStreamingDetokenizer(tokenizer: tokenizer)
            self.toolCallProcessor = ToolCallProcessor(format: toolCallFormat)
            self.continuation = continuation
        }
    }

    /// Create a handler that detokenizes tokens and yields `.chunk` / `.toolCall` events.
    static func text(
        continuation: AsyncStream<Generation>.Continuation,
        tokenizer: Tokenizer,
        toolCallFormat: ToolCallFormat
    ) -> SchedulerTokenHandler {
        let box = TextState(
            tokenizer: tokenizer,
            toolCallFormat: toolCallFormat,
            continuation: continuation
        )

        return SchedulerTokenHandler(
            mode: .decoded,
            processToken: { token in
                box.detokenizer.append(token: token)
                if let chunk = box.detokenizer.next() {
                    if let textToYield = box.toolCallProcessor.processChunk(chunk) {
                        if case .terminated = box.continuation.yield(.chunk(textToYield)) {
                            return false
                        }
                    }
                    if let toolCall = box.toolCallProcessor.toolCalls.popLast() {
                        if case .terminated = box.continuation.yield(.toolCall(toolCall)) {
                            return false
                        }
                    }
                }
                return true
            },
            processStopToken: { _ in
                // Decoded mode never emits stop tokens.
                return true
            },
            processEndOfSequence: {
                box.toolCallProcessor.processEOS()
                for toolCall in box.toolCallProcessor.toolCalls {
                    if case .terminated = box.continuation.yield(.toolCall(toolCall)) {
                        break
                    }
                }
            },
            yieldInfo: { info in
                _ = box.continuation.yield(.info(info))
            },
            finish: {
                box.continuation.finish()
            },
            onCancellation: { callback in
                box.continuation.onTermination = { termination in
                    if case .cancelled = termination {
                        callback()
                    }
                }
            }
        )
    }
}

// MARK: - Factory: Raw Token Mode

extension SchedulerTokenHandler {

    /// Create a handler that yields raw `.token(Int)` events.
    static func rawToken(
        continuation: AsyncStream<TokenGeneration>.Continuation,
        includeStopToken: Bool
    ) -> SchedulerTokenHandler {
        return SchedulerTokenHandler(
            mode: .rawTokens(includeStopToken: includeStopToken),
            processToken: { token in
                if case .terminated = continuation.yield(.token(token)) {
                    return false
                }
                return true
            },
            processStopToken: { token in
                guard includeStopToken else { return true }
                if case .terminated = continuation.yield(.token(token)) {
                    return false
                }
                return true
            },
            processEndOfSequence: {
                // No-op for raw token mode.
            },
            yieldInfo: { info in
                _ = continuation.yield(.info(info))
            },
            finish: {
                continuation.finish()
            },
            onCancellation: { callback in
                continuation.onTermination = { termination in
                    if case .cancelled = termination {
                        callback()
                    }
                }
            }
        )
    }
}
