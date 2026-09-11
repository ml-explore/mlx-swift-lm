// Copyright © 2026 Apple Inc.

/// A protocol-neutral semantic event decoded from a model's generated token stream.
package enum TokenStreamEvent: Sendable, Equatable {
    case reasoning(String)
    case response(String)
    case toolCall(ToolCall)
    /// A framed protocol rejected malformed output. Public generation logs it;
    /// package-level consumers can observe it and decide whether to retry.
    case protocolError(String)
    /// A tool-call-shaped model output rejected by parsing or authorization.
    case rejectedToolCall(RejectedToolCall)
    case stop
}

/// Decodes model-specific token streams into response and tool-call events.
///
/// The generation loop owns iteration and cancellation. A decoder owns the
/// response protocol: token framing, semantic stop tokens, and event routing.
/// This keeps model-specific protocols out of the generic evaluation loop.
package protocol TokenStreamDecoder {
    /// Semantic boundaries in addition to the model's ordinary EOS tokens.
    var additionalStopTokenIDs: Set<Int> { get }

    /// Whether semantic stop tokens must be passed through `push` before the
    /// generation loop terminates.
    var receivesStopTokens: Bool { get }

    /// Whether the decoder is currently consuming private reasoning payload.
    /// Sample this before feeding a token to attribute usage accurately.
    var isInsideReasoning: Bool { get }

    /// Tool-call-shaped outputs rejected so far during this generation.
    ///
    /// Decoders whose response protocol never rejects tool calls report zero.
    var rejectedToolCallCount: Int { get }

    /// Whether decoding can end here without flushing an incomplete payload.
    var canEndForSteering: Bool { get }

    /// Consumes one generated token. Returns `false` when decoding should stop
    /// because of either a semantic boundary or consumer termination.
    mutating func push(_ token: Int, emit: (TokenStreamEvent) -> Bool) -> Bool

    /// Flushes any buffered events at the end of generation. Returns `false`
    /// when decoding stopped before all buffered events were delivered.
    mutating func finish(emit: (TokenStreamEvent) -> Bool) -> Bool
}

extension TokenStreamDecoder {
    package var additionalStopTokenIDs: Set<Int> { [] }
    package var receivesStopTokens: Bool { false }
    package var isInsideReasoning: Bool { false }
    package var rejectedToolCallCount: Int { 0 }
    package var canEndForSteering: Bool { false }
}

/// Decoder for ordinary detokenized tool-call syntaxes.
struct StandardTokenStreamDecoder: TokenStreamDecoder {
    private var detokenizer: NaiveStreamingDetokenizer
    private let toolCallProcessor: ToolCallProcessor
    private var stopStringFilter: StopStringFilter
    private var hasCompleteText = false

    init(
        tokenizer: any Tokenizer,
        format: ToolCallFormat,
        tools: [[String: any Sendable]]?,
        stopStrings: Set<String>
    ) {
        self.detokenizer = NaiveStreamingDetokenizer(tokenizer: tokenizer)
        self.toolCallProcessor = ToolCallProcessor(format: format, tools: tools)
        self.stopStringFilter = StopStringFilter(stopStrings: stopStrings)
    }

    var rejectedToolCallCount: Int { toolCallProcessor.rejectedToolCallCount }

    var canEndForSteering: Bool {
        hasCompleteText && stopStringFilter.buffer.isEmpty
            && toolCallProcessor.canEndForSteering
    }

    mutating func push(_ token: Int, emit: (TokenStreamEvent) -> Bool) -> Bool {
        detokenizer.append(token: token)
        hasCompleteText = false
        guard let chunk = detokenizer.next() else { return true }
        hasCompleteText = !chunk.isEmpty

        let result = stopStringFilter.process(chunk)
        if let text = result.text,
            !emitOutputs(toolCallProcessor.processChunkOutputs(text), emit: emit)
        {
            return false
        }
        if result.stopped {
            _ = emit(.stop)
            return false
        }
        return true
    }

    mutating func finish(emit: (TokenStreamEvent) -> Bool) -> Bool {
        if let text = stopStringFilter.finish(),
            !emitOutputs(toolCallProcessor.processChunkOutputs(text), emit: emit)
        {
            return false
        }

        return emitOutputs(toolCallProcessor.processEOSOutputs(), emit: emit)
    }

    /// Maps ordered processor outputs onto stream events, keeping response text,
    /// accepted calls, and rejected calls in the order the model emitted them.
    private func emitOutputs(
        _ outputs: [ToolCallProcessor.Output],
        emit: (TokenStreamEvent) -> Bool
    ) -> Bool {
        for output in outputs {
            let event: TokenStreamEvent =
                switch output {
                case .response(let text): .response(text)
                case .toolCall(let call): .toolCall(call)
                case .rejectedToolCall(let rejection): .rejectedToolCall(rejection)
                }
            guard emit(event) else { return false }
        }
        return true
    }
}
