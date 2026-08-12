// Copyright © 2026 Apple Inc.

/// A semantic event decoded from a model's generated token stream.
enum TokenStreamEvent: Sendable {
    case response(String)
    case toolCall(ToolCall)
    case stop
}

/// Decodes model-specific token streams into response and tool-call events.
///
/// The generation loop owns iteration and cancellation. A decoder owns the
/// response protocol: token framing, semantic stop tokens, and event routing.
/// This keeps model-specific protocols out of the generic evaluation loop.
protocol TokenStreamDecoder {
    /// Semantic boundaries in addition to the model's ordinary EOS tokens.
    var additionalStopTokenIDs: Set<Int> { get }

    /// Whether semantic stop tokens must be passed through `push` before the
    /// generation loop terminates.
    var receivesStopTokens: Bool { get }

    /// Consumes one generated token. Returns `false` when decoding should stop
    /// because of either a semantic boundary or consumer termination.
    mutating func push(_ token: Int, emit: (TokenStreamEvent) -> Bool) -> Bool

    /// Flushes any buffered events at the end of generation. Returns `false`
    /// when decoding stopped before all buffered events were delivered.
    mutating func finish(emit: (TokenStreamEvent) -> Bool) -> Bool
}

extension TokenStreamDecoder {
    var additionalStopTokenIDs: Set<Int> { [] }
    var receivesStopTokens: Bool { false }
}

/// Decoder for ordinary detokenized tool-call syntaxes.
struct StandardTokenStreamDecoder: TokenStreamDecoder {
    private var detokenizer: NaiveStreamingDetokenizer
    private let toolCallProcessor: ToolCallProcessor
    private var stopStringFilter: StopStringFilter

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

    mutating func push(_ token: Int, emit: (TokenStreamEvent) -> Bool) -> Bool {
        detokenizer.append(token: token)
        guard let chunk = detokenizer.next() else { return true }

        let result = stopStringFilter.process(chunk)
        if let text = result.text, !emitProcessed(text, emit: emit) {
            return false
        }
        if result.stopped {
            _ = emit(.stop)
            return false
        }
        return true
    }

    mutating func finish(emit: (TokenStreamEvent) -> Bool) -> Bool {
        if let text = stopStringFilter.finish(), !emitProcessed(text, emit: emit) {
            return false
        }

        if let response = toolCallProcessor.processEOS(returnBufferedText: true),
            !response.isEmpty,
            !emit(.response(response))
        {
            return false
        }

        for toolCall in toolCallProcessor.drainToolCalls() {
            guard emit(.toolCall(toolCall)) else { return false }
        }
        return true
    }

    private func emitProcessed(
        _ text: String,
        emit: (TokenStreamEvent) -> Bool
    ) -> Bool {
        if let response = toolCallProcessor.processChunk(text), !emit(.response(response)) {
            return false
        }

        for toolCall in toolCallProcessor.drainToolCalls() {
            guard emit(.toolCall(toolCall)) else { return false }
        }
        return true
    }
}
