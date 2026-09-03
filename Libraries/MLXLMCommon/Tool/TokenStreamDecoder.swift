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
}

/// Decoder for ordinary detokenized tool-call syntaxes.
///
/// When the model declares a ``ReasoningConfig``, this also splits its thinking
/// spans out of the response before the tool-call parser sees them. Framed token
/// protocols (Harmony, Onyx) decode their own reasoning and never reach here.
struct StandardTokenStreamDecoder: TokenStreamDecoder {
    private var detokenizer: NaiveStreamingDetokenizer
    private let toolCallProcessor: ToolCallProcessor
    private var stopStringFilter: StopStringFilter

    /// Absent for models with no reasoning protocol, which then decode and cost
    /// exactly what they did before.
    private var reasoningEmitter: ReasoningEventEmitter?

    init(
        tokenizer: any Tokenizer,
        format: ToolCallFormat,
        tools: [[String: any Sendable]]?,
        stopStrings: Set<String>,
        reasoning: (config: ReasoningConfig, primedInside: Bool)? = nil
    ) {
        self.detokenizer = NaiveStreamingDetokenizer(tokenizer: tokenizer)
        self.toolCallProcessor = ToolCallProcessor(format: format, tools: tools)
        self.stopStringFilter = StopStringFilter(stopStrings: stopStrings)
        self.reasoningEmitter = reasoning.map {
            ReasoningEventEmitter(config: $0.config, primedInside: $0.primedInside)
        }
    }

    var rejectedToolCallCount: Int { toolCallProcessor.rejectedToolCallCount }

    var isInsideReasoning: Bool { reasoningEmitter?.isInsideReasoning ?? false }

    mutating func push(_ token: Int, emit: (TokenStreamEvent) -> Bool) -> Bool {
        detokenizer.append(token: token)
        guard let chunk = detokenizer.next() else { return true }

        let result = stopStringFilter.process(chunk)
        if let text = result.text, !route(text, emit: emit) {
            return false
        }
        if result.stopped {
            _ = emit(.stop)
            return false
        }
        return true
    }

    mutating func finish(emit: (TokenStreamEvent) -> Bool) -> Bool {
        if let text = stopStringFilter.finish(), !route(text, emit: emit) {
            return false
        }

        // Drain the reasoning scanner before the tool processor's EOS: an end
        // delimiter that never arrives as text must not strand held-back thinking,
        // and what it flushes may still be response text the parser has to see.
        if let segments = reasoningEmitter?.finalize(),
            !emitSegments(segments, emit: emit)
        {
            return false
        }

        return emitOutputs(toolCallProcessor.processEOSOutputs(), emit: emit)
    }

    /// Splits reasoning off first and lets only non-reasoning text reach the
    /// tool-call parser: a model that writes `<|tool_call>` inside its scratchpad
    /// would otherwise produce a phantom tool call.
    private mutating func route(_ text: String, emit: (TokenStreamEvent) -> Bool) -> Bool {
        guard let segments = reasoningEmitter?.process(text) else {
            return emitOutputs(toolCallProcessor.processChunkOutputs(text), emit: emit)
        }
        return emitSegments(segments, emit: emit)
    }

    /// `mutating` defensively rather than by necessity: it touches only the
    /// class-typed processor today, and marking it so means a later change that does
    /// advance the emitter here cannot silently write to a copy.
    private mutating func emitSegments(
        _ segments: [ReasoningEventEmitter.Segment],
        emit: (TokenStreamEvent) -> Bool
    ) -> Bool {
        for segment in segments {
            switch segment {
            case .reasoning(let reasoning):
                guard emit(.reasoning(reasoning)) else { return false }
            case .response(let response):
                guard emitOutputs(toolCallProcessor.processChunkOutputs(response), emit: emit)
                else { return false }
            }
        }
        return true
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
