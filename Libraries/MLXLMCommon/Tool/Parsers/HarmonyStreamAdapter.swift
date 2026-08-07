// Copyright © 2026 Apple Inc.

/// Token-stream decoder that owns a ``HarmonyFrameParser`` and a
/// ``HarmonyOutputRouter``.
///
/// The parser stays pure; the router encodes policy; this type bridges both
/// into the model-agnostic token-stream decoder contract.
struct HarmonyStreamAdapter: TokenStreamDecoder {
    private var parser: HarmonyFrameParser
    private var router: HarmonyOutputRouter
    private var stopStringFilter: StopStringFilter
    let additionalStopTokenIDs: Set<Int>
    let receivesStopTokens = true

    init?(
        tokenizer: any Tokenizer,
        tools: [[String: any Sendable]]?,
        stopStrings: Set<String>
    ) {
        guard let parser = HarmonyFrameParser(tokenizer: tokenizer) else {
            return nil
        }
        self.parser = parser
        self.router = HarmonyOutputRouter(
            tokenizer: tokenizer,
            allowedToolNames: HarmonyOutputRouter.allowedToolNames(from: tools))
        self.stopStringFilter = StopStringFilter(stopStrings: stopStrings)
        self.additionalStopTokenIDs = parser.semanticStopTokenIDs
    }

    /// Feeds one token. Returns `false` when the consumer terminated.
    mutating func push(_ token: Int, emit: (TokenStreamEvent) -> Bool) -> Bool {
        emitSteps(parser.push(token), emit: emit)
    }

    mutating func finish(emit: (TokenStreamEvent) -> Bool) -> Bool {
        guard emitSteps(parser.finish(), emit: emit) else { return false }
        guard emitEvents(router.finish(), emit: emit) else { return false }
        if let text = stopStringFilter.finish() {
            return emit(.response(text))
        }
        return true
    }

    private mutating func emitSteps(
        _ steps: [HarmonyParseStep],
        emit: (TokenStreamEvent) -> Bool
    ) -> Bool {
        for step in steps {
            if !emitEvents(router.route(step), emit: emit) {
                return false
            }
        }
        return true
    }

    private mutating func emitEvents(
        _ events: [HarmonyOutputRouter.Event],
        emit: (TokenStreamEvent) -> Bool
    ) -> Bool {
        for event in events {
            switch event {
            case .response(let text):
                let result = stopStringFilter.process(text)
                if let response = result.text, !emit(.response(response)) {
                    return false
                }
                if result.stopped {
                    _ = emit(.stop)
                    return false
                }

            case .toolCall(let call):
                // A non-text event is a hard boundary: a stop string cannot
                // span across a tool call.
                if let text = stopStringFilter.finish(), !emit(.response(text)) {
                    return false
                }
                guard emit(.toolCall(call)) else { return false }
            }
        }
        return true
    }
}
