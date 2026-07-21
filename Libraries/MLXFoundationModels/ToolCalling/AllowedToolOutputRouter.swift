// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration
#if canImport(FoundationModels, _version: 2)

import MLXLMCommon

struct AllowedToolOutputRouter {
    enum Event: Sendable, Equatable {
        case reasoning(String)
        case response(String)
        case toolCall(MLXLMCommon.ToolCall)
    }

    private var reasoningEmitter: ReasoningEventEmitter?
    private let toolProcessor: ToolCallProcessor

    init(
        format: ToolCallFormat,
        tools: [[String: any Sendable]],
        reasoning: (config: ReasoningConfig, primedInside: Bool)? = nil
    ) {
        self.toolProcessor = ToolCallProcessor(format: format, tools: tools)
        self.reasoningEmitter = reasoning.map {
            ReasoningEventEmitter(config: $0.config, primedInside: $0.primedInside)
        }
    }

    var isInsideReasoning: Bool {
        reasoningEmitter?.isInsideReasoning ?? false
    }

    mutating func process(_ chunk: String) -> [Event] {
        guard var emitter = reasoningEmitter else {
            return processResponse(chunk)
        }

        let segments = emitter.process(chunk)
        reasoningEmitter = emitter
        var events: [Event] = []
        for segment in segments {
            switch segment {
            case .reasoning(let text):
                events.append(.reasoning(text))
            case .response(let text):
                events.append(contentsOf: processResponse(text))
            }
        }
        return events
    }

    mutating func finish() -> [Event] {
        var events: [Event] = []
        if var emitter = reasoningEmitter {
            for segment in emitter.finalize() {
                switch segment {
                case .reasoning(let text): events.append(.reasoning(text))
                case .response(let text): events.append(contentsOf: processResponse(text))
                }
            }
            reasoningEmitter = emitter
        }
        if let text = toolProcessor.processEOS(returnBufferedText: true), !text.isEmpty {
            events.append(.response(text))
        }
        events.append(contentsOf: toolProcessor.drainToolCalls().map(Event.toolCall))
        return events
    }

    private mutating func processResponse(_ text: String) -> [Event] {
        var events: [Event] = []
        if let visible = toolProcessor.processChunk(text), !visible.isEmpty {
            events.append(.response(visible))
        }
        events.append(contentsOf: toolProcessor.drainToolCalls().map(Event.toolCall))
        return events
    }
}

#endif
#endif
