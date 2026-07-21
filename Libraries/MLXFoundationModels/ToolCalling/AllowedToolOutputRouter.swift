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
    private let allowedToolNames: Set<String>

    init(
        format: ToolCallFormat,
        tools: [[String: any Sendable]],
        reasoning: (config: ReasoningConfig, primedInside: Bool)? = nil
    ) {
        self.toolProcessor = ToolCallProcessor(format: format, tools: tools)
        self.allowedToolNames = Set(tools.compactMap { tool in
            (tool["function"] as? [String: any Sendable])?["name"] as? String
        })
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
        events.append(contentsOf: route(toolProcessor.processEOSOutputs()))
        return events
    }

    private mutating func processResponse(_ text: String) -> [Event] {
        route(toolProcessor.processChunkOutputs(text))
    }

    private func route(_ outputs: [ToolCallProcessor.Output]) -> [Event] {
        outputs.compactMap { output in
            switch output {
            case .response(let text):
                .response(text)
            case .toolCall(let call):
                allowedToolNames.contains(call.function.name) ? .toolCall(call) : nil
            }
        }
    }
}

#endif
#endif
