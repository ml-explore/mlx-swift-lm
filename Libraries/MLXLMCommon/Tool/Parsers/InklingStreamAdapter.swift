// Copyright © 2026 Apple Inc.

import Foundation

/// The payload kind a frame header ends with.
private enum InklingContentKind: Equatable {
    case thinking
    case text
    case toolJSON
    case toolText
}

private enum InklingParseStep {
    case consumed
    case opened(InklingContentKind)
    case payload(InklingContentKind, Int)
    case closed(InklingContentKind, name: String, tokens: [Int])
    case rejected(String)
}

/// Token-aware parser of Inkling 2's message frames:
/// `<|message_model|>name<|content_kind|>payload<|end_message|>`, repeated, then
/// `<|content_model_end_sampling|>`.
///
/// Structural markers are compared as token ids, so marker-shaped text in a payload is never
/// a frame boundary.
private struct InklingFrameParser {
    private struct Controls {
        var messageModel: Int
        var endMessage: Int
        var endSampling: Int
        var kinds: [Int: InklingContentKind]
    }

    private enum State {
        case start
        case betweenFrames(reportedUnexpectedToken: Bool)
        case header(tokens: [Int])
        case payload(kind: InklingContentKind, name: String, tokens: [Int])
        case finished
    }

    private let tokenizer: any Tokenizer
    private let controls: Controls
    private var state = State.start
    private let maximumHeaderTokens = 64

    init?(tokenizer: any Tokenizer) {
        guard let messageModel = tokenizer.convertTokenToId("<|message_model|>"),
            let endMessage = tokenizer.convertTokenToId("<|end_message|>"),
            let endSampling = tokenizer.convertTokenToId("<|content_model_end_sampling|>"),
            let thinking = tokenizer.convertTokenToId("<|content_thinking|>"),
            let text = tokenizer.convertTokenToId("<|content_text|>"),
            let toolJSON = tokenizer.convertTokenToId("<|content_invoke_tool_json|>")
        else { return nil }
        var kinds: [Int: InklingContentKind] = [
            thinking: .thinking, text: .text, toolJSON: .toolJSON,
        ]
        if let toolText = tokenizer.convertTokenToId("<|content_invoke_tool_text|>") {
            kinds[toolText] = .toolText
        }
        guard Set([messageModel, endMessage, endSampling] + kinds.keys).count == kinds.count + 3
        else { return nil }
        self.tokenizer = tokenizer
        self.controls = Controls(
            messageModel: messageModel, endMessage: endMessage, endSampling: endSampling,
            kinds: kinds)
    }

    var isInsideThinking: Bool {
        if case .payload(.thinking, _, _) = state { return true }
        return false
    }

    mutating func push(_ token: Int) -> InklingParseStep {
        if token == controls.endSampling {
            defer { state = .finished }
            switch state {
            case .payload(let kind, _, _):
                return .rejected(
                    "Inkling \(description(of: kind)) frame ended without <|end_message|>")
            case .header(let tokens) where !tokens.isEmpty:
                return .rejected(
                    "Incomplete Inkling frame header at <|content_model_end_sampling|>")
            default:
                return .consumed
            }
        }

        switch state {
        case .finished:
            return .consumed

        case .start:
            // A model turn opens with a frame marker, so a first token without one continues
            // the frame the prompt left open: a continued final message ends inside its text.
            let opensFrame = token == controls.messageModel || controls.kinds[token] != nil
            state =
                opensFrame
                ? .betweenFrames(reportedUnexpectedToken: false)
                : .payload(kind: .text, name: "", tokens: [])
            return push(token)

        case .betweenFrames(let reported):
            if token == controls.messageModel {
                state = .header(tokens: [])
                return .consumed
            }
            // Tolerate a frame that opens without its role marker.
            if let kind = controls.kinds[token] {
                state = .payload(kind: kind, name: "", tokens: [])
                return .opened(kind)
            }
            guard !reported else { return .consumed }
            state = .betweenFrames(reportedUnexpectedToken: true)
            let text = tokenizer.decode(tokenIds: [token], skipSpecialTokens: false)
            return .rejected("Unexpected token between Inkling frames: \(text)")

        case .header(var tokens):
            if let kind = controls.kinds[token] {
                let name = tokenizer.decode(tokenIds: tokens, skipSpecialTokens: false)
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                state = .payload(kind: kind, name: name, tokens: [])
                return .opened(kind)
            }
            if token == controls.messageModel || token == controls.endMessage {
                state = .betweenFrames(reportedUnexpectedToken: false)
                return .rejected("Inkling frame header ended before its content marker")
            }
            guard tokens.count < maximumHeaderTokens else {
                state = .betweenFrames(reportedUnexpectedToken: false)
                return .rejected("Inkling frame header exceeded \(maximumHeaderTokens) tokens")
            }
            tokens.append(token)
            state = .header(tokens: tokens)
            return .consumed

        case .payload(let kind, let name, var tokens):
            if token == controls.endMessage {
                state = .betweenFrames(reportedUnexpectedToken: false)
                return .closed(kind, name: name, tokens: tokens)
            }
            if token == controls.messageModel {
                state = .header(tokens: [])
                return .rejected(
                    "Inkling \(description(of: kind)) frame was interrupted by <|message_model|>")
            }
            if kind == .toolJSON || kind == .toolText {
                tokens.append(token)
            }
            state = .payload(kind: kind, name: name, tokens: tokens)
            return .payload(kind, token)
        }
    }

    /// The step for output that stopped inside a frame.
    mutating func finish() -> InklingParseStep? {
        defer { state = .finished }
        switch state {
        case .payload(let kind, let name, let tokens):
            if kind == .toolJSON || kind == .toolText {
                return .closed(kind, name: name, tokens: tokens)
            }
            return nil
        case .header(let tokens) where !tokens.isEmpty:
            return .rejected("Incomplete Inkling frame header at generation end")
        default:
            return nil
        }
    }

    private func description(of kind: InklingContentKind) -> String {
        switch kind {
        case .thinking: "thinking"
        case .text: "text"
        case .toolJSON, .toolText: "tool"
        }
    }
}

/// Routes Inkling frames to reasoning, response and tool-call events.
private struct InklingProtocolDecoder {
    private var parser: InklingFrameParser
    private let tokenizer: any Tokenizer
    private let tools: [[String: any Sendable]]?
    private let allowedToolNames: Set<String>?
    private let validationPolicy: ToolCallValidationPolicy
    private var detokenizer: any StreamingDetokenizer
    private var isFinished = false

    /// Whether the latest token streamed text that ends in a visible character.
    private(set) var endsInVisibleText = false

    var isInsideReasoning: Bool { parser.isInsideThinking }

    init?(
        tokenizer: any Tokenizer, tools: [[String: any Sendable]]?,
        toolCallPolicy: ToolCallPolicy = .init()
    ) {
        guard let parser = InklingFrameParser(tokenizer: tokenizer) else { return nil }
        self.parser = parser
        self.tokenizer = tokenizer
        self.tools = tools
        self.allowedToolNames = tools.map { Set($0.compactMap(Self.toolName)) }
        self.validationPolicy = toolCallPolicy.validation
        self.detokenizer = tokenizer.makeStreamingDetokenizer()
    }

    private static func toolName(_ tool: [String: any Sendable]) -> String? {
        if let function = tool["function"] as? [String: any Sendable] {
            return function["name"] as? String
        }
        return tool["name"] as? String
    }

    mutating func push(_ token: Int) -> TokenStreamEvent? {
        route(parser.push(token))
    }

    mutating func finish() -> TokenStreamEvent? {
        guard let step = parser.finish() else { return nil }
        if case .closed(_, let name, let tokens) = step {
            let text = tokenizer.decode(tokenIds: tokens, skipSpecialTokens: false)
            return .rejectedToolCall(
                rejection(.incompleteOutput, name: name.isEmpty ? nil : name, rawText: text))
        }
        return route(step)
    }

    private mutating func route(_ step: InklingParseStep) -> TokenStreamEvent? {
        endsInVisibleText = false
        switch step {
        case .consumed:
            return nil
        case .opened:
            detokenizer = tokenizer.makeStreamingDetokenizer()
            return nil
        case .payload(let kind, let token):
            guard kind == .thinking || kind == .text else { return nil }
            detokenizer.append(token: token)
            guard let text = detokenizer.next(), !text.isEmpty else { return nil }
            endsInVisibleText = text.last?.isWhitespace == false
            return kind == .thinking ? .reasoning(text) : .response(text)
        case .closed(let kind, let name, let tokens):
            switch kind {
            case .thinking:
                return detokenizer.finish().map(TokenStreamEvent.reasoning)
            case .text:
                return detokenizer.finish().map(TokenStreamEvent.response)
            case .toolJSON, .toolText:
                let text = tokenizer.decode(tokenIds: tokens, skipSpecialTokens: false)
                return toolCall(header: name, payload: text)
            }
        case .rejected(let message):
            detokenizer = tokenizer.makeStreamingDetokenizer()
            return .protocolError(message)
        }
    }

    /// A tool call from a `{"name": ..., "args": {...}}` payload.
    private func toolCall(header: String, payload: String) -> TokenStreamEvent {
        let headerName = header.isEmpty ? nil : header
        guard let data = payload.data(using: .utf8),
            let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else {
            return .rejectedToolCall(
                rejection(.malformedSyntax, name: headerName, rawText: payload))
        }
        guard let name = (object["name"] as? String) ?? headerName, !name.isEmpty else {
            return .rejectedToolCall(rejection(.missingToolName, name: nil, rawText: payload))
        }
        if let headerName, headerName != name {
            return .rejectedToolCall(rejection(.malformedSyntax, name: name, rawText: payload))
        }
        let rawArguments = object["args"] ?? object["arguments"] ?? [String: Any]()
        guard let arguments = rawArguments as? [String: Any],
            let argumentData = try? JSONSerialization.data(withJSONObject: arguments),
            let values = try? JSONDecoder().decode([String: JSONValue].self, from: argumentData)
        else {
            return .rejectedToolCall(rejection(.invalidArguments, name: name, rawText: payload))
        }
        guard allowedToolNames?.contains(name) ?? true else {
            return .rejectedToolCall(rejection(.undeclaredTool, name: name, rawText: payload))
        }

        let call = ToolArgumentNormalization.normalize(
            ToolCall(function: .init(name: name, arguments: values)), tools: tools)
        if validationPolicy == .strict,
            case .invalid(let violations) = ToolSchemaValidator.validate(
                arguments: call.function.arguments, forToolNamed: name, in: tools)
        {
            return .rejectedToolCall(
                rejection(
                    .invalidArguments, name: name, rawText: payload,
                    detail: ToolSchemaValidator.describe(violations)))
        }
        return .toolCall(
            ToolCall(function: call.function, id: ToolCallFormat.inkling.generateToolCallID()))
    }

    private func rejection(
        _ reason: RejectedToolCall.Reason, name: String?, rawText: String, detail: String? = nil
    ) -> RejectedToolCall {
        RejectedToolCall(
            reason: reason, format: .inkling, toolName: name, rawText: rawText,
            detail: detail ?? reason.diagnosticDetail)
    }
}

/// Token-stream decoder for Inkling 2's message frames and JSON tool calls.
struct InklingStreamAdapter: TokenStreamDecoder {
    private var protocolDecoder: InklingProtocolDecoder
    private var stopStringFilter: StopStringFilter
    private(set) var rejectedToolCallCount = 0
    let additionalStopTokenIDs: Set<Int>
    let receivesStopTokens = true

    var isInsideReasoning: Bool { protocolDecoder.isInsideReasoning }

    /// A step can end inside a frame after text that ends in a visible character, with no
    /// stop-string prefix held back.
    var canEndForSteering: Bool {
        protocolDecoder.endsInVisibleText && stopStringFilter.buffer.isEmpty
    }

    init?(
        tokenizer: any Tokenizer,
        tools: [[String: any Sendable]]?,
        stopStrings: Set<String>,
        toolCallPolicy: ToolCallPolicy = .init()
    ) {
        guard
            let decoder = InklingProtocolDecoder(
                tokenizer: tokenizer, tools: tools, toolCallPolicy: toolCallPolicy)
        else { return nil }
        self.protocolDecoder = decoder
        self.stopStringFilter = StopStringFilter(stopStrings: stopStrings)
        self.additionalStopTokenIDs = Set(
            ["<|content_model_end_sampling|>"].compactMap(tokenizer.convertTokenToId))
    }

    mutating func push(_ token: Int, emit: (TokenStreamEvent) -> Bool) -> Bool {
        guard let event = protocolDecoder.push(token) else { return true }
        return emitEvent(event, emit: emit)
    }

    mutating func finish(emit: (TokenStreamEvent) -> Bool) -> Bool {
        if let event = protocolDecoder.finish(), !emitEvent(event, emit: emit) { return false }
        if let text = stopStringFilter.finish() { return emit(.response(text)) }
        return true
    }

    private mutating func emitEvent(
        _ event: TokenStreamEvent, emit: (TokenStreamEvent) -> Bool
    ) -> Bool {
        switch event {
        case .reasoning(let text):
            return emit(.reasoning(text))
        case .response(let text):
            let result = stopStringFilter.process(text)
            if let response = result.text, !emit(.response(response)) { return false }
            if result.stopped {
                _ = emit(.stop)
                return false
            }
            return true
        case .toolCall(let call):
            if let response = stopStringFilter.finish(), !emit(.response(response)) {
                return false
            }
            return emit(.toolCall(call))
        case .rejectedToolCall(let rejection):
            rejectedToolCallCount += 1
            if let response = stopStringFilter.finish(), !emit(.response(response)) {
                return false
            }
            return emit(.rejectedToolCall(rejection))
        case .protocolError(let message):
            return emit(.protocolError(message))
        case .stop:
            return emit(.stop)
        }
    }
}

/// Parses Inkling 2 tool calls from text, for callers outside the token stream decoder:
/// `<|content_invoke_tool_json|>{"name": "f", "args": {...}}<|end_message|>`.
public struct InklingToolCallParser: ToolCallParser, Sendable {
    public let startTag: String? = "<|content_invoke_tool_json|>"
    public let endTag: String? = "<|end_message|>"

    public init() {}

    public func parse(content: String, tools: [[String: any Sendable]]?) -> ToolCall? {
        var text = content
        if let start = text.range(of: startTag!) {
            text = String(text[start.upperBound...])
        }
        if let end = text.range(of: endTag!, options: .backwards) {
            text = String(text[..<end.lowerBound])
        }
        guard
            let data = text.trimmingCharacters(in: .whitespacesAndNewlines).data(using: .utf8),
            let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            let name = object["name"] as? String,
            let arguments = (object["args"] ?? object["arguments"] ?? [String: Any]())
                as? [String: Any],
            let argumentData = try? JSONSerialization.data(withJSONObject: arguments),
            let values = try? JSONDecoder().decode([String: JSONValue].self, from: argumentData)
        else { return nil }
        return ToolCall(function: .init(name: name, arguments: values))
    }
}
