// Copyright © 2026 Apple Inc.

import Foundation

/// Routes completed Harmony frames (and streaming payload tokens) into the
/// events the generation loop understands.
///
/// Policy encoded here — not in the frame parser:
///
/// - `final` payload is user-visible response text
/// - `analysis` is private reasoning and is dropped from the public stream
/// - recipient-less `commentary` is a user-visible preamble
/// - `commentary to=functions.<name>` becomes a tool call **only** when the
///   name is in the caller's declared allowlist
/// - GPT-OSS commits **one** tool call per generation turn; further candidates
///   are ignored
/// - tool argument JSON is decoded strictly into `[String: JSONValue]`
package struct HarmonyOutputRouter {

    /// Events the generation adapter may emit on the public stream.
    package enum Event: Sendable, Equatable {
        case response(String)
        case toolCall(ToolCall)
        case rejectedToolCall(RejectedToolCall)
    }

    /// Declared tool names the caller is willing to dispatch. `nil` means no
    /// tools were offered for this generation, so no tool calls are accepted.
    private let allowedToolNames: Set<String>?
    private var hasEmittedToolCall = false
    private var responseDetokenizer: NaiveStreamingDetokenizer
    private let tokenizer: any Tokenizer

    package init(tokenizer: any Tokenizer, allowedToolNames: Set<String>?) {
        self.tokenizer = tokenizer
        self.allowedToolNames = allowedToolNames
        self.responseDetokenizer = NaiveStreamingDetokenizer(tokenizer: tokenizer)
    }

    /// Routes one parse step. `final` payloads and recipient-less `commentary`
    /// preambles stream as ``Event/response``; tool calls are emitted when a
    /// committed commentary frame targets an allowed function.
    package mutating func route(_ step: HarmonyParseStep) -> [Event] {
        switch step {
        case .consumed:
            return []

        case .payload(let header, let token):
            guard isPublicResponse(header) else { return [] }
            responseDetokenizer.append(token: token)
            if let text = responseDetokenizer.next(), !text.isEmpty {
                return [.response(text)]
            }
            return []

        case .closed(let frame):
            // Reset the response detokenizer between frames so a later final
            // frame starts clean after a tool turn.
            if isPublicResponse(frame.header) {
                responseDetokenizer = NaiveStreamingDetokenizer(tokenizer: tokenizer)
                return []
            }
            return routeClosedFrame(frame)
        }
    }

    /// Flushes any open response detokenizer state. Currently a no-op because
    /// public payload tokens are already streamed; kept for symmetry with the
    /// parser's ``HarmonyFrameParser/finish()``.
    package mutating func finish() -> [Event] {
        []
    }

    // MARK: - Frame → tool call

    private mutating func routeClosedFrame(_ frame: HarmonyFrame) -> [Event] {
        guard frame.header.channel == .commentary else {
            // Analysis and unknown channels are not public output.
            return []
        }
        guard let recipient = frame.header.recipient else {
            // Commentary preambles stream through the payload path.
            return []
        }
        let payload = decodePayload(frame.payloadTokenIds)
        guard let name = canonicalFunctionName(recipient) else {
            guard recipient.trimmingCharacters(in: .whitespacesAndNewlines) == "functions." else {
                return []
            }
            return [
                .rejectedToolCall(
                    rejection(.missingToolName, name: nil, rawText: payload))
            ]
        }

        guard frame.terminator == .call else {
            let reason: RejectedToolCall.Reason =
                frame.terminator == .incomplete ? .incompleteOutput : .malformedSyntax
            return [.rejectedToolCall(rejection(reason, name: name, rawText: payload))]
        }
        guard !hasEmittedToolCall else {
            // One committed tool call per generation turn.
            return []
        }
        guard let arguments = strictJSONObject(payload) else {
            return [
                .rejectedToolCall(
                    rejection(.invalidArguments, name: name, rawText: payload))
            ]
        }
        guard isAllowed(name) else {
            return [
                .rejectedToolCall(
                    rejection(.undeclaredTool, name: name, rawText: payload))
            ]
        }

        hasEmittedToolCall = true
        let toolCall = ToolCall(
            function: .init(name: name, arguments: arguments),
            id: ToolCallFormat.gptOSS.generateToolCallID())
        return [.toolCall(toolCall)]
    }

    private func rejection(
        _ reason: RejectedToolCall.Reason,
        name: String?,
        rawText: String
    ) -> RejectedToolCall {
        RejectedToolCall(
            reason: reason,
            format: .gptOSS,
            toolName: name,
            rawText: rawText,
            detail: reason.diagnosticDetail)
    }

    private func isAllowed(_ name: String) -> Bool {
        guard let allowedToolNames else {
            // No tool schema was provided for this generation.
            return false
        }
        return allowedToolNames.contains(name)
    }

    private func canonicalFunctionName(_ recipient: String) -> String? {
        let trimmed = recipient.trimmingCharacters(in: .whitespacesAndNewlines)
        guard trimmed.hasPrefix("functions.") else { return nil }
        let name = String(trimmed.dropFirst("functions.".count))
        return name.isEmpty ? nil : name
    }

    private func isPublicResponse(_ header: HarmonyHeader) -> Bool {
        header.channel == .final
            || (header.channel == .commentary && header.recipient == nil)
    }

    private func decodePayload(_ tokens: [Int]) -> String {
        tokenizer.decode(tokenIds: tokens, skipSpecialTokens: false)
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// Strict object decode. Rejects non-objects, empty payloads, and any
    /// wrapper recovery (code fences, double-encoded strings, nested
    /// `arguments` keys).
    private func strictJSONObject(_ text: String) -> [String: JSONValue]? {
        guard !text.isEmpty, let data = text.data(using: .utf8) else { return nil }
        guard let value = try? JSONDecoder().decode(JSONValue.self, from: data),
            case .object(let object) = value
        else {
            return nil
        }
        return object
    }
}

// MARK: - Allowlist extraction

extension HarmonyOutputRouter {
    /// Tool names declared in the generation's tool schemas.
    package static func allowedToolNames(
        from tools: [[String: any Sendable]]?
    ) -> Set<String>? {
        guard let tools, !tools.isEmpty else { return nil }
        var names = Set<String>()
        for tool in tools {
            if let function = tool["function"] as? [String: any Sendable],
                let name = function["name"] as? String, !name.isEmpty
            {
                names.insert(name)
            } else if let name = tool["name"] as? String, !name.isEmpty {
                names.insert(name)
            }
        }
        return names.isEmpty ? nil : names
    }
}
