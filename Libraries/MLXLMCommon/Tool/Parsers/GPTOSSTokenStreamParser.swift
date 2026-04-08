// Copyright © 2025 Apple Inc.

import Foundation
import Tokenizers

/// Token-level Harmony parser used only for GPT-OSS streaming.
///
/// The public `Generation` surface stays unchanged:
/// - visible Harmony frames still stream as `.chunk`
/// - commentary tool calls are hidden from `.chunk` and surfaced as `.toolCall`
struct GPTOSSTokenStreamParser {

    private enum Tag {
        static let start = "<|start|>"
        static let channel = "<|channel|>"
        static let message = "<|message|>"
        static let end = "<|end|>"
        static let call = "<|call|>"
        static let `return` = "<|return|>"
        static let constrain = "<|constrain|>"
    }

    private struct ControlTokens {
        let start: Int?
        let channel: Int
        let message: Int
        let end: Int?
        let call: Int?
        let `return`: Int?
    }

    private struct ToolHeader {
        let name: String
    }

    private enum State {
        case passthrough
        case collectingHeader([Int])
        case collectingToolCall(header: ToolHeader, payload: [Int])
    }

    private let tokenizer: Tokenizer
    private let controls: ControlTokens
    private var detokenizer: NaiveStreamingDetokenizer
    private var state: State = .passthrough

    init?(tokenizer: Tokenizer) {
        guard
            let channel = Self.resolveTokenID(Tag.channel, tokenizer: tokenizer),
            let message = Self.resolveTokenID(Tag.message, tokenizer: tokenizer)
        else {
            return nil
        }

        self.tokenizer = tokenizer
        self.detokenizer = NaiveStreamingDetokenizer(tokenizer: tokenizer)
        self.controls = ControlTokens(
            start: Self.resolveTokenID(Tag.start, tokenizer: tokenizer),
            channel: channel,
            message: message,
            end: Self.resolveTokenID(Tag.end, tokenizer: tokenizer),
            call: Self.resolveTokenID(Tag.call, tokenizer: tokenizer),
            return: Self.resolveTokenID(Tag.return, tokenizer: tokenizer)
        )
    }

    mutating func onToken(
        _ token: Int,
        emit: (sending Generation) -> AsyncStream<Generation>.Continuation.YieldResult
    ) -> Bool {
        switch state {
        case .passthrough:
            if startsHeader(token) {
                state = .collectingHeader([token])
                return true
            }
            return commitVisible([token], emit: emit)

        case .collectingHeader(var headerTokens):
            headerTokens.append(token)

            if token == controls.message {
                let visibleHeaderTokens = headerTokens
                let headerText = decode(headerTokens.dropLast())
                if let toolHeader = parseToolHeader(from: headerText) {
                    state = .collectingToolCall(header: toolHeader, payload: [])
                    return true
                } else {
                    state = .passthrough
                    return commitVisible(visibleHeaderTokens, emit: emit)
                }
            }

            if continuesRoleHeader(headerTokens) {
                state = .collectingHeader(headerTokens)
                return true
            }

            if isHardBoundary(token) {
                state = .passthrough
                return commitVisible(headerTokens, emit: emit)
            }

            state = .collectingHeader(headerTokens)
            return true

        case .collectingToolCall(let header, var payload):
            if isToolCallBoundary(token) {
                if !emitToolCall(header: header, payload: payload, emit: emit) {
                    return false
                }

                if startsHeader(token) {
                    state = .collectingHeader([token])
                } else {
                    state = .passthrough
                }
                return true
            }

            payload.append(token)
            state = .collectingToolCall(header: header, payload: payload)
            return true
        }
    }

    mutating func onGenerationEnd(
        emit: (sending Generation) -> AsyncStream<Generation>.Continuation.YieldResult
    ) {
        switch state {
        case .passthrough:
            break
        case .collectingHeader(let headerTokens):
            _ = commitVisible(headerTokens, emit: emit)
        case .collectingToolCall(let header, let payload):
            _ = emitToolCall(header: header, payload: payload, emit: emit)
        }
        state = .passthrough
    }

    private static func resolveTokenID(_ token: String, tokenizer: Tokenizer) -> Int? {
        if let id = tokenizer.convertTokenToId(token) {
            return id
        }

        let encoded = tokenizer.encode(text: token, addSpecialTokens: false)
        guard encoded.count == 1 else { return nil }
        return encoded[0]
    }

    private func decode<S: Sequence>(_ tokens: S) -> String where S.Element == Int {
        tokenizer.decode(tokens: Array(tokens), skipSpecialTokens: false)
    }

    private func startsHeader(_ token: Int) -> Bool {
        token == controls.channel || matches(token, controls.start)
    }

    private func isHardBoundary(_ token: Int) -> Bool {
        if token == controls.channel { return true }
        if matches(token, controls.start) { return true }
        if token == controls.message { return true }
        if matches(token, controls.end) { return true }
        if matches(token, controls.call) { return true }
        if matches(token, controls.return) { return true }
        return false
    }

    private func isToolCallBoundary(_ token: Int) -> Bool {
        if token == controls.channel { return true }
        if matches(token, controls.start) { return true }
        if matches(token, controls.call) { return true }
        if matches(token, controls.end) { return true }
        if matches(token, controls.return) { return true }
        return false
    }

    private func continuesRoleHeader(_ headerTokens: [Int]) -> Bool {
        guard headerTokens.last == controls.channel else { return false }
        guard matches(headerTokens.first ?? -1, controls.start) else { return false }
        return !headerTokens.dropLast().contains(controls.channel)
    }

    private func matches(_ token: Int, _ control: Int?) -> Bool {
        guard let control else { return false }
        return token == control
    }

    private mutating func commitVisible(
        _ tokens: [Int],
        emit: (sending Generation) -> AsyncStream<Generation>.Continuation.YieldResult
    ) -> Bool {
        for token in tokens {
            detokenizer.append(token: token)
            if let chunk = detokenizer.next() {
                if case .terminated = emit(.chunk(chunk)) {
                    return false
                }
            }
        }
        return true
    }

    private func emitToolCall(
        header: ToolHeader,
        payload: [Int],
        emit: (sending Generation) -> AsyncStream<Generation>.Continuation.YieldResult
    ) -> Bool {
        let payloadText = decode(payload)
        let normalized = normalizedArgumentsJSON(from: payloadText)
        guard let arguments = tryParseJSON(normalized) as? [String: any Sendable] else {
            return true
        }

        let toolCall = ToolCall(function: .init(name: header.name, arguments: arguments))
        if case .terminated = emit(.toolCall(toolCall)) {
            return false
        }
        return true
    }

    private func parseToolHeader(from header: String) -> ToolHeader? {
        let trimmedHeader = header.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let channelRange = trimmedHeader.range(of: Tag.channel, options: .backwards) else {
            return nil
        }

        let channelHeader = String(trimmedHeader[channelRange.upperBound...]).trimmingCharacters(
            in: .whitespacesAndNewlines)
        guard channelHeader.hasPrefix("commentary") else { return nil }

        let roleHeader = String(trimmedHeader[..<channelRange.lowerBound])
        let recipient = recipient(in: channelHeader) ?? recipient(in: roleHeader) ?? ""
        let normalizedName = canonicalName(from: recipient)
        guard !normalizedName.isEmpty else { return nil }

        return ToolHeader(name: normalizedName)
    }

    private func recipient(in headerSection: String) -> String? {
        guard let toRange = headerSection.range(of: "to=") else { return nil }
        var suffix = String(headerSection[toRange.upperBound...])
        if let constrainRange = suffix.range(of: Tag.constrain) {
            suffix = String(suffix[..<constrainRange.lowerBound])
        }
        let recipient =
            suffix
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .split(whereSeparator: { $0.isWhitespace || $0 == "<" || $0 == ">" })
            .first.map(String.init) ?? ""
        return recipient.isEmpty ? nil : recipient
    }

    private func canonicalName(from rawName: String) -> String {
        let trimmed = rawName.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return "" }
        if trimmed.hasPrefix("functions.") {
            return String(trimmed.dropFirst("functions.".count))
        }
        return trimmed
    }

    private func normalizedArgumentsJSON(from payload: String) -> String {
        let trimmed = payload.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return "{}" }

        let unfenced = stripCodeFenceIfNeeded(from: trimmed)

        if let unwrappedJSONString = unwrappedJSONString(from: unfenced) {
            return unwrappedJSONString
        }

        if let extractedObject = extractedNestedJSONObject(from: unfenced) {
            return extractedObject
        }

        return unfenced
    }

    private func stripCodeFenceIfNeeded(from text: String) -> String {
        guard text.hasPrefix("```") else { return text }
        var normalized = text
        if let firstNewline = normalized.firstIndex(of: "\n") {
            normalized = String(normalized[normalized.index(after: firstNewline)...])
        }
        if let closingFence = normalized.range(of: "```", options: .backwards) {
            normalized = String(normalized[..<closingFence.lowerBound])
        }
        return normalized.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private func unwrappedJSONString(from text: String) -> String? {
        guard let data = text.data(using: .utf8),
            let stringValue = try? JSONDecoder().decode(String.self, from: data)
        else {
            return nil
        }
        let normalized = stringValue.trimmingCharacters(in: .whitespacesAndNewlines)
        guard normalized.hasPrefix("{"), normalized.hasSuffix("}") else { return nil }
        return normalized
    }

    private func extractedNestedJSONObject(from text: String) -> String? {
        guard let data = text.data(using: .utf8),
            let json = try? JSONSerialization.jsonObject(with: data),
            let dict = json as? [String: Any]
        else {
            return nil
        }

        let wrapperKeys = ["arguments", "args", "input", "parameters", "kwargs"]
        for key in wrapperKeys {
            guard let nested = dict[key] else { continue }
            if let nestedDict = nested as? [String: Any],
                JSONSerialization.isValidJSONObject(nestedDict),
                let nestedData = try? JSONSerialization.data(
                    withJSONObject: nestedDict, options: [.sortedKeys]),
                let nestedJSON = String(data: nestedData, encoding: .utf8)
            {
                return nestedJSON
            }
            if let nestedString = nested as? String {
                let normalized = nestedString.trimmingCharacters(in: .whitespacesAndNewlines)
                if normalized.hasPrefix("{"), normalized.hasSuffix("}") {
                    return normalized
                }
            }
        }
        return nil
    }
}
