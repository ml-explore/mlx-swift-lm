// Copyright © 2026 Apple Inc.

/// Deterministic token-id state machine for the Harmony framing protocol.
///
/// Converts a stream of token IDs into typed ``HarmonyParseStep`` values. It
/// knows only about Harmony control tokens and header text. It does **not**:
///
/// - emit ``Generation`` events
/// - decode payload text
/// - authorize tool calls
/// - touch chat history
///
/// That separation keeps the parser unit-testable and free of recovery
/// heuristics that belong in higher layers.
package struct HarmonyFrameParser: Sendable {

    private enum Tag {
        static let start = "<|start|>"
        static let channel = "<|channel|>"
        static let message = "<|message|>"
        static let end = "<|end|>"
        static let call = "<|call|>"
        static let `return` = "<|return|>"
        static let constrain = "<|constrain|>"
    }

    private struct ControlTokens: Sendable {
        let start: Int
        let channel: Int
        let message: Int
        let end: Int
        let call: Int
        let `return`: Int
        let constrain: Int
    }

    private enum State: Sendable {
        case idle
        case header([Int])
        case body(header: HarmonyHeader, payload: [Int])
    }

    private let tokenizer: any Tokenizer
    private let controls: ControlTokens
    private var state: State = .idle

    /// Fails when the tokenizer has no Harmony control tokens, which means this
    /// is not a Harmony model and the caller should use a different path.
    package init?(tokenizer: any Tokenizer) {
        guard
            let start = Self.resolveTokenID(Tag.start, tokenizer: tokenizer),
            let channel = Self.resolveTokenID(Tag.channel, tokenizer: tokenizer),
            let message = Self.resolveTokenID(Tag.message, tokenizer: tokenizer),
            let end = Self.resolveTokenID(Tag.end, tokenizer: tokenizer),
            let call = Self.resolveTokenID(Tag.call, tokenizer: tokenizer),
            let `return` = Self.resolveTokenID(Tag.return, tokenizer: tokenizer),
            let constrain = Self.resolveTokenID(Tag.constrain, tokenizer: tokenizer)
        else {
            return nil
        }
        guard Set([start, channel, message, end, call, `return`, constrain]).count == 7 else {
            return nil
        }
        self.tokenizer = tokenizer
        self.controls = ControlTokens(
            start: start,
            channel: channel,
            message: message,
            end: end,
            call: call,
            return: `return`,
            constrain: constrain
        )
    }

    /// Harmony's two decode-time stop tokens. Resolve them from the tokenizer
    /// even when a converted checkpoint omitted `generation_config.json`.
    package static func stopTokenIDs(tokenizer: any Tokenizer) -> Set<Int>? {
        Self(tokenizer: tokenizer)?.semanticStopTokenIDs
    }

    package var semanticStopTokenIDs: Set<Int> {
        [controls.call, controls.return]
    }

    package var isInsideReasoning: Bool {
        guard case .body(let header, _) = state else { return false }
        return header.channel == .analysis
    }

    /// The commit token that separates an assistant function call from its
    /// tool result in a rendered Harmony conversation.
    package static func callTokenID(tokenizer: any Tokenizer) -> Int? {
        Self(tokenizer: tokenizer)?.controls.call
    }

    /// Feeds one generated token and returns the parse steps it produces
    /// (usually zero or one; a boundary token can close one frame and start
    /// the next header).
    package mutating func push(_ token: Int) -> [HarmonyParseStep] {
        switch state {
        case .idle:
            if startsHeader(token) {
                state = .header([token])
                return [.consumed]
            }
            if isControlToken(token) {
                // Never surface orphaned Harmony control tokens as response
                // text, even when the surrounding generation is malformed.
                return [.consumed]
            }
            // Non-conformant preamble: treat as a synthetic `final` payload so
            // plain text is not dropped when a model skips Harmony framing.
            let header = HarmonyHeader(
                channel: .final, rawChannel: nil, recipient: nil, constrain: nil)
            state = .body(header: header, payload: [token])
            return [.payload(header: header, token: token)]

        case .header(var headerTokens):
            headerTokens.append(token)
            if token == controls.message {
                let header = parseHeader(from: decode(headerTokens.dropLast()))
                state = .body(header: header, payload: [])
                return [.consumed]
            }
            if isFrameTerminator(token) {
                // Header without `<|message|>` is malformed; drop it.
                state = .idle
                return [.consumed]
            }
            state = .header(headerTokens)
            return [.consumed]

        case .body(let header, var payload):
            if isBoundary(token) {
                let terminator = terminator(for: token) ?? .incomplete
                let frame = HarmonyFrame(
                    header: header, payloadTokenIds: payload, terminator: terminator)
                var steps: [HarmonyParseStep] = [.closed(frame)]
                if startsHeader(token) {
                    state = .header([token])
                    steps.append(.consumed)
                } else {
                    state = .idle
                }
                return steps
            }
            payload.append(token)
            state = .body(header: header, payload: payload)
            return [.payload(header: header, token: token)]
        }
    }

    /// Flushes an open frame when generation ends.
    package mutating func finish() -> [HarmonyParseStep] {
        switch state {
        case .idle:
            return []
        case .header:
            state = .idle
            return []
        case .body(let header, let payload):
            state = .idle
            let frame = HarmonyFrame(
                header: header, payloadTokenIds: payload, terminator: .incomplete)
            return [.closed(frame)]
        }
    }

    // MARK: - Header decoding

    private func parseHeader(from text: String) -> HarmonyHeader {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)

        let roleSection: String
        let channelSection: String
        if let channelRange = trimmed.range(of: Tag.channel, options: .backwards) {
            roleSection = String(trimmed[..<channelRange.lowerBound])
            channelSection = String(trimmed[channelRange.upperBound...])
        } else {
            roleSection = trimmed
            channelSection = ""
        }

        let rawChannel =
            channelSection
            .split(whereSeparator: { $0.isWhitespace || $0 == "<" })
            .first
            .map(String.init)

        let channel: HarmonyChannel
        switch rawChannel {
        case "analysis": channel = .analysis
        case "commentary": channel = .commentary
        case "final": channel = .final
        default: channel = .other
        }

        let recipient = self.recipient(in: channelSection) ?? self.recipient(in: roleSection)
        let constrain = self.constrain(in: channelSection) ?? self.constrain(in: roleSection)

        return HarmonyHeader(
            channel: channel,
            rawChannel: channel == .other ? rawChannel : nil,
            recipient: recipient,
            constrain: constrain)
    }

    private func recipient(in section: String) -> String? {
        guard let toRange = standaloneAttributeRange(named: "to", in: section) else {
            return nil
        }
        var suffix = String(section[toRange.upperBound...])
        if let constrainRange = suffix.range(of: Tag.constrain) {
            suffix = String(suffix[..<constrainRange.lowerBound])
        }
        let recipient =
            suffix
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .split(whereSeparator: { $0.isWhitespace || $0 == "<" || $0 == ">" })
            .first
            .map(String.init) ?? ""
        return recipient.isEmpty ? nil : recipient
    }

    /// Finds `name=` only when it begins an actual header attribute. A raw
    /// substring search would accept malformed fields such as `not_to=` and
    /// could turn them into dispatchable recipients.
    private func standaloneAttributeRange(named name: String, in section: String) -> Range<
        String.Index
    >? {
        let marker = "\(name)="
        var searchStart = section.startIndex

        while searchStart < section.endIndex,
            let range = section.range(of: marker, range: searchStart ..< section.endIndex)
        {
            if range.lowerBound == section.startIndex {
                return range
            }

            let preceding = section[section.index(before: range.lowerBound)]
            if preceding.isWhitespace {
                return range
            }

            searchStart = range.upperBound
        }

        return nil
    }

    private func constrain(in section: String) -> String? {
        guard let range = section.range(of: Tag.constrain) else { return nil }
        let after = section[range.upperBound...]
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let value =
            after
            .split(whereSeparator: { $0.isWhitespace || $0 == "<" })
            .first
            .map(String.init) ?? ""
        return value.isEmpty ? nil : value
    }

    // MARK: - Token classification

    private static func resolveTokenID(_ token: String, tokenizer: any Tokenizer) -> Int? {
        if let id = tokenizer.convertTokenToId(token) {
            guard id != tokenizer.unknownTokenId else { return nil }
            return id
        }
        let encoded = tokenizer.encode(text: token, addSpecialTokens: false)
        guard encoded.count == 1, encoded[0] != tokenizer.unknownTokenId else { return nil }
        return encoded[0]
    }

    private func decode<S: Sequence>(_ tokens: S) -> String where S.Element == Int {
        tokenizer.decode(tokenIds: Array(tokens), skipSpecialTokens: false)
    }

    private func startsHeader(_ token: Int) -> Bool {
        token == controls.channel || token == controls.start
    }

    private func isFrameTerminator(_ token: Int) -> Bool {
        token == controls.end || token == controls.call || token == controls.return
    }

    private func isBoundary(_ token: Int) -> Bool {
        isFrameTerminator(token) || startsHeader(token) || token == controls.message
    }

    private func terminator(for token: Int) -> HarmonyTerminator? {
        if token == controls.end { return .end }
        if token == controls.call { return .call }
        if token == controls.return { return .return }
        return nil
    }

    private func isControlToken(_ token: Int) -> Bool {
        token == controls.start || token == controls.channel || token == controls.message
            || token == controls.end || token == controls.call || token == controls.return
            || token == controls.constrain
    }
}
