// Copyright © 2026 Apple Inc.

import Foundation

/// A bounded, streaming recovery pass for common textual tool-call dialects.
///
/// The selected `ToolCallParser` remains authoritative. This scanner only owns
/// dialect markers that the selected parser cannot consume, plus exact
/// `declaredTool[ARGS]{...}` rehearsals. It deliberately uses fixed-string
/// searches and structural JSON scanning instead of regular expressions or
/// speculative JSON decoding on ordinary response text.
///
/// Recovery is a *context lexer*, not just a collection of parsers. Text is
/// classified before any candidate is considered, and only committed response
/// text is eligible for promotion:
///
/// - **reasoning** (`<think>`, `<thinking>`, `[THINK]` spans): models rehearse
///   and reject candidate calls while reasoning; a call mentioned there is not
///   a committed action and is never promoted.
/// - **code** (Markdown fenced blocks and inline code spans): examples and
///   documentation of tool syntax are data, not actions.
/// - **JSON data** (objects, arrays and strings): ordinary structured output
///   remains response data; protocol-shaped string contents stay inert.
/// - **native frames** (the selected format's own protocol): opaque to
///   recovery, and a JSON or Qwen-XML payload frame only closes after a
///   structurally balanced payload, so a literal close marker inside a string
///   argument cannot expose fabricated suffixes to recovery.
///
/// Explicit protocol attempts (`<tool_call>`, `<|tool_call>`, `<function=`,
/// `[TOOL_CALLS]`) that are malformed, incomplete at end of stream, or name an
/// undeclared tool are reported as rejected rather than leaked as response
/// text. Ambiguous markerless candidates that fail validation remain text.
struct TextToolCallRecoveryScanner: Sendable {
    enum Output: Sendable {
        case text(String)
        /// Response data that has already been classified as non-executable.
        /// The downstream processor must emit it directly and must never feed
        /// it through the selected format's native tool parser.
        case protectedText(String)
        case toolCall(ToolCall, rawText: String)
        /// An explicit protocol attempt that could not become a call.
        case rejected(rawText: String, reason: RejectedToolCall.Reason, toolName: String?)
    }

    /// Explicit alternate-dialect protocol markers owned by recovery.
    private enum ExplicitKind: Sendable {
        case toolCallFrame
        case gemma4
        case qwenFunction
        case mistral
    }

    private enum Kind: Sendable, Equatable {
        case nativeFrame(endMarker: String)
        case nativeUntilEOS
        case nativeInlineJSON
        case opaqueJSONValue
        case reasoning(close: String)
        case explicit(ExplicitKind)
        case declaredArgs(name: String)
        /// A backtick run: an inline code span, or a fenced block at line start.
        case codeBacktick
        /// A tilde run: a fenced block at line start, otherwise ordinary text.
        case codeTilde
    }

    private enum Context: Sendable, Equatable {
        case response
        case reasoning(close: String)
        case codeFence(character: Character, count: Int)
        case codeSpan(count: Int)
        case jsonValue
        case nativeFrame(endMarker: String)
        case nativeUntilEOS
        /// Once a structured data value exceeds the inspection budget, remain
        /// fail-closed until EOS. This keeps memory and work bounded without
        /// ever re-entering executable response context from the middle of a
        /// JSON string or container.
        case opaqueUntilEOS
    }

    private struct Signal: Sendable {
        let text: String
        let kind: Kind
    }

    /// Avoid rescanning an incomplete candidate for every token. A candidate
    /// is structurally reconsidered only when the new chunk can contain a
    /// closing boundary (or when the byte limit/EOS forces a decision).
    private enum PendingCandidate: Sendable {
        case explicit(ExplicitKind)
        case declaredArgs(name: String)

        func shouldInspect(_ chunk: String) -> Bool {
            switch self {
            case .explicit(.mistral), .declaredArgs:
                chunk.contains("}")
            case .explicit(.toolCallFrame), .explicit(.gemma4), .explicit(.qwenFunction):
                chunk.contains(">")
            }
        }
    }

    private let primaryFormat: ToolCallFormat
    private let policy: ToolCallRecoveryPolicy
    private let tools: [[String: any Sendable]]
    private let allowedToolNames: Set<String>
    private let allowedToolNamesByLength: [String]
    private let signals: [Signal]
    private let potentialSignalPrefixes: Set<String>
    private let maximumPotentialSignalPrefixLength: Int
    private let potentialPrefixEndCharacters: Set<Character>
    private let maximumBufferedByteCount: Int
    private let jsonScanner = JSONLeadingObjectScanner(startCharacter: "{")
    private let gemmaScanner = StructuredTextScanner(
        quotes: ["\""], escapeMarker: "<|\"|>")
    private let framedParser = Qwen35ToolCallParser(
        startTag: "<tool_call>", endTag: "</tool_call>")
    private let gemmaParser = GemmaFunctionParser(
        startTag: "<|tool_call>", endTag: "<tool_call|>", escapeMarker: "<|\"|>")

    /// Provenance for every call promoted by recovery, in source order.
    private(set) var events: [ToolCallRecoveryEvent] = []

    private var buffer = ""
    private var context: Context = .response
    private var previousSourceCharacter: Character?
    private var pendingCandidate: PendingCandidate?

    /// Bytes that can begin a signal or context and therefore disqualify a
    /// chunk from the pass-through fast path.
    private static let interestingBytes: Set<UInt8> = [
        0x3C,  // <
        0x5B,  // [
        0x7B,  // {
        0x22,  // "
        0x60,  // `
        0x7E,  // ~
    ]

    init?(
        primaryFormat: ToolCallFormat,
        policy: ToolCallRecoveryPolicy,
        tools: [[String: any Sendable]]?,
        allowedToolNames: Set<String>?,
        maximumBufferedByteCount: Int = 65_536
    ) {
        guard let tools, let allowedToolNames, !allowedToolNames.isEmpty else { return nil }

        self.primaryFormat = primaryFormat
        self.policy = policy
        self.tools = tools
        self.allowedToolNames = allowedToolNames
        self.allowedToolNamesByLength = allowedToolNames.sorted { $0.count > $1.count }
        self.maximumBufferedByteCount = maximumBufferedByteCount

        var signals: [Signal] = []
        if Self.primaryOwnsToolCallFrame(primaryFormat) {
            signals.append(
                Signal(
                    text: "<tool_call>",
                    kind: .nativeFrame(endMarker: "</tool_call>")))
        } else if policy != .disabled {
            signals.append(Signal(text: "<tool_call>", kind: .explicit(.toolCallFrame)))
        }
        if primaryFormat == .gemma4 {
            signals.append(
                Signal(
                    text: "<|tool_call>",
                    kind: .nativeFrame(endMarker: "<tool_call|>")))
        } else if policy != .disabled {
            signals.append(Signal(text: "<|tool_call>", kind: .explicit(.gemma4)))
        }
        if policy != .disabled {
            signals.append(Signal(text: "<function=", kind: .explicit(.qwenFunction)))
        }
        // Reasoning spans are never eligible for recovery, regardless of the
        // selected dialect. Ordered so longer markers win ties.
        signals.append(Signal(text: "<thinking>", kind: .reasoning(close: "</thinking>")))
        signals.append(Signal(text: "<think>", kind: .reasoning(close: "</think>")))
        signals.append(Signal(text: "[THINK]", kind: .reasoning(close: "[/THINK]")))
        if primaryFormat != .mistral, policy != .disabled {
            signals.append(Signal(text: "[TOOL_CALLS]", kind: .explicit(.mistral)))
        }
        if primaryFormat == .llama3 {
            signals.append(Signal(text: "<|python_tag|>", kind: .nativeInlineJSON))
        }

        let primaryParser = primaryFormat.createParser()
        if let start = primaryParser.startTag,
            !signals.contains(where: { $0.text == start })
        {
            if let end = primaryParser.endTag {
                signals.append(Signal(text: start, kind: .nativeFrame(endMarker: end)))
            } else {
                signals.append(Signal(text: start, kind: .nativeUntilEOS))
            }
        }

        // Ordinary JSON values and Markdown code are data: protocol-shaped
        // text inside them must never be promoted. Single-character signals
        // are validated structurally when matched, so prose punctuation that
        // cannot begin a JSON value or code span falls through as text.
        signals.append(Signal(text: "{", kind: .opaqueJSONValue))
        signals.append(Signal(text: "[", kind: .opaqueJSONValue))
        signals.append(Signal(text: "\"", kind: .opaqueJSONValue))
        signals.append(Signal(text: "`", kind: .codeBacktick))
        signals.append(Signal(text: "~", kind: .codeTilde))

        self.signals = signals

        // Prefix lookup replaces an O(tool count) suffix search on every text
        // fragment. Declared names are consulted only after `[ARGS]` is present.
        var prefixes: Set<String> = []
        for signal in signals {
            for length in 1 ..< signal.text.count {
                prefixes.insert(String(signal.text.prefix(length)))
            }
        }
        if policy != .disabled {
            for name in allowedToolNames {
                let marker = name + "[ARGS]"
                for length in 1 ..< marker.count {
                    prefixes.insert(String(marker.prefix(length)))
                }
            }
        }
        self.potentialSignalPrefixes = prefixes
        self.maximumPotentialSignalPrefixLength = prefixes.map(\.count).max() ?? 0
        self.potentialPrefixEndCharacters = Set(prefixes.compactMap(\.last))
    }

    /// Removes and returns the provenance recorded since the last drain.
    mutating func drainEvents() -> [ToolCallRecoveryEvent] {
        let drained = events
        events.removeAll(keepingCapacity: true)
        return drained
    }

    /// Consume a streaming fragment and release all text that cannot still be
    /// the prefix of a recoverable call.
    mutating func consumeIfPassThrough(_ chunk: String) -> Bool {
        guard buffer.isEmpty, context == .response,
            !chunk.utf8.contains(where: { Self.interestingBytes.contains($0) })
        else { return false }
        guard longestRetainedSuffix(in: chunk) == 0 else { return false }
        if let last = chunk.last { previousSourceCharacter = last }
        return true
    }

    mutating func process(_ chunk: String) -> [Output] {
        if let pendingCandidate {
            buffer += chunk
            if buffer.utf8.count <= maximumBufferedByteCount,
                !pendingCandidate.shouldInspect(chunk)
            {
                return []
            }
            self.pendingCandidate = nil
        } else {
            if buffer.isEmpty, context == .response,
                !chunk.utf8.contains(where: { Self.interestingBytes.contains($0) })
            {
                let retained = longestRetainedSuffix(in: chunk)
                guard retained > 0 else {
                    if let last = chunk.last { previousSourceCharacter = last }
                    return chunk.isEmpty ? [] : [.text(chunk)]
                }

                let split = chunk.index(chunk.endIndex, offsetBy: -retained)
                buffer = String(chunk[split...])
                let released = String(chunk[..<split])
                if let last = released.last { previousSourceCharacter = last }
                return released.isEmpty ? [] : [.text(released)]
            }

            buffer += chunk
        }
        var output: [Output] = []

        scanLoop: while !buffer.isEmpty {
            switch context {
            case .reasoning(let close):
                if let range = buffer.range(of: close) {
                    appendProtectedText(String(buffer[..<range.upperBound]), to: &output)
                    buffer = String(buffer[range.upperBound...])
                    context = .response
                    continue
                }
                releaseProtectedAllButSuffix(
                    longestSuffix(in: buffer, matchingPrefixOf: close), &output)
                break scanLoop

            case .codeFence(let character, let count):
                if let close = closingFenceEnd(
                    in: buffer, character: character, minimumCount: count)
                {
                    appendProtectedText(String(buffer[..<close]), to: &output)
                    buffer = String(buffer[close...])
                    context = .response
                    continue
                }
                releaseProtectedAllButSuffix(
                    closingFenceRetention(in: buffer, character: character), &output)
                break scanLoop

            case .codeSpan(let count):
                if let close = spanCloseEnd(in: buffer, count: count) {
                    appendProtectedText(String(buffer[..<close]), to: &output)
                    buffer = String(buffer[close...])
                    context = .response
                    continue
                }
                releaseProtectedAllButSuffix(
                    trailingRunLength(in: buffer, character: "`"), &output)
                break scanLoop

            case .jsonValue:
                if buffer.utf8.count <= maximumBufferedByteCount,
                    !chunkMayCompleteJSONValue(chunk, opener: buffer.first)
                {
                    break scanLoop
                }
                switch jsonValueExtent(in: buffer) {
                case .complete(let end):
                    let value = String(buffer[..<end])
                    if value.utf8.count > maximumBufferedByteCount {
                        appendProtectedText(value, to: &output)
                    } else {
                        appendJSONData(value, to: &output)
                    }
                    buffer = String(buffer[end...])
                    context = .response
                    continue
                case .invalid:
                    // Not a JSON value after all; release the opener as text.
                    appendText(String(buffer[buffer.startIndex ... buffer.startIndex]), to: &output)
                    buffer.removeFirst()
                    context = .response
                    continue
                case .needMore:
                    if buffer.utf8.count > maximumBufferedByteCount {
                        appendProtectedText(buffer, to: &output)
                        buffer.removeAll(keepingCapacity: true)
                        context = .opaqueUntilEOS
                    }
                    break scanLoop
                }

            case .opaqueUntilEOS:
                appendProtectedText(buffer, to: &output)
                buffer.removeAll(keepingCapacity: true)
                break scanLoop

            case .nativeFrame(let endMarker):
                let frameEnd =
                    endMarker == ToolCallFrameScanner.endTag
                    ? ToolCallFrameScanner.frameEnd(in: buffer)
                    : buffer.range(of: endMarker).map(\.upperBound)
                if let frameEnd {
                    let raw = String(buffer[..<frameEnd])
                    buffer = String(buffer[frameEnd...])
                    context = .response
                    if raw.utf8.count > maximumBufferedByteCount {
                        previousSourceCharacter = raw.last
                        output.append(
                            .rejected(
                                rawText: raw, reason: .resourceLimitExceeded,
                                toolName: nil))
                    } else {
                        appendText(raw, to: &output)
                    }
                    continue
                }
                if buffer.utf8.count > maximumBufferedByteCount {
                    let raw = buffer
                    buffer.removeAll(keepingCapacity: true)
                    previousSourceCharacter = raw.last
                    output.append(
                        .rejected(
                            rawText: raw, reason: .resourceLimitExceeded,
                            toolName: nil))
                    context = .opaqueUntilEOS
                }
                break scanLoop

            case .nativeUntilEOS:
                if buffer.utf8.count > maximumBufferedByteCount {
                    let raw = buffer
                    buffer.removeAll(keepingCapacity: true)
                    previousSourceCharacter = raw.last
                    output.append(
                        .rejected(
                            rawText: raw, reason: .resourceLimitExceeded,
                            toolName: nil))
                    context = .opaqueUntilEOS
                }
                break scanLoop

            case .response:
                // A generic one-character construct (`[`, `{`, `"`, or a code
                // delimiter) must not win while that suffix may still grow
                // into a longer protocol marker split across token chunks.
                let retained = longestRetainedSuffix(in: buffer)
                if retained > 0 {
                    let retainedStart = buffer.index(buffer.endIndex, offsetBy: -retained)
                    if let match = earliestConstruct(in: buffer),
                        match.range.lowerBound >= retainedStart
                    {
                        if retainedStart != buffer.startIndex {
                            appendText(String(buffer[..<retainedStart]), to: &output)
                            buffer = String(buffer[retainedStart...])
                        }
                        break scanLoop
                    }
                }

                guard let match = earliestConstruct(in: buffer) else {
                    releaseAllButSuffix(longestRetainedSuffix(in: buffer), &output)
                    break scanLoop
                }

                if match.range.lowerBound != buffer.startIndex {
                    appendText(String(buffer[..<match.range.lowerBound]), to: &output)
                    buffer = String(buffer[match.range.lowerBound...])
                    continue
                }

                switch match.signal.kind {
                case .reasoning(let close):
                    appendProtectedText(match.signal.text, to: &output)
                    buffer.removeFirst(match.signal.text.count)
                    context = .reasoning(close: close)

                case .nativeFrame(let endMarker):
                    // Keep the whole frame buffered; it is emitted verbatim
                    // once its structural close arrives.
                    context = .nativeFrame(endMarker: endMarker)

                case .nativeUntilEOS:
                    context = .nativeUntilEOS

                case .nativeInlineJSON:
                    guard let end = nativeInlineJSONEnd(in: buffer) else {
                        if buffer.utf8.count > maximumBufferedByteCount {
                            let raw = buffer
                            buffer.removeAll(keepingCapacity: true)
                            previousSourceCharacter = raw.last
                            output.append(
                                .rejected(
                                    rawText: raw, reason: .resourceLimitExceeded,
                                    toolName: nil))
                            context = .opaqueUntilEOS
                        }
                        break scanLoop
                    }
                    let raw = String(buffer[..<end])
                    buffer = String(buffer[end...])
                    if raw.utf8.count > maximumBufferedByteCount {
                        previousSourceCharacter = raw.last
                        output.append(
                            .rejected(
                                rawText: raw, reason: .resourceLimitExceeded,
                                toolName: nil))
                    } else {
                        appendText(raw, to: &output)
                    }

                case .opaqueJSONValue:
                    context = .jsonValue

                case .codeBacktick:
                    let run = backtickRun(in: buffer)
                    guard run.terminated else { break scanLoop }
                    let isFence = run.count >= 3 && isLineStart(buffer.startIndex, in: buffer)
                    appendProtectedText(String(buffer[..<run.end]), to: &output)
                    buffer = String(buffer[run.end...])
                    context =
                        isFence
                        ? .codeFence(character: "`", count: run.count)
                        : .codeSpan(count: run.count)

                case .codeTilde:
                    let run = tildeRun(in: buffer)
                    guard run.terminated else { break scanLoop }
                    if run.count >= 3, isLineStart(buffer.startIndex, in: buffer) {
                        appendProtectedText(String(buffer[..<run.end]), to: &output)
                        buffer = String(buffer[run.end...])
                        context = .codeFence(character: "~", count: run.count)
                    } else {
                        appendText(String(buffer[..<run.end]), to: &output)
                        buffer = String(buffer[run.end...])
                    }

                case .explicit(let explicitKind):
                    switch candidateExtent(for: explicitKind, in: buffer) {
                    case .complete(let end):
                        let raw = String(buffer[..<end])
                        buffer = String(buffer[end...])
                        previousSourceCharacter = raw.last
                        if raw.utf8.count > maximumBufferedByteCount {
                            output.append(
                                .rejected(
                                    rawText: raw, reason: .resourceLimitExceeded,
                                    toolName: extractToolName(from: raw, kind: explicitKind)))
                        } else if let call = recoverExplicit(
                            raw, as: explicitKind, allowMissingOuterEnd: false)
                        {
                            recordEvent(
                                for: call, kind: match.signal.kind,
                                repair: .alternateDialect, wasIncompleteAtEOS: false)
                            output.append(.toolCall(call, rawText: raw))
                        } else {
                            output.append(rejection(for: raw, kind: explicitKind, atEOS: false))
                        }
                    case .needMore:
                        if buffer.utf8.count > maximumBufferedByteCount {
                            let raw = buffer
                            buffer.removeAll(keepingCapacity: true)
                            previousSourceCharacter = raw.last
                            output.append(
                                .rejected(
                                    rawText: raw, reason: .resourceLimitExceeded,
                                    toolName: extractToolName(from: raw, kind: explicitKind)))
                            context = .opaqueUntilEOS
                        } else {
                            pendingCandidate = .explicit(explicitKind)
                            break scanLoop
                        }
                    case .malformed(let consumed):
                        let raw = String(buffer[..<consumed])
                        buffer = String(buffer[consumed...])
                        previousSourceCharacter = raw.last
                        output.append(rejection(for: raw, kind: explicitKind, atEOS: false))
                    }

                case .declaredArgs(let name):
                    switch markerlessCandidateExtent(in: buffer) {
                    case .complete(let end):
                        let raw = String(buffer[..<end])
                        buffer = String(buffer[end...])
                        previousSourceCharacter = raw.last
                        if raw.utf8.count > maximumBufferedByteCount {
                            appendProtectedText(raw, to: &output)
                        } else if let call = recoverDeclaredArgs(raw, name: name) {
                            recordEvent(
                                for: call, kind: match.signal.kind,
                                repair: .alternateDialect, wasIncompleteAtEOS: false)
                            output.append(.toolCall(call, rawText: raw))
                        } else {
                            // Ambiguous markerless rehearsal: remain response text.
                            appendText(raw, to: &output)
                        }
                    case .needMore:
                        if buffer.utf8.count > maximumBufferedByteCount {
                            appendProtectedText(buffer, to: &output)
                            buffer.removeAll(keepingCapacity: true)
                            context = .opaqueUntilEOS
                        } else {
                            pendingCandidate = .declaredArgs(name: name)
                        }
                        break scanLoop
                    }
                }
            }
        }

        return output
    }

    /// Finish the recovery stream.
    ///
    /// A missing outer end marker may be repaired at EOS under the permissive
    /// policy when its inner payload is structurally complete. Under the
    /// conservative policy an explicit but incomplete attempt is rejected.
    /// Unterminated reasoning, code, and JSON-data spans flush as protected
    /// response text. Native frames alone remain the native parser's own
    /// end-of-stream responsibility.
    mutating func finish() -> [Output] {
        var output = process("")
        guard !buffer.isEmpty else {
            // `ToolCallProcessor` supports reuse after EOS. A native opening
            // marker delivered immediately before EOS must not leave recovery
            // shielding every subsequent generation.
            context = .response
            return output
        }

        defer {
            buffer.removeAll(keepingCapacity: true)
            context = .response
            pendingCandidate = nil
        }

        guard context == .response,
            let match = earliestConstruct(in: buffer),
            match.range.lowerBound == buffer.startIndex
        else {
            switch context {
            case .reasoning, .codeFence, .codeSpan, .jsonValue, .opaqueUntilEOS:
                appendProtectedText(buffer, to: &output)
            case .response, .nativeFrame, .nativeUntilEOS:
                appendText(buffer, to: &output)
            }
            return output
        }

        switch match.signal.kind {
        case .explicit(let explicitKind):
            let allowMissingOuterEnd = policy == .permissive
            if let call = recoverExplicit(
                buffer, as: explicitKind, allowMissingOuterEnd: allowMissingOuterEnd)
            {
                recordEvent(
                    for: call, kind: match.signal.kind,
                    repair: allowMissingOuterEnd ? .missingOuterClose : .alternateDialect,
                    wasIncompleteAtEOS: true)
                output.append(.toolCall(call, rawText: buffer))
                previousSourceCharacter = buffer.last
            } else {
                previousSourceCharacter = buffer.last
                output.append(rejection(for: buffer, kind: explicitKind, atEOS: true))
            }
        case .declaredArgs(let name):
            if let call = recoverDeclaredArgs(buffer, name: name) {
                recordEvent(
                    for: call, kind: match.signal.kind,
                    repair: .alternateDialect, wasIncompleteAtEOS: true)
                output.append(.toolCall(call, rawText: buffer))
                previousSourceCharacter = buffer.last
            } else {
                appendText(buffer, to: &output)
            }
        default:
            appendText(buffer, to: &output)
        }
        return output
    }

    /// Retry a complete payload rejected by the selected native parser.
    mutating func recoverCompletePayload(_ raw: String) -> ToolCall? {
        recoverCompletePayload(raw, atEOS: false)
    }

    /// Retry each EOS-delimited native segment independently. This primarily
    /// heals Mistral's observed `[TOOL_CALLS]name{json}` variant.
    mutating func recoverEOSPayloads(_ raw: String) -> [ToolCall] {
        guard policy != .disabled else { return [] }
        if raw.contains("[TOOL_CALLS]") {
            return raw.components(separatedBy: "[TOOL_CALLS]").compactMap { segment in
                guard !segment.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                    return nil
                }
                let recovered = recoverExplicit(
                    "[TOOL_CALLS]" + segment, as: .mistral,
                    allowMissingOuterEnd: policy == .permissive)
                if let recovered {
                    recordEvent(
                        for: recovered, kind: .explicit(.mistral),
                        repair: .alternateDialect, wasIncompleteAtEOS: true)
                }
                return recovered
            }
        }
        return recoverCompletePayload(raw, atEOS: true).map { [$0] } ?? []
    }

    private mutating func recoverCompletePayload(_ raw: String, atEOS: Bool) -> ToolCall? {
        guard policy != .disabled else { return nil }
        let allowMissingOuterEnd = policy == .permissive
        let candidates: [(String, ExplicitKind)] = [
            ("<tool_call>", .toolCallFrame),
            ("<|tool_call>", .gemma4),
            ("<function=", .qwenFunction),
            ("[TOOL_CALLS]", .mistral),
        ]
        for (marker, kind) in candidates where raw.contains(marker) {
            guard
                let call = recoverExplicit(
                    raw, as: kind, allowMissingOuterEnd: allowMissingOuterEnd)
            else { continue }
            recordEvent(
                for: call, kind: .explicit(kind),
                repair: .alternateDialect, wasIncompleteAtEOS: atEOS)
            return call
        }
        return nil
    }

    // MARK: - Construct scanning

    private func earliestConstruct(in text: String) -> (signal: Signal, range: Range<String.Index>)?
    {
        var earliest: (signal: Signal, range: Range<String.Index>)?
        for signal in signals {
            guard let range = text.range(of: signal.text) else { continue }
            if earliest == nil || range.lowerBound < earliest!.range.lowerBound {
                earliest = (signal, range)
            }
        }

        guard policy != .disabled else { return earliest }

        var argsSearchStart = text.startIndex
        while argsSearchStart < text.endIndex,
            let args = text.range(of: "[ARGS]", range: argsSearchStart ..< text.endIndex)
        {
            let prefix = text[..<args.lowerBound]
            if let name = allowedToolNamesByLength.first(where: { prefix.hasSuffix($0) }) {
                let start = text.index(args.lowerBound, offsetBy: -name.count)
                if text[..<start].hasSuffix("[TOOL_CALLS]") {
                    argsSearchStart = args.upperBound
                    continue
                }
                let predecessor =
                    start == text.startIndex
                    ? previousSourceCharacter
                    : text[text.index(before: start)]
                if predecessor.map(isToolNameContinuation) == true {
                    argsSearchStart = args.upperBound
                    continue
                }
                let range = start ..< args.upperBound
                if earliest == nil || range.lowerBound < earliest!.range.lowerBound {
                    earliest = (
                        Signal(text: name + "[ARGS]", kind: .declaredArgs(name: name)), range
                    )
                }
                break
            }
            argsSearchStart = args.upperBound
        }
        return earliest
    }

    private func isToolNameContinuation(_ character: Character) -> Bool {
        character.isLetter || character.isNumber
            || character == "_" || character == "." || character == "-"
    }

    private mutating func appendText(_ text: String, to output: inout [Output]) {
        guard !text.isEmpty else { return }
        output.append(.text(text))
        previousSourceCharacter = text.last
    }

    private mutating func appendProtectedText(_ text: String, to output: inout [Output]) {
        guard !text.isEmpty else { return }
        output.append(.protectedText(text))
        previousSourceCharacter = text.last
    }

    /// Bare top-level JSON remains eligible only when JSON is the selected
    /// native tool-call format. For every other dialect it is response data and
    /// must bypass native parsing, including protocol-shaped strings inside it.
    private mutating func appendJSONData(_ text: String, to output: inout [Output]) {
        if primaryFormat == .json, text.first == "{" {
            appendText(text, to: &output)
        } else {
            appendProtectedText(text, to: &output)
        }
    }

    private func chunkMayCompleteJSONValue(_ chunk: String, opener: Character?) -> Bool {
        guard !chunk.isEmpty else { return true }
        if opener == "\"" {
            return chunk.contains("\"") || chunk.contains("\n") || chunk.contains("\r")
        }
        return chunk.contains("}") || chunk.contains("]")
    }

    /// Release everything except a suffix that may still complete a signal or
    /// context boundary when the next token arrives.
    private mutating func releaseAllButSuffix(_ retained: Int, _ output: inout [Output]) {
        let releaseCount = buffer.count - retained
        guard releaseCount > 0 else { return }
        let split = buffer.index(buffer.startIndex, offsetBy: releaseCount)
        appendText(String(buffer[..<split]), to: &output)
        buffer = String(buffer[split...])
    }

    private mutating func releaseProtectedAllButSuffix(
        _ retained: Int, _ output: inout [Output]
    ) {
        let releaseCount = buffer.count - retained
        guard releaseCount > 0 else { return }
        let split = buffer.index(buffer.startIndex, offsetBy: releaseCount)
        appendProtectedText(String(buffer[..<split]), to: &output)
        buffer = String(buffer[split...])
    }

    private func longestRetainedSuffix(in text: String) -> Int {
        max(
            longestPotentialSignalSuffix(in: text),
            trailingRunLength(in: text, character: "`"),
            trailingRunLength(in: text, character: "~"))
    }

    /// Keep only a suffix that might become a signal when the next token arrives.
    private func longestPotentialSignalSuffix(in text: String) -> Int {
        guard let last = text.last, potentialPrefixEndCharacters.contains(last) else { return 0 }
        let upperBound = min(text.count, maximumPotentialSignalPrefixLength)
        guard upperBound > 0 else { return 0 }
        for length in stride(from: upperBound, through: 1, by: -1) {
            if potentialSignalPrefixes.contains(String(text.suffix(length))) {
                return length
            }
        }
        return 0
    }

    private func longestSuffix(in text: String, matchingPrefixOf marker: String) -> Int {
        let upperBound = min(text.count, marker.count - 1)
        guard upperBound > 0 else { return 0 }
        for length in stride(from: upperBound, through: 1, by: -1) {
            if text.suffix(length) == marker.prefix(length) { return length }
        }
        return 0
    }

    private func trailingRunLength(in text: String, character: Character) -> Int {
        var count = 0
        var index = text.endIndex
        while index > text.startIndex {
            let previous = text.index(before: index)
            guard text[previous] == character else { break }
            count += 1
            index = previous
        }
        return count
    }

    // MARK: - Code contexts

    private func backtickRun(in text: String) -> (count: Int, end: String.Index, terminated: Bool) {
        runLength(of: "`", in: text)
    }

    private func tildeRun(in text: String) -> (count: Int, end: String.Index, terminated: Bool) {
        runLength(of: "~", in: text)
    }

    /// Length of the marker run at the start of `text`. A run that reaches the
    /// end of the buffer is not terminated: the next chunk may extend it, and
    /// CommonMark open/close semantics depend on the exact run length.
    private func runLength(
        of character: Character, in text: String
    ) -> (count: Int, end: String.Index, terminated: Bool) {
        var index = text.startIndex
        while index < text.endIndex, text[index] == character {
            index = text.index(after: index)
        }
        return (text.distance(from: text.startIndex, to: index), index, index < text.endIndex)
    }

    /// Whether `index` begins a Markdown line: at most three spaces since the
    /// last newline (or since the start of the stream across chunks).
    private func isLineStart(_ index: String.Index, in text: String) -> Bool {
        var spaces = 0
        var current = index
        while current > text.startIndex {
            let previous = text.index(before: current)
            let character = text[previous]
            if character == "\n" { return spaces <= 3 }
            guard character == " " else { return false }
            spaces += 1
            current = previous
        }
        guard spaces <= 3 else { return false }
        guard let previous = previousSourceCharacter else { return true }
        // A preceding newline starts a line; preceding spaces may continue the
        // indentation of one (over-gating into code is the safe direction).
        return previous == "\n" || previous == " "
    }

    /// The end of a closing fence: a line starting with a run of `character`
    /// at least as long as the opening run.
    private func closingFenceEnd(
        in text: String, character: Character, minimumCount: Int
    ) -> String.Index? {
        var lineStart = text.startIndex
        while lineStart < text.endIndex {
            if isLineStart(lineStart, in: text) {
                var index = lineStart
                var spaces = 0
                while index < text.endIndex, text[index] == " " {
                    spaces += 1
                    index = text.index(after: index)
                }
                if spaces <= 3, index < text.endIndex, text[index] == character {
                    var runEnd = index
                    while runEnd < text.endIndex, text[runEnd] == character {
                        runEnd = text.index(after: runEnd)
                    }
                    let count = text.distance(from: index, to: runEnd)
                    if count >= minimumCount {
                        // A run touching the buffer end may still grow; wait.
                        guard runEnd < text.endIndex else { return nil }
                        return runEnd
                    }
                }
            }
            guard let newline = text.range(of: "\n", range: lineStart ..< text.endIndex)
            else { return nil }
            lineStart = newline.upperBound
        }
        return nil
    }

    /// Retention for a suffix that may become a closing fence line: optional
    /// indentation followed by a (possibly still growing) run of the fence
    /// character at a line start.
    private func closingFenceRetention(in text: String, character: Character) -> Int {
        let lineStart: String.Index
        if let newline = text.range(of: "\n", options: .backwards) {
            lineStart = newline.upperBound
        } else {
            guard isLineStart(text.startIndex, in: text) else { return 0 }
            lineStart = text.startIndex
        }
        var index = lineStart
        var spaces = 0
        while index < text.endIndex, text[index] == " " {
            spaces += 1
            index = text.index(after: index)
        }
        guard spaces <= 3 else { return 0 }
        while index < text.endIndex, text[index] == character {
            index = text.index(after: index)
        }
        guard index == text.endIndex else { return 0 }
        return text.distance(from: lineStart, to: text.endIndex)
    }

    /// The end of the backtick run that closes an inline code span. CommonMark
    /// requires the closing run to have exactly the opening run's length.
    private func spanCloseEnd(in text: String, count: Int) -> String.Index? {
        var index = text.startIndex
        while index < text.endIndex {
            guard text[index] == "`" else {
                index = text.index(after: index)
                continue
            }
            var runEnd = index
            while runEnd < text.endIndex, text[runEnd] == "`" {
                runEnd = text.index(after: runEnd)
            }
            // A run touching the buffer end may still grow; wait.
            guard runEnd < text.endIndex else { return nil }
            let length = text.distance(from: index, to: runEnd)
            if length == count { return runEnd }
            index = runEnd
        }
        return nil
    }

    // MARK: - JSON data opacity

    private enum ValueExtent {
        case complete(end: String.Index)
        case needMore
        case invalid
    }

    /// The extent of the JSON value starting at the beginning of `text`.
    /// Objects, arrays and strings are opaque data; numbers, booleans and null
    /// contain no interior text and need no shielding.
    private func jsonValueExtent(in text: String) -> ValueExtent {
        switch text[text.startIndex] {
        case "{", "[":
            return jsonContainerExtent(in: text)
        case "\"":
            return jsonStringExtent(in: text)
        default:
            return .invalid
        }
    }

    private func jsonContainerExtent(in text: String) -> ValueExtent {
        let start = text.startIndex
        var index = text.index(after: start)
        while index < text.endIndex, text[index].isWhitespace {
            index = text.index(after: index)
        }
        guard index < text.endIndex else { return .needMore }

        // Validate the first meaningful character so prose such as `[TOOL...`
        // or `{not json` is never treated as structured data.
        let first = text[index]
        let plausible: Bool
        if text[start] == "{" {
            plausible = first == "\"" || first == "}"
        } else {
            plausible =
                first == "\"" || first == "{" || first == "[" || first == "]"
                || first == "-" || first.isNumber
                || first == "t" || first == "f" || first == "n"
        }
        guard plausible else { return .invalid }

        var depth = 0
        var inString = false
        var isEscaped = false
        var scan = start
        while scan < text.endIndex {
            let character = text[scan]
            if inString {
                if isEscaped {
                    isEscaped = false
                } else if character == "\\" {
                    isEscaped = true
                } else if character == "\"" {
                    inString = false
                }
            } else {
                switch character {
                case "\"":
                    inString = true
                case "{", "[":
                    depth += 1
                case "}", "]":
                    depth -= 1
                    if depth == 0 {
                        return .complete(end: text.index(after: scan))
                    }
                default:
                    break
                }
            }
            scan = text.index(after: scan)
        }
        return .needMore
    }

    private func jsonStringExtent(in text: String) -> ValueExtent {
        var index = text.index(after: text.startIndex)
        var isEscaped = false
        while index < text.endIndex {
            let character = text[index]
            if isEscaped {
                isEscaped = false
            } else if character == "\\" {
                isEscaped = true
            } else if character == "\"" {
                return .complete(end: text.index(after: index))
            } else if character == "\n" || character == "\r" {
                // Valid JSON strings never contain a literal newline, so this
                // quote was prose punctuation rather than structured data.
                return .invalid
            }
            index = text.index(after: index)
        }
        return .needMore
    }

    // MARK: - Candidate extents

    private enum CandidateExtent {
        case complete(end: String.Index)
        case needMore
        /// The attempt is structurally impossible; `consumed` is the examined
        /// extent retained for a rejection diagnostic.
        case malformed(consumed: String.Index)
    }

    private func candidateExtent(for kind: ExplicitKind, in text: String) -> CandidateExtent {
        switch kind {
        case .qwenFunction:
            switch QwenXMLPayloadScanner.scan(text[...]) {
            case .complete(let payload):
                return .complete(end: payload.end)
            case .needMore:
                return .needMore
            case .malformed(let consumed):
                return .malformed(consumed: consumed)
            }

        case .toolCallFrame:
            var payloadStart = text.index(text.startIndex, offsetBy: "<tool_call>".count)
            while payloadStart < text.endIndex, text[payloadStart].isWhitespace {
                payloadStart = text.index(after: payloadStart)
            }
            guard payloadStart < text.endIndex else { return .needMore }

            if text[payloadStart] == "{" {
                // The frame closes only after a structurally balanced top-level
                // JSON payload: a literal `</tool_call>` inside a string
                // argument cannot end the candidate.
                let payloadAndSuffix = String(text[payloadStart...])
                guard let split = jsonScanner.splitLeadingObject(from: payloadAndSuffix) else {
                    return .needMore
                }
                var afterPayload = text.index(payloadStart, offsetBy: split.object.count)
                while afterPayload < text.endIndex, text[afterPayload].isWhitespace {
                    afterPayload = text.index(after: afterPayload)
                }
                guard afterPayload < text.endIndex else { return .needMore }
                if text[afterPayload...].hasPrefix("</tool_call>") {
                    return .complete(
                        end: text.index(afterPayload, offsetBy: "</tool_call>".count))
                }
                if let stray = text.range(
                    of: "</tool_call>", range: afterPayload ..< text.endIndex)
                {
                    return .malformed(consumed: stray.upperBound)
                }
                return .needMore
            }

            if text[payloadStart...].hasPrefix(QwenXMLPayloadScanner.functionOpen) {
                switch QwenXMLPayloadScanner.scan(text[payloadStart...]) {
                case .complete(let payload):
                    var afterPayload = payload.end
                    while afterPayload < text.endIndex, text[afterPayload].isWhitespace {
                        afterPayload = text.index(after: afterPayload)
                    }
                    guard afterPayload < text.endIndex else { return .needMore }
                    if text[afterPayload...].hasPrefix("</tool_call>") {
                        return .complete(
                            end: text.index(afterPayload, offsetBy: "</tool_call>".count))
                    }
                    if let stray = text.range(
                        of: "</tool_call>", range: afterPayload ..< text.endIndex)
                    {
                        return .malformed(consumed: stray.upperBound)
                    }
                    return .needMore
                case .needMore:
                    return .needMore
                case .malformed(let consumed):
                    return .malformed(consumed: consumed)
                }
            }

            // The payload may still be a split `<function=` prefix.
            if QwenXMLPayloadScanner.functionOpen.hasPrefix(String(text[payloadStart...])) {
                return .needMore
            }
            if let close = text.range(
                of: "</tool_call>", range: payloadStart ..< text.endIndex)
            {
                return .malformed(consumed: close.upperBound)
            }
            return .needMore

        case .gemma4:
            guard let brace = gemmaScanner.firstTopLevelIndex(of: "{", in: text[...]) else {
                if let close = text.range(of: "<tool_call|>") {
                    return .malformed(consumed: close.upperBound)
                }
                return .needMore
            }
            guard let braceEnd = gemmaScanner.endOfGroup(in: text[...], openedAt: brace)
            else {
                if let close = text.range(of: "<tool_call|>") {
                    return .malformed(consumed: close.upperBound)
                }
                return .needMore
            }
            guard
                let close = text.range(
                    of: "<tool_call|>", range: braceEnd ..< text.endIndex)
            else { return .needMore }
            return .complete(end: close.upperBound)

        case .mistral:
            guard let brace = text.firstIndex(of: "{") else { return .needMore }
            let tail = String(text[brace...])
            guard let split = jsonScanner.splitLeadingObject(from: tail) else { return .needMore }
            return .complete(end: text.index(brace, offsetBy: split.object.count))
        }
    }

    /// Markerless `name[ARGS]{...}` rehearsals only ever produce text or a
    /// call — never a rejection — so their extent has no malformed case.
    private enum MarkerlessExtent {
        case complete(end: String.Index)
        case needMore
    }

    private func markerlessCandidateExtent(in text: String) -> MarkerlessExtent {
        guard let brace = text.firstIndex(of: "{") else { return .needMore }
        let tail = String(text[brace...])
        guard let split = jsonScanner.splitLeadingObject(from: tail) else { return .needMore }
        return .complete(end: text.index(brace, offsetBy: split.object.count))
    }

    private func nativeInlineJSONEnd(in text: String) -> String.Index? {
        guard let brace = text.firstIndex(of: "{") else { return nil }
        let json = String(text[brace...])
        guard let split = jsonScanner.splitLeadingObject(from: json) else { return nil }
        return text.index(brace, offsetBy: split.object.count)
    }

    // MARK: - Parsing

    private mutating func recordEvent(
        for call: ToolCall,
        kind: Kind,
        repair: ToolCallRecoveryEvent.Repair,
        wasIncompleteAtEOS: Bool
    ) {
        let dialect: ToolCallRecoveryEvent.Dialect
        switch kind {
        case .explicit(.toolCallFrame): dialect = .toolCallFrame
        case .explicit(.gemma4): dialect = .gemma4
        case .explicit(.qwenFunction): dialect = .qwenFunction
        case .explicit(.mistral): dialect = .mistral
        case .declaredArgs: dialect = .declaredArgs
        case .nativeFrame, .nativeUntilEOS, .nativeInlineJSON, .opaqueJSONValue, .reasoning,
            .codeBacktick, .codeTilde:
            return
        }
        events.append(
            ToolCallRecoveryEvent(
                toolName: call.function.name,
                callID: call.id,
                selectedFormat: primaryFormat,
                dialect: dialect,
                repair: repair,
                wasIncompleteAtEOS: wasIncompleteAtEOS))
    }

    private func recoverExplicit(
        _ raw: String,
        as kind: ExplicitKind,
        allowMissingOuterEnd: Bool
    ) -> ToolCall? {
        let call: ToolCall?
        switch kind {
        case .toolCallFrame:
            guard allowMissingOuterEnd || raw.contains("</tool_call>") else { return nil }
            call = framedParser.parse(content: raw, tools: tools)
        case .gemma4:
            guard allowMissingOuterEnd || raw.contains("<tool_call|>") else { return nil }
            call = gemmaParser.parse(content: raw, tools: tools)
        case .qwenFunction:
            call = parseStrictQwenFunction(raw, allowMissingFunctionClose: allowMissingOuterEnd)
        case .mistral:
            call = parseMistralStyle(raw)
        }

        guard let call, allowedToolNames.contains(call.function.name) else { return nil }
        return call
    }

    private func recoverDeclaredArgs(_ raw: String, name: String) -> ToolCall? {
        guard raw.hasPrefix(name + "[ARGS]"),
            let call = parseMistralStyle(raw, expectedName: name)
        else { return nil }
        return call
    }

    /// Parse a bare `<function=...>` candidate with the shared structural
    /// scanner. The candidate must be canonical and fully consumed: every
    /// opened parameter must close, the function close is the structural close
    /// rather than the first textual one, and no arguments are fabricated for
    /// incomplete payloads.
    private func parseStrictQwenFunction(
        _ raw: String,
        allowMissingFunctionClose: Bool
    ) -> ToolCall? {
        QwenXMLPayloadScanner.parseCanonical(
            raw[...],
            tools: tools,
            allowMissingFunctionClose: allowMissingFunctionClose)
    }

    private func rejection(
        for raw: String,
        kind: ExplicitKind,
        atEOS: Bool
    ) -> Output {
        let name = extractToolName(from: raw, kind: kind)
        let reason: RejectedToolCall.Reason
        if atEOS {
            reason = .incompleteOutput
        } else if let name, !allowedToolNames.contains(name) {
            reason = .undeclaredTool
        } else {
            reason = .malformedSyntax
        }
        return .rejected(rawText: raw, reason: reason, toolName: name)
    }

    /// Best-effort function-name extraction for rejection diagnostics.
    private func extractToolName(from raw: String, kind: ExplicitKind) -> String? {
        switch kind {
        case .qwenFunction:
            return extractXMLFunctionName(from: raw)
        case .toolCallFrame:
            let payload = raw.dropFirst("<tool_call>".count)
                .trimmingCharacters(in: .whitespacesAndNewlines)
            if payload.hasPrefix("{"),
                let object = jsonScanner.splitLeadingObject(from: payload)?.object,
                let dictionary = tryParseJSON(object) as? [String: any Sendable]
            {
                return dictionary["name"] as? String
            }
            if payload.hasPrefix(QwenXMLPayloadScanner.functionOpen) {
                return extractXMLFunctionName(from: payload)
            }
            return nil
        case .gemma4:
            guard let callRange = raw.range(of: "call:") else { return nil }
            let remainder = raw[callRange.upperBound...]
            guard let brace = remainder.firstIndex(of: "{") else { return nil }
            let name = remainder[..<brace].trimmingCharacters(in: .whitespacesAndNewlines)
            return name.isEmpty ? nil : name
        case .mistral:
            var text = raw.trimmingCharacters(in: .whitespacesAndNewlines)
            if text.hasPrefix("[TOOL_CALLS]") {
                text.removeFirst("[TOOL_CALLS]".count)
                text = text.trimmingCharacters(in: .whitespacesAndNewlines)
            }
            guard
                let delimiter = text.range(of: "[ARGS]") ?? text.range(of: "{")
                    ?? text.range(of: "[CALL_ID]")
            else {
                return text.isEmpty ? nil : text
            }
            let name = text[..<delimiter.lowerBound]
                .trimmingCharacters(in: .whitespacesAndNewlines)
            return name.isEmpty ? nil : name
        }
    }

    private func extractXMLFunctionName(from text: String) -> String? {
        guard let openEnd = text.range(of: QwenXMLPayloadScanner.functionOpen)?.upperBound,
            let nameEnd = text[openEnd...].firstIndex(of: ">")
        else { return nil }
        let name = text[openEnd ..< nameEnd]
        return name.isEmpty || name.contains(where: \.isWhitespace) ? nil : String(name)
    }

    private func parseMistralStyle(_ raw: String, expectedName: String? = nil) -> ToolCall? {
        var text = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        if text.hasPrefix("[TOOL_CALLS]") {
            text.removeFirst("[TOOL_CALLS]".count)
            text = text.trimmingCharacters(in: .whitespacesAndNewlines)
        }

        guard let brace = text.firstIndex(of: "{") else { return nil }
        var header = String(text[..<brace]).trimmingCharacters(in: .whitespacesAndNewlines)
        let argumentsText = String(text[brace...])
        var callID: String?

        if header.hasSuffix("[ARGS]") {
            header.removeLast("[ARGS]".count)
            header = header.trimmingCharacters(in: .whitespacesAndNewlines)
        }
        if let idMarker = header.range(of: "[CALL_ID]") {
            let parsedID = header[idMarker.upperBound...]
                .trimmingCharacters(in: .whitespacesAndNewlines)
            callID = parsedID.isEmpty ? nil : String(parsedID)
            header = String(header[..<idMarker.lowerBound])
                .trimmingCharacters(in: .whitespacesAndNewlines)
        }

        guard !header.isEmpty, expectedName.map({ $0 == header }) ?? true,
            allowedToolNames.contains(header),
            let arguments = tryParseJSON(argumentsText) as? [String: any Sendable]
        else { return nil }

        return ToolCall(
            function: .init(name: header, arguments: arguments),
            id: callID)
    }

    private static func primaryOwnsToolCallFrame(_ format: ToolCallFormat) -> Bool {
        switch format {
        case .json, .xmlFunction, .qwen35, .gptOSS:
            true
        case .lfm2, .glm4, .gemma, .gemma4, .kimiK2, .minimaxM2, .atem, .mistral,
            .llama3:
            false
        }
    }
}
