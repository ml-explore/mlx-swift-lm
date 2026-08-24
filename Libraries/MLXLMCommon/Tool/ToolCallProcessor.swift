// Copyright © 2025 Apple Inc.

import Foundation

/// Processes generated text to detect and extract tool calls during streaming generation.
///
/// `ToolCallProcessor` handles the streaming detection of tool calls in model output,
/// buffering partial content and extracting complete tool calls when detected.
///
/// Example:
/// ```swift
/// let processor = ToolCallProcessor(format: .lfm2)
/// for chunk in generatedChunks {
///     if let text = processor.processChunk(chunk) {
///         // Regular text to display
///         print(text)
///     }
/// }
/// // After generation completes:
/// for toolCall in processor.toolCalls {
///     // Handle extracted tool calls
///     print(toolCall.function.name)
/// }
/// ```
public class ToolCallProcessor {

    /// An ordered item emitted while processing generated output.
    public enum Output: Sendable, Equatable {
        case response(String)
        case toolCall(ToolCall)
        case rejectedToolCall(RejectedToolCall)
    }

    // MARK: - Properties

    private let format: ToolCallFormat
    private let parser: any ToolCallParser
    private let tools: [[String: any Sendable]]?
    private let allowedToolNames: Set<String>?
    private let supportsBareJSONFallback: Bool
    private let maxJSONFallbackBufferLength = 32_768
    private let jsonObjectScanner = JSONLeadingObjectScanner(startCharacter: "{")
    private var state = State.normal
    private var toolCallBuffer = ""
    private var hasExplicitInlineMarker = false
    private var emittedToolCallIDs: Set<String> = []
    private var orderedOutputQueue: [Output] = []
    private var orderedOutputEnabled = false

    /// The tool calls extracted during processing.
    public var toolCalls: [ToolCall] = []

    /// Tool-call-shaped outputs that were parsed incompletely, were malformed,
    /// or failed authorization.
    public private(set) var rejectedToolCalls: [RejectedToolCall] = []

    /// Total rejected calls observed by this processor, including drained calls.
    public private(set) var rejectedToolCallCount = 0

    // MARK: - State Enum

    private enum State {
        case normal
        case potentialToolCall
        case collectingToolCall
        case collectingJSONToolCall
    }

    private enum TaggedStartMode {
        case none
        case tagged
        case bareJSON
    }

    // MARK: - Initialization

    /// Initialize with a specific tool call format.
    /// - Parameters:
    ///   - format: The tool call format to use (defaults to `.json` for standard JSON format)
    ///   - tools: Optional tool schemas for type-aware parsing and authorization.
    ///     `nil` accepts any parsed function name; a supplied array, including
    ///     an empty one, authorizes only the names it declares.
    public init(format: ToolCallFormat = .json, tools: [[String: any Sendable]]? = nil) {
        self.format = format
        self.parser = format.createParser()
        self.tools = tools
        self.allowedToolNames = tools.map { tools in
            Set(
                tools.compactMap { tool in
                    (tool["function"] as? [String: any Sendable])?["name"] as? String
                })
        }
        self.supportsBareJSONFallback = format == .json
    }

    // MARK: - Computed Properties

    /// Whether this processor uses inline format (no start tag).
    private var isInlineFormat: Bool {
        parser.startTag == nil
    }

    /// The first character of the start tag for quick detection.
    private var startTagFirstChar: Character? {
        parser.startTag?.first
    }

    // MARK: - Public Methods

    /// Process a generated text chunk and extract any tool call content.
    /// - Parameter chunk: The text chunk to process
    /// - Returns: Regular text that should be displayed (non-tool call content), or `nil` if buffering
    public func processChunk(_ chunk: String) -> String? {
        if isInlineFormat {
            return processInlineChunk(chunk)
        }
        return processTaggedChunk(chunk)
    }

    /// Processes a generated chunk and removes its output in source order.
    ///
    /// Tool protocol syntax that does not parse as a call is not emitted as a
    /// response. Use this streaming operation when tool calls and response text
    /// must retain their relative order. Do not mix this API with `processChunk`,
    /// `processEOS`, or `drainToolCalls()` on the same processor instance.
    public func processChunkOutputs(_ chunk: String) -> [Output] {
        orderedOutputEnabled = true
        let outputCount = orderedOutputQueue.count
        let visible = processChunk(chunk)
        if orderedOutputQueue.count == outputCount, let visible {
            recordResponse(sanitizingProtocol: visible)
        }
        _ = drainToolCalls()
        _ = drainRejectedToolCalls()
        return drainOrderedOutputs()
    }

    /// Removes and returns every parsed call in parse order.
    /// A second call returns an empty array until more chunks are processed.
    public func drainToolCalls() -> [ToolCall] {
        guard !toolCalls.isEmpty else { return [] }
        let drained = toolCalls
        toolCalls.removeAll(keepingCapacity: true)
        return drained
    }

    /// Removes and returns every rejected call in source order.
    /// A second call returns an empty array until more rejections are observed.
    public func drainRejectedToolCalls() -> [RejectedToolCall] {
        guard !rejectedToolCalls.isEmpty else { return [] }
        let drained = rejectedToolCalls
        rejectedToolCalls.removeAll(keepingCapacity: true)
        return drained
    }

    /// Process end-of-sequence, parsing any buffered content as tool call(s).
    ///
    /// Call this when generation ends (e.g., on EOS token) to handle formats
    /// whose end tag is never delivered as text (e.g., Mistral where `</s>`
    /// is intercepted at the token ID level).
    ///
    /// For formats with end tags that appear in the text stream, the buffer
    /// will already be empty at generation end, making this a no-op.
    public func processEOS() {
        _ = processEOS(returnBufferedText: false)
    }

    /// Process end-of-sequence and optionally return residual buffered text.
    ///
    /// Use this overload when callers need to preserve non-tool trailing content
    /// that remained buffered until generation end.
    ///
    /// - Parameter returnBufferedText: When `true`, returns residual text if no
    ///   tool call was parsed from the buffered content.
    /// - Returns: Residual buffered text that should be emitted as regular output,
    ///   or `nil` when the buffer was fully parsed as tool call content (or when
    ///   `returnBufferedText` is `false`).
    @discardableResult
    public func processEOS(returnBufferedText: Bool = true) -> String? {
        guard
            state == .collectingToolCall || state == .potentialToolCall
                || state == .collectingJSONToolCall
        else { return nil }
        guard !toolCallBuffer.isEmpty else {
            state = .normal
            hasExplicitInlineMarker = false
            return nil
        }

        let buffered = toolCallBuffer
        let terminalState = state
        let parsedCalls = parser.parseEOS(buffered, tools: tools)
        appendToolCalls(parsedCalls, rawText: buffered)

        let didReject: Bool
        if parsedCalls.isEmpty,
            let reason = rejectionReasonForResidual(
                buffered, state: terminalState, explicitInlineMarker: hasExplicitInlineMarker)
        {
            appendRejectedToolCall(
                reason: reason,
                rawText: buffered,
                detail: reason.diagnosticDetail)
            didReject = true
        } else {
            didReject = false
        }

        toolCallBuffer = ""
        state = .normal
        hasExplicitInlineMarker = false

        return returnBufferedText && parsedCalls.isEmpty && !didReject ? buffered : nil
    }

    /// Finishes processing and removes residual output in source order.
    ///
    /// This preserves non-tool text following EOS-delimited calls. Do not mix
    /// this API with the legacy processing and draining APIs.
    public func processEOSOutputs() -> [Output] {
        orderedOutputEnabled = true
        if format == .mistral, let outputs = processMistralEOSOutputs() {
            orderedOutputQueue.removeAll(keepingCapacity: true)
            return outputs
        }
        if format == .lfm2, let outputs = processLFM2EOSOutputs() {
            orderedOutputQueue.removeAll(keepingCapacity: true)
            return outputs
        }

        let outputCount = orderedOutputQueue.count
        let visible = processEOS(returnBufferedText: true)
        if orderedOutputQueue.count == outputCount, let visible {
            recordEOSResidual(visible)
        }
        _ = drainToolCalls()
        _ = drainRejectedToolCalls()
        return drainOrderedOutputs()
    }

    // MARK: - Private Methods

    /// Process chunk for inline formats (no wrapper tags).
    ///
    /// Uses quote-aware JSON object scanning to detect when output looks like a JSON tool call.
    /// While the object is incomplete the content is buffered (returns `nil`)
    /// so partial JSON is never leaked to the UI.
    private func processInlineChunk(_ chunk: String) -> String? {
        switch state {
        case .normal:
            // Check if this chunk starts what looks like a JSON tool call
            if let braceIndex = chunk.firstIndex(of: "{") {
                let leading = String(chunk[..<braceIndex])
                let jsonPart = String(chunk[braceIndex...])
                toolCallBuffer = jsonPart
                state = .collectingToolCall
                hasExplicitInlineMarker =
                    hasExplicitInlineMarker || leading.contains("<|python_tag|>")
                let visibleLeading = cleanInlineLeading(leading)

                if let toolCall = parser.parse(content: toolCallBuffer, tools: tools) {
                    recordResponse(visibleLeading)
                    appendToolCall(toolCall, rawText: leading + toolCallBuffer)
                    toolCallBuffer = ""
                    state = .normal
                    hasExplicitInlineMarker = false
                    return visibleLeading.isEmpty ? nil : visibleLeading
                }

                // Still collecting — check if the first JSON object is complete (would mean parse
                // failed on complete JSON, so it's not a tool call)
                if jsonObjectScanner.splitLeadingObject(from: toolCallBuffer) != nil {
                    let buffered = toolCallBuffer
                    recordResponse(visibleLeading)
                    let rejected = rejectInlinePayloadIfNeeded(
                        buffered, explicitMarker: hasExplicitInlineMarker)
                    state = .normal
                    toolCallBuffer = ""
                    hasExplicitInlineMarker = false
                    let response = rejected ? visibleLeading : visibleLeading + buffered
                    if !rejected { recordResponse(sanitizingProtocol: buffered) }
                    return response
                }

                recordResponse(visibleLeading)
                return visibleLeading.isEmpty ? nil : visibleLeading
            }

            // No brace seen — pass through as regular text
            if chunk.contains("<|python_tag|>") {
                hasExplicitInlineMarker = true
            }
            recordResponse(sanitizingProtocol: chunk)
            return chunk

        case .potentialToolCall, .collectingToolCall, .collectingJSONToolCall:
            toolCallBuffer += chunk

            if let toolCall = parser.parse(content: toolCallBuffer, tools: tools) {
                appendToolCall(toolCall, rawText: toolCallBuffer)
                toolCallBuffer = ""
                state = .normal
                hasExplicitInlineMarker = false
                return nil
            }

            // If the object is complete but parse failed, this isn't a tool call — flush
            if jsonObjectScanner.splitLeadingObject(from: toolCallBuffer) != nil {
                let buffered = toolCallBuffer
                let rejected = rejectInlinePayloadIfNeeded(
                    buffered, explicitMarker: hasExplicitInlineMarker)
                state = .normal
                toolCallBuffer = ""
                hasExplicitInlineMarker = false
                guard !rejected else { return nil }
                recordResponse(sanitizingProtocol: buffered)
                return buffered
            }

            // Still collecting
            return nil
        }
    }

    private func recordResponse(_ text: String) {
        guard orderedOutputEnabled, !text.isEmpty else { return }
        orderedOutputQueue.append(.response(text))
    }

    private func recordResponse(sanitizingProtocol text: String) {
        recordResponse(stripProtocolSpans(from: text))
    }

    private func recordEOSResidual(_ text: String) {
        recordResponse(sanitizeEOSResidual(text))
    }

    private func drainOrderedOutputs() -> [Output] {
        let outputs = orderedOutputQueue
        orderedOutputQueue.removeAll(keepingCapacity: true)
        return outputs
    }

    private func stripProtocolSpans(from text: String) -> String {
        var result = text
        let tags =
            [parser.startTag, parser.endTag].compactMap { $0 }
            + (format == .llama3 ? ["<|python_tag|>"] : [])

        for tag in tags {
            while let range = result.range(of: tag) {
                if tag == parser.startTag,
                    let endTag = parser.endTag,
                    let end = result.range(of: endTag, range: range.upperBound ..< result.endIndex)
                {
                    result.removeSubrange(range.lowerBound ..< end.upperBound)
                } else {
                    result.removeSubrange(range)
                }
            }

            guard let first = tag.first else { continue }
            var index = result.startIndex
            while index < result.endIndex {
                guard result[index] == first else {
                    index = result.index(after: index)
                    continue
                }
                let suffix = result[index...]
                let matchCount = zip(suffix, tag).prefix { $0 == $1 }.count
                guard matchCount >= nearCompleteMatchLength(for: tag) else {
                    index = result.index(after: index)
                    continue
                }
                let markerEnd =
                    suffix.firstIndex(of: ">")
                    ?? suffix.firstIndex(of: "]")
                let removalEnd = markerEnd.map { result.index(after: $0) } ?? result.endIndex
                result.removeSubrange(index ..< removalEnd)
            }
        }
        return result
    }

    private func sanitizeEOSResidual(_ text: String) -> String {
        guard let startTag = parser.startTag else {
            return stripProtocolSpans(from: text)
        }

        var searchStart = text.startIndex
        while let startRange = text.range(of: startTag, range: searchStart ..< text.endIndex) {
            guard
                let endTag = parser.endTag,
                let endRange = text.range(
                    of: endTag, range: startRange.upperBound ..< text.endIndex)
            else {
                return stripProtocolSpans(from: String(text[..<startRange.lowerBound]))
            }
            searchStart = endRange.upperBound
        }
        return stripProtocolSpans(from: text)
    }

    private func nearCompleteMatchLength(for tag: String) -> Int {
        max(tag.count - 2, 1)
    }

    private func processMistralEOSOutputs() -> [Output]? {
        guard
            state == .collectingToolCall || state == .potentialToolCall
                || state == .collectingJSONToolCall,
            !toolCallBuffer.isEmpty
        else { return nil }

        let startTag = "[TOOL_CALLS]"
        let argsTag = "[ARGS]"
        var remaining = toolCallBuffer

        while remaining.hasPrefix(startTag) {
            guard let argsRange = remaining.range(of: argsTag) else {
                appendRejectedToolCall(
                    reason: .incompleteOutput,
                    rawText: remaining,
                    detail: RejectedToolCall.Reason.incompleteOutput.diagnosticDetail)
                remaining = ""
                break
            }
            let arguments = String(remaining[argsRange.upperBound...])
            guard let split = jsonObjectScanner.splitLeadingObject(from: arguments) else {
                appendRejectedToolCall(
                    reason: .incompleteOutput,
                    rawText: remaining,
                    detail: RejectedToolCall.Reason.incompleteOutput.diagnosticDetail)
                remaining = ""
                break
            }

            let callText = String(remaining[..<argsRange.upperBound]) + split.object
            if let call = parser.parse(content: callText, tools: tools) {
                appendToolCall(call, rawText: callText)
            } else {
                let reason = classifyCompletePayload(callText)
                appendRejectedToolCall(
                    reason: reason,
                    rawText: callText,
                    detail: reason.diagnosticDetail)
            }
            remaining = split.trailing
        }

        toolCallBuffer = ""
        state = .normal

        if !remaining.isEmpty {
            recordEOSResidualOutputs(remaining)
        }
        _ = drainToolCalls()
        _ = drainRejectedToolCalls()
        return drainOrderedOutputs()
    }

    private func processLFM2EOSOutputs() -> [Output]? {
        guard
            state == .collectingToolCall || state == .potentialToolCall
                || state == .collectingJSONToolCall,
            !toolCallBuffer.isEmpty,
            let startTag = parser.startTag
        else { return nil }

        var remaining = toolCallBuffer

        while let startRange = remaining.range(of: startTag) {
            let responsePrefix = String(remaining[..<startRange.lowerBound])
            let callStart = startRange.upperBound
            recordEOSResidualOutputs(responsePrefix)
            guard let callEnd = balancedBracketEnd(in: remaining, from: callStart) else {
                appendRejectedToolCall(
                    reason: .incompleteOutput,
                    rawText: String(remaining[startRange.lowerBound...]),
                    detail: RejectedToolCall.Reason.incompleteOutput.diagnosticDetail)
                remaining = ""
                break
            }

            let callText = String(remaining[startRange.lowerBound ... callEnd])
            if let call = parser.parse(content: callText, tools: tools) {
                appendToolCall(call, rawText: callText)
            } else {
                let reason = classifyCompletePayload(callText)
                appendRejectedToolCall(
                    reason: reason,
                    rawText: callText,
                    detail: reason.diagnosticDetail)
            }
            remaining = String(remaining[remaining.index(after: callEnd)...])
        }

        toolCallBuffer = ""
        state = .normal

        if !remaining.isEmpty {
            recordEOSResidualOutputs(remaining)
        }
        _ = drainToolCalls()
        _ = drainRejectedToolCalls()
        return drainOrderedOutputs()
    }

    /// End of the bracketed call list beginning at `start`, ignoring brackets
    /// that appear inside quoted argument values.
    private func balancedBracketEnd(in text: String, from start: String.Index) -> String.Index? {
        let tail = text[start...]
        guard let open = Self.listScanner.firstTopLevelIndex(of: "[", in: tail) else { return nil }
        return Self.listScanner.endOfGroup(in: tail, openedAt: open)
    }

    private func recordEOSResidualOutputs(_ text: String) {
        guard let startTag = parser.startTag,
            let attempt = protocolMarkerAttempt(in: text, startTag: startTag),
            let range = text.range(of: attempt)
        else {
            recordResponse(sanitizeEOSResidual(text))
            return
        }

        recordResponse(sanitizeEOSResidual(String(text[..<range.lowerBound])))
        appendRejectedToolCall(
            reason: .malformedSyntax,
            rawText: attempt,
            detail: RejectedToolCall.Reason.malformedSyntax.diagnosticDetail)
        recordEOSResidualOutputs(String(text[range.upperBound...]))
    }

    private static let listScanner = StructuredTextScanner(quotes: ["'", "\""])

    /// Process chunk for tagged formats.
    private func processTaggedChunk(_ chunk: String) -> String? {
        guard let startTag = parser.startTag,
            let startChar = startTagFirstChar
        else {
            return chunk
        }

        let startMode =
            state == .normal
            ? taggedStartMode(in: chunk, startChar: startChar)
            : .none
        guard startMode != .none || state != .normal else {
            recordResponse(chunk)
            return chunk
        }

        toolCallBuffer += chunk
        var leadingToken: String?
        var leadingTokenWasRecorded = false

        switch state {
        case .normal:
            if startMode == .bareJSON {
                // Fallback for models that sporadically emit bare JSON tool calls.
                state = .collectingJSONToolCall

                leadingToken = separateToken(
                    from: &toolCallBuffer,
                    separator: String(jsonObjectScanner.startCharacter),
                    returnLeading: true
                )

                return processCollectingJSONToolCall(
                    startTag: startTag,
                    startChar: startChar,
                    leadingToken: leadingToken
                )
            }

            guard startMode == .tagged else {
                return chunk
            }

            // Change state to potential tagged tool call.
            state = .potentialToolCall

            leadingToken = separateToken(
                from: &toolCallBuffer, separator: String(startChar), returnLeading: true)

            fallthrough

        case .potentialToolCall:
            if partialMatch(buffer: toolCallBuffer, tag: startTag) {
                if toolCallBuffer.starts(with: startTag) {
                    state = .collectingToolCall
                    recordResponse(leadingToken ?? "")
                    leadingTokenWasRecorded = true
                    fallthrough
                } else {
                    recordResponse(leadingToken ?? "")
                    leadingTokenWasRecorded = true
                    return nil
                }
            } else {
                // Otherwise, return the collected text and reset the state.
                state = .normal
                let buffer = toolCallBuffer
                toolCallBuffer = ""
                if let attempt = protocolMarkerAttempt(in: buffer, startTag: startTag) {
                    recordResponse(leadingToken ?? "")
                    appendRejectedToolCall(
                        reason: .malformedSyntax,
                        rawText: attempt,
                        detail: RejectedToolCall.Reason.malformedSyntax.diagnosticDetail)
                    let remainder = buffer.replacingOccurrences(of: attempt, with: "")
                    recordResponse(sanitizingProtocol: remainder)
                    return combine(leadingToken, stripProtocolSpans(from: remainder))
                }
                let response = (leadingToken ?? "") + buffer
                recordResponse(sanitizingProtocol: response)
                return response
            }

        case .collectingToolCall:
            guard let endTag = parser.endTag else {
                return nil
            }

            if toolCallBuffer.contains(endTag) {
                // Separate the trailing token.
                let trailingToken = separateToken(
                    from: &toolCallBuffer, separator: endTag, returnLeading: false)

                let bufferedToolCall = toolCallBuffer

                // Parse the tool call using the parser.
                if let toolCall = parser.parse(content: bufferedToolCall, tools: tools) {
                    if !leadingTokenWasRecorded {
                        recordResponse(leadingToken ?? "")
                    }
                    appendToolCall(toolCall, rawText: bufferedToolCall)
                    state = .normal
                    toolCallBuffer = ""

                    // If trailing content may contain another tool call, recurse.
                    if let trailingToken,
                        tokenCouldContainToolStart(trailingToken, startChar: startChar)
                    {
                        return combine(leadingToken, processChunk(trailingToken))
                    }

                    // Otherwise, return trailing text if non-empty.
                    let trailingText = trailingToken?.isEmpty ?? true ? nil : trailingToken
                    if let trailingText { recordResponse(trailingText) }
                    return combine(leadingToken, trailingText)
                }

                // A complete tagged payload is unambiguously intended as a tool
                // call. Report it rather than leaking protocol text as response.
                state = .normal
                toolCallBuffer = ""
                if !leadingTokenWasRecorded {
                    recordResponse(leadingToken ?? "")
                }
                let reason = classifyCompletePayload(bufferedToolCall)
                appendRejectedToolCall(
                    reason: reason,
                    rawText: bufferedToolCall,
                    detail: reason.diagnosticDetail)
                if let trailingToken,
                    tokenCouldContainToolStart(trailingToken, startChar: startChar)
                {
                    return combine(leadingToken, processChunk(trailingToken))
                }
                if let trailingToken { recordResponse(trailingToken) }
                return combine(leadingToken, trailingToken)
            }

            return nil

        case .collectingJSONToolCall:
            return processCollectingJSONToolCall(
                startTag: startTag,
                startChar: startChar,
                leadingToken: leadingToken
            )
        }
    }

    private func processCollectingJSONToolCall(
        startTag: String,
        startChar: Character,
        leadingToken: String?
    ) -> String? {
        if toolCallBuffer.count > maxJSONFallbackBufferLength {
            // Safety valve: flush pathological unmatched JSON-like buffers as text.
            state = .normal
            let buffered = toolCallBuffer
            toolCallBuffer = ""
            let response = (leadingToken ?? "") + buffered
            recordResponse(sanitizingProtocol: response)
            return response
        }

        switch jsonObjectScanner.evaluatePrefix(in: toolCallBuffer) {
        case .invalidObject:
            state = .normal
            let buffered = toolCallBuffer
            toolCallBuffer = ""
            // vLLM-style recovery: if a tagged tool call exists later, retry tagged parsing.
            if buffered.contains(startTag) {
                recordResponse(leadingToken ?? "")
                return combine(leadingToken, processChunk(buffered))
            }
            let response = (leadingToken ?? "") + buffered
            recordResponse(sanitizingProtocol: response)
            return response
        case .needsMore, .validObject:
            break
        }

        guard let split = jsonObjectScanner.splitLeadingObject(from: toolCallBuffer) else {
            // Continue buffering until a complete top-level JSON object is available.
            recordResponse(leadingToken ?? "")
            return leadingToken?.isEmpty ?? true ? nil : leadingToken
        }

        let jsonCandidate = split.object
        let trailingToken = split.trailing
        recordResponse(leadingToken ?? "")

        if let toolCall = parser.parse(content: jsonCandidate, tools: tools) {
            appendToolCall(toolCall, rawText: jsonCandidate)

            state = .normal
            toolCallBuffer = ""

            if trailingToken.isEmpty {
                return leadingToken?.isEmpty ?? true ? nil : leadingToken
            }

            if tokenCouldContainToolStart(trailingToken, startChar: startChar) {
                return combine(leadingToken, processChunk(trailingToken))
            }

            recordResponse(trailingToken)
            return combine(leadingToken, trailingToken)
        }

        // Bare JSON remains response text unless it carries clear tool-call
        // intent. This avoids classifying ordinary JSON as malformed protocol.
        let rejected = rejectBareJSONPayloadIfNeeded(jsonCandidate)
        state = .normal
        toolCallBuffer = ""
        if tokenCouldContainToolStart(trailingToken, startChar: startChar) {
            if !rejected {
                recordResponse(jsonCandidate)
            }
            return combine(
                leadingToken,
                combine(rejected ? nil : jsonCandidate, processChunk(trailingToken)))
        }
        let response = (rejected ? "" : jsonCandidate) + trailingToken
        recordResponse(sanitizingProtocol: response)
        return combine(leadingToken, response)
    }

    private func taggedStartMode(
        in chunk: String,
        startChar: Character
    ) -> TaggedStartMode {
        let taggedStartIndex = chunk.firstIndex(of: startChar)
        let jsonStartIndex =
            supportsBareJSONFallback
            ? chunk.firstIndex(of: jsonObjectScanner.startCharacter)
            : nil

        switch (taggedStartIndex, jsonStartIndex) {
        case (nil, nil):
            return .none
        case (.some, nil):
            return .tagged
        case (nil, .some):
            return .bareJSON
        case (.some(let tagged), .some(let json)):
            if json >= tagged {
                return .tagged
            }

            // If the earlier `{` cannot begin a JSON object, prefer tagged parsing.
            if case .invalidObject = jsonObjectScanner.evaluatePrefix(in: chunk, from: json) {
                return .tagged
            }

            return .bareJSON
        }
    }

    private func tokenCouldContainToolStart(_ token: String, startChar: Character) -> Bool {
        token.contains(startChar)
            || (supportsBareJSONFallback && token.contains(jsonObjectScanner.startCharacter))
    }

    private func combine(_ first: String?, _ second: String?) -> String? {
        let merged = (first ?? "") + (second ?? "")
        return merged.isEmpty ? nil : merged
    }

    private func appendToolCalls(_ calls: [ToolCall], rawText: String) {
        for call in calls {
            appendToolCall(call, rawText: rawText)
        }
    }

    private func appendToolCall(_ call: ToolCall, rawText: String) {
        guard allowedToolNames?.contains(call.function.name) ?? true else {
            appendRejectedToolCall(
                reason: .undeclaredTool,
                rawText: rawText,
                toolName: call.function.name,
                callID: call.id,
                detail: RejectedToolCall.Reason.undeclaredTool.diagnosticDetail)
            return
        }

        let normalized = normalizedToolCall(call)
        toolCalls.append(normalized)
        if orderedOutputEnabled {
            orderedOutputQueue.append(.toolCall(normalized))
        }
    }

    private func appendRejectedToolCall(
        reason: RejectedToolCall.Reason,
        rawText: String,
        toolName: String? = nil,
        callID: String? = nil,
        detail: String? = nil
    ) {
        let rejection = RejectedToolCall(
            reason: reason,
            format: format,
            toolName: toolName,
            callID: callID,
            rawText: rawText,
            detail: detail)
        rejectedToolCalls.append(rejection)
        rejectedToolCallCount += 1
        if orderedOutputEnabled {
            orderedOutputQueue.append(.rejectedToolCall(rejection))
        }
    }

    private func rejectionReasonForResidual(
        _ text: String,
        state: State,
        explicitInlineMarker: Bool
    ) -> RejectedToolCall.Reason? {
        switch state {
        case .normal:
            return nil
        case .potentialToolCall:
            guard let startTag = parser.startTag,
                protocolMarkerAttempt(in: text, startTag: startTag) != nil
            else { return nil }
            return partialMatch(buffer: text, tag: startTag)
                ? .incompleteOutput : .malformedSyntax
        case .collectingToolCall:
            if isInlineFormat {
                guard explicitInlineMarker || inspectJSONToolIntent(text) != nil else {
                    return nil
                }
                if jsonObjectScanner.splitLeadingObject(from: text) == nil {
                    return .incompleteOutput
                }
                return classifyCompletePayload(text)
            }
            if let endTag = parser.endTag, text.contains(endTag) {
                return classifyCompletePayload(text)
            }
            return .incompleteOutput
        case .collectingJSONToolCall:
            guard inspectJSONToolIntent(text) != nil else { return nil }
            if jsonObjectScanner.splitLeadingObject(from: text) == nil {
                return .incompleteOutput
            }
            return classifyCompletePayload(text)
        }
    }

    private func rejectInlinePayloadIfNeeded(
        _ text: String, explicitMarker: Bool
    ) -> Bool {
        guard explicitMarker || inspectJSONToolIntent(text) != nil else { return false }
        let inspection = inspectJSONToolIntent(text)
        let reason = inspection?.reason ?? classifyCompletePayload(text)
        appendRejectedToolCall(
            reason: reason,
            rawText: text,
            toolName: inspection?.toolName,
            callID: inspection?.callID,
            detail: reason.diagnosticDetail)
        return true
    }

    private func rejectBareJSONPayloadIfNeeded(_ text: String) -> Bool {
        guard let inspection = inspectJSONToolIntent(text) else { return false }
        appendRejectedToolCall(
            reason: inspection.reason,
            rawText: text,
            toolName: inspection.toolName,
            callID: inspection.callID,
            detail: inspection.reason.diagnosticDetail)
        return true
    }

    private func classifyCompletePayload(_ text: String) -> RejectedToolCall.Reason {
        if let inspection = inspectJSONToolIntent(text) {
            return inspection.reason
        }
        if text.range(of: #"<function\s*=\s*>"#, options: .regularExpression) != nil {
            return .missingToolName
        }
        return .malformedSyntax
    }

    private struct JSONToolIntent {
        let reason: RejectedToolCall.Reason
        let toolName: String?
        let callID: String?
    }

    /// Identifies JSON as tool protocol only when it has tool-specific keys.
    /// Ordinary JSON remains response text.
    private func inspectJSONToolIntent(_ rawText: String) -> JSONToolIntent? {
        var text =
            rawText
            .replacingOccurrences(of: "<|python_tag|>", with: "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        if let startTag = parser.startTag, let range = text.range(of: startTag) {
            text = String(text[range.upperBound...])
        }
        if let endTag = parser.endTag, let range = text.range(of: endTag) {
            text = String(text[..<range.lowerBound])
        }
        text = text.trimmingCharacters(in: .whitespacesAndNewlines)

        guard let data = text.data(using: .utf8),
            let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else {
            let hasToolKeys =
                text.contains("\"name\"")
                && (text.contains("\"arguments\"") || text.contains("\"parameters\""))
            return hasToolKeys
                ? JSONToolIntent(reason: .malformedSyntax, toolName: nil, callID: nil)
                : nil
        }

        let call = (root["function"] as? [String: Any]) ?? root
        let name = call["name"] as? String
        let callID = (root["id"] as? String) ?? (call["id"] as? String)
        let argumentValue = call["arguments"] ?? call["parameters"]
        let hasArgumentKey = call.keys.contains("arguments") || call.keys.contains("parameters")

        guard name != nil || hasArgumentKey else { return nil }
        guard let name, !name.isEmpty else {
            return JSONToolIntent(
                reason: .missingToolName, toolName: nil, callID: callID)
        }
        guard hasArgumentKey else {
            // A name by itself is common in ordinary JSON and is not enough to
            // infer a bare tool call. Explicit tagged formats are classified by
            // `classifyCompletePayload` instead.
            return nil
        }

        let argumentsAreValid: Bool
        if argumentValue is [String: Any] {
            argumentsAreValid = true
        } else if let string = argumentValue as? String,
            let data = string.data(using: .utf8),
            (try? JSONSerialization.jsonObject(with: data) as? [String: Any]) != nil
        {
            argumentsAreValid = true
        } else {
            argumentsAreValid = false
        }

        return JSONToolIntent(
            reason: argumentsAreValid ? .malformedSyntax : .invalidArguments,
            toolName: name,
            callID: callID)
    }

    private func protocolMarkerAttempt(in text: String, startTag: String) -> String? {
        guard let first = startTag.first, let start = text.firstIndex(of: first) else {
            return nil
        }
        let suffix = text[start...]
        let matchCount = zip(suffix, startTag).prefix { $0 == $1 }.count
        guard matchCount >= nearCompleteMatchLength(for: startTag) else { return nil }
        let end =
            suffix.firstIndex(of: ">")
            .map { text.index(after: $0) }
            ?? suffix.firstIndex(of: "]").map { text.index(after: $0) }
            ?? text.endIndex
        return String(text[start ..< end])
    }

    private func cleanInlineLeading(_ text: String) -> String {
        text.replacingOccurrences(of: "<|python_tag|>", with: "")
    }

    private func normalizedToolCall(_ call: ToolCall) -> ToolCall {
        if let id = call.id, !id.isEmpty, emittedToolCallIDs.insert(id).inserted {
            return call
        }

        return ToolCall(function: call.function, id: generateToolCallID())
    }

    private func generateToolCallID() -> String {
        while true {
            let id = format.generateToolCallID()
            if emittedToolCallIDs.insert(id).inserted {
                return id
            }
        }
    }

    /// Separates a token from a string buffer based on a separator
    /// - Parameters:
    ///   - buffer: The string buffer to modify
    ///   - separator: The separator string to search for
    ///   - returnLeading: If true, returns text before separator; if false, returns text after
    /// - Returns: The separated token, or nil if separator not found
    private func separateToken(from buffer: inout String, separator: String, returnLeading: Bool)
        -> String?
    {
        guard let range = buffer.range(of: separator) else { return nil }

        let token: String
        if returnLeading {
            token = String(buffer[..<range.lowerBound])
            buffer = String(buffer[range.lowerBound...])
        } else {
            token = String(buffer[range.upperBound...])
            buffer = String(buffer[..<range.upperBound])
        }

        return token
    }

    private func partialMatch(buffer: String, tag: String) -> Bool {
        for (tagIndex, bufferIndex) in zip(tag.indices, buffer.indices) {
            if buffer[bufferIndex] != tag[tagIndex] {
                return false
            }
        }

        return true
    }
}

struct JSONLeadingObjectScanner {
    enum PrefixState {
        case needsMore
        case validObject
        case invalidObject
    }

    let startCharacter: Character

    func evaluatePrefix(in buffer: String) -> PrefixState {
        guard let start = buffer.firstIndex(where: { !$0.isWhitespace }) else {
            return .invalidObject
        }
        return evaluatePrefix(in: buffer, from: start)
    }

    func evaluatePrefix(in buffer: String, from start: String.Index) -> PrefixState {
        var openingIndex = start
        while openingIndex < buffer.endIndex, buffer[openingIndex].isWhitespace {
            openingIndex = buffer.index(after: openingIndex)
        }

        guard openingIndex < buffer.endIndex, buffer[openingIndex] == startCharacter else {
            return .invalidObject
        }

        var index = buffer.index(after: openingIndex)
        while index < buffer.endIndex, buffer[index].isWhitespace {
            index = buffer.index(after: index)
        }

        guard index < buffer.endIndex else {
            return .needsMore
        }

        let firstToken = buffer[index]
        if firstToken == "\"" || firstToken == "}" {
            return .validObject
        }

        return .invalidObject
    }

    /// Splits a buffer that starts with optional whitespace + startCharacter into:
    /// 1) the first complete top-level JSON object
    /// 2) trailing remainder after that object
    func splitLeadingObject(from buffer: String) -> (object: String, trailing: String)? {
        guard let start = buffer.firstIndex(where: { !$0.isWhitespace }),
            buffer[start] == startCharacter
        else { return nil }

        var depth = 0
        var inString = false
        var isEscaped = false

        var index = start
        while index < buffer.endIndex {
            let character = buffer[index]

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
                case "{":
                    depth += 1
                case "}":
                    depth -= 1
                    if depth == 0 {
                        let object = String(buffer[start ... index])
                        let trailingStart = buffer.index(after: index)
                        let trailing =
                            trailingStart < buffer.endIndex
                            ? String(buffer[trailingStart...])
                            : ""
                        return (object, trailing)
                    }
                default:
                    break
                }
            }

            index = buffer.index(after: index)
        }

        return nil
    }
}
