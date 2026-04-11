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

    // MARK: - Properties

    private let parser: any ToolCallParser
    private let tools: [[String: any Sendable]]?
    private let supportsBareJSONFallback: Bool
    private let maxJSONFallbackBufferLength = 32_768
    private let jsonObjectScanner = JSONLeadingObjectScanner(startCharacter: "{")
    private var state = State.normal
    private var toolCallBuffer = ""

    /// The tool calls extracted during processing.
    public var toolCalls: [ToolCall] = []

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
    ///   - tools: Optional tool schemas for type-aware parsing
    public init(format: ToolCallFormat = .json, tools: [[String: any Sendable]]? = nil) {
        self.parser = format.createParser()
        self.tools = tools
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

    /// Drain queued tool calls in parse order.
    ///
    /// This is useful for streaming consumers that need deterministic FIFO dispatch
    /// of all complete calls discovered while processing a token chunk.
    func drainToolCalls() -> [ToolCall] {
        guard !toolCalls.isEmpty else { return [] }
        let drained = toolCalls
        toolCalls.removeAll(keepingCapacity: true)
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
            return nil
        }

        let buffered = toolCallBuffer
        let parsedCalls = parser.parseEOS(buffered, tools: tools)
        toolCalls.append(contentsOf: parsedCalls)

        toolCallBuffer = ""
        state = .normal

        return returnBufferedText && parsedCalls.isEmpty ? buffered : nil
    }

    // MARK: - Private Methods

    /// Process chunk for inline formats (no wrapper tags).
    ///
    /// Uses brace counting to detect when output looks like a JSON tool call.
    /// While braces are unbalanced the content is buffered (returns `nil`)
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

                if let toolCall = parser.parse(content: toolCallBuffer, tools: tools) {
                    toolCalls.append(toolCall)
                    toolCallBuffer = ""
                    state = .normal
                    return leading.isEmpty ? nil : leading
                }

                // Still collecting — check if braces are balanced (would mean parse
                // failed on complete JSON, so it's not a tool call)
                if jsonBracesBalanced(toolCallBuffer) {
                    state = .normal
                    let buffer = toolCallBuffer
                    toolCallBuffer = ""
                    return leading + buffer
                }

                return leading.isEmpty ? nil : leading
            }

            // No brace seen — pass through as regular text
            return chunk

        case .potentialToolCall, .collectingToolCall, .collectingJSONToolCall:
            toolCallBuffer += chunk

            if let toolCall = parser.parse(content: toolCallBuffer, tools: tools) {
                toolCalls.append(toolCall)
                toolCallBuffer = ""
                state = .normal
                return nil
            }

            // If braces are balanced but parse failed, this isn't a tool call — flush
            if jsonBracesBalanced(toolCallBuffer) {
                state = .normal
                let buffer = toolCallBuffer
                toolCallBuffer = ""
                return buffer
            }

            // Still collecting
            return nil
        }
    }

    /// Check whether open/close braces are balanced in the string.
    private func jsonBracesBalanced(_ text: String) -> Bool {
        var depth = 0
        for ch in text {
            if ch == "{" { depth += 1 } else if ch == "}" { depth -= 1 }
        }
        return depth == 0
    }

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
            return chunk
        }

        toolCallBuffer += chunk
        var leadingToken: String?

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
                    fallthrough
                } else {
                    return nil
                }
            } else {
                // Otherwise, return the collected text and reset the state.
                state = .normal
                let buffer = toolCallBuffer
                toolCallBuffer = ""
                return (leadingToken ?? "") + buffer
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
                    toolCalls.append(toolCall)
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
                    return combine(leadingToken, trailingText)
                }

                // Preserve unparsed tagged payload as plain text, then continue scanning.
                state = .normal
                toolCallBuffer = ""
                if let trailingToken,
                    tokenCouldContainToolStart(trailingToken, startChar: startChar)
                {
                    return combine(
                        leadingToken,
                        combine(bufferedToolCall, processChunk(trailingToken))
                    )
                }
                return combine(leadingToken, combine(bufferedToolCall, trailingToken))
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
            return combine(leadingToken, buffered)
        }

        switch jsonObjectScanner.evaluatePrefix(in: toolCallBuffer) {
        case .invalidObject:
            state = .normal
            let buffered = toolCallBuffer
            toolCallBuffer = ""
            // vLLM-style recovery: if a tagged tool call exists later, retry tagged parsing.
            if buffered.contains(startTag) {
                return combine(leadingToken, processChunk(buffered))
            }
            return combine(leadingToken, buffered)
        case .needsMore, .validObject:
            break
        }

        guard let split = jsonObjectScanner.splitLeadingObject(from: toolCallBuffer) else {
            // Continue buffering until a complete top-level JSON object is available.
            return leadingToken?.isEmpty ?? true ? nil : leadingToken
        }

        let jsonCandidate = split.object
        let trailingToken = split.trailing

        if let toolCall = parser.parse(content: jsonCandidate, tools: tools) {
            toolCalls.append(toolCall)

            state = .normal
            toolCallBuffer = ""

            if trailingToken.isEmpty {
                return leadingToken?.isEmpty ?? true ? nil : leadingToken
            }

            if tokenCouldContainToolStart(trailingToken, startChar: startChar) {
                return combine(leadingToken, processChunk(trailingToken))
            }

            return combine(leadingToken, trailingToken)
        }

        // If it looked like JSON but is not a valid tool call payload,
        // flush it back as normal text while still scanning trailing content.
        state = .normal
        toolCallBuffer = ""
        if tokenCouldContainToolStart(trailingToken, startChar: startChar) {
            return combine(leadingToken, combine(jsonCandidate, processChunk(trailingToken)))
        }
        return combine(leadingToken, combine(jsonCandidate, trailingToken))
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

private struct JSONLeadingObjectScanner {
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
