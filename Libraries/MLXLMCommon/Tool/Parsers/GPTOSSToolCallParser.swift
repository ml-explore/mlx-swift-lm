// Copyright © 2025 Apple Inc.

import Foundation

/// Parser for GPT-OSS Harmony tool call format.
/// Example:
/// <|channel|>commentary to=get_weather
/// <|message|>{"location": "Tokyo"}
/// <|call|>
public struct GPTOSSToolCallParser: ToolCallParser, Sendable {
    public let startTag: String? = Tag.channelTool
    public let startTags: [String] = [Tag.channelTool, Tag.assistantTool, Tag.assistantChannelTool]
    public let endTag: String? = "<|call|>"

    private enum Tag {
        static let channel = "<|channel|>"
        static let message = "<|message|>"
        static let end = "<|end|>"
        static let call = "<|call|>"
        static let `return` = "<|return|>"
        static let constrain = "<|constrain|>"
        static let assistantTool = "<|start|>assistant to="
        static let assistantChannelTool = "<|start|>assistant<|channel|>commentary to="
        static let channelTool = "<|channel|>commentary to="
    }

    public init() {}

    public func parse(content: String, tools: [[String: any Sendable]]?) -> ToolCall? {
        extractSingle(from: content)
    }

    public func parseEOS(_ toolCallBuffer: String, tools: [[String: any Sendable]]?) -> [ToolCall] {
        extractAll(from: toolCallBuffer)
    }

    private func extractAll(from text: String) -> [ToolCall] {
        var calls: [ToolCall] = []
        var cursor = text.startIndex

        while cursor < text.endIndex {
            guard
                let messageRange = text.range(
                    of: Tag.message, range: cursor ..< text.endIndex)
            else {
                break
            }

            let rawHeader = String(text[cursor ..< messageRange.lowerBound])
            let boundary = firstBoundary(in: text, from: messageRange.upperBound)
            let payloadEnd = boundary?.lowerBound ?? text.endIndex
            let payload = String(text[messageRange.upperBound ..< payloadEnd])

            if let function = parseFunction(header: rawHeader, payload: payload) {
                calls.append(ToolCall(function: function))
            }

            if let boundary {
                // If the boundary is a new-call channel opener, keep the tag so the next
                // iteration's rawHeader includes it (parseFunction requires <|channel|>).
                cursor =
                    String(text[boundary]) == Tag.channel
                    ? boundary.lowerBound : boundary.upperBound
            } else {
                cursor = text.endIndex
            }
        }
        return calls
    }

    private func extractSingle(from chunk: String) -> ToolCall? {
        guard let messageRange = chunk.range(of: Tag.message) else { return nil }
        let rawChannelHeader = String(chunk[..<messageRange.lowerBound])
        let payload = String(chunk[messageRange.upperBound...])
        var cleanPayload = payload
        for tag in [Tag.end, Tag.call, Tag.return, Tag.channel] {
            if let tagRange = cleanPayload.range(of: tag) {
                cleanPayload = String(cleanPayload[..<tagRange.lowerBound])
            }
        }

        guard let function = parseFunction(header: rawChannelHeader, payload: cleanPayload) else {
            return nil
        }
        return ToolCall(function: function)
    }

    private func parseFunction(header: String, payload: String) -> ToolCall.Function? {
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

        let normalizedArguments = normalizedArgumentsJSON(from: payload)
        guard let argsDict = tryParseJSON(normalizedArguments) as? [String: any Sendable] else {
            return nil
        }

        return ToolCall.Function(name: normalizedName, arguments: argsDict)
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

    private func firstBoundary(in text: String, from index: String.Index) -> Range<String.Index>? {
        guard index < text.endIndex else { return nil }
        let searchRange = index ..< text.endIndex

        let tags = [Tag.end, Tag.call, Tag.return]
        var minRange: Range<String.Index>? = nil

        for tag in tags {
            if let range = text.range(of: tag, range: searchRange) {
                if minRange == nil || range.lowerBound < minRange!.lowerBound {
                    minRange = range
                }
            }
        }

        if let channelRange = text.range(of: Tag.channel, range: searchRange) {
            let nextSearch = channelRange.upperBound ..< text.endIndex
            if text.range(of: Tag.message, range: nextSearch) != nil {
                if minRange == nil || channelRange.lowerBound < minRange!.lowerBound {
                    minRange = channelRange
                }
            }
        }

        return minRange
    }
}
