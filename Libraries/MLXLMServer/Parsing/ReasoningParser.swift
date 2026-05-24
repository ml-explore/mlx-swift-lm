// Copyright © 2026 Apple Inc.

import Foundation

public enum ReasoningParserFormat: String, Codable, Sendable, CaseIterable {
    case none
    case deepseekR1 = "deepseek_r1"
    case qwen3
    case harmony

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        let raw = try container.decode(String.self)
            .lowercased()
            .replacingOccurrences(of: "-", with: "_")

        switch raw {
        case "none", "off", "disabled":
            self = .none
        case "deepseek_r1", "deepseek", "r1", "think":
            self = .deepseekR1
        case "qwen3", "qwen":
            self = .qwen3
        case "harmony", "gpt_oss", "openai_harmony":
            self = .harmony
        default:
            throw DecodingError.dataCorruptedError(
                in: container,
                debugDescription: "Unsupported reasoning parser '\(raw)'"
            )
        }
    }
}

public struct ParsedReasoning: Sendable, Equatable {
    public var content: String
    public var reasoningContent: String?
}

public struct ReasoningParser: Sendable {
    public var format: ReasoningParserFormat

    public init(format: ReasoningParserFormat) {
        self.format = format
    }

    public func parse(_ text: String) -> ParsedReasoning {
        switch format {
        case .none:
            return .init(content: text, reasoningContent: nil)
        case .deepseekR1, .qwen3:
            return parseThinkTags(text)
        case .harmony:
            return parseHarmony(text)
        }
    }

    private func parseThinkTags(_ text: String) -> ParsedReasoning {
        var remaining = text
        var reasoning: [String] = []

        while let start = remaining.range(of: "<think>"),
            let end = remaining.range(of: "</think>", range: start.upperBound..<remaining.endIndex)
        {
            reasoning.append(String(remaining[start.upperBound..<end.lowerBound]))
            remaining.removeSubrange(start.lowerBound..<end.upperBound)
        }

        let reasoningText = reasoning.joined(separator: "\n").trimmedForReasoning
        return .init(
            content: remaining.trimmedForReasoning,
            reasoningContent: reasoningText.isEmpty ? nil : reasoningText
        )
    }

    private func parseHarmony(_ text: String) -> ParsedReasoning {
        let channelMarker = "<|channel|>"
        let messageMarker = "<|message|>"
        let endMarker = "<|end|>"
        var final: [String] = []
        var analysis: [String] = []
        var cursor = text.startIndex

        while let channelStart = text.range(of: channelMarker, range: cursor..<text.endIndex) {
            let channelNameStart = channelStart.upperBound
            guard let messageStart = text.range(
                of: messageMarker,
                range: channelNameStart..<text.endIndex
            ) else { break }

            let channel = String(text[channelNameStart..<messageStart.lowerBound])
            let contentStart = messageStart.upperBound
            let contentEnd = text.range(of: endMarker, range: contentStart..<text.endIndex)
            let messageEnd = contentEnd?.lowerBound ?? text.endIndex
            let content = String(text[contentStart..<messageEnd])

            switch channel {
            case "analysis":
                analysis.append(content)
            case "final":
                final.append(content)
            default:
                break
            }

            cursor = contentEnd?.upperBound ?? text.endIndex
        }

        if final.isEmpty && analysis.isEmpty {
            return .init(content: text, reasoningContent: nil)
        }

        let reasoningText = analysis.joined(separator: "\n").trimmedForReasoning
        return .init(
            content: final.joined(separator: "\n").trimmedForReasoning,
            reasoningContent: reasoningText.isEmpty ? nil : reasoningText
        )
    }
}

extension String {
    fileprivate var trimmedForReasoning: String {
        trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
