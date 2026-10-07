// Copyright © 2026 Apple Inc.

import Foundation

/// Plain ChatML, as used by the Qwen family without its tool extensions.
///
/// ```
/// <|im_start|>system\n...<|im_end|>\n
/// <|im_start|>user\n...<|im_end|>\n
/// <|im_start|>assistant\n...<|im_end|>\n
/// <|im_start|>assistant\n            generation prompt
/// ```
///
/// Role names are ordinary text, so any role renders. `<|im_end|>` is the default EOS:
/// a turn's prompt plus its generated tokens (including EOS) is a prefix of the next
/// turn's rendering, since the `\n` after `<|im_end|>` comes after that prefix.
///
/// Plain ChatML has no tool convention, so tools and tool calls throw
/// ``ScriptedTemplateError/unsupported(_:)``.
public struct ChatMLTemplate: ScriptedChatTemplate {
    public static let imStart = "<|im_start|>"
    public static let imEnd = "<|im_end|>"

    public init() {}

    public var specialTokens: [String] { [Self.imStart, Self.imEnd] }
    public var corpus: [String] { ["system\n", "user\n", "assistant\n", "tool\n"] }
    public var endOfTurnToken: String? { Self.imEnd }

    public func render(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [String] {
        if let tools, !tools.isEmpty {
            throw ScriptedTemplateError.unsupported("tools")
        }

        var segments: [String] = []
        for message in messages {
            if message["tool_calls"] != nil {
                throw ScriptedTemplateError.unsupported("tool_calls")
            }
            guard let role = message["role"] as? String, !role.isEmpty else {
                throw ScriptedTemplateError.unknownRole("")
            }
            segments.append(Self.imStart)
            segments.append(role + "\n")

            let content = ScriptedTemplateMessage.content(of: message)
            if !content.isEmpty {
                segments.append(content)
            }
            segments.append(Self.imEnd)
            segments.append("\n")
        }

        let addGenerationPrompt = additionalContext?["add_generation_prompt"] as? Bool ?? true
        if addGenerationPrompt {
            segments.append(Self.imStart)
            segments.append("assistant\n")
        }
        return segments
    }
}
