// Copyright © 2026 Apple Inc.

import Foundation

/// A minimal one-marker-per-role template for mechanics tests. Not a real model family:
/// prompts stay short and segment assertions stay readable.
///
/// ```
/// <|tools|>[...]<|end|>          only when tools are present
/// <|system|>...<|end|>
/// <|user|>...<|end|>
/// <|assistant|>...<|tool_call|>{...}<|end|>
/// <|tool|>...<|end|>
/// <|assistant|>                  generation prompt
/// ```
///
/// An assistant turn ends with `<|end|>`, which is also the default EOS. So the prompt
/// for one turn plus the generated tokens (including EOS) is a prefix of the next
/// turn's rendering.
///
/// Tool calls render as canonical JSON (sorted keys).
public struct MinimalChatTemplate: ScriptedChatTemplate {

    public struct Markers: Sendable, Equatable {
        public var system = "<|system|>"
        public var user = "<|user|>"
        public var assistant = "<|assistant|>"
        public var tool = "<|tool|>"
        public var tools = "<|tools|>"
        public var toolCall = "<|tool_call|>"
        public var end = "<|end|>"

        public init() {}

        public var all: [String] { [system, user, assistant, tool, tools, toolCall, end] }
    }

    public var markers: Markers
    /// Markers registered as multi-token text instead of atomic specials.
    public var nonAtomicMarkers: Set<String>

    public init(markers: Markers = .init(), nonAtomicMarkers: Set<String> = []) {
        self.markers = markers
        self.nonAtomicMarkers = nonAtomicMarkers
    }

    public var specialTokens: [String] {
        markers.all.filter { !nonAtomicMarkers.contains($0) }
    }

    public var textMarkers: [String] {
        markers.all.filter { nonAtomicMarkers.contains($0) }
    }

    public var endOfTurnToken: String? {
        nonAtomicMarkers.contains(markers.end) ? nil : markers.end
    }

    public func render(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [String] {
        var segments: [String] = []

        if let tools, !tools.isEmpty {
            segments.append(markers.tools)
            segments.append(try ScriptedTemplateMessage.canonicalJSON(tools))
            segments.append(markers.end)
        }

        for message in messages {
            let role = message["role"] as? String ?? ""
            segments.append(try roleMarker(role))

            let content = ScriptedTemplateMessage.content(of: message)
            if !content.isEmpty {
                segments.append(content)
            }

            if let calls = message["tool_calls"] as? [[String: any Sendable]] {
                for call in calls {
                    segments.append(markers.toolCall)
                    segments.append(
                        try ScriptedTemplateMessage.canonicalJSON(call["function"] ?? call))
                }
            }
            segments.append(markers.end)
        }

        let addGenerationPrompt = additionalContext?["add_generation_prompt"] as? Bool ?? true
        if addGenerationPrompt {
            segments.append(markers.assistant)
        }
        return segments
    }

    private func roleMarker(_ role: String) throws -> String {
        switch role {
        case "system": markers.system
        case "user": markers.user
        case "assistant": markers.assistant
        case "tool": markers.tool
        default: throw ScriptedTemplateError.unknownRole(role)
        }
    }
}
