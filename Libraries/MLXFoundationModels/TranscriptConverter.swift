// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration
#if canImport(FoundationModels, _version: 2)

import Foundation
import FoundationModels
import MLXLMCommon
import os.log

/// Converts FoundationModels transcript entries to MLX chat message format.
@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
struct TranscriptConverter {

    private static let logger = Logger(
        subsystem: "com.apple.FoundationModels-MLX", category: "TranscriptConverter")

    /// The MLX `Chat.Message` array for a collection of transcript entries.
    ///
    /// - Parameter entries: Transcript entries from FoundationModels
    /// - Returns: Array of MLX Chat.Message objects
    static func mlxMessages(for entries: some Collection<Transcript.Entry>) -> [Chat
        .Message]
    {
        entries.compactMap { entry -> Chat.Message? in
            switch entry {
            case .instructions(let instructions):
                // System message for model instructions. Labeled image
                // attachments ride along as message images, mirroring the
                // prompt path, so the `.vision` gate sees them and they are
                // not silently dropped.
                let text = extractText(from: instructions.segments)
                let images = extractImages(from: instructions.segments)
                guard text != nil || !images.isEmpty else {
                    logger.warning(
                        "Skipping instructions entry with no text or image content")
                    return nil
                }
                return Chat.Message.system(text ?? "", images: images)

            case .prompt(let prompt):
                // User message for prompts. Labeled image attachments
                // (public `.attachment` segments) ride along as message
                // images; text is still concatenated as before.
                let text = extractText(from: prompt.segments)
                let images = extractImages(from: prompt.segments)
                guard text != nil || !images.isEmpty else {
                    logger.warning("Skipping prompt entry with no text or image content")
                    return nil
                }
                return Chat.Message.user(text ?? "", images: images)

            case .response(let response):
                // Assistant message for previous responses
                guard let text = extractText(from: response.segments) else {
                    logger.warning("Skipping response entry with no text content")
                    return nil
                }
                return Chat.Message.assistant(text)

            case .reasoning:
                // Prior-turn reasoning is intentionally NOT replayed into the
                // model's chat history (per SKILL.md): the answer carries
                // forward, the chain-of-thought does not. Dropped explicitly so
                // a future SDK change is reviewed here rather than silently
                // absorbed by the catch-all below.
                logger.debug("Skipping reasoning entry (not replayed into chat history)")
                return nil

            case .toolCalls(let toolCalls):
                // Replay prior tool calls as an assistant message carrying the
                // structured calls. The model's tool-aware chat template renders
                // these into its native tool-call channel; DefaultMessageGenerator
                // serializes each id/name/arguments (see ToolCallIdTests). Without
                // this, a continuation round would re-issue the same call.
                let calls = toolCalls.map { call -> MLXLMCommon.ToolCall in
                    let argumentsData = Data(call.arguments.jsonString.utf8)
                    let arguments: [String: JSONValue]
                    if let decoded = try? JSONDecoder().decode(
                        [String: JSONValue].self, from: argumentsData)
                    {
                        arguments = decoded
                    } else {
                        logger.warning(
                            "Failed to decode arguments for tool: \(call.toolName, privacy: .public)"
                        )
                        arguments = [:]
                    }
                    return MLXLMCommon.ToolCall(
                        function: .init(name: call.toolName, arguments: arguments),
                        id: call.id)
                }
                guard !calls.isEmpty else {
                    logger.warning("Skipping toolCalls entry with no calls")
                    return nil
                }
                return Chat.Message.assistant("", toolCalls: calls)

            case .toolOutput(let output):
                // Replay the tool result as a `tool` message correlated to its
                // originating call by id. FoundationModels sets ToolOutput.id to
                // the call id (ToolCallCoordinator), so the template resolves the
                // tool name via the matching tool_call_id.
                let text = extractText(from: output.segments) ?? ""
                return Chat.Message.tool(text, id: output.id)

            default:
                // Skip unsupported entry types. Explicit `return nil` is a
                // tripwire: a newly added SDK entry type surfaces here for review
                // rather than being silently coerced into the wrong role.
                logger.debug("Skipping unsupported entry type")
                return nil
            }
        }
    }

    /// Extracts text content from transcript segments.
    ///
    /// Concatenates all text segments with newlines.
    /// Skips images, structured content, and other non-text segments.
    ///
    /// - Parameter segments: Array of transcript segments
    /// - Returns: Concatenated text, or nil if no text content found
    private static func extractText(from segments: [Transcript.Segment]) -> String? {
        let texts = segments.compactMap { segment -> String? in
            switch segment {
            case .text(let textSegment):
                return textSegment.content

            default:
                // Skip images, structured content, and local attention segment types
                logger.debug("Skipping non-text segment in extractText")
                return nil
            }
        }

        let combined = texts.joined(separator: "\n")
        return combined.isEmpty ? nil : combined
    }

    /// Extracts image inputs from image attachment segments.
    ///
    /// Each image attachment is handed over as its already-decoded
    /// `CIImage`. Segments that carry no image produce no input.
    ///
    /// - Parameter segments: Array of transcript segments
    /// - Returns: The image inputs found, in segment order
    private static func extractImages(from segments: [Transcript.Segment])
        -> [UserInput.Image]
    {
        segments.compactMap { segment -> UserInput.Image? in
            guard case .attachment(let attachment) = segment,
                case .image(let imageAttachment) = attachment.content
            else {
                return nil
            }
            return .ciImage(imageAttachment.ciImage)
        }
    }
}

#endif  // canImport(FoundationModels)
#endif  // FoundationModelsIntegration
