// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration
#if canImport(FoundationModels, _version: 2)

import CoreImage
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
    static func mlxMessages(for entries: some Collection<Transcript.Entry>) throws -> [Chat
        .Message]
    {
        try entries.compactMap { entry -> Chat.Message? in
            switch entry {
            case .instructions(let instructions):
                // System message for model instructions. Attachments are
                // dropped, matching FoundationModels, which ignores images in
                // an instructions entry (rdar://163210652). Dropping is also the
                // only safe option here: the Qwen3-VL and Gemma4 templates emit
                // no vision placeholder for system content, while `UserInput`
                // still collects the pixels from every message, so carrying them
                // would leave the placeholder and image counts disagreeing.
                let text = extractText(from: instructions.segments)
                let dropped = try extractLabeledImages(from: instructions.segments, in: entry)
                if !dropped.isEmpty {
                    logger.warning(
                        "Dropping \(dropped.count, privacy: .public) image attachment(s) in an instructions entry; attach images to a prompt instead so the model receives them"
                    )
                }
                guard let text else {
                    logger.warning("Skipping instructions entry with no text content")
                    return nil
                }
                return Chat.Message.system(text)

            case .prompt(let prompt):
                // User message for prompts. Labeled image attachments ride
                // along as message images, and the renderer names them in the
                // text so the model can refer to a specific picture and quote
                // the label back.
                let text = extractText(from: prompt.segments)
                let labeled = try extractLabeledImages(from: prompt.segments, in: entry)
                let legend = AttachmentLabelRenderer.default.legend(
                    for: labeled.map(\.label))
                // Legend first: the images render ahead of the text, so the
                // legend's "above" is accurate and the caller's own text stays
                // last, closest to the model's turn.
                let content = [legend, text].compactMap { $0 }.joined(separator: "\n")
                guard !content.isEmpty || !labeled.isEmpty else {
                    logger.warning("Skipping prompt entry with no text or image content")
                    return nil
                }
                return Chat.Message.user(content, images: labeled.map(\.image))

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
                // originating call by id. Text remains verbatim; structured
                // GeneratedContent is serialized as JSON so the native chat
                // template can expose it to the continuation model turn.
                let content = extractToolOutputContent(from: output.segments)
                return Chat.Message.tool(content, id: output.id)

            default:
                // Skip unsupported entry types. Explicit `return nil` is a
                // tripwire: a newly added SDK entry type surfaces here for review
                // rather than being silently coerced into the wrong role.
                logger.debug("Skipping unsupported entry type")
                return nil
            }
        }
    }

    /// Extracts supported tool-output content in transcript segment order.
    ///
    /// Foundation Models lowers `String` outputs to `.text` and
    /// `GeneratedContent`/`@Generable` outputs to `.structure`. MLX chat
    /// templates accept tool results as strings, so structured values retain
    /// their JSON representation. Attachments and custom segments are deferred
    /// until their media and prompt-representation contracts are implemented.
    private static func extractToolOutputContent(
        from segments: [Transcript.Segment]
    ) -> String {
        segments.compactMap { segment -> String? in
            switch segment {
            case .text(let textSegment):
                return textSegment.content
            case .structure(let structuredSegment):
                return structuredSegment.content.jsonString
            case .attachment(let attachment):
                // FoundationModels renders tool-output attachments; this adapter
                // does not yet. Warn rather than logging at debug so a dropped
                // image is visible in the log instead of silently missing from
                // the model's view.
                logger.warning(
                    "Dropping an attachment in tool output (label: \(attachment.label ?? "none", privacy: .public)); tool-output images are not yet forwarded to the model"
                )
                return nil
            default:
                logger.debug("Skipping unsupported tool-output segment")
                return nil
            }
        }.joined(separator: "\n")
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

    /// One image input plus its label, in segment order.
    private struct LabeledImage {
        let image: UserInput.Image
        let label: String?
    }

    /// Extracts image inputs and their labels from attachment segments.
    ///
    /// Two things happen here that the naive reading of the SDK misses. First,
    /// `Transcript.ImageAttachment` keeps `orientation` as metadata and hands
    /// back unrotated pixels from `ciImage`, so the transform has to be applied
    /// here or an attachment created with an explicit orientation reaches the
    /// model sideways. FoundationModels itself passes the whole image buffer,
    /// orientation included, down to its own renderer. Second, the label rides
    /// along so the message text can name each image.
    ///
    /// - Parameters:
    ///   - segments: Array of transcript segments
    ///   - entry: The entry these segments belong to, for error reporting
    /// - Returns: The labeled image inputs found, in segment order
    /// - Throws: `LanguageModelError.unsupportedTranscriptContent` if an
    ///   attachment carries content this adapter cannot render. FoundationModels
    ///   throws the same error for its own non-image attachment case, and the
    ///   `@unknown default` makes a future SDK attachment kind surface as a
    ///   typed error rather than a silently missing input.
    private static func extractLabeledImages(
        from segments: [Transcript.Segment],
        in entry: Transcript.Entry
    ) throws -> [LabeledImage] {
        try segments.compactMap { segment -> LabeledImage? in
            guard case .attachment(let attachment) = segment else { return nil }
            switch attachment.content {
            case .image(let imageAttachment):
                return LabeledImage(
                    image: .ciImage(imageAttachment.ciImage.oriented(imageAttachment.orientation)),
                    label: attachment.label)
            @unknown default:
                throw LanguageModelError.unsupportedTranscriptContent(
                    LanguageModelError.UnsupportedTranscriptContent(
                        unsupportedContent: [entry],
                        debugDescription:
                            "This attachment carries content the MLX adapter cannot render. Only image attachments are supported."
                    ))
            }
        }
    }

    /// A distinct attachment label and the entry that carried it.
    ///
    /// The entry rides along so a label rejected by ``AttachmentLabelValidator``
    /// can name the prompt it came from. `Transcript.Entry` is `Sendable`, so this
    /// crosses the actor hop into `ModelContainer.perform` unboxed.
    struct LabeledAttachment: Sendable {
        let label: String
        let entry: Transcript.Entry
    }

    /// The distinct attachment labels present in `entries`, in first-seen order,
    /// each paired with the entry it was first seen in.
    ///
    /// Only prompt entries are considered: instructions attachments are dropped
    /// outright, matching FoundationModels, so a prompt is the only place an
    /// image with a label reaches the model.
    static func labeledAttachments(in entries: some Collection<Transcript.Entry>)
        -> [LabeledAttachment]
    {
        var seen = Set<String>()
        var ordered: [LabeledAttachment] = []
        for entry in entries {
            guard case .prompt(let prompt) = entry else { continue }
            for segment in prompt.segments {
                guard case .attachment(let attachment) = segment,
                    let label = attachment.label
                else { continue }
                if seen.insert(label).inserted {
                    ordered.append(LabeledAttachment(label: label, entry: entry))
                }
            }
        }
        return ordered
    }

    /// The distinct attachment labels present in `entries`, in first-seen order.
    ///
    /// Used to constrain a guided `ImageReference` to a label that can actually
    /// resolve.
    static func attachmentLabels(in entries: some Collection<Transcript.Entry>) -> [String] {
        labeledAttachments(in: entries).map(\.label)
    }
}

#endif  // canImport(FoundationModels)
#endif  // FoundationModelsIntegration
