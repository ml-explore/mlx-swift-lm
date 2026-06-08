// Copyright © 2025 Apple Inc.

#if FoundationModelsIntegration
    #if canImport(FoundationModels, _version: 2)

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
                        // System message for model instructions
                        guard let text = extractText(from: instructions.segments) else {
                            logger.warning("Skipping instructions entry with no text content")
                            return nil
                        }
                        return Chat.Message.system(text)

                    case .prompt(let prompt):
                        // User message for prompts
                        guard let text = extractText(from: prompt.segments) else {
                            logger.warning("Skipping prompt entry with no text content")
                            return nil
                        }
                        return Chat.Message.user(text)

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

                    default:
                        // Skip unsupported entry types (toolCalls, toolOutput, etc.)
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
        }

    #endif  // canImport(FoundationModels)
#endif  // FoundationModelsIntegration
