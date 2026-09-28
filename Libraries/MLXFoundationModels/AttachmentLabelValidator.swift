// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration
#if canImport(FoundationModels, _version: 2)

import Foundation
import FoundationModels
import MLXLMCommon

/// Refuses an attachment label that a model would read as a picture marker.
///
/// A label that holds a marker character is always refused. A label whose rendered
/// `[label]` form encodes to a special token is refused on the models where it does.
@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
struct AttachmentLabelValidator {

    /// The validator used by ``MLXLanguageModel``.
    static let `default` = AttachmentLabelValidator()

    /// Refuses each label that would reach the model as a picture marker.
    ///
    /// - Parameters:
    ///   - attachments: The labels to check. Each one carries the entry it came from.
    ///   - tokenizer: The tokenizer that encodes the prompt.
    /// - Throws: `LanguageModelError.unsupportedTranscriptContent`.
    func validate(
        _ attachments: [TranscriptConverter.LabeledAttachment],
        with tokenizer: any MLXLMCommon.Tokenizer
    ) throws {
        for attachment in attachments {
            // Check the tokenizer first. It names the token, which is a better error.
            if let names = tokenizer.specialTokenNames(inImageLabel: attachment.label) {
                let named =
                    names.isEmpty
                    ? "a tokenizer special token"
                    : names.map { "`\($0)`" }.joined(separator: ", ")
                throw Self.rejection(
                    attachment,
                    because:
                        "holds \(named). This model's tokenizer turns that into a special "
                        + "token instead of text, which corrupts the prompt's image "
                        + "placeholders."
                )
            }

            if let character = attachment.label.first(
                where: UserInput.Image.markerCharacters.contains)
            {
                throw Self.rejection(
                    attachment,
                    because:
                        "holds `\(character)`. Vision models build their image placeholders "
                        + "from `<`, `>`, `|`, `[` and `]`. A label that holds one of them can "
                        + "reach the prompt as a placeholder, and then the model counts more "
                        + "images than you gave it."
                )
            }
        }
    }

    private static func rejection(
        _ attachment: TranscriptConverter.LabeledAttachment, because reason: String
    ) -> LanguageModelError {
        LanguageModelError.unsupportedTranscriptContent(
            LanguageModelError.UnsupportedTranscriptContent(
                unsupportedContent: [attachment.entry],
                debugDescription:
                    "The image attachment label \"\(attachment.label)\" \(reason) "
                    + "Use a label made of ordinary text."
            ))
    }
}

#endif  // canImport(FoundationModels)
#endif  // FoundationModelsIntegration
