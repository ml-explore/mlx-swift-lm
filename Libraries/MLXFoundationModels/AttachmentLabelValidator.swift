// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration
#if canImport(FoundationModels, _version: 2)

import Foundation
import FoundationModels
import MLXLMCommon

@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
struct AttachmentLabelValidator {

    static let `default` = AttachmentLabelValidator()

    func validate(
        _ attachments: [TranscriptConverter.LabeledAttachment],
        with tokenizer: any MLXLMCommon.Tokenizer
    ) throws {
        for attachment in attachments {
            // Run the tokenizer check first, because its error names the special token.
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
