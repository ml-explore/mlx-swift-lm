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
            // A message generator writes the label into the prompt inside brackets.
            let rendered = "[\(attachment.label)]"
            let ids = tokenizer.encode(text: rendered, addSpecialTokens: false)
            if containsSpecialToken(ids, tokenizer) {
                let names = specialTokenNames(in: ids, tokenizer)
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

    /// Whether `ids` contains any token the tokenizer treats as special.
    ///
    /// Nothing in the protocol enumerates a tokenizer's special tokens, so this
    /// decodes the same ids with and without `skipSpecialTokens` and compares: the
    /// two differ only when a special id is present. The call site passes
    /// `addSpecialTokens: false`, or the tokenizer's own BOS rejects every label.
    ///
    /// This sees only the tokens a tokenizer flags special. An added token without
    /// that flag stays invisible here, which is why ``validate(_:with:)`` also
    /// refuses the marker characters.
    private func containsSpecialToken(_ ids: [Int], _ tokenizer: any MLXLMCommon.Tokenizer)
        -> Bool
    {
        tokenizer.decode(tokenIds: ids, skipSpecialTokens: false)
            != tokenizer.decode(tokenIds: ids, skipSpecialTokens: true)
    }

    /// The names of the special tokens in `ids`, for the error message. The name
    /// comes from `convertIdToToken`, because decoding one id runs the decoder's
    /// cleanup and need not reproduce the token string.
    private func specialTokenNames(in ids: [Int], _ tokenizer: any MLXLMCommon.Tokenizer)
        -> [String]
    {
        var names: [String] = []
        var seen = Set<Int>()
        for id in ids where containsSpecialToken([id], tokenizer) {
            guard seen.insert(id).inserted else { continue }
            names.append(tokenizer.convertIdToToken(id) ?? "token \(id)")
        }
        return names
    }
}

#endif  // canImport(FoundationModels)
#endif  // FoundationModelsIntegration
