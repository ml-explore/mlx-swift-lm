// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration
#if canImport(FoundationModels, _version: 2)

import Foundation
import FoundationModels
import MLXLMCommon

/// Rejects an attachment label that a tokenizer would read as a special token.
///
/// The check runs on the rendered form, `[label]`, because that is what reaches the
/// tokenizer: a label of `IMG` renders as `[IMG]`, a real image token on
/// Mistral-family models. It catches tokenizer special tokens only, so a label
/// holding a string a processor treats as magic, such as FastVLM's `<image>`, still
/// passes.
@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
struct AttachmentLabelValidator {

    /// The validator used by ``MLXLanguageModel``.
    static let `default` = AttachmentLabelValidator()

    /// Throws if any label, as rendered into the prompt, contains a token
    /// `tokenizer` treats as special.
    ///
    /// - Parameters:
    ///   - attachments: The distinct labels to check, each with the entry it
    ///     came from so a rejection can point at the offending prompt.
    ///   - tokenizer: The tokenizer that will encode the prompt. The answer is
    ///     per-model: `<|image_pad|>` is special for Qwen and ordinary text for
    ///     Gemma 4, and the reverse holds for `<|image|>`.
    /// - Throws: `LanguageModelError.unsupportedTranscriptContent`, naming the
    ///   label and the token.
    func validate(
        _ attachments: [TranscriptConverter.LabeledAttachment],
        with tokenizer: any MLXLMCommon.Tokenizer
    ) throws {
        for attachment in attachments {
            // The rendered form, matching what a message generator emits.
            let rendered = "[\(attachment.label)]"
            let ids = tokenizer.encode(text: rendered, addSpecialTokens: false)
            guard containsSpecialToken(ids, tokenizer) else { continue }

            let names = specialTokenNames(in: ids, tokenizer)
            let named =
                names.isEmpty
                ? "a tokenizer special token"
                : names.map { "`\($0)`" }.joined(separator: ", ")
            throw LanguageModelError.unsupportedTranscriptContent(
                LanguageModelError.UnsupportedTranscriptContent(
                    unsupportedContent: [attachment.entry],
                    debugDescription:
                        "The image attachment label \"\(attachment.label)\" contains \(named), "
                        + "which this model's tokenizer turns into a special token rather than "
                        + "text. Special tokens in a label corrupt the prompt's image "
                        + "placeholders. Use a label made of ordinary text."
                ))
        }
    }

    /// Whether `ids` contains any token the tokenizer treats as special.
    ///
    /// Nothing in the protocol enumerates a tokenizer's special tokens, so this
    /// decodes the same ids with and without `skipSpecialTokens` and compares: the
    /// two differ only when a special id is present. The call site passes
    /// `addSpecialTokens: false`, or the tokenizer's own BOS rejects every label.
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
