// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration
#if canImport(FoundationModels, _version: 2)

import Foundation
import FoundationModels
import MLXLMCommon

/// Rejects attachment labels that a tokenizer would turn into special tokens.
///
/// ``AttachmentLabelRenderer`` interpolates app-supplied labels into the message
/// text, and that text is tokenized. HuggingFace tokenizers match their added
/// tokens inside ordinary text -- swift-transformers splits the input on a regex
/// built from `tokenizer.json`'s added tokens and maps any exact match straight
/// to its id -- so a label containing one contributes a real special token to
/// the prompt rather than the characters the app wrote.
///
/// That is not a cosmetic problem. Measured on a real
/// `Qwen3-VL-4B-Instruct-4bit` tokenizer, the text `The image above is
/// [<|image_pad|>].` encodes to `[785, 2168, 3403, 374, 508, 151655, 936]`,
/// where 151655 is the image padding token, and routing the same string through
/// the chat template keeps it. Downstream, an extra image placeholder either
/// fails the request (Gemma 4 throws on a surplus placeholder) or, worse,
/// survives as an image position with no vision features behind it and
/// misaligns the feature scatter and the rotary position counts. Rejecting the
/// label is the only outcome that keeps the label usable as an identifier: a
/// silently stripped or rewritten label no longer matches what the app looks up,
/// or what a guided `ImageReference` was pinned to.
///
/// ## The rule is about special tokens, not image placeholders
///
/// Any special token is rejected, including a chat-structure token such as
/// `<|im_start|>`. That is intended, for two reasons. ``MLXLMCommon/Tokenizer``
/// cannot tell a vision placeholder from a chat token -- which ids mean "image"
/// lives in per-model model configuration, not in the tokenizer -- and a label
/// that injects a turn boundary deserves rejection on its own merits.
///
/// ## Known limitation
///
/// This makes labels safe against *tokenizer* special tokens, not against every
/// model's prompt conventions. FastVLM's processor splits the decoded prompt on
/// the literal string `<image>` (`MLXVLM/Models/FastVLM.swift`), and `<image>`
/// is not an added token in its tokenizer, so a label containing it passes this
/// check and still inserts an image marker. Catching that would need a per-model
/// list of magic strings, which cannot be kept correct across models; it is
/// recorded here rather than half-solved.
///
/// The check also guards one code path rather than the type system: it is called
/// from `MLXLanguageModel.Executor.respond`, which holds the module's only
/// transcript conversion. A second path that prepared a prompt from a transcript
/// would have to call it too.
@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
struct AttachmentLabelValidator {

    /// The validator used by ``MLXLanguageModel``.
    static let `default` = AttachmentLabelValidator()

    /// Throws if any label in `attachments`, as rendered into the prompt, contains
    /// a token `tokenizer` treats as special.
    ///
    /// The string checked is the label wrapped in the brackets
    /// ``AttachmentLabelRenderer`` writes around it, not the bare label, because
    /// the brackets are part of what reaches the tokenizer. That is deliberate and
    /// load-bearing: Mistral-family image tokens are themselves bracketed, so a
    /// label of `IMG` renders as `[IMG]` and becomes a genuine extra image
    /// placeholder even though the bare label is ordinary text. The brackets are
    /// also the right granularity to stop at. Encoding the whole legend sentence
    /// would let the renderer's own wording contribute a token the app's label did
    /// not, and blaming the app for that would be a false positive; `[label]` is
    /// precisely the substring the app controls.
    ///
    /// - Parameters:
    ///   - attachments: The distinct labels to check, each with the entry it
    ///     came from so a rejection can point at the offending prompt.
    ///   - tokenizer: The tokenizer that will encode the prompt. The answer is
    ///     per-model: `<|image_pad|>` is a special token for Qwen and ordinary
    ///     text for Gemma 4, and the reverse holds for `<|image|>`.
    /// - Throws: `LanguageModelError.unsupportedTranscriptContent`, naming the
    ///   label and the token, matching the error the converter already throws
    ///   for attachment content it cannot render.
    func validate(
        _ attachments: [TranscriptConverter.LabeledAttachment],
        with tokenizer: any MLXLMCommon.Tokenizer
    ) throws {
        for attachment in attachments {
            // The rendered form, matching `AttachmentLabelRenderer.legend` exactly.
            // Checking the bare label would miss a label whose bracketed form is
            // the special token, which is the live case on Mistral-family models.
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
    /// ``MLXLMCommon/Tokenizer`` exposes no way to enumerate a tokenizer's added
    /// or special tokens, and neither does swift-transformers (`addedTokens` and
    /// `specialTokens` are internal there), so this asks the question the one way
    /// the protocol allows: `skipSpecialTokens` drops exactly the special ids, so
    /// decoding the same ids both ways and comparing detects them without
    /// knowing any model's vocabulary in advance.
    ///
    /// A false positive is impossible by construction rather than merely
    /// unobserved: both calls decode the *same* ids through the same decoder and
    /// the same cleanup, so with no special id present the two results are
    /// byte-identical. Normalization, whitespace collapsing and byte-level BPE
    /// quirks cancel out because they apply equally to both sides.
    ///
    /// `addSpecialTokens: false` at the call site keeps the tokenizer's own
    /// BOS/EOS out of the comparison; those are special, and adding them would
    /// reject every label.
    private func containsSpecialToken(_ ids: [Int], _ tokenizer: any MLXLMCommon.Tokenizer)
        -> Bool
    {
        tokenizer.decode(tokenIds: ids, skipSpecialTokens: false)
            != tokenizer.decode(tokenIds: ids, skipSpecialTokens: true)
    }

    /// The names of the special tokens in `ids`, for the error message.
    ///
    /// Classification uses the same two-decode comparison per id, which stays
    /// valid for a single id because both sides see identical input. The *name*
    /// comes from `convertIdToToken`, not from decoding, because a single-id
    /// decode renders through the decoder's cleanup and is not required to
    /// reproduce the token string.
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
