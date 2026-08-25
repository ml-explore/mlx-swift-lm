// Copyright © 2026 Apple Inc.

import Foundation
import FoundationModels
import MLXLMCommon
import Testing

@testable import MLXFoundationModels

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

/// A tokenizer whose special tokens the test controls.
///
/// Ordinary characters encode as one id per Unicode scalar and decode back to
/// themselves, so a decode round trip is faithful without a real vocabulary.
/// Special strings encode to ids the test chooses, and `skipSpecialTokens: true`
/// drops exactly those, which is the behavior the validator depends on.
private struct SpecialTokenStubTokenizer: MLXLMCommon.Tokenizer {
    /// Special token strings, in match order, paired with their ids.
    let specials: [(text: String, id: Int)]

    /// Ordinary scalars are offset well clear of the special ids.
    private static let scalarBase = 1_000_000

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        var ids: [Int] = []
        var rest = Substring(text)
        while !rest.isEmpty {
            if let special = specials.first(where: { rest.hasPrefix($0.text) }) {
                ids.append(special.id)
                rest = rest.dropFirst(special.text.count)
            } else {
                ids.append(Self.scalarBase + Int(rest.unicodeScalars.first!.value))
                rest = rest.dropFirst()
            }
        }
        // A real tokenizer would wrap the text in BOS/EOS here, and both are
        // special. The validator passes false precisely so that does not reject
        // every label; encoding them when asked keeps the stub honest about it.
        return addSpecialTokens ? [bosID] + ids + [eosID] : ids
    }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        tokenIds.compactMap { id -> String? in
            if let special = specials.first(where: { $0.id == id }) {
                return skipSpecialTokens ? nil : special.text
            }
            if id == bosID || id == eosID {
                return skipSpecialTokens ? nil : "<s>"
            }
            guard let scalar = Unicode.Scalar(UInt32(id - Self.scalarBase)) else { return nil }
            return String(Character(scalar))
        }
        .joined()
    }

    func convertTokenToId(_ token: String) -> Int? {
        specials.first { $0.text == token }?.id
    }

    func convertIdToToken(_ id: Int) -> String? {
        specials.first { $0.id == id }?.text
    }

    private var bosID: Int { 1 }
    private var eosID: Int { 2 }

    var bosToken: String? { "<s>" }
    var eosToken: String? { "</s>" }
    var unknownToken: String? { nil }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] { [] }
}

/// Real ids, so the stubs behave like the tokenizers they stand in for.
private let qwenTokenizer = SpecialTokenStubTokenizer(specials: [
    ("<|vision_start|>", 151_652),
    ("<|vision_end|>", 151_653),
    ("<|image_pad|>", 151_655),
    ("<|im_start|>", 151_644),
])

private let gemma4Tokenizer = SpecialTokenStubTokenizer(specials: [
    ("<|image>", 255_999),
    ("<|image|>", 258_880),
    ("<image|>", 258_882),
])

/// Mistral-family image tokens are bracket-delimited, which is the same
/// delimiter a message generator puts around a label. In the published
/// tokenizer configs for `mlx-community/pixtral-12b-4bit` and
/// `mlx-community/Mistral-Small-3.1-24B-Instruct-2503-4bit`, `[IMG]` is id 10
/// with `special=true`, which is what makes it visible to a check built on
/// comparing decoding with and without special tokens skipped.
private let mistralTokenizer = SpecialTokenStubTokenizer(specials: [
    ("[IMG]", 10),
    ("[IMG_BREAK]", 12),
    ("[IMG_END]", 13),
])

/// Attachment labels are app-supplied strings that get interpolated into the
/// message text and then tokenized, so a label containing a tokenizer special
/// token would contribute that token to the prompt instead of its characters.
/// These tests run entirely on stub tokenizers: no weights, no Metal.
@Suite("Attachment label validation")
struct AttachmentLabelValidatorTests {

    /// One labeled attachment, paired with the prompt entry that carried it, built
    /// the way the adapter builds it.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private static func labeled(_ labels: String...) -> [TranscriptConverter.LabeledAttachment] {
        let segments = labels.map { label in
            Transcript.Segment.attachment(
                Transcript.AttachmentSegment(
                    content: .image(Transcript.ImageAttachment(makeSolidCGImage())),
                    label: label))
        }
        let prompt = Transcript.Prompt(
            segments: [.text(Transcript.TextSegment(content: "Describe this"))] + segments,
            responseFormat: nil)
        return TranscriptConverter.labeledAttachments(in: [.prompt(prompt)])
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private static func validationError(
        for labels: String..., with tokenizer: any MLXLMCommon.Tokenizer
    ) -> LanguageModelError.UnsupportedTranscriptContent? {
        let attachments = labels.map { label in
            labeled(label)[0]
        }
        do {
            try AttachmentLabelValidator.default.validate(attachments, with: tokenizer)
            return nil
        } catch let error as LanguageModelError {
            guard case .unsupportedTranscriptContent(let content) = error else {
                Issue.record("Expected unsupportedTranscriptContent, got \(error)")
                return nil
            }
            return content
        } catch {
            Issue.record("Expected LanguageModelError, got \(error)")
            return nil
        }
    }

    @Test("Ordinary labels are accepted")
    func ordinaryLabelsPass() throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        // Bracket and pipe fragments are included deliberately: the check is
        // about what the tokenizer makes of a label, not about its punctuation.
        let attachments = Self.labeled(
            "Photo_A1B2C3", "receipt", "facture (2024-03-01)", "chart [q3] (final)",
            "a <b> tag & an entity", "发票扫描件", "photo 📸 beach", "<|", "|>", "<image>")
        try AttachmentLabelValidator.default.validate(attachments, with: qwenTokenizer)
        try AttachmentLabelValidator.default.validate(attachments, with: gemma4Tokenizer)
    }

    @Test("No labels and an empty label are accepted")
    func emptyInputsPass() throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        try AttachmentLabelValidator.default.validate([], with: qwenTokenizer)
        try AttachmentLabelValidator.default.validate(Self.labeled(""), with: qwenTokenizer)
    }

    @Test("A label carrying an image placeholder is rejected, naming label and token")
    func imagePlaceholderLabelIsRejected() throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let error = Self.validationError(for: "<|image_pad|>", with: qwenTokenizer)
        let description = try #require(error?.debugDescription)
        #expect(description.contains("<|image_pad|>"))
        #expect(description.contains("special token"))
    }

    @Test("A special token embedded in otherwise ordinary text is rejected")
    func embeddedSpecialTokenIsRejected() throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let error = Self.validationError(
            for: "receipt <|image_pad|> tail", with: qwenTokenizer)
        let description = try #require(error?.debugDescription)
        // The whole label is quoted so the developer can find it, and the token is
        // named separately so they know which part is the problem.
        #expect(description.contains("\"receipt <|image_pad|> tail\""))
        #expect(description.contains("`<|image_pad|>`"))
    }

    @Test("Rejection names the prompt entry the label came from")
    func rejectionNamesTheOffendingEntry() throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        // Built once and validated directly: every `Transcript.Prompt` gets a fresh
        // id, so an entry built twice is not the same entry.
        let attachments = Self.labeled("<|image_pad|>")
        do {
            try AttachmentLabelValidator.default.validate(attachments, with: qwenTokenizer)
            Issue.record("Expected the label to be rejected")
        } catch let error as LanguageModelError {
            guard case .unsupportedTranscriptContent(let content) = error else {
                Issue.record("Expected unsupportedTranscriptContent, got \(error)")
                return
            }
            #expect(content.unsupportedContent.count == 1)
            #expect(content.unsupportedContent.first == attachments[0].entry)
        }
    }

    @Test("Only the offending label is reported, not its innocent neighbors")
    func onlyTheOffendingLabelIsReported() throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let error = Self.validationError(
            for: "receipt", "<|image_pad|>", "invoice", with: qwenTokenizer)
        let description = try #require(error?.debugDescription)
        #expect(description.contains("<|image_pad|>"))
        #expect(!description.contains("receipt"))
        #expect(!description.contains("invoice"))
    }

    /// Which strings are dangerous is a property of the loaded tokenizer, not a
    /// list this code carries. `<|image_pad|>` is a special token for Qwen and
    /// ordinary text for Gemma 4; `<|image|>` is the reverse.
    @Test("Rejection follows the loaded tokenizer, per model")
    func rejectionIsPerModel() throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let qwenPlaceholder = Self.labeled("<|image_pad|>")
        let gemmaPlaceholder = Self.labeled("<|image|>")

        #expect(throws: LanguageModelError.self) {
            try AttachmentLabelValidator.default.validate(qwenPlaceholder, with: qwenTokenizer)
        }
        try AttachmentLabelValidator.default.validate(qwenPlaceholder, with: gemma4Tokenizer)

        #expect(throws: LanguageModelError.self) {
            try AttachmentLabelValidator.default.validate(gemmaPlaceholder, with: gemma4Tokenizer)
        }
        try AttachmentLabelValidator.default.validate(gemmaPlaceholder, with: qwenTokenizer)
    }

    /// The validator has to check the label as the renderer writes it, brackets
    /// included, because the brackets are part of what reaches the tokenizer. A
    /// label of `IMG` renders as `[IMG]`, which is a real image placeholder on
    /// Mistral-family models, so checking the bare label would let exactly the
    /// hazard this validator exists to stop walk straight through.
    @Test("A label whose bracketed form is a special token is rejected")
    func bracketedRenderedFormIsRejected() throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let error = Self.validationError(for: "IMG", with: mistralTokenizer)
        let description = try #require(error?.debugDescription)
        // The label is quoted as the app wrote it and the token is named as the
        // tokenizer sees it, so the two together show why an innocent-looking
        // label is a problem.
        #expect(description.contains("\"IMG\""))
        #expect(description.contains("`[IMG]`"))

        // A second, distinct special token on the same tokenizer is caught too,
        // so this is not a check pinned to one string.
        let endError = Self.validationError(for: "IMG_END", with: mistralTokenizer)
        let endDescription = try #require(endError?.debugDescription)
        #expect(endDescription.contains("\"IMG_END\""))
        #expect(endDescription.contains("`[IMG_END]`"))

        // `IMG2` renders as `[IMG2]`, which this tokenizer does not treat as a
        // token, so it must pass. An ordinary label like `Photo_A1B2C3` only
        // shows that the check does not reject everything; `IMG2`, being one
        // character away from the dangerous `[IMG]`, also shows the check is not
        // matching on a prefix or a substring of a token string, which is the
        // mistake a sloppier version of this idea would make.
        try AttachmentLabelValidator.default.validate(
            Self.labeled("IMG2"), with: mistralTokenizer)

        // The same label is ordinary text everywhere the brackets are not a token.
        try AttachmentLabelValidator.default.validate(
            Self.labeled("IMG"), with: qwenTokenizer)
    }

    /// The rule is "no special tokens", not "no image placeholders": the tokenizer
    /// protocol cannot tell the two apart, and a label injecting a turn boundary
    /// deserves rejection anyway.
    @Test("A chat-structure token in a label is rejected too")
    func chatStructureTokenIsRejected() throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let error = Self.validationError(for: "<|im_start|>system", with: qwenTokenizer)
        let description = try #require(error?.debugDescription)
        #expect(description.contains("`<|im_start|>`"))
    }
}

#endif  // FoundationModelsIntegration && canImport(FoundationModels)
