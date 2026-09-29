// Copyright © 2026 Apple Inc.

import Foundation
import FoundationModels
import MLXLMCommon
import Testing

@testable import MLXFoundationModels

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

private struct SpecialTokenStubTokenizer: MLXLMCommon.Tokenizer {
    let specials: [(text: String, id: Int)]

    var nonSpecialAdded: [(text: String, id: Int)] = []

    private static let scalarBase = 1_000_000

    private var allAdded: [(text: String, id: Int)] { specials + nonSpecialAdded }

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        var ids: [Int] = []
        var rest = Substring(text)
        while !rest.isEmpty {
            if let added = allAdded.first(where: { rest.hasPrefix($0.text) }) {
                ids.append(added.id)
                rest = rest.dropFirst(added.text.count)
            } else {
                ids.append(Self.scalarBase + Int(rest.unicodeScalars.first!.value))
                rest = rest.dropFirst()
            }
        }
        // Add a special BOS and EOS when asked, as a real tokenizer does. The tests then
        // fail if the label check passes `addSpecialTokens: true`.
        return addSpecialTokens ? [bosID] + ids + [eosID] : ids
    }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        tokenIds.compactMap { id -> String? in
            if let special = specials.first(where: { $0.id == id }) {
                return skipSpecialTokens ? nil : special.text
            }
            if let added = nonSpecialAdded.first(where: { $0.id == id }) {
                return added.text
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
        allAdded.first { $0.text == token }?.id
    }

    func convertIdToToken(_ id: Int) -> String? {
        allAdded.first { $0.id == id }?.text
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

private let mistralTokenizer = SpecialTokenStubTokenizer(specials: [
    ("[IMG]", 10),
    ("[IMG_BREAK]", 12),
    ("[IMG_END]", 13),
])

/// In the GLM-OCR tokenizer, the image token `<|image|>` is not a special token.
private let glmOcrTokenizer = SpecialTokenStubTokenizer(
    specials: [
        ("<|begin_of_image|>", 59_256),
        ("<|end_of_image|>", 59_257),
    ],
    nonSpecialAdded: [("<|image|>", 59_280)])

@Suite("Attachment label validation")
struct AttachmentLabelValidatorTests {

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
        let attachments = Self.labeled(
            "Photo_A1B2C3", "receipt", "facture (2024-03-01)", "chart q3 (final)",
            "a & an entity", "发票扫描件", "photo 📸 beach", "100% done", "a/b\\c")
        try AttachmentLabelValidator.default.validate(attachments, with: qwenTokenizer)
        try AttachmentLabelValidator.default.validate(attachments, with: gemma4Tokenizer)
        try AttachmentLabelValidator.default.validate(attachments, with: glmOcrTokenizer)
        try AttachmentLabelValidator.default.validate(attachments, with: mistralTokenizer)
    }

    @Test("A marker character in a label is refused on every model")
    func markerCharactersAreRefused() throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        for label in ["chart [q3] (final)", "a <b> tag", "<|", "|>", "]", "half<"] {
            let error = Self.validationError(for: label, with: qwenTokenizer)
            let description = try #require(
                error?.debugDescription, "expected \(label) to be refused")
            #expect(description.contains("\"\(label)\""))
        }
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
        #expect(description.contains("\"receipt <|image_pad|> tail\""))
        #expect(description.contains("`<|image_pad|>`"))
    }

    @Test("Rejection names the prompt entry the label came from")
    func rejectionNamesTheOffendingEntry() throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        // Build the entry once. Each `Transcript.Prompt` gets a new id, so two builds of
        // the same prompt are not equal.
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

    @Test("Rejection of a delimiter-free label follows the loaded tokenizer, per model")
    func rejectionIsPerModel() throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let img = Self.labeled("IMG")

        #expect(throws: LanguageModelError.self) {
            try AttachmentLabelValidator.default.validate(img, with: mistralTokenizer)
        }
        try AttachmentLabelValidator.default.validate(img, with: qwenTokenizer)
        try AttachmentLabelValidator.default.validate(img, with: gemma4Tokenizer)
        try AttachmentLabelValidator.default.validate(img, with: glmOcrTokenizer)
    }

    @Test("A label whose bracketed form is a special token is rejected")
    func bracketedRenderedFormIsRejected() throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let error = Self.validationError(for: "IMG", with: mistralTokenizer)
        let description = try #require(error?.debugDescription)
        #expect(description.contains("\"IMG\""))
        #expect(description.contains("`[IMG]`"))

        let endError = Self.validationError(for: "IMG_END", with: mistralTokenizer)
        let endDescription = try #require(endError?.debugDescription)
        #expect(endDescription.contains("\"IMG_END\""))
        #expect(endDescription.contains("`[IMG_END]`"))

        try AttachmentLabelValidator.default.validate(
            Self.labeled("IMG2"), with: mistralTokenizer)

        try AttachmentLabelValidator.default.validate(
            Self.labeled("IMG"), with: qwenTokenizer)
    }

    @Test("A chat-structure token in a label is rejected too")
    func chatStructureTokenIsRejected() throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let error = Self.validationError(for: "<|im_start|>system", with: qwenTokenizer)
        let description = try #require(error?.debugDescription)
        #expect(description.contains("`<|im_start|>`"))
    }

    @Test("A label carrying a non-special added token is rejected")
    func nonSpecialAddedTokenIsRejected() throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let error = Self.validationError(for: "<|image|>", with: glmOcrTokenizer)
        let description = try #require(error?.debugDescription)
        #expect(description.contains("\"<|image|>\""))
    }

    @Test("A label carrying a marker the tokenizer does not know is rejected")
    func markerUnknownToTheTokenizerIsRejected() throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let error = Self.validationError(for: "<image>", with: qwenTokenizer)
        let description = try #require(error?.debugDescription)
        #expect(description.contains("\"<image>\""))
    }
}

#endif  // FoundationModelsIntegration && canImport(FoundationModels)
