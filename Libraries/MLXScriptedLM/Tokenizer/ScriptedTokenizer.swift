// Copyright © 2026 Apple Inc.

import Foundation
import MLXLMCommon

/// A deterministic `Tokenizer` for scripted tests.
///
/// Encoding is longest-match over a ``ScriptedVocabulary``, with atomic specials matched
/// first. `decode(encode(s)) == s` for every string, so text survives the decode and
/// re-encode that `ChatSession` performs on each turn.
public struct ScriptedTokenizer: MLXLMCommon.Tokenizer {

    /// Which token, if any, ends generation.
    public enum EndOfSequence: Sendable, Equatable {
        /// The template's end-of-turn token.
        case template
        /// A specific atomic special.
        case token(String)
        /// No EOS: generation runs until `maxTokens`.
        case none
    }

    public let vocabulary: ScriptedVocabulary
    public let template: any ScriptedChatTemplate
    public let bosToken: String?
    public let eosToken: String?
    /// Byte fallback encodes everything, so there is no unknown token.
    public var unknownToken: String? { nil }

    /// Uses a prebuilt vocabulary, which must hold the template's special tokens.
    ///
    /// - Parameters:
    ///   - vocabulary: must hold the template's special tokens, and the EOS and BOS tokens.
    ///   - template: renders chat messages.
    ///   - eos: defaults to the template's end-of-turn token.
    ///   - bosToken: added by `encode(text:addSpecialTokens: true)`. Must be an atomic special.
    public init(
        vocabulary: ScriptedVocabulary,
        template: any ScriptedChatTemplate = MinimalChatTemplate(),
        eos: EndOfSequence = .template,
        bosToken: String? = nil
    ) {
        let eosToken = Self.resolve(eos, template: template)
        for token in template.specialTokens + [eosToken, bosToken].compactMap({ $0 }) {
            precondition(
                vocabulary.specialID(token) != nil,
                "\(token) must be registered as an atomic special")
        }
        for marker in template.textMarkers {
            precondition(
                vocabulary.specialID(marker) == nil,
                "template text marker \(marker) is registered as an atomic special")
        }
        self.vocabulary = vocabulary
        self.template = template
        self.eosToken = eosToken
        self.bosToken = bosToken
    }

    /// Builds the vocabulary from `corpus` plus everything the template declares.
    ///
    /// The corpus should hold every string a test will encode, so it tokenizes into
    /// corpus pieces rather than byte fallback. Unlisted text still round-trips.
    /// `specials` and `textMarkers` are for markers the scenario adds itself, such as
    /// `<think>`; the template's own markers need not be listed.
    public init(
        corpus: [String],
        template: any ScriptedChatTemplate = MinimalChatTemplate(),
        specials: [String] = [],
        textMarkers: [String] = [],
        configuration: VocabularyBuilder.Configuration = .init(),
        eos: EndOfSequence = .template,
        bosToken: String? = nil
    ) {
        let eosToken = Self.resolve(eos, template: template)
        var builder = VocabularyBuilder(configuration: configuration)
        for special in template.specialTokens + specials + [eosToken, bosToken].compactMap({ $0 }) {
            builder.addSpecial(special)
        }
        for marker in template.textMarkers + textMarkers {
            builder.addSpecial(marker, atomic: false)
        }
        builder.add(texts: template.corpus + corpus)
        self.init(
            vocabulary: builder.build(), template: template, eos: eos, bosToken: bosToken)
    }

    private static func resolve(_ eos: EndOfSequence, template: any ScriptedChatTemplate)
        -> String?
    {
        switch eos {
        case .template: template.endOfTurnToken
        case .token(let token): token
        case .none: nil
        }
    }

    // MARK: - Tokenizer

    public func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        let ids = vocabulary.encode(text)
        if addSpecialTokens, let bosToken, let bos = vocabulary.specialID(bosToken) {
            return [bos] + ids
        }
        return ids
    }

    public func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        vocabulary.decode(tokenIds, skipSpecialTokens: skipSpecialTokens)
    }

    public func convertTokenToId(_ token: String) -> Int? {
        vocabulary.id(forToken: token)
    }

    public func convertIdToToken(_ id: Int) -> String? {
        vocabulary.entry(id)?.name
    }

    public func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        try render(messages: messages, tools: tools, additionalContext: additionalContext)
            .flatMap { vocabulary.encode($0) }
    }

    // MARK: - Test helpers

    public func render(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]? = nil,
        additionalContext: [String: any Sendable]? = nil
    ) throws -> [String] {
        try template.render(
            messages: messages, tools: tools, additionalContext: additionalContext)
    }
}
