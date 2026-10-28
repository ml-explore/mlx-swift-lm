// Copyright © 2026 Apple Inc.

import Foundation
import MLXLMCommon

/// A conversation to script: what the user says and what the model replies.
///
/// Write it as an array in code, or load it from JSON:
///
/// ```json
/// {
///   "turns": [
///     { "system": "Be brief." },
///     { "user": "hello" },
///     { "assistant": "hi there" },
///     { "user": "count" },
///     { "assistant": { "text": "one two three", "terminator": "none" } }
///   ]
/// }
/// ```
public struct Transcript: Sendable, Equatable, Codable {

    public enum Turn: Sendable, Equatable {
        case system(String)
        case user(String)
        case assistant(Assistant)
    }

    /// What ends an assistant turn's script.
    public enum Terminator: String, Sendable, Equatable, Codable {
        /// The tokenizer's EOS token, so generation stops on its own.
        case endOfSequence = "eos"
        /// Nothing: generation runs until `maxTokens`, emitting unscripted tokens after the
        /// script. Those are never a stop token or the turn marker, so the turn cannot end
        /// early. Stop strings can still match the decoded text.
        case none
    }

    public struct Assistant: Sendable, Equatable, ExpressibleByStringLiteral {
        public var text: String
        public var terminator: Terminator

        public init(_ text: String, terminator: Terminator = .endOfSequence) {
            self.text = text
            self.terminator = terminator
        }

        public init(stringLiteral value: String) {
            self.init(value)
        }
    }

    public var turns: [Turn]

    public init(_ turns: [Turn]) {
        self.turns = turns
    }

    public init(contentsOf url: URL) throws {
        self = try JSONDecoder().decode(Transcript.self, from: Data(contentsOf: url))
    }

    public var assistantTurns: [Assistant] {
        turns.compactMap {
            if case .assistant(let assistant) = $0 { assistant } else { nil }
        }
    }

    public var userMessages: [String] {
        turns.compactMap {
            if case .user(let text) = $0 { text } else { nil }
        }
    }

    /// All the text in the transcript, for building a tokenizer's vocabulary.
    public var corpus: [String] {
        turns.map {
            switch $0 {
            case .system(let text), .user(let text): text
            case .assistant(let assistant): assistant.text
            }
        }
    }

    /// The tokens each assistant turn emits: its text, then EOS if it has one.
    public func scripts(tokenizer: any Tokenizer) -> [[Int]] {
        assistantTurns.map { assistant in
            var tokens = tokenizer.encode(text: assistant.text, addSpecialTokens: false)
            if assistant.terminator == .endOfSequence {
                guard let eos = tokenizer.eosToken.flatMap(tokenizer.convertTokenToId) else {
                    preconditionFailure(
                        "the transcript ends a turn with EOS but the tokenizer has none")
                }
                tokens.append(eos)
            }
            return tokens
        }
    }
}

// MARK: - Tokenizer and scripts

extension ScriptedTokenizer {
    /// Builds a tokenizer whose vocabulary covers the transcript's text.
    public init(
        transcript: Transcript,
        template: any ScriptedChatTemplate = MinimalChatTemplate(),
        specials: [String] = [],
        textMarkers: [String] = [],
        configuration: VocabularyBuilder.Configuration = .init()
    ) {
        self.init(
            corpus: transcript.corpus, template: template, specials: specials,
            textMarkers: textMarkers, configuration: configuration)
    }
}

extension TurnScripts {

    /// Initialize the TurnScripts.
    ///
    /// The transcript's assistant scripts, with the template's
    /// generation prompt as the turn marker.
    ///
    /// An assistant turn that follows a user turn ends its prompt
    /// with that user message, rendered alone. This assumes the
    /// template renders a lone user message without a default
    /// system prompt.
    ///
    /// - Parameters:
    ///   - transcript: the transcript
    ///   - tokenizer: the tokenizer
    ///   - stopTokens: token ids that stop generation besides the
    ///     tokenizer's EOS
    public init(
        transcript: Transcript, tokenizer: ScriptedTokenizer, stopTokens: Set<Int> = []
    ) throws {
        let marker = try tokenizer.applyChatTemplate(
            messages: [], tools: nil, additionalContext: nil)

        var endings: [[Int]?] = []
        for (index, turn) in transcript.turns.enumerated() {
            guard case .assistant = turn else { continue }
            if index > 0, case .user(let text) = transcript.turns[index - 1] {
                endings.append(
                    try tokenizer.applyChatTemplate(
                        messages: [["role": "user", "content": text]], tools: nil,
                        additionalContext: nil))
            } else {
                endings.append(nil)
            }
        }

        self.init(
            scripts: transcript.scripts(tokenizer: tokenizer), turnMarker: marker,
            promptEndings: endings,
            endOfSequence: tokenizer.eosTokenId, stopTokens: stopTokens)
    }
}

// MARK: - Codable

extension Transcript.Turn: Codable {
    private enum CodingKeys: String, CodingKey {
        case system, user, assistant
    }

    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        guard container.allKeys.count == 1, let key = container.allKeys.first else {
            throw DecodingError.dataCorrupted(
                .init(
                    codingPath: container.codingPath,
                    debugDescription: "A turn needs exactly one of system, user, or assistant"))
        }
        switch key {
        case .system: self = .system(try container.decode(String.self, forKey: key))
        case .user: self = .user(try container.decode(String.self, forKey: key))
        case .assistant:
            self = .assistant(try container.decode(Transcript.Assistant.self, forKey: key))
        }
    }

    public func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        switch self {
        case .system(let text): try container.encode(text, forKey: .system)
        case .user(let text): try container.encode(text, forKey: .user)
        case .assistant(let assistant): try container.encode(assistant, forKey: .assistant)
        }
    }
}

extension Transcript.Assistant: Codable {
    private enum CodingKeys: String, CodingKey {
        case text, terminator
    }

    /// Decodes a plain string, or an object with `text` and an optional `terminator`.
    public init(from decoder: any Decoder) throws {
        if let text = try? decoder.singleValueContainer().decode(String.self) {
            self.init(text)
            return
        }
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.init(
            try container.decode(String.self, forKey: .text),
            terminator: try container.decodeIfPresent(
                Transcript.Terminator.self, forKey: .terminator) ?? .endOfSequence)
    }

    /// Encodes a plain string when the terminator is the default.
    public func encode(to encoder: any Encoder) throws {
        if terminator == .endOfSequence {
            var container = encoder.singleValueContainer()
            try container.encode(text)
        } else {
            var container = encoder.container(keyedBy: CodingKeys.self)
            try container.encode(text, forKey: .text)
            try container.encode(terminator, forKey: .terminator)
        }
    }
}
