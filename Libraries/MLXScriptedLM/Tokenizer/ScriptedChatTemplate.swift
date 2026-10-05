// Copyright © 2026 Apple Inc.

import Foundation

/// A chat template for scripted tests.
///
/// ``render(messages:tools:additionalContext:)`` returns strings that the tokenizer
/// encodes one at a time and concatenates. Encoding each string separately keeps a
/// message's tokens independent of its neighbors, which `ChatSession` prefix reuse
/// depends on. Output is vocabulary-independent.
///
/// A template declares the tokens it needs. ``ScriptedTokenizer`` adds them to the
/// vocabulary, so scenario authors list only their own text.
public protocol ScriptedChatTemplate: Sendable {
    /// Markers that must be single (atomic) special tokens, such as `<|im_end|>`.
    var specialTokens: [String] { get }

    /// Markers that must encode to two or more ordinary tokens.
    var textMarkers: [String] { get }

    /// Literal text the template emits, such as role names, added to the corpus so it
    /// tokenizes into pieces rather than byte fallback.
    var corpus: [String] { get }

    /// The token that ends an assistant turn, used as the default EOS. Must be one of
    /// ``specialTokens``.
    var endOfTurnToken: String? { get }

    func render(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [String]
}

extension ScriptedChatTemplate {
    public var textMarkers: [String] { [] }
    public var corpus: [String] { [] }
    public var endOfTurnToken: String? { nil }
}

public enum ScriptedTemplateError: Error, Equatable {
    case unknownRole(String)
    case invalidJSON(String)
    /// The template has no rendering for this input, such as tools in plain ChatML.
    case unsupported(String)
}

/// Message helpers shared by template implementations.
public enum ScriptedTemplateMessage {
    /// String content, or the concatenated `text` fields of a content-parts array.
    public static func content(of message: [String: any Sendable]) -> String {
        switch message["content"] {
        case let text as String:
            return text
        case let parts as [[String: any Sendable]]:
            return parts.compactMap { $0["text"] as? String }.joined()
        case let parts as [[String: String]]:
            return parts.compactMap { $0["text"] }.joined()
        default:
            return ""
        }
    }

    public static func canonicalJSON(_ value: any Sendable) throws -> String {
        guard JSONSerialization.isValidJSONObject(value) else {
            throw ScriptedTemplateError.invalidJSON(String(describing: value))
        }
        let data = try JSONSerialization.data(
            withJSONObject: value, options: [.sortedKeys, .withoutEscapingSlashes])
        return String(decoding: data, as: UTF8.self)
    }
}
