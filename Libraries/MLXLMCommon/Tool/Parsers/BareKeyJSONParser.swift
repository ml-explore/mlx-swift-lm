// Copyright © 2026 Apple Inc.

import Foundation

/// Reads JSON that a dialect wrote with unquoted object keys.
///
/// Gemma writes a nested object as `{city: "Paris"}`. That is JSON in every respect except that its
/// keys carry no quotes, which `JSONSerialization` rejects, so the value falls back to its own raw
/// text: a parameter the schema declares an `object` reaches the tool as a string, and the call
/// fails at the far end with a schema error the model has no way to act on.
///
/// Only keys are rewritten. A bare *value* is left alone and the payload refused, because quoting it
/// would invent meaning — `{ok: true}` is a boolean while `{ok: yes}` is nothing JSON can name — and
/// a parser that guesses there is worse than one that hands back the literal text.
struct BareKeyJSONParser: Sendable {

    /// Keys are the only bare spans this parser rewrites, so quoted values stay opaque while the
    /// structure is walked.
    private let scanner = StructuredTextScanner(quotes: ["\""])

    /// Characters a bare key may contain. Anything else is a payload this parser refuses.
    private static let keyCharacters = CharacterSet.alphanumerics
        .union(CharacterSet(charactersIn: "_-."))

    /// Parses `text` as JSON, retrying once with bare object keys quoted.
    func parse(_ text: String) -> (any Sendable)? {
        if let value = tryParseJSON(text) { return value }
        guard let quoted = quotingBareKeys(text[...]) else { return nil }
        return tryParseJSON(quoted)
    }

    /// Rewrites `text` with every bare object key quoted.
    ///
    /// Scalars are returned verbatim, so the rewrite reaches keys and nothing else. Returns `nil`
    /// when a group is unbalanced, a member has no `:`, or a key is not a bare identifier.
    private func quotingBareKeys(_ text: Substring) -> String? {
        let literal = text.trimmingWhitespace()

        switch literal.first {
        case "{":
            guard let body = groupBody(of: literal) else { return nil }
            var members: [String] = []
            for member in scanner.splitTopLevel(body, separator: ",") {
                guard let colon = scanner.firstTopLevelIndex(of: ":", in: member),
                    let key = quotedKey(member[..<colon].trimmingWhitespace()),
                    let value = quotingBareKeys(member[member.index(after: colon)...])
                else { return nil }
                members.append("\(key):\(value)")
            }
            return "{\(members.joined(separator: ","))}"

        case "[":
            guard let body = groupBody(of: literal) else { return nil }
            var elements: [String] = []
            for element in scanner.splitTopLevel(body, separator: ",") {
                guard let value = quotingBareKeys(element) else { return nil }
                elements.append(value)
            }
            return "[\(elements.joined(separator: ","))]"

        default:
            return String(literal)
        }
    }

    /// The contents of a group whose closing bracket ends `literal`.
    private func groupBody(of literal: Substring) -> Substring? {
        guard let end = scanner.endOfGroup(in: literal, openedAt: literal.startIndex),
            end == literal.index(before: literal.endIndex)
        else { return nil }
        return literal[literal.index(after: literal.startIndex) ..< end]
    }

    /// A quoted form of `key`, or `nil` when it is neither already quoted nor a bare identifier.
    private func quotedKey(_ key: Substring) -> String? {
        if key.count > 1, key.hasPrefix("\""), key.hasSuffix("\"") {
            return String(key)
        }
        guard !key.isEmpty,
            key.unicodeScalars.allSatisfy(Self.keyCharacters.contains)
        else { return nil }
        return "\"\(key)\""
    }
}
