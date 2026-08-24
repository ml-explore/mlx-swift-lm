// Copyright © 2026 Apple Inc.

import Foundation

/// Locates *structural* punctuation in a tool-call payload.
///
/// Dialects that separate arguments with punctuation — Pythonic
/// `[f(a='x', b=2)]`, Gemma `call:f{a:1,b:2}` — all have to answer one question
/// before they can split anything: is this comma / colon / bracket a separator,
/// or is it part of a value? Answering it with `firstIndex(of:)` truncates any
/// value that happens to contain the delimiter, which is how a quoted `")]"` or
/// a nested `{"a":1,"b":2}` ends up silently mangled.
///
/// This type answers that question once, for every such parser. It walks text
/// while treating three kinds of span as opaque:
///
/// - quoted spans, honoring backslash escapes (`'a,b'`),
/// - marker-delimited spans, where the same multi-character marker both opens
///   and closes the span (Gemma's `<|"|>`),
/// - bracket nesting across `()`, `[]`, and `{}`.
///
/// Only characters outside every opaque span are reported, each with the
/// bracket depth it sits at, so callers can ask for *top level* positions and
/// get exactly those.
struct StructuredTextScanner: Sendable {

    /// Characters that open and close a quoted span.
    let quotes: Set<Character>

    /// A marker that both opens and closes an opaque span, if the dialect has one.
    let escapeMarker: String?

    init(quotes: Set<Character> = [], escapeMarker: String? = nil) {
        self.quotes = quotes
        self.escapeMarker = escapeMarker
    }

    /// Bracket pairs tracked as nesting. All three share one depth counter, so
    /// a value may mix them freely as long as it is balanced.
    private static let brackets: [Character: Character] = ["(": ")", "[": "]", "{": "}"]
    private static let closers: Set<Character> = Set(brackets.values)

    // MARK: - Queries

    /// The first `character` that sits at the top level, or `nil` if it only
    /// ever appears inside a quoted span, a marker span, or nested brackets.
    func firstTopLevelIndex(of character: Character, in text: Substring) -> String.Index? {
        var found: String.Index?
        scan(text) { index, scanned, depth in
            guard depth == 0, scanned == character else { return .continue }
            found = index
            return .stop
        }
        return found
    }

    /// Splits `text` on every top-level `separator`.
    ///
    /// Interior empty fields are preserved so callers can detect malformed
    /// input; a trailing field is dropped when it is empty or all whitespace,
    /// which is what a trailing separator produces.
    func splitTopLevel(_ text: Substring, separator: Character) -> [Substring] {
        var fields: [Substring] = []
        var fieldStart = text.startIndex

        scan(text) { index, scanned, depth in
            guard depth == 0, scanned == separator else { return .continue }
            fields.append(text[fieldStart ..< index])
            fieldStart = text.index(after: index)
            return .continue
        }

        let last = text[fieldStart...]
        if !last.allSatisfy(\.isWhitespace) {
            fields.append(last)
        }
        return fields
    }

    /// The index of the bracket that closes the group opened at `openingIndex`,
    /// or `nil` when the group is unbalanced or `openingIndex` is not an opener.
    ///
    /// Delimiters inside quoted, marker-delimited, or more deeply nested spans
    /// cannot close the group, so a value containing `)]` or `}` is safe.
    func endOfGroup(in text: Substring, openedAt openingIndex: String.Index) -> String.Index? {
        guard openingIndex < text.endIndex,
            let closingCharacter = Self.brackets[text[openingIndex]]
        else { return nil }

        var closingIndex: String.Index?
        scan(text[openingIndex...]) { index, scanned, depth in
            guard depth == 0, index != openingIndex else { return .continue }
            guard scanned == closingCharacter else {
                // A closer of a different kind at this depth means the group is
                // malformed; refuse rather than guess where it ended.
                return Self.closers.contains(scanned) ? .stop : .continue
            }
            closingIndex = index
            return .stop
        }
        return closingIndex
    }

    // MARK: - Core scan

    private enum Step {
        case `continue`
        case stop
    }

    /// Visits every character outside an opaque span, with its bracket depth.
    ///
    /// Depth is reported *outside* the bracket for both ends of a pair, so in
    /// `a{b}c` the braces are reported at depth 0 and `b` at depth 1.
    private func scan(_ text: Substring, _ visit: (String.Index, Character, Int) -> Step) {
        var index = text.startIndex
        var depth = 0
        var activeQuote: Character?
        var isEscaped = false

        while index < text.endIndex {
            let character = text[index]

            if let marker = escapeMarker, activeQuote == nil,
                text[index...].hasPrefix(marker)
            {
                // Skip the whole marker span; a value may contain anything.
                let contentStart = text.index(index, offsetBy: marker.count)
                guard let closing = text[contentStart...].range(of: marker) else { return }
                index = closing.upperBound
                continue
            }

            if let quote = activeQuote {
                if isEscaped {
                    isEscaped = false
                } else if character == #"\"# {
                    isEscaped = true
                } else if character == quote {
                    activeQuote = nil
                }
                index = text.index(after: index)
                continue
            }

            if quotes.contains(character) {
                activeQuote = character
                index = text.index(after: index)
                continue
            }

            if Self.brackets[character] != nil {
                if visit(index, character, depth) == .stop { return }
                depth += 1
            } else if Self.closers.contains(character) {
                depth = max(0, depth - 1)
                if visit(index, character, depth) == .stop { return }
            } else {
                if visit(index, character, depth) == .stop { return }
            }

            index = text.index(after: index)
        }
    }
}
