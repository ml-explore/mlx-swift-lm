// Copyright © 2026 Apple Inc.

import Foundation
import Testing

@testable import MLXLMCommon

@Suite("StructuredTextScanner")
struct StructuredTextScannerTests {

    private let pythonic = StructuredTextScanner(quotes: ["'", "\""])
    private let gemma = StructuredTextScanner(quotes: ["\""], escapeMarker: "<|\"|>")

    // MARK: - Top-level index

    @Test("Finds a delimiter at the top level")
    func findsTopLevelDelimiter() throws {
        let text = "a=1"[...]
        let index = try #require(pythonic.firstTopLevelIndex(of: "=", in: text))
        #expect(text[..<index] == "a")
    }

    @Test("Ignores delimiters inside quotes, brackets, and marker spans")
    func ignoresNonStructuralDelimiters() {
        #expect(pythonic.firstTopLevelIndex(of: "=", in: "'a=b'"[...]) == nil)
        #expect(pythonic.firstTopLevelIndex(of: ",", in: "[1, 2]"[...]) == nil)
        #expect(pythonic.firstTopLevelIndex(of: ":", in: #"{"a": 1}"#[...]) == nil)
        #expect(gemma.firstTopLevelIndex(of: ",", in: #"<|"|>a, b<|"|>"#[...]) == nil)
    }

    @Test("An escaped quote does not end a quoted span")
    func escapedQuoteStaysInsideSpan() {
        #expect(pythonic.firstTopLevelIndex(of: ",", in: #"'a\',b'"#[...]) == nil)
    }

    // MARK: - Splitting

    @Test("Splits only on top-level separators")
    func splitsOnTopLevelSeparators() {
        let text = #"a='x,y', b=[1, 2], c={"k": 3}"#[...]
        let fields = pythonic.splitTopLevel(text, separator: ",")
        #expect(fields.count == 3)
        #expect(fields[0] == "a='x,y'")
        #expect(fields[1].trimmingCharacters(in: .whitespaces) == "b=[1, 2]")
        #expect(fields[2].trimmingCharacters(in: .whitespaces) == #"c={"k": 3}"#)
    }

    @Test("Marker spans keep separators out of the split")
    func markerSpansAreOpaque() {
        let fields = gemma.splitTopLevel(#"note:<|"|>a,b<|"|>,count:2"#[...], separator: ",")
        #expect(fields.count == 2)
        #expect(fields[0] == #"note:<|"|>a,b<|"|>"#)
        #expect(fields[1] == "count:2")
    }

    @Test("A trailing separator does not produce an empty field")
    func trailingSeparatorIsDropped() {
        #expect(pythonic.splitTopLevel("a=1, "[...], separator: ",").count == 1)
    }

    @Test("Interior empty fields are preserved for malformed input")
    func interiorEmptyFieldsPreserved() {
        #expect(pythonic.splitTopLevel("a=1,,b=2"[...], separator: ",").count == 3)
    }

    // MARK: - Balanced groups

    @Test("Finds the closing bracket of a balanced group")
    func findsBalancedEnd() throws {
        let text = "(a, b)tail"[...]
        let end = try #require(gemma.endOfGroup(in: text, openedAt: text.startIndex))
        #expect(text[text.index(after: end)...] == "tail")
    }

    @Test("Nested and quoted delimiters cannot close a group")
    func nestedAndQuotedDelimitersDoNotClose() throws {
        for source in [#"(note='a)]b')"#, "(inner=(1, 2))", #"({"k": "}"})"#] {
            let text = source[...]
            let end = try #require(
                pythonic.endOfGroup(in: text, openedAt: text.startIndex), "\(source)")
            #expect(end == text.index(before: text.endIndex), "\(source)")
        }
    }

    @Test("An unbalanced or mismatched group has no end")
    func unbalancedGroupHasNoEnd() {
        let unterminated = "(a, b"[...]
        #expect(pythonic.endOfGroup(in: unterminated, openedAt: unterminated.startIndex) == nil)

        let mismatched = "(a]"[...]
        #expect(pythonic.endOfGroup(in: mismatched, openedAt: mismatched.startIndex) == nil)

        let notAnOpener = "a)"[...]
        #expect(pythonic.endOfGroup(in: notAnOpener, openedAt: notAnOpener.startIndex) == nil)
    }

    @Test("An unterminated marker span consumes the remainder")
    func unterminatedMarkerSpanIsOpaque() {
        #expect(gemma.firstTopLevelIndex(of: ",", in: #"a:<|"|>b,c"#[...]) == nil)
    }
}
