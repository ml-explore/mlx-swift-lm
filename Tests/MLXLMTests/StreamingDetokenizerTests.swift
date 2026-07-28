// Copyright © 2026 Apple Inc.

import Foundation
import XCTest

@testable import MLXLMCommon

/// A tokenizer whose `decode` is deliberately NOT append-only: a two-space
/// token followed by a comma decodes to `" ,"` (one space), mirroring the
/// SentencePiece whitespace normalization that Gemma's real tokenizer performs
/// (`decode([\n,"  "]) == "\n  "` but `decode([\n,"  ",","]) == "\n ,"`).
private struct WhitespaceCollapsingTokenizer: MLXLMCommon.Tokenizer {
    static let pieces: [Int: String] = [10: "\n", 20: "  ", 30: ",", 40: "\""]

    func encode(text: String, addSpecialTokens: Bool) -> [Int] { [] }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        let joined = tokenIds.map { Self.pieces[$0] ?? "" }.joined()
        // SentencePiece-style: collapse the run of spaces before a comma.
        return joined.replacingOccurrences(of: "  ,", with: " ,")
    }

    func convertTokenToId(_ token: String) -> Int? {
        Self.pieces.first { $0.value == token }?.key
    }
    func convertIdToToken(_ id: Int) -> String? { Self.pieces[id] }

    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] { [] }
}

/// A normal append-only tokenizer: `decode` is exactly the concatenation of
/// per-token pieces. Guards against the fix changing well-behaved tokenizers.
private struct AppendOnlyTokenizer: MLXLMCommon.Tokenizer {
    let pieces: [Int: String]

    func encode(text: String, addSpecialTokens: Bool) -> [Int] { [] }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        tokenIds.map { pieces[$0] ?? "" }.joined()
    }
    func convertTokenToId(_ token: String) -> Int? { pieces.first { $0.value == token }?.key }
    func convertIdToToken(_ id: Int) -> String? { pieces[id] }
    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }
    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] { [] }
}

/// Models a multi-byte character split across two byte-fallback tokens: token
/// 50 alone decodes to the Unicode REPLACEMENT CHARACTER (incomplete), and the
/// pair [50, 51] decodes to the completed character. Mirrors how a SentencePiece
/// byte-fallback tokenizer emits a partial UTF-8 sequence mid-generation.
private struct SplitMultibyteTokenizer: MLXLMCommon.Tokenizer {
    func encode(text: String, addSpecialTokens: Bool) -> [Int] { [] }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        var out = ""
        var i = 0
        while i < tokenIds.count {
            switch tokenIds[i] {
            case 50:
                if i + 1 < tokenIds.count, tokenIds[i + 1] == 51 {
                    out += "\u{4E2D}"  // 中 — completed once the second half arrives
                    i += 2
                    continue
                }
                out += "\u{fffd}"  // incomplete: only the first half so far
            case 51:
                out += "\u{fffd}"  // second half without its first half
            case 65: out += "a"
            case 66: out += "b"
            default: break
            }
            i += 1
        }
        return out
    }

    func convertTokenToId(_ token: String) -> Int? { nil }
    func convertIdToToken(_ id: Int) -> String? { nil }
    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }
    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] { [] }
}

final class StreamingDetokenizerTests: XCTestCase {

    private func stream(_ tokens: [Int], _ tokenizer: any Tokenizer) -> String {
        var det = NaiveStreamingDetokenizer(tokenizer: tokenizer)
        var out = ""
        for id in tokens {
            det.append(token: id)
            if let text = det.next() { out += text }
        }
        return out
    }

    /// Regression for the guided-generation "dropped required property" bug:
    /// a structural comma committed to the grammar must not vanish from the
    /// emitted text just because the tokenizer's decode is not append-only.
    func testDoesNotDropCommaWhenDecodeIsNotAppendOnly() {
        let tokenizer = WhitespaceCollapsingTokenizer()
        let tokens = [10, 20, 30, 40]  // \n  "  "  ,  "

        let full = tokenizer.decode(tokenIds: tokens, skipSpecialTokens: false)
        XCTAssertTrue(full.contains(","), "sanity: full decode should contain the comma")

        let streamed = stream(tokens, tokenizer)
        XCTAssertTrue(
            streamed.contains(","),
            "streaming detokenizer dropped the comma: \(streamed.debugDescription)")
    }

    /// The fix must not change behavior for a normal append-only tokenizer.
    func testAppendOnlyStreamMatchesFullDecode() {
        let tokenizer = AppendOnlyTokenizer(pieces: [1: "Hel", 2: "lo", 3: ", ", 4: "world"])
        let tokens = [1, 2, 3, 4]
        XCTAssertEqual(
            stream(tokens, tokenizer),
            tokenizer.decode(tokenIds: tokens, skipSpecialTokens: false))
    }

    /// A multi-byte character split across tokens must be emitted exactly once,
    /// after its final byte arrives — not dropped, not duplicated, and the
    /// intermediate REPLACEMENT CHARACTER must be suppressed. Exercises the
    /// `new.last == "\u{fffd}"` early-return branch in `next()`.
    func testSplitMultibyteUnicodeEmitsCompletedCharacterOnce() {
        let tokenizer = SplitMultibyteTokenizer()
        let tokens = [65, 50, 51, 66]  // "a", 中(first half), 中(second half), "b"

        let streamed = stream(tokens, tokenizer)
        XCTAssertEqual(
            streamed, "a\u{4E2D}b",
            "expected 'a中b', got \(streamed.debugDescription)")
        XCTAssertFalse(
            streamed.contains("\u{fffd}"),
            "replacement character leaked into output: \(streamed.debugDescription)")
        XCTAssertEqual(
            streamed, tokenizer.decode(tokenIds: tokens, skipSpecialTokens: false))
    }

    /// The newline-triggered `startNewSegment()` reset must still preserve a
    /// structural character when the post-reset decode is non-append-only.
    func testNewlineResetPreservesCommaWithNonAppendOnlyDecode() {
        let tokenizer = WhitespaceCollapsingTokenizer()
        let tokens = [10, 20, 30]  // "\n" (resets segment), "  ", "," (collapses "  ," -> " ,")

        let streamed = stream(tokens, tokenizer)
        XCTAssertTrue(
            streamed.contains(","),
            "comma dropped across a newline segment reset: \(streamed.debugDescription)")
    }
}
