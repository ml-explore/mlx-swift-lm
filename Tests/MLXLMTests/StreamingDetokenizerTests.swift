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
            case 60: out += "\u{FE0F}"  // VARIATION SELECTOR-16, its own token
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
        var det = tokenizer.makeStreamingDetokenizer()
        var out = ""
        for id in tokens {
            det.append(token: id)
            if let text = det.next() { out += text }
        }
        if let text = det.finish() { out += text }
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

    // MARK: grapheme clusters that grow across a token boundary

    /// Exact scalar sequences: `String ==` compares under canonical
    /// equivalence, which would let `e` + U+0301 equal `é`.
    private func scalars(_ text: String) -> [UInt32] {
        text.unicodeScalars.map(\.value)
    }

    /// A token that appends a combining scalar to the previous token's
    /// character must emit only that scalar. A `Character`-level common
    /// prefix treats `🏳` and `🏳️` as different characters and re-emits the
    /// base, so `🏳️‍🌈` streamed as `🏳🏳️🏳️‍🏳️‍🌈`.
    func testCombiningScalarAppendedToPreviousCharacterIsEmittedOnce() {
        let tokenizer = AppendOnlyTokenizer(pieces: [
            1: "\u{1F3F3}",  // 🏳 WAVING WHITE FLAG
            2: "\u{FE0F}",  // VARIATION SELECTOR-16
            3: "\u{200D}",  // ZERO WIDTH JOINER
            4: "\u{1F308}",  // 🌈 RAINBOW
        ])
        let tokens = [1, 2, 3, 4]
        let streamed = stream(tokens, tokenizer)
        XCTAssertEqual(
            scalars(streamed), [0x1F3F3, 0xFE0F, 0x200D, 0x1F308],
            "base character re-emitted: \(streamed.debugDescription)")
        XCTAssertEqual(
            streamed, tokenizer.decode(tokenIds: tokens, skipSpecialTokens: false))
    }

    /// The same growth on an ASCII base: `'` then U+FE0F then `'` is three
    /// scalars, not four.
    func testVariationSelectorAfterASCIIDoesNotRepeatTheBase() {
        let tokenizer = AppendOnlyTokenizer(pieces: [1: "'", 2: "\u{FE0F}", 3: "'"])
        let streamed = stream([1, 2, 3], tokenizer)
        XCTAssertEqual(scalars(streamed), [0x27, 0xFE0F, 0x27], streamed.debugDescription)
    }

    /// A combining accent arriving as its own token: `e` + U+0301 + `x`.
    func testCombiningAccentDoesNotRepeatTheBase() {
        let tokenizer = AppendOnlyTokenizer(pieces: [1: "e", 2: "\u{0301}", 3: "x"])
        let streamed = stream([1, 2, 3], tokenizer)
        XCTAssertEqual(scalars(streamed), [0x65, 0x0301, 0x78], streamed.debugDescription)
    }

    /// The REPLACEMENT CHARACTER hold-back and the cluster growth compose:
    /// a multi-byte character completed across two tokens, then a variation
    /// selector token joining it, emits each scalar exactly once.
    func testVariationSelectorAfterSplitMultibyteCharacterIsEmittedOnce() {
        let tokenizer = SplitMultibyteTokenizer()
        let tokens = [65, 50, 51, 60, 66]  // "a", 中(first half), 中(second half), U+FE0F, "b"
        let streamed = stream(tokens, tokenizer)
        XCTAssertEqual(scalars(streamed), [0x61, 0x4E2D, 0xFE0F, 0x62], streamed.debugDescription)
        XCTAssertEqual(
            streamed, tokenizer.decode(tokenIds: tokens, skipSpecialTokens: false))
    }
}
