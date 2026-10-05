// Copyright © 2026 Apple Inc.

import Foundation
import MLXScriptedLM
import Testing

/// SplitMix64: a seeded generator so fuzzed inputs are identical on every run.
struct SeededGenerator: RandomNumberGenerator {
    private var state: UInt64

    init(seed: UInt64) { state = seed }

    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}

enum FuzzText {
    /// ASCII words and punctuation, JSON, whitespace, multi-scalar emoji, combining marks,
    /// CJK, and marker text.
    static let fragments = [
        "a", "b", "hello", " world", " international", "\n", "\t", "  ", "\r\n", "{", "}",
        "\"", "key", ":", "1", "23", ",", "[", "]", "?", "!", "é", "e\u{301}", "中", "文",
        "👍🏽", "🏳️‍🌈", "ß", "…", "\u{0}", "<|end|>", "<|user|>", "<think>", "</think>",
        "<|", "|>", "x", "Z", " ",
    ]

    /// Characters and text that `fragments` never uses, including near-misses of specials.
    static let outsideFragments = [
        "🚀 Ωmega naïve façade", "Привет мир", "𝔘𝔫𝔦𝔠𝔬𝔡𝔢", "quux_zzz-42",
        "\u{7F}\u{1}\u{1B}[0m", "  \t \u{2028}", "<|en", "x<|end", "d|>", "<|unregistered|>",
    ]

    static func strings(count: Int, seed: UInt64) -> [String] {
        var rng = SeededGenerator(seed: seed)
        return (0 ..< count).map { _ in
            (0 ..< Int.random(in: 0 ... 20, using: &rng))
                .map { _ in fragments.randomElement(using: &rng)! }
                .joined()
        }
    }
}

struct ScriptedVocabularyTests {

    static func vocabulary(
        corpus: [String],
        specials: [String] = ["<|end|>", "<|user|>"],
        textMarkers: [String] = [],
        configuration: VocabularyBuilder.Configuration = .init()
    ) -> ScriptedVocabulary {
        var builder = VocabularyBuilder(configuration: configuration)
        for special in specials { builder.addSpecial(special) }
        for marker in textMarkers { builder.addSpecial(marker, atomic: false) }
        builder.add(texts: corpus)
        return builder.build()
    }

    static let configurations = [
        VocabularyBuilder.Configuration(),
        VocabularyBuilder.Configuration(maxPieceLength: nil),
        VocabularyBuilder.Configuration(multibyteCharacters: .pieces),
    ]

    static func assertRoundTrip(_ texts: [String], _ vocabulary: ScriptedVocabulary) {
        for text in texts {
            let ids = vocabulary.encode(text)
            #expect(vocabulary.decode(ids) == text, "\(text.debugDescription) -> \(ids)")
            #expect(vocabulary.encode(vocabulary.decode(ids)) == ids)
        }
    }

    @Test(arguments: configurations)
    func `round trip holds for fuzzed strings`(configuration: VocabularyBuilder.Configuration) {
        let corpus = FuzzText.strings(count: 50, seed: 1)
        let vocabulary = Self.vocabulary(
            corpus: corpus, textMarkers: ["<think>"], configuration: configuration)

        // The corpus covers every fragment, so new fuzzed strings add only new
        // combinations: words like "hellokey" that the corpus never saw.
        Self.assertRoundTrip(corpus + FuzzText.strings(count: 500, seed: 2), vocabulary)

        // Characters the fragments never use. Assert they reach multibyte byte fallback,
        // so this keeps testing unseen characters even in `.pieces` mode.
        Self.assertRoundTrip(FuzzText.outsideFragments, vocabulary)
        let fallback = FuzzText.outsideFragments.flatMap(vocabulary.encode).contains {
            vocabulary.entry($0).map { $0.kind == .byte && $0.bytes[0] >= 0x80 } ?? false
        }
        #expect(fallback, "unseen characters should use multibyte byte fallback")
    }

    @Test(arguments: configurations)
    func `round trip holds with an empty corpus`(configuration: VocabularyBuilder.Configuration) {
        let vocabulary = Self.vocabulary(corpus: [], configuration: configuration)
        Self.assertRoundTrip(
            FuzzText.strings(count: 200, seed: 4) + FuzzText.outsideFragments, vocabulary)
    }

    @Test func `specials are never split or absorbed into pieces`() {
        let vocabulary = Self.vocabulary(corpus: ["a<|end|>b", "word<|user|>word", "<|end|>"])

        for entry in vocabulary.entries where entry.kind == .piece {
            #expect(!entry.name.contains("<|"), "piece \(entry.name.debugDescription)")
        }

        let end = vocabulary.specialID("<|end|>")!
        let x = vocabulary.byteBase + Int(UInt8(ascii: "x"))
        let y = vocabulary.byteBase + Int(UInt8(ascii: "y"))
        #expect(vocabulary.encode("x<|end|>y") == [x, end, y])
    }

    @Test func `a piece never extends over the start of a special`() {
        let vocabulary = Self.vocabulary(
            corpus: ["aEO"], specials: ["EOT"],
            configuration: .init(maxPieceLength: nil))

        #expect(vocabulary.id(forToken: "aEO") != nil)
        let a = vocabulary.byteBase + Int(UInt8(ascii: "a"))
        #expect(vocabulary.encode("aEOT") == [a, vocabulary.specialID("EOT")!])
    }

    @Test func `byte fallback covers text outside the corpus`() {
        let thumbs = "👍🏽"
        let bytesOnly = Self.vocabulary(corpus: [thumbs])
        let ids = bytesOnly.encode(thumbs)
        #expect(ids.count == thumbs.utf8.count)
        #expect(ids.allSatisfy { bytesOnly.entry($0)?.kind == .byte })

        let withPieces = Self.vocabulary(
            corpus: [thumbs], configuration: .init(multibyteCharacters: .pieces))
        #expect(withPieces.encode(thumbs).count == thumbs.unicodeScalars.count)
    }

    @Test func `builds are deterministic`() {
        let corpus = FuzzText.strings(count: 50, seed: 3)
        let first = Self.vocabulary(corpus: corpus, textMarkers: ["<think>"])
        let second = Self.vocabulary(corpus: corpus, textMarkers: ["<think>"])
        #expect(first == second)
    }

    @Test func `id layout is stable as the corpus grows`() {
        let small = Self.vocabulary(corpus: ["how are you?"])
        let large = Self.vocabulary(corpus: ["how are you?", "fine, thanks for asking"])

        #expect(small.specialID("<|end|>") == 0)
        #expect(small.specialID("<|user|>") == 1)
        #expect(small.entry(2)?.name == "<|reserved_0|>")
        #expect(small.entry(2)?.kind == .special)
        #expect(small.byteBase == 64)
        #expect(large.entries.starts(with: small.entries))
        #expect(large.size > small.size)
    }

    @Test func `long words are multi-token unless maxPieceLength is nil`() {
        let word = " international"
        #expect(Self.vocabulary(corpus: [word]).encode(word).count > 1)

        let whole = Self.vocabulary(corpus: [word], configuration: .init(maxPieceLength: nil))
        #expect(whole.encode(word).count == 1)
    }

    @Test func `non-atomic markers are multi-token ordinary text`() {
        let vocabulary = Self.vocabulary(corpus: [], textMarkers: ["<think>"])
        let ids = vocabulary.encode("<think>")

        #expect(ids.count >= 2)
        #expect(!ids.contains { vocabulary.isSpecial($0) })
        #expect(vocabulary.decode(ids, skipSpecialTokens: true) == "<think>")
        #expect(vocabulary.specialID("<think>") == nil)
    }
}
