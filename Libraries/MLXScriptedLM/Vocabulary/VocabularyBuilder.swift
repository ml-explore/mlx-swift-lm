// Copyright © 2026 Apple Inc.

import Foundation

/// Builds a ``ScriptedVocabulary`` from a scenario corpus.
///
/// Register specials and add every string a scenario will encode, then call ``build()``.
/// Text is pretokenized at build time, after all specials are known, so special text is
/// never absorbed into a corpus piece.
///
/// Pretokenization (ASCII-oriented, deterministic):
/// - a word is an optional leading space plus a run of ASCII letters and digits
/// - a whitespace run is its own piece
/// - ASCII punctuation is left to byte fallback
/// - non-ASCII characters follow ``Configuration/multibyteCharacters``
///
/// Words and whitespace runs are chunked to ``Configuration/maxPieceLength`` bytes so
/// long words are always multi-token. Single-byte pieces are dropped: byte tokens cover them.
public struct VocabularyBuilder: Sendable {

    public struct Configuration: Sendable, Equatable {
        public enum MultibyteCharacters: Sendable, Equatable {
            /// Always byte fallback, so non-ASCII text exercises partial UTF-8 decoding.
            case bytes
            /// Each non-ASCII scalar becomes its own piece.
            case pieces
        }

        /// Maximum piece length in bytes. `nil` keeps whole words.
        public var maxPieceLength: Int?
        /// Ids reserved for atomic specials, below all other tokens.
        public var reservedSpecialCount: Int
        public var multibyteCharacters: MultibyteCharacters

        public init(
            maxPieceLength: Int? = 4,
            reservedSpecialCount: Int = 64,
            multibyteCharacters: MultibyteCharacters = .bytes
        ) {
            precondition(maxPieceLength.map { $0 >= 2 } ?? true, "maxPieceLength must be >= 2")
            self.maxPieceLength = maxPieceLength
            self.reservedSpecialCount = reservedSpecialCount
            self.multibyteCharacters = multibyteCharacters
        }
    }

    public let configuration: Configuration
    private var specials: [String] = []
    private var textMarkers: [String] = []
    private var texts: [String] = []

    public init(configuration: Configuration = .init()) {
        self.configuration = configuration
    }

    /// Registers a marker such as `<|end|>` or `<think>`.
    ///
    /// Atomic markers become single special tokens: never split, and dropped by
    /// `decode(skipSpecialTokens: true)`. Non-atomic markers are ordinary text that must
    /// encode to two or more tokens, like multi-token markers in some real model families.
    public mutating func addSpecial(_ token: String, atomic: Bool = true) {
        precondition(!token.isEmpty, "special token must not be empty")
        if atomic {
            precondition(
                !textMarkers.contains(token), "\(token) is already registered as non-atomic")
            if !specials.contains(token) {
                specials.append(token)
            }
        } else {
            precondition(!specials.contains(token), "\(token) is already registered as atomic")
            if !textMarkers.contains(token) {
                textMarkers.append(token)
            }
        }
    }

    public mutating func add(text: String) {
        texts.append(text)
    }

    public mutating func add(texts: some Sequence<String>) {
        self.texts.append(contentsOf: texts)
    }

    public func build() -> ScriptedVocabulary {
        let reserved = configuration.reservedSpecialCount
        precondition(
            specials.count <= reserved,
            "\(specials.count) specials exceed reservedSpecialCount \(reserved)")

        var entries: [ScriptedVocabulary.Entry] = []
        var specialNames = specials
        var filler = 0
        while specialNames.count < reserved {
            let name = "<|reserved_\(filler)|>"
            filler += 1
            if !specials.contains(name) {
                specialNames.append(name)
            }
        }
        for name in specialNames {
            entries.append(
                .init(id: entries.count, kind: .special, bytes: Array(name.utf8), name: name))
        }

        for value in 0 ... 255 {
            let byte = UInt8(value)
            entries.append(
                .init(
                    id: entries.count, kind: .byte, bytes: [byte],
                    name: ScriptedVocabulary.byteName(byte)))
        }

        var seen = Set<[UInt8]>()
        let specialBytes = specialNames.map { Array($0.utf8) }
        for text in texts + textMarkers {
            for segment in Self.split(text, around: specialBytes) {
                for piece in pretokenize(segment) where seen.insert(piece).inserted {
                    entries.append(
                        .init(
                            id: entries.count, kind: .piece, bytes: piece,
                            name: String(decoding: piece, as: UTF8.self)))
                }
            }
        }

        let vocabulary = ScriptedVocabulary(
            entries: entries, reservedSpecialCount: reserved, textMarkers: textMarkers)
        for marker in textMarkers {
            precondition(
                vocabulary.encode(marker).count >= 2,
                "non-atomic marker \(marker) encodes to a single token; use a longer marker")
        }
        return vocabulary
    }

    // MARK: - Pretokenization

    /// Splits `text` into the byte runs between occurrences of any special.
    static func split(_ text: String, around specials: [[UInt8]]) -> [[UInt8]] {
        let bytes = Array(text.utf8)
        let ordered = specials.sorted { $0.count > $1.count }
        var segments: [[UInt8]] = []
        var current: [UInt8] = []
        var i = 0
        outer: while i < bytes.count {
            for special in ordered where special.count <= bytes.count - i {
                if Array(bytes[i ..< i + special.count]) == special {
                    segments.append(current)
                    current = []
                    i += special.count
                    continue outer
                }
            }
            current.append(bytes[i])
            i += 1
        }
        segments.append(current)
        return segments.filter { !$0.isEmpty }
    }

    func pretokenize(_ bytes: [UInt8]) -> [[UInt8]] {
        let scalars = Array(String(decoding: bytes, as: UTF8.self).unicodeScalars)
        var pieces: [[UInt8]] = []

        func isWordByte(_ s: Unicode.Scalar) -> Bool {
            s.isASCII && (s.properties.isAlphabetic || ("0" ... "9").contains(s))
        }
        func isSpace(_ s: Unicode.Scalar) -> Bool {
            s == " " || s == "\n" || s == "\t" || s == "\r"
        }
        func startsWord(_ i: Int) -> Bool {
            scalars[i] == " " && i + 1 < scalars.count && isWordByte(scalars[i + 1])
        }

        var i = 0
        while i < scalars.count {
            let s = scalars[i]
            if isWordByte(s) || startsWord(i) {
                var word: [UInt8] = []
                if s == " " {
                    word.append(0x20)
                    i += 1
                }
                while i < scalars.count && isWordByte(scalars[i]) {
                    word.append(UInt8(scalars[i].value))
                    i += 1
                }
                pieces += chunk(word)
            } else if isSpace(s) {
                var run: [UInt8] = []
                while i < scalars.count && isSpace(scalars[i]) && !startsWord(i) {
                    run.append(UInt8(scalars[i].value))
                    i += 1
                }
                pieces += chunk(run)
            } else {
                if !s.isASCII && configuration.multibyteCharacters == .pieces {
                    pieces.append(Array(String(s).utf8))
                }
                i += 1
            }
        }
        return pieces
    }

    private func chunk(_ bytes: [UInt8]) -> [[UInt8]] {
        guard let size = configuration.maxPieceLength else {
            return bytes.count >= 2 ? [bytes] : []
        }
        return stride(from: 0, to: bytes.count, by: size)
            .map { Array(bytes[$0 ..< Swift.min($0 + size, bytes.count)]) }
            .filter { $0.count >= 2 }
    }
}
