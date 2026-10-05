// Copyright © 2026 Apple Inc.

import Foundation

/// A deterministic vocabulary for scripted tests.
///
/// Id layout:
/// - `0 ..< reservedSpecialCount`: atomic special tokens, then `<|reserved_N|>` fillers
/// - next 256 ids: one byte-fallback token per byte value
/// - remaining ids: corpus pieces, in first-seen order
///
/// Every entry decodes to a fixed byte string, so `decode(encode(s)) == s` for any `s`.
public struct ScriptedVocabulary: Sendable, Equatable {

    package enum Kind: Sendable, Equatable {
        case special
        case byte
        case piece
    }

    package struct Entry: Sendable, Equatable {
        package let id: Int
        package let kind: Kind
        /// Bytes this token contributes to decoded text.
        package let bytes: [UInt8]
        /// Name returned by `convertIdToToken`.
        package let name: String
    }

    package let entries: [Entry]
    package let reservedSpecialCount: Int
    /// Markers registered as ordinary text (multi-token), in registration order.
    package let textMarkers: [String]

    private let specialIDs: [String: Int]
    /// Specials sorted longest first, so the longest special wins at a position.
    private let specialsByLength: [(bytes: [UInt8], id: Int)]
    private let pieceIDs: [[UInt8]: Int]
    private let maxPieceLength: Int

    public var size: Int { entries.count }
    package var byteBase: Int { reservedSpecialCount }

    init(entries: [Entry], reservedSpecialCount: Int, textMarkers: [String]) {
        self.entries = entries
        self.reservedSpecialCount = reservedSpecialCount
        self.textMarkers = textMarkers

        var specialIDs: [String: Int] = [:]
        var pieceIDs: [[UInt8]: Int] = [:]
        var specials: [(bytes: [UInt8], id: Int)] = []
        for entry in entries {
            switch entry.kind {
            case .special:
                specialIDs[entry.name] = entry.id
                specials.append((entry.bytes, entry.id))
            case .piece:
                pieceIDs[entry.bytes] = entry.id
            case .byte:
                break
            }
        }
        self.specialIDs = specialIDs
        self.pieceIDs = pieceIDs
        self.specialsByLength = specials.sorted {
            $0.bytes.count != $1.bytes.count ? $0.bytes.count > $1.bytes.count : $0.id < $1.id
        }
        self.maxPieceLength = pieceIDs.keys.map(\.count).max() ?? 1
    }

    public static func == (lhs: ScriptedVocabulary, rhs: ScriptedVocabulary) -> Bool {
        lhs.entries == rhs.entries && lhs.reservedSpecialCount == rhs.reservedSpecialCount
            && lhs.textMarkers == rhs.textMarkers
    }

    // MARK: - Lookup

    package func entry(_ id: Int) -> Entry? {
        entries.indices.contains(id) ? entries[id] : nil
    }

    public func isSpecial(_ id: Int) -> Bool {
        entry(id)?.kind == .special
    }

    /// Id of an atomic special token, or `nil`.
    public func specialID(_ token: String) -> Int? {
        specialIDs[token]
    }

    /// Id of a token by name: a special, a piece, or a byte (`"a"` or `"<0x0A>"`).
    public func id(forToken token: String) -> Int? {
        if let id = specialIDs[token] {
            return id
        }
        let bytes = Array(token.utf8)
        if let id = pieceIDs[bytes] {
            return id
        }
        if bytes.count == 1 {
            return byteBase + Int(bytes[0])
        }
        if let value = Self.parseByteName(token) {
            return byteBase + Int(value)
        }
        return nil
    }

    // MARK: - Encode / decode

    /// Longest-match encoding. Specials are matched first at every position, and a piece
    /// never extends over the start of a special. Unmatched bytes use byte fallback.
    public func encode(_ text: String) -> [Int] {
        let bytes = Array(text.utf8)
        var ids: [Int] = []
        var i = 0
        while i < bytes.count {
            if let (length, id) = special(at: i, in: bytes) {
                ids.append(id)
                i += length
                continue
            }

            var limit = Swift.min(maxPieceLength, bytes.count - i)
            if let next = nextSpecialStart(after: i, within: limit, in: bytes) {
                limit = next - i
            }

            var matched = false
            if limit >= 2 {
                for length in stride(from: limit, through: 2, by: -1) {
                    if let id = pieceIDs[Array(bytes[i ..< i + length])] {
                        ids.append(id)
                        i += length
                        matched = true
                        break
                    }
                }
            }
            if !matched {
                ids.append(byteBase + Int(bytes[i]))
                i += 1
            }
        }
        return ids
    }

    /// Concatenates token bytes. Invalid UTF-8 (a partial character) decodes to U+FFFD.
    /// Ids outside the vocabulary are ignored.
    public func decode(_ ids: [Int], skipSpecialTokens: Bool = false) -> String {
        var bytes: [UInt8] = []
        for id in ids {
            guard let entry = entry(id) else { continue }
            if skipSpecialTokens && entry.kind == .special { continue }
            bytes.append(contentsOf: entry.bytes)
        }
        return String(decoding: bytes, as: UTF8.self)
    }

    // MARK: - Private

    private func special(at i: Int, in bytes: [UInt8]) -> (Int, Int)? {
        for special in specialsByLength where matches(special.bytes, at: i, in: bytes) {
            return (special.bytes.count, special.id)
        }
        return nil
    }

    private func nextSpecialStart(after i: Int, within limit: Int, in bytes: [UInt8]) -> Int? {
        guard limit >= 2 else { return nil }
        for j in (i + 1) ..< (i + limit) where special(at: j, in: bytes) != nil {
            return j
        }
        return nil
    }

    private func matches(_ pattern: [UInt8], at i: Int, in bytes: [UInt8]) -> Bool {
        guard i + pattern.count <= bytes.count else { return false }
        for k in 0 ..< pattern.count where bytes[i + k] != pattern[k] {
            return false
        }
        return true
    }

    static func byteName(_ value: UInt8) -> String {
        (0x20 ... 0x7E).contains(value)
            ? String(UnicodeScalar(value)) : String(format: "<0x%02X>", value)
    }

    private static func parseByteName(_ token: String) -> UInt8? {
        guard token.count == 6, token.hasPrefix("<0x"), token.hasSuffix(">") else {
            return nil
        }
        return UInt8(token.dropFirst(3).prefix(2), radix: 16)
    }
}
