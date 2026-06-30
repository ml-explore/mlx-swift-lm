// Copyright © 2026 Apple Inc.

import MLX
import MLXLMCommon

/// Utility that identifies whitespace-only tokens in a tokenizer's vocabulary
/// and produces a negative logit bias array.
///
/// Classification decodes each token through a private `tokenToBytes` helper
/// so that BPE-encoded whitespace (e.g. Qwen's `Ċ` for `\n`, `Ġ` for space),
/// SentencePiece space markers, and byte-fallback whitespace all classify
/// correctly.
public enum WhitespaceTokenBias {

    // MARK: - Constants

    private static let biasMagnitude: Float = -200.0

    /// Byte values that are JSON whitespace: tab, newline, carriage return, space.
    private static let whitespaceByteCodes: Set<UInt8> = [0x09, 0x0A, 0x0D, 0x20]

    // MARK: - Public API

    /// Returns an MLXArray of shape [vocabSize] with -200.0 for whitespace-only
    /// tokens and 0.0 for all others, plus the set of whitespace token IDs.
    public static func compute(tokenizer: any Tokenizer) -> (bias: MLXArray, tokenIDs: Set<Int>) {
        // Discover vocab size by scanning token IDs
        var vocabSize = 0
        while tokenizer.convertIdToToken(vocabSize) != nil {
            vocabSize += 1
            if vocabSize > 500_000 { break }
        }

        var biases = [Float](repeating: 0.0, count: vocabSize)
        var whitespaceIDs = Set<Int>()

        for id in 0 ..< vocabSize {
            if let token = tokenizer.convertIdToToken(id),
                isWhitespaceOnly(token)
            {
                biases[id] = biasMagnitude
                whitespaceIDs.insert(id)
            }
        }

        return (MLXArray(biases), whitespaceIDs)
    }

    // MARK: - Private

    /// A token is "whitespace-only" if every byte of its decoded form is
    /// JSON whitespace. Decoding goes through the same path as the vocab
    /// extractor so BPE/SentencePiece encodings are handled uniformly.
    private static func isWhitespaceOnly(_ token: String) -> Bool {
        let bytes = tokenToBytes(token)
        guard !bytes.isEmpty else { return false }
        return bytes.allSatisfy { whitespaceByteCodes.contains($0) }
    }

    /// Convert a token piece string to its actual decoded byte representation.
    ///
    /// Handles (in order):
    /// 1. `<0xNN>` SentencePiece byte-fallback → single byte with value `0xNN`.
    /// 2. SentencePiece space marker `\u{2581}` → ASCII space.
    /// 3. GPT-2 BPE byte-to-unicode: each Unicode scalar in the remaining
    ///    string is mapped back to its original byte through
    ///    `bpeUnicodeToByte`. Scalars outside the mapping (e.g. a multi-byte
    ///    Unicode char in a SentencePiece tokenizer's piece text) fall back
    ///    to the scalar's UTF-8 encoding.
    private static func tokenToBytes(_ token: String) -> [UInt8] {
        // SentencePiece byte-fallback: <0x00> through <0xFF>
        if token.count == 6,
            token.hasPrefix("<0x"),
            token.hasSuffix(">"),
            let byte = UInt8(token.dropFirst(3).dropLast(), radix: 16)
        {
            return [byte]
        }

        // Replace SentencePiece space marker with real space
        let normalized = token.replacingOccurrences(of: "\u{2581}", with: " ")

        // BPE inverse: each scalar either maps back to a byte, or falls
        // through as UTF-8. Identity scalars (Latin-1 printables) map to
        // their own byte value, so SentencePiece Unicode text passes
        // through unchanged.
        var bytes: [UInt8] = []
        bytes.reserveCapacity(normalized.utf8.count)
        for scalar in normalized.unicodeScalars {
            if let byte = bpeUnicodeToByte[scalar.value] {
                bytes.append(byte)
            } else {
                bytes.append(contentsOf: String(scalar).utf8)
            }
        }
        return bytes
    }

    /// HuggingFace `bytes_to_unicode()` map, inverted.
    ///
    /// Shape: `[codepoint: byte]`. Covers all 256 single-byte values.
    /// 223 of them are identity-mapped (printable Latin-1 ranges); the
    /// remaining 33 control/whitespace bytes are mapped to codepoints
    /// `U+0100` through `U+0120` in iteration order.
    ///
    /// Examples:
    /// - `U+010A` (`Ċ`) → byte `0x0A` (`\n`)
    /// - `U+0120` (`Ġ`) → byte `0x20` (space)
    /// - `U+0121` (`ġ`) → byte `0x7F` (DEL)
    ///
    /// Identity mapping covers `0x21-0x7E`, `0xA1-0xAC`, `0xAE-0xFF`.
    private static let bpeUnicodeToByte: [UInt32: UInt8] = {
        var map: [UInt32: UInt8] = [:]
        map.reserveCapacity(256)
        var extendedCodepoint: UInt32 = 0x100
        for b in 0 ..< 256 {
            let isIdentity =
                (b >= 0x21 && b <= 0x7E)
                || (b >= 0xA1 && b <= 0xAC)
                || (b >= 0xAE && b <= 0xFF)
            if isIdentity {
                map[UInt32(b)] = UInt8(b)
            } else {
                map[extendedCodepoint] = UInt8(b)
                extendedCodepoint += 1
            }
        }
        return map
    }()
}
