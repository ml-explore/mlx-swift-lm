// Copyright © 2025 Apple Inc.

import MLXLMCommon

/// Extracts vocabulary byte data from a HuggingFace Tokenizer.
///
/// Two vocab shapes are exposed:
/// - `extract(from:)` returns a packed `(tokenBytes, tokenLens)` buffer
///   useful for testing that the per-token byte decoding agrees with
///   the tokenizer's own `decode(ids)` output.
/// - `extractForGrammar(from:)` returns the raw per-token piece strings
///   plus a detected `VocabType`, which xgrammar consumes directly.
///
/// Three token-model conventions are normalized by `tokenToBytes` (used
/// by the packed-buffer path):
/// - **SentencePiece space marker** `\u{2581}` (LOWER ONE EIGHTH BLOCK) ->
///   ASCII space `0x20`.
/// - **SentencePiece byte-fallback** `<0xNN>` -> the literal byte.
/// - **GPT-2-style BPE byte-to-unicode mapping** (used by Qwen, Llama,
///   Mistral-family, etc.): the vocab stores bytes that can't appear
///   literally in a string (controls, space, some punctuation) as mapped
///   codepoints. e.g. `\n` (`0x0A`) is stored as `Ċ` (`U+010A`); space is
///   stored as `Ġ` (`U+0120`). `bpeUnicodeToByte` reverses that mapping.
///   Identity-mapped Latin-1 printables (`0x21-0x7E`, `0xA1-0xAC`,
///   `0xAE-0xFF`) pass through unchanged, so SentencePiece tokens that
///   happen to share the identity range are unaffected.
public enum TokenizerVocabExtractor {

    struct VocabData {
        let tokenBytes: [UInt8]
        let tokenLens: [UInt32]
        let eosToken: UInt32
        let vocabSize: Int
    }

    /// Extract vocabulary bytes from a Tokenizer.
    ///
    /// Iterates through token IDs, decoding each to get its string representation,
    /// then converts to UTF-8 bytes. Handles SentencePiece conventions:
    /// - Replaces `\u{2581}` with ASCII space (0x20)
    /// - Decodes `<0xNN>` byte-fallback tokens to their literal byte value
    static func extract(from tokenizer: any Tokenizer) -> VocabData {
        let eosToken = UInt32(tokenizer.eosTokenId ?? 0)

        // Discover vocab size by scanning token IDs
        var vocabSize = 0
        while tokenizer.convertIdToToken(vocabSize) != nil {
            vocabSize += 1
            if vocabSize > 500_000 { break }  // safety limit
        }

        var allBytes: [UInt8] = []
        var lens: [UInt32] = []
        allBytes.reserveCapacity(vocabSize * 4)  // rough estimate
        lens.reserveCapacity(vocabSize)

        for id in 0 ..< vocabSize {
            if let token = tokenizer.convertIdToToken(id) {
                let bytes = tokenToBytes(token)
                allBytes.append(contentsOf: bytes)
                lens.append(UInt32(bytes.count))
            } else {
                // Gaps in vocab: use empty token
                lens.append(0)
            }
        }

        return VocabData(
            tokenBytes: allBytes,
            tokenLens: lens,
            eosToken: eosToken,
            vocabSize: vocabSize
        )
    }

    /// Vocab data in the shape xgrammar's `TokenizerInfo` expects:
    /// one piece string per token id, plus a `VocabType` selecting
    /// xgrammar's in-process decoder.
    ///
    /// xgrammar applies the SentencePiece or GPT-2 byte-level decoding
    /// itself based on `vocabType`, so unlike `extract(from:)` this
    /// helper hands over the raw piece strings (`<0xNN>` byte-fallback
    /// tokens, `▁`-prefixed SentencePiece pieces, `Ġ`/`Ċ`-mapped BPE
    /// pieces) unmodified. Pre-normalizing here would duplicate
    /// xgrammar's decoding path and lose fidelity for non-UTF-8 raw
    /// bytes when transporting through Swift `String`.
    public struct GrammarVocab {
        public let vocab: [String]
        public let vocabType: VocabType
    }

    /// Extract vocabulary for xgrammar.
    ///
    /// Detects the tokenizer family by scanning a bounded sample of
    /// tokens:
    /// - any `<0xNN>` byte-fallback piece -> `XG_VOCAB_TYPE_BYTE_FALLBACK`
    /// - any codepoint in the GPT-2 byte-to-unicode extended range
    ///   (`U+0100`-`U+0143`) -> `XG_VOCAB_TYPE_BYTE_LEVEL`
    /// - otherwise -> `XG_VOCAB_TYPE_RAW`
    ///
    /// Detection is intentionally a scan of the full vocab (not the
    /// first few tokens) so tokenizers that sprinkle byte-fallback
    /// tokens beyond the ASCII prefix are still classified correctly.
    /// The cost is one pass at construction time, which is negligible
    /// next to xgrammar's own vocab-processing work.
    public static func extractForGrammar(from tokenizer: any Tokenizer) -> GrammarVocab {
        var vocabSize = 0
        while tokenizer.convertIdToToken(vocabSize) != nil {
            vocabSize += 1
            if vocabSize > 500_000 { break }  // safety limit
        }

        var vocab: [String] = []
        vocab.reserveCapacity(vocabSize)

        var sawByteFallback = false
        var sawByteLevelScalar = false

        for id in 0 ..< vocabSize {
            let token = tokenizer.convertIdToToken(id) ?? ""
            vocab.append(token)

            if !sawByteFallback, isByteFallbackToken(token) {
                sawByteFallback = true
            }
            if !sawByteLevelScalar, containsByteLevelScalar(token) {
                sawByteLevelScalar = true
            }
        }

        let vocabType: VocabType
        if sawByteFallback {
            vocabType = .byteFallback
        } else if sawByteLevelScalar {
            vocabType = .byteLevel
        } else {
            vocabType = .raw
        }

        return GrammarVocab(vocab: vocab, vocabType: vocabType)
    }

    /// True for SentencePiece `<0xNN>` byte-fallback piece strings.
    private static func isByteFallbackToken(_ token: String) -> Bool {
        guard token.count == 6,
            token.hasPrefix("<0x"),
            token.hasSuffix(">")
        else {
            return false
        }
        return UInt8(token.dropFirst(3).dropLast(), radix: 16) != nil
    }

    /// True if any scalar of `token` falls in the GPT-2
    /// `bytes_to_unicode` extended codepoint range (`U+0100`-`U+0143`).
    /// These codepoints only appear in byte-level BPE tokenizers, so
    /// any sighting is decisive.
    private static func containsByteLevelScalar(_ token: String) -> Bool {
        for scalar in token.unicodeScalars {
            if scalar.value >= 0x100 && scalar.value <= 0x143 {
                return true
            }
        }
        return false
    }

    /// Convert a token piece string to its actual decoded byte representation.
    ///
    /// Handles (in order):
    /// 1. `<0xNN>` SentencePiece byte-fallback -> single byte with value `0xNN`.
    /// 2. SentencePiece space marker `\u{2581}` -> ASCII space.
    /// 3. GPT-2 BPE byte-to-unicode: each Unicode scalar in the remaining
    ///    string is mapped back to its original byte through
    ///    `bpeUnicodeToByte`. Scalars outside the mapping (e.g. a multi-byte
    ///    Unicode char in a SentencePiece tokenizer's piece text) fall back
    ///    to the scalar's UTF-8 encoding.
    ///
    /// `WhitespaceTokenBias` (in MLXLMCommon) inlines an identical helper so
    /// the bias's whitespace classification agrees with what this extractor
    /// reports as a token's "real bytes".
    static func tokenToBytes(_ token: String) -> [UInt8] {
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
    /// - `U+010A` (`Ċ`) -> byte `0x0A` (`\n`)
    /// - `U+0120` (`Ġ`) -> byte `0x20` (space)
    /// - `U+0121` (`ġ`) -> byte `0x7F` (DEL)
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
