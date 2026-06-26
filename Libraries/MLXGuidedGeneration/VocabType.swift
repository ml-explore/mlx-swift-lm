// Copyright © 2026 Apple Inc.

import MLXCXGrammar

/// Selects xgrammar's token-decoding path when building a ``GrammarTokenizer``.
///
/// Wraps `CXGrammar`'s `XGVocabType` so callers of this library do not need to
/// import the C shim to construct a tokenizer.
public enum VocabType: Sendable {
    /// Each vocab string is literal UTF-8 bytes (`XG_VOCAB_TYPE_RAW`).
    case raw
    /// SentencePiece `<0xNN>` byte-fallback + `▁` decoding
    /// (`XG_VOCAB_TYPE_BYTE_FALLBACK`).
    case byteFallback
    /// GPT-2 `bytes_to_unicode` byte-level decoding
    /// (`XG_VOCAB_TYPE_BYTE_LEVEL`).
    case byteLevel

    /// The matching `CXGrammar` C enum value.
    var xgVocabType: XGVocabType {
        switch self {
        case .raw: return XG_VOCAB_TYPE_RAW
        case .byteFallback: return XG_VOCAB_TYPE_BYTE_FALLBACK
        case .byteLevel: return XG_VOCAB_TYPE_BYTE_LEVEL
        }
    }
}
