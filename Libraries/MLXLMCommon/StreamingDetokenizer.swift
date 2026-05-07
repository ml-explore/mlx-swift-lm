import Foundation

/// Streams a sequence of token IDs into incrementally emitted text chunks,
/// implementing the upstream Hugging Face `step_decode_stream` algorithm.
///
/// Feed tokens one at a time through ``consume(_:)-(Int)``. Each call returns either
/// a non-empty text chunk that can be displayed, or `nil` to indicate that the
/// detokenizer has buffered the token because it does not yet form a complete
/// scalar. Internal state is bounded: after every successful emission the
/// stored prefix and id buffer are trimmed so memory does not grow with stream
/// length.
///
/// The decoder must be byte-prefix monotonic: `decode(ids + [newId])` must
/// start with `decode(ids)`. Tokenizers that apply post-decode cleanup in
/// `decode` (for example whitespace fixups around contractions and
/// punctuation) can violate that invariant on otherwise-normal output and
/// throw ``TokenizerError/invalidStreamingPrefix(tokenId:expectedPrefix:actualString:)``.
/// Tokenizers that have a raw path should conform to
/// ``StreamingDecodeTokenizer``; this detokenizer uses it automatically when
/// available and falls back to `decode` otherwise.
///
/// `StreamingDetokenizer` is single-consumer by design and is therefore not
/// `Sendable`.
public final class StreamingDetokenizer {
    private let tokenizer: any Tokenizer
    private let skipSpecialTokens: Bool
    internal private(set) var ids: [Int]
    internal private(set) var prefix: String
    internal private(set) var prefixIndex: Int

    /// Creates an empty stream over `tokenizer`.
    public convenience init(tokenizer: any Tokenizer, skipSpecialTokens: Bool = false) {
        self.init(tokenizer: tokenizer, skipSpecialTokens: skipSpecialTokens, initialTokenIds: [])
    }

    /// Creates a stream seeded with prior token IDs. The first ``consume(_:)-(Int)``
    /// uses `initialTokenIds` to establish the prefix and emits only the new
    /// token's chunk. Use this for resuming after interruption.
    public init(
        tokenizer: any Tokenizer,
        skipSpecialTokens: Bool = false,
        initialTokenIds: [Int]
    ) {
        self.tokenizer = tokenizer
        self.skipSpecialTokens = skipSpecialTokens
        ids = initialTokenIds
        prefix = ""
        prefixIndex = 0
    }

    /// Consumes a token and returns any complete chunk it produced.
    ///
    /// - Parameter id: The next token ID to feed into the stream.
    /// - Returns: A non-empty chunk if the buffer can now emit one, or `nil`
    ///   if the buffer ends mid-scalar and the caller should feed the next
    ///   token. The returned chunk is guaranteed to be non-empty.
    /// - Throws: ``TokenizerError/invalidStreamingPrefix(tokenId:expectedPrefix:actualString:)``
    ///   when the decoder produces output that does not begin with the cached
    ///   prefix, or any error propagated from the tokenizer's decode call. On
    ///   any throw, internal state is unchanged from before the call.
    public func consume(_ id: Int) throws -> String? {
        var workingIds = ids
        var workingPrefix = prefix
        var workingPrefixIndex = prefixIndex

        if workingPrefix.isEmpty && !workingIds.isEmpty {
            let seeded = try rawDecode(workingIds)
            if !seeded.hasSuffix("\u{fffd}") {
                workingPrefix = seeded
                workingPrefixIndex = workingIds.count
            }
        }

        workingIds.append(id)
        let string = try rawDecode(workingIds)

        if string.utf8.count <= workingPrefix.utf8.count || string.hasSuffix("\u{fffd}") {
            ids = workingIds
            prefix = workingPrefix
            prefixIndex = workingPrefixIndex
            return nil
        }

        guard string.utf8.starts(with: workingPrefix.utf8) else {
            throw TokenizerError.invalidStreamingPrefix(
                tokenId: id,
                expectedPrefix: workingPrefix,
                actualString: string
            )
        }

        let newChunk = String(
            decoding: string.utf8.dropFirst(workingPrefix.utf8.count),
            as: UTF8.self
        )

        let trimmed = Array(workingIds[workingPrefixIndex...])
        let refreshed = try rawDecode(trimmed)

        ids = trimmed
        prefix = refreshed
        prefixIndex = trimmed.count
        return newChunk
    }

    private func rawDecode(_ tokenIds: [Int]) throws -> String {
        if let raw = tokenizer as? StreamingDecodeTokenizer {
            return try raw.rawDecode(tokenIds: tokenIds, skipSpecialTokens: skipSpecialTokens)
        }
        // TODO: Consider warning once on fallback — Swift Transformers has
        // no cleanup-free decode, so streaming silently inherits its cleanup.
        return try tokenizer.decode(tokenIds: tokenIds, skipSpecialTokens: skipSpecialTokens)
    }

    /// Consumes a batch of tokens. Returns the concatenation of every chunk
    /// produced, or `nil` if no chunk was produced.
    ///
    /// Each token is consumed atomically, but the batch as a whole is not:
    /// if a token mid-batch throws, earlier tokens have already been
    /// committed to internal state.
    public func consume(_ ids: [Int]) throws -> String? {
        var combined = ""
        for id in ids {
            if let chunk = try consume(id) {
                combined.append(chunk)
            }
        }
        return combined.isEmpty ? nil : combined
    }
}

extension Tokenizer {
    /// Returns a fresh ``StreamingDetokenizer`` over this tokenizer.
    public func streamingDetokenizer(skipSpecialTokens: Bool = false) -> StreamingDetokenizer {
        StreamingDetokenizer(tokenizer: self, skipSpecialTokens: skipSpecialTokens)
    }

    /// Returns a ``StreamingDetokenizer`` seeded with `initialTokenIds`.
    public func streamingDetokenizer(
        skipSpecialTokens: Bool = false,
        initialTokenIds: [Int]
    ) -> StreamingDetokenizer {
        StreamingDetokenizer(
            tokenizer: self,
            skipSpecialTokens: skipSpecialTokens,
            initialTokenIds: initialTokenIds
        )
    }
}

@available(
    *, unavailable,
    message:
        "Replaced by StreamingDetokenizer. Use tokenizer.streamingDetokenizer() and try consume(_:) in place of append(token:)/next(). See PR #271."
)
public struct NaiveStreamingDetokenizer {
    public init(tokenizer: any Tokenizer) {}

    @available(
        *, unavailable,
        message: "Use try StreamingDetokenizer.consume(_:) instead of append(token:)/next()."
    )
    public mutating func append(token: Int) {}

    @available(
        *, unavailable,
        message: "Use try StreamingDetokenizer.consume(_:) instead of append(token:)/next()."
    )
    public mutating func next() -> String? { nil }
}
