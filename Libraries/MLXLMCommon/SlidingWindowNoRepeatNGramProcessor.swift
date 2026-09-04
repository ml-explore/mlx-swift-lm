// Copyright © 2026 Apple Inc.

import Foundation
import MLX

/// Sliding-window no-repeat n-gram logits processor (DeepSeek-OCR / Unlimited-OCR).
///
/// Port of Baidu / DeepSeek-OCR `NoRepeatNGramLogitsProcessor` /
/// `SlidingWindowNoRepeatNgramProcessor`: within the last `windowSize` tokens,
/// if the current `(ngramSize - 1)`-token prefix already completed an n-gram
/// ending in token `t`, ban `t` by setting its logit to `-inf`.
///
/// **Opt-in.** mlx-vlm and upstream Unlimited-OCR leave this off by default
/// (`no_repeat_ngram_size=0`). Enable by attaching it through
/// ``GenerationComponents/logitProcessorFactory`` (typical Unlimited examples:
/// `ngramSize=35`, `windowSize=128` single-image or `1024` multi-page/PDF):
///
/// ```swift
/// let components = GenerationComponents(
///     logitProcessorFactory: { SlidingWindowNoRepeatNGramProcessor.unlimitedOCRSingleImage() }
/// )
/// ```
public struct SlidingWindowNoRepeatNGramProcessor: LogitProcessor {

    /// Upstream Unlimited-OCR example n-gram size when the guard is enabled.
    public static let unlimitedOCRNgramSize = 35
    /// Upstream single-image example window.
    public static let unlimitedOCRSingleImageWindow = 128
    /// Upstream multi-page / PDF example window.
    public static let unlimitedOCRMultiPageWindow = 1024

    public let ngramSize: Int
    public let windowSize: Int
    public let whitelistTokenIds: Set<Int>

    /// The last `windowSize` prompt + generated tokens, GPU-resident. Only n-grams
    /// inside this window can ban a token, so the window is the whole history the
    /// scan needs (the Python processor slices the same suffix of `input_ids`).
    private var ring: TokenRing
    /// Row `i` gathers the `(ngramSize - 1)`-token prefix of the n-gram that starts
    /// at history position `i`; ``process(logits:)`` slices it to the current length.
    private let prefixIndices: MLXArray
    /// `whitelistTokenIds` as an `int32` vector, or `nil` when empty.
    private let whitelist: MLXArray?

    public init(
        ngramSize: Int,
        windowSize: Int = SlidingWindowNoRepeatNGramProcessor.unlimitedOCRSingleImageWindow,
        whitelistTokenIds: Set<Int> = []
    ) {
        precondition(ngramSize > 0, "ngramSize must be a strictly positive integer")
        precondition(windowSize > 0, "windowSize must be a strictly positive integer")
        self.ngramSize = ngramSize
        self.windowSize = windowSize
        self.whitelistTokenIds = whitelistTokenIds
        self.ring = TokenRing(capacity: windowSize)
        let ngramStarts = MLXArray.arange(max(0, windowSize - ngramSize + 1))
        self.prefixIndices =
            ngramStarts[0..., .newAxis] + MLXArray.arange(ngramSize - 1)[.newAxis, 0...]
        self.whitelist =
            whitelistTokenIds.isEmpty
            ? nil : MLXArray(whitelistTokenIds.sorted().map { Int32($0) })
    }

    /// Convenience for Unlimited-OCR single-image example settings (`n=35`, window `128`).
    public static func unlimitedOCRSingleImage(
        whitelistTokenIds: Set<Int> = []
    ) -> SlidingWindowNoRepeatNGramProcessor {
        SlidingWindowNoRepeatNGramProcessor(
            ngramSize: unlimitedOCRNgramSize,
            windowSize: unlimitedOCRSingleImageWindow,
            whitelistTokenIds: whitelistTokenIds
        )
    }

    /// Convenience for Unlimited-OCR multi-page/PDF example settings (`n=35`, window `1024`).
    public static func unlimitedOCRMultiPage(
        whitelistTokenIds: Set<Int> = []
    ) -> SlidingWindowNoRepeatNGramProcessor {
        SlidingWindowNoRepeatNGramProcessor(
            ngramSize: unlimitedOCRNgramSize,
            windowSize: unlimitedOCRMultiPageWindow,
            whitelistTokenIds: whitelistTokenIds
        )
    }

    public mutating func prompt(_ prompt: MLXArray) {
        ring.loadPrompt(prompt)
    }

    /// Bans the n-gram completions entirely on the GPU.
    ///
    /// Nothing here reads a token back to the CPU: the history, the prefix match
    /// and the scatter of `-inf` are lazy ops on the same stream as the logits, so
    /// the sampled token `TokenIterator` hands to ``didSample(token:)`` stays in
    /// flight and the one-step pipeline lookahead is preserved. The only host-side
    /// value used is the ring's element count.
    public func process(logits: MLXArray) -> MLXArray {
        let historyLength = ring.count
        guard historyLength >= ngramSize, let history = ring.orderedTokens else {
            return logits
        }

        // The n-gram starting at position i has prefix history[i ..< i + n - 1] and
        // completion history[i + n - 1]; the current prefix is the last n - 1 tokens.
        let ngramCount = historyLength - ngramSize + 1
        let prefixes = history[prefixIndices[..<ngramCount]]
        let currentPrefix = history[ngramCount...]
        let completions = history[(ngramSize - 1)...]
        var matches = (prefixes .== currentPrefix).all(axis: 1)
        if let whitelist {
            let whitelisted = (completions[0..., .newAxis] .== whitelist[.newAxis, 0...])
                .any(axis: 1)
            matches = logicalAnd(matches, logicalNot(whitelisted))
        }

        // A token may complete several matching n-grams: accumulate with scatter-add
        // (well-defined for duplicate indices) rather than writing -inf per match.
        let banned =
            MLXArray.zeros([logits.dim(-1)], type: Int32.self)
            .at[completions].add(matches.asType(.int32)) .> 0
        let negInf = MLXArray(-Float.infinity).asType(logits.dtype)
        return which(banned[.newAxis, 0...], negInf, logits)
    }

    public mutating func didSample(token: MLXArray) {
        ring.append(token)
    }

    /// CPU reference of the ban rule that ``process(logits:)`` evaluates on the GPU:
    /// tokens that would complete a repeated n-gram given `history`. Used by the
    /// equivalence tests.
    public func bannedTokens(in history: [Int]) -> Set<Int> {
        guard history.count >= ngramSize else { return [] }

        let prefixLen = ngramSize - 1
        let currentPrefix = Array(history.suffix(prefixLen))
        let searchStart = max(0, history.count - windowSize)
        let searchEnd = history.count - ngramSize + 1

        var banned = Set<Int>()
        if searchStart < searchEnd {
            for i in searchStart ..< searchEnd {
                let ngram = Array(history[i ..< (i + ngramSize)])
                if Array(ngram.dropLast()) == currentPrefix {
                    banned.insert(ngram[ngramSize - 1])
                }
            }
        }
        return banned.subtracting(whitelistTokenIds)
    }
}
