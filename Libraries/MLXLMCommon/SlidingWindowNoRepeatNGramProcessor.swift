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

    /// Full prompt + generated token history (CPU), matching transformers'
    /// `input_ids` passed to the Python logits processor each step.
    private var tokens: [Int] = []

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
        tokens = prompt.asArray(Int.self)
    }

    public func process(logits: MLXArray) -> MLXArray {
        let banned = bannedTokens(in: tokens)
        guard !banned.isEmpty else { return logits }

        let bannedList = Array(banned)
        let indices = MLXArray(bannedList.map { Int32($0) }).asType(.uint32)[.newAxis, 0...]
        let negInf = MLXArray(Array(repeating: -Float.infinity as Float, count: bannedList.count))[
            .newAxis, 0...
        ]
        // Match dtype of incoming logits (bf16 decode path is common).
        return putAlong(logits, indices, values: negInf.asType(logits.dtype), axis: -1)
    }

    public mutating func didSample(token: MLXArray) {
        tokens.append(token.item(Int.self))
    }

    /// Tokens that would complete a repeated n-gram given the current history.
    /// Exposed for unit tests.
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
