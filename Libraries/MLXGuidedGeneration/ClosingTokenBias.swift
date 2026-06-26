// Copyright © 2025 Apple Inc.

import MLX
import MLXLMCommon

/// Utility that identifies JSON-closing tokens in a tokenizer's vocabulary
/// and produces a logit bias array.
public enum ClosingTokenBias {

    // MARK: - Constants

    private static let tier1Bias: Float = 200.0
    private static let tier2Bias: Float = 100.0

    private static let tier2Characters: Set<String> = [
        "\"", "}", "]",
        "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    ]

    // MARK: - Public API

    /// Returns an MLXArray of shape [vocabSize]. Closing tokens get a large
    /// positive value (tiered by priority), all others get 0.0.
    ///
    /// Tier 1 (+200): EOS token
    /// Tier 2 (+100): `"`, `}`, `]`, single digits `0`-`9`
    public static func compute(tokenizer: any Tokenizer, eosTokenId: Int?) -> MLXArray {
        // Discover vocab size by scanning token IDs
        var vocabSize = 0
        while tokenizer.convertIdToToken(vocabSize) != nil {
            vocabSize += 1
            if vocabSize > 500_000 { break }
        }

        var biases = [Float](repeating: 0.0, count: vocabSize)

        for id in 0 ..< vocabSize {
            if let token = tokenizer.convertIdToToken(id),
                tier2Characters.contains(token)
            {
                biases[id] = tier2Bias
            }
        }

        // Tier 1 applied last so it overrides tier 2 if EOS overlaps
        if let eos = eosTokenId, eos >= 0, eos < vocabSize {
            biases[eos] = tier1Bias
        }

        return MLXArray(biases)
    }
}
