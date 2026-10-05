// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN

package func draftMTPTokenBlock(
    targetEmbedTokens: Embedding,
    lmHead: Linear?,
    inputEmbedding: Embedding,
    lastToken: MLXArray,
    lastHidden: MLXArray,
    queryOffset: Int,
    blockSize: Int,
    sampler: any LogitSampler,
    cache: [KVCache],
    forward: (
        _ inputsEmbeds: MLXArray, _ hiddenStates: MLXArray, _ cache: [KVCache],
        _ positionOffset: Int
    ) -> MLXArray
) -> MLXArray {
    precondition(blockSize >= 2, "blockSize must be >= 2")

    var tok = lastToken.ndim == 1 ? lastToken.reshaped([lastToken.dim(0), 1]) : lastToken
    var hidden = lastHidden
    precondition(!cache.isEmpty, "Qwen MTP drafter cache must not be empty")
    var tokens: [MLXArray] = []
    tokens.reserveCapacity(blockSize - 1)

    for stepIndex in 0 ..< (blockSize - 1) {
        let mtpHidden = forward(
            inputEmbedding(tok),
            hidden,
            cache,
            queryOffset + stepIndex
        )
        hidden = mtpHidden

        let logits: MLXArray
        if let lmHead {
            logits = lmHead(mtpHidden)
        } else {
            logits = targetEmbedTokens.asLinear(mtpHidden)
        }

        let next = sampler.sample(logits: logits[0..., -1, 0...])
        tok = next.ndim == 1 ? next.reshaped([next.dim(0), 1]) : next
        tokens.append(tok)
    }

    return concatenated(tokens, axis: 1)
}

package func normalizedMTPTokenBatch(_ tokens: MLXArray) -> MLXArray {
    switch tokens.ndim {
    case 1:
        return tokens[.newAxis, 0...]
    default:
        return tokens
    }
}

package func normalizedMTPColumn(_ tokens: MLXArray) -> MLXArray {
    let tokens = normalizedMTPTokenBatch(tokens)
    return tokens.dim(-1) == 1 ? tokens : tokens[0..., (-1)...]
}

package func sampleMTPSeed(
    hidden: MLXArray,
    targetEmbedTokens: Embedding,
    lmHead: Linear?,
    sampler: any LogitSampler
) -> MLXArray {
    let logits = lmHead.map { $0(hidden) } ?? targetEmbedTokens.asLinear(hidden)
    let sampled = sampler.sample(logits: logits[0..., -1, 0...])
    return normalizedMTPColumn(sampled)
}
