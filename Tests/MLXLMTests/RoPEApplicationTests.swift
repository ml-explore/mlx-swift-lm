import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

@Suite(.serialized)
struct RoPEApplicationTests {

    /// Gemma3nAttention applies rope, updates the cache and applies rope again.
    /// Ensure that it is correctly implemented.  We can observe prefill vs single token
    /// generaton and they should produce the same answers if implemented correctly.
    @Test func gemma3nAttentionTest() {
        let config = Gemma3nTextConfiguration()
        let attention = Gemma3nAttention(config, layerIdx: 3)

        #expect(!attention.isKvSharedLayer)

        let B = 1
        let L = 4
        let D = config.hiddenSize

        MLXRandom.seed(42)
        let x = MLXRandom.normal([B, L, D])
        eval(x)

        // Batch: process all L tokens at once with a causal mask
        let cacheBatch = KVCacheSimple()
        let causalMask = createAttentionMask(h: x, cache: cacheBatch)
        let outputBatch = attention(x, mask: causalMask, cache: cacheBatch)
        eval(outputBatch)

        // Sequential: process one token at a time (mask=.none since L=1 with cache)
        let cacheSeq = KVCacheSimple()
        var seqOutputs: [MLXArray] = []
        for i in 0 ..< L {
            let token = x[0..., i ..< (i + 1), 0...]
            let mask = createAttentionMask(h: token, cache: cacheSeq)
            let out = attention(token, mask: mask, cache: cacheSeq)
            seqOutputs.append(out)
        }
        let outputSeq = concatenated(seqOutputs, axis: 1)
        eval(outputSeq)

        // With correct RoPE these would match.  The buggy code would use
        // different offsets for keys/queries.
        let match = allClose(outputBatch, outputSeq, atol: 1e-4)
        eval(match)
        print(outputBatch)
        print(outputSeq)
        print(abs(outputSeq - outputBatch))
        #expect(match.item(Bool.self))
    }
}
