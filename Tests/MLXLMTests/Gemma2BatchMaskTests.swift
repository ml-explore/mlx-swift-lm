// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLLM
import Testing

@testable import MLXLMCommon

/// A cache whose mask carries each row's length, as a batched cache's mask does.
private final class RowLengthCache: KVCacheSimple {
    let lengths: [Int32]

    init(lengths: [Int32]) {
        self.lengths = lengths
        super.init()
    }

    override func makeMask(
        n: Int, windowSize: Int?, returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        .array(
            createCausalMask(
                n: n, offset: offset, windowSize: windowSize, lengths: MLXArray(lengths)))
    }
}

@Suite(.serialized)
struct Gemma2BatchMaskTests {

    @Test("A per-row array mask applies to each row of grouped-query scores")
    func perRowMask() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = try tinyGemma2()
            let prompts: [[Int32]] = [[3, 7, 1], [12, 5, 9, 2, 8]]
            let longest = prompts.map(\.count).max()!

            var expected: [MLXArray] = []
            for prompt in prompts {
                let logits = model(
                    MLXArray(prompt).reshaped([1, prompt.count]),
                    cache: try model.newCache(parameters: nil))
                expected.append(logits[0, prompt.count - 1])
            }

            let lengths = prompts.map { Int32($0.count) }
            let cache = try model.newCache(parameters: nil).map { _ in
                RowLengthCache(lengths: lengths) as KVCache
            }
            let padded = prompts.flatMap { $0 + Array(repeating: 0, count: longest - $0.count) }
            let logits = model(MLXArray(padded).reshaped([prompts.count, longest]), cache: cache)
            for (row, prompt) in prompts.enumerated() {
                #expect(
                    maxAbsDifference(logits[row, prompt.count - 1], expected[row]) < 1e-4,
                    "row \(row)")
            }
        }
    }
}

private func maxAbsDifference(_ lhs: MLXArray, _ rhs: MLXArray) -> Float {
    abs(lhs.asType(.float32) - rhs.asType(.float32)).max().item(Float.self)
}

/// Two layers of grouped-query attention: four heads over two key-value heads.
func tinyGemma2Configuration() throws -> Gemma2Configuration {
    try JSONDecoder().decode(
        Gemma2Configuration.self,
        from: Data(
            """
            {
              "hidden_size": 16, "num_hidden_layers": 2, "intermediate_size": 32,
              "num_attention_heads": 4, "head_dim": 8, "rms_norm_eps": 1e-6,
              "vocab_size": 32, "num_key_value_heads": 2, "attn_logit_softcapping": 50.0,
              "final_logit_softcapping": 30.0, "query_pre_attn_scalar": 144.0
            }
            """.utf8))
}

func tinyGemma2() throws -> Gemma2Model {
    let configuration = try tinyGemma2Configuration()
    return withRandomState(MLXRandom.RandomState(seed: 13)) {
        let model = Gemma2Model(configuration)
        eval(model)
        return model
    }
}
