import Foundation
import MLX
import MLXNN
import MLXVLM
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

struct RoPEApplicationTests {

    /// Gemma3nAttention applies rope, updates the cache and applies rope again.
    /// Ensure that it is correctly implemented.  We can observe prefill vs single token
    /// generaton and they should produce the same answers if implemented correctly.
    @Test func gemma3nAttentionTest() {
        // Task-local rather than MLXRandom.seed: parallel tests must not share
        // (or perturb) the global random stream. The scope covers the module
        // initializer too, so the attention weights are pinned as well.
        withRandomState(MLXRandom.RandomState(seed: 42)) {
            let config = Gemma3nTextConfiguration()
            let attention = Gemma3nAttention(config, layerIdx: 3)

            #expect(!attention.isKvSharedLayer)

            let B = 1
            let L = 4
            let D = config.hiddenSize

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
            // different offsets for keys/queries.  The tolerance is wider on
            // machines whose float32 matmuls run as TF32; see MatmulPrecision.
            let match = allClose(
                outputBatch, outputSeq,
                atol: MatmulPrecision.tolerance(float32: 1e-4, reduced: 5e-3))
            eval(match)
            #expect(
                match.item(Bool.self),
                """
                batch and sequential attention differ by up to \
                \(abs(outputSeq - outputBatch).max().item(Float.self))
                """)
        }
    }
}

/// Rows at different positions, left-padded to one width, as a batched cache holds them.
private final class RowOffsetCache: KVCacheSimple, BatchPositionedKVCache {
    private let leftPadding: [Int32]

    /// Joins prefilled single-sequence caches into one batch.
    init(rows: [any KVCache]) {
        let longest = rows.map(\.offset).max() ?? 0
        leftPadding = rows.map { Int32(longest - $0.offset) }
        super.init()
        var keys: [MLXArray] = []
        var values: [MLXArray] = []
        for row in rows {
            let widths: [IntOrPair] = [0, 0, .init((longest - row.offset, 0)), 0]
            keys.append(MLX.padded(row.state[0], widths: widths))
            values.append(MLX.padded(row.state[1], widths: widths))
        }
        state = [concatenated(keys), concatenated(values)]
    }

    var batchOffset: MLXArray { MLXArray(leftPadding.map { Int32(offset) - $0 }) }

    override var ropeOffset: RoPEOffset { .batch(batchOffset) }

    override func makeMask(
        n: Int, windowSize: Int?, returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        let causal = createCausalMask(n: n, offset: offset, windowSize: windowSize)
        let positions = MLXArray(Int32(0) ..< Int32(offset + n))
        let padding = MLXArray(leftPadding)[0..., .newAxis, .newAxis, .newAxis]
        return .array(causal & (positions .>= padding))
    }
}

/// An image row and a text row prefilled separately, then decoded as one batch, must each get
/// the logits they get on their own.
@Suite(.serialized)
struct VLMRowPositionTests {

    @Test("FastVLM positions each row of a batch at its own offset")
    func fastVLM() throws {
        try Device.withDefaultDevice(.cpu) {
            let configuration = try JSONDecoder().decode(
                FastVLMConfiguration.self,
                from: Data(
                    """
                    {
                      "model_type": "fastvlm", "hidden_size": 16, "num_hidden_layers": 1,
                      "intermediate_size": 32, "num_attention_heads": 2, "num_key_value_heads": 1,
                      "vocab_size": 16, "eos_token_id": 15, "mm_projector_type": "mlp2x_gelu",
                      "mm_hidden_size": 16, "tokenizer_model_max_length": 1024,
                      "tokenizer_padding_side": "right",
                      "vision_config": {
                        "cls_ratio": 1, "down_patch_size": 3, "down_stride": 2,
                        "downsamples": [false, false], "embed_dims": [16, 16],
                        "hidden_size": 16, "image_size": 128, "intermediate_size": 32,
                        "layers": [1, 1], "layer_scale_init_value": 0.00001,
                        "mlp_ratios": [2, 2], "num_classes": 16, "patch_size": 99,
                        "pos_embs_shapes": [null, null], "projection_dim": 16,
                        "repmixer_kernel_size": 3, "token_mixers": ["repmixer", "repmixer"]
                      }
                    }
                    """.utf8))
            let model = withRandomState(MLXRandom.RandomState(seed: 501)) {
                let model = FastVLM(configuration)
                eval(model)
                return model
            }
            let inputs = [
                LMInput(
                    text: .init(tokens: MLXArray([1, -200, 2]).reshaped([1, 3])),
                    image: .init(pixels: MLXArray.ones([1, 3, 16, 16]), frames: [])),
                LMInput(tokens: MLXArray([1, 2, 3]).reshaped([1, 3])),
            ]
            try expectRowsMatchSingles(model, inputs: inputs, positions: [18, 3])
        }
    }

    @Test("Gemma 3 positions each row of a batch at its own offset")
    func gemma3() throws {
        try Device.withDefaultDevice(.cpu) {
            let configuration = try JSONDecoder().decode(
                Gemma3Configuration.self,
                from: Data(
                    """
                    {
                      "model_type": "gemma3", "mm_tokens_per_image": 4,
                      "text_config": {
                        "model_type": "gemma3_text", "hidden_size": 16, "num_hidden_layers": 6,
                        "intermediate_size": 32, "num_attention_heads": 2, "num_key_value_heads": 1,
                        "head_dim": 8, "query_pre_attn_scalar": 8, "sliding_window": 64
                      },
                      "vision_config": {
                        "model_type": "siglip_vision_model", "num_hidden_layers": 1,
                        "hidden_size": 16, "intermediate_size": 32, "num_attention_heads": 2,
                        "patch_size": 2, "image_size": 8
                      }
                    }
                    """.utf8))
            let model = withRandomState(MLXRandom.RandomState(seed: 502)) {
                let model = Gemma3(configuration)
                eval(model)
                return model
            }
            let inputs = [
                LMInput(
                    text: .init(
                        tokens: MLXArray([1, 262144, 262144, 262144, 262144, 2])
                            .reshaped([1, 6])),
                    image: .init(pixels: MLXArray.ones([1, 3, 8, 8]), frames: [])),
                LMInput(tokens: MLXArray([1, 2, 3]).reshaped([1, 3])),
            ]
            try expectRowsMatchSingles(model, inputs: inputs, positions: [6, 3])
        }
    }
}

/// Prefills each input alone, joins the caches as one batch and decodes three tokens,
/// comparing every row's logits with the same sequence continued alone.
private func expectRowsMatchSingles(
    _ model: any LanguageModel, inputs: [LMInput], positions: [Int]
) throws {
    func prefill(_ input: LMInput) throws -> [any KVCache] {
        let cache = try model.newCache(parameters: nil)
        switch try model.prepare(input, cache: cache, state: nil, prefill: .init()) {
        case .logits(let output): eval(output.logits)
        case .tokens: Issue.record("The model must complete its own prefill")
        }
        eval(cache)
        return cache
    }

    let singles = try inputs.map(prefill)
    #expect(singles.map { $0[0].offset } == positions)
    let batch = singles[0].indices.map { layer in
        RowOffsetCache(rows: singles.map { $0[layer] }) as KVCache
    }
    #expect((batch[0] as? RowOffsetCache)?.batchOffset.asArray(Int.self) == positions)

    for step in 0 ..< 3 {
        let token = Int32(4 + step)
        let output = model(MLXArray([token, token]).reshaped([2, 1]), cache: batch)
        for row in 0 ..< 2 {
            let expected = model(MLXArray([token]).reshaped([1, 1]), cache: singles[row])
            #expect(
                maxAbsDifference(output[row ..< row + 1], expected) < 1e-4,
                "row \(row), step \(step)")
        }
    }
}

func maxAbsDifference(_ lhs: MLXArray, _ rhs: MLXArray) -> Float {
    abs(lhs.asType(.float32) - rhs.asType(.float32)).max().item(Float.self)
}
