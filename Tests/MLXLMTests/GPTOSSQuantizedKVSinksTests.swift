import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXLLM

/// GPT-OSS attention sinks must still apply once `kvBits` turns the full-attention
/// layer caches into `QuantizedKVCache`. Sliding-window layers keep their
/// `RotatingKVCache`, which the affine conversion leaves alone.
final class GPTOSSQuantizedKVSinksTests: XCTestCase {

    private let prompt: [Int32] = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3]
    private let decodeToken: [Int32] = [7]
    private let chunk: [Int32] = [8, 11, 13]

    private func tinyModel() throws -> GPTOSSModel {
        // head_dim 64 so the default KV group size 64 can quantize the cache.
        let config = try JSONDecoder().decode(
            GPTOSSConfiguration.self,
            from: Data(
                """
                {
                    "model_type": "gpt_oss", "num_hidden_layers": 2, "num_local_experts": 4,
                    "num_experts_per_tok": 2, "vocab_size": 64, "rms_norm_eps": 1e-5,
                    "hidden_size": 64, "intermediate_size": 64, "head_dim": 64,
                    "num_attention_heads": 4, "num_key_value_heads": 2,
                    "sliding_window": 32, "rope_theta": 10000.0,
                    "layer_types": ["sliding_attention", "full_attention"]
                }
                """.utf8))
        MLXRandom.seed(1234)
        let model = GPTOSSModel(config)
        eval(model)
        return model
    }

    /// Layer 0 (sliding) always keeps real sinks; only layer 1 (full attention,
    /// the layer whose cache gets quantized) switches between real and neutral sinks.
    private func setSinks(_ model: GPTOSSModel, fullAttentionSinks: MLXArray) {
        let real = MLXArray([1.5, 2.0, 2.5, 3.0] as [Float])
        model.update(
            parameters: ModuleParameters.unflattened([
                "model.layers.0.self_attn.sinks": real,
                "model.layers.1.self_attn.sinks": fullAttentionSinks,
            ]))
        eval(model)
    }

    /// Prefill, optionally convert the cache the way generation does under `kvBits`,
    /// then run one decode token and one three-token chunk over that cache.
    private func run(_ model: GPTOSSModel, kvBits: Int?) throws -> (MLXArray, MLXArray) {
        var cache = try model.newCache(parameters: nil)
        eval(model(MLXArray(prompt).reshaped(1, -1), cache: cache))

        if let kvBits {
            maybeQuantizeKVCache(cache: &cache, kvBits: kvBits, kvGroupSize: 64)
            XCTAssertTrue(cache[0] is RotatingKVCache, "sliding layer: \(type(of: cache[0]))")
            XCTAssertTrue(cache[1] is QuantizedKVCache, "full layer: \(type(of: cache[1]))")
        }

        let step = model(MLXArray(decodeToken).reshaped(1, 1), cache: cache)
        eval(step)
        let chunkLogits = model(MLXArray(chunk).reshaped(1, -1), cache: cache)
        eval(chunkLogits)
        return (step, chunkLogits)
    }

    private func maxAbsDiff(_ a: MLXArray, _ b: MLXArray) -> Float {
        abs(a - b).max().item(Float.self)
    }

    func testQuantizedCacheKeepsAttentionSinks() throws {
        try Stream.withNewDefaultStream(device: .cpu) {
            let model = try tinyModel()

            setSinks(model, fullAttentionSinks: MLXArray([1.5, 2.0, 2.5, 3.0] as [Float]))
            let (plainReal, plainRealChunk) = try run(model, kvBits: nil)
            let (quantReal, quantRealChunk) = try run(model, kvBits: 8)

            // A very negative sink takes no softmax mass, which equals no sink at all.
            setSinks(model, fullAttentionSinks: MLXArray.full([4], values: MLXArray(Float(-1e9))))
            let (plainNeutral, _) = try run(model, kvBits: nil)
            let (quantNeutral, _) = try run(model, kvBits: 8)

            let plainVsQuant = maxAbsDiff(plainReal, quantReal)
            let plainVsQuantChunk = maxAbsDiff(plainRealChunk, quantRealChunk)
            let plainSinkEffect = maxAbsDiff(plainReal, plainNeutral)
            let quantSinkEffect = maxAbsDiff(quantReal, quantNeutral)
            // The sinks on the full-attention layer matter for the unquantized cache ...
            XCTAssertGreaterThan(plainSinkEffect, 0.05)
            // ... and must matter the same way once that layer's cache is quantized.
            XCTAssertGreaterThan(quantSinkEffect, 0.5 * plainSinkEffect)
            // With 8-bit keys and values the quantized cache tracks the unquantized one
            // up to quantization noise, well below the effect of the sinks.
            XCTAssertLessThan(plainVsQuant, 0.1 * plainSinkEffect)
            XCTAssertLessThan(plainVsQuantChunk, 0.1 * plainSinkEffect)
        }
    }
}
