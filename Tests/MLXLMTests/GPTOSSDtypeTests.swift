import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXLLM

/// The GPT-OSS SwiGLU clip bounds must follow the activation dtype. A float32
/// bound promotes the residual stream to float32, and every affine quantized
/// matmul then casts its full scales and biases to float32 on each forward.
final class GPTOSSDtypeTests: XCTestCase {

    private func tinyConfig() throws -> GPTOSSConfiguration {
        try JSONDecoder().decode(
            GPTOSSConfiguration.self,
            from: Data(
                """
                {
                    "model_type": "gpt_oss", "num_hidden_layers": 2, "num_local_experts": 4,
                    "num_experts_per_tok": 2, "vocab_size": 64, "rms_norm_eps": 1e-5,
                    "hidden_size": 64, "intermediate_size": 64, "head_dim": 16,
                    "num_attention_heads": 4, "num_key_value_heads": 2,
                    "sliding_window": 8, "rope_theta": 10000.0,
                    "layer_types": ["sliding_attention", "full_attention"]
                }
                """.utf8))
    }

    private func bf16Model(quantized: Bool) throws -> GPTOSSModel {
        let model = GPTOSSModel(try tinyConfig())
        model.update(parameters: model.parameters().mapValues { $0.asType(.bfloat16) })
        if quantized {
            quantize(model: model, groupSize: 32, bits: 4, mode: .affine)
        }
        eval(model)
        return model
    }

    func testMLPBlockKeepsBFloat16() throws {
        let mlp = MLPBlock(try tinyConfig())
        mlp.update(parameters: mlp.parameters().mapValues { $0.asType(.bfloat16) })
        let x = MLXRandom.normal([1, 3, 64]).asType(.bfloat16) * 4
        let y = mlp(x)
        eval(y)
        XCTAssertEqual(y.dtype, .bfloat16)
    }

    func testForwardKeepsResidualStreamInInputDType() throws {
        for quantized in [false, true] {
            let model = try bf16Model(quantized: quantized)
            let cache = try model.newCache(parameters: nil)
            let inputs = MLXArray(Array(Int32(1) ... Int32(5))).reshaped(1, 5)

            let hidden = model.model(inputs, cache: cache)
            eval(hidden)
            XCTAssertEqual(hidden.dtype, .bfloat16, "prefill, quantized=\(quantized)")

            let step = model(MLXArray([Int32(6)]).reshaped(1, 1), cache: cache)
            eval(step)
            XCTAssertEqual(step.dtype, .bfloat16, "decode logits, quantized=\(quantized)")

            // Every layer's cache keeps its keys in the input dtype.
            for (layer, layerCache) in cache.enumerated() {
                XCTAssertEqual(
                    layerCache.state.first?.dtype, .bfloat16,
                    "cache layer \(layer), quantized=\(quantized)")
            }
        }
    }
}
