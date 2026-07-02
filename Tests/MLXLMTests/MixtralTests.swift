import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXLLM

final class MixtralTests: XCTestCase {

    private func makeConfig() throws -> MixtralConfiguration {
        let json = """
            {
                "model_type": "mixtral",
                "vocab_size": 32,
                "hidden_size": 8,
                "intermediate_size": 16,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "num_local_experts": 2,
                "num_experts_per_tok": 2,
                "rms_norm_eps": 1e-5,
                "rope_theta": 1000000.0
            }
            """
        return try JSONDecoder().decode(MixtralConfiguration.self, from: Data(json.utf8))
    }

    func testForwardPassProducesLogitsShape() throws {
        let model = MixtralModel(try makeConfig())
        let inputs = MLXArray([1, 2, 3] as [Int32]).reshaped(1, 3)

        let logits = model(inputs, cache: nil)
        eval(logits)

        // [batch, sequence, vocab]
        XCTAssertEqual(logits.shape, [1, 3, 32])
    }

    func testSanitizeStacksPerExpertWeightsIntoSwitchMLP() throws {
        let config = try makeConfig()
        let model = MixtralModel(config)

        // Per-expert projections as the HF checkpoint stores them: w1/w3 are
        // [intermediate, hidden], w2 is [hidden, intermediate].
        let (hidden, inter, experts) = (config.hiddenSize, config.intermediateSize, 2)
        var weights: [String: MLXArray] = [:]
        for e in 0 ..< experts {
            let p = "model.layers.0.block_sparse_moe.experts.\(e)"
            weights["\(p).w1.weight"] = MLXArray.zeros([inter, hidden])
            weights["\(p).w2.weight"] = MLXArray.zeros([hidden, inter])
            weights["\(p).w3.weight"] = MLXArray.zeros([inter, hidden])
        }

        let out = model.sanitize(weights: weights)

        // Per-expert keys are consumed; stacked switch_mlp keys appear with an
        // expert axis prepended, mapped w1->gate, w2->down, w3->up.
        for e in 0 ..< experts {
            XCTAssertNil(out["model.layers.0.block_sparse_moe.experts.\(e).w1.weight"])
        }
        let base = "model.layers.0.block_sparse_moe.switch_mlp"
        XCTAssertEqual(out["\(base).gate_proj.weight"]?.shape, [experts, inter, hidden])
        XCTAssertEqual(out["\(base).down_proj.weight"]?.shape, [experts, hidden, inter])
        XCTAssertEqual(out["\(base).up_proj.weight"]?.shape, [experts, inter, hidden])
    }
}
