import Foundation
import MLX
import XCTest

@testable import MLXLLM

final class Qwen3MoESanitizeTests: XCTestCase {
    private func makeConfig(tieWordEmbeddings: Bool) throws -> Qwen3MoEConfiguration {
        let json = """
            {
                "model_type": "qwen3_moe",
                "hidden_size": 8,
                "num_hidden_layers": 1,
                "intermediate_size": 16,
                "num_attention_heads": 1,
                "num_key_value_heads": 1,
                "head_dim": 8,
                "num_experts": 2,
                "num_experts_per_tok": 1,
                "decoder_sparse_step": 1,
                "mlp_only_layers": [],
                "moe_intermediate_size": 8,
                "rms_norm_eps": 1e-6,
                "vocab_size": 32,
                "tie_word_embeddings": \(tieWordEmbeddings)
            }
            """
        return try JSONDecoder().decode(
            Qwen3MoEConfiguration.self,
            from: Data(json.utf8)
        )
    }

    func testTiedEmbeddingsRemoveEveryQuantizedLMHeadTensor() throws {
        let model = Qwen3MoEModel(try makeConfig(tieWordEmbeddings: true))
        let value = MLXArray.zeros([1])
        let sanitized = model.sanitize(weights: [
            "lm_head.weight": value,
            "lm_head.scales": value,
            "lm_head.biases": value,
            "model.embed_tokens.weight": value,
        ])

        XCTAssertFalse(sanitized.keys.contains { $0.hasPrefix("lm_head.") })
        XCTAssertNotNil(sanitized["model.embed_tokens.weight"])
    }

    func testUntiedEmbeddingsPreserveQuantizedLMHeadTensors() throws {
        let model = Qwen3MoEModel(try makeConfig(tieWordEmbeddings: false))
        let value = MLXArray.zeros([1])
        let sanitized = model.sanitize(weights: [
            "lm_head.weight": value,
            "lm_head.scales": value,
            "lm_head.biases": value,
        ])

        XCTAssertNotNil(sanitized["lm_head.weight"])
        XCTAssertNotNil(sanitized["lm_head.scales"])
        XCTAssertNotNil(sanitized["lm_head.biases"])
    }
}
