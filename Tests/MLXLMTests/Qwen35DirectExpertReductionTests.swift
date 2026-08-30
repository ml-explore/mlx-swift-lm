import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXLLM

final class Qwen35DirectExpertReductionTests: XCTestCase {
    private func configuration() throws -> Qwen35TextConfiguration {
        let json = """
            {
                "model_type": "qwen3_5_moe",
                "hidden_size": 64,
                "num_hidden_layers": 1,
                "intermediate_size": 128,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 16,
                "linear_num_value_heads": 4,
                "linear_num_key_heads": 2,
                "linear_key_head_dim": 16,
                "linear_value_head_dim": 16,
                "linear_conv_kernel_dim": 4,
                "vocab_size": 128,
                "full_attention_interval": 1,
                "num_experts": 8,
                "num_experts_per_tok": 8,
                "moe_intermediate_size": 64,
                "shared_expert_intermediate_size": 64
            }
            """
        return try JSONDecoder().decode(
            Qwen35TextConfiguration.self,
            from: Data(json.utf8))
    }

    func testQuantizedPrefillMatchesEstablishedPathExactly() throws {
        MLXRandom.seed(31)
        let block = Qwen35SparseMoeBlock(try configuration())
        quantize(model: block, groupSize: 32, bits: 4)
        let input = MLXRandom.normal([1, 19, 64]).asType(.bfloat16)

        var gates = block.gate(input)
        gates = softmax(gates, axis: -1, precise: true)
        let (indices, scores) = moeRouterTopK(
            gates, k: block.topK, normalize: block.normTopkProb)
        let expertOutput = weightedExpertSum(block.switchMLP(input, indices), scores)
        let sharedOutput = sigmoid(block.sharedExpertGate(input)) * block.sharedExpert(input)
        let expected = expertOutput + sharedOutput
        let actual = block.forward(input)
        eval(expected, actual)

        XCTAssertTrue(allClose(actual, expected, rtol: 0, atol: 0).item(Bool.self))
    }
}
