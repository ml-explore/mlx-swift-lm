import Foundation
import MLX
import MLXLMCommon
import XCTest

@testable import MLXLLM

final class MoERouterFusedIntegrationTests: XCTestCase {
    private let input = MLXArray([
        Float(0.25), -0.5, 0.75, 1.0, -1.25, 0.125, 0.625, -0.875,
    ]).reshaped(1, 1, 8)

    private func decode<T: Decodable>(_ type: T.Type, _ json: String) throws -> T {
        try JSONDecoder().decode(type, from: Data(json.utf8))
    }

    private func assertBitIdentical(
        _ actual: MLXArray, _ expected: MLXArray,
        file: StaticString = #filePath, line: UInt = #line
    ) {
        eval(actual, expected)
        XCTAssertEqual(actual.dtype, expected.dtype, file: file, line: line)
        XCTAssertEqual(actual.shape, expected.shape, file: file, line: line)
        let actualBits = actual.asType(.float32).asArray(Float.self).map(\.bitPattern)
        let expectedBits = expected.asType(.float32).asArray(Float.self).map(\.bitPattern)
        XCTAssertEqual(actualBits, expectedBits, file: file, line: line)
    }

    private func descendingIndices(_ selection: MLXArray, k: Int) -> MLXArray {
        MLX.argPartition(-selection, kth: k - 1, axis: -1)[.ellipsis, ..<k]
    }

    func testQwen3MoEDecodeMatchesGenericRouter() throws {
        let config = try decode(
            Qwen3MoEConfiguration.self,
            """
            {
                "hidden_size": 8,
                "num_hidden_layers": 1,
                "intermediate_size": 16,
                "num_attention_heads": 2,
                "num_experts": 4,
                "num_experts_per_tok": 2,
                "decoder_sparse_step": 1,
                "mlp_only_layers": [],
                "moe_intermediate_size": 12,
                "rms_norm_eps": 1e-5,
                "vocab_size": 32,
                "num_key_value_heads": 1,
                "head_dim": 4,
                "norm_topk_prob": true
            }
            """)
        MLXRandom.seed(101)
        let block = Qwen3MoESparseMoeBlock(config)

        let actual = block(input)
        let logits = block.gate(input)
        let probabilities = MLX.softmax(logits, axis: -1, precise: true)
        let indices = descendingIndices(logits, k: block.topK)
        var scores = MLX.takeAlong(probabilities, indices, axis: -1)
        scores = scores / MLX.sum(scores, axis: -1, keepDims: true)
        let expected = weightedExpertSum(block.switchMLP(input, indices), scores)

        assertBitIdentical(actual, expected)
    }

    func testOlmoEDecodeMatchesGenericRouter() throws {
        let config = try decode(
            OlmoEConfiguration.self,
            """
            {
                "hidden_size": 8,
                "num_hidden_layers": 1,
                "intermediate_size": 12,
                "num_attention_heads": 2,
                "rms_norm_eps": 1e-5,
                "vocab_size": 32,
                "num_key_value_heads": 1,
                "num_experts": 4,
                "num_experts_per_tok": 2,
                "norm_topk_prob": true
            }
            """)
        MLXRandom.seed(102)
        let block = OlmoeSparseMoeBlock(config)

        let actual = block(input)
        let probabilities = MLX.softmax(block.gate(input), axis: -1, precise: true)
        let indices = descendingIndices(probabilities, k: block.topK)
        var scores = MLX.takeAlong(probabilities, indices, axis: -1)
        scores = scores / MLX.sum(scores, axis: -1, keepDims: true)
        let expected = weightedExpertSum(block.switchMLP(input, indices), scores)

        assertBitIdentical(actual, expected)
    }

    func testMixtralDecodeMatchesGenericRouter() throws {
        let config = try decode(
            MixtralConfiguration.self,
            """
            {
                "hidden_size": 8,
                "intermediate_size": 12,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "num_local_experts": 4,
                "num_experts_per_tok": 2
            }
            """)
        MLXRandom.seed(103)
        let block = MixtralSparseMoeBlock(config)

        let actual = block(input)
        let logits = block.gate(input)
        let indices = descendingIndices(logits, k: block.topK)
        let scores = MLX.softmax(
            MLX.takeAlong(logits, indices, axis: -1), axis: -1, precise: true)
        let expected = weightedExpertSum(block.switchMLP(input, indices), scores)

        assertBitIdentical(actual, expected)
    }

    func testJambaDecodeMatchesGenericRouter() throws {
        let config = try decode(
            JambaConfiguration.self,
            """
            {
                "model_type": "jamba",
                "hidden_size": 8,
                "intermediate_size": 12,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "attn_layer_offset": 0,
                "attn_layer_period": 1,
                "expert_layer_offset": 0,
                "expert_layer_period": 1,
                "mamba_d_conv": 2,
                "mamba_d_state": 4,
                "mamba_expand": 2,
                "num_experts": 4,
                "num_experts_per_tok": 2,
                "rms_norm_eps": 1e-5,
                "max_position_embeddings": 32,
                "vocab_size": 32
            }
            """)
        MLXRandom.seed(104)
        let block = JambaSparseMoeBlock(config)

        let actual = block(input)
        let logits = block.router(input)
        let indices = descendingIndices(logits, k: block.numExpertsPerTok)
        let scores = MLX.softmax(
            MLX.takeAlong(logits, indices, axis: -1), axis: -1, precise: true)
        let expected = weightedExpertSum(block.switchMLP(input, indices), scores)

        assertBitIdentical(actual, expected)
    }

    func testGraniteMoeHybridDecodeMatchesGenericRouter() {
        MLXRandom.seed(105)
        let router = GraniteMoeHybridTopKGating(inputSize: 8, numExperts: 4, topK: 2)

        let actual = router(input)
        let logits = router.layer(input)
        let expectedIndices = descendingIndices(logits, k: router.topK)
        let expectedScores = MLX.softmax(
            MLX.takeAlong(logits, expectedIndices, axis: -1), axis: -1, precise: true)
        eval(actual.0, actual.1, expectedIndices, expectedScores)

        XCTAssertEqual(
            actual.0.reshaped(-1).asArray(UInt32.self),
            expectedIndices.reshaped(-1).asArray(UInt32.self))
        assertBitIdentical(actual.1, expectedScores)
    }
}
