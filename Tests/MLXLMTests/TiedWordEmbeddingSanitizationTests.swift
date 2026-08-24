import MLX
import XCTest

@testable import MLXLLM
@testable import MLXLMCommon

final class TiedWordEmbeddingSanitizationTests: XCTestCase {
    private let value = MLXArray.zeros([1])

    func testSharedFilterRemovesEveryLMHeadParameterAtAnyNamespace() {
        let sanitized = filterLMHeadWeights(
            from: [
                "lm_head.weight": value,
                "lm_head.scales": value,
                "lm_head.biases": value,
                "language_model.lm_head.weight": value,
                "language_model.lm_head.scales": value,
                "model.embed_tokens.weight": value,
                "model.lm_head_projection.weight": value,
            ],
            tiedWordEmbeddings: true)

        XCTAssertFalse(
            sanitized.keys.contains { $0.split(separator: ".").contains("lm_head") })
        XCTAssertNotNil(sanitized["model.embed_tokens.weight"])
        XCTAssertNotNil(sanitized["model.lm_head_projection.weight"])
    }

    func testSharedFilterPreservesLMHeadParametersWhenEmbeddingsAreUntied() {
        let weights = [
            "lm_head.weight": value,
            "lm_head.scales": value,
            "lm_head.biases": value,
        ]

        let sanitized = filterLMHeadWeights(
            from: weights, tiedWordEmbeddings: false)

        XCTAssertEqual(Set(sanitized.keys), Set(weights.keys))
    }

    func testPreviouslyUnsanitizedLlamaModelUsesSharedFilter() {
        let configuration = LlamaConfiguration(
            hiddenSize: 8,
            hiddenLayers: 1,
            intermediateSize: 16,
            attentionHeads: 1,
            rmsNormEps: 1e-6,
            vocabularySize: 32,
            kvHeads: 1,
            tieWordEmbeddings: true)
        let model = LlamaModel(configuration)

        let sanitized = model.sanitize(weights: [
            "lm_head.weight": value,
            "lm_head.scales": value,
            "lm_head.biases": value,
            "model.embed_tokens.weight": value,
        ])

        XCTAssertFalse(sanitized.keys.contains { $0.hasPrefix("lm_head.") })
        XCTAssertNotNil(sanitized["model.embed_tokens.weight"])
    }
}
