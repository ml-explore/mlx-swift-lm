import Foundation
import MLX
import MLXNN
import XCTest

@testable import MLXLLM

final class LFM2MoeRoutingTests: XCTestCase {

    private func makeConfig(useExpertBias: Bool, normTopkProb: Bool = false) throws
        -> LFM2MoEConfiguration
    {
        let json = """
            {
                "model_type": "lfm2_moe",
                "vocab_size": 32,
                "hidden_size": 4,
                "intermediate_size": 8,
                "moe_intermediate_size": 8,
                "num_hidden_layers": 1,
                "num_experts": 4,
                "num_experts_per_tok": 2,
                "norm_topk_prob": \(normTopkProb),
                "num_attention_heads": 1,
                "num_key_value_heads": 1,
                "max_position_embeddings": 128,
                "use_expert_bias": \(useExpertBias),
                "num_dense_layers": 0,
                "norm_eps": 1e-5,
                "conv_bias": false,
                "conv_L_cache": 3
            }
            """
        return try JSONDecoder().decode(LFM2MoEConfiguration.self, from: Data(json.utf8))
    }

    private func makeBlock(
        useExpertBias: Bool, expertBias: [Float]? = nil,
        normTopkProb: Bool = false
    ) throws -> Lfm2MoeSparseMoeBlock {
        let block = Lfm2MoeSparseMoeBlock(
            try makeConfig(useExpertBias: useExpertBias, normTopkProb: normTopkProb))
        var params: [String: MLXArray] = ["gate.weight": MLX.eye(4)]
        if let expertBias {
            params["expert_bias"] = MLXArray(expertBias)
        }
        try block.update(parameters: ModuleParameters.unflattened(params), verify: [])
        eval(block)
        return block
    }

    private let logits: [Float] = [2, 1, 0, -1]
    private func x() -> MLXArray { MLXArray(logits).reshaped(1, 1, 4) }
    private func sig(_ v: Float) -> Float { 1 / (1 + expf(-v)) }

    private func routed(_ block: Lfm2MoeSparseMoeBlock) -> [Int: Float] {
        let r = block.route(x())
        let idx = r.indices.reshaped(-1).asArray(Int32.self).map(Int.init)
        let w = r.weights.reshaped(-1).asArray(Float.self)
        return Dictionary(uniqueKeysWithValues: zip(idx, w))
    }

    func testExpertBiasSteersSelectionOnly() throws {
        let block = try makeBlock(useExpertBias: true, expertBias: [0, 0, 1, 0])
        let m = routed(block)

        XCTAssertEqual(Set(m.keys), [0, 2], "expert_bias must move expert 2 into the top-k")
        XCTAssertEqual(m[2] ?? .nan, sig(0), accuracy: 1e-4)
        XCTAssertEqual(m[0] ?? .nan, sig(2), accuracy: 1e-4)
    }

    func testGateIsSigmoidNotSoftmax() throws {
        let block = try makeBlock(useExpertBias: false)
        let m = routed(block)

        XCTAssertEqual(Set(m.keys), [0, 1])
        XCTAssertEqual(m[0] ?? .nan, sig(2), accuracy: 1e-4)
        XCTAssertEqual(m[1] ?? .nan, sig(1), accuracy: 1e-4)
    }

    func testNormTopKProbRenormalizesUnbiasedWeights() throws {
        let block = try makeBlock(useExpertBias: false, normTopkProb: true)
        let m = routed(block)

        XCTAssertEqual(Set(m.keys), [0, 1])
        let denom = sig(2) + sig(1)
        XCTAssertEqual(m[0] ?? .nan, sig(2) / denom, accuracy: 1e-4)
        XCTAssertEqual(m[1] ?? .nan, sig(1) / denom, accuracy: 1e-4)
        XCTAssertEqual((m[0] ?? 0) + (m[1] ?? 0), 1, accuracy: 1e-4)
    }
}
