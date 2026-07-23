import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXLLM

/// Routing-correctness tests for the GLM-4 MoE gates (noaux_tc grouped top-k).
///
/// These models had zero coverage. Unlike a cache offset bug, a mis-wired router
/// still runs and produces plausible logits — the defect only shows as subtly wrong
/// token probabilities, so it hides without a targeted test. Two structural properties
/// are pinned here:
///
///  1. Group-limited routing: with `n_group`/`topk_group`, selection must be confined
///     to the winning group(s) (the DeepSeek group-limited-greedy shape).
///  2. `e_score_correction_bias` steers *which* experts are selected, but the returned
///     gate weights are gathered from the *unbiased* scores. A port that gathers the
///     biased `selectionScores` instead would inflate every routed weight — the single
///     easiest routing bug to introduce and the hardest to notice.
final class GLM4MoERoutingTests: XCTestCase {

    // MARK: - Config builders

    private func glmConfig(
        nRouted: Int, nGroup: Int, topkGroup: Int, topK: Int,
        hidden: Int = 4, moeInter: Int = 8, layers: Int = 1, firstKDense: Int = 0,
        normTopk: Bool = false, scaling: Float = 1.0
    ) throws -> GLM4MoEConfiguration {
        let json = """
            {
                "model_type": "glm4_moe", "vocab_size": 32, "hidden_size": \(hidden),
                "intermediate_size": 16, "max_position_embeddings": 128,
                "moe_intermediate_size": \(moeInter), "norm_topk_prob": \(normTopk),
                "num_attention_heads": 2, "n_group": \(nGroup), "head_dim": 2,
                "topk_group": \(topkGroup), "n_shared_experts": null,
                "n_routed_experts": \(nRouted), "routed_scaling_factor": \(scaling),
                "num_experts_per_tok": \(topK), "first_k_dense_replace": \(firstKDense),
                "num_hidden_layers": \(layers), "num_key_value_heads": 1,
                "rms_norm_eps": 1e-6, "rope_theta": 10000.0, "use_qk_norm": false,
                "tie_word_embeddings": true, "attention_bias": false,
                "partial_rotary_factor": 1.0, "scoring_func": "sigmoid",
                "topk_method": "noaux_tc"
            }
            """
        return try JSONDecoder().decode(GLM4MoEConfiguration.self, from: Data(json.utf8))
    }

    /// One-hot gate weight: expert e responds only to hidden dim e, so the input vector
    /// directly sets each expert's pre-sigmoid logit.
    private func oneHot(nRouted: Int, hidden: Int) -> MLXArray {
        var w = [Float](repeating: 0, count: nRouted * hidden)
        for e in 0 ..< nRouted { w[e * hidden + e] = 1 }
        return MLXArray(w).reshaped(nRouted, hidden)
    }

    // MARK: - Group-limited routing

    func testGroupLimitedRoutingConfinesToWinningGroup() throws {
        // 4 experts, 2 groups → group0={0,1}, group1={2,3}; keep 1 group, top-2 experts.
        let gate = GLM4MoEGate(try glmConfig(nRouted: 4, nGroup: 2, topkGroup: 1, topK: 2))
        try gate.update(
            parameters: ModuleParameters.unflattened(["weight": oneHot(nRouted: 4, hidden: 4)]),
            verify: [])

        // Drive experts 2,3 (group1) high, 0,1 (group0) low.
        let x = MLXArray([0.1, 0.1, 3.0, 2.0] as [Float]).reshaped(1, 1, 4)
        let (inds, scores) = gate(x)
        eval(inds, scores)

        let chosen = Set(inds.reshaped(-1).asArray(Int32.self).map(Int.init))
        XCTAssertEqual(chosen, [2, 3], "noaux_tc must confine top-k to the winning group")
        XCTAssertEqual(scores.reshaped(-1).asArray(Float.self).count, 2)
    }

    // MARK: - Correction bias steers selection but not the returned weight

    func testCorrectionBiasSteersSelectionNotReturnedWeights() throws {
        // 2 experts, single group, top-1. expert0 logit 2.0 (sigmoid≈0.881),
        // expert1 logit 0.0 (sigmoid 0.5). Without bias, expert0 wins.
        let gate = GLM4MoEGate(try glmConfig(nRouted: 2, nGroup: 1, topkGroup: 1, topK: 1))
        try gate.update(
            parameters: ModuleParameters.unflattened([
                "weight": oneHot(nRouted: 2, hidden: 4),
                // +5 bias on expert1 flips the *selection* to expert1.
                "e_score_correction_bias": MLXArray([0.0, 5.0] as [Float]),
            ]), verify: [])

        let x = MLXArray([2.0, 0.0, 0.0, 0.0] as [Float]).reshaped(1, 1, 4)
        let (inds, scores) = gate(x)
        eval(inds, scores)

        let chosen = inds.reshaped(-1).asArray(Int32.self).map(Int.init)
        let weight = scores.reshaped(-1).asArray(Float.self)[0]

        XCTAssertEqual(chosen, [1], "correction bias must steer selection to expert 1")
        // The returned weight is the UNBIASED sigmoid (0.5), not sigmoid+bias (5.5).
        XCTAssertEqual(
            weight, 0.5, accuracy: 1e-3,
            "gate weight must come from unbiased scores; got \(weight) "
                + "(≈5.5 would mean the biased selectionScores were gathered)")
    }

    // MARK: - Sanitize stacks per-expert weights and drops the MTP layer

    func testSanitizeStacksExpertsIntoSwitchMLPAndDropsMTP() throws {
        let config = try glmConfig(
            nRouted: 4, nGroup: 1, topkGroup: 1, topK: 2, hidden: 8, moeInter: 8,
            layers: 2, firstKDense: 1)
        let model = GLM4MoEModel(config)

        let (hidden, moeInter, experts) = (8, 8, 4)
        var weights: [String: MLXArray] = [:]
        for e in 0 ..< experts {
            let p = "model.layers.1.mlp.experts.\(e)"  // layer 1 is MoE (first_k_dense_replace=1)
            weights["\(p).gate_proj.weight"] = MLXArray.zeros([moeInter, hidden])
            weights["\(p).up_proj.weight"] = MLXArray.zeros([moeInter, hidden])
            weights["\(p).down_proj.weight"] = MLXArray.zeros([hidden, moeInter])
        }
        // A leftover MTP layer (index == hiddenLayers) must be dropped.
        weights["model.layers.2.embed_tokens.weight"] = MLXArray.zeros([32, hidden])

        let out = model.sanitize(weights: weights)

        XCTAssertNil(out["model.layers.1.mlp.experts.0.gate_proj.weight"])
        let base = "model.layers.1.mlp.switch_mlp"
        XCTAssertEqual(out["\(base).gate_proj.weight"]?.shape, [experts, moeInter, hidden])
        XCTAssertEqual(out["\(base).up_proj.weight"]?.shape, [experts, moeInter, hidden])
        XCTAssertEqual(out["\(base).down_proj.weight"]?.shape, [experts, hidden, moeInter])
        XCTAssertNil(out["model.layers.2.embed_tokens.weight"], "MTP layer must be dropped")
    }

    // MARK: - GLM4MOELite (separate gate implementation, same noaux_tc contract)

    private func liteConfig(
        nRouted: Int, nGroup: Int, topkGroup: Int, topK: Int, hidden: Int = 4
    ) throws -> GLM4MoELiteConfiguration {
        let json = """
            {
                "model_type": "glm4_moe_lite", "vocab_size": 32, "hidden_size": \(hidden),
                "intermediate_size": 16, "moe_intermediate_size": 8, "num_hidden_layers": 1,
                "num_attention_heads": 2, "num_key_value_heads": 1,
                "n_routed_experts": \(nRouted), "routed_scaling_factor": 1.0,
                "kv_lora_rank": 8, "qk_rope_head_dim": 2, "qk_nope_head_dim": 2, "v_head_dim": 4,
                "norm_topk_prob": false, "n_group": \(nGroup), "topk_group": \(topkGroup),
                "num_experts_per_tok": \(topK), "first_k_dense_replace": 0,
                "max_position_embeddings": 128, "rms_norm_eps": 1e-6, "rope_theta": 10000.0,
                "attention_bias": false, "partial_rotary_factor": 1.0,
                "topk_method": "noaux_tc", "scoring_func": "sigmoid"
            }
            """
        return try JSONDecoder().decode(GLM4MoELiteConfiguration.self, from: Data(json.utf8))
    }

    func testLiteGroupLimitedRoutingConfinesToWinningGroup() throws {
        let gate = GLM4MoELiteGate(try liteConfig(nRouted: 4, nGroup: 2, topkGroup: 1, topK: 2))
        try gate.update(
            parameters: ModuleParameters.unflattened(["weight": oneHot(nRouted: 4, hidden: 4)]),
            verify: [])

        let x = MLXArray([0.1, 0.1, 3.0, 2.0] as [Float]).reshaped(1, 1, 4)
        let (inds, scores) = gate(x)
        eval(inds, scores)

        let chosen = Set(inds.reshaped(-1).asArray(Int32.self).map(Int.init))
        XCTAssertEqual(chosen, [2, 3], "noaux_tc must confine top-k to the winning group")
        XCTAssertEqual(scores.reshaped(-1).asArray(Float.self).count, 2)
    }

    func testLiteCorrectionBiasSteersSelectionNotReturnedWeights() throws {
        let gate = GLM4MoELiteGate(try liteConfig(nRouted: 2, nGroup: 1, topkGroup: 1, topK: 1))
        try gate.update(
            parameters: ModuleParameters.unflattened([
                "weight": oneHot(nRouted: 2, hidden: 4),
                "e_score_correction_bias": MLXArray([0.0, 5.0] as [Float]),
            ]), verify: [])

        let x = MLXArray([2.0, 0.0, 0.0, 0.0] as [Float]).reshaped(1, 1, 4)
        let (inds, scores) = gate(x)
        eval(inds, scores)

        let chosen = inds.reshaped(-1).asArray(Int32.self).map(Int.init)
        let weight = scores.reshaped(-1).asArray(Float.self)[0]

        XCTAssertEqual(chosen, [1], "correction bias must steer selection to expert 1")
        XCTAssertEqual(
            weight, 0.5, accuracy: 1e-3,
            "gate weight must come from unbiased scores; got \(weight)")
    }
}
