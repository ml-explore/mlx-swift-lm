import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXLLM

final class DeepseekV2Tests: XCTestCase {

    // Tiny MoE config: 4 routed experts in 2 groups, top-1 group, top-2 experts.
    private func makeConfig(topkMethod: String = "group_limited_greedy") throws
        -> DeepseekV2Configuration
    {
        let json = """
            {
                "model_type": "deepseek_v2",
                "vocab_size": 32,
                "hidden_size": 8,
                "intermediate_size": 16,
                "moe_intermediate_size": 8,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "num_key_value_heads": 2,
                "n_routed_experts": 4,
                "n_shared_experts": 1,
                "num_experts_per_tok": 2,
                "n_group": 2,
                "topk_group": 1,
                "topk_method": "\(topkMethod)",
                "routed_scaling_factor": 1.0,
                "first_k_dense_replace": 1,
                "moe_layer_freq": 1,
                "q_lora_rank": null,
                "kv_lora_rank": 4,
                "qk_rope_head_dim": 2,
                "qk_nope_head_dim": 2,
                "v_head_dim": 4,
                "rms_norm_eps": 1e-6,
                "rope_theta": 10000.0,
                "max_position_embeddings": 128,
                "rope_scaling": {"type": "yarn", "factor": 1.0, "mscale_all_dim": 0}
            }
            """
        return try JSONDecoder().decode(DeepseekV2Configuration.self, from: Data(json.utf8))
    }

    func testForwardPassProducesLogitsShape() throws {
        let model = DeepseekV2Model(try makeConfig())
        let inputs = MLXArray([1, 2, 3] as [Int32]).reshaped(1, 3)

        let logits = model(inputs, cache: nil)
        eval(logits)

        XCTAssertEqual(logits.shape, [1, 3, 32])
    }

    /// Attention must update the KV cache exactly once per forward pass.
    ///
    /// The attention path builds `keys`/`values` and hands them to
    /// `attentionWithCacheUpdate`, which performs the cache update itself. An
    /// additional explicit `cache.update(...)` beforehand appended the returned
    /// history a second time, so a prompt of L tokens left the cache at 2L and
    /// every subsequent token attended over duplicated keys.
    func testForwardPassUpdatesCacheExactlyOnce() throws {
        let model = DeepseekV2Model(try makeConfig())
        let cache = model.newCache(parameters: nil)
        let tokenCount = 5
        let inputs = MLXArray(Array(Int32(1) ... Int32(tokenCount))).reshaped(1, tokenCount)

        let logits = model(inputs, cache: cache)
        eval(logits)

        for (i, layerCache) in cache.enumerated() {
            XCTAssertEqual(
                layerCache.offset, tokenCount,
                "layer \(i) cache advanced by \(layerCache.offset) for \(tokenCount) tokens")
        }
    }

    func testGroupLimitedGreedyMasksLosingGroup() throws {
        let gate = DeepseekV2MoEGate(config: try makeConfig())

        // weight is [n_routed_experts, hidden] = [4, 8]. Make expert e respond only
        // to hidden dim e, so the input directly controls each expert's logit.
        // Group layout (n_group=2): group0={e0,e1}, group1={e2,e3}.
        var w = [Float](repeating: 0, count: 4 * 8)
        for e in 0 ..< 4 { w[e * 8 + e] = 1 }
        try gate.update(
            parameters: ModuleParameters.unflattened(["weight": MLXArray(w).reshaped(4, 8)]),
            verify: [])

        // Drive experts 2 and 3 (group1) high, 0 and 1 (group0) low.
        let x = MLXArray([0.1, 0.1, 3.0, 2.0, 0, 0, 0, 0] as [Float]).reshaped(1, 1, 8)
        let (inds, scores) = gate(x)
        eval(inds, scores)

        let chosen = Set(inds.reshaped(-1).asArray(Int32.self).map(Int.init))
        // top-1 group is group1 → only experts {2,3} may be selected (top-2 within it).
        XCTAssertEqual(
            chosen, [2, 3], "group-limited-greedy must confine top-k to the winning group")
        XCTAssertEqual(scores.reshaped(-1).asArray(Float.self).count, 2)
    }

    func testNullQLoraRankDecodesToNil() throws {
        // DeepSeek-V2-Lite sets q_lora_rank: null → direct q_proj (no q LoRA).
        // Regression: a null must NOT fall back to the 1536 default (that would
        // build the q_a/q_b path and fail to load the checkpoint's q_proj).
        XCTAssertNil(try makeConfig().qLoraRank)

        let withRank = try JSONDecoder().decode(
            DeepseekV2Configuration.self,
            from: Data(#"{"hidden_size":8,"q_lora_rank":1536}"#.utf8))
        XCTAssertEqual(withRank.qLoraRank, 1536)
    }

    func testSanitizeStacksPerExpertWeightsIntoSwitchMLP() throws {
        let config = try makeConfig()
        let model = DeepseekV2Model(config)

        let (hidden, moeInter, experts) = (config.hiddenSize, config.moeIntermediateSize, 4)
        var weights: [String: MLXArray] = [:]
        for e in 0 ..< experts {
            let p = "model.layers.1.mlp.experts.\(e)"  // layer 1 is MoE (first_k_dense_replace=1)
            weights["\(p).gate_proj.weight"] = MLXArray.zeros([moeInter, hidden])
            weights["\(p).down_proj.weight"] = MLXArray.zeros([hidden, moeInter])
            weights["\(p).up_proj.weight"] = MLXArray.zeros([moeInter, hidden])
        }

        let out = model.sanitize(weights: weights)

        XCTAssertNil(out["model.layers.1.mlp.experts.0.gate_proj.weight"])
        let base = "model.layers.1.mlp.switch_mlp"
        XCTAssertEqual(out["\(base).gate_proj.weight"]?.shape, [experts, moeInter, hidden])
        XCTAssertEqual(out["\(base).down_proj.weight"]?.shape, [experts, hidden, moeInter])
        XCTAssertEqual(out["\(base).up_proj.weight"]?.shape, [experts, moeInter, hidden])
    }
}
