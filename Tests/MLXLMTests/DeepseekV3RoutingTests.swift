import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXLLM

/// The DeepSeek-V3 `noaux_tc` gate must return the same expert indices and
/// routing weights as mlx-lm's `group_expert_select`: `e_score_correction_bias`
/// only chooses the experts, the weights are gathered from the uncorrected
/// sigmoid scores, and `routed_scaling_factor` applies whether or not the
/// weights are normalised. The scores are computed in float32 whatever the
/// activation dtype, and the weights come back in the activation dtype.
///
/// Expected values come from mlx-lm 0.32.2 (mlx 0.32.4) `MoEGate`, run on the
/// CPU. The first three tests run in float32 with the weights, bias and inputs
/// below. All inputs and gate
/// weights are exact multiples of 1/32 or 1/16. On every token the selected
/// group wins by at least 0.18 and the k-th expert by at least 0.038, so the
/// selection has no near ties.
final class DeepseekV3RoutingTests: XCTestCase {

    private static let hiddenSize = 16
    private static let routedExperts = 8

    private static let gateWeight: [[Float]] = [
        [
            -0.25, -0.15625, -0.0625, 0.03125, 0.125, 0.21875, -0.21875, -0.125, -0.03125,
            0.0625, 0.15625, 0.25, -0.1875, -0.09375, 0, 0.09375,
        ],
        [
            -0.03125, 0.0625, 0.15625, 0.25, -0.1875, -0.09375, 0, 0.09375, 0.1875, -0.25,
            -0.15625, -0.0625, 0.03125, 0.125, 0.21875, -0.21875,
        ],
        [
            0.1875, -0.25, -0.15625, -0.0625, 0.03125, 0.125, 0.21875, -0.21875, -0.125,
            -0.03125, 0.0625, 0.15625, 0.25, -0.1875, -0.09375, 0,
        ],
        [
            -0.125, -0.03125, 0.0625, 0.15625, 0.25, -0.1875, -0.09375, 0, 0.09375, 0.1875,
            -0.25, -0.15625, -0.0625, 0.03125, 0.125, 0.21875,
        ],
        [
            0.09375, 0.1875, -0.25, -0.15625, -0.0625, 0.03125, 0.125, 0.21875, -0.21875,
            -0.125, -0.03125, 0.0625, 0.15625, 0.25, -0.1875, -0.09375,
        ],
        [
            -0.21875, -0.125, -0.03125, 0.0625, 0.15625, 0.25, -0.1875, -0.09375, 0, 0.09375,
            0.1875, -0.25, -0.15625, -0.0625, 0.03125, 0.125,
        ],
        [
            0, 0.09375, 0.1875, -0.25, -0.15625, -0.0625, 0.03125, 0.125, 0.21875, -0.21875,
            -0.125, -0.03125, 0.0625, 0.15625, 0.25, -0.1875,
        ],
        [
            0.21875, -0.21875, -0.125, -0.03125, 0.0625, 0.15625, 0.25, -0.1875, -0.09375, 0,
            0.09375, 0.1875, -0.25, -0.15625, -0.0625, 0.03125,
        ],
    ]

    private static let correctionBias: [Float] = [
        0.25, -0.125, 0.0625, -0.25, 0.1875, 0.375, -0.0625, 0.125,
    ]

    private static let inputs: [[Float]] = [
        [
            -0.375, 0.3125, 0.1875, 0.0625, -0.0625, -0.1875, -0.3125, 0.375, 0.25, 0.125, 0,
            -0.125, -0.25, -0.375, 0.3125, 0.1875,
        ],
        [
            -0.0625, -0.1875, -0.3125, 0.375, 0.25, 0.125, 0, -0.125, -0.25, -0.375, 0.3125,
            0.1875, 0.0625, -0.0625, -0.1875, -0.3125,
        ],
        [
            0.25, 0.125, 0, -0.125, -0.25, -0.375, 0.3125, 0.1875, 0.0625, -0.0625, -0.1875,
            -0.3125, 0.375, 0.25, 0.125, 0,
        ],
    ]

    private func config(
        hiddenSize: Int = DeepseekV3RoutingTests.hiddenSize,
        routedExperts: Int = DeepseekV3RoutingTests.routedExperts, nGroup: Int,
        topkGroup: Int = 1, topK: Int = 2, normTopkProb: Bool
    ) throws -> DeepseekV3Configuration {
        try JSONDecoder().decode(
            DeepseekV3Configuration.self,
            from: Data(
                """
                {
                    "model_type": "deepseek_v3", "vocab_size": 32,
                    "hidden_size": \(hiddenSize), "intermediate_size": 32,
                    "moe_intermediate_size": 8, "num_hidden_layers": 1,
                    "num_attention_heads": 2, "num_key_value_heads": 2,
                    "n_routed_experts": \(routedExperts), "n_shared_experts": null,
                    "num_experts_per_tok": \(topK), "n_group": \(nGroup),
                    "topk_group": \(topkGroup),
                    "norm_topk_prob": \(normTopkProb), "routed_scaling_factor": 2.5,
                    "first_k_dense_replace": 0, "moe_layer_freq": 1,
                    "q_lora_rank": 4, "kv_lora_rank": 4, "qk_rope_head_dim": 2,
                    "qk_nope_head_dim": 2, "v_head_dim": 4, "rms_norm_eps": 1e-6,
                    "rope_theta": 10000.0, "max_position_embeddings": 128,
                    "attention_bias": false
                }
                """.utf8))
    }

    /// Runs the gate on the CPU in float32 and compares each token against the
    /// expected (expert, weight) pairs. The order of the top-k indices is
    /// implementation-defined (argPartition), so the pairs are compared sorted
    /// by expert index.
    private func assertGate(
        nGroup: Int, normTopkProb: Bool, expected: [[(Int, Float)]],
        file: StaticString = #filePath, line: UInt = #line
    ) throws {
        try Device.withDefaultDevice(.cpu) {
            let gate = MoEGate(config: try config(nGroup: nGroup, normTopkProb: normTopkProb))
            try gate.update(
                parameters: ModuleParameters.unflattened([
                    "weight": MLXArray(Self.gateWeight.flatMap { $0 })
                        .reshaped(Self.routedExperts, Self.hiddenSize),
                    "e_score_correction_bias": MLXArray(Self.correctionBias),
                ]),
                verify: [.noUnusedKeys])

            let x = MLXArray(Self.inputs.flatMap { $0 })
                .reshaped(1, Self.inputs.count, Self.hiddenSize)
            let (inds, weights) = gate(x)
            eval(inds, weights)
            XCTAssertEqual(weights.dtype, .float32, file: file, line: line)

            let k = inds.dim(-1)
            let flatInds = inds.reshaped(-1).asArray(Int32.self).map(Int.init)
            let flatWeights = weights.reshaped(-1).asArray(Float.self)
            for (token, want) in expected.enumerated() {
                let got = (0 ..< k)
                    .map { (flatInds[token * k + $0], flatWeights[token * k + $0]) }
                    .sorted { $0.0 < $1.0 }
                XCTAssertEqual(
                    got.map(\.0), want.map(\.0), "token \(token) experts", file: file,
                    line: line)
                for ((_, g), (_, w)) in zip(got, want) {
                    XCTAssertEqual(
                        g, w, accuracy: 1e-5,
                        "token \(token) weights: got \(got), mlx-lm \(want)", file: file,
                        line: line)
                }
            }
        }
    }

    /// Two groups of four, one group kept, top-2, normalised, scaling 2.5.
    func testGroupedNormalisedRoutingMatchesMLXLM() throws {
        try assertGate(
            nGroup: 2, normTopkProb: true,
            expected: [
                [(4, 1.11309505), (5, 1.38690495)],
                [(4, 1.25699544), (5, 1.24300468)],
                [(4, 1.42427695), (5, 1.07572293)],
            ])
    }

    /// Without normalisation mlx-lm still multiplies by `routed_scaling_factor`.
    func testGroupedUnnormalisedRoutingMatchesMLXLM() throws {
        try assertGate(
            nGroup: 2, normTopkProb: false,
            expected: [
                [(4, 1.07656538), (5, 1.34138942)],
                [(4, 1.3134222), (5, 1.29880333)],
                [(4, 1.39220405), (5, 1.05149889)],
            ])
    }

    /// `n_group` 1: no group masking, top-2 over all eight experts.
    func testSingleGroupRoutingMatchesMLXLM() throws {
        try assertGate(
            nGroup: 1, normTopkProb: true,
            expected: [
                [(0, 1.22925091), (5, 1.27074897)],
                [(0, 1.28411388), (5, 1.21588612)],
                [(4, 1.42427695), (5, 1.07572293)],
            ])
    }

    // MARK: - bfloat16 activations with a realistic correction bias

    // 64 experts in 8 groups, 4 groups kept, top-8, normalised, scaling 2.5:
    // the DeepSeek-V3 routing shape with a quarter of its experts. The gate
    // weight and the inputs are bfloat16, as in a bfloat16 checkpoint. The
    // expected values come from mlx-lm's `MoEGate` fed the same bfloat16 arrays.
    // In float32 the kept groups win by at least 0.0013 and the 8th expert by
    // at least 0.0024 on every token, so the expected sets are unambiguous.
    // The bias sits near 4.06, where the bfloat16 step is 1/32: selection
    // scores computed in bfloat16 collapse to about 20 distinct values for 64
    // experts.
    private static let bf16HiddenSize = 32
    private static let bf16RoutedExperts = 64
    // Rows of the input formula; row 9 has a float32 expert margin of only
    // 6e-5 and is left out.
    private static let bf16Tokens = Array(0 ... 8) + Array(10 ... 16)

    // The first 64 values of layer 10's `e_score_correction_bias` in
    // mlx-community/DeepSeek-V3.1-4bit (stored as float32). Rounded to
    // bfloat16 all of them become 4.0625, as in DeepSeek-V3-0324-4bit.
    private static let realisticBias: [Float] = [
        4.05898857, 4.05698872, 4.06098843, 4.06198835, 4.06098843, 4.06078815, 4.0599885,
        4.06278801, 4.0587883, 4.06498814, 4.06098843, 4.06198835, 4.05698872, 4.06178808,
        4.05698872, 4.06278801, 4.06398821, 4.05898857, 4.06378794, 4.06278801, 4.05898857,
        4.05898857, 4.06178808, 4.05498886, 4.06098843, 4.06378794, 4.06098843, 4.06098843,
        4.06378794, 4.05378866, 4.06098843, 4.06298828, 4.0599885, 4.05398893, 4.06278801,
        4.0599885, 4.06098843, 4.06198835, 4.05778837, 4.06198835, 4.05798864, 4.05698872,
        4.06198835, 4.06078815, 4.05898857, 4.06178808, 4.0587883, 4.06078815, 4.05978823,
        4.06178808, 4.06198835, 4.0587883, 4.0599885, 4.06098843, 4.05298901, 4.06078815,
        4.06578779, 4.05778837, 4.04978895, 4.06178808, 4.06098843, 4.05798864, 4.06078815,
        4.06078815,
    ]

    private static let expectedFloat32Bias: [[(Int, Float)]] = [
        [
            (0, 0.291834444), (1, 0.289257824), (5, 0.359914899), (7, 0.307416677),
            (
                17,
                0.321741641
            ), (50, 0.348742306), (61, 0.291834444), (62, 0.289257824),
        ],
        [
            (19, 0.322388291), (21, 0.314305305), (24, 0.288677901), (28, 0.32512486),
            (
                39,
                0.358081818
            ), (48, 0.347934723), (52, 0.271743536), (53, 0.271743536),
        ],
        [
            (16, 0.305768937), (20, 0.296182603), (22, 0.312940508), (30, 0.329864293),
            (
                33,
                0.306116194
            ), (37, 0.340921134), (46, 0.310578465), (47, 0.297627866),
        ],
        [
            (19, 0.31276834), (24, 0.276915312), (25, 0.301821858), (26, 0.365294933),
            (
                32,
                0.272086293
            ), (35, 0.368906915), (36, 0.278018862), (48, 0.324187398),
        ],
        [
            (16, 0.299849212), (18, 0.277562916), (27, 0.264366984), (30, 0.316596687),
            (
                33,
                0.358671486
            ), (34, 0.323782206), (49, 0.366520762), (51, 0.292649686),
        ],
        [
            (24, 0.308599859), (31, 0.375282645), (33, 0.28642258), (34, 0.282222629),
            (
                36,
                0.284520507
            ), (41, 0.312720865), (47, 0.28413865), (50, 0.366092265),
        ],
        [
            (16, 0.286636084), (20, 0.334301531), (28, 0.290739298), (29, 0.348359644),
            (
                35,
                0.273527801
            ), (36, 0.297733665), (39, 0.310974956), (47, 0.35772723),
        ],
        [
            (1, 0.31390214), (7, 0.294717997), (16, 0.31424883), (18, 0.348887622),
            (
                27,
                0.333628297
            ), (28, 0.28940168), (56, 0.291311502), (62, 0.31390214),
        ],
        [
            (0, 0.305997282), (3, 0.294529915), (19, 0.31652227), (20, 0.325829983),
            (
                25,
                0.322173089
            ), (27, 0.296305746), (30, 0.323401928), (43, 0.315239847),
        ],
        [
            (12, 0.382956922), (14, 0.29798016), (28, 0.28515783), (29, 0.293145746),
            (
                35,
                0.294642508
            ), (39, 0.303086609), (56, 0.344681978), (58, 0.298348427),
        ],
        [
            (10, 0.352149725), (24, 0.315835804), (30, 0.33124125), (33, 0.311079204),
            (
                35,
                0.2894454
            ), (48, 0.290572166), (49, 0.274808228), (52, 0.334868163),
        ],
        [
            (7, 0.308911443), (8, 0.380024612), (9, 0.332111537), (26, 0.298175007),
            (
                31,
                0.300122321
            ), (48, 0.287852556), (49, 0.29895547), (52, 0.293847084),
        ],
        [
            (2, 0.351266384), (3, 0.301261514), (6, 0.338469088), (33, 0.269654274),
            (
                35,
                0.309525996
            ), (48, 0.313920498), (49, 0.26463604), (63, 0.351266384),
        ],
        [
            (4, 0.342832357), (6, 0.299530774), (14, 0.293693185), (21, 0.324964553),
            (
                23,
                0.317849517
            ), (24, 0.291469753), (28, 0.311810464), (31, 0.317849517),
        ],
        [
            (8, 0.288331747), (9, 0.308237553), (44, 0.349641144), (46, 0.296536386),
            (
                52,
                0.336827368
            ), (55, 0.273807526), (56, 0.34060514), (63, 0.306013107),
        ],
        [
            (0, 0.296447635), (5, 0.292022943), (7, 0.320947886), (9, 0.305894673),
            (
                12,
                0.298836648
            ), (35, 0.351856112), (45, 0.342376292), (47, 0.2916179),
        ],
    ]

    private static let expectedBfloat16Bias: [[(Int, Float)]] = [
        [
            (0, 0.291834444), (1, 0.289257824), (5, 0.359914899), (7, 0.307416677),
            (
                17,
                0.321741641
            ), (50, 0.348742306), (61, 0.291834444), (62, 0.289257824),
        ],
        [
            (19, 0.322388291), (21, 0.314305305), (24, 0.288677901), (28, 0.32512486),
            (
                39,
                0.358081818
            ), (48, 0.347934723), (52, 0.271743536), (53, 0.271743536),
        ],
        [
            (16, 0.305768937), (20, 0.296182603), (22, 0.312940508), (30, 0.329864293),
            (
                33,
                0.306116194
            ), (37, 0.340921134), (46, 0.310578465), (47, 0.297627866),
        ],
        [
            (19, 0.31276834), (24, 0.276915312), (25, 0.301821858), (26, 0.365294933),
            (
                32,
                0.272086293
            ), (35, 0.368906915), (36, 0.278018862), (48, 0.324187398),
        ],
        [
            (16, 0.299849212), (18, 0.277562916), (27, 0.264366984), (30, 0.316596687),
            (
                33,
                0.358671486
            ), (34, 0.323782206), (49, 0.366520762), (51, 0.292649686),
        ],
        [
            (24, 0.308599859), (31, 0.375282645), (33, 0.28642258), (34, 0.282222629),
            (
                36,
                0.284520507
            ), (41, 0.312720865), (47, 0.28413865), (50, 0.366092265),
        ],
        [
            (16, 0.286636084), (20, 0.334301531), (28, 0.290739298), (29, 0.348359644),
            (
                35,
                0.273527801
            ), (36, 0.297733665), (39, 0.310974956), (47, 0.35772723),
        ],
        [
            (1, 0.313521028), (7, 0.294360191), (16, 0.313867331), (18, 0.348464072),
            (
                27,
                0.333223224
            ), (28, 0.289050311), (40, 0.306473702), (46, 0.301040143),
        ],
        [
            (0, 0.305997282), (3, 0.294529915), (19, 0.31652227), (20, 0.325829983),
            (
                25,
                0.322173089
            ), (27, 0.296305746), (30, 0.323401928), (43, 0.315239847),
        ],
        [
            (12, 0.382956922), (14, 0.29798016), (28, 0.28515783), (29, 0.293145746),
            (
                35,
                0.294642508
            ), (39, 0.303086609), (56, 0.344681978), (58, 0.298348427),
        ],
        [
            (10, 0.352149725), (24, 0.315835804), (30, 0.33124125), (33, 0.311079204),
            (
                35,
                0.2894454
            ), (48, 0.290572166), (49, 0.274808228), (52, 0.334868163),
        ],
        [
            (7, 0.308911443), (8, 0.380024612), (9, 0.332111537), (26, 0.298175007),
            (
                31,
                0.300122321
            ), (48, 0.287852556), (49, 0.29895547), (52, 0.293847084),
        ],
        [
            (2, 0.351266384), (3, 0.301261514), (6, 0.338469088), (33, 0.269654274),
            (
                35,
                0.309525996
            ), (48, 0.313920498), (49, 0.26463604), (63, 0.351266384),
        ],
        [
            (4, 0.342832357), (6, 0.299530774), (14, 0.293693185), (21, 0.324964553),
            (
                23,
                0.317849517
            ), (24, 0.291469753), (28, 0.311810464), (31, 0.317849517),
        ],
        [
            (8, 0.288331747), (9, 0.308237553), (44, 0.349641144), (46, 0.296536386),
            (
                52,
                0.336827368
            ), (55, 0.273807526), (56, 0.34060514), (63, 0.306013107),
        ],
        [
            (0, 0.296447635), (5, 0.292022943), (7, 0.320947886), (9, 0.305894673),
            (
                12,
                0.298836648
            ), (35, 0.351856112), (45, 0.342376292), (47, 0.2916179),
        ],
    ]

    /// Multiples of 1/64 with at most five significant bits: exact in bfloat16.
    private static func bf16GateWeight() -> MLXArray {
        var values = [Float]()
        for e in 0 ..< bf16RoutedExperts {
            for h in 0 ..< bf16HiddenSize {
                values.append(Float((e * 37 + h * 101 + e * h * 7) % 61 - 30) / 64)
            }
        }
        return MLXArray(values).reshaped(bf16RoutedExperts, bf16HiddenSize)
            .asType(.bfloat16)
    }

    /// Multiples of 1/32 with at most five significant bits: exact in bfloat16.
    private static func bf16Inputs() -> MLXArray {
        var values = [Float]()
        for t in bf16Tokens {
            for h in 0 ..< bf16HiddenSize {
                values.append(Float((t * 53 + h * 29 + t * h * 11) % 47 - 23) / 32)
            }
        }
        return MLXArray(values).reshaped(1, bf16Tokens.count, bf16HiddenSize)
            .asType(.bfloat16)
    }

    private func assertBFloat16Gate(
        bias: MLXArray, expected: [[(Int, Float)]],
        file: StaticString = #filePath, line: UInt = #line
    ) throws {
        try Device.withDefaultDevice(.cpu) {
            let gate = MoEGate(
                config: try config(
                    hiddenSize: Self.bf16HiddenSize, routedExperts: Self.bf16RoutedExperts,
                    nGroup: 8, topkGroup: 4, topK: 8, normTopkProb: true))
            try gate.update(
                parameters: ModuleParameters.unflattened([
                    "weight": Self.bf16GateWeight(), "e_score_correction_bias": bias,
                ]),
                verify: [.noUnusedKeys])

            let (inds, weights) = gate(Self.bf16Inputs())
            eval(inds, weights)
            XCTAssertEqual(weights.dtype, .bfloat16, file: file, line: line)

            let k = inds.dim(-1)
            let flatInds = inds.reshaped(-1).asArray(Int32.self).map(Int.init)
            let flatWeights = weights.asType(.float32).reshaped(-1).asArray(Float.self)
            var differentSets = 0
            var largestDifference: Float = 0
            for (token, want) in expected.enumerated() {
                let got = (0 ..< k)
                    .map { (flatInds[token * k + $0], flatWeights[token * k + $0]) }
                    .sorted { $0.0 < $1.0 }
                guard got.map(\.0) == want.map(\.0) else {
                    differentSets += 1
                    continue
                }
                for ((_, g), (_, w)) in zip(got, want) {
                    largestDifference = max(largestDifference, abs(g - w))
                    // The weights come back in bfloat16, rounded from float32
                    // values that equal mlx-lm's: allow half a bfloat16 step
                    // (8 significant bits) at the weight's magnitude.
                    let halfStep = Float(pow(2, log2(Double(abs(w))).rounded(.down) - 8))
                    XCTAssertEqual(
                        g, w, accuracy: halfStep + 1e-6,
                        "token \(token) weights: got \(got), mlx-lm \(want)", file: file,
                        line: line)
                }
            }
            print(
                "DeepseekV3 bf16 routing: \(differentSets) of \(expected.count) tokens select "
                    + "a different expert set; largest weight difference on matching sets "
                    + "\(largestDifference)")
            XCTAssertEqual(
                differentSets, 0,
                "\(differentSets) of \(expected.count) tokens select a different expert set "
                    + "than mlx-lm", file: file, line: line)
        }
    }

    func testBFloat16RoutingWithFloat32BiasMatchesMLXLM() throws {
        try assertBFloat16Gate(
            bias: MLXArray(Self.realisticBias), expected: Self.expectedFloat32Bias)
    }

    func testBFloat16RoutingWithBFloat16BiasMatchesMLXLM() throws {
        try assertBFloat16Gate(
            bias: MLXArray(Self.realisticBias).asType(.bfloat16),
            expected: Self.expectedBfloat16Bias)
    }

    /// With float32 gate scores the weights must come back in the activation
    /// dtype, or a float32 bias would turn the MoE output into float32.
    func testMoEBlockKeepsBFloat16WithFloat32Bias() throws {
        let moe = DeepseekV3MoE(config: try config(nGroup: 2, normTopkProb: true))
        let parameters = moe.parameters().flattened().map { key, value in
            if key.hasSuffix("e_score_correction_bias") {
                return (key, MLXArray(Self.correctionBias))
            }
            return (key, MLXRandom.normal(value.shape).asType(.bfloat16))
        }
        try moe.update(parameters: ModuleParameters.unflattened(parameters), verify: [.all])

        let y = moe(MLXRandom.normal([1, 3, Self.hiddenSize]).asType(.bfloat16))
        eval(y)
        XCTAssertEqual(moe.gate.e_score_correction_bias.dtype, .float32)
        XCTAssertEqual(y.dtype, .bfloat16)
    }
}
