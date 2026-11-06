// Copyright © 2026 Apple Inc.
//
// The Qwen3.5 VLM rotary must apply YaRN from rope_parameters like HF
// `_compute_yarn_parameters`, and leave non-YaRN configs unchanged.

import Foundation
import MLX
import MLXLMCommon
import XCTest

@testable import MLXVLM

final class Qwen35VLMYarnTests: XCTestCase {

    // sensenova/SenseNova-6.8-Code2Video text_config: head_dim 256, partial 0.25.
    private let dim = 64
    private let base: Float = 10_000_000
    private let yarn: [String: StringOrNumber] = [
        "type": .string("yarn"),
        "rope_type": .string("yarn"),
        "factor": .float(2.0),
        "original_max_position_embeddings": .int(262_144),
        "rope_theta": .float(10_000_000),
        "mrope_section": .ints([11, 11, 10]),
    ]

    // transformers 5.22.1 `ROPE_INIT_FUNCTIONS["yarn"]` on the same config.
    private let hfInvFreq: [Float] = [
        1, 0.604296386, 0.365174145, 0.220673397, 0.133352146, 0.0805842131,
        0.0486967526, 0.0294272732, 0.0177827943, 0.0107460786, 0.00649381662,
        0.0039241896, 0.00237137382, 0.00143301266, 0.000865964335, 0.000490592909,
        0.000276699313, 0.000155264934, 8.66086484e-05, 4.79758528e-05, 2.63560341e-05,
        1.43341686e-05, 7.69963208e-06, 4.6528603e-06, 2.81170674e-06, 1.69910413e-06,
        1.02676245e-06, 6.20468882e-07, 3.7494712e-07, 2.26579189e-07, 1.36920988e-07,
        8.27408542e-08,
    ]
    private let hfAttentionFactor: Float = 1.0693147180559945

    func testYarnParametersMatchHF() {
        let (invFreq, attentionFactor) = Qwen35Language.yarnParameters(
            dim: dim, base: base, maxPositionEmbeddings: 524_288, yarn)
        XCTAssertEqual(invFreq.count, hfInvFreq.count)
        for (ours, ref) in zip(invFreq, hfInvFreq) {
            XCTAssertEqual(ours, ref, accuracy: ref * 1e-5)
        }
        XCTAssertEqual(attentionFactor, hfAttentionFactor, accuracy: 1e-6)
    }

    func testMissingOriginalMaxFallsBackToMaxPositionEmbeddings() {
        var withoutOriginal = yarn
        withoutOriginal["original_max_position_embeddings"] = nil
        let fallback = Qwen35Language.yarnParameters(
            dim: dim, base: base, maxPositionEmbeddings: 262_144, withoutOriginal)
        let explicit = Qwen35Language.yarnParameters(
            dim: dim, base: base, maxPositionEmbeddings: 524_288, yarn)
        XCTAssertEqual(fallback.invFreq, explicit.invFreq)
        XCTAssertEqual(fallback.attentionFactor, explicit.attentionFactor)
    }

    func testYarnScalesCosAndSin() {
        let plain = Qwen35Language.RotaryEmbedding(
            dim: dim, base: base, mropeSection: [11, 11, 10])
        let scaled = Qwen35Language.RotaryEmbedding(
            dim: dim, base: base, mropeSection: [11, 11, 10], ropeParameters: yarn)
        XCTAssertEqual(plain.attentionScaling, 1)
        XCTAssertEqual(scaled.attentionScaling, hfAttentionFactor, accuracy: 1e-6)

        let x = MLXArray.zeros([1, 1, 1, 256], dtype: .float32)
        let positions = MLXArray([Int32(0)]).reshaped(1, 1)
        let (cosValues, sinValues) = scaled(x: x, positionIds: positions)
        XCTAssertEqual(cosValues.max().item(Float.self), hfAttentionFactor, accuracy: 1e-6)
        XCTAssertEqual(sinValues.abs().max().item(Float.self), 0)
    }

    func testNonYarnConfigIsUnchanged() {
        let defaultParameters: [String: StringOrNumber] = [
            "type": .string("default"), "mrope_section": .ints([11, 11, 10]),
        ]
        let plain = Qwen35Language.RotaryEmbedding(
            dim: dim, base: base, mropeSection: [11, 11, 10])
        let withDefault = Qwen35Language.RotaryEmbedding(
            dim: dim, base: base, mropeSection: [11, 11, 10], ropeParameters: defaultParameters)
        XCTAssertEqual(withDefault.attentionScaling, 1)

        let x = MLXArray.zeros([1, 1, 7, 256], dtype: .bfloat16)
        let positions = MLXArray(0 ..< 7).asType(.int32).reshaped(1, 7)
        let (c0, s0) = plain(x: x, positionIds: positions)
        let (c1, s1) = withDefault(x: x, positionIds: positions)
        XCTAssertTrue(arrayEqual(c0, c1).item(Bool.self))
        XCTAssertTrue(arrayEqual(s0, s1).item(Bool.self))
    }
}
