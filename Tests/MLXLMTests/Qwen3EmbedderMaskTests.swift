// Copyright © 2026 Apple Inc.
//
// The MLXEmbedders Qwen3 key-padding fold: real tokens must not attend to
// padding keys, and the fold must line up with whatever key window the causal
// mask is sized for.

import Foundation
import MLX
import MLXLMCommon
import XCTest

@testable import MLXEmbedders

final class Qwen3EmbedderMaskTests: XCTestCase {

    private func makeTinyModel() throws -> Qwen3Model {
        let json = """
            {
                "hidden_size": 32,
                "num_hidden_layers": 2,
                "intermediate_size": 64,
                "num_attention_heads": 4,
                "head_dim": 8,
                "rms_norm_eps": 1e-5,
                "vocab_size": 512,
                "num_key_value_heads": 2,
                "rope_theta": 10000.0
            }
            """
        let config = try JSONDecoder().decode(Qwen3Configuration.self, from: Data(json.utf8))
        // Pin the initializer weights. Task-local rather than MLXRandom.seed:
        // parallel tests must not share (or perturb) the global random stream.
        return withRandomState(MLXRandom.RandomState(seed: 1)) { Qwen3Model(config) }
    }

    /// Changing only a masked pad token must leave the real positions' hidden
    /// states untouched; without the mask the pad key is visible and they move.
    func testPaddingMaskHidesPaddedKeys() throws {
        let model = try makeTinyModel()
        var tokensB: [Int32] = [7, 11, 13, 17]
        tokensB[0] = 23
        let a = MLXArray([Int32(7), 11, 13, 17]).expandedDimensions(axis: 0)
        let b = MLXArray(tokensB).expandedDimensions(axis: 0)
        let padding = MLXArray([Int32(0), 1, 1, 1]).expandedDimensions(axis: 0)

        let maskedA = model(a, attentionMask: padding).hiddenStates!
        let maskedB = model(b, attentionMask: padding).hiddenStates!
        eval(maskedA, maskedB)
        XCTAssertLessThanOrEqual(
            abs(maskedA[0..., 1..., 0...] - maskedB[0..., 1..., 0...]).max().item(Float.self),
            1e-5)

        let openA = model(a).hiddenStates!
        let openB = model(b).hiddenStates!
        eval(openA, openB)
        XCTAssertGreaterThan(
            abs(openA[0..., 1..., 0...] - openB[0..., 1..., 0...]).max().item(Float.self),
            1e-3)
    }
}
