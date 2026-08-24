// Copyright © 2025 Apple Inc.

import Foundation
import MLX
import XCTest

@testable import MLXLLM
@testable import MLXLMCommon

/// Offline architecture checks for the Helium port. No weights, no network:
/// builds `HeliumModel` from a tiny synthetic config and exercises the forward
/// pass + `sanitize`. Mirrors the pattern in `FalconH1Tests` / `Mamba2Tests`.
///
/// A clean real-weight load already proves parameter keys/shapes match the
/// checkpoint (`Load.swift` uses `verify: [.all]`), so these tests only guard
/// the shape/plumbing/sanitize logic that a synthetic config can reach. Numeric
/// forward-pass parity against Python mlx-lm is a separate (network) exercise.
final class HeliumTests: XCTestCase {

    private func tinyConfiguration() throws -> HeliumConfiguration {
        let json = """
            {
                "hidden_size": 32,
                "num_hidden_layers": 2,
                "intermediate_size": 64,
                "num_attention_heads": 4,
                "num_key_value_heads": 4,
                "rms_norm_eps": 1e-5,
                "vocab_size": 100,
                "rope_theta": 100000,
                "tie_word_embeddings": false
            }
            """.data(using: .utf8)!
        return try JSONDecoder().decode(HeliumConfiguration.self, from: json)
    }

    func testForwardPassProducesLogitsShape() throws {
        let model = HeliumModel(try tinyConfiguration())
        let inputs = MLXArray([0, 1, 2, 3, 4] as [Int32]).reshaped(1, 5)
        let logits = model(inputs, cache: nil)
        eval(logits)
        XCTAssertEqual(logits.shape, [1, 5, 100])
    }

    func testSanitizeDropsRotaryInvFreq() throws {
        let model = HeliumModel(try tinyConfiguration())
        let sanitized = model.sanitize(weights: [
            "model.layers.0.self_attn.rotary_emb.inv_freq": MLXArray.zeros([8]),
            "model.embed_tokens.weight": MLXArray.zeros([100, 32]),
        ])
        XCTAssertNil(sanitized["model.layers.0.self_attn.rotary_emb.inv_freq"])
        XCTAssertNotNil(sanitized["model.embed_tokens.weight"])
    }
}
