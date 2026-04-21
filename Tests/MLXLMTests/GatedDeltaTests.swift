// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import XCTest

public class GatedDeltaTests: XCTestCase {

    /// Decode a minimal Qwen35 config from JSON -- small dims for fast testing.
    /// Exercises the GatedDelta kernel at T>1 during prefill.
    private func makeTestConfig() throws -> Qwen35TextConfiguration {
        let json = """
            {
                "model_type": "qwen3_5",
                "hidden_size": 64,
                "num_hidden_layers": 2,
                "intermediate_size": 128,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "linear_num_value_heads": 4,
                "linear_num_key_heads": 2,
                "linear_key_head_dim": 32,
                "linear_value_head_dim": 16,
                "linear_conv_kernel_dim": 4,
                "rms_norm_eps": 1e-6,
                "vocab_size": 100,
                "full_attention_interval": 4
            }
            """
        return try JSONDecoder().decode(
            Qwen35TextConfiguration.self, from: json.data(using: .utf8)!)
    }

    /// Test that the GatedDelta kernel produces finite, non-zero output at T>1.
    /// This catches the precision bug where bf16 state accumulation produced
    /// divergent results across recurrence steps.
    func testGatedDeltaMultiStepPrefill() throws {
        let config = try makeTestConfig()
        let model = Qwen35TextModel(config)

        // T=8 triggers multi-step GDN recurrence (the kernel path at T>1)
        let tokens = MLXArray(Array(repeating: Int32(1), count: 8))[.newAxis, .ellipsis]
        let cache = model.newCache(parameters: nil)
        let output = model(tokens, cache: cache)

        eval(output)

        let outputData = output.asArray(Float.self)
        let hasNaN = outputData.contains(where: { $0.isNaN })
        let hasInf = outputData.contains(where: { $0.isInfinite })
        let allZero = outputData.allSatisfy { $0 == 0 }

        XCTAssertFalse(hasNaN, "GDN kernel produced NaN at T>1")
        XCTAssertFalse(hasInf, "GDN kernel produced Inf at T>1")
        XCTAssertFalse(allZero, "GDN kernel produced all zeros at T>1")
        XCTAssertEqual(output.shape, [1, 8, 100])
    }

    /// Test that running the same input twice produces identical output.
    /// Precision bugs in state accumulation cause non-determinism when
    /// intermediate values overflow fp16 range.
    func testGatedDeltaDeterministic() throws {
        let config = try makeTestConfig()
        let model = Qwen35TextModel(config)

        let tokens = MLXArray(Array(repeating: Int32(1), count: 8))[.newAxis, .ellipsis]

        let cache1 = model.newCache(parameters: nil)
        let output1 = model(tokens, cache: cache1)
        eval(output1)

        let cache2 = model.newCache(parameters: nil)
        let output2 = model(tokens, cache: cache2)
        eval(output2)

        let diff = abs(output1 - output2).max()
        eval(diff)

        let maxDiff = diff.item(Float.self)
        XCTAssertEqual(maxDiff, 0.0, "GDN kernel not deterministic across runs")
    }
}
