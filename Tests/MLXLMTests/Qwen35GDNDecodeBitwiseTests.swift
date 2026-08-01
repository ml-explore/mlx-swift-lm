// Copyright © 2026 Apple Inc.
//
// Pins `decodeConv` bit-for-bit against `generalConv` (MLX's `Convolution`
// kernel); a mismatch means compiled decode silently diverges from prefill.

import Foundation
import MLX
import MLXNN
import XCTest

@testable import MLXLLM

final class Qwen35GDNDecodeBitwiseTests: XCTestCase {

    /// The f32 upcast is injective on finite f16/bf16/f32 values, so
    /// bit-comparing the upcast compares the originals.
    private func assertBitIdentical(
        _ got: MLXArray, _ want: MLXArray, _ label: String,
        file: StaticString = #filePath, line: UInt = #line
    ) {
        XCTAssertEqual(got.dtype, want.dtype, "\(label): dtype", file: file, line: line)
        XCTAssertEqual(got.shape, want.shape, "\(label): shape", file: file, line: line)
        let a = got.asType(.float32).asArray(Float.self)
        let b = want.asType(.float32).asArray(Float.self)
        let mismatches = zip(a, b).lazy.filter { $0.bitPattern != $1.bitPattern }.count
        XCTAssertEqual(
            mismatches, 0, "\(label): \(mismatches)/\(a.count) elements differ bitwise",
            file: file, line: line)
    }

    /// 256 conv channels — wide enough that an accumulation-order change in
    /// MLX's Convolution kernel cannot pass by coincidence (native-dtype
    /// accumulation diverges in ~47% of channels).
    private func tinyGDNConfiguration() throws -> Qwen35TextConfiguration {
        let json = """
            {
                "model_type": "qwen3_5_moe",
                "hidden_size": 64,
                "num_hidden_layers": 2,
                "intermediate_size": 64,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 32,
                "linear_num_value_heads": 4,
                "linear_num_key_heads": 2,
                "linear_key_head_dim": 32,
                "linear_value_head_dim": 32,
                "linear_conv_kernel_dim": 4,
                "vocab_size": 32,
                "full_attention_interval": 2,
                "num_experts": 16,
                "num_experts_per_tok": 4,
                "moe_intermediate_size": 32,
                "shared_expert_intermediate_size": 32
            }
            """
        return try JSONDecoder().decode(
            Qwen35TextConfiguration.self, from: Data(json.utf8))
    }

    func testDecodeConvMatchesConvolutionKernelBitwise() throws {
        let config = try tinyGDNConfiguration()
        for dtype in [DType.float16, DType.bfloat16] {
            MLXRandom.seed(11)
            let gdn = Qwen35GatedDeltaNet(config)
            gdn.conv1d.update(
                parameters: gdn.conv1d.parameters().mapValues { $0.asType(dtype) })

            let qkv = MLXRandom.normal([1, 1, gdn.convDim]).asType(dtype)
            let convState = MLXRandom.normal(
                [1, gdn.convKernelSize - 1, gdn.convDim]
            ).asType(dtype)
            eval(qkv, convState)

            let (conv, state) = gdn.decodeConv(convState: convState, qkv: qkv)
            let (refConv, refState) = gdn.generalConv(convState: convState, qkv: qkv)
            eval(conv, state, refConv, refState)

            assertBitIdentical(conv, refConv, "conv output (\(dtype))")
            assertBitIdentical(state, refState, "conv state (\(dtype))")
        }
    }
}
