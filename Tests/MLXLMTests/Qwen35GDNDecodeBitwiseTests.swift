// Copyright © 2026 Apple Inc.
//
// CI pin for the bitwise contract GDN decode rests on: `decodeConv`
// (elementwise multiply-adds, f32 accumulation) must reproduce MLX's
// `Convolution` kernel bit-for-bit, or compiled decode silently diverges
// from prefill after an MLX bump.

import Foundation
import MLX
import MLXNN
import XCTest

@testable import MLXLLM
@testable import MLXLMCommon

final class Qwen35GDNDecodeBitwiseTests: XCTestCase {

    /// Bitwise equality, dtype and shape included. The f32 upcast is
    /// injective for f16/bf16/f32 finite values, so bit-comparing the upcast
    /// compares the originals.
    private func assertBitIdentical(
        _ got: MLXArray, _ want: MLXArray, _ label: String,
        file: StaticString = #filePath, line: UInt = #line
    ) {
        XCTAssertEqual(got.dtype, want.dtype, "\(label): dtype", file: file, line: line)
        XCTAssertEqual(got.shape, want.shape, "\(label): shape", file: file, line: line)
        let a = got.asType(.float32).asArray(Float.self)
        let b = want.asType(.float32).asArray(Float.self)
        let mismatches = zip(a, b).filter { $0.bitPattern != $1.bitPattern }.count
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
            gdn.update(parameters: gdn.parameters().mapValues { $0.asType(dtype) })

            let qkv = MLXRandom.normal([1, 1, gdn.convDim]).asType(dtype)
            let convState = MLXRandom.normal(
                [1, gdn.convKernelSize - 1, gdn.convDim]
            ).asType(dtype)
            eval(qkv, convState)

            let (convOut, newConvState) = gdn.decodeConv(convState: convState, qkv: qkv)

            let convInput = concatenated([convState, qkv], axis: 1)
            let refOut = silu(gdn.conv1d(convInput))
            let refState = contiguous(convInput[0..., (-(gdn.convKernelSize - 1))..., 0...])
            eval(convOut, newConvState, refOut, refState)

            assertBitIdentical(convOut, refOut, "conv output (\(dtype))")
            assertBitIdentical(newConvState, refState, "conv state (\(dtype))")
        }
    }
}
