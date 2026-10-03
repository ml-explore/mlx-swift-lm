// Copyright © 2026 Apple Inc.
//
// The Qwen3.5 and Qwen3-Next gated delta nets must normalize q and k like the
// reference l2norm, x / sqrt(sum(x^2) + 1e-6). Each test builds the layer
// output again from the layer's own modules with that l2norm and compares.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXLLM
@testable import MLXVLM

final class GatedDeltaQKNormTests: XCTestCase {

    // One key head keeps q, k and v contiguous in every input projection.
    private let hiddenSize = 32
    private let numVHeads = 2
    private let headKDim = 64
    private let headVDim = 16
    private let convKernelSize = 4

    private var configurationJSON: Data {
        Data(
            """
            {
                "model_type": "qwen3_5",
                "hidden_size": \(hiddenSize),
                "num_hidden_layers": 1,
                "intermediate_size": 64,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 16,
                "vocab_size": 32,
                "linear_num_value_heads": \(numVHeads),
                "linear_num_key_heads": 1,
                "linear_key_head_dim": \(headKDim),
                "linear_value_head_dim": \(headVDim),
                "linear_conv_kernel_dim": \(convKernelSize),
                "rms_norm_eps": 1e-6,
                "num_experts": 0, "num_experts_per_tok": 0, "decoder_sparse_step": 1,
                "shared_expert_intermediate_size": 64, "mlp_only_layers": [],
                "moe_intermediate_size": 64,
                "rope_theta": 10000.0, "partial_rotary_factor": 0.25,
                "max_position_embeddings": 128
            }
            """.utf8)
    }

    /// Token scales from 1 to 1e-6 put sum(q^2) and sum(k^2) above, near and
    /// below the eps.
    private func makeInputs() -> MLXArray {
        let scales = (0 ..< 28).map { Float(pow(10, -Double($0 / 4))) }
        let x = MLXRandom.normal([1, scales.count, hiddenSize])
        return x * MLXArray(scales).reshaped(1, -1, 1)
    }

    private func l2norm(_ x: MLXArray) -> MLXArray {
        x * rsqrt((x * x).sum(axis: -1, keepDims: true) + 1e-6)
    }

    /// The layer output with the reference q/k normalization. `mixedQKV` is
    /// the conv input, laid out as [q, k, v].
    private func referenceOutput(
        mixedQKV: MLXArray, z: MLXArray, b: MLXArray, a: MLXArray,
        conv1d: Conv1d, aLog: MLXArray, dtBias: MLXArray,
        norm: (MLXArray, MLXArray) -> MLXArray, outProj: Linear
    ) -> MLXArray {
        let B = mixedQKV.dim(0)
        let S = mixedQKV.dim(1)
        let convState = MLXArray.zeros([B, convKernelSize - 1, mixedQKV.dim(-1)])
        let convOut = silu(conv1d(concatenated([convState, mixedQKV], axis: 1)))
        let parts = MLX.split(convOut, indices: [headKDim, 2 * headKDim], axis: -1)
        let q = l2norm(parts[0].reshaped(B, S, 1, headKDim)) * pow(Float(headKDim), -0.5)
        let k = l2norm(parts[1].reshaped(B, S, 1, headKDim))
        let v = parts[2].reshaped(B, S, numVHeads, headVDim)
        let (out, _) = gatedDeltaUpdate(
            q: q, k: k, v: v, a: a, b: b, aLog: aLog, dtBias: dtBias, useKernel: false)
        let gated = norm(out, z.reshaped(B, S, numVHeads, headVDim))
        return outProj(gated.reshaped(B, S, -1))
    }

    /// Compares per token, because the small tokens have small outputs.
    private func assertMatches(
        _ output: MLXArray, _ reference: MLXArray,
        file: StaticString = #filePath, line: UInt = #line
    ) {
        let error = MLX.abs(output - reference).max(axis: -1) / MLX.abs(reference).max(axis: -1)
        let worst = error.max().item(Float.self)
        XCTAssertLessThan(worst, 1e-4, "max per-token relative error", file: file, line: line)
    }

    func testQwen35() throws {
        let config = try JSONDecoder().decode(
            Qwen35TextConfiguration.self, from: configurationJSON)
        withRandomState(MLXRandom.RandomState(seed: 0)) {
            let gdn = Qwen35GatedDeltaNet(config)
            let x = makeInputs()
            let reference = referenceOutput(
                mixedQKV: gdn.inProjQKV(x), z: gdn.inProjZ(x),
                b: gdn.inProjB(x), a: gdn.inProjA(x),
                conv1d: gdn.conv1d, aLog: gdn.aLog, dtBias: gdn.dtBias,
                norm: { gdn.norm($0, gate: $1) }, outProj: gdn.outProj)
            assertMatches(gdn(x), reference)
        }
    }

    func testQwen35VLM() throws {
        let config = try JSONDecoder().decode(
            MLXVLM.Qwen35Configuration.TextConfiguration.self, from: configurationJSON)
        withRandomState(MLXRandom.RandomState(seed: 0)) {
            let gdn = Qwen35Language.GatedDeltaNet(config)
            let x = makeInputs()
            let reference = referenceOutput(
                mixedQKV: gdn.inProjQKV(x), z: gdn.inProjZ(x),
                b: gdn.inProjB(x), a: gdn.inProjA(x),
                conv1d: gdn.conv1d, aLog: gdn.aLog, dtBias: gdn.dtBias,
                norm: { gdn.norm($0, gate: $1) }, outProj: gdn.outProj)
            assertMatches(gdn(x), reference)
        }
    }

    func testQwen3Next() throws {
        let config = try JSONDecoder().decode(
            Qwen3NextConfiguration.self, from: configurationJSON)
        withRandomState(MLXRandom.RandomState(seed: 0)) {
            let gdn = Qwen3NextGatedDeltaNet(config)
            let x = makeInputs()
            // With one key head, in_proj_qkvz is [q, k, v, z] and in_proj_ba is [b, a].
            let qkvz = gdn.inProjQKVZ(x)
            let ba = gdn.inProjBA(x)
            let qkvDim = 2 * headKDim + numVHeads * headVDim
            let reference = referenceOutput(
                mixedQKV: qkvz[.ellipsis, ..<qkvDim], z: qkvz[.ellipsis, qkvDim...],
                b: ba[.ellipsis, ..<numVHeads], a: ba[.ellipsis, numVHeads...],
                conv1d: gdn.conv1d, aLog: gdn.aLog, dtBias: gdn.dtBias,
                norm: { gdn.norm($0, gate: $1) }, outProj: gdn.outProj)
            assertMatches(gdn(x), reference)
        }
    }
}
