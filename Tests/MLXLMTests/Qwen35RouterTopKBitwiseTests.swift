// Copyright © 2026 Apple Inc.
//
// Pins fusedRouterTopK bitwise-equal to chainRouterTopK. If an MLX bump
// changes the sort's tie-breaking or the reduce's accumulation, this
// fails instead of decode (fused) silently splitting from prefill (chain).

import MLX
import XCTest

@testable import MLXLMCommon

final class Qwen35RouterTopKBitwiseTests: XCTestCase {

    /// Bitwise equality, dtype and shape included; the f32 upcast is exact
    /// for f16/bf16/f32, so comparing upcast bits compares the originals.
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

    private func assertRouterMatchesChain(
        _ gates: MLXArray, k: Int, normalize: Bool, _ label: String,
        indicesOnly: Bool = false,
        file: StaticString = #filePath, line: UInt = #line
    ) {
        let (chainInds, chainScores) = chainRouterTopK(gates, k: k, normalize: normalize)
        let (fusedInds, fusedScores) = fusedRouterTopK(gates, k: k, normalize: normalize)
        eval(chainInds, chainScores, fusedInds, fusedScores)

        XCTAssertEqual(
            fusedInds.dtype, chainInds.dtype,
            "\(label): index dtype must match argPartition's", file: file, line: line)
        let a = chainInds.reshaped(-1).asArray(UInt32.self)
        let b = fusedInds.reshaped(-1).asArray(UInt32.self)
        XCTAssertEqual(a, b, "\(label): expert selection order", file: file, line: line)

        // NaN score payloads are propagation-order-dependent and out of
        // contract (production gates are softmax outputs, NaN-free).
        if !indicesOnly {
            assertBitIdentical(
                fusedScores.reshaped(-1, k), chainScores.reshaped(-1, k),
                "\(label): scores", file: file, line: line)
        }
    }

    func testFusedRouterTopKMatchesChain() {
        let rows = 64
        for (e, k) in [(256, 8), (128, 8)] {
            for dtype in [DType.float16, DType.bfloat16, DType.float32] {
                for normalize in [true, false] {
                    withRandomState(
                        MLXRandom.RandomState(seed: UInt64(e + k + (normalize ? 1 : 0)))
                    ) {
                        let tag = "E=\(e) K=\(k) \(dtype) norm=\(normalize)"

                        // The production distribution: softmax outputs.
                        let soft = MLX.softmax(
                            MLXRandom.normal([rows, e]), axis: -1, precise: true
                        ).asType(dtype)
                        assertRouterMatchesChain(soft, k: k, normalize: normalize, "softmax \(tag)")

                        // Heavy ties — the stable tie-break is the hard part.
                        let ties = (MLX.round(MLXRandom.normal([rows, e]) * 2) / 2).asType(dtype)
                        assertRouterMatchesChain(ties, k: k, normalize: normalize, "ties \(tag)")

                        // All-equal rows: pure index-order selection.
                        let equal = MLX.full([rows, e], values: MLXArray(Float(0.25))).asType(dtype)
                        assertRouterMatchesChain(
                            equal, k: k, normalize: normalize, "all-equal \(tag)")

                        // Signed zeros: compare equal, differ bitwise.
                        let signs = MLX.where(
                            MLXRandom.uniform(low: Float(0), high: 1, [rows, e]) .< 0.5,
                            MLXArray(Float(-0.0)), MLXArray(Float(0.0)))
                        let zeros = MLX.where(
                            MLXRandom.uniform(low: Float(0), high: 1, [rows, e]) .< 0.5,
                            signs, MLXRandom.normal([rows, e])
                        ).asType(dtype)
                        assertRouterMatchesChain(zeros, k: k, normalize: normalize, "±0 \(tag)")

                        // NaN above everything, all NaNs tie; indices only.
                        var nans = MLX.where(
                            MLXRandom.uniform(low: Float(0), high: 1, [rows, e]) .< 0.05,
                            MLXArray(Float.nan), MLXRandom.normal([rows, e])
                        ).asType(dtype)
                        nans[0] = MLX.full([e], values: MLXArray(Float.nan)).asType(dtype)
                        assertRouterMatchesChain(
                            nans, k: k, normalize: normalize, "NaN \(tag)", indicesOnly: true)
                    }
                }
            }
        }
    }

    func testMoERouterTopKMatchesChainAcrossDispatchPaths() {
        let cases = [
            (shape: [1, 256], k: 8, label: "single-row fused"),
            (shape: [4, 256], k: 8, label: "multi-row chain"),
            (shape: [1, 1025], k: 8, label: "expert-limit chain"),
        ]

        for testCase in cases {
            let gates = MLX.softmax(
                MLXRandom.normal(testCase.shape), axis: -1, precise: true
            ).asType(.float16)
            let (wantInds, wantScores) = chainRouterTopK(
                gates, k: testCase.k, normalize: true)
            let (gotInds, gotScores) = moeRouterTopK(
                gates, k: testCase.k, normalize: true)
            eval(wantInds, wantScores, gotInds, gotScores)

            XCTAssertEqual(
                gotInds.reshaped(-1).asArray(UInt32.self),
                wantInds.reshaped(-1).asArray(UInt32.self),
                "\(testCase.label): expert selection order")
            assertBitIdentical(
                gotScores.reshaped(-1, testCase.k),
                wantScores.reshaped(-1, testCase.k),
                "\(testCase.label): scores")
        }
    }
}
