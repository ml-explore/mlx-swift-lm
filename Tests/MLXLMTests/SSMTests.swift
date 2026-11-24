// Copyright © 2026 Apple Inc.

import MLX
import MLXLLM
import Testing

/// Compare two float32 results that differ only in how the work was split.
///
/// The tolerances are wider on machines whose float32 matmuls run as TF32 — see
/// ``MatmulPrecision``. The message reports the worst element so a failure says how far off it
/// was, not just that it was.
private func expectAllClose(
    _ actual: MLXArray, _ expected: MLXArray, label: String,
    rtol: Double = MatmulPrecision.tolerance(float32: 1e-4, reduced: 5e-3),
    atol: Double = MatmulPrecision.tolerance(float32: 1e-5, reduced: 5e-4)
) {
    guard actual.shape == expected.shape else {
        Issue.record("\(label) shape \(actual.shape) != \(expected.shape)")
        return
    }
    #expect(
        allClose(actual, expected, rtol: rtol, atol: atol).item(Bool.self),
        "\(label) values differ by up to \(MLX.abs(actual - expected).max().item(Float.self))")
}

@Test func testSSMAttnChunkedMatchesUnchunkedValuesAndGradients() {
    withRandomState(MLXRandom.RandomState(seed: 11)) {
        let batch = 1
        let sequence = 5
        let heads = 4
        let headDimension = 3
        let groups = 2
        let stateDimension = 3

        let parameters = [
            MLXRandom.normal([batch, sequence, heads, headDimension]),
            MLXRandom.normal([heads]) * 0.1,
            MLXRandom.normal([batch, sequence, groups, stateDimension]) * 0.2,
            MLXRandom.normal([batch, sequence, groups, stateDimension]) * 0.2,
            MLXRandom.normal([heads]) * 0.2,
            MLXRandom.normal([batch, sequence, heads]) * 0.1,
            MLXRandom.normal([heads]) * 0.1,
            MLXRandom.normal([batch, heads, headDimension, stateDimension]) * 0.2,
        ]

        func forward(_ parameters: [MLXArray], step: Int) -> (MLXArray, MLXArray) {
            ssmAttn(
                x: parameters[0],
                ALog: parameters[1],
                B: parameters[2],
                C: parameters[3],
                D: parameters[4],
                dt: parameters[5],
                dtBias: parameters[6],
                state: parameters[7],
                step: step
            )
        }

        let (referenceOutput, referenceState) = forward(parameters, step: sequence)

        func valueAndGradient(step: Int) -> ([MLXArray], [MLXArray]) {
            valueAndGrad(
                { parameters in
                    let (output, state) = forward(parameters, step: step)
                    return [(output.square().mean() + state.square().mean())]
                },
                argumentNumbers: parameters.indices
            )(parameters)
        }

        let (referenceValue, referenceGradients) = valueAndGradient(step: sequence)
        #expect(referenceGradients.count == parameters.count)
        let gradientLabels = ["x", "ALog", "B", "C", "D", "dt", "dtBias", "state"]

        for step in [1, 2, 3] {
            let (chunkedOutput, chunkedState) = forward(parameters, step: step)
            let (chunkedValue, chunkedGradients) = valueAndGradient(step: step)
            eval(
                chunkedOutput, chunkedState, chunkedValue, chunkedGradients, referenceOutput,
                referenceState, referenceValue, referenceGradients)

            expectAllClose(chunkedOutput, referenceOutput, label: "output (step \(step))")
            expectAllClose(chunkedState, referenceState, label: "state (step \(step))")
            expectAllClose(chunkedValue[0], referenceValue[0], label: "loss (step \(step))")
            #expect(chunkedGradients.count == parameters.count)
            for (index, pair) in zip(chunkedGradients, referenceGradients).enumerated() {
                expectAllClose(
                    pair.0, pair.1, label: "\(gradientLabels[index]) gradient (step \(step))",
                    rtol: MatmulPrecision.tolerance(float32: 5e-4, reduced: 5e-3),
                    atol: MatmulPrecision.tolerance(float32: 5e-5, reduced: 5e-4))
            }
        }
    }
}

@Test func testSSMAttnFinalStateGradientCrossesChunkBoundaries() {
    withRandomState(MLXRandom.RandomState(seed: 12)) {
        let sequence = 5
        let x = MLXRandom.normal([1, sequence, 2, 2])
        let aLog = MLXRandom.normal([2]) * 0.1
        let inputMixing = MLXRandom.normal([1, sequence, 1, 3]) * 0.2
        let outputMixing = MLXArray.zeros([1, sequence, 1, 3])
        let residualScale = MLXArray.zeros([2])
        let dt = MLXRandom.normal([1, sequence, 2]) * 0.1
        let dtBias = MLXRandom.normal([2]) * 0.1

        let gradient = grad { input in
            let (_, state) = ssmAttn(
                x: input,
                ALog: aLog,
                B: inputMixing,
                C: outputMixing,
                D: residualScale,
                dt: dt,
                dtBias: dtBias,
                step: 2
            )
            return state.square().sum()
        }(x)
        eval(gradient)

        let firstChunkMagnitude = gradient[0..., ..<2, 0..., 0...].abs().sum().item(Float.self)
        #expect(firstChunkMagnitude.isFinite)
        #expect(firstChunkMagnitude > 0, "final state is disconnected from the first chunk")
    }
}

@Test func testSSMAttnPreservesRecurrentStateDTypeAcrossChunks() throws {
    withRandomState(MLXRandom.RandomState(seed: 7)) {
        let dtype = DType.bfloat16
        let batch = 1
        let sequence = 5
        let heads = 4
        let headDim = 3
        let groups = 2
        let stateDim = 8

        let x = MLXRandom.normal([batch, sequence, heads, headDim]).asType(dtype)
        let aLog = MLXRandom.normal([heads]).asType(dtype)
        let B = MLXRandom.normal([batch, sequence, groups, stateDim]).asType(dtype)
        let C = MLXRandom.normal([batch, sequence, groups, stateDim]).asType(dtype)
        let D = MLXRandom.normal([heads]).asType(dtype)
        let dt = MLXRandom.normal([batch, sequence, heads]).asType(dtype)
        let dtBias = MLXRandom.normal([heads]).asType(dtype)

        let (freshY, freshState) = ssmAttn(
            x: x,
            ALog: aLog,
            B: B,
            C: C,
            D: D,
            dt: dt,
            dtBias: dtBias,
            step: 2
        )
        eval(freshY, freshState)

        #expect(freshY.dtype == dtype)
        #expect(freshState.dtype == dtype)

        let previousState = MLXRandom.normal([batch, heads, headDim, stateDim]).asType(dtype)
        let (continuedY, continuedState) = ssmAttn(
            x: x,
            ALog: aLog,
            B: B,
            C: C,
            D: D,
            dt: dt,
            dtBias: dtBias,
            state: previousState,
            step: 2
        )
        eval(continuedY, continuedState)

        #expect(continuedY.dtype == dtype)
        #expect(continuedState.dtype == previousState.dtype)
    }
}

/// Compare two `segsum` results that may hold -inf at masked positions.
private func expectSameSegsum(_ actual: MLXArray, _ expected: MLXArray, label: String) {
    guard actual.shape == expected.shape else {
        Issue.record("\(label) shape \(actual.shape) != \(expected.shape)")
        return
    }
    #expect(
        MLX.arrayEqual(MLX.isInf(actual), MLX.isInf(expected)).item(Bool.self),
        "\(label) masks differ")
    let zero = MLXArray(Float(0))
    #expect(
        MLX.allClose(
            MLX.which(MLX.isInf(expected), zero, actual),
            MLX.which(MLX.isInf(expected), zero, expected)
        ).item(Bool.self), "\(label) values differ")
}

/// `segsum` with a (B, L) mask and a (B, H, L) input must apply row b's mask to
/// every head of row b, as running each row on its own does. When B == H the
/// pair mask built from the unexpanded mask still broadcasts, but against the
/// heads, so head h of every row got row h's mask.
@Test func testSegsumBatchMaskMatchesPerRowSegsum() {
    withRandomState(MLXRandom.RandomState(seed: 13)) {
        let batch = 2
        let heads = 2
        let length = 4
        let x = MLXRandom.normal([batch, heads, length])
        let mask = MLXArray([1, 1, 1, 1, 0, 0, 1, 1].map { Float($0) }, [batch, length])

        let batched = segsum(x, mask: mask)
        #expect(batched.shape == [batch, heads, length, length])

        for row in 0 ..< batch {
            let single = segsum(x[row ..< (row + 1)], mask: mask[row ..< (row + 1)])
            expectSameSegsum(batched[row ..< (row + 1)], single, label: "row \(row)")
        }
    }
}

/// With B > 1, H > 1 and B != H the unexpanded pair mask, (B, L, L), does not
/// broadcast against (B, H, L, L) at all.
@Test func testSegsumBatchMaskBroadcastsAcrossHeads() {
    withRandomState(MLXRandom.RandomState(seed: 14)) {
        let batch = 2
        let heads = 3
        let length = 5
        let x = MLXRandom.normal([batch, heads, length])
        let mask = MLXArray([1, 1, 1, 1, 1, 0, 0, 1, 1, 1].map { Float($0) }, [batch, length])

        let masked = segsum(x, mask: mask)
        #expect(masked.shape == [batch, heads, length, length])

        // An all-ones mask is the same as no mask.
        expectSameSegsum(
            segsum(x, mask: MLXArray.ones([batch, length])), segsum(x), label: "all-ones")

        // Row 1 pads columns 0 and 1, so those are -inf in every head.
        for head in 0 ..< heads {
            let padded = masked[1, head, 2, ..<2].asArray(Float.self)
            #expect(padded.allSatisfy { $0 == -Float.infinity }, "head \(head)")
        }
    }
}
