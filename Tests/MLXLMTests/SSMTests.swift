// Copyright © 2026 Apple Inc.

import MLX
import MLXLLM
import Testing

private func expectAllClose(
    _ actual: MLXArray, _ expected: MLXArray, label: String, rtol: Double = 1e-4,
    atol: Double = 1e-5
) {
    guard actual.shape == expected.shape else {
        Issue.record("\(label) shape \(actual.shape) != \(expected.shape)")
        return
    }
    #expect(
        allClose(actual, expected, rtol: rtol, atol: atol).item(Bool.self),
        "\(label) values differ")
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
                    rtol: 5e-4, atol: 5e-5)
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
