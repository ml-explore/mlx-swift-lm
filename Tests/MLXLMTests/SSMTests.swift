// Copyright © 2026 Apple Inc.

import MLX
import MLXLLM
import Testing

@Test func testSSMAttnSupportsGradientTransformsAcrossChunks() {
    let x = MLXArray.ones([1, 5, 2, 2])
    let aLog = MLXArray.zeros([2])
    let inputMixing = MLXArray.ones([1, 5, 1, 3])
    let outputMixing = MLXArray.ones([1, 5, 1, 3])
    let residualScale = MLXArray.ones([2])
    let dt = MLXArray.ones([1, 5, 2])
    let dtBias = MLXArray.zeros([2])

    let gradient = grad { input in
        let (output, state) = ssmAttn(
            x: input,
            ALog: aLog,
            B: inputMixing,
            C: outputMixing,
            D: residualScale,
            dt: dt,
            dtBias: dtBias,
            step: 2
        )
        return output.sum() + state.sum()
    }(x)
    eval(gradient)

    let magnitude = gradient.abs().sum().item(Float.self)
    #expect(gradient.shape == x.shape)
    #expect(magnitude.isFinite)
    #expect(magnitude > 0)
}

@Test func testSSMAttnPreservesRecurrentStateDTypeAcrossChunks() throws {
    MLXRandom.seed(7)

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
