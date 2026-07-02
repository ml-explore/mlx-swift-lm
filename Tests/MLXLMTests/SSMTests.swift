// Copyright © 2026 Apple Inc.

import MLX
import MLXLLM
import Testing

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
