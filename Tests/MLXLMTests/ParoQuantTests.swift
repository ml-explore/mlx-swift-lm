// Copyright © 2025 INQTR. All rights reserved.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

public class ParoQuantTests: XCTestCase {

    // MARK: - Pair Packing

    func testPairPackingEncodesCorrectly() {
        let groupSize = 8
        let krot = 2
        let dim = 16  // 2 groups of 8

        // krot=0: [0,1,2,3,4,5,6,7, 0,1,2,3,4,5,6,7]
        // krot=1: [7,6,5,4,3,2,1,0, 7,6,5,4,3,2,1,0]
        let row0 = (0 ..< dim).map { Int16($0 % groupSize) }
        let row1 = (0 ..< dim).map { Int16((groupSize - 1) - ($0 % groupSize)) }
        let pairs = MLXArray(row0 + row1).reshaped(krot, dim)

        let packed = packPairsForTesting(pairs, groupSize: groupSize)
        XCTAssertEqual(packed.shape, [krot, dim / 2])

        let values = packed.asArray(Int32.self)

        // krot=0, group 0: pairs [0,1,2,3,4,5,6,7]
        //   even=[0,2,4,6], odd=[1,3,5,7]  →  packed = lo | (hi << 16)
        XCTAssertEqual(values[0] & 0xFFFF, 0)
        XCTAssertEqual(values[0] >> 16, 1)
        XCTAssertEqual(values[1] & 0xFFFF, 2)
        XCTAssertEqual(values[1] >> 16, 3)
        XCTAssertEqual(values[2] & 0xFFFF, 4)
        XCTAssertEqual(values[2] >> 16, 5)
        XCTAssertEqual(values[3] & 0xFFFF, 6)
        XCTAssertEqual(values[3] >> 16, 7)

        // krot=1, group 0: pairs [7,6,5,4,3,2,1,0]
        //   even=[7,5,3,1], odd=[6,4,2,0]
        let offset = dim / 2
        XCTAssertEqual(values[offset + 0] & 0xFFFF, 7)
        XCTAssertEqual(values[offset + 0] >> 16, 6)
        XCTAssertEqual(values[offset + 1] & 0xFFFF, 5)
        XCTAssertEqual(values[offset + 1] >> 16, 4)
    }

    func testPairPackingRoundTrip() {
        let groupSize = 128
        let krot = 8
        let dim = 256

        let pairs = makeRandomPairs(krot: krot, dim: dim, groupSize: groupSize)
        let packed = packPairsForTesting(pairs, groupSize: groupSize)

        let packedValues = packed.asArray(Int32.self)
        let originalValues = pairs.asArray(Int16.self)

        for k in 0 ..< krot {
            for g in 0 ..< (dim / groupSize) {
                for t in 0 ..< (groupSize / 2) {
                    let packedIdx = k * (dim / 2) + g * (groupSize / 2) + t
                    let lo = packedValues[packedIdx] & 0xFFFF
                    let hi = packedValues[packedIdx] >> 16

                    let evenIdx = k * dim + g * groupSize + t * 2
                    let oddIdx = evenIdx + 1

                    XCTAssertEqual(lo, Int32(originalValues[evenIdx]))
                    XCTAssertEqual(hi, Int32(originalValues[oddIdx]))
                }
            }
        }
    }

    // MARK: - AutoAWQ Conversion

    /// Verifies bias = (-scales_f32 * zeros_f32).T.float16 using known values.
    func testAWQBiasComputation() {
        let outputDims = 4

        // scales [4,1], NOT transposed yet (AWQ format)
        let scalesData: [Float16] = [2.0, 4.0, 1.0, 0.5]
        let scales = MLXArray(scalesData).reshaped(outputDims, 1)

        // qzeros: all zero-points = 3 → packed as 0x33333333
        let qzeros = MLXArray([UInt32(0x3333_3333)]).reshaped(1, 1)

        let zeros = unpackAndReorderForTesting(qzeros).asType(.float32)
        let zerosValues = zeros.asArray(Float.self)
        for z in zerosValues {
            XCTAssertEqual(z, 3.0, accuracy: 1e-6)
        }

        // biases = (-scales * zeros).T → [8, 4]
        let biases = (-scales.asType(.float32) * zeros).transposed().asType(.float16)
        XCTAssertEqual(biases.shape, [8, 4])

        // biases[:, i] = -scales[i] * 3.0
        let biasValues = biases.asArray(Float16.self)
        let expected: [Float] = [-6.0, -12.0, -3.0, -1.5]
        for j in 0 ..< 8 {
            for (i, exp) in expected.enumerated() {
                XCTAssertEqual(Float(biasValues[j * 4 + i]), exp, accuracy: 0.01)
            }
        }
    }

    func testAWQUnpackReorderPackRoundTrip() {
        // All-zeros should unpack to all-zeros
        let zeros = MLXArray([UInt32(0)]).reshaped(1, 1)
        let unpackedValues = unpackAndReorderForTesting(zeros).asArray(UInt8.self)
        for v in unpackedValues {
            XCTAssertEqual(v, 0)
        }

        // AWQ stores nibbles in order [0,2,4,6,1,3,5,7].
        // Pack sequential values 0..7 in that order and verify unpack recovers [0,1,2,...,7].
        let awqPacked: UInt32 =
            (0 << 0) | (2 << 4) | (4 << 8) | (6 << 12)
            | (1 << 16) | (3 << 20) | (5 << 24) | (7 << 28)

        let result = unpackAndReorderForTesting(MLXArray([awqPacked]).reshaped(1, 1))
        let resultValues = result.asArray(UInt8.self)
        XCTAssertEqual(resultValues.count, 8)
        for i in 0 ..< 8 {
            XCTAssertEqual(resultValues[i], UInt8(i), "Mismatch at index \(i)")
        }
    }

    // MARK: - Rotation + Quantization Round-Trip

    func testQuantizationRoundTrip() {
        let w = MLXRandom.normal([32, 128]).asType(.float16)
        let (wq, scales, biases) = quantized(w, groupSize: 64, bits: 4)
        let wRecon = dequantized(wq, scales: scales, biases: biases, groupSize: 64, bits: 4)

        let relError = relativeRMSError(w, wRecon)
        XCTAssertLessThan(relError, 0.15, "Quantization round-trip error: \(relError)")
    }

    func testQuantizedMatmulApproximatesFullPrecision() {
        let x = MLXRandom.normal([4, 128]).asType(.float16)
        let w = MLXRandom.normal([64, 128]).asType(.float16)
        eval(x, w)

        let yRef = matmul(x, w.transposed())

        let (wq, scales, biases) = quantized(w, groupSize: 64, bits: 4)
        let yQuant = quantizedMM(
            x, wq, scales: scales, biases: biases,
            transpose: true, groupSize: 64, bits: 4
        )

        let relError = relativeRMSError(yRef, yQuant)
        XCTAssertLessThan(relError, 0.15, "Quantized matmul error: \(relError)")
    }

    // MARK: - Pre-Rotation Equivalence

    func testPreRotationEquivalence() throws {
        let layer = try makeTestLayer(hasBias: false)
        let x = MLXRandom.normal([4, 128]).asType(.float16)
        eval(x)

        // Output BEFORE pre-rotation (runtime rotation path)
        let yBefore = layer(x)
        eval(yBefore)

        // Pre-rotate: bakes rotation into weights
        try layer.preRotateWeights()
        eval(layer)

        // Output AFTER pre-rotation (fast path)
        let yAfter = layer(x)
        eval(yAfter)

        // One extra quantization round → allow up to 25% relative error
        let relError = relativeRMSError(yBefore, yAfter)
        XCTAssertLessThan(relError, 0.25, "Pre-rotation equivalence error: \(relError)")
    }

    func testRotateQuantizedLinearProducesValidOutput() throws {
        let layer = try makeTestLayer(hasBias: true)

        let y1 = layer(MLXRandom.normal([1, 128]).asType(.float16))
        eval(y1)
        XCTAssertEqual(y1.shape, [1, 64])

        let y1Values = y1.asType(.float32).asArray(Float.self)
        XCTAssertTrue(y1Values.allSatisfy { $0.isFinite }, "Output contains non-finite values")
        XCTAssertTrue(y1Values.contains { $0 != 0 }, "Output is all zeros")

        let y4 = layer(MLXRandom.normal([4, 128]).asType(.float16))
        eval(y4)
        XCTAssertEqual(y4.shape, [4, 64])
    }
}

// MARK: - Test Helpers

private let testInDim = 128
private let testOutDim = 64
private let testGroupSize = 128
private let testBits = 4
private let testKrot = 2

/// Creates a RotateQuantizedLinear layer with random weights and rotation parameters.
private func makeTestLayer(hasBias: Bool) throws -> RotateQuantizedLinear {
    let layer = RotateQuantizedLinear(
        inputDims: testInDim, outputDims: testOutDim, hasBias: hasBias,
        groupSize: testGroupSize, bits: testBits, krot: testKrot
    )

    let w = MLXRandom.normal([testOutDim, testInDim]).asType(.float16)
    let (wq, scales, biases) = quantized(w, groupSize: testGroupSize, bits: testBits)

    // Small rotation angles keep the rotation near identity
    let theta = (MLXRandom.normal([testKrot, testInDim / 2]) * 0.1).asType(.float16)
    let pairs = makeRandomPairs(krot: testKrot, dim: testInDim, groupSize: testGroupSize)
    let channelScales = (MLXRandom.normal([1, testInDim]) * 0.1 + 1.0).asType(.float16)

    var params: [String: MLXArray] = [
        "theta": theta,
        "pairs": pairs,
        "channel_scales": channelScales,
        "weight": wq,
        "scales": scales,
        "biases": biases ?? MLXArray.zeros(scales.shape),
    ]
    if hasBias {
        params["bias"] = MLXRandom.normal([testOutDim]).asType(.float16)
    }
    try layer.update(parameters: ModuleParameters.unflattened(params), verify: [])
    eval(layer)
    return layer
}

/// Generates random permutation pair indices for Givens rotations within each group.
private func makeRandomPairs(krot: Int, dim: Int, groupSize: Int) -> MLXArray {
    var data = [Int16]()
    data.reserveCapacity(krot * dim)
    for _ in 0 ..< krot {
        for _ in 0 ..< (dim / groupSize) {
            var perm = Array(0 ..< groupSize).map { Int16($0) }
            perm.shuffle()
            data.append(contentsOf: perm)
        }
    }
    return MLXArray(data).reshaped(krot, dim)
}

/// Relative RMS error between two arrays: sqrt(mean((a-b)²) / mean(a²)).
private func relativeRMSError(_ a: MLXArray, _ b: MLXArray) -> Float {
    let diff = (a - b).asType(.float32)
    let ref = a.asType(.float32)
    let mse = mean(diff * diff).item(Float.self)
    let refVar = mean(ref * ref).item(Float.self)
    return sqrt(mse / max(refVar, 1e-10))
}

/// Mirrors `packPairs` from RotateQuantizedLinear.swift (file-private in production).
private func packPairsForTesting(_ pairs: MLXArray, groupSize: Int) -> MLXArray {
    let krot = pairs.dim(0)
    let numGroups = pairs.dim(1) / groupSize
    let p = pairs.reshaped(krot, numGroups, groupSize).asType(.int32)
    let lo = p[0..., 0..., .stride(by: 2)]
    let hi = p[0..., 0..., .stride(from: 1, by: 2)]
    return (lo | (hi << 16)).reshaped(krot, -1)
}

/// Mirrors `unpackAndReorder` from ParoQuantLoader.swift (file-private in production).
private func unpackAndReorderForTesting(_ packed: MLXArray) -> MLXArray {
    let rows = packed.dim(0)
    let cols = packed.dim(1)
    let shifts = MLXArray([0, 4, 8, 12, 16, 20, 24, 28].map { Int64($0) }).reshaped(1, 1, 8)
    let mask: Int64 = 0xF
    let inverseReorder = MLXArray([0, 4, 1, 5, 2, 6, 3, 7].map { Int32($0) })

    let expanded = packed.asType(.int64).expandedDimensions(axis: 2)
    let raw = ((expanded >> shifts) & mask).asType(.uint8)
    let reordered = raw.take(inverseReorder, axis: 2)
    return reordered.reshaped(rows, cols * 8)
}
