import Foundation
import MLX
import MLXNN
import XCTest

@testable import MLXLMCommon

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

        let packed = packPairs(pairs, groupSize: groupSize)
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
        let packed = packPairs(pairs, groupSize: groupSize)

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

    /// The converter must handle **every** `.qweight` prefix — MoE per-expert
    /// weights carry no sibling `theta` (their rotations are shared per layer)
    /// and were silently skipped by the old theta-filter — and must emit
    /// float16 scales to match the f16 biases in `quantizedMM`.
    func testAWQConversionCoversThetaLessPrefixesAndCastsScales() {
        // AWQ layout for in=8, out=8, groupSize=8:
        //   qweight [in, out/8] int32, qzeros [in/gs, out/8] int32, scales [in/gs, out] f32
        let dense = "model.layers.0.mlp.gate_proj."
        let expert = "model.layers.0.mlp.experts.0.gate_proj."

        var weights: [String: MLXArray] = [:]
        for pfx in [dense, expert] {
            weights["\(pfx)qweight"] = MLXArray(Array(repeating: UInt32(0x7654_3210), count: 8))
                .reshaped(8, 1)
            weights["\(pfx)qzeros"] = MLXArray([UInt32(0x3333_3333)]).reshaped(1, 1)
            weights["\(pfx)scales"] = MLXArray((0 ..< 8).map { Float($0) + 1.0 }).reshaped(1, 8)
        }
        // Only the dense prefix has rotation params, as in real MoE checkpoints.
        weights["\(dense)theta"] = MLXArray.zeros([2, 4])

        convertAutoAWQ(&weights, groupSize: 8)

        for pfx in [dense, expert] {
            XCTAssertNil(weights["\(pfx)qweight"], "\(pfx): qweight not consumed")
            XCTAssertNil(weights["\(pfx)qzeros"], "\(pfx): qzeros not consumed")
            let weight = try? XCTUnwrap(weights["\(pfx)weight"], "\(pfx): missing converted weight")
            XCTAssertEqual(weight?.dtype, .uint32)
            let scales = try? XCTUnwrap(weights["\(pfx)scales"], "\(pfx): missing scales")
            XCTAssertEqual(scales?.dtype, .float16, "\(pfx): scales not cast to f16")
            XCTAssertEqual(scales?.shape, [8, 1], "\(pfx): scales not transposed")
            XCTAssertEqual(weights["\(pfx)biases"]?.dtype, .float16)
        }
        XCTAssertNotNil(weights["\(dense)theta"], "theta must pass through untouched")
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

    /// Regression gate — the old implementation had a
    /// `nonisolated(unsafe)` kernel cache and an eval-time `CachedRotation?`
    /// field that mutated on the first forward pass. Both are unsafe under
    /// the multi-threaded usage that `ModelContainer.perform { ... }`
    /// allows in production.
    ///
    /// Uses `DispatchQueue.concurrentPerform` (the same dispatch primitive
    /// the model container path ends up on via its worker queue) so the
    /// layer is hit from several threads simultaneously without any
    /// isolation in between. Mixes batch=1 and batch=4 so both tile sizes
    /// race into the kernel cache on the first iteration.
    func testRotateQuantizedLinearConcurrentSafe() throws {
        let layer = SharedLayerRef(try makeTestLayer(hasBias: true))
        let numTasks = 8
        let buffer = SynchronizedShapeBuffer()

        DispatchQueue.concurrentPerform(iterations: numTasks) { i in
            let batch = i % 2 == 0 ? 1 : 4
            let x = MLXRandom.normal([batch, 128]).asType(.float16)
            let y = layer.layer(x)
            eval(y)
            buffer.append(y.shape)
        }

        let shapes = buffer.snapshot()
        XCTAssertEqual(shapes.count, numTasks)
        for shape in shapes {
            XCTAssertTrue(
                shape == [1, 64] || shape == [4, 64],
                "Unexpected output shape under concurrent load: \(shape)")
        }
    }

    // MARK: - PairwiseRotation

    /// The checkpoint key contract `RotateSwitchGLU` relies on: Module
    /// reflection must expose exactly `theta` / `pairs` / `channel_scales`
    /// (so nested keys like `switch_mlp.gate_up_rot.theta` load), and none
    /// of the underscore-prefixed derived state.
    func testPairwiseRotationExposesCheckpointKeys() {
        let rot = PairwiseRotation(dims: 16, groupSize: 8, krot: 2)
        let keys = Set(rot.parameters().flattened().map { $0.0 })
        XCTAssertEqual(keys, ["theta", "pairs", "channel_scales"])
    }

    /// Freshly-initialized parameters (theta = 0, scales = 1) must be an
    /// exact identity: cos = 1 / sin = 0 rotations and unit channel scales
    /// round-trip every value bit-for-bit through the kernel.
    func testPairwiseRotationDefaultIsIdentity() {
        let rot = PairwiseRotation(dims: 16, groupSize: 8, krot: 2)
        rot.prepareDerivedRotationState()

        let x = MLXRandom.normal([4, 16]).asType(.float16)
        eval(x)
        let y = rot.rotate(x)
        XCTAssertTrue(allClose(y, x, rtol: 0.0, atol: 0.0).item(Bool.self))
    }

    /// Kernel output vs a scalar CPU re-implementation of the same math
    /// (channel scaling, then krot rounds of within-group Givens rotations),
    /// on both tile paths (batch 1 → tile 1, batch 5 → tile 4).
    func testPairwiseRotationMatchesCPUReference() throws {
        let dim = 16
        let groupSize = 8
        let krot = 3

        let rot = PairwiseRotation(dims: dim, groupSize: groupSize, krot: krot)
        let theta = (MLXRandom.normal([krot, dim / 2]) * 0.5).asType(.float16)
        let pairs = makeRandomPairs(krot: krot, dim: dim, groupSize: groupSize)
        let channelScales = (MLXRandom.normal([1, dim]) * 0.1 + 1.0).asType(.float16)
        try rot.update(
            parameters: ModuleParameters.unflattened([
                "theta": theta, "pairs": pairs, "channel_scales": channelScales,
            ]),
            verify: [.all])
        rot.prepareDerivedRotationState()

        for batch in [1, 5] {
            let x = MLXRandom.normal([batch, dim]).asType(.float16)
            eval(x)

            let expected = referenceRotate(
                x: x, pairs: pairs, theta: theta, channelScales: channelScales,
                groupSize: groupSize)
            let y = rot.rotate(x)

            let relError = relativeRMSError(expected, y)
            XCTAssertLessThan(
                relError, 0.01, "batch \(batch): kernel diverges from CPU reference")
        }
    }

    /// The reusability contract for gathered MoE activations: leading shape
    /// is preserved for N-D input, and a zero-row input passes through
    /// (no kernel dispatch on an empty grid).
    func testPairwiseRotationPreservesLeadingShapeAndHandlesEmpty() {
        let rot = PairwiseRotation(dims: 16, groupSize: 8, krot: 2)
        rot.prepareDerivedRotationState()

        let x4d = MLXRandom.normal([2, 3, 1, 16]).asType(.float16)
        eval(x4d)
        let y4d = rot.rotate(x4d)
        XCTAssertEqual(y4d.shape, [2, 3, 1, 16])
        // Same data through the 2-D path must give the same rows.
        let yFlat = rot.rotate(x4d.reshaped(-1, 16))
        XCTAssertTrue(allClose(y4d.reshaped(-1, 16), yFlat, rtol: 0.0, atol: 0.0).item(Bool.self))

        let empty = MLXArray.zeros([0, 16]).asType(.float16)
        let yEmpty = rot.rotate(empty)
        XCTAssertEqual(yEmpty.shape, [0, 16])
    }
}

/// Reference used to carry one layer into the `@Sendable` closure of
/// `DispatchQueue.concurrentPerform`. `@unchecked Sendable` because the point of
/// `testRotateQuantizedLinearConcurrentSafe` is deliberately unsynchronized
/// concurrent access to the shared layer.
private final class SharedLayerRef: @unchecked Sendable {
    let layer: RotateQuantizedLinear

    init(_ layer: RotateQuantizedLinear) {
        self.layer = layer
    }
}

/// Thread-safe `[[Int]]` accumulator used by `testRotateQuantizedLinearConcurrentSafe`.
/// `@unchecked Sendable` because all mutation is serialised by the internal lock.
private final class SynchronizedShapeBuffer: @unchecked Sendable {
    private let lock = NSLock()
    private var shapes: [[Int]] = []

    func append(_ shape: [Int]) {
        lock.withLock { shapes.append(shape) }
    }

    func snapshot() -> [[Int]] {
        lock.withLock { shapes }
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
    // Mirror the loader contract: derive rotation state after the checkpoint
    // params are loaded, before any forward pass.
    layer.prepareDerivedRotationState()
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

/// Scalar CPU re-implementation of the pairwise-rotation kernel's math for
/// `testPairwiseRotationMatchesCPUReference`: scale each channel, then for
/// each rotation round k apply the within-group Givens rotations
/// `(a, b) → (a·cos + b·sin, b·cos − a·sin)`. Rounds are sequential; pairs
/// within a round are a disjoint permutation, so element order is free.
private func referenceRotate(
    x: MLXArray, pairs: MLXArray, theta: MLXArray, channelScales: MLXArray,
    groupSize: Int
) -> MLXArray {
    let batch = x.dim(0)
    let dim = x.dim(1)
    let krot = theta.dim(0)
    let halfGroup = groupSize / 2

    let xValues = x.asType(.float32).asArray(Float.self)
    let thetaValues = theta.asType(.float32).asArray(Float.self)
    let pairValues = pairs.asArray(Int16.self)
    let scaleValues = channelScales.asType(.float32).asArray(Float.self)

    var out = [Float]()
    out.reserveCapacity(batch * dim)
    for row in 0 ..< batch {
        var v = (0 ..< dim).map { xValues[row * dim + $0] * scaleValues[$0] }
        for k in 0 ..< krot {
            for g in 0 ..< (dim / groupSize) {
                for t in 0 ..< halfGroup {
                    let i = g * groupSize + Int(pairValues[k * dim + g * groupSize + 2 * t])
                    let j = g * groupSize + Int(pairValues[k * dim + g * groupSize + 2 * t + 1])
                    let angle = thetaValues[k * (dim / 2) + g * halfGroup + t]
                    let (c, s) = (cos(angle), sin(angle))
                    let (a, b) = (v[i], v[j])
                    v[i] = a * c + b * s
                    v[j] = b * c - a * s
                }
            }
        }
        out.append(contentsOf: v)
    }
    return MLXArray(out).reshaped(batch, dim).asType(x.dtype)
}

/// Relative RMS error between two arrays: sqrt(mean((a-b)²) / mean(a²)).
private func relativeRMSError(_ a: MLXArray, _ b: MLXArray) -> Float {
    let diff = (a - b).asType(.float32)
    let ref = a.asType(.float32)
    let mse = mean(diff * diff).item(Float.self)
    let refVar = mean(ref * ref).item(Float.self)
    return sqrt(mse / max(refVar, 1e-10))
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
