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
    /// scales and biases in the checkpoint's own float dtype (read from the
    /// rotation tensors), so `quantizedMM` never promotes a bf16 activation
    /// stream to f32 against f16 scales.
    func testAWQConversionCoversThetaLessPrefixesAndMatchesCheckpointDType() {
        for floatDType in [DType.float16, DType.bfloat16] {
            // AWQ layout for in=8, out=8, groupSize=8:
            //   qweight [in, out/8] int32, qzeros [in/gs, out/8] int32, scales [in/gs, out] f32
            let dense = "model.layers.0.mlp.gate_proj."
            let expert = "model.layers.0.mlp.experts.0.gate_proj."

            var weights: [String: MLXArray] = [:]
            for pfx in [dense, expert] {
                weights["\(pfx)qweight"] = MLXArray(
                    Array(repeating: UInt32(0x7654_3210), count: 8)
                ).reshaped(8, 1)
                weights["\(pfx)qzeros"] = MLXArray([UInt32(0x3333_3333)]).reshaped(1, 1)
                weights["\(pfx)scales"] = MLXArray((0 ..< 8).map { Float($0) + 1.0 }).reshaped(1, 8)
            }
            // Only the dense prefix has rotation params, as in real MoE checkpoints;
            // their dtype is the checkpoint's float dtype.
            weights["\(dense)theta"] = MLXArray.zeros([2, 4]).asType(floatDType)

            convertAutoAWQ(&weights, groupSize: 8)

            for pfx in [dense, expert] {
                XCTAssertNil(weights["\(pfx)qweight"], "\(pfx): qweight not consumed")
                XCTAssertNil(weights["\(pfx)qzeros"], "\(pfx): qzeros not consumed")
                let weight = try? XCTUnwrap(
                    weights["\(pfx)weight"], "\(pfx): missing converted weight")
                XCTAssertEqual(weight?.dtype, .uint32)
                let scales = try? XCTUnwrap(weights["\(pfx)scales"], "\(pfx): missing scales")
                XCTAssertEqual(
                    scales?.dtype, floatDType, "\(pfx): scales not cast to \(floatDType)")
                XCTAssertEqual(scales?.shape, [8, 1], "\(pfx): scales not transposed")
                XCTAssertEqual(
                    weights["\(pfx)biases"]?.dtype, floatDType,
                    "\(pfx): biases not cast to \(floatDType)")
            }
            XCTAssertNotNil(weights["\(dense)theta"], "theta must pass through untouched")
        }
    }

    /// Without any rotation tensor to read the dtype from, conversion keeps
    /// the historical float16 default.
    func testCheckpointFloatDTypeDefaultsToFloat16() {
        XCTAssertEqual(checkpointFloatDType([:]), .float16)
        XCTAssertEqual(
            checkpointFloatDType(["a.channel_scales": MLXArray.ones([1, 4]).asType(.bfloat16)]),
            .bfloat16)
        // Non-float rotation tensors (pairs) never decide the dtype.
        XCTAssertEqual(
            checkpointFloatDType(["a.pairs": MLXArray.zeros([1, 4], type: Int16.self)]), .float16)
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

    /// Rotation parameters are checkpoint constants: `PairwiseRotation`
    /// freezes itself like `QuantizedLinear`, so a `RotateSwitchGLU` over
    /// quantized experts reports no trainable parameters — the condition
    /// `SwitchGLU.supportsDirectWeightedReduction` checks before taking the
    /// fused reduction path.
    func testPairwiseRotationIsFrozen() {
        let rot = PairwiseRotation(dims: 16, groupSize: 8, krot: 2)
        XCTAssertTrue(rot.trainableParameters().flattened().isEmpty)
        XCTAssertEqual(rot.parameters().flattened().count, 3)

        let glu = RotateSwitchGLU(
            inputDims: 64, hiddenDims: 64, numExperts: 8, groupSize: 32, krot: 2)
        quantize(model: glu, groupSize: 32, bits: 4)
        XCTAssertTrue(glu.trainableParameters().flattened().isEmpty)
    }

    /// The geometry every kernel assumes, rejected up front with a reason:
    /// whole groups, whole pairs, a half-group that fits one threadgroup,
    /// at least one round.
    func testRotationGeometryProblems() {
        XCTAssertNil(rotationGeometryProblem(dims: 256, groupSize: 128, krot: 8))
        XCTAssertNil(rotationGeometryProblem(dims: 16, groupSize: 8, krot: 1))
        XCTAssertNil(rotationGeometryProblem(dims: 4096, groupSize: 2048, krot: 3))
        XCTAssertNotNil(rotationGeometryProblem(dims: 20, groupSize: 8, krot: 2), "partial group")
        XCTAssertNotNil(rotationGeometryProblem(dims: 18, groupSize: 9, krot: 2), "odd group")
        XCTAssertNotNil(rotationGeometryProblem(dims: 8, groupSize: 16, krot: 2), "dims < group")
        XCTAssertNotNil(rotationGeometryProblem(dims: 8192, groupSize: 4096, krot: 2), "CTA limit")
        XCTAssertNotNil(rotationGeometryProblem(dims: 16, groupSize: 8, krot: 0), "no rounds")
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
    /// on both kernels (groupSize 8 → generic, 128 → simdgroup), both tile
    /// paths (batch 1 → tile 1, batch 5 → tile 4), and every activation
    /// dtype the kernels are instantiated for.
    func testPairwiseRotationMatchesCPUReference() throws {
        let krot = 3
        for groupSize in [8, 128] {
            let dim = groupSize * 2

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

            // bfloat16 keeps 8 significand bits, so it rounds ~8x coarser than
            // float16 on the same values.
            let tolerances: [(DType, Float)] = [
                (.float16, 0.01), (.bfloat16, 0.05), (.float32, 0.001),
            ]
            for (dtype, tolerance) in tolerances {
                for batch in [1, 5] {
                    let x = MLXRandom.normal([batch, dim]).asType(dtype)
                    eval(x)

                    let expected = referenceRotate(
                        x: x, pairs: pairs, theta: theta, channelScales: channelScales,
                        groupSize: groupSize)
                    let y = rot.rotate(x)
                    XCTAssertEqual(y.dtype, dtype)

                    let relError = relativeRMSError(expected, y)
                    XCTAssertLessThan(
                        relError, tolerance,
                        "groupSize \(groupSize) \(dtype) batch \(batch): kernel diverges from CPU reference"
                    )
                }
            }
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

    // MARK: - MoE Loader Passes

    /// `…experts.{e}.{proj}.{suffix}` triples stack into 3-D `switch_mlp`
    /// tensors in expert order, consuming the per-expert keys; groups with a
    /// mid-gap expert are left untouched (the strict update fails loudly on
    /// the real gap instead); non-stackable suffixes and shared rotation
    /// keys are ignored.
    func testStackMoEExpertWeightsStacksAndRemoves() {
        let base = "model.layers.0.mlp"
        var weights = [String: MLXArray]()
        for e in 0 ..< 2 {
            for suffix in ["weight", "scales", "biases"] {
                weights["\(base).experts.\(e).gate_proj.\(suffix)"] =
                    MLXArray(Array(repeating: Float(e), count: 8)).reshaped(4, 2)
            }
        }
        // Mid-gap group (experts 0 and 2, no 1) must not stack.
        weights["\(base).experts.0.up_proj.weight"] = MLXArray.zeros([4, 2])
        weights["\(base).experts.2.up_proj.weight"] = MLXArray.zeros([4, 2])
        // Unconverted AWQ suffix and shared rotation keys must be ignored.
        weights["\(base).experts.0.down_proj.qweight"] = MLXArray.zeros([4, 2])
        weights["\(base).experts.gate_up_weight_theta"] = MLXArray.zeros([2, 8])

        stackMoEExpertWeights(&weights)

        for suffix in ["weight", "scales", "biases"] {
            let stacked = weights["\(base).switch_mlp.gate_proj.\(suffix)"]
            XCTAssertEqual(stacked?.shape, [2, 4, 2])
            // Expert order: slice e must hold expert e's values.
            XCTAssertEqual(stacked?[1].max().item(Float.self), 1.0)
            XCTAssertEqual(stacked?[1].min().item(Float.self), 1.0)
            XCTAssertNil(weights["\(base).experts.0.gate_proj.\(suffix)"])
            XCTAssertNil(weights["\(base).experts.1.gate_proj.\(suffix)"])
        }
        XCTAssertNil(weights["\(base).switch_mlp.up_proj.weight"])
        XCTAssertNotNil(weights["\(base).experts.0.up_proj.weight"])
        XCTAssertNotNil(weights["\(base).experts.2.up_proj.weight"])
        XCTAssertNotNil(weights["\(base).experts.0.down_proj.qweight"])
        XCTAssertNotNil(weights["\(base).experts.gate_up_weight_theta"])
    }

    /// The shared per-layer rotation keys remap onto the *nested*
    /// `PairwiseRotation` children (deliberate divergence from upstream's
    /// flat `gate_up_rot_theta` layout), with 1-D channel_scales normalised
    /// to `[1, dims]`.
    func testRemapSharedMoERotationsProducesNestedKeys() {
        let base = "model.layers.3.mlp"
        var weights = [String: MLXArray]()
        weights["\(base).experts.gate_up_weight_theta"] = MLXArray.zeros([8, 1024])
        weights["\(base).experts.gate_up_weight_pairs"] = MLXArray.zeros(
            [8, 2048], type: Int16.self)
        weights["\(base).experts.gate_up_weight_channel_scales"] = MLXArray.ones([1, 2048])
        weights["\(base).experts.down_weight_theta"] = MLXArray.zeros([8, 256])
        weights["\(base).experts.down_weight_pairs"] = MLXArray.zeros([8, 512], type: Int16.self)
        // 1-D channel_scales must be normalised like convertAutoAWQ does.
        weights["\(base).experts.down_weight_channel_scales"] = MLXArray.ones([512])
        // Per-expert keys must not be touched.
        weights["\(base).experts.0.gate_proj.weight"] = MLXArray.zeros([4, 2])

        remapSharedMoERotations(&weights)

        XCTAssertEqual(weights["\(base).switch_mlp.gate_up_rot.theta"]?.shape, [8, 1024])
        XCTAssertEqual(weights["\(base).switch_mlp.gate_up_rot.pairs"]?.shape, [8, 2048])
        XCTAssertEqual(
            weights["\(base).switch_mlp.gate_up_rot.channel_scales"]?.shape, [1, 2048])
        XCTAssertEqual(weights["\(base).switch_mlp.down_rot.theta"]?.shape, [8, 256])
        XCTAssertEqual(weights["\(base).switch_mlp.down_rot.channel_scales"]?.shape, [1, 512])
        XCTAssertNil(weights["\(base).experts.gate_up_weight_theta"])
        XCTAssertNil(weights["\(base).experts.down_weight_channel_scales"])
        XCTAssertNotNil(weights["\(base).experts.0.gate_proj.weight"])
    }

    /// The text_config flatten must keep the checkpoint's *top-level*
    /// model_type (the registry key) instead of letting text_config's
    /// "*_text" variant — or the old hard-coded "qwen3_5" — clobber it.
    func testFlattenTextConfigPreservesModelType() {
        let moe: [String: Any] = [
            "model_type": "qwen3_5_moe",
            "text_config": ["model_type": "qwen3_5_moe_text", "num_experts": 256],
        ]
        let flatMoe = flattenParoQuantTextConfig(moe)
        XCTAssertEqual(flatMoe["model_type"] as? String, "qwen3_5_moe")
        XCTAssertEqual(flatMoe["num_experts"] as? Int, 256)

        let dense: [String: Any] = [
            "model_type": "qwen3_5",
            "text_config": ["model_type": "qwen3_5_text"],
        ]
        XCTAssertEqual(
            flattenParoQuantTextConfig(dense)["model_type"] as? String, "qwen3_5")
    }

    /// Detection accepts both the dense and the MoE ForConditionalGeneration
    /// architecture strings, and still rejects everything else.
    func testDetectionAcceptsMoEArchitecture() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }

        func writeConfig(architecture: String) throws {
            let config: [String: Any] = [
                "architectures": [architecture],
                "quantization_config": ["quant_method": "paroquant"],
            ]
            try JSONSerialization.data(withJSONObject: config)
                .write(to: dir.appendingPathComponent("config.json"))
        }

        try writeConfig(architecture: "Qwen3_5MoeForConditionalGeneration")
        XCTAssertTrue(isParoQuantModel(directory: dir))
        try writeConfig(architecture: "Qwen3_5ForConditionalGeneration")
        XCTAssertTrue(isParoQuantModel(directory: dir))
        try writeConfig(architecture: "LlamaForCausalLM")
        XCTAssertFalse(isParoQuantModel(directory: dir))
    }

    // MARK: - RotateSwitchGLU

    /// Nested checkpoint keys: the two `PairwiseRotation` children expose
    /// `gate_up_rot.*` / `down_rot.*` alongside the inherited projections —
    /// exactly what `remapSharedMoERotations` emits.
    func testRotateSwitchGLUExposesNestedRotationKeys() {
        let glu = RotateSwitchGLU(
            inputDims: 16, hiddenDims: 8, numExperts: 4, groupSize: 8, krot: 2)
        let keys = Set(glu.parameters().flattened().map { $0.0 })
        for child in ["gate_up_rot", "down_rot"] {
            for suffix in ["theta", "pairs", "channel_scales"] {
                XCTAssertTrue(keys.contains("\(child).\(suffix)"), "missing \(child).\(suffix)")
            }
        }
        XCTAssertTrue(keys.contains("gate_proj.weight"))
    }

    /// With identity rotations (the fresh-init state), RotateSwitchGLU must
    /// reproduce stock SwitchGLU bit-for-bit on the same projection weights —
    /// on both the broadcast (<64 indices) and gather/sort (≥64) paths.
    func testRotateSwitchGLUIdentityMatchesSwitchGLU() throws {
        let inputDims = 16
        let hiddenDims = 8
        let numExperts = 4

        let reference = SwitchGLU(
            inputDims: inputDims, hiddenDims: hiddenDims, numExperts: numExperts)
        let rotated = RotateSwitchGLU(
            inputDims: inputDims, hiddenDims: hiddenDims, numExperts: numExperts,
            groupSize: 8, krot: 2)

        // Copy the reference projections; rotations stay at their identity
        // defaults, so `verify: []` (the rotation keys are legitimately absent).
        try rotated.update(parameters: reference.parameters(), verify: [])
        for case let rot as PairwiseRotation in rotated.modules() {
            rot.prepareDerivedRotationState()
        }

        // Broadcast path: 2×2 indices (size 4 < 64).
        let xSmall = MLXRandom.normal([1, 2, inputDims])
        let idxSmall = MLXArray([Int32(0), 1, 2, 3]).reshaped(1, 2, 2)
        eval(xSmall)
        XCTAssertTrue(
            allClose(
                rotated(xSmall, idxSmall), reference(xSmall, idxSmall),
                rtol: 0.0, atol: 0.0
            ).item(Bool.self))

        // Gather/sort path: 32×2 indices (size 64 ≥ 64).
        let xLarge = MLXRandom.normal([1, 32, inputDims])
        let idxLarge = randInt(Int32(0) ..< Int32(numExperts), [1, 32, 2])
        eval(xLarge)
        XCTAssertTrue(
            allClose(
                rotated(xLarge, idxLarge), reference(xLarge, idxLarge),
                rtol: 0.0, atol: 0.0
            ).item(Bool.self))
    }

    /// End-to-end loader mechanics on a miniature MoE block: the GLU swap
    /// installs RotateSwitchGLU, the step-9 quantize pass converts its
    /// SwitchLinear children to QuantizedSwitchLinear (checkpoint `.scales`
    /// present, no `.theta`), the strict checkpoint update succeeds, and a
    /// forward pass through both index paths produces finite output.
    func testPatchMoESwitchGLULayersEndToEnd() throws {
        let inputDims = 128
        let hiddenDims = 64
        let numExperts = 4
        let groupSize = 64
        let bits = 4
        let krot = 2

        let host = MoEHostBlock(
            inputDims: inputDims, hiddenDims: hiddenDims, numExperts: numExperts)

        // Synthetic stacked checkpoint: quantized 3-D expert projections
        // (what stackMoEExpertWeights emits) + nested rotation parameters
        // (what remapSharedMoERotations emits).
        var weights = [String: MLXArray]()
        for (proj, outDims, inDims) in [
            ("gate_proj", hiddenDims, inputDims),
            ("up_proj", hiddenDims, inputDims),
            ("down_proj", inputDims, hiddenDims),
        ] {
            let w = MLXRandom.normal([numExperts, outDims, inDims]).asType(.float16)
            let (wq, scales, biases) = quantized(w, groupSize: groupSize, bits: bits)
            weights["switch_mlp.\(proj).weight"] = wq
            weights["switch_mlp.\(proj).scales"] = scales
            weights["switch_mlp.\(proj).biases"] = biases ?? MLXArray.zeros(scales.shape)
        }
        for (child, dims) in [("gate_up_rot", inputDims), ("down_rot", hiddenDims)] {
            weights["switch_mlp.\(child).theta"] =
                (MLXRandom.normal([krot, dims / 2]) * 0.1).asType(.float16)
            weights["switch_mlp.\(child).pairs"] = makeRandomPairs(
                krot: krot, dim: dims, groupSize: groupSize)
            weights["switch_mlp.\(child).channel_scales"] =
                (MLXRandom.normal([1, dims]) * 0.1 + 1.0).asType(.float16)
        }

        // Mirror the loader: GLU swap → quantize by checkpoint keys → strict
        // update → derived rotation state.
        try patchMoESwitchGLULayers(model: host, weights: weights, groupSize: groupSize)
        XCTAssertTrue(host.switchMLP is RotateSwitchGLU)

        quantize(model: host) { path, module in
            guard module is Quantizable else { return nil }
            guard weights["\(path).scales"] != nil, weights["\(path).theta"] == nil else {
                return nil
            }
            return (groupSize, bits, .affine)
        }
        let leaves = Dictionary(uniqueKeysWithValues: host.leafModules().flattened())
        for proj in ["gate_proj", "up_proj", "down_proj"] {
            XCTAssertTrue(
                leaves["switch_mlp.\(proj)"] is QuantizedSwitchLinear, "\(proj) not quantized")
        }

        try host.update(
            parameters: ModuleParameters.unflattened(weights),
            verify: [.allModelKeysSet, .shapeMismatch])
        for case let rot as PairwiseRotation in host.modules() {
            rot.prepareDerivedRotationState()
        }
        eval(host)

        for tokens in [2, 32] {  // 2×2=4 broadcast path, 32×2=64 gather/sort path
            let x = MLXRandom.normal([1, tokens, inputDims]).asType(.float16)
            let indices = randInt(Int32(0) ..< Int32(numExperts), [1, tokens, 2])
            eval(x)
            let y = host.switchMLP(x, indices)
            XCTAssertEqual(y.shape, [1, tokens, 2, inputDims])
            XCTAssertTrue(all(isFinite(y)).item(Bool.self), "\(tokens) tokens: non-finite output")
        }
    }

    // MARK: - Prepared Checkpoint

    /// Representative converted dict: packed quantized weights, f16
    /// scales/biases, and verbatim rotation keys across dtypes.
    private func makePreparedWeights() -> [String: MLXArray] {
        let weights: [String: MLXArray] = [
            "layers.0.q_proj.weight": MLXArray(
                (0 ..< 64).map { UInt32($0) &* 2_654_435_761 }
            ).reshaped(8, 8),
            "layers.0.q_proj.scales": (MLXRandom.normal([8, 2]) * 0.02).asType(.float16),
            "layers.0.q_proj.biases": (MLXRandom.normal([8, 2]) * 0.01).asType(.float16),
            "layers.0.q_proj.theta": (MLXRandom.normal([2, 32]) * 0.1).asType(.float16),
            "layers.0.q_proj.pairs": makeRandomPairs(krot: 2, dim: 64, groupSize: 8),
            "layers.0.q_proj.channel_scales": (MLXRandom.normal([1, 64]) * 0.1 + 1.0)
                .asType(.float16),
        ]
        eval(Array(weights.values))
        return weights
    }

    private func makeTempCheckpointDir() throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("prepared-checkpoint-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    private static let testManifest = ParoQuantPreparedCheckpoint.Manifest(
        formatVersion: ParoQuantPreparedCheckpoint.formatVersion,
        sources: [
            .init(name: "config.json", size: 42, mtimeNs: 1_000),
            .init(name: "model.safetensors", size: 4_096, mtimeNs: 2_000),
        ]
    )

    func testPreparedCheckpointRoundTripIsBitExact() throws {
        let dir = try makeTempCheckpointDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        let weights = makePreparedWeights()

        ParoQuantPreparedCheckpoint.write(
            weights: weights, manifest: Self.testManifest, directory: dir)
        let loaded = try XCTUnwrap(
            ParoQuantPreparedCheckpoint.load(directory: dir, manifest: Self.testManifest))

        XCTAssertEqual(Set(loaded.keys), Set(weights.keys))
        for (key, original) in weights {
            let restored = try XCTUnwrap(loaded[key], key)
            XCTAssertEqual(restored.dtype, original.dtype, key)
            XCTAssertEqual(restored.shape, original.shape, key)
            XCTAssertTrue(arrayEqual(original, restored).item(Bool.self), "\(key) not bit-equal")
        }
    }

    func testPreparedCheckpointStaleManifestSelfHeals() throws {
        let dir = try makeTempCheckpointDir()
        defer { try? FileManager.default.removeItem(at: dir) }

        ParoQuantPreparedCheckpoint.write(
            weights: makePreparedWeights(), manifest: Self.testManifest, directory: dir)
        let artifact = dir.appendingPathComponent(ParoQuantPreparedCheckpoint.fileName)
        XCTAssertTrue(FileManager.default.fileExists(atPath: artifact.path))

        // A source changed (size differs) — the artifact is stale.
        let staleExpectation = ParoQuantPreparedCheckpoint.Manifest(
            formatVersion: ParoQuantPreparedCheckpoint.formatVersion,
            sources: [
                .init(name: "config.json", size: 42, mtimeNs: 1_000),
                .init(name: "model.safetensors", size: 8_192, mtimeNs: 2_000),
            ]
        )
        XCTAssertNil(
            ParoQuantPreparedCheckpoint.load(directory: dir, manifest: staleExpectation))
        XCTAssertFalse(
            FileManager.default.fileExists(atPath: artifact.path),
            "stale artifact must be deleted so the rewrite can publish a fresh one")
    }

    func testPreparedCheckpointCorruptArtifactSelfHeals() throws {
        let dir = try makeTempCheckpointDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        let artifact = dir.appendingPathComponent(ParoQuantPreparedCheckpoint.fileName)
        try Data("not a safetensors file".utf8).write(to: artifact)

        XCTAssertNil(ParoQuantPreparedCheckpoint.load(directory: dir, manifest: Self.testManifest))
        XCTAssertFalse(
            FileManager.default.fileExists(atPath: artifact.path),
            "corrupt artifact must be deleted")
    }

    func testPreparedCheckpointManifestExcludesArtifactNames() throws {
        let dir = try makeTempCheckpointDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        try Data("{}".utf8).write(to: dir.appendingPathComponent("config.json"))
        try Data([0x00]).write(to: dir.appendingPathComponent("model.safetensors"))
        for name in ParoQuantPreparedCheckpoint.excludedFileNames {
            try Data([0x00]).write(to: dir.appendingPathComponent(name))
        }

        let manifest = try ParoQuantPreparedCheckpoint.currentManifest(directory: dir)
        XCTAssertEqual(
            manifest.sources.map(\.name).sorted(), ["config.json", "model.safetensors"],
            "artifact names must never count as conversion sources")
    }
}

/// Miniature stand-in for an MoE block (e.g. `Qwen35SparseMoeBlock`): a
/// `switch_mlp: SwitchGLU` property that `patchMoESwitchGLULayers` must be
/// able to replace with a `RotateSwitchGLU` through `update(modules:)`.
private final class MoEHostBlock: Module {
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU

    init(inputDims: Int, hiddenDims: Int, numExperts: Int) {
        self._switchMLP.wrappedValue = SwitchGLU(
            inputDims: inputDims, hiddenDims: hiddenDims, numExperts: numExperts)
        super.init()
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
