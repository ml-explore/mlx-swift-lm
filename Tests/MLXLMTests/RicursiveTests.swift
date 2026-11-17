import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXLLM

/// Ricursive against the model author's `modeling_ricursive.py` (ricursiveAI/Ripple-65B-Preview,
/// revision and sha256 in `RicursiveReferenceValues`). The tiny model has four sliding layers
/// and one full layer, and an 8-token window shorter than its 24-token prompt. Its weights
/// come from the closed-form formula in `tools/ricursive-reference.py`, which also writes the
/// expected values in `RicursiveReferenceValues`.
final class RicursiveTests: XCTestCase {

    private func tinyConfiguration() throws -> RicursiveConfiguration {
        let config = try JSONDecoder().decode(
            RicursiveConfiguration.self,
            from: Data(RicursiveReferenceValues.configuration.utf8))
        try config.validateModelConfiguration()
        return config
    }

    /// `formula_weights` in `tools/ricursive-reference.py`, keyed by the reference's names.
    private func tinyWeights() -> [String: MLXArray] {
        let seed = RicursiveReferenceValues.seed
        var weights = [String: MLXArray]()
        for (i, (name, shape)) in zip(
            RicursiveReferenceValues.parameterNames, RicursiveReferenceValues.parameterShapes
        ).enumerated() {
            let values = (0 ..< shape.reduce(1, *)).map { j -> Float in
                let v = (j * j * 7 + j * (13 + 2 * i) + i * 31 + seed) % 97 - 48
                if name.hasSuffix("value_res_lambda") {
                    return Float(v >= 0 ? 2 * v + 65 : 2 * v - 63) / 64
                } else if name.hasSuffix("norm.weight") {
                    return Float(v) / 128 + 1
                } else if name.hasSuffix("gate_proj.weight") || name.hasSuffix("up_proj.weight") {
                    return Float(v) / 64
                }
                return Float(v) / 256
            }
            weights[name] = MLXArray(values, shape)
        }
        return weights
    }

    private func makeTiny() throws -> RicursiveModel {
        let model = RicursiveModel(try tinyConfiguration())
        // The parameter names and shapes must be the reference model's.
        let parameters = model.parameters().flattened().sorted { $0.0 < $1.0 }
        XCTAssertEqual(parameters.map(\.0), RicursiveReferenceValues.parameterNames)
        XCTAssertEqual(parameters.map(\.1.shape), RicursiveReferenceValues.parameterShapes)
        try model.update(
            parameters: ModuleParameters.unflattened(tinyWeights()), verify: [.all])
        model.train(false)
        eval(model)
        return model
    }

    private func reference(_ values: [Float]) throws -> [[Float]] {
        let vocabulary = try tinyConfiguration().vocabularySize
        return stride(from: 0, to: values.count, by: vocabulary).map {
            Array(values[$0 ..< $0 + vocabulary])
        }
    }

    private func maxDifference(_ mine: [[Float]], _ theirs: [[Float]]) -> Float {
        XCTAssertEqual(mine.count, theirs.count)
        return zip(mine, theirs).map { a, b in
            XCTAssertEqual(a.count, b.count)
            return zip(a, b).map { abs($0 - $1) }.max() ?? 0
        }.max() ?? 0
    }

    private func rows(_ array: MLXArray) -> [[Float]] {
        (0 ..< array.dim(0)).map { array[$0].asType(.float32).asArray(Float.self) }
    }

    private func decode(_ model: RicursiveModel, prompt: [Int], step: Int, count: Int) throws -> (
        [Int], [[Float]]
    ) {
        let cache = try model.newCache(parameters: nil)
        var rest = prompt[...]
        while rest.count > 1 {
            let n = min(step, rest.count - 1)
            let chunk = MLXArray(rest.prefix(n).map(Int32.init)).reshaped(1, n)
            eval(model(chunk, cache: cache))
            eval(cache)
            rest = rest.dropFirst(n)
        }
        var input = MLXArray(rest.map(Int32.init)).reshaped(1, 1)
        var tokens = [Int]()
        var logits = [[Float]]()
        for _ in 0 ..< count {
            let row = model(input, cache: cache)[0, -1]
            eval(row)
            eval(cache)
            let token = argMax(row, axis: -1).item(Int.self)
            tokens.append(token)
            logits.append(row.asType(.float32).asArray(Float.self))
            input = MLXArray([Int32(token)]).reshaped(1, 1)
        }
        return (tokens, logits)
    }

    func testUncachedForwardMatchesReference() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = try makeTiny()
            let prompt = RicursiveReferenceValues.prompt
            let input = MLXArray(prompt.map(Int32.init)).reshaped(1, prompt.count)
            let difference = maxDifference(
                rows(model(input, cache: nil)[0]),
                try reference(RicursiveReferenceValues.fullLogits))
            print("ricursive parity uncached: max |dlogit| = \(difference)")
            XCTAssertLessThan(difference, 2e-5)
        }
    }

    func testCachedGreedyDecodeMatchesReferenceAtEveryChunking() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = try makeTiny()
            let prompt = RicursiveReferenceValues.prompt
            let expected = try reference(RicursiveReferenceValues.decodeLogits)
            XCTAssertGreaterThan(prompt.count, model.configuration.slidingWindow)
            for step in RicursiveReferenceValues.prefillSteps {
                let (tokens, logits) = try decode(
                    model, prompt: prompt, step: step, count: expected.count)
                XCTAssertEqual(tokens, RicursiveReferenceValues.greedy, "prefill step \(step)")
                let difference = maxDifference(logits, expected)
                print("ricursive parity step \(step): max |dlogit| = \(difference)")
                XCTAssertLessThan(difference, 2e-5, "prefill step \(step)")
            }
        }
    }

    func testPrepareLeavesTheLastTokenAndKeepsParity() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = try makeTiny()
            let prompt = RicursiveReferenceValues.prompt
            let cache = try model.newCache(parameters: nil)
            let input = LMInput(tokens: MLXArray(prompt.map(Int32.init)))
            // Chunks wider than the sliding window, and a prompt longer than one chunk, so
            // the default `LLMModel.prepare` chunks it and leaves the last token.
            let step = model.configuration.slidingWindow + 3
            XCTAssertGreaterThan(prompt.count, step)
            guard
                case .tokens(let rest) = try model.prepare(
                    input, cache: cache, state: nil, prefill: PrefillParameters(stepSize: step))
            else {
                return XCTFail("prepare should hand tokens to the iterator")
            }
            XCTAssertEqual(rest.tokens.size, 1)
            let logits = model(rest.tokens.reshaped(1, 1), cache: cache)[0, -1]
            eval(logits)
            let difference = maxDifference(
                [logits.asArray(Float.self)],
                [try reference(RicursiveReferenceValues.decodeLogits)[0]])
            print("ricursive parity prepare: max |dlogit| = \(difference)")
            XCTAssertLessThan(difference, 2e-5)
        }
    }

    func testCacheTypesFollowLayerTypes() throws {
        let caches = try RicursiveModel(try tinyConfiguration()).newCache(parameters: nil)
        XCTAssertEqual(caches.count, 5)
        for cache in caches.prefix(4) {
            XCTAssertTrue(cache is RotatingKVCache)
            XCTAssertEqual(cache.maxSize, 8)
        }
        XCTAssertTrue(caches[4] is KVCacheSimple)
    }

    func testRejectsInconsistentConfiguration() throws {
        func configuration(_ change: (inout [String: Any]) -> Void) throws -> RicursiveConfiguration
        {
            let data = Data(RicursiveReferenceValues.configuration.utf8)
            var json = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
            change(&json)
            return try JSONDecoder().decode(
                RicursiveConfiguration.self,
                from: JSONSerialization.data(withJSONObject: json))
        }
        XCTAssertThrowsError(
            try configuration { $0["layer_types"] = ["full_attention"] }
                .validateModelConfiguration())
        XCTAssertThrowsError(
            try configuration { $0["num_key_value_heads"] = 3 }.validateModelConfiguration())
        XCTAssertThrowsError(
            try configuration { $0["hidden_act"] = "gelu" }.validateModelConfiguration())
        XCTAssertNoThrow(
            try configuration { $0.removeValue(forKey: "layer_types") }
                .validateModelConfiguration())
    }

    func testMixedPrecisionQuantizedCheckpointLoads() throws {
        try Device.withDefaultDevice(.cpu) {
            let source = try makeTiny()
            let quantization: (String, Module) -> (Int, Int, QuantizationMode)? = { path, _ in
                path == "lm_head" || path.hasSuffix("embed_tokens")
                    || path.hasSuffix("g_proj") ? (32, 8, .affine) : (32, 4, .affine)
            }
            quantize(model: source, filter: quantization)
            eval(source)

            let directory = FileManager.default.temporaryDirectory
                .appendingPathComponent("ricursive-\(UUID().uuidString)")
            try FileManager.default.createDirectory(
                at: directory, withIntermediateDirectories: true)
            defer { try? FileManager.default.removeItem(at: directory) }
            let weights = Dictionary(uniqueKeysWithValues: source.parameters().flattened())
            XCTAssertNotNil(weights["lm_head.scales"])
            XCTAssertNotNil(weights["model.layers.0.self_attn.value_res_lambda"])
            XCTAssertNil(weights["model.layers.0.input_layernorm.scales"])
            try save(arrays: weights, url: directory.appendingPathComponent("model.safetensors"))

            let per: [String: BaseConfiguration.QuantizationOption] = [
                "lm_head": .quantize(.init(groupSize: 32, bits: 8)),
                "model.embed_tokens": .quantize(.init(groupSize: 32, bits: 8)),
            ]
            var layers = per
            for i in 0 ..< 5 {
                layers["model.layers.\(i).self_attn.g_proj"] = .quantize(
                    .init(groupSize: 32, bits: 8))
            }
            let loaded = RicursiveModel(try tinyConfiguration())
            try loadWeights(
                modelDirectory: directory, model: loaded,
                perLayerQuantization: .init(
                    quantization: .init(groupSize: 32, bits: 4), perLayerQuantization: layers))
            loaded.train(false)

            let input = MLXArray((0 ..< 12).map { Int32($0) }).reshaped(1, 12)
            let (a, b) = (source(input, cache: nil), loaded(input, cache: nil))
            let difference = abs(a - b).max().item(Float.self)
            print("ricursive quantized reload: max |dlogit| = \(difference)")
            XCTAssertLessThan(difference, 1e-5)

            // With a quantized lm_head, `nextTokenLogits` projects only trailing rows; the
            // last row must stay bitwise equal to the full forward pass.
            for length in [97, 200] {
                let text = LMInput.Text(
                    tokens: MLXArray((0 ..< length).map { Int32(($0 * 7 + 3) % 64) })
                        .reshaped(1, length))
                let expected = loaded(text.tokens, cache: nil)[0..., -1, 0...]
                let actual = loaded.nextTokenLogits(text, cache: nil, state: nil).logits
                XCTAssertLessThan(actual.dim(1), length)
                XCTAssertTrue(
                    arrayEqual(actual[0..., -1, 0...], expected).item(Bool.self),
                    "length \(length)")
            }
        }
    }

    /// Positions above 256 are not exact in bfloat16, so RoPE must build them in float32.
    /// Like the reference, cos and sin are rounded to the input precision.
    func testRoPEKeepsPositionsInFloat32() throws {
        try Device.withDefaultDevice(.cpu) {
            let x = MLXRandom.normal([1, 2, 3, 16], key: MLXRandom.key(0)).asType(.bfloat16)
            let inverseFrequencies =
                1
                / pow(
                    MLXArray(Float(10_000)),
                    MLXArray(
                        stride(from: 0, to: 16, by: 2).map {
                            Float($0) / 16
                        }))
            let angles =
                MLXArray([Float(300), 301, 302])[0..., .newAxis]
                * inverseFrequencies[.newAxis, 0...]
            let both = concatenated([angles, angles], axis: -1)
            let cosine = cos(both).asType(.bfloat16).asType(.float32)
            let sine = sin(both).asType(.bfloat16).asType(.float32)
            let source = x.asType(.float32)
            let rotated = concatenated(
                [-source[.ellipsis, 8...], source[.ellipsis, ..<8]], axis: -1)
            let expected = (source * cosine + rotated * sine).asType(.bfloat16)
            let actual = ricursiveRoPE(x, offset: 300, inverseFrequencies: inverseFrequencies)
            XCTAssertEqual(actual.dtype, .bfloat16)
            XCTAssertTrue(arrayEqual(actual, expected).item(Bool.self))
        }
    }

    /// With `gate_full_precision`, the sigmoid runs in float32 and is rounded once.
    func testGateRunsInFloat32() throws {
        try Device.withDefaultDevice(.cpu) {
            let output = MLXRandom.normal([1, 4, 64], key: MLXRandom.key(1)).asType(.bfloat16)
            let gate = (6 * MLXRandom.normal([1, 4, 64], key: MLXRandom.key(2))).asType(.bfloat16)
            let expected = output * sigmoid(gate.asType(.float32)).asType(.bfloat16)
            // The inputs must tell the two precisions apart.
            XCTAssertFalse(
                arrayEqual(ricursiveGate(output, gate: gate, fullPrecision: false), expected)
                    .item(Bool.self))
            let actual = ricursiveGate(output, gate: gate, fullPrecision: true)
            XCTAssertEqual(actual.dtype, .bfloat16)
            XCTAssertTrue(arrayEqual(actual, expected).item(Bool.self))
        }
    }

    /// PyTorch multiplies a bfloat16 tensor by a Python float in float32 and rounds once.
    func testEmbedScaleRoundsOnce() throws {
        try Device.withDefaultDevice(.cpu) {
            let h = MLXRandom.normal([1, 4, 64], key: MLXRandom.key(3)).asType(.bfloat16)
            let scale = Float(5120).squareRoot()
            let expected = (h.asType(.float32) * scale).asType(.bfloat16)
            XCTAssertFalse(
                arrayEqual(h * MLXArray(scale).asType(.bfloat16), expected).item(Bool.self))
            let actual = ricursiveEmbedScale(h, scale: scale)
            XCTAssertEqual(actual.dtype, .bfloat16)
            XCTAssertTrue(arrayEqual(actual, expected).item(Bool.self))
        }
    }

    /// Bit patterns of `1.0 / (10000.0 ** (torch.arange(0, 128, 2).float() / 128))` from
    /// torch 2.15, the reference's RoPE frequencies at Ripple's head size.
    func testInverseFrequenciesMatchTorch() throws {
        let torch: [UInt32] = [
            0x3F80_0000, 0x3F5D_AFD7, 0x3F3F_F911, 0x3F26_3DE0, 0x3F0F_F59A, 0x3EF9_53CF,
            0x3ED7_E89B, 0x3EBA_F81B, 0x3EA1_E89B, 0x3E8C_3504, 0x3E72_D423, 0x3E52_47ED,
            0x3E36_1887, 0x3E1D_B040, 0x3E08_8D77, 0x3DEC_7FD6, 0x3DCC_CCCD, 0x3DB1_5978,
            0x3D99_940D, 0x3D84_FE4D, 0x3D66_55C2, 0x3D47_763F, 0x3D2C_BA15, 0x3D15_9348,
            0x3D01_86E3, 0x3CE0_54D2, 0x3CC2_434F, 0x3CA8_398B, 0x3C91_AD39, 0x3C7C_4D33,
            0x3C5A_7BF2, 0x3C3D_3311, 0x3C23_D70A, 0x3C0D_E12D, 0x3BF5_B9B0, 0x3BD4_CA15,
            0x3BB8_449C, 0x3B9F_91CC, 0x3B8A_2E77, 0x3B6F_520E, 0x3B4F_3E38, 0x3B33_770F,
            0x3B1B_690D, 0x3B06_946F, 0x3AE9_1528, 0x3AC9_D75C, 0x3AAE_C98E, 0x3A97_5C0E,
            0x3A83_126F, 0x3A63_01E2, 0x3A44_948C, 0x3A2A_3B44, 0x3A13_6A16, 0x39FF_4FAC,
            0x39DD_1725, 0x39BF_74D7, 0x39A5_CB60, 0x398F_9272, 0x3978_A815, 0x3957_53E4,
            0x393A_7753, 0x3921_7916, 0x390B_D472, 0x38F2_2CE2,
        ]
        let frequencies = ricursiveInverseFrequencies(theta: 10_000, dimensions: 128)
        XCTAssertEqual(frequencies.asArray(Float.self).map(\.bitPattern), torch)
    }

    /// `torch.lerp` on bfloat16 computes in float32 and rounds once, and the blend is the
    /// sigmoid of the bfloat16 lambda rounded to bfloat16. At lambda -2 MLX's bfloat16 sigmoid
    /// rounds differently from the float32 one.
    func testValueResidualBlendRoundsOnce() throws {
        try Device.withDefaultDevice(.cpu) {
            let x = MLXRandom.normal([1, 4, 64], key: MLXRandom.key(4)).asType(.bfloat16)
            let residual = MLXRandom.normal([1, 4, 64], key: MLXRandom.key(5)).asType(.bfloat16)
            for value: Float in [-2, 0.7, 2.39] {
                let lambda = MLXArray([value]).asType(.bfloat16)
                let blend = MLXArray(Float(1 / (1 + Foundation.exp(-Double(value)))))
                    .asType(.bfloat16).asType(.float32)
                let start = residual.asType(.float32)
                let expected = (start + blend * (x.asType(.float32) - start)).asType(.bfloat16)
                let stepwise = residual + sigmoid(lambda[0]) * (x - residual)
                XCTAssertFalse(arrayEqual(stepwise, expected).item(Bool.self), "lambda \(value)")
                let actual = ricursiveValueInput(x, residual: residual, lambda: lambda)
                XCTAssertEqual(actual.dtype, .bfloat16)
                XCTAssertTrue(arrayEqual(actual, expected).item(Bool.self), "lambda \(value)")
            }
        }
    }

    /// `F.silu` on bfloat16 computes in float32 and rounds once.
    func testSiLURoundsOnce() throws {
        try Device.withDefaultDevice(.cpu) {
            let gate = (8 * MLXRandom.normal([4, 256], key: MLXRandom.key(6))).asType(.bfloat16)
            let wide = gate.asType(.float32)
            let expected = (wide / (1 + exp(-wide))).asType(.bfloat16)
            XCTAssertFalse(arrayEqual(silu(gate), expected).item(Bool.self))
            let actual = ricursiveSiLU(gate)
            XCTAssertEqual(actual.dtype, .bfloat16)
            XCTAssertTrue(arrayEqual(actual, expected).item(Bool.self))
        }
    }
}
