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

    /// Parameter `i` in sorted name order, element `j` in row-major order, takes
    /// `v = (j * j * 7 + j * (13 + 2 * i) + i * 31 + seed) % 97 - 48`, then
    /// `(2 * v + 65) / 64` (`v >= 0`) or `(2 * v - 63) / 64` for `value_res_lambda`,
    /// `1 + v / 128` for norm weights, `v / 64` for the MLP gate and up projections and
    /// `v / 256` otherwise. The same formula as the Python reference, keyed by the
    /// reference's parameter names.
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
            XCTAssertLessThan(difference, 1e-3)
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
                XCTAssertLessThan(difference, 1e-3, "prefill step \(step)")
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
            XCTAssertLessThan(difference, 1e-3)
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
        }
    }
}
