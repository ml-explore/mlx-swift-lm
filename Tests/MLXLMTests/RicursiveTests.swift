import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXLLM

/// Ricursive against the model author's `modeling_ricursive.py` (ricursiveAI/Ripple-65B-Preview,
/// revision and sha256 in `RicursiveReferenceValues`). The tiny model has four sliding layers
/// and one full layer, and an 8-token window shorter than its 24-token prompt. Its weights
/// come from the closed-form formula in `tools/ricursive_parity/ricursive-reference.py`,
/// which also writes the expected values in `RicursiveReferenceValues`.
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

    /// Checks that a checkpoint's weight names are exactly the ones this class needs.
    /// Set `RICURSIVE_CHECKPOINT_DIRS` to a colon-separated list of directories holding
    /// `config.json` and `model.safetensors.index.json`; no weights are read. The model is
    /// built at a tiny width because the names depend only on the layer stack.
    func testCheckpointWeightNames() throws {
        guard let list = ProcessInfo.processInfo.environment["RICURSIVE_CHECKPOINT_DIRS"] else {
            throw XCTSkip("RICURSIVE_CHECKPOINT_DIRS is not set")
        }
        for path in list.split(separator: ":") {
            let directory = URL(fileURLWithPath: String(path))
            let configData = try Data(contentsOf: directory.appendingPathComponent("config.json"))
            var json = try XCTUnwrap(
                JSONSerialization.jsonObject(with: configData) as? [String: Any])
            let tiny: [String: Any] = [
                "vocab_size": 64, "hidden_size": 32, "intermediate_size": 64,
                "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
                "sliding_window": 8,
            ]
            json.merge(tiny) { $1 }
            let config = try JSONDecoder().decode(
                RicursiveConfiguration.self,
                from: JSONSerialization.data(withJSONObject: json))
            try config.validateModelConfiguration()
            let base = try JSONDecoder().decode(BaseConfiguration.self, from: configData)

            let index = try XCTUnwrap(
                JSONSerialization.jsonObject(
                    with: Data(
                        contentsOf: directory.appendingPathComponent(
                            "model.safetensors.index.json"))) as? [String: Any])
            let map = try XCTUnwrap(index["weight_map"] as? [String: Any])
            let names = Set(map.keys)

            let model = RicursiveModel(config)
            if let perLayer = base.perLayerQuantization {
                quantize(model: model) { path, _ in
                    guard names.contains("\(path).scales"),
                        let option = perLayer.quantization(layer: path)
                    else { return nil }
                    return (32, option.bits, option.mode)
                }
            }
            let expected = Set(model.parameters().flattened().map(\.0))
            XCTAssertEqual(
                expected.subtracting(names).sorted(), [], "\(path): missing from the checkpoint")
            XCTAssertEqual(
                names.subtracting(expected).sorted(), [], "\(path): not used by the model")
            print("ricursive names \(directory.lastPathComponent): \(names.count) keys match")
        }
    }
}

/// Full-size parity against an oracle made by `tools/ricursive_parity/run_job.sh`.
/// Needs `RICURSIVE_PARITY_MODEL` and `RICURSIVE_PARITY_ORACLE`. With
/// `RICURSIVE_PARITY_MODE=exact` (the default) every greedy token must match; `report`
/// only records the comparison, for quantized weights. Set `RICURSIVE_PARITY_REPORT`
/// to write the numbers as JSON.
final class RicursivePytorchParityTests: XCTestCase {
    private struct Row: Encodable {
        let length: Int
        let steps: Int
        let firstMismatch: Int?
        let matchingTokens: Int
        let meanKL: Float
        let maxLogitDifference: Float
        let prefillSeconds: Double
        let decodeTokensPerSecond: Double
    }

    func testGreedyTokensMatchPytorchOracle() throws {
        let environment = ProcessInfo.processInfo.environment
        guard let modelPath = environment["RICURSIVE_PARITY_MODEL"],
            let oraclePath = environment["RICURSIVE_PARITY_ORACLE"]
        else {
            throw XCTSkip("RICURSIVE_PARITY_MODEL and RICURSIVE_PARITY_ORACLE are not set")
        }
        let exact = (environment["RICURSIVE_PARITY_MODE"] ?? "exact") == "exact"
        var prefill = PrefillParameters()
        if let step = Int(environment["RICURSIVE_PARITY_STEP"] ?? "") {
            prefill.stepSize = step
        }

        let directory = URL(fileURLWithPath: modelPath)
        let configData = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        let configuration = try JSONDecoder().decode(RicursiveConfiguration.self, from: configData)
        try configuration.validateModelConfiguration()
        let base = try JSONDecoder().decode(BaseConfiguration.self, from: configData)
        let model = RicursiveModel(configuration)
        try loadWeights(
            modelDirectory: directory, model: model,
            perLayerQuantization: base.perLayerQuantization)
        model.train(false)

        let oracleDirectory = URL(fileURLWithPath: oraclePath)
        let oracles = try FileManager.default.contentsOfDirectory(
            at: oracleDirectory, includingPropertiesForKeys: nil
        )
        .filter { $0.lastPathComponent.hasPrefix("oracle-") && $0.pathExtension == "safetensors" }
        .sorted {
            $0.path.compare($1.path, options: .numeric) == .orderedAscending
        }
        XCTAssertFalse(oracles.isEmpty, "no oracle files in \(oraclePath)")

        var report = [Row]()
        for url in oracles {
            let arrays = try loadArrays(url: url)
            let prompt = try XCTUnwrap(arrays["prompt"])
            let expected = try XCTUnwrap(arrays["greedy"]).asArray(Int32.self).map(Int.init)
            let logits = try XCTUnwrap(arrays["logits"])

            let cache = try model.newCache(parameters: nil)
            let begin = Date()
            guard
                case .tokens(let rest) = try model.prepare(
                    LMInput(tokens: prompt), cache: cache, state: nil, prefill: prefill)
            else {
                return XCTFail("prepare should hand tokens to the iterator")
            }
            eval(cache)
            let prefillSeconds = Date().timeIntervalSince(begin)

            var input = rest.tokens.reshaped(1, rest.tokens.size)
            var tokens = [Int]()
            var divergences = [Float]()
            var worst: Float = 0
            let decodeBegin = Date()
            for step in expected.indices {
                // The iterator's path: `nextTokenLogits` may trim the vocabulary projection.
                let row = model.nextTokenLogits(
                    LMInput.Text(tokens: input), cache: cache, state: nil
                ).logits[0, -1].asType(.float32)
                eval(row)
                eval(cache)
                let token = argMax(row, axis: -1).item(Int.self)
                tokens.append(token)

                let reference = logits[step]
                let referenceLog = reference - logSumExp(reference, axis: -1, keepDims: true)
                let log = row - logSumExp(row, axis: -1, keepDims: true)
                divergences.append(
                    (exp(referenceLog) * (referenceLog - log)).sum().item(Float.self))
                worst = max(worst, abs(row - reference).max().item(Float.self))

                // Follow the oracle's token so one miss does not corrupt later comparisons.
                input = MLXArray([Int32(expected[step])]).reshaped(1, 1)
            }
            let decodeSeconds = Date().timeIntervalSince(decodeBegin)

            let matches = zip(tokens, expected).filter { $0 == $1 }.count
            let result = Row(
                length: prompt.size, steps: expected.count,
                firstMismatch: zip(tokens, expected).enumerated().first {
                    $0.element.0 != $0.element.1
                }?
                .offset,
                matchingTokens: matches,
                meanKL: divergences.reduce(0, +) / Float(divergences.count),
                maxLogitDifference: worst,
                prefillSeconds: prefillSeconds,
                decodeTokensPerSecond: Double(expected.count) / decodeSeconds)
            report.append(result)
            print("ricursive parity \(result)")
            if exact {
                XCTAssertEqual(tokens, expected, "greedy tokens at prompt length \(prompt.size)")
            }
        }

        if let path = environment["RICURSIVE_PARITY_REPORT"] {
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            try encoder.encode(report).write(to: URL(fileURLWithPath: path))
        }
    }
}
