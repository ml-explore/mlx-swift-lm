// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN
import XCTest

@testable import MLXLLM
@testable import MLXLMCommon

/// DeepSeek-V4.1 against mlx-lm's `deepseek_v41.py`. Reference values come from
/// `tools/deepseek-v4-reference/deepseek-v41-reference.py`, which builds the same tiny
/// model with the same `deterministicWeight` weights.
final class DeepseekV41Tests: XCTestCase {

    static let configurationJSON = """
        {
            "model_type": "deepseek_v41",
            "vocab_size": 128,
            "hidden_size": 64,
            "num_hidden_layers": 6,
            "num_attention_heads": 4,
            "head_dim": 32,
            "q_lora_rank": 32,
            "qk_rope_head_dim": 8,
            "o_groups": 2,
            "o_lora_rank": 16,
            "moe_intermediate_size": 32,
            "n_routed_experts": 8,
            "num_experts_per_tok": 2,
            "sliding_window": 4,
            "compress_ratios": [0, 2, 2, 2, 1, 1],
            "kv_source_layer_ids": [1, 4],
            "index_source_layer_ids": [1, 2, 4, 5],
            "index_n_heads": 4,
            "index_head_dim": 32,
            "index_topk": 3,
            "candidate_source_layer_id": 4,
            "candidate_topk_blocks": 2,
            "candidate_block_size": 2,
            "rope_scaling": {
                "type": "yarn",
                "factor": 4.0,
                "original_max_position_embeddings": 16,
                "beta_fast": 32,
                "beta_slow": 1
            },
            "max_position_embeddings": 64
        }
        """

    private func configuration(_ json: String = configurationJSON) throws
        -> DeepseekV41Configuration
    {
        try JSONDecoder().decode(DeepseekV41Configuration.self, from: Data(json.utf8))
    }

    // MARK: - Deterministic weights, as tools/deepseek-v4-reference/deepseek-v41-reference.py

    private static func phase(_ name: String) -> Double {
        var h = 0
        for c in name.utf8 {
            h = (h * 31 + Int(c)) % 1_000_003
        }
        return Double(h % 997) / 997 * 2 * Double.pi
    }

    private static func deterministicWeight(_ name: String, _ shape: [Int]) -> MLXArray {
        let count = shape.reduce(1, *)
        let x = sin(
            MLXArray(0 ..< count).asType(.float32) * Float(0.7713) + Float(phase(name))
        ).reshaped(shape)
        if name.hasSuffix("norm.weight") || name.hasSuffix(".scale") {
            return Float(1) + Float(0.1) * x
        }
        if shape.count >= 2 {
            return x / Float(Double(shape[shape.count - 1]).squareRoot())
        }
        return Float(0.5) * x
    }

    private func referenceModel() throws -> DeepseekV41Model {
        let model = DeepseekV41Model(try configuration())
        let weights = model.parameters().flattened().map { name, value in
            (name, Self.deterministicWeight(name, value.shape))
        }
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: [.all])
        eval(model)
        return model
    }

    /// The reference weights packed like DeepSeek's checkpoint, mxfp8 projections and mxfp4
    /// experts, loaded through the shared loader under the checkpoint's `quantization_config`.
    /// The config names `layers.0.self_attn.kv_proj` as unquantized.
    private func packedModel() throws -> DeepseekV41Model {
        let model = DeepseekV41Model(try configuration())
        let modules = Dictionary(uniqueKeysWithValues: model.leafModules().flattened())
        var weights = [String: MLXArray]()
        for (name, value) in model.parameters().flattened() {
            weights[name] = Self.deterministicWeight(name, value.shape)
        }
        for (path, module) in modules where module is Linear || module is SwitchLinear {
            guard !path.hasSuffix("lm_head") else { continue }
            let experts = path.contains(".switch_mlp.")
            let (wq, scales, _) = quantized(
                weights["\(path).weight"]!, groupSize: 32, bits: experts ? 4 : 8,
                mode: experts ? .mxfp4 : .mxfp8)
            weights["\(path).weight"] = wq
            weights["\(path).scales"] = scales
        }
        weights["model.layers.0.self_attn.kv_proj.scales"] = nil
        weights["model.layers.0.self_attn.kv_proj.weight"] = Self.deterministicWeight(
            "model.layers.0.self_attn.kv_proj.weight", [32, 64])

        let base = try JSONDecoder().decode(
            BaseConfiguration.self,
            from: Data(
                #"""
                {"model_type": "deepseek_v41", "quantization_config": {"quant_method": "fp8",
                 "scale_fmt": "ue8m0", "weight_block_size": [32, 32]}}
                """#.utf8))
        var settings = try XCTUnwrap(base.perLayerQuantization)
        XCTAssertEqual(settings.quantization?.mode, .mxfp8)
        settings.perLayerQuantization["model.layers.0.self_attn.kv_proj"] = .skip

        try applyCheckpoint(
            model.prepareCheckpoint(
                ModelCheckpoint(weights: weights, perLayerQuantization: settings)),
            to: model)
        eval(model)
        return model
    }

    private static func floats(_ text: String) -> [Float] {
        text.split(whereSeparator: \.isWhitespace).map { Float(String($0))! }
    }

    private static var tokens: MLXArray {
        MLXArray(DeepseekV41Reference.tokens.flatMap { $0 }).reshaped(2, 16)
    }

    // MARK: - Parity with mlx-lm

    func testParameterNamesMatchMLXLM() throws {
        let model = DeepseekV41Model(try configuration())
        let shapes = Dictionary(
            uniqueKeysWithValues: model.parameters().flattened().map { ($0, $1.shape) })
        XCTAssertEqual(shapes, DeepseekV41Reference.parameterShapes)
    }

    func testLogitsMatchMLXLM() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = try referenceModel()
            let logits = model(Self.tokens, cache: nil)
            let expected = MLXArray(Self.floats(DeepseekV41Reference.logits)).reshaped(2, 16, 128)
            XCTAssertEqual(logits.shape, [2, 16, 128])
            assertClose(logits, expected, rtol: 1e-4, atol: 1e-4, reducedAtol: 2e-2, "logits")
        }
    }

    func testCachedDecodeMatchesFullForward() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = try referenceModel()
            let tokens = Self.tokens
            let full = model(tokens, cache: nil)
            let cache = try model.newCache(parameters: nil)
            let steps = [(0, 7), (7, 15), (15, 16)].map { start, end in
                model(tokens[0..., start ..< end], cache: cache)
            }
            XCTAssertEqual(cache.map(\.offset), Array(repeating: 16, count: 6))
            assertClose(
                concatenated(steps, axis: 1), full, rtol: 1e-4, atol: 1e-4, reducedAtol: 2e-2,
                "chunked")

            let single = try model.newCache(parameters: nil)
            let tokenByToken = (0 ..< 16).map { i in
                model(tokens[0..., i ..< (i + 1)], cache: single)
            }
            assertClose(
                concatenated(tokenByToken, axis: 1), full, rtol: 1e-4, atol: 1e-4,
                reducedAtol: 2e-2, "token by token")
        }
    }

    func testFakeQuantMatchesMLXLM() {
        Device.withDefaultDevice(.cpu) {
            let x = Float(3) * sin(MLXArray(0 ..< 128).asType(.float32) * Float(1.37))
            let input = x.reshaped(2, 64)
            let cases: [(MLXArray, String)] = [
                (deepseekV41FakeQuantFP8(input, blockSize: 32), DeepseekV41Reference.fakeQuantFP8),
                (deepseekV41FakeQuantFP4(input, blockSize: 32), DeepseekV41Reference.fakeQuantFP4),
                (
                    deepseekV41FakeQuantFP4(input, blockSize: 16, e4m3Scale: true),
                    DeepseekV41Reference.fakeQuantFP4E4M3
                ),
            ]
            for (index, (actual, reference)) in cases.enumerated() {
                let expected = MLXArray(Self.floats(reference)).reshaped(2, 64)
                XCTAssertTrue(allClose(actual, expected, atol: 1e-6).item(), "case \(index)")
            }
        }
    }

    func testRopeInverseRoundTrip() {
        let rope = YarnRoPE(
            dimensions: 4, traditional: true, base: 160000, scalingFactor: 2,
            originalMaxPositionEmbeddings: 16, mscale: 0, mscaleAllDim: 0)
        for shape in [[1, 5, 8], [1, 5, 2, 8]] {
            let x = MLXRandom.normal(shape, key: MLXRandom.key(0))
            var y = deepseekV41ApplyRope(x, rope, ropeDim: 4, offset: 4, scale: 2)
            XCTAssertFalse(allClose(y, x, atol: 1e-3).item(), "rotated \(shape)")
            y = deepseekV41ApplyRope(y, rope, ropeDim: 4, offset: 4, scale: 2, inverse: true)
            XCTAssertTrue(allClose(y, x, atol: 1e-5).item(), "round trip \(shape)")
        }
    }

    func testSinkhornKernelMatchesOps() throws {
        guard Device.defaultDevice().deviceType == .gpu else {
            throw XCTSkip("The Sinkhorn kernel needs a GPU")
        }
        let logits = MLXRandom.normal([3, 5, 4, 4], key: MLXRandom.key(1))
        let comb = softmax(logits, axis: -1) + Float(1e-6)
        let eps = MLXArray([Float(1e-6)])
        let kernel = deepseekV41Sinkhorn(comb, iterations: 20, eps: eps)
        let ops = deepseekV41Sinkhorn(comb, iterations: 20, eps: eps, useKernel: false)
        XCTAssertEqual(kernel.shape, comb.shape)
        XCTAssertTrue(allClose(kernel, ops, atol: 1e-6).item())
        // Doubly stochastic within the last row pass.
        XCTAssertTrue(allClose(kernel.sum(axis: -2), MLXArray.ones([3, 5, 4]), atol: 1e-4).item())
    }

    // MARK: - Weights

    func testRenameMatchesMLXLM() {
        for (raw, expected) in DeepseekV41Reference.renames {
            XCTAssertEqual(DeepseekV41Weights.rename(raw), expected, raw)
        }
    }

    func testRepackMatchesMLXLM() {
        let weight = (MLXArray(0 ..< (64 * 64)).asType(.uint32) % UInt32(251)).asType(.uint8)
            .reshaped(64, 64)
        let scale = MLXArray([0.5, 2.0, 4.0, 0.125] as [Float]).reshaped(2, 2)
        let (packed, scales) = DeepseekV41Weights.repack(weight: weight, scale: scale)
        XCTAssertEqual(packed.dtype, .uint32)
        XCTAssertEqual(packed.shape, DeepseekV41Reference.repackedShape)
        XCTAssertEqual(
            packed.reshaped(-1)[..<8].asArray(UInt32.self), DeepseekV41Reference.repackedWords)
        XCTAssertEqual(scales.dtype, .uint8)
        XCTAssertEqual(scales.shape, DeepseekV41Reference.repackedScalesShape)
        XCTAssertEqual(
            scales[.stride(by: 31)].reshaped(-1).asArray(UInt8.self),
            DeepseekV41Reference.repackedScales)
    }

    func testSanitizeRawCheckpoint() throws {
        let args = try configuration()
        let model = DeepseekV41Model(args)
        var raw = [String: MLXArray]()
        for expert in 0 ..< 8 {
            for projection in ["w1", "w2", "w3"] {
                let prefix = "layers.0.ffn.experts.\(expert).\(projection)"
                raw["\(prefix).weight"] = MLXArray.zeros([32, 32], dtype: .int8)
                raw["\(prefix).scale"] = MLXArray.ones([32, 2], dtype: .uint8)
            }
        }
        raw["layers.0.attn.wo_a.weight"] = MLXArray.zeros([32, 16], dtype: .uint8)
        raw["layers.0.attn.wo_a.scale"] = MLXArray.ones([1, 2], dtype: .float32)
        raw["layers.0.ffn.gate.bias"] = MLXArray.zeros([8], dtype: .float32)
        raw["layers.0.ffn.gate.bias_vl"] = MLXArray.zeros([8], dtype: .float32)
        raw["mtp.0.norm.weight"] = MLXArray.zeros([64])
        raw["vision.patch.weight"] = MLXArray.zeros([4])
        raw["layers.0.engram.table.weight"] = MLXArray.zeros([4])

        let clean = try model.sanitize(weights: raw)
        XCTAssertEqual(
            Set(clean.keys),
            [
                "model.layers.0.mlp.experts.switch_mlp.gate_proj.weight",
                "model.layers.0.mlp.experts.switch_mlp.gate_proj.scales",
                "model.layers.0.mlp.experts.switch_mlp.up_proj.weight",
                "model.layers.0.mlp.experts.switch_mlp.up_proj.scales",
                "model.layers.0.mlp.experts.switch_mlp.down_proj.weight",
                "model.layers.0.mlp.experts.switch_mlp.down_proj.scales",
                "model.layers.0.self_attn.o_a_proj.weight",
                "model.layers.0.self_attn.o_a_proj.scales",
                "model.layers.0.mlp.gate.e_score_correction_bias",
            ])
        let experts = clean["model.layers.0.mlp.experts.switch_mlp.gate_proj.weight"]!
        XCTAssertEqual(experts.shape, [8, 32, 8])
        XCTAssertEqual(experts.dtype, .uint32)
        XCTAssertEqual(
            clean["model.layers.0.mlp.experts.switch_mlp.gate_proj.scales"]!.shape, [8, 32, 2])
        // fp8 tile scales repeat over 32 rows, then split into the output groups.
        XCTAssertEqual(clean["model.layers.0.self_attn.o_a_proj.weight"]!.shape, [2, 16, 4])
        let oScales = clean["model.layers.0.self_attn.o_a_proj.scales"]!
        XCTAssertEqual(oScales.shape, [2, 16, 2])
        XCTAssertEqual(oScales.dtype, .uint8)
        XCTAssertEqual(oScales.max().item(Int.self), 127)

        raw["layers.0.ffn.experts.7.w2.weight"] = nil
        raw["layers.0.ffn.experts.7.w2.scale"] = nil
        XCTAssertThrowsError(try model.sanitize(weights: raw))
    }

    func testSanitizeKeepsConvertedCheckpoint() throws {
        let model = DeepseekV41Model(try configuration())
        let weights = [
            "model.norm.weight": MLXArray.ones([64]), "layers.0.ffn.gate.bias": MLXArray.zeros([8]),
        ]
        XCTAssertEqual(Set(try model.sanitize(weights: weights).keys), Set(weights.keys))
    }

    /// DeepSeek's checkpoint loads its mxfp8 projections and mxfp4 experts packed: the
    /// `quantization_config` gives the mxfp8 default, the loader infers the experts' mode, and a
    /// layer the config names keeps its setting.
    func testLoaderInfersQuantization() throws {
        let model = try packedModel()
        let loaded = Dictionary(uniqueKeysWithValues: model.leafModules().flattened())
        let experts = try XCTUnwrap(
            loaded["model.layers.0.mlp.experts.switch_mlp.up_proj"] as? QuantizedSwitchLinear)
        XCTAssertEqual(experts.mode, .mxfp4)
        XCTAssertEqual(experts.bits, 4)
        let output = try XCTUnwrap(loaded["model.layers.3.self_attn.o_a_proj"] as? Quantized)
        XCTAssertEqual(output.mode, .mxfp8)
        XCTAssertEqual(output.bits, 8)
        XCTAssertFalse(loaded["model.layers.0.self_attn.kv_proj"] is Quantized)

        try Device.withDefaultDevice(.cpu) {
            let logits = model(Self.tokens, cache: try model.newCache(parameters: nil))
            XCTAssertEqual(logits.shape, [2, 16, 128])
            XCTAssertTrue(all(isFinite(logits)).item(Bool.self))
        }
    }

    // MARK: - Configuration, registry and cache

    func testConfigurationReadsTextConfig() throws {
        let nested = """
            {"model_type": "deepseek_v41_text", "text_config": \(Self.configurationJSON)}
            """
        let args = try configuration(nested)
        XCTAssertEqual(args.modelType, "deepseek_v41_text")
        XCTAssertEqual(args.compressRatios, [0, 2, 2, 2, 1, 1])
        XCTAssertEqual(args.hiddenSize, 64)
        XCTAssertNoThrow(try args.validateModelConfiguration())

        let defaults = try configuration("{}")
        XCTAssertEqual(defaults.rmsNormEps, 1e-20)
        XCTAssertEqual(defaults.candidateSourceLayerId, -1)
        XCTAssertEqual(defaults.hcSinkhornIters, 20)
    }

    func testConfigurationRejectsUnsupportedCheckpoints() throws {
        let engram = try configuration(#"{"engram_layer_ids": [3]}"#)
        XCTAssertThrowsError(try engram.validateModelConfiguration())

        let blockFP8 = try configuration(
            #"{"quantization_config": {"quant_method": "fp8", "weight_block_size": [128, 128]}}"#)
        XCTAssertThrowsError(try blockFP8.validateModelConfiguration())

        let mxfp8 = try configuration(
            #"""
            {"quantization_config": {"quant_method": "fp8", "scale_fmt": "ue8m0",
             "weight_block_size": [32, 32]}}
            """#)
        XCTAssertNoThrow(try mxfp8.validateModelConfiguration())
    }

    func testRegisteredModelTypes() async throws {
        for modelType in ["deepseek_v41", "deepseek_v41_text"] {
            let model = try await LLMTypeRegistry.shared.createModel(
                configuration: Data(Self.configurationJSON.utf8), modelType: modelType)
            XCTAssertTrue(model is DeepseekV41Model, modelType)
        }
    }

    func testCacheIsArchitectureOwned() throws {
        let model = DeepseekV41Model(try configuration())
        let cache = try model.newCache(parameters: nil)
        XCTAssertEqual(cache.count, 6)
        XCTAssertFalse(cache[0].isTrimmable)

        _ = model(Self.tokens[0..., ..<5], cache: cache)
        XCTAssertEqual(cache.map(\.offset), Array(repeating: 5, count: 6))
        let copy = cache.map { $0.copy() }
        _ = model(Self.tokens[0..., 5 ..< 6], cache: cache)
        XCTAssertEqual(copy.map(\.offset), Array(repeating: 5, count: 6))
        XCTAssertEqual(cache.map(\.offset), Array(repeating: 6, count: 6))

        // A capped cache and a typed quantization request are rejected; the legacy
        // `kvBits` request allows partial coverage, so it leaves these caches as they are.
        XCTAssertThrowsError(try GenerateParameters(maxKVSize: 8).kvCachePlan().validated(cache))
        XCTAssertThrowsError(
            try validateKVCacheCompatibility(
                cache, configuration: KVCacheConfiguration(strategy: .affine(.fourBit))))
        let legacy = try GenerateParameters(kvBits: 4).kvCachePlan()
        var quantized = try legacy.validated(cache)
        legacy.apply(to: &quantized)
        XCTAssertTrue(quantized.allSatisfy { $0 is DeepseekV41LayerCache })
    }

    /// A saved cache, with evicted window rows, open groups and empty slots, decodes on after
    /// loading exactly as the cache it was saved from.
    func testPromptCacheRoundTrip() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = try referenceModel()
            let tokens = Self.tokens
            let cache = try model.newCache(parameters: nil)
            _ = model(tokens[0..., ..<11], cache: cache)
            let layers = cache.compactMap { $0 as? DeepseekV41LayerCache }
            XCTAssertEqual(layers[0].metaState, ["4", "11", "10000"])
            XCTAssertTrue(layers.contains { $0.compressKV != nil && $0.pendingKV != nil })

            let url = FileManager.default.temporaryDirectory
                .appending(component: "deepseek-v41-\(UUID().uuidString).safetensors")
            defer { try? FileManager.default.removeItem(at: url) }
            try savePromptCache(url: url, cache: cache, metadata: ["model": "deepseek_v41"])
            let (restored, metadata) = try loadPromptCache(url: url)
            XCTAssertEqual(metadata, ["model": "deepseek_v41"])
            XCTAssertTrue(restored.allSatisfy { $0 is DeepseekV41LayerCache })
            XCTAssertEqual(restored.map(\.metaState), cache.map(\.metaState))

            let next = tokens[0..., 11 ..< 16]
            let expected = model(next, cache: cache)
            let actual = model(next, cache: restored)
            XCTAssertTrue(arrayEqual(actual, expected).item(Bool.self))
        }
    }

    func testPromptCacheRejectsCorruptMetadata() throws {
        let state = [MLXArray.zeros([1, 2, 8])]
        for metaState in [
            ["4", "2"], ["4", "2", "11000"], ["0", "2", "10000"], ["4", "x", "10000"],
        ] {
            XCTAssertThrowsError(
                try DeepseekV41LayerCache.restore(state: state, metaState: metaState),
                "\(metaState)")
        }
        XCTAssertThrowsError(
            try DeepseekV41LayerCache.restore(
                state: [MLXArray.zeros([1, 5, 8])], metaState: ["4", "5", "10000"]))
        XCTAssertNoThrow(
            try DeepseekV41LayerCache.restore(state: state, metaState: ["4", "2", "10000"]))
    }
}
