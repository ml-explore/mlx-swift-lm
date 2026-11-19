// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN
import XCTest

@testable import MLXLLM
@testable import MLXLMCommon

final class Qwen4ExpTests: XCTestCase {

    // MARK: - Helpers

    private func configurationData(
        _ edit: (inout [String: Any]) -> Void = { _ in }
    ) throws -> Data {
        var root = try XCTUnwrap(
            JSONSerialization.jsonObject(with: Data(Qwen4ExpReferenceValues.configuration.utf8))
                as? [String: Any])
        var text = try XCTUnwrap(root["text_config"] as? [String: Any])
        edit(&text)
        root["text_config"] = text
        return try JSONSerialization.data(withJSONObject: root)
    }

    private func configuration(
        _ edit: (inout [String: Any]) -> Void = { _ in }
    ) throws -> Qwen4ExpConfiguration {
        let configuration = try JSONDecoder.json5().decode(
            Qwen4ExpConfiguration.self, from: configurationData(edit))
        try configuration.validateModelConfiguration()
        return configuration
    }

    /// Weights in sorted parameter-name order, as in tools/qwen4-exp-reference.py.
    private func referenceModel() throws -> Qwen4ExpModel {
        let model = Qwen4ExpModel(try configuration())
        var weights = [String: MLXArray]()
        let parameters = model.parameters().flattened().sorted { $0.0 < $1.0 }
        for (i, (name, parameter)) in parameters.enumerated() {
            let indices = MLXArray(Int32(0) ..< Int32(parameter.size))
            var values = ((indices * 7 + Int32(i * 13)) % 97 - 48).asType(.float32) / 256
            if parameter.ndim == 1 && name.hasSuffix(".weight") {
                values = values + 1
            }
            weights[name] = values.reshaped(parameter.shape)
        }
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: [.all])
        eval(model)
        return model
    }

    private var referenceTokens: MLXArray {
        let rows = Qwen4ExpReferenceValues.tokens
        return MLXArray(rows.flatMap { $0 }).reshaped(rows.count, -1)
    }

    private func assertMatchesReference(
        _ logits: MLXArray, _ label: String, file: StaticString = #filePath, line: UInt = #line
    ) throws {
        let positions = MLXArray(Qwen4ExpReferenceValues.positions.map(Int32.init))
        let slice = take(logits, positions, axis: 1)[
            .ellipsis, ..<Qwen4ExpReferenceValues.vocabularySlice]
        let actual = ["logits": slice, "logsumexp": logits.logSumExp(axis: -1)]
        for (key, array) in actual {
            let values = try XCTUnwrap(Qwen4ExpReferenceValues.outputs[key], key)
            XCTAssertEqual(array.size, values.count, key, file: file, line: line)
            let difference = abs(array - MLXArray(values, array.shape)).max().item(Float.self)
            XCTAssertLessThan(difference, 1e-4, "\(label) \(key)", file: file, line: line)
        }
    }

    private static let foldOne = [
        "q_layernorm.weight", "k_layernorm.weight", "q_norm.weight", "k_norm.weight",
        "hc_norm.weight", "norm_key.weight", "norm_query.weight", "norm_conv.weight",
    ]

    private func flattened(_ model: Module) -> [String: MLXArray] {
        Dictionary(uniqueKeysWithValues: model.parameters().flattened())
    }

    private func assertSame(
        _ actual: [String: MLXArray], _ expected: [String: MLXArray], _ label: String,
        file: StaticString = #filePath, line: UInt = #line
    ) {
        XCTAssertEqual(Set(actual.keys), Set(expected.keys), label, file: file, line: line)
        for (key, value) in expected {
            guard let other = actual[key] else { continue }
            XCTAssertTrue(
                other.shape == value.shape && arrayEqual(other, value).item(Bool.self),
                "\(label): \(key)", file: file, line: line)
        }
    }

    // MARK: - Configuration and registry

    func testRegistryCreatesModel() async throws {
        let model = try await LLMTypeRegistry.shared.createModel(
            configuration: try configurationData(), modelType: "qwen4_exp")
        XCTAssertTrue(model is Qwen4ExpModel)
        XCTAssertEqual(model.toolCallFormat, .xmlFunction)
        XCTAssertEqual(model.reasoningConfig, QwenReasoningProtocol.tagged)
    }

    func testConfigurationDefaults() throws {
        let text = try configuration().text
        XCTAssertEqual(text.eosTokenId, 1)
        XCTAssertEqual(text.ropeTheta, 10_000_000)
        XCTAssertEqual(text.partialRotaryFactor, 0.25)
        XCTAssertEqual(text.seed, 1234)
        XCTAssertEqual(text.pleConvKernelSize, 4)
        XCTAssertEqual(text.outputGateType, "silu")
        XCTAssertEqual(
            text.layerTypes,
            ["linear_attention", "linear_attention", "linear_attention", "full_attention"])
    }

    func testReleasedConfigurationDecodes() throws {
        // Qwen/Qwen3.8-Flash-Next config.json; `layer_types` repeats 3 linear + 1 full.
        let layerTypes: [String] = Array(
            Array(
                repeating: [
                    "linear_attention", "linear_attention", "linear_attention", "full_attention",
                ], count: 12
            ).joined())
        let layerTypesJSON = layerTypes.map { "\"\($0)\"" }.joined(separator: ",")
        let json = """
            {
              "architectures": ["Qwen4ExpForConditionalGeneration"],
              "image_token_id": 248056, "language_model_only": false,
              "model_type": "qwen4_exp", "tie_word_embeddings": false,
              "video_token_id": 248057, "vision_end_token_id": 248054,
              "vision_start_token_id": 248053,
              "vision_config": {"depth": 27, "hidden_size": 1152, "model_type": "qwen4_exp"},
              "text_config": {
                "attention_bias": false, "attention_dropout": 0.0, "bos_token_id": 248044,
                "dtype": "bfloat16", "eos_token_id": 248044, "full_attention_interval": 4,
                "hc_count": 4, "hc_lowrank": 320, "head_dim": 256, "heads_per_ngram": 8,
                "hidden_act": "silu", "hidden_size": 2560, "indexer_budget": 2048,
                "indexer_compress_ratio": 4, "indexer_head_dim": 128, "indexer_kv_heads": 1,
                "indexer_n_heads": 4, "initializer_range": 0.02,
                "layer_types": [\(layerTypesJSON)],
                "linear_conv_kernel_dim": 4, "linear_key_head_dim": 128,
                "linear_num_key_heads": 16, "linear_num_value_heads": 48,
                "linear_value_head_dim": 128, "make_ngram_vocab_size_divisible_by": 128,
                "mamba_ssm_dtype": "float32", "max_position_embeddings": 262144,
                "model_type": "qwen4_exp_text", "moe_intermediate_size": 640,
                "mtp": {
                  "hybrid": true, "layer_types": ["full_attention"],
                  "mtp_use_hidden_state_from_layer": null, "num_hidden_layers": 1,
                  "rope_theta": 10000000
                },
                "mtp_num_hidden_layers": 1, "mtp_use_dedicated_embeddings": false,
                "ngram_size": 3, "ngram_vocab_size_base": 20000000, "num_attention_heads": 24,
                "num_experts": 512, "num_experts_per_tok": 10, "num_hidden_layers": 48,
                "num_key_value_heads": 2, "output_gate_type": "sigmoid",
                "output_router_logits": false, "pad_token_id": null,
                "partial_rotary_factor": 0.25, "ple_conv_kernel_size": 4,
                "ple_embed_dim": 2560, "ple_layer_ids": [2], "rms_norm_eps": 1e-06,
                "rope_parameters": {
                  "mrope_interleaved": true, "mrope_section": [11, 11, 10],
                  "partial_rotary_factor": 0.25, "rope_theta": 10000000,
                  "rope_type": "default"
                },
                "router_aux_loss_coef": 0.001, "shared_expert_intermediate_size": 640,
                "split_ngram_parts": 128, "tie_word_embeddings": false, "use_cache": true,
                "vocab_size": 248320
              }
            }
            """
        let configuration = try JSONDecoder.json5().decode(
            Qwen4ExpConfiguration.self, from: Data(json.utf8))
        try configuration.validateModelConfiguration()
        let text = configuration.text
        XCTAssertEqual(text.layerTypes, layerTypes)
        XCTAssertEqual(text.eosTokenId, 248044)
        XCTAssertEqual(text.ropeTheta, 10_000_000)
        XCTAssertEqual(text.partialRotaryFactor, 0.25)
        XCTAssertEqual(text.outputGateType, "sigmoid")
        XCTAssertEqual(text.pleLayerIds, [2])
    }

    func testConfigurationRejectsLayoutsThePortCannotRun() throws {
        let invalid: [(String, (inout [String: Any]) -> Void)] = [
            ("PLE on an attention layer", { $0["ple_layer_ids"] = [4] }),
            ("two indexer key heads", { $0["indexer_kv_heads"] = 2 }),
            ("no attention layer", { $0["full_attention_interval"] = 8 }),
            ("indexer budget below one block", { $0["indexer_budget"] = 2 }),
            ("PLE width not divisible by heads", { $0["ple_embed_dim"] = 130 }),
        ]
        for (label, edit) in invalid {
            XCTAssertThrowsError(try configuration(edit), label)
        }
    }

    // MARK: - Python reference

    func testLogitsMatchPythonReference() throws {
        // CPU pins IEEE FP32, as the reference does.
        try Device.withDefaultDevice(.cpu) {
            let model = try referenceModel()
            let tokens = referenceTokens
            let full = model(tokens, cache: nil)
            try assertMatchesReference(full, "full forward")

            let prefill = Qwen4ExpReferenceValues.prefill
            let cache = try model.newCache(parameters: nil)
            var steps = [model(tokens[0..., ..<prefill], cache: cache)]
            for i in prefill ..< tokens.dim(1) {
                steps.append(model(tokens[0..., i ..< (i + 1)], cache: cache))
            }
            try assertMatchesReference(concatenated(steps, axis: 1), "prefill and decode")
        }
    }

    func testNGramConstantsAreCheckedThenDropped() throws {
        let model = Qwen4ExpModel(try configuration())
        XCTAssertEqual(Qwen4ExpReferenceValues.ngramConstants.count, 6)
        for (key, values) in Qwen4ExpReferenceValues.ngramConstants {
            // Model, HF and mlx-vlm names.
            for prefix in ["model.", "model.language_model.", "language_model.model."] {
                let name = prefix + key.dropFirst("model.".count)
                XCTAssertEqual(try model.sanitize(weights: [name: MLXArray(values)]).count, 0, name)
            }
            var wrong = values
            wrong[0] += 2
            XCTAssertThrowsError(try model.sanitize(weights: [key: MLXArray(wrong)]), key)
        }
    }

    // MARK: - Checkpoint layouts

    /// The layouts of mlx-lm's test: a split block inject, n-gram row shards, and norms
    /// as `w` for HF and mlx-vlm.
    func testSanitizeLoadsEveryQuantizedCheckpointLayout() throws {
        let model = try referenceModel()
        quantize(model: model, groupSize: 32, bits: 4) { _, module in
            guard module is Quantizable,
                let weight = module.parameters().flattened().first(where: { $0.0 == "weight" })
            else { return false }
            return weight.1.dim(-1) % 32 == 0
        }
        let weights = flattened(model)
        XCTAssertNotNil(weights["model.layers.0.ple.ple_embedding.ngram_embedding.0.scales"])
        assertSame(try model.sanitize(weights: weights), weights, "already sanitized")
        // Quantized n-gram tables take the int64 hash rows.
        let logits = model(referenceTokens, cache: try model.newCache(parameters: nil))
        XCTAssertTrue(all(isFinite(logits)).item(Bool.self))

        let layouts = [
            ("model.language_model.", true, "shard_"),  // HF
            ("language_model.model.", true, "shards."),  // mlx-vlm
            ("language_model.model.", false, "shard_"),  // TensorFold
            ("model.", false, "shards."),  // older mlx-lm conversions
        ]
        for (prefix, raw, shard) in layouts {
            let checkpoint = try checkpointLayout(
                weights, lowrank: model.configuration.hcLowrank, prefix: prefix, raw: raw,
                shard: shard)
            assertSame(try model.sanitize(weights: checkpoint), weights, prefix + shard)
        }
    }

    /// `weights` of a quantized model as a checkpoint stores them: under `prefix`, with
    /// unfolded norms when `raw`, the block inject split out, and n-gram row shards.
    private func checkpointLayout(
        _ weights: [String: MLXArray], lowrank: Int, prefix: String, raw: Bool, shard: String
    ) throws -> [String: MLXArray] {
        var checkpoint = [String: MLXArray]()
        for (key, value) in weights {
            var key = key
            if let range = key.range(of: "model.") {
                key.replaceSubrange(range, with: prefix)
            }
            let fold = raw && Self.foldOne.contains { key.hasSuffix($0) }
            checkpoint[key] = fold ? value - 1 : value
        }
        for key in checkpoint.keys where key.hasSuffix("input_mix_weight_down.weight") {
            let base = String(key.dropLast("weight".count))
            let inject = base.replacingOccurrences(
                of: "input_mix_weight_down", with: "block_inject_weight")
            for name in ["weight", "scales", "biases"] {
                let value = try XCTUnwrap(checkpoint.removeValue(forKey: base + name))
                checkpoint[base + name] = value[..<lowrank]
                checkpoint[inject + name] = value[lowrank...]
            }
        }
        for layer in [0, 1] {
            let tables = "\(prefix)layers.\(layer).ple.ple_embedding.ngram_embedding."
            for name in ["weight", "scales", "biases"] {
                let parts = (0 ..< 4).map {
                    checkpoint.removeValue(forKey: "\(tables)\($0).\(name)")
                }
                let table = concatenated(try parts.map { try XCTUnwrap($0) })
                let rows = (table.dim(0) + 127) / 128 * 128
                let padded = concatenated([
                    table,
                    MLXArray.zeros([rows - table.dim(0), table.dim(1)], dtype: table.dtype),
                ])
                for (i, part) in padded.split(parts: 4).enumerated() {
                    checkpoint["\(tables)\(shard)\(i).\(name)"] = part
                }
            }
        }
        return checkpoint
    }

    /// The loader picks each layer's precision by its model path, so the per-layer
    /// settings must follow the merged block inject and the split n-gram shards.
    func testPerLayerQuantizationFollowsMergedAndSplitLayers() throws {
        let model = try referenceModel()
        let tables = BaseConfiguration.Quantization(groupSize: 32, bits: 4)
        let rest = BaseConfiguration.Quantization(groupSize: 32, bits: 8)
        quantize(model: model) { path, module in
            guard module is Quantizable,
                let weight = module.parameters().flattened().first(where: { $0.0 == "weight" }),
                weight.1.dim(-1) % 32 == 0
            else { return nil }
            return (path.contains(".ngram_embedding.") ? tables : rest).asTuple
        }
        let weights = flattened(model)
        let checkpoint = try checkpointLayout(
            weights, lowrank: model.configuration.hcLowrank, prefix: "language_model.model.",
            raw: true, shard: "shards.")
        var overrides = [String: BaseConfiguration.QuantizationOption]()
        for key in checkpoint.keys
        where key.contains(".ngram_embedding.shards.") && key.hasSuffix(".scales") {
            overrides[String(key.dropLast(".scales".count))] = .quantize(tables)
        }
        let settings = BaseConfiguration.PerLayerQuantization(
            quantization: rest, perLayerQuantization: overrides)

        let loaded = Qwen4ExpModel(try configuration())
        try applyCheckpoint(
            try loaded.prepareCheckpoint(
                ModelCheckpoint(weights: checkpoint, perLayerQuantization: settings)),
            to: loaded)
        assertSame(flattened(loaded), weights, "per-layer quantization")

        let inject = try XCTUnwrap(
            checkpoint.keys.first { $0.hasSuffix("block_inject_weight.scales") })
        var conflicting = settings
        conflicting.perLayerQuantization[String(inject.dropLast(".scales".count))] =
            .quantize(tables)
        XCTAssertThrowsError(
            try Qwen4ExpModel(try configuration()).prepareCheckpoint(
                ModelCheckpoint(weights: checkpoint, perLayerQuantization: conflicting)))
    }

    /// The bf16 HF checkpoint also stores torch convolutions and fused experts, and
    /// carries MTP and vision tensors the text model drops.
    func testSanitizeLoadsTransformersLayout() throws {
        let model = try referenceModel()
        let weights = flattened(model)
        let lowrank = model.configuration.hcLowrank

        var checkpoint = [
            "mtp.fc.weight": MLXArray.zeros([2]),
            "model.visual.blocks.0.weight": MLXArray.zeros([2]),
        ]
        for (name, value) in weights {
            let key = name.hasPrefix("model.") ? "model.language_model." + name.dropFirst(6) : name
            if name.hasSuffix("switch_mlp.up_proj.weight")
                || name.hasSuffix("switch_mlp.down_proj.weight")
            {
                continue
            }
            if name.hasSuffix("switch_mlp.gate_proj.weight") {
                let base = name.dropLast("switch_mlp.gate_proj.weight".count)
                let up = try XCTUnwrap(weights["\(base)switch_mlp.up_proj.weight"])
                let down = try XCTUnwrap(weights["\(base)switch_mlp.down_proj.weight"])
                let target = key.dropLast("switch_mlp.gate_proj.weight".count)
                checkpoint["\(target)experts.gate_up_proj"] = concatenated([value, up], axis: -2)
                checkpoint["\(target)experts.down_proj"] = down
                continue
            }
            var value = value
            if Self.foldOne.contains(where: { key.hasSuffix($0) }) {
                value = value - 1
            }
            if key.hasSuffix("conv1d.weight") {
                value = value.transposed(0, 2, 1)
            }
            if key.hasSuffix("input_mix_weight_down.weight") {
                let inject = key.replacingOccurrences(
                    of: "input_mix_weight_down", with: "block_inject_weight")
                checkpoint[key] = value[..<lowrank]
                checkpoint[inject] = value[lowrank...]
                continue
            }
            checkpoint[key] = value
        }
        for layer in [0, 1] {
            let tables = "model.language_model.layers.\(layer).ple.ple_embedding.ngram_embedding."
            let table = concatenated(
                try (0 ..< 4).map {
                    try XCTUnwrap(checkpoint.removeValue(forKey: "\(tables)\($0).weight"))
                })
            let rows = (table.dim(0) + 1) / 2
            let padded = concatenated([
                table, MLXArray.zeros([2 * rows - table.dim(0), table.dim(1)], dtype: table.dtype),
            ])
            for (i, part) in padded.split(parts: 2).enumerated() {
                checkpoint["\(tables)shard_\(i).weight"] = part
            }
        }
        assertSame(try model.sanitize(weights: checkpoint), weights, "transformers")
    }

    // MARK: - Caches

    func testCachePerLayerType() throws {
        let model = try referenceModel()
        let cache = try model.newCache(parameters: nil)
        XCTAssertEqual(cache.count, 4)
        XCTAssertEqual(cache.prefix(3).map { $0 is ArraysCache }, [true, true, true])
        let list = try XCTUnwrap(cache[3] as? CacheList)
        XCTAssertTrue(list[0] is KVCacheSimple)
        XCTAssertTrue(list[1] is any AuxiliaryKVCache)

        _ = model(referenceTokens[0..., ..<10], cache: cache)
        eval(cache)
        // A PLE layer adds its conv state and n-gram context.
        XCTAssertEqual(cache.prefix(3).map { $0.innerState().count }, [4, 4, 2])
        XCTAssertEqual(list[0].offset, 10)
        XCTAssertEqual(list[1].offset, 10)
    }

    func testBoundedCacheIsRefused() throws {
        let model = try referenceModel()
        XCTAssertThrowsError(try model.newCache(parameters: GenerateParameters(maxKVSize: 64)))
    }

    /// KV quantization rewrites the attention keys and values, never the indexer keys.
    func testKVQuantizationLeavesIndexerKeys() throws {
        let model = try referenceModel()
        let tokens = referenceTokens
        var cache = try model.newCache(parameters: nil)
        _ = model(tokens[0..., ..<15], cache: cache)
        maybeQuantizeKVCache(cache: &cache, kvBits: 8, kvGroupSize: 32, quantizedKVStart: 0)

        let list = try XCTUnwrap(cache[3] as? CacheList)
        XCTAssertTrue(list[0] is QuantizedKVCache)
        XCTAssertTrue(list[1] is any AuxiliaryKVCache)
        let next = model(tokens[0..., 15 ..< 16], cache: cache)
        eval(next)
        XCTAssertEqual(list[1].offset, 16)
        XCTAssertTrue(all(isFinite(next)).item(Bool.self))
    }

    func testPromptCacheRoundTrip() throws {
        let model = try referenceModel()
        let tokens = referenceTokens
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("qwen4-exp-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        // A fresh cache has layers that were never updated.
        let fresh = directory.appendingPathComponent("fresh.safetensors")
        try savePromptCache(url: fresh, cache: try model.newCache(parameters: nil))
        let (restoredFresh, _) = try loadPromptCache(url: fresh)
        XCTAssertEqual(restoredFresh.map(\.offset), [0, 0, 0, 0])

        let cache = try model.newCache(parameters: nil)
        _ = model(tokens[0..., ..<15], cache: cache)
        let url = directory.appendingPathComponent("warm.safetensors")
        try savePromptCache(url: url, cache: cache)
        let (restored, _) = try loadPromptCache(url: url)

        let next = tokens[0..., 15 ..< 16]
        let expected = model(next, cache: cache)
        let actual = model(next, cache: restored)
        XCTAssertTrue(arrayEqual(actual, expected).item(Bool.self))
        XCTAssertTrue((restored[3] as? CacheList)?[1] is any AuxiliaryKVCache)
    }

    // MARK: - Consistency

    func testChunkedPrefillMatchesOneShot() throws {
        let model = try referenceModel()
        let tokens = referenceTokens
        let full = model(tokens, cache: nil)

        let cache = try model.newCache(parameters: nil)
        var chunks: [MLXArray] = []
        var start = 0
        for size in [3, 5, 9, 1, 4] {
            chunks.append(model(tokens[0..., start ..< (start + size)], cache: cache))
            start += size
        }
        XCTAssertEqual(start, tokens.dim(1))
        assertClose(
            concatenated(chunks, axis: 1), full, atol: 1e-4,
            reducedAtol: Double(MatmulPrecision.splitTolerance))
    }

    /// Right-padded rows (`withPreparedCache`) match each row alone, including the
    /// deltanet, PLE conv and n-gram state they leave for the next step.
    func testRightPaddedRowsMatchEachRowAlone() throws {
        let model = try referenceModel()
        let rows = [
            Qwen4ExpReferenceValues.tokens[0], Array(Qwen4ExpReferenceValues.tokens[1].prefix(7)),
        ]
        let width = rows.map(\.count).max()!
        let padded = rows.flatMap { $0 + Array(repeating: 0, count: width - $0.count) }

        let cache = try model.newCache(parameters: nil)
        let batched = withPreparedCache(cache, lengths: rows.map(\.count)) {
            model(MLXArray(padded).reshaped(rows.count, width), cache: cache)
        }

        for (row, tokens) in rows.enumerated() {
            let alone = try model.newCache(parameters: nil)
            let expected = model(MLXArray(tokens).reshaped(1, -1), cache: alone)
            assertClose(
                batched[row ..< (row + 1), ..<tokens.count], expected, atol: 1e-4,
                reducedAtol: Double(MatmulPrecision.splitTolerance), "logits, row \(row)")
            for layer in 0 ..< 3 {
                let batchedState = cache[layer].innerState()
                let aloneState = alone[layer].innerState()
                XCTAssertEqual(batchedState.count, aloneState.count)
                for (slot, (b, a)) in zip(batchedState, aloneState).enumerated() {
                    let b = b[row ..< (row + 1)]
                    if a.dtype.isFloatingPoint {
                        assertClose(
                            b, a, atol: 1e-4,
                            reducedAtol: Double(MatmulPrecision.splitTolerance),
                            "layer \(layer), slot \(slot), row \(row)")
                    } else {
                        XCTAssertTrue(
                            arrayEqual(b, a).item(Bool.self), "layer \(layer), n-gram, row \(row)")
                    }
                }
            }
        }
    }
}
