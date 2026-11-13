import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXLLM

final class GLM5NextTests: XCTestCase {

    // MARK: - Fixtures

    /// Six layers: three dense KDA, then DSA (own indexer) + MoE, KDA + MoE,
    /// and DSA with a shared indexer + MoE.
    private func textConfigJSON(indexTopk: Int = 8, indexHeads: Int = 2) -> String {
        """
        {
            "model_type": "glm5_next_text", "vocab_size": 64, "hidden_size": 64,
            "intermediate_size": 96, "moe_intermediate_size": 32,
            "num_hidden_layers": 6, "num_nextn_predict_layers": 1,
            "num_attention_heads": 2, "num_key_value_heads": 2,
            "n_shared_experts": 1, "n_routed_experts": 4, "num_experts_per_tok": 2,
            "routed_scaling_factor": 2.5, "n_group": 1, "topk_group": 1,
            "norm_topk_prob": true, "scoring_func": "sigmoid", "topk_method": "noaux_tc",
            "kv_lora_rank": 32, "q_lora_rank": 32, "qk_rope_head_dim": 0,
            "qk_nope_head_dim": 32, "v_head_dim": 32, "mla_use_nope": true,
            "rms_norm_eps": 1e-5, "tie_word_embeddings": false, "attention_bias": false,
            "index_topk": \(indexTopk), "index_head_dim": 16, "index_n_heads": \(indexHeads),
            "index_kpool": 4, "index_kpool_always_select_tail": true,
            "layer_types": ["linear_attention", "linear_attention", "linear_attention",
                "deepseek_sparse_attention", "linear_attention", "deepseek_sparse_attention"],
            "mlp_layer_types": ["dense", "dense", "dense", "sparse", "sparse", "sparse"],
            "indexer_types": ["full", "full", "full", "full", "full", "shared"],
            "swiglu_limit": 10.0,
            "linear_attn_config": {"num_heads": 2, "head_dim": 32,
                "short_conv_kernel_size": 4, "gate_lower_bound": -5.0},
            "hc_mult": 4, "hc_eps": 1e-6, "hc_sinkhorn_iters": 20,
            "first_k_dense_replace": 3
        }
        """
    }

    private func configuration(indexTopk: Int = 8, indexHeads: Int = 2, nested: Bool = false)
        throws -> GLM5NextConfiguration
    {
        let text = textConfigJSON(indexTopk: indexTopk, indexHeads: indexHeads)
        let json = nested ? #"{"model_type": "glm5_next", "text_config": \#(text)}"# : text
        let config = try JSONDecoder().decode(GLM5NextConfiguration.self, from: Data(json.utf8))
        try config.validateModelConfiguration()
        return config
    }

    /// Replace every parameter with seeded noise so routing and top-k selection have no ties.
    private func randomize(_ model: Module, seed: UInt64 = 7) {
        let parameters = model.parameters().flattened()
        let updated: [(String, MLXArray)] = withRandomState(MLXRandom.RandomState(seed: seed)) {
            parameters.map { key, value in
                let noise = MLXRandom.normal(value.shape).asType(value.dtype)
                if key.hasSuffix("norm.weight") || key.hasSuffix("layernorm.weight") {
                    return (key, 1 + 0.1 * noise)
                }
                return (key, 0.15 * noise)
            }
        }
        model.update(parameters: ModuleParameters.unflattened(updated))
        eval(model)
    }

    private func tokens(_ count: Int, seed: UInt64 = 3) -> MLXArray {
        withRandomState(MLXRandom.RandomState(seed: seed)) {
            MLXRandom.randInt(0 ..< 64, [1, count]).asType(.int32)
        }
    }

    /// Run `body` on the CPU with compilation off. MLX's compile cache matches
    /// entries on its global default stream, not on `withDefaultDevice`'s
    /// task-local device, so a function compiled by an earlier GPU test would
    /// replay its GPU trace here.
    private func onCPU<R>(_ body: () throws -> R) rethrows -> R {
        compile(enable: false)
        defer { compile(enable: true) }
        return try Device.withDefaultDevice(.cpu) { try body() }
    }

    private func maxAbsDifference(_ a: MLXArray, _ b: MLXArray) -> Float {
        abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
    }

    // MARK: - Shapes and configuration

    func testForwardShapesWithAndWithoutCache() throws {
        for nested in [false, true] {
            let config = try configuration(nested: nested)
            let model: any LanguageModel =
                nested ? GLM5NextModel(config) : GLM5NextTextModel(config)
            randomize(model)
            let input = tokens(13)

            let uncached = model(input, cache: nil)
            XCTAssertEqual(uncached.shape, [1, 13, 64])

            let cache = try model.newCache(parameters: nil)
            XCTAssertEqual(cache.count, 6)
            XCTAssertTrue(cache[0] is MambaCache)
            XCTAssertTrue(cache[3] is GLM5NextAttentionCache)
            let prefill = model(input, cache: cache)
            let step = model(MLXArray([Int32(5)]).reshaped(1, 1), cache: cache)
            eval(prefill, step)
            XCTAssertEqual(prefill.shape, [1, 13, 64])
            XCTAssertEqual(step.shape, [1, 1, 64])
            XCTAssertEqual(cache[3].offset, 14)
            XCTAssertLessThan(maxAbsDifference(uncached, prefill), 1e-4)
        }
    }

    func testConfigurationRejectsInvalidValues() throws {
        func rejects(_ from: String, _ to: String, line: UInt = #line) {
            let json = textConfigJSON()
            XCTAssertTrue(json.contains(from), "fixture lacks \(from)", line: line)
            XCTAssertThrowsError(
                try JSONDecoder().decode(
                    GLM5NextConfiguration.self,
                    from: Data(json.replacingOccurrences(of: from, with: to).utf8)
                ).validateModelConfiguration(), to, line: line)
        }
        XCTAssertNoThrow(try configuration())
        rejects(#""index_topk": 8"#, #""index_topk": 0"#)
        rejects(#""index_topk": 8"#, #""index_topk": 2"#)
        rejects(#""index_topk": 8"#, #""index_topk": 10"#)
        rejects(#""index_kpool": 4"#, #""index_kpool": 0"#)
        rejects(#""qk_rope_head_dim": 0"#, #""qk_rope_head_dim": 64"#)
        rejects(#""num_key_value_heads": 2"#, #""num_key_value_heads": 1"#)
        rejects(#""mlp_layer_types": ["dense""#, #""mlp_layer_types": ["moe""#)
        rejects(#""full", "full", "shared"]"#, #""full", "full", "Shared"]"#)
        // The first sparse-attention layer (3) cannot share.
        rejects(#""full", "full", "full", "shared"]"#, #""full", "shared", "full", "shared"]"#)
        rejects(#""num_experts_per_tok": 2"#, #""num_experts_per_tok": 5"#)
    }

    func testConfigurationDerivesLayerPatternFromLinearAttentionConfig() throws {
        let kda = (0 ..< 45).filter { $0 % 4 != 3 }
        let json = """
            {
                "model_type": "glm5_next",
                "text_config": {
                    "model_type": "glm5_next_text", "num_hidden_layers": 45,
                    "index_kpool": 4, "index_topk": 2048, "first_k_dense_replace": 3,
                    "linear_attn_config": {"num_heads": 64, "head_dim": 128,
                        "short_conv_kernel_size": 4, "gate_lower_bound": -5.0,
                        "kda_layers": \(kda)}
                }
            }
            """
        let config = try JSONDecoder().decode(GLM5NextConfiguration.self, from: Data(json.utf8))
        try config.validateModelConfiguration()
        XCTAssertEqual(config.layerTypes.filter { $0 == "linear_attention" }.count, 34)
        XCTAssertEqual(
            config.layerTypes.indices.filter { config.layerTypes[$0] != "linear_attention" },
            (0 ..< 45).filter { $0 % 4 == 3 })
        XCTAssertEqual(config.mlpLayerTypes.prefix(4), ["dense", "dense", "dense", "sparse"])
        XCTAssertTrue(config.indexerTypes.allSatisfy { $0 == "full" })
        XCTAssertEqual(config.linearHeadDim, 128)
        XCTAssertEqual(config.linearLowerBound, -5.0)
    }

    /// transformers' configuration names the sparse layers `indexed_attention`,
    /// and `save_pretrained` writes that name.
    func testConfigurationAcceptsTransformersLayerTypeName() throws {
        let json = textConfigJSON().replacingOccurrences(
            of: "deepseek_sparse_attention", with: "indexed_attention")
        let config = try JSONDecoder().decode(GLM5NextConfiguration.self, from: Data(json.utf8))
        try config.validateModelConfiguration()
        XCTAssertEqual(config.layerTypes, try configuration().layerTypes)
        let model = GLM5NextTextModel(config)
        XCTAssertTrue(model.model.layers[3].selfAttn is GLM5NextAttention)
        XCTAssertTrue(model.model.layers[4].selfAttn is GLM5NextLinearAttention)
    }

    // MARK: - Cache consistency

    /// Token-by-token decode reproduces the full-prefill logits while the
    /// indexer keeps a strict subset of the pools. Eight indexer heads and this
    /// seed keep every top-k boundary clear of ties (with two heads, scores the
    /// ReLU clamps to exactly 0 often tie at the boundary). It runs on the CPU:
    /// on an M5 GPU a multi-row float32 matmul differs from the one-row path by
    /// about 1e-3 relative, enough to flip a close selection.
    func testIncrementalDecodeMatchesPrefill() throws {
        try onCPU {
            let model = GLM5NextTextModel(try configuration(indexTopk: 8, indexHeads: 8))
            randomize(model)
            let input = tokens(40)
            let inner = model.model

            let prefill = model(input, cache: try model.newCache(parameters: nil))
            eval(prefill)

            // The model's forward, one layer at a time, to read the selection
            // of the first sparse-attention layer (3) at each step.
            let cache = try model.newCache(parameters: nil)
            var maxDifference: Float = 0
            var strictSubsetSteps = 0
            for t in 0 ..< 40 {
                var h = repeated(
                    expandedDimensions(inner.embedTokens(input[0..., t ..< (t + 1)]), axis: 2),
                    count: inner.hcMult, axis: 2)
                var topk: MLXArray?
                var selection: MLXArray?
                for (index, layer) in inner.layers.enumerated() {
                    (h, topk) = layer(h, cache: cache[index], previousTopk: topk)
                    if index == 3 { selection = topk }
                }
                let step = model.lmHead!(inner.norm(h.mean(axis: 2)))
                eval(step)
                maxDifference = max(maxDifference, maxAbsDifference(step[0, 0], prefill[0, t]))

                // Selected pool tokens (the first index_topk entries; the open tail follows).
                if let selection {
                    let picked = (selection[0, 0, ..<8] .>= 0).sum().item(Int.self)
                    if picked < (t + 1) / 4 * 4 { strictSubsetSteps += 1 }
                }
            }
            XCTAssertLessThan(maxDifference, 1e-3, "decode diverged from prefill")
            // Every step with more than two complete pools (t >= 11) drops some.
            XCTAssertEqual(strictSubsetSteps, (0 ..< 40).filter { ($0 + 1) / 4 > 2 }.count)
        }
    }

    /// Prefill in chunks whose boundaries split pool windows matches one-shot
    /// prefill, and decode continues the same from either.
    func testChunkedPrefillMatchesSinglePrefill() throws {
        let model = GLM5NextTextModel(try configuration(indexTopk: 8))
        randomize(model)
        let input = tokens(30)
        let next = MLXArray([Int32(11)]).reshaped(1, 1)

        let single = try model.newCache(parameters: nil)
        let full = model(input, cache: single)
        let fullNext = model(next, cache: single)
        eval(full, fullNext)

        let chunked = try model.newCache(parameters: nil)
        var pieces = [MLXArray]()
        for (start, end) in [(0, 9), (9, 18), (18, 30)] {
            pieces.append(model(input[0..., start ..< end], cache: chunked))
        }
        let chunkedNext = model(next, cache: chunked)
        eval(pieces, chunkedNext)

        XCTAssertLessThan(maxAbsDifference(concatenated(pieces, axis: 1), full), 1e-3)
        XCTAssertLessThan(maxAbsDifference(chunkedNext, fullNext), 1e-3)
    }

    /// A copied cache continues exactly like the original.
    func testCacheCopyContinuesLikeOriginal() throws {
        let model = GLM5NextTextModel(try configuration(indexTopk: 8))
        randomize(model)
        let cache = try model.newCache(parameters: nil)
        eval(model(tokens(22), cache: cache))
        let copy = cache.map { $0.copy() }

        let next = tokens(3, seed: 5)
        let a = model(next, cache: cache)
        let b = model(next, cache: copy)
        eval(a, b)
        XCTAssertEqual(maxAbsDifference(a, b), 0)
    }

    /// Past `index_topk` the indexer drops tokens: the same weights with a
    /// larger `index_topk` (every token selected) give different logits.
    func testIndexerSelectionIsActivePastTopk() throws {
        let sparse = GLM5NextTextModel(try configuration(indexTopk: 8))
        randomize(sparse)
        let dense = GLM5NextTextModel(try configuration(indexTopk: 64))
        try dense.update(parameters: sparse.parameters(), verify: [.all])
        let input = tokens(40)

        let sparseLogits = sparse(input, cache: try sparse.newCache(parameters: nil))
        let denseLogits = dense(input, cache: try dense.newCache(parameters: nil))
        eval(sparseLogits, denseLogits)
        // Up to index_topk + kpool - 1 tokens every query sees its whole prefix.
        XCTAssertLessThan(
            maxAbsDifference(sparseLogits[0..., ..<11], denseLogits[0..., ..<11]), 1e-4)
        XCTAssertGreaterThan(maxAbsDifference(sparseLogits, denseLogits), 1e-2)
    }

    /// Greedy generation through `TokenIterator` (chunked prefill, cached
    /// decode) matches greedy decoding with a full uncached forward per token.
    func testGreedyGenerationMatchesUncachedForward() throws {
        let model = GLM5NextTextModel(try configuration(indexTopk: 8))
        randomize(model)
        let prompt = tokens(30).reshaped(-1)

        var iterator = try TokenIterator(
            input: LMInput(text: .init(tokens: prompt)), model: model,
            parameters: GenerateParameters(
                maxTokens: 8, temperature: 0, prefill: .init(stepSize: 7)))
        var generated = [Int]()
        while let token = iterator.next() {
            generated.append(token)
        }

        var sequence = prompt.asArray(Int32.self)
        var expected = [Int]()
        for _ in 0 ..< 8 {
            let logits = model(MLXArray(sequence).reshaped(1, -1), cache: nil)
            let token = argMax(logits[0, -1], axis: -1).item(Int.self)
            expected.append(token)
            sequence.append(Int32(token))
        }
        XCTAssertEqual(generated, expected)
    }

    func testVectorizedGatedDeltaKernelMatchesOps() throws {
        let (B, T, H, Dk, Dv) = (1, 12, 2, 32, 32)
        let (q, k, v, g, beta, state) = withRandomState(MLXRandom.RandomState(seed: 5)) {
            (
                MLXRandom.normal([B, T, H, Dk]) * 0.2,
                MLXRandom.normal([B, T, H, Dk]) * 0.2,
                MLXRandom.normal([B, T, H, Dv]),
                exp(-MLXRandom.uniform(low: 0, high: 1, [B, T, H, Dk])),
                sigmoid(MLXRandom.normal([B, T, H])),
                MLXRandom.normal([B, H, Dv, Dk]) * 0.1
            )
        }
        let (yKernel, sKernel) = gatedDeltaUpdateVectorized(
            q: q, k: k, v: v, g: g, beta: beta, state: state, useKernel: true)
        let (yOps, sOps) = gatedDeltaUpdateVectorized(
            q: q, k: k, v: v, g: g, beta: beta, state: state, useKernel: false)
        eval(yKernel, sKernel, yOps, sOps)
        XCTAssertLessThan(maxAbsDifference(yKernel, yOps), 1e-4)
        XCTAssertLessThan(maxAbsDifference(sKernel, sOps), 1e-4)
    }

    // MARK: - Weights

    /// Parameter names of one KDA layer and one DSA + MoE layer of the
    /// GLM-5.3-Flash MLX checkpoint, under `language_model.model.layers.<i>.`.
    private static let checkpointLayerKeys: [Int: [String]] = [
        0: [
            "attn_hc.base", "attn_hc.fn", "attn_hc.scale", "ffn_hc.base", "ffn_hc.fn",
            "ffn_hc.scale", "input_layernorm.weight", "mlp.down_proj.weight",
            "mlp.gate_proj.weight", "mlp.up_proj.weight", "post_attention_layernorm.weight",
            "self_attn.b_proj.weight", "self_attn.conv1d.weight", "self_attn.forget_gate.A_log",
            "self_attn.forget_gate.dt_bias", "self_attn.forget_gate.f_a_proj.weight",
            "self_attn.forget_gate.f_b_proj.weight", "self_attn.g_a_proj.weight",
            "self_attn.g_b_proj.weight", "self_attn.k_proj.weight", "self_attn.o_norm.weight",
            "self_attn.o_proj.weight", "self_attn.q_proj.weight", "self_attn.v_proj.weight",
        ],
        3: [
            "attn_hc.base", "attn_hc.fn", "attn_hc.scale", "ffn_hc.base", "ffn_hc.fn",
            "ffn_hc.scale", "input_layernorm.weight", "mlp.gate.e_score_correction_bias",
            "mlp.gate.weight", "mlp.shared_experts.down_proj.weight",
            "mlp.shared_experts.gate_proj.weight", "mlp.shared_experts.up_proj.weight",
            "mlp.switch_mlp.down_proj.weight", "mlp.switch_mlp.gate_proj.weight",
            "mlp.switch_mlp.up_proj.weight", "post_attention_layernorm.weight",
            "self_attn.embed_q.weight", "self_attn.indexer.index_kpool_compress_ape",
            "self_attn.indexer.index_kpool_compress_gate", "self_attn.indexer.k_norm.bias",
            "self_attn.indexer.k_norm.weight", "self_attn.indexer.weights_proj.weight",
            "self_attn.indexer.wk.weight", "self_attn.indexer.wq_b.weight",
            "self_attn.kv_a_layernorm.weight", "self_attn.kv_a_proj_with_mqa.weight",
            "self_attn.o_proj.weight", "self_attn.q_a_layernorm.weight",
            "self_attn.q_a_proj.weight", "self_attn.q_b_proj.weight",
            "self_attn.unembed_out.weight",
        ],
    ]

    func testModuleNamesMatchCheckpointTensorNames() throws {
        let model = GLM5NextModel(try configuration(nested: true))
        let keys = Set(model.parameters().flattened().map(\.0))
        for (layer, expected) in Self.checkpointLayerKeys {
            let prefix = "language_model.model.layers.\(layer)."
            let actual = Set(
                keys.filter { $0.hasPrefix(prefix) }.map { String($0.dropFirst(prefix.count)) })
            XCTAssertEqual(actual, Set(expected), "layer \(layer)")
        }
        for global in [
            "language_model.lm_head.weight", "language_model.model.embed_tokens.weight",
            "language_model.model.norm.weight",
        ] {
            XCTAssertTrue(keys.contains(global), global)
        }
        XCTAssertFalse(keys.contains { $0.contains("mtp") })
    }

    /// The MLX checkpoint layout loads unchanged; vision tensors and the
    /// multi-token-prediction layer are dropped.
    func testSanitizeKeepsCheckpointLayoutAndDropsVisionAndMTP() throws {
        let source = GLM5NextModel(try configuration(nested: true))
        randomize(source)
        var weights = Dictionary(uniqueKeysWithValues: source.parameters().flattened())
        weights["vision_model.blocks.0.attn.qkv.weight"] = MLXArray.zeros([4, 4])
        weights["vision_tower.merger.proj.weight"] = MLXArray.zeros([4, 4])
        weights["language_model.mtp.0.eh_proj.weight"] = MLXArray.zeros([4, 4])
        weights["language_model.model.layers.6.enorm.weight"] = MLXArray.zeros([4])
        weights["language_model.model.layers.6.mlp.gate.weight"] = MLXArray.zeros([4, 4])

        let target = GLM5NextModel(try configuration(nested: true))
        let sanitized = target.sanitize(weights: weights)
        XCTAssertFalse(sanitized.keys.contains { $0.hasPrefix("vision") })
        XCTAssertFalse(sanitized.keys.contains { $0.contains("mtp") || $0.contains("layers.6.") })
        try target.update(parameters: ModuleParameters.unflattened(sanitized), verify: [.all])

        let input = tokens(10)
        XCTAssertEqual(maxAbsDifference(source(input, cache: nil), target(input, cache: nil)), 0)
    }

    /// Hugging Face layout: `model.language_model.*`, `hc_attn_*`, top-level
    /// forget-gate tensors, per-projection convs in torch layout, per-expert
    /// weights, `kv_b_proj`, and the nextn layer at `layers.<num_hidden_layers>`.
    func testSanitizeMapsHuggingFaceLayout() throws {
        let config = try configuration(nested: true)
        let source = GLM5NextModel(config)
        randomize(source)
        let reference = Dictionary(uniqueKeysWithValues: source.parameters().flattened())

        var hf = [String: MLXArray]()
        var kvB = [String: (wk: MLXArray?, wv: MLXArray?)]()
        for (name, value) in reference {
            let text = name.dropFirst("language_model.".count)
            let key =
                (text.hasPrefix("model.")
                ? "model.language_model." + text.dropFirst("model.".count) : String(text))
                .replacingOccurrences(of: ".attn_hc.", with: ".hc_attn_")
                .replacingOccurrences(of: ".ffn_hc.", with: ".hc_ffn_")
                .replacingOccurrences(of: ".forget_gate.", with: ".")

            if key.hasSuffix(".self_attn.conv1d.weight") {
                let prefix = String(key.dropLast("conv1d.weight".count))
                let torchLayout = value.movedAxis(source: 1, destination: 2)
                let parts = split(torchLayout, parts: 3, axis: 0)
                for (name, part) in zip(["q", "k", "v"], parts) {
                    hf[prefix + "\(name)_conv1d.weight"] = part
                }
                continue
            }
            if let range = key.range(of: ".mlp.switch_mlp.") {
                for expert in 0 ..< value.dim(0) {
                    let name =
                        key[..<range.lowerBound] + ".mlp.experts.\(expert)."
                        + key[range.upperBound...]
                    hf[String(name)] = value[expert]
                }
                continue
            }
            if key.hasSuffix(".self_attn.embed_q.weight") {
                kvB[String(key.dropLast("embed_q.weight".count)), default: (nil, nil)].wk = value
                continue
            }
            if key.hasSuffix(".self_attn.unembed_out.weight") {
                kvB[String(key.dropLast("unembed_out.weight".count)), default: (nil, nil)].wv =
                    value
                continue
            }
            hf[key] = value
        }
        for (prefix, pair) in kvB {
            let combined = concatenated([pair.wk!.swappedAxes(-1, -2), pair.wv!], axis: 1)
            hf[prefix + "kv_b_proj.weight"] = combined.reshaped(-1, combined.dim(-1))
        }
        let nextn = "model.language_model.layers.\(config.hiddenLayers)."
        hf[nextn + "eh_proj.weight"] = MLXArray.zeros([4, 4])
        hf[nextn + "shared_head.head.weight"] = MLXArray.zeros([4, 4])
        hf[nextn + "mlp.experts.0.up_proj.weight"] = MLXArray.zeros([4, 4])
        hf["model.visual.patch_embed.proj.weight"] = MLXArray.zeros([4, 4])

        let target = GLM5NextModel(config)
        let sanitized = target.sanitize(weights: hf)
        XCTAssertEqual(Set(sanitized.keys), Set(reference.keys))
        for (key, value) in reference {
            XCTAssertLessThan(maxAbsDifference(sanitized[key]!, value), 1e-6, key)
        }
        try target.update(parameters: ModuleParameters.unflattened(sanitized), verify: [.all])
    }

    /// Mixed-precision checkpoint through `loadWeights`: per-layer bit widths
    /// keyed by `language_model.*` paths, as in the checkpoint's `quantization`.
    /// Vision and MTP tensors in the file are not read.
    func testLoadWeightsWithPerLayerQuantization() throws {
        let config = try configuration(nested: true)
        let source = GLM5NextModel(config)
        randomize(source)

        func bits(_ path: String) -> Int {
            if path.hasSuffix("embed_q") || path.hasSuffix("b_proj") { return 5 }
            if path.contains("indexer") || path.hasSuffix("lm_head") { return 8 }
            return 4
        }
        quantize(
            model: source,
            filter: {
                (path: String, module: Module) -> (
                    groupSize: Int, bits: Int, mode: QuantizationMode
                )? in
                module is Quantizable ? (32, bits(path), .affine) : nil
            })
        eval(source)

        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("glm5next-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        var weights = Dictionary(uniqueKeysWithValues: source.parameters().flattened())
        weights["vision_tower.post_layernorm.weight"] = MLXArray.ones([8])
        weights["language_model.mtp.0.enorm.weight"] = MLXArray.ones([8])
        weights["language_model.model.layers.6.hnorm.weight"] = MLXArray.ones([8])
        try save(arrays: weights, url: directory.appendingPathComponent("model.safetensors"))

        var perLayer = [String: BaseConfiguration.QuantizationOption]()
        for (path, _) in source.leafModules().flattened() where bits(path) != 4 {
            perLayer[path] = .quantize(.init(groupSize: 32, bits: bits(path)))
        }
        let target = GLM5NextModel(config)
        try loadWeights(
            modelDirectory: directory, model: target,
            perLayerQuantization: .init(
                quantization: .init(groupSize: 32, bits: 4), perLayerQuantization: perLayer))

        let attention = target.languageModel.model.layers[3].selfAttn as! GLM5NextAttention
        XCTAssertEqual((attention.embedQ as? QuantizedMultiLinear)?.bits, 5)

        let input = tokens(12)
        XCTAssertEqual(maxAbsDifference(source(input, cache: nil), target(input, cache: nil)), 0)
    }

    // MARK: - Reference parity

    /// `Resources/glm5_next_tiny`, from `tools/glm5_next_parity/make_tiny_fixture.py`: a
    /// 5-layer checkpoint saved by transformers' own `glm5_next` model, with that model's
    /// float32 CPU log-probabilities for several prefill plans, teacher-forced on its greedy
    /// tokens.
    private struct TinyOracle: Decodable {
        struct Case: Decodable {
            let name: String
            /// Ends of the prefill chunks; one-token decode steps follow.
            let prefillEnds: [Int]
            /// Log-softmax at every position of `prompt + greedy.dropLast()`.
            let logprobs: [[Float]]

            enum CodingKeys: String, CodingKey {
                case name, logprobs
                case prefillEnds = "prefill_ends"
            }
        }
        let tolerance: Float
        let prompt: [Int]
        let greedy: [Int]
        let cases: [Case]
    }

    private var tinyFixture: URL {
        get throws {
            try XCTUnwrap(Bundle.module.url(forResource: "glm5_next_tiny", withExtension: nil))
        }
    }

    /// Through the model registry and `loadWeights`, so configuration decoding,
    /// validation and `sanitize` all run on the checkpoint as transformers saved it.
    private func loadTiny() async throws -> any LanguageModel {
        let data = try Data(contentsOf: tinyFixture.appendingPathComponent("config.json"))
        let base = try JSONDecoder().decode(BaseConfiguration.self, from: data)
        XCTAssertEqual(base.modelType, "glm5_next")
        let model = try await LLMTypeRegistry.shared.createModel(
            configuration: data, modelType: base.modelType)
        try await loadWeights(
            modelDirectory: tinyFixture, model: model,
            perLayerQuantization: base.perLayerQuantization)
        return model
    }

    /// Prefill `tokens` in chunks ending at `prefillEnds`, then one token per step;
    /// returns the float32 log-softmax at every position.
    private func teacherForced(
        _ model: any LanguageModel, tokens: [Int], prefillEnds: [Int]
    ) throws -> [[Float]] {
        let cache = try model.newCache(parameters: nil)
        let ends =
            prefillEnds + Array(stride(from: prefillEnds.last! + 1, through: tokens.count, by: 1))
        var rows = [[Float]]()
        var start = 0
        for end in ends {
            let chunk = MLXArray(tokens[start ..< end].map(Int32.init)).reshaped(1, end - start)
            let logits = model(chunk, cache: cache)[0].asType(.float32)
            let logprobs = logits - logSumExp(logits, axis: -1, keepDims: true)
            eval(logprobs)
            for row in 0 ..< (end - start) {
                rows.append(logprobs[row].asArray(Float.self))
            }
            start = end
        }
        return rows
    }

    func testTinyCheckpointMatchesTransformersReference() async throws {
        let oracle = try JSONDecoder().decode(
            TinyOracle.self,
            from: Data(contentsOf: tinyFixture.appendingPathComponent("oracle.json")))
        XCTAssertEqual(
            oracle.cases.map(\.name),
            ["one_call", "chunks_5_13_22_32", "chunks_of_7", "token_by_token"])
        let model = try await loadTiny()
        let tokens = oracle.prompt + oracle.greedy.dropLast()

        try onCPU {
            var worst: Float = 0
            for item in oracle.cases {
                let rows = try teacherForced(model, tokens: tokens, prefillEnds: item.prefillEnds)
                XCTAssertEqual(rows.count, item.logprobs.count, item.name)
                var difference: Float = 0
                for (row, expected) in zip(rows, item.logprobs) {
                    XCTAssertEqual(row.count, expected.count, item.name)
                    for (a, b) in zip(row, expected) {
                        difference = max(difference, abs(a - b))
                    }
                }
                worst = max(worst, difference)
                print("glm5_next parity \(item.name): max |dlogprob| = \(difference)")
                XCTAssertLessThanOrEqual(difference, oracle.tolerance, item.name)

                // The fixture's argmax margins exceed its threshold, so greedy agrees exactly.
                let greedy = rows[(oracle.prompt.count - 1)...].map { row in
                    row.indices.max { row[$0] < row[$1] }!
                }
                XCTAssertEqual(greedy, oracle.greedy, item.name)
            }
            print("glm5_next parity max |dlogprob| = \(worst) tolerance = \(oracle.tolerance)")
        }
    }

    // MARK: - Local checkpoint

    /// Greedy parity with a reference run of the Python model (mlx-vlm
    /// `glm5_next`) on a local checkpoint. Skipped unless
    /// `MLX_GLM5NEXT_MODEL` names the checkpoint directory and
    /// `MLX_GLM5NEXT_ORACLES` lists reference JSON files (comma-separated),
    /// each `{"prompt": [token ids], "greedy": [token ids]}` from greedy
    /// decoding at batch 1. Prints decode tokens/s and peak memory.
    func testLocalCheckpointGreedyParity() async throws {
        let environment = ProcessInfo.processInfo.environment
        guard let path = environment["MLX_GLM5NEXT_MODEL"], !path.isEmpty,
            let oracles = environment["MLX_GLM5NEXT_ORACLES"], !oracles.isEmpty
        else {
            throw XCTSkip("set MLX_GLM5NEXT_MODEL and MLX_GLM5NEXT_ORACLES to run")
        }

        let directory = URL(fileURLWithPath: path)
        let data = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        let base = try JSONDecoder().decode(BaseConfiguration.self, from: data)
        let config = try JSONDecoder().decode(GLM5NextConfiguration.self, from: data)
        try config.validateModelConfiguration()
        let model: any LanguageModel =
            base.modelType == "glm5_next" ? GLM5NextModel(config) : GLM5NextTextModel(config)
        try await loadWeights(
            modelDirectory: directory, model: model,
            perLayerQuantization: base.perLayerQuantization)

        // Keep the weights wired, as a caller of `generate` does with a wired
        // memory ticket. Unwired, a checkpoint this size is paged on every
        // decode step and decodes at about 1% of the wired rate.
        let weightBytes = model.parameters().flattened().reduce(0) { $0 + $1.1.nbytes }
        let ticket = MLX.WiredSumPolicy().ticket(size: weightBytes, kind: .active)
        try await ticket.withWiredLimit {
            try runOracles(oracles, model: model)
        }
    }

    private func runOracles(_ oracles: String, model: any LanguageModel) throws {
        struct Oracle: Decodable {
            let prompt: [Int]
            let greedy: [Int]
        }
        for file in oracles.split(separator: ",") {
            let oracle = try JSONDecoder().decode(
                Oracle.self, from: Data(contentsOf: URL(fileURLWithPath: String(file))))
            Memory.clearCache()
            Memory.peakMemory = 0

            var iterator = try TokenIterator(
                input: LMInput(text: .init(tokens: MLXArray(oracle.prompt.map { Int32($0) }))),
                model: model,
                parameters: GenerateParameters(
                    maxTokens: oracle.greedy.count, temperature: 0,
                    prefill: .init(stepSize: 2048)))
            var generated = [Int]()
            var decodeStart = Date()
            while let token = iterator.next() {
                generated.append(token)
                if generated.count == 1 {
                    decodeStart = Date()
                }
            }
            let decodeSeconds = Date().timeIntervalSince(decodeStart)

            let matching = zip(generated, oracle.greedy).prefix { $0.0 == $0.1 }.count
            print(
                "glm5_next parity \(file): prompt=\(oracle.prompt.count)",
                "matching=\(matching)/\(oracle.greedy.count)",
                "prefill_s=\(String(format: "%.2f", iterator.promptPrefillTime))",
                "decode_tps=\(String(format: "%.2f", Double(generated.count - 1) / decodeSeconds))",
                "peak_gb=\(String(format: "%.1f", Double(Memory.peakMemory) / 1e9))")
            XCTAssertEqual(generated, oracle.greedy, "first divergence at token \(matching)")
        }
    }
}
