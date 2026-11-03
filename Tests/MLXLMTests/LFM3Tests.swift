// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN
import XCTest

@testable import MLXLLM
@testable import MLXLMCommon

final class LFM3Tests: XCTestCase {

    /// `config.json` of LiquidAI/LFM3-1.7B.
    private let realConfigJSON = """
        {
          "architectures": ["Lfm3ForCausalLM"],
          "bos_token_id": 124894,
          "conv_L_cache": 7,
          "conv_bias": false,
          "dtype": "bfloat16",
          "eos_token_id": 124900,
          "head_dim": 128,
          "hidden_size": 2048,
          "intermediate_size": 7680,
          "layer_types": [
            "conv", "conv", "full_attention", "conv", "sliding_attention", "conv",
            "full_attention", "conv", "sliding_attention", "conv", "full_attention", "conv",
            "sliding_attention", "conv", "full_attention", "conv", "sliding_attention", "conv",
            "full_attention", "conv", "sliding_attention", "conv", "full_attention", "conv"
          ],
          "max_position_embeddings": 262144,
          "model_type": "lfm3",
          "norm_eps": 1e-05,
          "num_attention_heads": 16,
          "num_hidden_layers": 24,
          "num_key_value_heads": 4,
          "pad_token_id": 124893,
          "rope_parameters": {
            "full_attention": {
              "partial_rotary_factor": 0.5, "rope_theta": 1000000.0, "rope_type": "default"
            },
            "sliding_attention": {"rope_theta": 10000, "rope_type": "default"}
          },
          "sliding_window": 256,
          "tie_word_embeddings": true,
          "transformers_version": "5.12.1",
          "use_cache": true,
          "use_sandwich_norm": true,
          "vocab_size": 128000
        }
        """

    private static let tinyLayerTypes = [
        "conv", "sliding_attention", "full_attention", "conv", "sliding_attention",
        "full_attention",
    ]

    private func decode(_ json: String) throws -> LFM3Configuration {
        try JSONDecoder().decode(LFM3Configuration.self, from: Data(json.utf8))
    }

    /// A tiny config with the real layer kinds and a sliding window shorter than the
    /// test prompts.
    private func tinyConfiguration(
        layerTypes: [String] = tinyLayerTypes, tieWordEmbeddings: Bool = true,
        intermediateSize: Int = 48,
        fullRope: String =
            #"{"rope_type": "default", "rope_theta": 1000000.0, "partial_rotary_factor": 0.5}"#
    ) throws -> LFM3Configuration {
        let layerTypesJSON = layerTypes.map { "\"\($0)\"" }.joined(separator: ", ")
        return try decode(
            """
            {
                "model_type": "lfm3", "vocab_size": 64, "hidden_size": 32,
                "intermediate_size": \(intermediateSize),
                "num_hidden_layers": \(layerTypes.count),
                "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
                "norm_eps": 1e-5, "conv_L_cache": 3, "conv_bias": false,
                "sliding_window": 4, "use_sandwich_norm": true,
                "tie_word_embeddings": \(tieWordEmbeddings),
                "layer_types": [\(layerTypesJSON)],
                "rope_parameters": {
                    "full_attention": \(fullRope),
                    "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0}
                }
            }
            """)
    }

    private func tinyModel(_ configuration: LFM3Configuration? = nil) throws -> LFM3Model {
        let configuration = try configuration ?? tinyConfiguration()
        return withRandomState(MLXRandom.RandomState(seed: 7)) {
            let model = LFM3Model(configuration)
            eval(model)
            return model
        }
    }

    // MARK: - Configuration

    func testRealConfigDecodes() throws {
        let c = try decode(realConfigJSON)

        XCTAssertEqual(c.hiddenLayers, 24)
        XCTAssertEqual(c.headDim, 128)
        XCTAssertEqual(c.kvHeads, 4)
        XCTAssertEqual(c.intermediateSize, 7680)
        XCTAssertEqual(c.convLCache, 7)
        XCTAssertFalse(c.convBias)
        XCTAssertEqual(c.slidingWindow, 256)
        XCTAssertTrue(c.useSandwichNorm)
        XCTAssertTrue(c.tieWordEmbeddings)

        let indices = { (type: LFM3Configuration.LayerType) in
            c.layerTypes.indices.filter { c.layerTypes[$0] == type }
        }
        XCTAssertEqual(indices(.fullAttention), [2, 6, 10, 14, 18, 22])
        XCTAssertEqual(indices(.slidingAttention), [4, 8, 12, 16, 20])
        XCTAssertEqual(indices(.conv).count, 13)

        let full = try XCTUnwrap(c.ropeParameters["full_attention"])
        XCTAssertEqual(full.ropeTheta, 1_000_000)
        XCTAssertEqual(full.partialRotaryFactor, 0.5)
        let sliding = try XCTUnwrap(c.ropeParameters["sliding_attention"])
        XCTAssertEqual(sliding.ropeTheta, 10_000)
        XCTAssertEqual(sliding.partialRotaryFactor, 1.0)
    }

    func testMissingFieldsTakeReferenceDefaults() throws {
        let c = try decode(
            """
            {
                "hidden_size": 32, "intermediate_size": 48, "num_hidden_layers": 9,
                "num_attention_heads": 4, "num_key_value_heads": 2
            }
            """)

        XCTAssertEqual(c.headDim, 8)
        XCTAssertEqual(c.slidingWindow, 256)
        XCTAssertEqual(c.convLCache, 7)
        XCTAssertEqual(
            c.layerTypes,
            [
                .conv, .conv, .fullAttention, .conv, .slidingAttention, .conv, .fullAttention,
                .conv, .slidingAttention,
            ])
        XCTAssertEqual(c.ropeParameters["full_attention"]?.partialRotaryFactor, 0.5)
        XCTAssertEqual(c.ropeParameters["sliding_attention"]?.ropeTheta, 10_000)
    }

    func testInvalidConfigsThrow() throws {
        XCTAssertThrowsError(
            try tinyConfiguration(layerTypes: ["conv", "linear_attention"]))
        XCTAssertThrowsError(
            try tinyConfiguration(
                fullRope: #"{"rope_type": "yarn", "rope_theta": 1000000.0, "factor": 4.0}"#))
        // layer_types must cover every layer.
        let json = realConfigJSON.replacingOccurrences(
            of: "\"num_hidden_layers\": 24", with: "\"num_hidden_layers\": 23")
        XCTAssertThrowsError(try decode(json))
    }

    /// The LFM2 key `tie_embedding` overrides `tie_word_embeddings`, as in the reference.
    func testTieEmbeddingKeyOverridesTieWordEmbeddings() throws {
        let tiny = { (keys: String) in
            try self.decode(
                """
                {
                    "hidden_size": 32, "intermediate_size": 48, "num_hidden_layers": 3,
                    "num_attention_heads": 4, "num_key_value_heads": 2\(keys)
                }
                """)
        }
        XCTAssertTrue(try tiny("").tieWordEmbeddings)
        XCTAssertFalse(try tiny(#", "tie_word_embeddings": false"#).tieWordEmbeddings)
        XCTAssertFalse(try tiny(#", "tie_embedding": false"#).tieWordEmbeddings)
        XCTAssertFalse(
            try tiny(#", "tie_embedding": false, "tie_word_embeddings": true"#).tieWordEmbeddings)
        XCTAssertTrue(
            try tiny(#", "tie_embedding": true, "tie_word_embeddings": false"#).tieWordEmbeddings)
    }

    func testRegisteredModelType() async throws {
        XCTAssertTrue(LLMRegistry.shared.contains(id: "LiquidAI/LFM3-1.7B-MLX-4bit"))
        let model = try await LLMTypeRegistry.shared.createModel(
            configuration: Data(realConfigJSON.utf8), modelType: "lfm3")
        XCTAssertTrue(model is LFM3Model)
    }

    func testDeclaresAlwaysOnThinking() throws {
        XCTAssertEqual(LFM3Model(try tinyConfiguration()).reasoningConfig, .alwaysOnThinking)
    }

    // MARK: - Model

    func testForwardShape() throws {
        let model = try tinyModel()
        let inputs = MLXArray((0 ..< 14).map { Int32($0 * 5 % 64) }).reshaped(2, 7)

        let logits = model(inputs, cache: nil)

        XCTAssertEqual(logits.shape, [2, 7, 64])
        XCTAssertNil(model.lmHead)
    }

    func testUntiedEmbeddingsUseLMHead() throws {
        let model = try tinyModel(tinyConfiguration(tieWordEmbeddings: false))
        let lmHead = try XCTUnwrap(model.lmHead)
        let tokens = MLXArray([Int32(1), 2, 3]).reshaped(1, 3)
        let logits = model(tokens, cache: nil)
        XCTAssertEqual(logits.shape, [1, 3, 64])
        XCTAssertEqual(abs(logits - lmHead(model.model(tokens))).max().item(Float.self), 0)
    }

    /// Distinct values for every tensor, from a hash of one running index over the parameters
    /// in name order: `1 + u` for norms and `u / 2` otherwise, with `u` in [-0.5, 0.5).
    private func hashedWeights(_ model: Module) -> [String: MLXArray] {
        var start = 0
        var weights = [String: MLXArray]()
        for (key, value) in model.parameters().flattened().sorted(by: { $0.0 < $1.0 }) {
            let values = (start ..< start + value.size).map { i in
                let u = Double(UInt64(i) * 2_654_435_761 % (1 << 32)) / 4_294_967_296 - 0.5
                return Float(value.ndim == 1 ? 1 + u : u / 2)
            }
            weights[key] = MLXArray(values, value.shape)
            start += value.size
        }
        return weights
    }

    /// Expected logits come from modeling_lfm3.py in float32 on the CPU with the same weights
    /// and tokens: the first 8 at positions 3, 10 and 18, with a sliding window of 4.
    func testLogitsMatchReference() throws {
        let model = try tinyModel()
        try model.update(
            parameters: ModuleParameters.unflattened(hashedWeights(model)), verify: [.all])
        let tokens = MLXArray((0 ..< 19).map { Int32(($0 * 7 + 3) % 64) }).reshaped(1, 19)
        let expected: [Int: [Float]] = [
            3: [
                0.124086142, -0.628987074, 0.808857799, -0.596109271, 0.524976909, -1.24413228,
                0.590065658, -0.482998401,
            ],
            10: [
                0.897291958, -1.01679277, -0.422243804, -0.528702438, -0.0963748991, 2.48546743,
                -0.584653139, 0.0170748997,
            ],
            18: [
                0.402512223, -0.151311055, -0.361906171, -0.781367064, 0.596630931, 0.113322325,
                0.043631494, -0.346791685,
            ],
        ]

        let logits = model(tokens, cache: nil)

        for (position, values) in expected {
            assertClose(
                logits[0, position, 0 ..< 8], MLXArray(values), atol: 1e-4, reducedAtol: 2e-2,
                "position \(position)")
        }
    }

    /// The quantized fast path projects only the trailing rows, and the last row must stay
    /// bitwise equal to the full forward pass, tied or untied.
    func testNextTokenLogitsMatchFullForward() throws {
        for (tied, bits) in [(true, 4), (true, 8), (false, 6)] {
            let model = try tinyModel(
                tinyConfiguration(tieWordEmbeddings: tied, intermediateSize: 64))
            quantize(model: model, groupSize: 32, bits: bits)
            for length in [1, 97, 200] {
                let text = LMInput.Text(
                    tokens: MLXArray((0 ..< length).map { Int32(($0 * 7 + 3) % 64) })
                        .reshaped(1, length))
                let expected = model(text.tokens, cache: nil)[0..., -1, 0...]
                let actual = model.nextTokenLogits(text, cache: nil, state: nil).logits
                if length > 96 {
                    XCTAssertLessThan(actual.dim(1), length)
                }
                XCTAssertTrue(
                    arrayEqual(actual[0..., -1, 0...], expected).item(Bool.self),
                    "tied \(tied), bits \(bits), length \(length)")
            }
        }
    }

    func testCacheTypesPerLayer() throws {
        let model = try tinyModel()

        let cache = try model.newCache(parameters: nil)
        XCTAssertEqual(cache.count, 6)
        for (i, layerType) in model.configuration.layerTypes.enumerated() {
            switch layerType {
            case .conv:
                XCTAssertTrue(cache[i] is MambaCache, "layer \(i)")
            case .slidingAttention:
                let rotating = try XCTUnwrap(cache[i] as? RotatingKVCache, "layer \(i)")
                XCTAssertEqual(rotating.maxSize, 4)
            case .fullAttention:
                XCTAssertTrue(cache[i] is KVCacheSimple, "layer \(i)")
            }
        }

        // maxKVSize bounds full attention and never widens the sliding window.
        let bounded = try model.newCache(parameters: GenerateParameters(maxKVSize: 16))
        XCTAssertEqual((bounded[1] as? RotatingKVCache)?.maxSize, 4)
        XCTAssertEqual((bounded[2] as? RotatingKVCache)?.maxSize, 16)
        XCTAssertTrue(bounded[0] is MambaCache)
    }

    func testPartialRoPEMatchesReference() throws {
        let attention = try XCTUnwrap(try tinyModel().model.layers[2].attention)
        let (headDim, base, offset) = (16, Float(1_000_000), 3)
        let x = withRandomState(MLXRandom.RandomState(seed: 1)) {
            MLXRandom.normal([1, 2, 5, headDim])
        }

        // The reference keeps the full-head frequencies for the first
        // `partial_rotary_factor * headDim / 2` angles and zeroes the rest.
        let angles = 4
        let invFreq = MLXArray(
            (0 ..< headDim / 2).map { j in
                j < angles ? pow(base, -Float(2 * j) / Float(headDim)) : 0
            })
        let positions = MLXArray((offset ..< offset + 5).map(Float.init))
        let freqs = positions.expandedDimensions(axis: -1) * invFreq
        let emb = concatenated([freqs, freqs], axis: -1)
        let halves = split(x, parts: 2, axis: -1)
        let rotated = concatenated([-halves[1], halves[0]], axis: -1)
        let expected = x * cos(emb) + rotated * sin(emb)

        let actual = applyRotaryPosition(attention.rope, to: x, offset: .scalar(offset))

        XCTAssertLessThan(abs(actual - expected).max().item(Float.self), 1e-4)
    }

    func testSanitizeLoadsMLXAndTransformersLayouts() throws {
        let model = try tinyModel()
        let parameters = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
        let convKeys = parameters.keys.filter { $0.hasSuffix("conv.conv.weight") }
        XCTAssertEqual(convKeys.count, 2)

        // MLX bundles: wrapped under `language_model`, plus tensors the text model drops.
        var mlx = Dictionary(
            uniqueKeysWithValues: parameters.map { ("language_model." + $0.key, $0.value) })
        mlx["language_model.lm_head.weight"] = MLXArray.zeros([64, 32])
        mlx["vision_tower.encoder.weight"] = MLXArray.zeros([4])

        // Transformers checkpoints store the depthwise conv as [C, 1, K].
        var transformers = parameters
        for key in convKeys {
            transformers[key] = parameters[key]!.transposed(0, 2, 1)
        }

        for weights in [mlx, transformers] {
            let sanitized = try model.sanitize(weights: weights)
            XCTAssertEqual(Set(sanitized.keys), Set(parameters.keys))
            for (key, value) in sanitized {
                XCTAssertEqual(value.shape, parameters[key]!.shape, key)
            }
            try model.update(parameters: ModuleParameters.unflattened(sanitized), verify: [.all])
        }
    }

    /// Prefill in one call, in chunks, and one token at a time must give the same logits,
    /// including past the sliding window where the rotating caches wrap.
    func testChunkedPrefillAndDecodeMatchOneShot() throws {
        let model = try tinyModel()
        let tokenCount = 19
        let tokens = MLXArray((0 ..< tokenCount).map { Int32(($0 * 7 + 3) % 64) })
            .reshaped(1, tokenCount)

        let cacheless = model(tokens, cache: nil)
        let oneShotCache = try model.newCache(parameters: nil)
        let oneShot = model(tokens, cache: oneShotCache)
        eval(cacheless, oneShot)
        XCTAssertLessThan(
            abs(oneShot - cacheless).max().item(Float.self), MatmulPrecision.splitTolerance)

        let decodeCount = 6
        for chunkSize in [1, 3, 4, 5, 13] {
            let cache = try model.newCache(parameters: nil)
            var logits = [MLXArray]()
            var start = 0
            while start < tokenCount {
                let prefillEnd = tokenCount - decodeCount
                let end = start < prefillEnd ? min(start + chunkSize, prefillEnd) : start + 1
                logits.append(model(tokens[0..., start ..< end], cache: cache))
                eval(logits.last!)
                start = end
            }

            let drift = abs(concatenated(logits, axis: 1) - oneShot).max().item(Float.self)
            XCTAssertLessThan(drift, MatmulPrecision.splitTolerance, "chunk size \(chunkSize)")
            XCTAssertEqual(cache.map(\.offset), oneShotCache.map(\.offset))
            // Like LFM2, conv caches hold a state window and do not track an offset.
            for layerCache in cache {
                if layerCache is MambaCache {
                    XCTAssertFalse(layerCache.innerState().isEmpty)
                } else {
                    XCTAssertEqual(layerCache.offset, tokenCount)
                }
            }
        }
    }

    // MARK: - Backbone entry points

    /// Token ids and their embeddings take the same path, so logits match exactly.
    func testEmbeddingInputMatchesTokenInput() throws {
        for tied in [true, false] {
            let model = try tinyModel(tinyConfiguration(tieWordEmbeddings: tied))
            let tokens = MLXArray((0 ..< 14).map { Int32($0 * 5 % 64) }).reshaped(2, 7)

            let fromTokens = model(tokens, cache: nil)
            let hidden = model.model(embeddings: model.model.embed(tokens))
            let fromEmbeddings = model.projectLogits(hidden)

            XCTAssertEqual(hidden.shape, [2, 7, model.model.hiddenSize])
            XCTAssertEqual(
                abs(fromEmbeddings - fromTokens).max().item(Float.self), 0, "tied \(tied)")
            XCTAssertEqual(
                abs(model.hiddenStates(tokens, cache: nil) - hidden).max().item(Float.self), 0)
        }
    }

    /// Embedding prefill in chunks and then one step at a time must match one shot, past
    /// the sliding window where the rotating caches wrap.
    func testChunkedEmbeddingPrefillMatchesOneShot() throws {
        let model = try tinyModel()
        let tokenCount = 19
        let tokens = MLXArray((0 ..< tokenCount).map { Int32(($0 * 5 + 1) % 64) })
            .reshaped(1, tokenCount)
        let embeddings = model.model.embed(tokens)

        let oneShot = model(tokens, cache: nil)
        eval(oneShot)

        let decodeCount = 5
        for chunkSize in [2, 4, 7, 14] {
            let cache = try model.newCache(parameters: nil)
            var logits = [MLXArray]()
            var start = 0
            while start < tokenCount {
                let prefillEnd = tokenCount - decodeCount
                let end = start < prefillEnd ? min(start + chunkSize, prefillEnd) : start + 1
                let hidden = model.model(embeddings: embeddings[0..., start ..< end], cache: cache)
                logits.append(model.projectLogits(hidden))
                eval(logits.last!)
                start = end
            }

            let drift = abs(concatenated(logits, axis: 1) - oneShot).max().item(Float.self)
            XCTAssertLessThan(drift, MatmulPrecision.splitTolerance, "chunk size \(chunkSize)")
            for (i, layerCache) in cache.enumerated() where !(layerCache is MambaCache) {
                XCTAssertEqual(layerCache.offset, tokenCount, "layer \(i)")
            }
        }
    }

    /// Masks passed per attention type replace the defaults for that type only.
    func testAttentionMaskOverrides() throws {
        let model = try tinyModel()
        let tokens = MLXArray((0 ..< 9).map { Int32($0 * 3 % 64) }).reshaped(1, 9)
        let window = model.configuration.slidingWindow
        let defaults = model.model(tokens)

        let difference = { (masks: LFM3AttentionMasks) in
            abs(model.model(tokens, masks: masks) - defaults).max().item(Float.self)
        }

        // An override equal to its type's default changes nothing, so each one reaches only
        // its own layers.
        let windowed = MLXFast.ScaledDotProductAttentionMaskMode.array(
            createCausalMask(n: 9, offset: 0, windowSize: window))
        XCTAssertLessThan(difference(.init(fullAttention: .causal)), 1e-5)
        XCTAssertLessThan(difference(.init(slidingAttention: windowed)), 1e-5)

        // Any other mask changes the output: no mask on full layers, no window on sliding ones.
        let unmasked = MLXFast.ScaledDotProductAttentionMaskMode.none
        XCTAssertGreaterThan(difference(.init(fullAttention: unmasked)), 1e-3)
        XCTAssertGreaterThan(difference(.init(slidingAttention: .causal)), 1e-3)
    }

    // MARK: - Mixed-precision checkpoint

    /// An MLX conversion wrapped in `language_model` with 4-bit attention and conv, 6-bit
    /// MLP, and an 8-bit untied `lm_head`, declared per layer in `config.json`. The prefix
    /// mapping must rename tensors and settings together or strict loading fails.
    func testLoadsMixedPrecisionWrappedCheckpoint() throws {
        let groupSize = 32
        let configuration = try tinyConfiguration(tieWordEmbeddings: false, intermediateSize: 64)
        let reference = try tinyModel(configuration)

        let bits = { (path: String) in
            path == "lm_head" ? 8 : path.contains(".feed_forward.") ? 6 : 4
        }
        var arrays = [String: MLXArray]()
        var quantizedPaths = [String]()
        for (path, module) in reference.leafModules().flattened() {
            let weight: MLXArray
            switch module {
            case let linear as Linear: weight = linear.weight
            case let embedding as Embedding: weight = embedding.weight
            default: continue
            }
            let (wq, scales, biases) = quantized(weight, groupSize: groupSize, bits: bits(path))
            arrays["language_model.\(path).weight"] = wq
            arrays["language_model.\(path).scales"] = scales
            arrays["language_model.\(path).biases"] = biases
            quantizedPaths.append(path)
        }
        for (key, value) in reference.parameters().flattened() {
            let module = key.split(separator: ".").dropLast().joined(separator: ".")
            if !quantizedPaths.contains(module) {
                arrays["language_model.\(key)"] = value
            }
        }
        arrays["vision_tower.proj.weight"] = MLXArray.zeros([4, 4])
        XCTAssertTrue(quantizedPaths.contains("lm_head"))
        XCTAssertTrue(quantizedPaths.contains { $0.hasSuffix("conv.in_proj") })

        // Per-layer entries use the checkpoint's own (wrapped) names.
        let overrides =
            quantizedPaths.filter { bits($0) != 4 }.map { path in
                #""language_model.\#(path)": {"group_size": \#(groupSize), "bits": \#(bits(path))}"#
            } + [#""vision_tower.proj": {"group_size": \#(groupSize), "bits": 8}"#]
        let baseConfiguration = try JSONDecoder().decode(
            BaseConfiguration.self,
            from: Data(
                """
                {
                    "model_type": "lfm3",
                    "quantization": {
                        "group_size": \(groupSize), "bits": 4,
                        \(overrides.joined(separator: ",\n"))
                    }
                }
                """.utf8))

        let directory = FileManager.default.temporaryDirectory
            .appending(component: "lfm3-mixed-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try save(arrays: arrays, url: directory.appending(component: "model.safetensors"))

        let model = LFM3Model(configuration)
        try loadWeights(
            modelDirectory: directory, model: model,
            perLayerQuantization: baseConfiguration.perLayerQuantization)

        let modules = Dictionary(uniqueKeysWithValues: model.leafModules().flattened())
        for path in quantizedPaths {
            let module = try XCTUnwrap(modules[path] as? Quantized, path)
            XCTAssertEqual(module.bits, bits(path), path)
            XCTAssertEqual(module.groupSize, groupSize, path)
        }

        let logits = model(MLXArray([Int32(1), 2, 3, 4, 5]).reshaped(1, 5), cache: nil)
        XCTAssertEqual(logits.shape, [1, 5, 64])
        XCTAssertFalse(any(isNaN(logits)).item(Bool.self))
    }
}
