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

    func testInvalidDimensionsFailDuringDecoding() throws {
        let invalid: [(String, Any)] = [
            ("hidden_size", 0), ("hidden_size", -1),
            ("num_attention_heads", 0), ("num_attention_heads", -1),
            ("num_key_value_heads", 0), ("num_key_value_heads", 3),
            ("head_dim", 0), ("head_dim", -2), ("head_dim", 127),
            ("num_hidden_layers", 0), ("num_hidden_layers", -1),
            ("vocab_size", 0), ("intermediate_size", -1),
            ("sliding_window", 0), ("sliding_window", -1),
            ("conv_L_cache", 0), ("conv_L_cache", -1),
            ("norm_eps", 0), ("norm_eps", -1), ("max_position_embeddings", 0),
        ]
        for (key, value) in invalid {
            var json = try XCTUnwrap(
                JSONSerialization.jsonObject(with: Data(realConfigJSON.utf8)) as? [String: Any])
            json[key] = value
            // Exercise the inferred head dimension too: this used to divide by zero.
            if key == "num_attention_heads" { json.removeValue(forKey: "head_dim") }
            let data = try JSONSerialization.data(withJSONObject: json)
            XCTAssertThrowsError(try JSONDecoder().decode(LFM3Configuration.self, from: data), key)
            {
                XCTAssertTrue($0 is DecodingError, "\(key): \($0)")
            }
        }
    }

    func testInvalidRotaryParametersThrow() throws {
        for rope in [
            #"{"rope_theta": 0}"#, #"{"rope_theta": -10000}"#,
            #"{"rope_theta": 10000, "partial_rotary_factor": -0.1}"#,
            #"{"rope_theta": 10000, "partial_rotary_factor": 1.1}"#,
        ] {
            XCTAssertThrowsError(try tinyConfiguration(fullRope: rope))
        }
        var json = try XCTUnwrap(
            JSONSerialization.jsonObject(with: Data(realConfigJSON.utf8)) as? [String: Any])
        json["rope_parameters"] = ["sliding_attention": ["rope_theta": 10000]]
        XCTAssertThrowsError(
            try JSONDecoder().decode(
                LFM3Configuration.self, from: JSONSerialization.data(withJSONObject: json)))
        json.removeValue(forKey: "hidden_size")
        XCTAssertThrowsError(
            try JSONDecoder().decode(
                LFM3Configuration.self, from: JSONSerialization.data(withJSONObject: json)))
    }

    func testTokenizerCompatibilityProbes() throws {
        XCTAssertNoThrow(try LFM3Model.validateTokenizer(ProbeTokenizer()))
        for broken in [".reduce", "line.\r\nThis"] {
            XCTAssertThrowsError(try LFM3Model.validateTokenizer(ProbeTokenizer(broken: broken))) {
                XCTAssertTrue($0.localizedDescription.contains("ignore_merges"))
            }
        }
        // A custom vocabulary without these byte-level entries is not the affected tokenizer.
        XCTAssertNoThrow(try LFM3Model.validateTokenizer(ProbeTokenizer(hasProbeVocabulary: false)))
    }

    func testFactoryRejectsIncompatibleTokenizer() async throws {
        let directory = FileManager.default.temporaryDirectory.appending(
            component: UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let config = try tinyConfiguration()
        try JSONEncoder().encode(config).write(to: directory.appending(component: "config.json"))
        let model = try tinyModel(config)
        try save(
            arrays: Dictionary(uniqueKeysWithValues: model.parameters().flattened()),
            url: directory.appending(component: "model.safetensors"))
        do {
            _ = try await LLMModelFactory.shared.load(
                from: directory, using: ProbeTokenizerLoader())
            XCTFail("The factory accepted a tokenizer that ignores ignore_merges")
        } catch {
            XCTAssertTrue(error.localizedDescription.contains("ignore_merges"), "\(error)")
        }
    }

    private struct ProbeTokenizerLoader: TokenizerLoader {
        func load(from directory: URL) async throws -> any Tokenizer {
            ProbeTokenizer(broken: ".reduce")
        }
    }

    private struct ProbeTokenizer: Tokenizer {
        var broken: String? = nil
        var hasProbeVocabulary = true
        let bosToken: String? = nil
        let eosToken: String? = nil
        let unknownToken: String? = nil
        private let vocabulary = [
            "<|startoftext|>": 124894, ".reduce": 96744, "line": 1421, ".čĊ": 64122, "This": 1942,
        ]

        func encode(text: String, addSpecialTokens: Bool) -> [Int] {
            if text == broken { return [1, 2] }
            return text == ".reduce" ? [96744] : [1421, 64122, 1942]
        }
        func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { "" }
        func convertTokenToId(_ token: String) -> Int? {
            hasProbeVocabulary ? vocabulary[token] : nil
        }
        func convertIdToToken(_ id: Int) -> String? { nil }
        func applyChatTemplate(
            messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
            additionalContext: [String: any Sendable]?
        ) throws -> [Int] { [] }
    }

    func testOfficialXMLToolCallFormat() throws {
        let model = try tinyModel()
        XCTAssertEqual(model.toolCallFormat, .xmlFunction)
        XCTAssertEqual(model.reasoningConfig?.startDelimiter, "<think>")
        XCTAssertEqual(model.reasoningConfig?.endDelimiter, "</think>")
        // Rendered by LiquidAI's chat_template.jinja, not LFM2's Python-style format.
        let payload = """
            <tool_call>
            <function=get_weather>
            <parameter=location>
            Paris
            </parameter>
            </function>
            </tool_call>
            """
        let parser = XMLFunctionParser(startTag: "<tool_call>", endTag: "</tool_call>")
        let call = try XCTUnwrap(parser.parse(content: payload, tools: nil))
        XCTAssertEqual(call.function.name, "get_weather")
        XCTAssertEqual(call.function.arguments["location"], .string("Paris"))
        XCTAssertNil(
            parser.parse(
                content: payload.replacingOccurrences(of: "</parameter>", with: ""), tools: nil))
    }

    func testRegisteredModelType() async throws {
        XCTAssertTrue(LLMRegistry.shared.contains(id: "LiquidAI/LFM3-1.7B-MLX-4bit"))
        let model = try await LLMTypeRegistry.shared.createModel(
            configuration: Data(realConfigJSON.utf8), modelType: "lfm3")
        XCTAssertTrue(model is LFM3Model)
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
        XCTAssertNotNil(model.lmHead)
        XCTAssertEqual(
            model(MLXArray([Int32(1), 2, 3]).reshaped(1, 3), cache: nil).shape, [1, 3, 64])
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

    func testRotaryFactorsAndLongOffsets() throws {
        let headDim = 16
        let x = MLXArray((0 ..< 3 * headDim).map { Float($0 % 17 - 8) / 8 })
            .reshaped(1, 1, 3, headDim)
        for factor: Float in [0, 0.25, 0.6, 0.75, 1] {
            let config = try tinyConfiguration(
                fullRope: "{\"rope_theta\": 1000000, \"partial_rotary_factor\": \(factor)}")
            let attention = try XCTUnwrap(try tinyModel(config).model.layers[2].attention)
            for offset in [0, 257, 4096, 16384, 32768] {
                let angles = Int(factor * Float(headDim)) / 2
                let invFreq = MLXArray(
                    (0 ..< headDim / 2).map { j in
                        j < angles ? pow(Float(1_000_000), -Float(2 * j) / Float(headDim)) : 0
                    })
                let positions = MLXArray((offset ..< offset + 3).map(Float.init))
                let freqs = positions.expandedDimensions(axis: -1) * invFreq
                let emb = concatenated([freqs, freqs], axis: -1)
                let halves = split(x, parts: 2, axis: -1)
                let rotated = concatenated([-halves[1], halves[0]], axis: -1)
                let expected = x * cos(emb) + rotated * sin(emb)
                let actual = applyRotaryPosition(attention.rope, to: x, offset: .scalar(offset))
                XCTAssertLessThan(
                    abs(actual - expected).max().item(Float.self), 1e-3,
                    "factor \(factor), offset \(offset)")
            }
        }
    }

    func testConvolutionStateMatchesExplicitCausalSum() throws {
        for width in [1, 3, 7] {
            var json = try XCTUnwrap(
                JSONSerialization.jsonObject(with: JSONEncoder().encode(tinyConfiguration()))
                    as? [String: Any])
            json["conv_L_cache"] = width
            json["conv_bias"] = true
            json["use_sandwich_norm"] = false
            let config = try JSONDecoder().decode(
                LFM3Configuration.self, from: JSONSerialization.data(withJSONObject: json))
            let conv = try XCTUnwrap(try tinyModel(config).model.layers[0].conv)
            let values: [Float] = (0 ..< 704).map { Float($0 % 19 - 9) / 10 }
            let x = MLXArray(values).reshaped(2, 11, 32)
            let parts = conv.inProj(x).split(parts: 3, axis: -1)
            let bx = parts[0] * parts[2]
            let padded = concatenated([MLXArray.zeros([2, width - 1, 32]), bx], axis: 1)
            var sum = MLXArray.zeros([2, 11, 32])
            for k in 0 ..< width {
                sum = sum + padded[0..., k ..< k + 11, 0...] * conv.conv.weight[0..., k, 0]
            }
            if let bias = conv.conv.bias { sum = sum + bias }
            let expected = conv.outProj(parts[1] * sum)
            assertClose(conv(x, cache: nil), expected, atol: 1e-5, reducedAtol: 5e-4)
            for chunkSize in [1, 2, 5, 11] {
                let cache = MambaCache()
                var outputs = [MLXArray]()
                for start in stride(from: 0, to: 11, by: chunkSize) {
                    outputs.append(
                        conv(x[0..., start ..< min(start + chunkSize, 11)], cache: cache))
                    eval(outputs.last!)
                }
                assertClose(concatenated(outputs, axis: 1), expected, atol: 1e-5, reducedAtol: 5e-4)
                let state = try XCTUnwrap(cache[0])
                XCTAssertEqual(state.shape, [2, width - 1, 32])
                if width > 1 {
                    XCTAssertEqual(
                        abs(state - bx[0..., (11 - (width - 1))..., 0...]).max().item(Float.self), 0
                    )
                }
            }
        }
    }

    func testSlidingWindowBoundariesAndCacheCopy() throws {
        var json = try XCTUnwrap(
            JSONSerialization.jsonObject(with: JSONEncoder().encode(tinyConfiguration()))
                as? [String: Any])
        json["sliding_window"] = 256
        json["conv_L_cache"] = 7
        let config = try JSONDecoder().decode(
            LFM3Configuration.self, from: JSONSerialization.data(withJSONObject: json))
        let model = try tinyModel(config)
        for length in [256, 257, 512, 1024] {
            let tokens = MLXArray((0 ..< 2 * length).map { Int32(($0 * 7 + 3) % 64) }).reshaped(
                2, length)
            let expected = model(tokens, cache: nil)
            eval(expected)
            for chunkSize in [127, 257] {
                var cache = try model.newCache(parameters: nil)
                var outputs = [MLXArray]()
                for start in stride(from: 0, to: length, by: chunkSize) {
                    outputs.append(
                        model(tokens[0..., start ..< min(start + chunkSize, length)], cache: cache))
                    eval(outputs.last!)
                    eval(cache)
                    // Copies model prompt-cache reuse, including convolution state.
                    cache = cache.map { $0.copy() }
                }
                XCTAssertLessThan(
                    abs(concatenated(outputs, axis: 1) - expected).max().item(Float.self),
                    MatmulPrecision.splitTolerance)
                let decode = MLXArray([Int32(1), 2]).reshaped(2, 1)
                let actual = model(decode, cache: cache)
                let fresh = try model.newCache(parameters: nil)
                let oneShot = model(concatenated([tokens, decode], axis: 1), cache: fresh)[
                    0..., length...]
                XCTAssertLessThan(
                    abs(actual - oneShot).max().item(Float.self), MatmulPrecision.splitTolerance)
                XCTAssertTrue(all(isFinite(actual)).item(Bool.self))
            }
        }
    }

    func testConvolutionPreparedLengthsPreserveLastValidState() throws {
        var json = try XCTUnwrap(
            JSONSerialization.jsonObject(with: JSONEncoder().encode(tinyConfiguration()))
                as? [String: Any])
        json["conv_L_cache"] = 7
        json["conv_bias"] = true
        let config = try JSONDecoder().decode(
            LFM3Configuration.self, from: JSONSerialization.data(withJSONObject: json))
        let conv = try XCTUnwrap(try tinyModel(config).model.layers[0].conv)
        let values: [Float] = (0 ..< 768).map { Float($0 % 19 - 9) / 10 }
        let x = MLXArray(values).reshaped(3, 8, 32)
        let lengths = [0, 3, 8]
        for leftPadded in [false, true] {
            let independent = lengths.enumerated().map { batch, length in
                let cache = MambaCache()
                if length > 0 {
                    let start = leftPadded ? 8 - length : 0
                    eval(conv(x[batch ..< batch + 1, start ..< start + length], cache: cache))
                }
                return cache
            }
            for chunkSize in [2, 5, 8] {
                let cache = MambaCache(leftPadding: leftPadded ? lengths.map { 8 - $0 } : nil)
                withPreparedCache([cache], lengths: leftPadded ? nil : lengths) {
                    for start in stride(from: 0, to: 8, by: chunkSize) {
                        eval(conv(x[0..., start ..< min(start + chunkSize, 8)], cache: cache))
                    }
                }
                let next = x[0..., 0 ..< 1]
                let expected = concatenated(
                    independent.enumerated().map { batch, reference in
                        conv(next[batch ..< batch + 1], cache: reference.copy() as? MambaCache)
                    }, axis: 0)
                assertClose(conv(next, cache: cache), expected, atol: 1e-5, reducedAtol: 1e-4)
            }
        }
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

    func testMalformedConvolutionWeightsThrow() throws {
        let model = try tinyModel()
        for shape in [[], [32], [32, 3]] {
            XCTAssertThrowsError(
                try model.sanitize(weights: [
                    "model.layers.0.conv.conv.weight": MLXArray.zeros(shape)
                ]))
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

        let explicit = model.model(
            tokens,
            masks: .init(
                fullAttention: .causal,
                slidingAttention: .array(createCausalMask(n: 9, offset: 0, windowSize: window))))
        XCTAssertLessThan(abs(explicit - defaults).max().item(Float.self), 1e-5)

        // A plain causal mask lets sliding layers see past the window.
        let unwindowed = model.model(tokens, masks: .init(slidingAttention: .causal))
        XCTAssertGreaterThan(abs(unwindowed - defaults).max().item(Float.self), 1e-3)
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
