// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXLLM

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
        fullRope: String =
            #"{"rope_type": "default", "rope_theta": 1000000.0, "partial_rotary_factor": 0.5}"#
    ) throws -> LFM3Configuration {
        let layerTypesJSON = layerTypes.map { "\"\($0)\"" }.joined(separator: ", ")
        return try decode(
            """
            {
                "model_type": "lfm3", "vocab_size": 64, "hidden_size": 32,
                "intermediate_size": 48, "num_hidden_layers": \(layerTypes.count),
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
}
