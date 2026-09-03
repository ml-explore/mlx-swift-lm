// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXLLM

final class LFM2Tests: XCTestCase {
    private struct TokenizerLoaderStub: MLXLMCommon.TokenizerLoader {
        func load(from directory: URL) async throws -> any Tokenizer {
            TestTokenizer(vocabularySize: 65_536)
        }
    }

    private static let tinyConfigJSON = """
        {
          "model_type": "lfm2",
          "vocab_size": 32,
          "hidden_size": 8,
          "num_hidden_layers": 2,
          "num_attention_heads": 2,
          "num_key_value_heads": 1,
          "max_position_embeddings": 128,
          "norm_eps": 1e-5,
          "conv_bias": false,
          "conv_L_cache": 3,
          "block_dim": 8,
          "intermediate_size": 24,
          "block_multiple_of": 8,
          "block_ffn_dim_multiplier": 1.0,
          "block_auto_adjust_ff_dim": false,
          "layer_types": ["conv", "full_attention"],
          "rope_parameters": { "rope_theta": 10000000.0 }
        }
        """

    private func config() throws -> LFM2Configuration {
        try JSONDecoder().decode(
            LFM2Configuration.self, from: Data(Self.tinyConfigJSON.utf8))
    }

    func testTransformersIntermediateSizeBuildsReleasedCheckpointShape() throws {
        let model = LFM2Model(try config())
        let parameters = Dictionary(uniqueKeysWithValues: model.parameters().flattened())

        XCTAssertEqual(
            parameters["model.layers.0.feed_forward.w1.weight"]?.shape, [24, 8])
        XCTAssertEqual(
            parameters["model.layers.0.feed_forward.w2.weight"]?.shape, [8, 24])
        XCTAssertEqual(
            parameters["model.layers.0.feed_forward.w3.weight"]?.shape, [24, 8])
    }

    func testConfigurationRejectsConflictingLayerLayoutDeclarations() throws {
        let json = Self.tinyConfigJSON.replacingOccurrences(
            of: #""layer_types": ["conv", "full_attention"]"#,
            with: #""full_attn_idxs": [0], "layer_types": ["conv", "full_attention"]"#)
        let configuration = try JSONDecoder().decode(
            LFM2Configuration.self, from: Data(json.utf8))

        XCTAssertThrowsError(try configuration.validateModelConfiguration())
    }

    /// Regression for the Lampo failure:
    /// `keyNotFound(path: ["model", "layers", "0", "conv", "in_proj", "weight"])`.
    func testCommunityLanguageModelNamespaceLoadsWithFullVerification() throws {
        let model = LFM2Model(try config())
        let communityWeights = Dictionary(
            uniqueKeysWithValues: model.parameters().flattened().map { name, value in
                ("language_model.\(name)", value)
            })

        let sanitized = model.sanitize(weights: communityWeights)

        XCTAssertNotNil(sanitized["model.layers.0.conv.in_proj.weight"])
        XCTAssertNil(sanitized["language_model.model.layers.0.conv.in_proj.weight"])
        try model.update(
            parameters: ModuleParameters.unflattened(sanitized), verify: [.all])
    }

    func testNativeNamespaceIsNeverRewritten() throws {
        let model = LFM2Model(try config())
        let native = [
            "model.layers.0.conv.in_proj.weight": MLXArray.zeros([24, 8]),
            "language_model.metadata": MLXArray([Float(1)]),
        ]

        let sanitized = model.sanitize(weights: native)

        XCTAssertNotNil(sanitized["model.layers.0.conv.in_proj.weight"])
        XCTAssertNotNil(sanitized["language_model.metadata"])
    }

    func testHybridCacheAndWarmContinuationMatchColdPrefill() throws {
        MLXRandom.seed(17)
        let model = LFM2Model(try config())
        let prefix = MLXArray([Int32(1), 2, 3, 4]).reshaped(1, 4)
        let suffix = MLXArray([Int32(5), 6]).reshaped(1, 2)
        let full = concatenated([prefix, suffix], axis: 1)

        let coldCache = try model.newCache(parameters: nil)
        let cold = model(full, cache: coldCache)[0..., (-2)..., 0...]

        let warmCache = try model.newCache(parameters: nil)
        _ = model(prefix, cache: warmCache)
        let warm = model(suffix, cache: warmCache)
        eval(cold, warm)

        let difference = abs(cold - warm).max().item(Float.self)
        XCTAssertLessThanOrEqual(difference, 1e-4)
        XCTAssertTrue(warmCache[0] is RewindableConvolutionCache)
        XCTAssertTrue(warmCache[1] is KVCacheSimple)
        XCTAssertEqual(warmCache[1].offset, 6)
    }

    func testRaggedBatchStoresEachRowsLogicalConvolutionEndpoint() throws {
        let conv = LFM2ShortConv(try config(), layerIdx: 0)
        let cache = MambaCache()
        let input = MLXArray((0 ..< 64).map(Float.init)).reshaped(2, 4, 8)
        cache.prepare(lengths: [4, 2])

        _ = conv(input, mask: cache.makeMask(N: 4), cache: cache)
        let state = try XCTUnwrap(cache[0])
        eval(state)

        XCTAssertEqual(state.shape, [2, 2, 8])
        XCTAssertEqual(cache.currentLengths?.asArray(Int.self), [0, -2])
    }

    func testDeclaresArchitectureToolConventionWithoutAssumingReasoning() throws {
        let model = LFM2Model(try config())

        XCTAssertEqual(model.toolCallFormat, .lfm2)
        XCTAssertNil(model.reasoningConfig)
    }

    func testAttentionCacheHonorsLongContextMemoryControls() throws {
        let model = LFM2Model(try config())

        let quantized = try model.newCache(
            parameters: GenerateParameters(kvBits: 4, quantizedKVStart: 0))
        XCTAssertTrue(quantized[0] is RewindableConvolutionCache)
        XCTAssertTrue(quantized[1] is QuantizedKVCache)

        let bounded = try model.newCache(parameters: GenerateParameters(maxKVSize: 64))
        XCTAssertTrue(bounded[0] is RewindableConvolutionCache)
        XCTAssertTrue(bounded[1] is RotatingKVCache)
        XCTAssertEqual(bounded[1].maxSize, 64)
    }

    func testHybridCacheCanRewindAndBranchWithoutRebuilding() throws {
        MLXRandom.seed(29)
        let model = LFM2Model(try config())
        let prefix = MLXArray([Int32(1), 2, 3, 4]).reshaped(1, 4)
        let discarded = MLXArray([Int32(5), 6]).reshaped(1, 2)
        let replacement = MLXArray([Int32(7), 8]).reshaped(1, 2)

        let rewoundCache = try model.newCache(parameters: nil)
        _ = model(prefix, cache: rewoundCache)
        _ = model(discarded, cache: rewoundCache)
        XCTAssertEqual(trimPromptCache(rewoundCache, numTokens: 2), 2)
        let rewound = model(replacement, cache: rewoundCache)

        let coldCache = try model.newCache(parameters: nil)
        let cold = model(concatenated([prefix, replacement], axis: 1), cache: coldCache)[
            0..., (-2)..., 0...]
        eval(rewound, cold)

        XCTAssertLessThanOrEqual(abs(rewound - cold).max().item(Float.self), 1e-4)
    }

    /// Opt-in integration test. Set `MLX_LFM2_MODEL_PATH` to a downloaded
    /// LFM2.5 checkpoint to verify every real tensor name and shape.
    func testDownloadedCheckpointLoadsThroughFactory() async throws {
        guard let path = ProcessInfo.processInfo.environment["MLX_LFM2_MODEL_PATH"] else {
            throw XCTSkip("Set MLX_LFM2_MODEL_PATH to run the real-checkpoint test.")
        }
        let directory = URL(filePath: path, directoryHint: .isDirectory)
        let generationData = try Data(
            contentsOf: directory.appending(component: "generation_config.json"))
        let generationConfig = try JSONDecoder.json5().decode(
            GenerationConfigFile.self, from: generationData)
        let context = try await LLMModelFactory.shared.load(
            from: directory, using: TokenizerLoaderStub())
        let model = try XCTUnwrap(context.model as? LFM2Model)
        let config = model.configuration

        let tokens = MLXArray([Int32(1), 2, 3]).reshaped(1, 3)
        let logits = model(tokens, cache: try model.newCache(parameters: nil))
        eval(logits)
        XCTAssertEqual(logits.shape, [1, 3, config.vocabularySize])
        XCTAssertEqual(context.configuration.toolCallFormat, .lfm2)
        XCTAssertEqual(context.configuration.reasoningConfig?.promptStrategy, .alwaysOn)
        XCTAssertNotNil(context.configuration.generationConfig)
        XCTAssertEqual(generationConfig.temperature, 0.1)
    }

    /// Opt-in device benchmark for the real checkpoint. This deliberately
    /// measures model forward throughput without tokenizer or UI overhead.
    func testDownloadedCheckpointPerformance() throws {
        guard ProcessInfo.processInfo.environment["MLX_LFM2_BENCHMARK"] == "1" else {
            throw XCTSkip("Set MLX_LFM2_BENCHMARK=1 to run the device benchmark.")
        }
        guard let path = ProcessInfo.processInfo.environment["MLX_LFM2_MODEL_PATH"] else {
            throw XCTSkip("Set MLX_LFM2_MODEL_PATH to run the device benchmark.")
        }
        let directory = URL(filePath: path, directoryHint: .isDirectory)
        let data = try Data(contentsOf: directory.appending(component: "config.json"))
        let config = try JSONDecoder().decode(LFM2Configuration.self, from: data)
        let model = LFM2Model(config)
        try loadWeights(modelDirectory: directory, model: model)

        // Warm Metal kernels and allocator state outside the timed samples.
        let warmCache = try model.newCache(parameters: nil)
        let warm = model(
            MLXArray([Int32](repeating: 42, count: 16)).reshaped(1, 16), cache: warmCache)
        eval(warm)
        for _ in 0 ..< 4 {
            eval(model(MLXArray([Int32(43)]).reshaped(1, 1), cache: warmCache))
        }

        Memory.peakMemory = 0
        let promptTokens = 256
        let decodeTokens = 32
        let cache = try model.newCache(parameters: nil)
        let prompt = MLXArray([Int32](repeating: 42, count: promptTokens)).reshaped(1, promptTokens)

        let prefillStart = Date.timeIntervalSinceReferenceDate
        eval(model(prompt, cache: cache))
        let prefillSeconds = Date.timeIntervalSinceReferenceDate - prefillStart

        let decodeStart = Date.timeIntervalSinceReferenceDate
        for _ in 0 ..< decodeTokens {
            eval(model(MLXArray([Int32(43)]).reshaped(1, 1), cache: cache))
        }
        let decodeSeconds = Date.timeIntervalSinceReferenceDate - decodeStart

        let prefillTPS = Double(promptTokens) / prefillSeconds
        let decodeTPS = Double(decodeTokens) / decodeSeconds
        let peakGiB = Double(Memory.peakMemory) / 1_073_741_824
        print(
            String(
                format: "LFM2.5-2.6B bf16: prefill %.2f tok/s, decode %.2f tok/s, peak %.2f GiB",
                prefillTPS, decodeTPS, peakGiB))

        XCTAssertTrue(prefillTPS.isFinite && prefillTPS > 0)
        XCTAssertTrue(decodeTPS.isFinite && decodeTPS > 0)
    }
}
