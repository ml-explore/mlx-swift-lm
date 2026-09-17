import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXLLM

final class LFM2MoeRoutingTests: XCTestCase {

    private struct TokenizerLoaderStub: MLXLMCommon.TokenizerLoader {
        func load(from directory: URL) async throws -> any Tokenizer {
            TestTokenizer(vocabularySize: 128_000)
        }
    }

    private func makeConfig(useExpertBias: Bool, normTopkProb: Bool = false) throws
        -> LFM2MoEConfiguration
    {
        let json = """
            {
                "model_type": "lfm2_moe",
                "vocab_size": 32,
                "hidden_size": 4,
                "intermediate_size": 8,
                "moe_intermediate_size": 8,
                "num_hidden_layers": 1,
                "num_experts": 4,
                "num_experts_per_tok": 2,
                "norm_topk_prob": \(normTopkProb),
                "num_attention_heads": 1,
                "num_key_value_heads": 1,
                "max_position_embeddings": 128,
                "use_expert_bias": \(useExpertBias),
                "num_dense_layers": 0,
                "norm_eps": 1e-5,
                "conv_bias": false,
                "conv_L_cache": 3
            }
            """
        return try JSONDecoder().decode(LFM2MoEConfiguration.self, from: Data(json.utf8))
    }

    private func makeHybridConfig(numExpertsPerToken: Int = 2) throws -> LFM2MoEConfiguration {
        let json = """
            {
                "model_type": "lfm2_moe",
                "vocab_size": 32,
                "hidden_size": 8,
                "intermediate_size": 16,
                "moe_intermediate_size": 16,
                "num_hidden_layers": 2,
                "num_experts": 4,
                "num_experts_per_tok": \(numExpertsPerToken),
                "norm_topk_prob": true,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "max_position_embeddings": 128,
                "use_expert_bias": true,
                "num_dense_layers": 1,
                "norm_eps": 1e-5,
                "conv_bias": false,
                "conv_L_cache": 3,
                "layer_types": ["conv", "full_attention"],
                "rope_parameters": { "rope_theta": 5000000.0 }
            }
            """
        return try JSONDecoder().decode(LFM2MoEConfiguration.self, from: Data(json.utf8))
    }

    private func makeBlock(
        useExpertBias: Bool, expertBias: [Float]? = nil,
        normTopkProb: Bool = false
    ) throws -> Lfm2MoeSparseMoeBlock {
        let block = Lfm2MoeSparseMoeBlock(
            try makeConfig(useExpertBias: useExpertBias, normTopkProb: normTopkProb))
        var params: [String: MLXArray] = ["gate.weight": MLX.eye(4)]
        if let expertBias {
            params["expert_bias"] = MLXArray(expertBias)
        }
        try block.update(parameters: ModuleParameters.unflattened(params), verify: [])
        eval(block)
        return block
    }

    private let logits: [Float] = [2, 1, 0, -1]
    private func x() -> MLXArray { MLXArray(logits).reshaped(1, 1, 4) }
    private func sig(_ v: Float) -> Float { 1 / (1 + expf(-v)) }

    private func routed(_ block: Lfm2MoeSparseMoeBlock) -> [Int: Float] {
        let r = block.route(x())
        let idx = r.indices.reshaped(-1).asArray(Int32.self).map(Int.init)
        let w = r.weights.reshaped(-1).asArray(Float.self)
        return Dictionary(uniqueKeysWithValues: zip(idx, w))
    }

    func testExpertBiasSteersSelectionOnly() throws {
        let block = try makeBlock(useExpertBias: true, expertBias: [0, 0, 1, 0])
        let m = routed(block)

        XCTAssertEqual(Set(m.keys), [0, 2], "expert_bias must move expert 2 into the top-k")
        XCTAssertEqual(m[2] ?? .nan, sig(0), accuracy: 1e-4)
        XCTAssertEqual(m[0] ?? .nan, sig(2), accuracy: 1e-4)
    }

    func testGateIsSigmoidNotSoftmax() throws {
        let block = try makeBlock(useExpertBias: false)
        let m = routed(block)

        XCTAssertEqual(Set(m.keys), [0, 1])
        XCTAssertEqual(m[0] ?? .nan, sig(2), accuracy: 1e-4)
        XCTAssertEqual(m[1] ?? .nan, sig(1), accuracy: 1e-4)
    }

    func testNormTopKProbRenormalizesUnbiasedWeights() throws {
        let block = try makeBlock(useExpertBias: false, normTopkProb: true)
        let m = routed(block)

        XCTAssertEqual(Set(m.keys), [0, 1])
        let denom = sig(2) + sig(1)
        XCTAssertEqual(m[0] ?? .nan, sig(2) / denom, accuracy: 1e-4)
        XCTAssertEqual(m[1] ?? .nan, sig(1) / denom, accuracy: 1e-4)
        XCTAssertEqual((m[0] ?? 0) + (m[1] ?? 0), 1, accuracy: 1e-4)
    }

    func testConfigurationValidationRejectsImpossibleExpertRouting() throws {
        let config = try makeHybridConfig(numExpertsPerToken: 5)
        XCTAssertThrowsError(try config.validateModelConfiguration())
    }

    func testConfigurationRejectsConflictingLayerLayoutDeclarations() throws {
        let json = """
            {
                "model_type": "lfm2_moe",
                "vocab_size": 32,
                "hidden_size": 8,
                "intermediate_size": 16,
                "moe_intermediate_size": 16,
                "num_hidden_layers": 2,
                "num_experts": 4,
                "num_experts_per_tok": 2,
                "norm_topk_prob": true,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "max_position_embeddings": 128,
                "use_expert_bias": true,
                "num_dense_layers": 1,
                "norm_eps": 1e-5,
                "conv_bias": false,
                "conv_L_cache": 3,
                "full_attn_idxs": [0],
                "layer_types": ["conv", "full_attention"]
            }
            """
        let configuration = try JSONDecoder().decode(
            LFM2MoEConfiguration.self, from: Data(json.utf8))

        XCTAssertThrowsError(try configuration.validateModelConfiguration())
    }

    func testHybridCacheAndWarmContinuationMatchColdPrefill() throws {
        MLXRandom.seed(41)
        let model = LFM2MoEModel(try makeHybridConfig())
        let prefix = MLXArray([Int32(1), 2, 3, 4]).reshaped(1, 4)
        let suffix = MLXArray([Int32(5), 6]).reshaped(1, 2)

        let cold = model(
            concatenated([prefix, suffix], axis: 1),
            cache: try model.newCache(parameters: nil))[0..., (-2)..., 0...]
        let warmCache = try model.newCache(parameters: nil)
        _ = model(prefix, cache: warmCache)
        let warm = model(suffix, cache: warmCache)
        eval(cold, warm)

        XCTAssertLessThanOrEqual(abs(cold - warm).max().item(Float.self), 1e-4)
        XCTAssertTrue(warmCache[0] is RewindableConvolutionCache)
        XCTAssertTrue(warmCache[1] is KVCacheSimple)
        XCTAssertEqual(warmCache[1].offset, 6)
    }

    func testNoCacheForwardIsCausal() throws {
        MLXRandom.seed(43)
        let model = LFM2MoEModel(try makeHybridConfig())
        let first = MLXArray([Int32(1), 2, 3, 4]).reshaped(1, 4)
        let second = MLXArray([Int32(1), 2, 3, 9]).reshaped(1, 4)

        let firstLogits = model(first, cache: nil)[0..., ..<3, 0...]
        let secondLogits = model(second, cache: nil)[0..., ..<3, 0...]
        eval(firstLogits, secondLogits)

        XCTAssertLessThanOrEqual(
            abs(firstLogits - secondLogits).max().item(Float.self), 1e-5)
    }

    func testRaggedBatchStoresEachRowsLogicalConvolutionEndpoint() throws {
        let config = try makeHybridConfig()
        let conv = LFM2MoEShortConv(config, layerIdx: 0)
        let cache = MambaCache()
        let input = MLXArray((0 ..< 64).map(Float.init)).reshaped(2, 4, 8)
        cache.prepare(lengths: [4, 2])

        _ = conv(input, mask: cache.makeMask(N: 4), cache: cache)
        let state = try XCTUnwrap(cache[0])
        eval(state)

        XCTAssertEqual(state.shape, [2, 2, 8])
        XCTAssertEqual(cache.currentLengths?.asArray(Int.self), [0, -2])
    }

    func testAttentionCacheHonorsLongContextMemoryControls() throws {
        let model = LFM2MoEModel(try makeHybridConfig())

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
        MLXRandom.seed(47)
        let model = LFM2MoEModel(try makeHybridConfig())
        let prefix = MLXArray([Int32(1), 2, 3, 4]).reshaped(1, 4)
        let discarded = MLXArray([Int32(5), 6]).reshaped(1, 2)
        let replacement = MLXArray([Int32(7), 8]).reshaped(1, 2)

        let rewoundCache = try model.newCache(parameters: nil)
        _ = model(prefix, cache: rewoundCache)
        _ = model(discarded, cache: rewoundCache)
        XCTAssertEqual(trimPromptCache(rewoundCache, numTokens: 2), 2)
        let rewound = model(replacement, cache: rewoundCache)

        let cold = model(
            concatenated([prefix, replacement], axis: 1),
            cache: try model.newCache(parameters: nil))[0..., (-2)..., 0...]
        eval(rewound, cold)

        XCTAssertLessThanOrEqual(abs(rewound - cold).max().item(Float.self), 1e-4)
    }

    func testCommunityNamespaceLoadsStrictlyAndNativeNamespaceIsPreserved() throws {
        let model = LFM2MoEModel(try makeHybridConfig())
        let communityWeights = Dictionary(
            uniqueKeysWithValues: model.parameters().flattened().map { name, value in
                ("language_model.\(name)", value)
            })
        let sanitized = model.sanitize(weights: communityWeights)

        XCTAssertNotNil(sanitized["model.layers.0.conv.in_proj.weight"])
        try model.update(parameters: ModuleParameters.unflattened(sanitized), verify: [.all])

        let native = [
            "model.layers.0.conv.in_proj.weight": MLXArray.zeros([24, 8]),
            "language_model.metadata": MLXArray([Float(1)]),
        ]
        let untouched = model.sanitize(weights: native)
        XCTAssertNotNil(untouched["model.layers.0.conv.in_proj.weight"])
        XCTAssertNotNil(untouched["language_model.metadata"])
    }

    /// Opt-in strict integration test for the released mixed-precision model.
    /// Factory loading verifies every checkpoint tensor, then this asserts that
    /// the checkpoint's 8-bit router override coexists with its 4-bit modules.
    func testDownloadedCheckpointLoadsWithMixedPrecisionRouting() async throws {
        guard let path = ProcessInfo.processInfo.environment["MLX_LFM2_MOE_MODEL_PATH"] else {
            throw XCTSkip("Set MLX_LFM2_MOE_MODEL_PATH to run the real-checkpoint test.")
        }
        let directory = URL(filePath: path, directoryHint: .isDirectory)
        let context = try await LLMModelFactory.shared.load(
            from: directory, using: TokenizerLoaderStub())
        let model = try XCTUnwrap(context.model as? LFM2MoEModel)
        let modules = Dictionary(uniqueKeysWithValues: model.leafModules().flattened())
        let router = try XCTUnwrap(
            modules["model.layers.2.feed_forward.gate"] as? QuantizedLinear)
        let convolution = try XCTUnwrap(
            modules["model.layers.0.conv.in_proj"] as? QuantizedLinear)

        XCTAssertEqual(router.bits, 8)
        XCTAssertEqual(router.groupSize, 64)
        XCTAssertEqual(convolution.bits, 4)
        XCTAssertEqual(convolution.groupSize, 64)
        XCTAssertEqual(context.configuration.toolCallFormat, .lfm2)
        XCTAssertEqual(context.configuration.reasoningConfig?.promptStrategy, .alwaysOn)
        XCTAssertEqual(context.configuration.reasoningConfig?.isSpecialToken, false)

        let cache = try model.newCache(parameters: nil)
        let logits = model(MLXArray([Int32(1), 2, 3]).reshaped(1, 3), cache: cache)
        eval(logits)
        XCTAssertEqual(logits.shape, [1, 3, 128_000])
        XCTAssertEqual(cache.count, 24)
    }

    /// Opt-in device benchmark of model forward throughput. Tokenization and UI
    /// work are intentionally excluded so this measures the MLX integration.
    func testDownloadedCheckpointPerformance() throws {
        guard ProcessInfo.processInfo.environment["MLX_LFM2_MOE_BENCHMARK"] == "1" else {
            throw XCTSkip("Set MLX_LFM2_MOE_BENCHMARK=1 to run the device benchmark.")
        }
        guard let path = ProcessInfo.processInfo.environment["MLX_LFM2_MOE_MODEL_PATH"] else {
            throw XCTSkip("Set MLX_LFM2_MOE_MODEL_PATH to run the device benchmark.")
        }
        let directory = URL(filePath: path, directoryHint: .isDirectory)
        let configData = try Data(contentsOf: directory.appending(component: "config.json"))
        let config = try JSONDecoder().decode(LFM2MoEConfiguration.self, from: configData)
        let baseConfig = try JSONDecoder.json5().decode(BaseConfiguration.self, from: configData)
        let model = LFM2MoEModel(config)
        try loadWeights(
            modelDirectory: directory,
            model: model,
            perLayerQuantization: baseConfig.perLayerQuantization)

        let warmCache = try model.newCache(parameters: nil)
        eval(
            model(
                MLXArray([Int32](repeating: 42, count: 16)).reshaped(1, 16),
                cache: warmCache))
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
                format:
                    "LFM2.5-8B-A1B mixed 4/8-bit: prefill %.2f tok/s, decode %.2f tok/s, peak %.2f GiB",
                prefillTPS, decodeTPS, peakGiB))

        XCTAssertTrue(prefillTPS.isFinite && prefillTPS > 0)
        XCTAssertTrue(decodeTPS.isFinite && decodeTPS > 0)
    }
}
