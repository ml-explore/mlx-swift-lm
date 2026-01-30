// Copyright © 2025 Apple Inc.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import XCTest

public class NemotronHTests: XCTestCase {

    /// Create a minimal test configuration for NemotronH
    /// Uses small dimensions to keep tests fast
    private func makeTestConfig(pattern: String = "M*M-E") -> NemotronHConfiguration {
        NemotronHConfiguration(
            vocabSize: 100,
            hiddenSize: 64,
            numHiddenLayers: pattern.count,
            numAttentionHeads: 4,
            numKeyValueHeads: 2,
            mambaNumHeads: 4,
            mambaHeadDim: 16,
            ssmStateSize: 16,
            convKernel: 4,
            nGroups: 2,
            intermediateSize: 128,
            moeIntermediateSize: 64,
            moeSharedExpertIntermediateSize: 64,
            nRoutedExperts: 4,
            numExpertsPerTok: 2,
            hybridOverridePattern: pattern,
            layerNormEpsilon: 1e-5,
            nGroup: 2,
            topkGroup: 1
        )
    }

    // MARK: - Configuration Decoding Tests

    func testConfigurationDecodingFromJSON() throws {
        let json = """
            {
                "model_type": "nemotron_h",
                "vocab_size": 131072,
                "hidden_size": 4096,
                "num_hidden_layers": 32,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "mamba_num_heads": 64,
                "mamba_head_dim": 64,
                "ssm_state_size": 128,
                "conv_kernel": 4,
                "n_groups": 8,
                "intermediate_size": 16384,
                "moe_intermediate_size": 1024,
                "moe_shared_expert_intermediate_size": 8192,
                "n_routed_experts": 64,
                "num_experts_per_tok": 4,
                "hybrid_override_pattern": "M*M-E*",
                "layer_norm_epsilon": 1e-5,
                "n_group": 4,
                "topk_group": 2
            }
            """

        let config = try JSONDecoder().decode(
            NemotronHConfiguration.self, from: json.data(using: .utf8)!)

        XCTAssertEqual(config.vocabSize, 131072)
        XCTAssertEqual(config.hiddenSize, 4096)
        XCTAssertEqual(config.numHiddenLayers, 32)
        XCTAssertEqual(config.numAttentionHeads, 32)
        XCTAssertEqual(config.numKeyValueHeads, 8)
        XCTAssertEqual(config.mambaNumHeads, 64)
        XCTAssertEqual(config.mambaHeadDim, 64)
        XCTAssertEqual(config.ssmStateSize, 128)
        XCTAssertEqual(config.convKernel, 4)
        XCTAssertEqual(config.nGroups, 8)
        XCTAssertEqual(config.intermediateSize, 16384)
        XCTAssertEqual(config.moeIntermediateSize, 1024)
        XCTAssertEqual(config.nRoutedExperts, 64)
        XCTAssertEqual(config.numExpertsPerTok, 4)
        XCTAssertEqual(config.hybridOverridePattern, "M*M-E*")
        XCTAssertEqual(config.nGroup, 4)
        XCTAssertEqual(config.topkGroup, 2)
    }

    func testConfigurationDecodingWithArrayPattern() throws {
        // Some configs have hybrid_override_pattern as array of strings
        let json = """
            {
                "vocab_size": 100,
                "hidden_size": 64,
                "num_hidden_layers": 4,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "mamba_num_heads": 4,
                "mamba_head_dim": 16,
                "ssm_state_size": 16,
                "conv_kernel": 4,
                "n_groups": 2,
                "intermediate_size": 128,
                "moe_intermediate_size": 64,
                "moe_shared_expert_intermediate_size": 64,
                "n_routed_experts": 4,
                "num_experts_per_tok": 2,
                "hybrid_override_pattern": ["M", "*", "M", "-"]
            }
            """

        let config = try JSONDecoder().decode(
            NemotronHConfiguration.self, from: json.data(using: .utf8)!)

        XCTAssertEqual(config.hybridOverridePattern, "M*M-")
    }

    func testConfigurationDecodingWithTimeStepLimitArray() throws {
        // time_step_limit can be an array [min, max]
        let json = """
            {
                "vocab_size": 100,
                "hidden_size": 64,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "mamba_num_heads": 4,
                "mamba_head_dim": 16,
                "ssm_state_size": 16,
                "conv_kernel": 4,
                "n_groups": 2,
                "intermediate_size": 128,
                "moe_intermediate_size": 64,
                "moe_shared_expert_intermediate_size": 64,
                "n_routed_experts": 4,
                "num_experts_per_tok": 2,
                "hybrid_override_pattern": "M*",
                "time_step_limit_min": [0.0, 1000.0]
            }
            """

        let config = try JSONDecoder().decode(
            NemotronHConfiguration.self, from: json.data(using: .utf8)!)

        XCTAssertEqual(config.timeStepLimitMin, 0.0)
        XCTAssertEqual(config.timeStepLimitMax, 1000.0)
    }

    func testConfigurationDecodingWithDefaults() throws {
        // Minimal config - should use defaults for optional fields
        let json = """
            {
                "vocab_size": 100,
                "hidden_size": 64,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "mamba_num_heads": 4,
                "mamba_head_dim": 16,
                "ssm_state_size": 16,
                "conv_kernel": 4,
                "n_groups": 2,
                "intermediate_size": 128,
                "moe_intermediate_size": 64,
                "moe_shared_expert_intermediate_size": 64,
                "n_routed_experts": 4,
                "num_experts_per_tok": 2,
                "hybrid_override_pattern": "M*"
            }
            """

        let config = try JSONDecoder().decode(
            NemotronHConfiguration.self, from: json.data(using: .utf8)!)

        // Check defaults
        XCTAssertEqual(config.attentionBias, false)
        XCTAssertEqual(config.mambaProjBias, false)
        XCTAssertEqual(config.mlpBias, false)
        XCTAssertEqual(config.useConvBias, true)
        XCTAssertEqual(config.tieWordEmbeddings, false)
        XCTAssertEqual(config.layerNormEpsilon, 1e-5)
        XCTAssertEqual(config.ropeTheta, 10000.0)
        XCTAssertEqual(config.nGroup, 1)
        XCTAssertEqual(config.topkGroup, 1)
        XCTAssertEqual(config.normTopkProb, true)
        XCTAssertEqual(config.routedScalingFactor, 1.0)
    }

    // MARK: - Weight Sanitization Tests

    func testSanitizeConv1dWeights() throws {
        let config = makeTestConfig(pattern: "M*")
        let model = NemotronHModel(config)

        // The sanitization swaps axes 1 and 2 when dim(-1) != 1
        // Python format comes in as [convDim, inputChannels, kernelSize]
        // Swift expects: [convDim, kernelSize, inputChannels]
        let convDim =
            config.mambaNumHeads * config.mambaHeadDim + 2 * config.nGroups * config.ssmStateSize
        // Create weight with shape [convDim, 1, kernelSize] - this has dim(-1) = kernelSize != 1
        let mockConvWeight = MLXArray.ones([convDim, 1, config.convKernel])

        var weights = [String: MLXArray]()
        weights["backbone.layers.0.mixer.conv1d.weight"] = mockConvWeight

        let sanitized = model.sanitize(weights: weights)

        // After swapping axes 1 and 2: [convDim, kernelSize, 1]
        let sanitizedConv = sanitized["backbone.layers.0.mixer.conv1d.weight"]!
        XCTAssertEqual(sanitizedConv.shape, [convDim, config.convKernel, 1])
    }

    func testSanitizeConv1dWeightsNoOpWhenAlreadyCorrect() throws {
        let config = makeTestConfig(pattern: "M*")
        let model = NemotronHModel(config)

        // When dim(-1) == 1, no transpose needed
        let convDim =
            config.mambaNumHeads * config.mambaHeadDim + 2 * config.nGroups * config.ssmStateSize
        let mockConvWeight = MLXArray.ones([convDim, config.convKernel, 1])

        var weights = [String: MLXArray]()
        weights["backbone.layers.0.mixer.conv1d.weight"] = mockConvWeight

        let sanitized = model.sanitize(weights: weights)

        // Should remain unchanged [convDim, kernelSize, 1]
        let sanitizedConv = sanitized["backbone.layers.0.mixer.conv1d.weight"]!
        XCTAssertEqual(sanitizedConv.shape, [convDim, config.convKernel, 1])
    }

    func testSanitizeExpertWeights() throws {
        let config = makeTestConfig(pattern: "E")
        let model = NemotronHModel(config)

        // Create mock expert weights that need stacking
        var weights = [String: MLXArray]()
        for e in 0 ..< config.nRoutedExperts {
            weights["backbone.layers.0.mixer.experts.\(e).up_proj.weight"] =
                MLXArray.ones([config.moeIntermediateSize, config.hiddenSize])
            weights["backbone.layers.0.mixer.experts.\(e).down_proj.weight"] =
                MLXArray.ones([config.hiddenSize, config.moeIntermediateSize])
        }

        let sanitized = model.sanitize(weights: weights)

        // Experts should be stacked into switch_mlp format
        let stackedFc1 = sanitized["backbone.layers.0.mixer.switch_mlp.fc1.weight"]
        let stackedFc2 = sanitized["backbone.layers.0.mixer.switch_mlp.fc2.weight"]

        XCTAssertNotNil(stackedFc1)
        XCTAssertNotNil(stackedFc2)
        XCTAssertEqual(
            stackedFc1!.shape,
            [config.nRoutedExperts, config.moeIntermediateSize, config.hiddenSize])
        XCTAssertEqual(
            stackedFc2!.shape,
            [config.nRoutedExperts, config.hiddenSize, config.moeIntermediateSize])

        // Original expert keys should be removed
        XCTAssertNil(sanitized["backbone.layers.0.mixer.experts.0.up_proj.weight"])
    }

    func testSanitizePreservesOtherWeights() throws {
        let config = makeTestConfig(pattern: "M*")
        let model = NemotronHModel(config)

        var weights = [String: MLXArray]()
        weights["backbone.embeddings.weight"] = MLXArray.ones([config.vocabSize, config.hiddenSize])
        weights["backbone.norm_f.weight"] = MLXArray.ones([config.hiddenSize])

        let sanitized = model.sanitize(weights: weights)

        XCTAssertNotNil(sanitized["backbone.embeddings.weight"])
        XCTAssertNotNil(sanitized["backbone.norm_f.weight"])
        XCTAssertEqual(
            sanitized["backbone.embeddings.weight"]!.shape, [config.vocabSize, config.hiddenSize])
    }

    // MARK: - Basic Forward Pass Tests

    func testNemotronHForwardPass() throws {
        let config = makeTestConfig(pattern: "M*")
        let model = NemotronHModel(config)

        let input = MLXArray([1, 2, 3, 4, 5])[.newAxis, .ellipsis]
        let output = model.callAsFunction(input, cache: nil)

        XCTAssertEqual(output.shape, [1, 5, 100])
    }

    func testNemotronHWithMambaOnly() throws {
        let config = makeTestConfig(pattern: "MMM")
        let model = NemotronHModel(config)

        let input = MLXArray([1, 2, 3])[.newAxis, .ellipsis]
        let output = model.callAsFunction(input, cache: nil)

        XCTAssertEqual(output.shape, [1, 3, 100])
    }

    func testNemotronHWithAttentionOnly() throws {
        let config = makeTestConfig(pattern: "***")
        let model = NemotronHModel(config)

        let input = MLXArray([1, 2, 3])[.newAxis, .ellipsis]
        let output = model.callAsFunction(input, cache: nil)

        XCTAssertEqual(output.shape, [1, 3, 100])
    }

    func testNemotronHWithMLP() throws {
        let config = makeTestConfig(pattern: "M-*")
        let model = NemotronHModel(config)

        let input = MLXArray([1, 2, 3])[.newAxis, .ellipsis]
        let output = model.callAsFunction(input, cache: nil)

        XCTAssertEqual(output.shape, [1, 3, 100])
    }

    func testNemotronHWithMoE() throws {
        let config = makeTestConfig(pattern: "ME*")
        let model = NemotronHModel(config)

        let input = MLXArray([1, 2, 3])[.newAxis, .ellipsis]
        let output = model.callAsFunction(input, cache: nil)

        XCTAssertEqual(output.shape, [1, 3, 100])
    }

    func testNemotronHFullPattern() throws {
        // Test a pattern with all block types
        let config = makeTestConfig(pattern: "M-E*M-E*")
        let model = NemotronHModel(config)

        let input = MLXArray([1, 2, 3, 4])[.newAxis, .ellipsis]
        let output = model.callAsFunction(input, cache: nil)

        XCTAssertEqual(output.shape, [1, 4, 100])
    }

    // MARK: - Cache Tests

    func testNemotronHCacheCreation() throws {
        // Pattern: M*M- has 2 Mamba + 1 Attention = 3 caches
        let config = makeTestConfig(pattern: "M*M-")
        let model = NemotronHModel(config)

        let cache = model.newCache(parameters: nil)

        // Only Mamba (M) and Attention (*) layers have caches
        // Pattern M*M- has M, *, M = 3 cacheable layers
        XCTAssertEqual(cache.count, 3)
    }

    func testNemotronHCacheCountMambaOnly() throws {
        let config = makeTestConfig(pattern: "MMM")
        let model = NemotronHModel(config)

        let cache = model.newCache(parameters: nil)

        // 3 Mamba layers = 3 caches
        XCTAssertEqual(cache.count, 3)
    }

    func testNemotronHCacheCountAttentionOnly() throws {
        let config = makeTestConfig(pattern: "***")
        let model = NemotronHModel(config)

        let cache = model.newCache(parameters: nil)

        // 3 Attention layers = 3 caches
        XCTAssertEqual(cache.count, 3)
    }

    func testNemotronHCacheCountMixed() throws {
        // Pattern with MLP (-) and MoE (E) which don't have caches
        let config = makeTestConfig(pattern: "M-E*-E")
        let model = NemotronHModel(config)

        let cache = model.newCache(parameters: nil)

        // Only M and * have caches: M, * = 2 caches
        XCTAssertEqual(cache.count, 2)
    }

    // MARK: - Incremental Generation Tests

    func testNemotronHIncrementalGeneration() throws {
        let config = makeTestConfig(pattern: "M*")
        let model = NemotronHModel(config)

        // First pass - process prompt
        let prompt = MLXArray([1, 2, 3, 4, 5])[.newAxis, .ellipsis]
        let cache = model.newCache(parameters: nil)
        let promptOutput = model.callAsFunction(prompt, cache: cache)

        XCTAssertEqual(promptOutput.shape, [1, 5, 100])

        // Second pass - generate next token
        let nextToken = MLXArray([6])[.newAxis, .ellipsis]
        let nextOutput = model.callAsFunction(nextToken, cache: cache)

        XCTAssertEqual(nextOutput.shape, [1, 1, 100])
    }

    // MARK: - KV Heads Tests

    func testNemotronHKVHeads() throws {
        let config = makeTestConfig(pattern: "M*M*")
        let model = NemotronHModel(config)

        // kvHeads should have entries for Mamba (0) and Attention (numKeyValueHeads)
        // Pattern M*M* = [0, 2, 0, 2] where 2 is numKeyValueHeads
        XCTAssertEqual(model.kvHeads.count, 4)
        XCTAssertEqual(model.kvHeads[0], 0)  // Mamba
        XCTAssertEqual(model.kvHeads[1], 2)  // Attention
        XCTAssertEqual(model.kvHeads[2], 0)  // Mamba
        XCTAssertEqual(model.kvHeads[3], 2)  // Attention
    }

    // MARK: - Vocabulary Size Tests

    func testNemotronHVocabularySize() throws {
        let config = makeTestConfig(pattern: "M*")
        let model = NemotronHModel(config)

        XCTAssertEqual(model.vocabularySize, 100)
    }

    // MARK: - Batch Processing Tests

    func testNemotronHBatchProcessing() throws {
        let config = makeTestConfig(pattern: "M*")
        let model = NemotronHModel(config)

        // Batch of 2 sequences - use reshaped to create 2D input
        let flat = MLXArray([1, 2, 3, 4, 5, 6])
        let input = flat.reshaped(2, 3)
        let output = model.callAsFunction(input, cache: nil)

        XCTAssertEqual(output.shape, [2, 3, 100])
    }

    // MARK: - Tied Embeddings Test

    func testNemotronHTiedEmbeddings() throws {
        let config = NemotronHConfiguration(
            vocabSize: 100,
            hiddenSize: 64,
            numHiddenLayers: 2,
            numAttentionHeads: 4,
            numKeyValueHeads: 2,
            mambaNumHeads: 4,
            mambaHeadDim: 16,
            ssmStateSize: 16,
            convKernel: 4,
            nGroups: 2,
            intermediateSize: 128,
            moeIntermediateSize: 64,
            moeSharedExpertIntermediateSize: 64,
            nRoutedExperts: 4,
            numExpertsPerTok: 2,
            hybridOverridePattern: "M*",
            tieWordEmbeddings: true
        )
        let model = NemotronHModel(config)

        let input = MLXArray([1, 2, 3])[.newAxis, .ellipsis]
        let output = model.callAsFunction(input, cache: nil)

        XCTAssertEqual(output.shape, [1, 3, 100])
    }

    func testNemotronHUntiedEmbeddings() throws {
        let config = NemotronHConfiguration(
            vocabSize: 100,
            hiddenSize: 64,
            numHiddenLayers: 2,
            numAttentionHeads: 4,
            numKeyValueHeads: 2,
            mambaNumHeads: 4,
            mambaHeadDim: 16,
            ssmStateSize: 16,
            convKernel: 4,
            nGroups: 2,
            intermediateSize: 128,
            moeIntermediateSize: 64,
            moeSharedExpertIntermediateSize: 64,
            nRoutedExperts: 4,
            numExpertsPerTok: 2,
            hybridOverridePattern: "M*",
            tieWordEmbeddings: false
        )
        let model = NemotronHModel(config)

        let input = MLXArray([1, 2, 3])[.newAxis, .ellipsis]
        let output = model.callAsFunction(input, cache: nil)

        XCTAssertEqual(output.shape, [1, 3, 100])
    }

    // MARK: - Shared Experts Test

    func testNemotronHWithSharedExperts() throws {
        let config = NemotronHConfiguration(
            vocabSize: 100,
            hiddenSize: 64,
            numHiddenLayers: 1,
            numAttentionHeads: 4,
            numKeyValueHeads: 2,
            mambaNumHeads: 4,
            mambaHeadDim: 16,
            ssmStateSize: 16,
            convKernel: 4,
            nGroups: 2,
            intermediateSize: 128,
            moeIntermediateSize: 64,
            moeSharedExpertIntermediateSize: 64,
            nRoutedExperts: 4,
            numExpertsPerTok: 2,
            hybridOverridePattern: "E",
            nSharedExperts: 1
        )
        let model = NemotronHModel(config)

        let input = MLXArray([1, 2, 3])[.newAxis, .ellipsis]
        let output = model.callAsFunction(input, cache: nil)

        XCTAssertEqual(output.shape, [1, 3, 100])
    }

    // MARK: - Cast Predicate Test

    func testCastPredicateExcludesSpecialParameters() throws {
        let config = makeTestConfig(pattern: "ME")
        let model = NemotronHModel(config)

        let castPredicate = model.castPredicate!

        // These should NOT be cast (return false)
        XCTAssertFalse(castPredicate("backbone.layers.0.mixer.e_score_correction_bias"))
        XCTAssertFalse(castPredicate("backbone.layers.0.mixer.A_log"))

        // Regular parameters should be cast (return true)
        XCTAssertTrue(castPredicate("backbone.layers.0.mixer.in_proj.weight"))
        XCTAssertTrue(castPredicate("backbone.embeddings.weight"))
        XCTAssertTrue(castPredicate("backbone.norm_f.weight"))
    }

    // MARK: - LoRA Layers Test

    func testNemotronHLoRALayers() throws {
        let config = makeTestConfig(pattern: "M*M-E")
        let model = NemotronHModel(config)

        // loraLayers should return backbone.layers
        XCTAssertEqual(model.loraLayers.count, 5)
    }

    // MARK: - Edge Cases

    func testNemotronHSingleMambaLayer() throws {
        let config = makeTestConfig(pattern: "M")
        let model = NemotronHModel(config)

        let input = MLXArray([1, 2, 3])[.newAxis, .ellipsis]
        let output = model.callAsFunction(input, cache: nil)

        XCTAssertEqual(output.shape, [1, 3, 100])
        XCTAssertEqual(model.kvHeads, [0])  // Mamba has 0 kv heads
    }

    func testNemotronHSingleAttentionLayer() throws {
        let config = makeTestConfig(pattern: "*")
        let model = NemotronHModel(config)

        let input = MLXArray([1, 2, 3])[.newAxis, .ellipsis]
        let output = model.callAsFunction(input, cache: nil)

        XCTAssertEqual(output.shape, [1, 3, 100])
        XCTAssertEqual(model.kvHeads, [2])  // numKeyValueHeads = 2
    }

    func testNemotronHLongSequence() throws {
        let config = makeTestConfig(pattern: "M*")
        let model = NemotronHModel(config)

        // Test with a longer sequence
        let input = MLXArray(0 ..< 128)[.newAxis, .ellipsis]
        let output = model.callAsFunction(input, cache: nil)

        XCTAssertEqual(output.shape, [1, 128, 100])
    }

    func testNemotronHMultipleGenerationSteps() throws {
        let config = makeTestConfig(pattern: "M*M*")
        let model = NemotronHModel(config)

        let cache = model.newCache(parameters: nil)

        // Initial prompt
        let prompt = MLXArray([1, 2, 3, 4, 5])[.newAxis, .ellipsis]
        let _ = model.callAsFunction(prompt, cache: cache)

        // Multiple generation steps
        for tokenId in 6 ..< 10 {
            let nextToken = MLXArray([tokenId])[.newAxis, .ellipsis]
            let output = model.callAsFunction(nextToken, cache: cache)
            XCTAssertEqual(output.shape, [1, 1, 100])
        }
    }

    // MARK: - Complex Pattern Tests

    func testNemotronHAlternatingPattern() throws {
        // Alternating Mamba and Attention layers
        let config = makeTestConfig(pattern: "M*M*M*M*")
        let model = NemotronHModel(config)

        let input = MLXArray([1, 2, 3])[.newAxis, .ellipsis]
        let output = model.callAsFunction(input, cache: nil)

        XCTAssertEqual(output.shape, [1, 3, 100])

        // kvHeads should alternate: [0, 2, 0, 2, 0, 2, 0, 2]
        XCTAssertEqual(model.kvHeads, [0, 2, 0, 2, 0, 2, 0, 2])
    }

    func testNemotronHMoEHeavyPattern() throws {
        // Pattern with multiple MoE layers
        let config = makeTestConfig(pattern: "MEE*EE")
        let model = NemotronHModel(config)

        let input = MLXArray([1, 2, 3])[.newAxis, .ellipsis]
        let output = model.callAsFunction(input, cache: nil)

        XCTAssertEqual(output.shape, [1, 3, 100])

        // Only M and * contribute to kvHeads
        XCTAssertEqual(model.kvHeads, [0, 2])
    }
}
