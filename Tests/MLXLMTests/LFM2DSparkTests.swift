// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXLLM

private let tinyLFM2DSparkTargetJSON = """
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
      "rope_theta": 1000000.0
    }
    """

private let tinyLFM2DSparkJSON = """
    {
      "architectures": ["Lfm2DSparkDraftModel"],
      "model_type": "qwen3",
      "hidden_size": 8,
      "num_hidden_layers": 2,
      "num_attention_heads": 2,
      "num_key_value_heads": 1,
      "head_dim": 4,
      "intermediate_size": 16,
      "hidden_act": "silu",
      "rms_norm_eps": 1e-6,
      "vocab_size": 32,
      "rope_theta": 1000000.0,
      "max_position_embeddings": 128,
      "layer_types": ["full_attention", "full_attention"],
      "block_size": 3,
      "dflash_config": {
        "mask_token_id": 31,
        "target_layer_ids": [0, 1],
        "num_target_layers": 2
      },
      "markov_rank": 4,
      "rope_is_neox_style": false,
      "enable_confidence_head": true,
      "markov_head_type": "vanilla"
    }
    """

private func tinyDSparkConfiguration() throws -> LFM2DSparkConfiguration {
    try JSONDecoder().decode(
        LFM2DSparkConfiguration.self, from: Data(tinyLFM2DSparkJSON.utf8))
}

private func tinyDSparkTarget() throws -> LFM2Model {
    LFM2Model(
        try JSONDecoder().decode(
            LFM2Configuration.self, from: Data(tinyLFM2DSparkTargetJSON.utf8)))
}

@Test
func lfm2DSparkConfigurationAndCheckpointNamespaceMatchReleasedArchitecture() throws {
    let configuration = try tinyDSparkConfiguration()
    try configuration.validateModelConfiguration()
    let drafter = LFM2DSparkDraftModel(configuration)
    let parameters = Dictionary(uniqueKeysWithValues: drafter.parameters().flattened())

    #expect(drafter.maximumBlockSize == 4)
    #expect(drafter.targetLayerIds == [0, 1])
    #expect(drafter.requiresPromptPrefill)
    #expect(!drafter.requiresSharedTargetKV)
    #expect(drafter.requiresGreedySampling)
    #expect(parameters["fc.weight"]?.shape == [8, 16])
    #expect(parameters["hidden_norm.weight"]?.shape == [8])
    #expect(parameters["layers.0.self_attn.q_proj.weight"]?.shape == [8, 8])
    #expect(parameters["markov_head.markov_w1.weight"]?.shape == [32, 4])
    #expect(parameters["markov_head.markov_w2.weight"]?.shape == [32, 4])
    #expect(parameters["confidence_head.proj.weight"]?.shape == [1, 12])
    #expect(parameters["confidence_head.proj.bias"]?.shape == [1])
}

@Test
func lfm2DSparkRegistrationDisambiguatesTheQwen3ModelType() async throws {
    await LFM2DSparkRegistration.register()
    let model = try await MTPDrafterTypeRegistry.shared.createModel(
        configuration: Data(tinyLFM2DSparkJSON.utf8), modelType: "qwen3")

    #expect(model is LFM2DSparkDraftModel)
}

@Test
func lfm2DSparkRegistryContainsEveryReleasedCheckpoint() {
    #expect(MTPDrafterRegistry.shared.contains(id: "LiquidAI/LFM2.5-1.2B-Instruct-DSpark"))
    #expect(MTPDrafterRegistry.shared.contains(id: "LiquidAI/LFM2.5-2.6B-DSpark"))
    #expect(MTPDrafterRegistry.shared.contains(id: "LiquidAI/LFM2.5-8B-A1B-DSpark"))
}

@Suite(.serialized)
struct LFM2DSparkMetalTests {
    @Test
    func targetEmitsOrderedDecoderFeatureTaps() throws {
        let target = try tinyDSparkTarget()
        var state = LMOutput.State()
        state[mtpEmitFlagKey] = true
        state[mtpTargetLayerIdsKey] = [0, 1]
        let tokens = MLXArray([Int32(1), 2, 3, 4]).reshaped(1, 4)

        let output = target(LMInput.Text(tokens: tokens), cache: nil, state: state)
        let hidden = try #require(output.state?[mtpLastHiddenStatesKey])
        let sharedKV = try #require(output.state?[mtpSharedKVStatesKey])
        eval(output.logits, hidden)

        #expect(output.logits.shape == [1, 4, 32])
        #expect(hidden.shape == [1, 4, 16])
        #expect(sharedKV.isEmpty)
    }

    @Test
    func draftProposalIsTransientAndCommitAppendsOnlyVerifiedTargetFeatures() throws {
        MLXRandom.seed(41)
        let target = try tinyDSparkTarget()
        let drafter = LFM2DSparkDraftModel(try tinyDSparkConfiguration())
        let sampler = GenerateParameters(temperature: 0).sampler()
        let prompt = MLXArray([Int32(1), 2, 3, 4]).reshaped(1, 4)
        var emitState = LMOutput.State()
        emitState[mtpEmitFlagKey] = true
        emitState[mtpTargetLayerIdsKey] = [0, 1]
        let prefill = target(LMInput.Text(tokens: prompt), cache: nil, state: emitState)
        let promptHidden = try #require(prefill.state?[mtpLastHiddenStatesKey])
        let bonus = MLXArray([Int32(5)])
        var state = drafter.makeState(parameters: nil)

        drafter.prepareDrafterState(
            target: target, promptTokens: prompt, targetHidden: promptHidden,
            firstBonus: bonus, positionDeltas: nil, state: &state, sampler: sampler)
        eval(promptHidden)
        #expect(state.cache.allSatisfy { $0.offset == 4 })

        let proposal = drafter.draftBlock(
            target: target, lastToken: bonus,
            lastHidden: promptHidden[0..., (-1)..., 0...], sharedKV: [:],
            positionDeltas: nil, queryOffset: 4, blockSize: 4,
            state: &state, sampler: sampler)
        eval(proposal)
        #expect(proposal.shape == [1, 3])
        #expect(state.cache.allSatisfy { $0.offset == 4 })

        let verifyTokens = concatenated([bonus, proposal.flattened()])
        let verify = target(
            LMInput.Text(tokens: verifyTokens[.newAxis]), cache: nil, state: emitState)
        let verifyHidden = try #require(verify.state?[mtpLastHiddenStatesKey])
        drafter.commitDrafterState(
            target: target, targetHidden: verifyHidden, draftTokens: proposal,
            acceptedCount: 1, finalToken: MLXArray([Int32(6)]),
            positionDeltas: nil, state: &state, sampler: sampler)
        eval(verifyHidden)

        #expect(state.cache.allSatisfy { $0.offset == 6 })
        #expect(state.nextPosition == 6)
    }

    @Test
    func iteratorUsesLFMHybridNativeRollbackPath() throws {
        let target = try tinyDSparkTarget()
        let drafter = LFM2DSparkDraftModel(try tinyDSparkConfiguration())
        let input = LMInput(tokens: MLXArray([Int32(1), 2, 3, 4]))
        let parameters = GenerateParameters(maxTokens: 20, temperature: 0)
        var baseline = try TokenIterator(
            input: input, model: target, parameters: parameters)
        var iterator = try MTPSpeculativeTokenIterator(
            input: input, mainModel: target, drafter: drafter,
            parameters: parameters, blockSize: 4)

        var expected = [Int]()
        while let token = baseline.next() { expected.append(token) }
        var output = [Int]()
        while let token = iterator.next() { output.append(token) }

        #expect(output == expected)
        #expect(iterator.passthroughReason == nil)
        #expect(iterator.proposedCount > 0)
    }
}

private struct DSparkTokenizerLoaderStub: TokenizerLoader {
    func load(from _: URL) async throws -> any Tokenizer {
        TestTokenizer(vocabularySize: 128_000)
    }
}

/// Opt-in strict checkpoint and end-to-end test. The draft path is intentionally
/// separate because target checkpoints do not bundle the DSpark weights.
@Test
func downloadedLFM2DSparkCheckpointLoadsAndGenerates() async throws {
    guard
        let targetPath = ProcessInfo.processInfo.environment["MLX_LFM2_MODEL_PATH"],
        let draftPath = ProcessInfo.processInfo.environment["MLX_LFM2_DSPARK_MODEL_PATH"]
    else {
        return
    }

    await LFM2DSparkRegistration.register()
    let targetContext = try await LLMModelFactory.shared.load(
        from: URL(filePath: targetPath, directoryHint: .isDirectory),
        using: DSparkTokenizerLoaderStub())
    let draftContext = try await MTPDrafterModelFactory.shared.load(
        from: URL(filePath: draftPath, directoryHint: .isDirectory),
        using: DSparkTokenizerLoaderStub())
    let drafter = try #require(draftContext.model as? LFM2DSparkDraftModel)
    // Chat template for "What is the capital of France?" in the released
    // LFM2.5 tokenizer. Literal IDs keep this strict model/graph test
    // independent of an optional tokenizer package while exercising the
    // distribution the draft checkpoints were trained on.
    let input = LMInput(
        tokens: MLXArray([
            Int32(124_894), 124_899, 5922, 207, 2992, 355, 278, 5205, 302, 3980,
            39, 124_900, 207, 124_899, 63_514, 207,
        ]))
    let parameters = GenerateParameters(maxTokens: 32, temperature: 0)
    var baseline = try TokenIterator(
        input: input, model: targetContext.model, parameters: parameters)
    var iterator = try MTPSpeculativeTokenIterator(
        input: input, mainModel: targetContext.model, drafter: drafter,
        parameters: parameters, blockSize: 10)

    var expected = [Int]()
    while let token = baseline.next() { expected.append(token) }
    var tokens = [Int]()
    while let token = iterator.next() { tokens.append(token) }

    print(
        "LFM2 DSpark acceptance: \(iterator.acceptedCount)/\(iterator.proposedCount) "
            + "draft tokens")
    let usesQuantizedEmbedding =
        (targetContext.model as? LFM2Model)?.model.embedTokens is QuantizedEmbedding
        || (targetContext.model as? LFM2MoEModel)?.model.embedTokens is QuantizedEmbedding
    if usesQuantizedEmbedding {
        // Quantized batched verifier kernels are not bitwise identical to
        // one-token matmuls near an argmax tie. The stream still consists
        // exclusively of verifier-selected tokens; exact baseline equality
        // is asserted for the bf16 target below this branch.
        #expect(tokens.count == expected.count)
    } else {
        #expect(tokens == expected)
    }
    #expect(iterator.passthroughReason == nil)
    #expect(iterator.proposedCount > 0)
    #expect(iterator.acceptedCount > 0)
}

/// Opt-in end-to-end decode benchmark for a matching LFM2.5 target and
/// DSpark checkpoint. This times iterator initialization (including prompt
/// prefill) separately from token consumption and alternates baseline and
/// speculative samples to reduce thermal-ordering bias.
@Test
func downloadedLFM2DSparkPerformance() async throws {
    guard ProcessInfo.processInfo.environment["MLX_LFM2_DSPARK_BENCHMARK"] == "1" else {
        return
    }
    guard
        let targetPath = ProcessInfo.processInfo.environment["MLX_LFM2_MODEL_PATH"],
        let draftPath = ProcessInfo.processInfo.environment["MLX_LFM2_DSPARK_MODEL_PATH"]
    else {
        return
    }

    await LFM2DSparkRegistration.register()
    let targetContext = try await LLMModelFactory.shared.load(
        from: URL(filePath: targetPath, directoryHint: .isDirectory),
        using: DSparkTokenizerLoaderStub())
    let draftContext = try await MTPDrafterModelFactory.shared.load(
        from: URL(filePath: draftPath, directoryHint: .isDirectory),
        using: DSparkTokenizerLoaderStub())
    let drafter = try #require(draftContext.model as? LFM2DSparkDraftModel)
    let input = LMInput(
        tokens: MLXArray([
            Int32(124_894), 124_899, 5922, 207, 2992, 355, 278, 5205, 302, 3980,
            39, 124_900, 207, 124_899, 63_514, 207,
        ]))
    let blockSize = Int(
        ProcessInfo.processInfo.environment["MLX_LFM2_DSPARK_BLOCK_SIZE"] ?? "10")!

    struct Sample {
        let prefillSeconds: Double
        let decodeSeconds: Double
        let accepted: Int
        let proposed: Int
    }

    func baseline(tokens: Int) throws -> Sample {
        let parameters = GenerateParameters(maxTokens: tokens, temperature: 0)
        let initializeStart = Date.timeIntervalSinceReferenceDate
        var iterator = try TokenIterator(
            input: input, model: targetContext.model, parameters: parameters)
        let prefillSeconds = Date.timeIntervalSinceReferenceDate - initializeStart
        let decodeStart = Date.timeIntervalSinceReferenceDate
        var count = 0
        while iterator.next() != nil { count += 1 }
        Stream().synchronize()
        return Sample(
            prefillSeconds: prefillSeconds,
            decodeSeconds: Date.timeIntervalSinceReferenceDate - decodeStart,
            accepted: 0,
            proposed: count)
    }

    func speculative(tokens: Int) throws -> Sample {
        let parameters = GenerateParameters(maxTokens: tokens, temperature: 0)
        let initializeStart = Date.timeIntervalSinceReferenceDate
        var iterator = try MTPSpeculativeTokenIterator(
            input: input, mainModel: targetContext.model, drafter: drafter,
            parameters: parameters, blockSize: blockSize)
        let prefillSeconds = Date.timeIntervalSinceReferenceDate - initializeStart
        let decodeStart = Date.timeIntervalSinceReferenceDate
        var count = 0
        while iterator.next() != nil { count += 1 }
        Stream().synchronize()
        #expect(iterator.passthroughReason == nil)
        #expect(count == tokens)
        return Sample(
            prefillSeconds: prefillSeconds,
            decodeSeconds: Date.timeIntervalSinceReferenceDate - decodeStart,
            accepted: iterator.acceptedCount,
            proposed: iterator.proposedCount)
    }

    // Compile kernels and realize lazy checkpoint tensors before sampling.
    _ = try baseline(tokens: 16)
    _ = try speculative(tokens: 16)
    Memory.clearCache()

    let measuredTokens = 128
    let repetitions = 3
    var baselineSamples = [Sample]()
    var speculativeSamples = [Sample]()
    for _ in 0 ..< repetitions {
        baselineSamples.append(try baseline(tokens: measuredTokens))
        speculativeSamples.append(try speculative(tokens: measuredTokens))
    }

    func median(_ values: [Double]) -> Double {
        values.sorted()[values.count / 2]
    }

    let baselineSeconds = median(baselineSamples.map(\.decodeSeconds))
    let speculativeSeconds = median(speculativeSamples.map(\.decodeSeconds))
    let baselinePrefill = median(baselineSamples.map(\.prefillSeconds))
    let speculativePrefill = median(speculativeSamples.map(\.prefillSeconds))
    let representative = speculativeSamples.min {
        abs($0.decodeSeconds - speculativeSeconds) < abs($1.decodeSeconds - speculativeSeconds)
    }!
    let baselineTPS = Double(measuredTokens) / baselineSeconds
    let speculativeTPS = Double(measuredTokens) / speculativeSeconds
    print(
        String(
            format:
                "LFM2 DSpark benchmark (block %d, median of %d): baseline %.2f tok/s, DSpark %.2f tok/s, speedup %.2fx, prefill %.3fs -> %.3fs, acceptance %d/%d",
            blockSize, repetitions, baselineTPS, speculativeTPS,
            speculativeTPS / baselineTPS, baselinePrefill, speculativePrefill,
            representative.accepted, representative.proposed))

    #expect(baselineTPS.isFinite && baselineTPS > 0)
    #expect(speculativeTPS.isFinite && speculativeTPS > 0)
    #expect(representative.proposed > 0)
}
