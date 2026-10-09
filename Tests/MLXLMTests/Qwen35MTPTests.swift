import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXLMCommon
@testable import MLXVLM

@Test
func testQwen35TextConfigurationDecodesMTPFields() throws {
    let cfg = try JSONDecoder().decode(
        MLXLLM.Qwen35TextConfiguration.self,
        from: Data(qwen35TextConfigJSON(mtpLayers: 1).utf8))

    #expect(cfg.mtpNumHiddenLayers == 1)
    #expect(cfg.mtpUseDedicatedEmbeddings == false)
}

@Test
func testQwen35VLMTextConfigurationDecodesMTPFields() throws {
    let cfg = try JSONDecoder().decode(
        MLXVLM.Qwen35Configuration.TextConfiguration.self,
        from: Data(qwen35TextConfigJSON(mtpLayers: 1).utf8))

    #expect(cfg.mtpNumHiddenLayers == 1)
    #expect(cfg.mtpUseDedicatedEmbeddings == false)
}

@Test
func testQwen35MTPDraftSanitizeKeepsAndShiftsMTPNorms() throws {
    let cfg = try JSONDecoder().decode(
        MLXLLM.Qwen35TextConfiguration.self,
        from: Data(qwen35TextConfigJSON(mtpLayers: 1).utf8))
    let drafter = MLXLLM.Qwen35MTPDraftModel(cfg)

    let sanitized = try drafter.sanitize(weights: [
        "mtp.norm.weight": MLXArray.zeros([16]),
        "mtp.pre_fc_norm_embedding.weight": MLXArray.zeros([16]),
        "mtp.layers.0.self_attn.q_proj.weight": MLXArray.zeros([32, 16]),
        "mtp.layers.0.mlp.experts.gate_up_proj": MLXArray.zeros([2, 32, 16]),
        "mtp.layers.0.mlp.experts.down_proj": MLXArray.zeros([2, 16, 16]),
        "model.embed_tokens.weight": MLXArray.zeros([16, 16]),
    ])

    #expect(sanitized["model.embed_tokens.weight"] == nil)
    #expect(sanitized["mtp.layers.0.self_attn.q_proj.weight"] != nil)
    #expect(sanitized["mtp.layers.0.mlp.experts.gate_up_proj"] == nil)
    #expect(sanitized["mtp.layers.0.mlp.experts.down_proj"] == nil)
    #expect(sanitized["mtp.layers.0.mlp.switch_mlp.gate_proj.weight"]?.shape == [2, 16, 16])
    #expect(sanitized["mtp.layers.0.mlp.switch_mlp.up_proj.weight"]?.shape == [2, 16, 16])
    #expect(sanitized["mtp.layers.0.mlp.switch_mlp.down_proj.weight"]?.shape == [2, 16, 16])
    let norm = try #require(sanitized["mtp.norm.weight"])
    let pre = try #require(sanitized["mtp.pre_fc_norm_embedding.weight"])
    eval(norm, pre)
    #expect(allClose(norm, MLXArray.ones([16]), rtol: 0, atol: 0).item(Bool.self))
    #expect(allClose(pre, MLXArray.ones([16]), rtol: 0, atol: 0).item(Bool.self))
}

@Test(arguments: [false, true])
func testQwen35MTPDraftSanitizeRejectsCompetingComponents(vision: Bool) throws {
    let data = Data(qwen35TextConfigJSON(mtpLayers: 1).utf8)
    let drafter: any BaseLanguageModel
    if vision {
        drafter = MLXVLM.Qwen35VLMNextNDraftModel(
            try JSONDecoder().decode(MLXVLM.Qwen35Configuration.TextConfiguration.self, from: data))
    } else {
        drafter = MLXLLM.Qwen35MTPDraftModel(
            try JSONDecoder().decode(MLXLLM.Qwen35TextConfiguration.self, from: data))
    }
    let weights = [
        "mtp.norm.weight": MLXArray.zeros([16]),
        "language_model.mtp.norm.weight": MLXArray.ones([16]),
    ]
    #expect(throws: CheckpointComponent.SelectionError.self) {
        try drafter.sanitize(weights: weights)
    }
    #expect(throws: CheckpointComponent.SelectionError.self) {
        try drafter.sanitize(weights: weights, metadata: [:])
    }
}

@Test
func testQwen35StandaloneMTPDoesNotDoubleShiftConvertedNorms() throws {
    let cfg = try JSONDecoder().decode(
        MLXLLM.Qwen35Configuration.self,
        from: Data(qwen35StandaloneMTPConfigJSON().utf8))
    let drafter = MLXLLM.Qwen35MTPDraftModel(cfg, preconvertedNorms: true)

    let weight = MLXArray.zeros([16])
    let sanitized = try drafter.sanitize(weights: ["mtp.norm.weight": weight])
    let norm = try #require(sanitized["mtp.norm.weight"])
    eval(norm)
    #expect(allClose(norm, weight, rtol: 0, atol: 0).item(Bool.self))
}

@Test
func testQwen35MTPDraftSanitizeStacksPerExpertMoEWeights() throws {
    let cfg = try JSONDecoder().decode(
        MLXLLM.Qwen35TextConfiguration.self,
        from: Data(qwen35TextConfigJSON(mtpLayers: 1, numExperts: 2).utf8))
    let drafter = MLXLLM.Qwen35MTPDraftModel(cfg)

    let weights: [String: MLXArray] = [
        "mtp.layers.0.mlp.experts.0.gate_proj.weight": MLXArray.zeros([16, 16]),
        "mtp.layers.0.mlp.experts.1.gate_proj.weight": MLXArray.ones([16, 16]),
        "mtp.layers.0.mlp.experts.0.up_proj.weight": MLXArray.zeros([16, 16]),
        "mtp.layers.0.mlp.experts.1.up_proj.weight": MLXArray.ones([16, 16]),
        "mtp.layers.0.mlp.experts.0.down_proj.weight": MLXArray.zeros([16, 16]),
        "mtp.layers.0.mlp.experts.1.down_proj.weight": MLXArray.ones([16, 16]),
    ]

    let sanitized = try drafter.sanitize(weights: weights)

    #expect(sanitized["mtp.layers.0.mlp.experts.0.gate_proj.weight"] == nil)
    #expect(sanitized["mtp.layers.0.mlp.switch_mlp.gate_proj.weight"]?.shape == [2, 16, 16])
    #expect(sanitized["mtp.layers.0.mlp.switch_mlp.up_proj.weight"]?.shape == [2, 16, 16])
    #expect(sanitized["mtp.layers.0.mlp.switch_mlp.down_proj.weight"]?.shape == [2, 16, 16])
}

@Test
func testQwen35MTPDraftInstantiatesDedicatedEmbeddingWhenConfigured() throws {
    let cfg = try JSONDecoder().decode(
        MLXLLM.Qwen35TextConfiguration.self,
        from: Data(
            qwen35TextConfigJSON(mtpLayers: 1, mtpUseDedicatedEmbeddings: true).utf8))
    let drafter = MLXLLM.Qwen35MTPDraftModel(cfg)

    #expect(drafter.mtp.embedTokens != nil)
    let sanitized = try drafter.sanitize(weights: [
        "mtp.embed_tokens.weight": MLXArray.zeros([16, 16]),
        "model.embed_tokens.weight": MLXArray.ones([16, 16]),
    ])
    #expect(sanitized["mtp.embed_tokens.weight"] != nil)
    #expect(sanitized["model.embed_tokens.weight"] == nil)
}

/// Bound for comparing a gated-delta-net state reached two ways. Wider on machines whose
/// float32 matmuls run as TF32; see ``MatmulPrecision``.
private let gdnStateTolerance = MatmulPrecision.tolerance(float32: 1e-5, reduced: 2e-3)

@Suite(.serialized)
struct Qwen35MTPMetalTests {
    @Test
    func testQwen35MTPPredictorAdvancesEveryLayerCachePerToken() throws {
        let cfg = try JSONDecoder().decode(
            MLXLLM.Qwen35TextConfiguration.self,
            from: Data(qwen35TextConfigJSON(mtpLayers: 2).utf8))
        let predictor = MLXLLM.Qwen35MTPPredictor(cfg)
        let cache = predictor.newCache()
        let embeds = MLXArray.zeros([1, 1, 16])
        let hidden = MLXArray.zeros([1, 1, 16])

        let first = predictor(
            inputsEmbeds: embeds, hiddenStates: hidden, cache: cache,
            positionOffset: 128)
        eval(first)
        #expect(cache[0].offset == 1)
        #expect(cache[1].offset == 1)

        let second = predictor(
            inputsEmbeds: embeds, hiddenStates: first, cache: cache,
            positionOffset: 129)
        eval(second)
        #expect(cache[0].offset == 2)
        #expect(cache[1].offset == 2)
    }

    @Test
    func testQwen35VLMMTPPositionIdsApplyMultimodalDelta() throws {
        let positionIds = MLXVLM.qwen35MTPPositionIds(
            offset: 10,
            batchSize: 2,
            positionDeltas: MLXArray([Int32(3), 5])
        )

        eval(positionIds)
        #expect(positionIds.shape == [3, 2, 1])
        #expect(positionIds[0, 0, 0].item(Int32.self) == 13)
        #expect(positionIds[1, 0, 0].item(Int32.self) == 13)
        #expect(positionIds[2, 0, 0].item(Int32.self) == 13)
        #expect(positionIds[0, 1, 0].item(Int32.self) == 15)
        #expect(positionIds[1, 1, 0].item(Int32.self) == 15)
        #expect(positionIds[2, 1, 0].item(Int32.self) == 15)
    }

    @Test
    func testQwen35VLMMTPPositionIdsRepeatAndTrimShortBatchDeltas() throws {
        let positionIds = MLXVLM.qwen35MTPPositionIds(
            offset: 10,
            batchSize: 4,
            positionDeltas: MLXArray([Int32(3), 5])
        )

        eval(positionIds)
        #expect(positionIds.shape == [3, 4, 1])
        #expect(positionIds[0, 0, 0].item(Int32.self) == 13)
        #expect(positionIds[0, 1, 0].item(Int32.self) == 15)
        #expect(positionIds[0, 2, 0].item(Int32.self) == 13)
        #expect(positionIds[0, 3, 0].item(Int32.self) == 15)
    }

    @Test
    func testQwen35TextModelEmitDrafterStateBySynthetic() throws {
        let cfg = try JSONDecoder().decode(
            MLXLLM.Qwen35TextConfiguration.self,
            from: Data(qwen35TextConfigJSON(mtpLayers: 1).utf8))
        let model = MLXLLM.Qwen35TextModel(cfg)
        let cache = try model.newCache(parameters: nil as GenerateParameters?)
        var state = LMOutput.State()
        state[mtpEmitFlagKey] = true

        let input = LMInput.Text(tokens: MLXArray([Int32(1), 2, 3, 4]).reshaped([1, 4]))
        let out = model(input, cache: cache, state: state)

        let hidden = try #require(out.state?[mtpLastHiddenStatesKey])
        let sharedKV = try #require(out.state?[mtpSharedKVStatesKey])
        let sharedKVOffsets = try #require(out.state?[mtpSharedKVOffsetsKey])
        eval(out.logits, hidden)
        #expect(out.logits.shape == [1, 4, 16])
        #expect(hidden.shape == [1, 4, 16])
        #expect(Set(sharedKV.keys) == ["full_attention"])
        #expect(sharedKVOffsets == ["full_attention": 4])
        let full = try #require(sharedKV["full_attention"])
        eval(full.0, full.1)
        #expect(full.0.shape.count == 4)
        #expect(full.1.shape.count == 4)
    }

    @Test
    func testQwen35TextModelEmitsPostFinalNormHiddenState() throws {
        let cfg = try JSONDecoder().decode(
            MLXLLM.Qwen35TextConfiguration.self,
            from: Data(qwen35TextConfigJSON(mtpLayers: 1).utf8))
        let model = MLXLLM.Qwen35TextModel(cfg)
        let tokens = MLXArray([Int32(1), 2, 3, 4]).reshaped([1, 4])
        let expected = model.model.forward(tokens, applyFinalNorm: false)
        let normalized = model.model.norm(expected)
        var state = LMOutput.State()
        state[mtpEmitFlagKey] = true

        let output = model(LMInput.Text(tokens: tokens), cache: nil, state: state)
        let emitted = try #require(output.state?[mtpLastHiddenStatesKey])
        eval(expected, normalized, emitted)

        #expect(allClose(emitted, normalized, rtol: 0, atol: 0).item(Bool.self))
        #expect(!allClose(emitted, expected, rtol: 0, atol: 0).item(Bool.self))
    }

    @Test
    func testQwen35VLMEmitsPostFinalNormHiddenState() throws {
        let cfg = try JSONDecoder().decode(
            MLXVLM.Qwen35Configuration.self,
            from: Data(qwen35VLMConfigJSON(mtpLayers: 1).utf8))
        let model = MLXVLM.Qwen35(cfg)
        let tokens = MLXArray([Int32(1), 2, 3, 4]).reshaped([1, 4])
        let base = MLXArray(0 ..< 4).asType(.int32).reshaped([1, 1, 4])
        let positionIds = broadcast(base, to: [3, 1, 4])
        let expected = model.languageModel.model(
            tokens, positionIds: positionIds, applyFinalNorm: false)
        let normalized = model.languageModel.model.norm(expected)
        var state = LMOutput.State()
        state[mtpEmitFlagKey] = true

        let output = model.languageModel(
            tokens, cache: nil, state: state, positionIds: positionIds)
        let emitted = try #require(output.state?[mtpLastHiddenStatesKey])
        eval(expected, normalized, emitted)

        #expect(allClose(emitted, normalized, rtol: 0, atol: 0).item(Bool.self))
        #expect(!allClose(emitted, expected, rtol: 0, atol: 0).item(Bool.self))
    }

    @Test
    func testQwen35DrafterCacheTracksVerifiedSequenceAcrossAcceptRejectPatterns() throws {
        let cfg = try JSONDecoder().decode(
            MLXLLM.Qwen35TextConfiguration.self,
            from: Data(qwen35TextConfigJSON(mtpLayers: 1).utf8))
        let target = MLXLLM.Qwen35TextModel(cfg)
        let drafter = MLXLLM.Qwen35MTPDraftModel(cfg)
        let sampler = GenerateParameters(temperature: 0).sampler()
        let prompt = MLXArray([Int32(1), 2, 3]).reshaped([1, 3])

        var targetState = LMOutput.State()
        targetState[mtpEmitFlagKey] = true
        let targetOutput = target(LMInput.Text(tokens: prompt), cache: nil, state: targetState)
        let promptHidden = try #require(targetOutput.state?[mtpLastHiddenStatesKey])

        for pattern in [[0, 0], [0, 1], [1, 0], [1, 1]] {
            var state = drafter.makeState(parameters: nil)
            var bonus = MLXArray([Int32(4)])
            drafter.prepareDrafterState(
                target: target, promptTokens: prompt, targetHidden: promptHidden,
                firstBonus: bonus, positionDeltas: nil, state: &state, sampler: sampler)
            eval(state.seedToken!, state.seedHidden!)
            #expect(state.cache.allSatisfy { $0.offset == 3 })
            #expect(state.nextPosition == 3)

            var expectedPosition = 3
            for accepted in pattern {
                let proposal = drafter.draftBlock(
                    target: target, lastToken: bonus,
                    lastHidden: promptHidden[0..., (-1)..., 0...], sharedKV: [:],
                    positionDeltas: nil, queryOffset: expectedPosition, blockSize: 2,
                    state: &state, sampler: sampler)
                eval(proposal)
                #expect(state.cache.allSatisfy { $0.offset == expectedPosition })

                let verifyHidden = MLXArray.zeros([1, 2, cfg.hiddenSize])
                let finalToken = MLXArray([Int32(8 + accepted)])
                drafter.commitDrafterState(
                    target: target, targetHidden: verifyHidden, draftTokens: proposal,
                    acceptedCount: accepted, finalToken: finalToken, positionDeltas: nil,
                    state: &state, sampler: sampler)
                eval(state.seedToken!, state.seedHidden!)
                expectedPosition += accepted + 1
                #expect(state.cache.allSatisfy { $0.offset == expectedPosition })
                #expect(state.nextPosition == expectedPosition)
                bonus = finalToken
            }
        }
    }

    @Test
    func testQwen35GDNCheckpointMatchesPrefixWithoutReplayingProjections() throws {
        let cfg = try JSONDecoder().decode(
            MLXLLM.Qwen35TextConfiguration.self,
            from: Data(qwen35TextConfigJSON(mtpLayers: 1).utf8))
        let layer = MLXLLM.Qwen35GatedDeltaNet(cfg)
        let input = MLXRandom.normal([1, 2, 16])
        let fullCache = MambaCache()
        let speculativeCache = MambaCache()
        let prefixCache = MambaCache()

        let full = layer(input, cache: fullCache)
        let speculative = layer(input, cache: speculativeCache, checkpointAfter: 1)
        _ = layer(input[0..., ..<1, 0...], cache: prefixCache)
        eval(full, speculative)

        #expect(allClose(speculative, full, rtol: 1e-5, atol: 1e-5).item(Bool.self))
        #expect(speculativeCache.hasSpeculativeCheckpoint)
        #expect(speculativeCache.restoreSpeculativeCheckpoint())

        let restored = speculativeCache.state
        let expected = prefixCache.state
        #expect(restored.count == expected.count)
        for (actual, reference) in zip(restored, expected) {
            eval(actual, reference)
            // The checkpoint and the prefix run reach the same state through
            // differently shaped matmuls, so the bound follows the machine's
            // float32 matmul precision; see MatmulPrecision.
            #expect(
                allClose(actual, reference, rtol: gdnStateTolerance, atol: gdnStateTolerance)
                    .item(Bool.self))
        }
    }

    @Test
    func testQwen35VLMGDNCheckpointMatchesPrefix() throws {
        let cfg = try JSONDecoder().decode(
            MLXVLM.Qwen35Configuration.TextConfiguration.self,
            from: Data(qwen35TextConfigJSON(mtpLayers: 1).utf8))
        let layer = MLXVLM.Qwen35Language.GatedDeltaNet(cfg)
        let input = MLXRandom.normal([1, 2, 16])
        let fullCache = MambaCache()
        let speculativeCache = MambaCache()
        let prefixCache = MambaCache()

        let full = layer(input, cache: fullCache)
        let speculative = layer(input, cache: speculativeCache, checkpointAfter: 1)
        _ = layer(input[0..., ..<1, 0...], cache: prefixCache)
        eval(full, speculative)

        #expect(allClose(speculative, full, rtol: 1e-5, atol: 1e-5).item(Bool.self))
        #expect(speculativeCache.restoreSpeculativeCheckpoint())
        for (actual, reference) in zip(speculativeCache.state, prefixCache.state) {
            eval(actual, reference)
            #expect(
                allClose(actual, reference, rtol: gdnStateTolerance, atol: gdnStateTolerance)
                    .item(Bool.self))
        }
    }

    @Test
    func testQwen35HybridCacheRewindRestoresAttentionAndRecurrentStateAtomically() {
        let attention = KVCacheSimple()
        let keys = MLXArray.zeros([1, 1, 3, 2])
        _ = attention.update(keys: keys, values: keys)

        let recurrent = MambaCache()
        let checkpointConv = MLXArray.ones([1, 1, 4])
        let checkpointState = MLXArray.ones([1, 2, 2, 2])
        recurrent.saveSpeculativeCheckpoint(
            convState: checkpointConv, recurrentState: checkpointState, advancedBy: 1)
        recurrent[0] = MLXArray.zeros([1, 1, 4])
        recurrent[1] = MLXArray.zeros([1, 2, 2, 2])

        let rewound = rewindSpeculativePromptCache([attention, recurrent], numTokens: 1)
        #expect(rewound == 1)
        #expect(attention.offset == 2)
        #expect(!recurrent.hasSpeculativeCheckpoint)

        let restored = recurrent.state
        #expect(restored.count == 2)
        eval(restored[0], restored[1])
        #expect(allClose(restored[0], checkpointConv, rtol: 0, atol: 0).item(Bool.self))
        #expect(allClose(restored[1], checkpointState, rtol: 0, atol: 0).item(Bool.self))
    }
}

@Suite(.serialized)
struct Qwen35MTPRegistrationTests {
    @Test
    func registrationsCreateTextAndVLMDrafters() async throws {
        await MLXLLM.Qwen35TextMTPRegistration.register()

        let textModel = try await MTPDrafterTypeRegistry.shared.createModel(
            configuration: Data(qwen35TextConfigJSON(mtpLayers: 1).utf8),
            modelType: "qwen3_5_text")
        #expect(textModel is MLXLLM.Qwen35MTPDraftModel)

        await MLXVLM.Qwen35VLMMTPRegistration.register()

        let wrappedTextModel = try await MTPDrafterTypeRegistry.shared.createModel(
            configuration: Data(qwen35WrappedTextConfigJSON(modelType: "qwen3_5").utf8),
            modelType: "qwen3_5")
        #expect(wrappedTextModel is MLXLLM.Qwen35MTPDraftModel)

        let vlmModel = try await MTPDrafterTypeRegistry.shared.createModel(
            configuration: Data(qwen35VLMConfigJSON(mtpLayers: 1).utf8),
            modelType: "qwen3_5")
        #expect(vlmModel is MLXVLM.Qwen35VLMNextNDraftModel)

        let standalone = try await MTPDrafterTypeRegistry.shared.createModel(
            configuration: Data(qwen35StandaloneMTPConfigJSON().utf8),
            modelType: "qwen3_5_mtp")
        #expect(standalone is MLXLLM.Qwen35MTPDraftModel)
        #expect(standalone.maximumBlockSize == 2)
        #expect(standalone.requiresPromptPrefill)
        #expect(!standalone.requiresSharedTargetKV)
        #expect(standalone.requiresGreedySampling)
    }

    @Test
    func registrationsAreOrderIndependentForSharedModelTypes() async throws {
        await MLXVLM.Qwen35VLMMTPRegistration.register()
        await MLXLLM.Qwen35TextMTPRegistration.register()

        let vlmModel = try await MTPDrafterTypeRegistry.shared.createModel(
            configuration: Data(qwen35VLMConfigJSON(mtpLayers: 1).utf8),
            modelType: "qwen3_5")
        #expect(vlmModel is MLXVLM.Qwen35VLMNextNDraftModel)

        let textModel = try await MTPDrafterTypeRegistry.shared.createModel(
            configuration: Data(qwen35WrappedTextConfigJSON(modelType: "qwen3_5").utf8),
            modelType: "qwen3_5")
        #expect(textModel is MLXLLM.Qwen35MTPDraftModel)
    }
}

private func qwen35TextConfigJSON(
    mtpLayers: Int,
    mtpUseDedicatedEmbeddings: Bool = false,
    numExperts: Int = 0
) -> String {
    """
    {
      "model_type": "qwen3_5_text",
      "hidden_size": 16,
      "num_hidden_layers": 1,
      "intermediate_size": 32,
      "num_attention_heads": 2,
      "num_key_value_heads": 1,
      "head_dim": 8,
      "linear_num_value_heads": 2,
      "linear_num_key_heads": 1,
      "linear_key_head_dim": 8,
      "linear_value_head_dim": 8,
      "linear_conv_kernel_dim": 2,
      "rms_norm_eps": 1e-6,
      "vocab_size": 16,
      "rope_theta": 100000.0,
      "partial_rotary_factor": 0.25,
      "max_position_embeddings": 64,
      "tie_word_embeddings": true,
      "attention_bias": false,
      "full_attention_interval": 1,
      "mtp_num_hidden_layers": \(mtpLayers),
      "mtp_use_dedicated_embeddings": \(mtpUseDedicatedEmbeddings),
      "num_experts": \(numExperts),
      "num_experts_per_tok": \(numExperts == 0 ? 0 : 1),
      "moe_intermediate_size": 16,
      "shared_expert_intermediate_size": 16,
      "rope_parameters": {
        "type": "default",
        "rope_theta": 100000.0,
        "partial_rotary_factor": 0.25
      }
    }
    """
}

private func qwen35VLMConfigJSON(mtpLayers: Int) -> String {
    """
    {
      "model_type": "qwen3_5",
      "text_config": \(qwen35TextConfigJSON(mtpLayers: mtpLayers)),
      "vision_config": {
        "model_type": "qwen3_5_vit",
        "depth": 1,
        "hidden_size": 16,
        "intermediate_size": 32,
        "out_hidden_size": 16,
        "num_heads": 2,
        "patch_size": 2,
        "spatial_merge_size": 1,
        "temporal_patch_size": 1,
        "num_position_embeddings": 16
      }
    }
    """
}

private func qwen35WrappedTextConfigJSON(modelType: String) -> String {
    """
    {
      "model_type": "\(modelType)",
      "text_config": \(qwen35TextConfigJSON(mtpLayers: 1))
    }
    """
}

private func qwen35StandaloneMTPConfigJSON() -> String {
    """
    {
      "model_type": "qwen3_5_mtp",
      "block_size": 3,
      "text_config": \(qwen35TextConfigJSON(mtpLayers: 1)),
      "tie_word_embeddings": true,
      "vision_config": {}
    }
    """
}

@Suite(.serialized)
struct Qwen35CheckpointLoadingTests {
    @Test(arguments: [false, true])
    func convertedEmbeddedFactoryPreservesMixedPrecision(vision: Bool) async throws {
        let fixture = try await makeFixture(
            vision: vision, standalone: false, prefix: "language_model.mtp.", bits: 4)
        defer { try? FileManager.default.removeItem(at: fixture.directory) }
        let context = try await fixture.factory.load(
            from: fixture.directory, using: UnusedTokenizerLoader())
        let fc = try #require(
            (context.model as? MLXLLM.Qwen35MTPDraftModel)?.mtp.fc as? QuantizedLinear
                ?? (context.model as? MLXVLM.Qwen35VLMNextNDraftModel)?.mtp.fc as? QuantizedLinear)
        #expect(fc.bits == 8)
        #expect(
            context.model.modules().compactMap { $0 as? QuantizedLinear }.contains { $0.bits == 4 })
    }

    /// A head embedded in a target checkpoint reads only its own tensors; a standalone head
    /// reads every tensor.
    @Test(arguments: [false, true])
    func drafterDoesNotReadTargetTensors(vision: Bool) async throws {
        for standalone in [false, true] {
            let fixture = try await makeFixture(
                vision: vision, standalone: standalone, prefix: standalone ? "" : "mtp.")
            defer { try? FileManager.default.removeItem(at: fixture.directory) }
            let context = try await fixture.factory.load(
                from: fixture.directory, using: UnusedTokenizerLoader())

            let url = fixture.directory.appendingPathComponent("model.safetensors")
            let all = try loadModelCheckpoint(urls: [url]).weights.keys
            let read = try loadModelCheckpoint(
                urls: [url], excludedNamespaces: context.model.excludedCheckpointNamespaces
            ).weights.keys
            if standalone {
                #expect(Set(read) == Set(all))
            } else {
                #expect(all.contains("model.embed_tokens.weight"))
                #expect(Set(read) == Set(all.filter { $0.hasPrefix("mtp.") }))
            }
        }
    }

    @Test(arguments: [false, true])
    func targetLoaderExcludesConvertedMTPComponent(vision: Bool) throws {
        let data = try fixtureConfiguration(vision: vision, standalone: false, bits: nil)
        let target: any BaseLanguageModel
        if vision {
            target = MLXVLM.Qwen35(
                try JSONDecoder().decode(MLXVLM.Qwen35Configuration.self, from: data))
        } else {
            target = MLXLLM.Qwen35Model(
                try JSONDecoder().decode(MLXLLM.Qwen35Configuration.self, from: data))
        }
        var weights = Dictionary(uniqueKeysWithValues: target.parameters().flattened())
        let count = weights.count
        weights["language_model.mtp.norm.weight"] = MLXArray.zeros([64])
        weights["language_model.mtp.fc.weight"] = MLXArray.zeros([64, 128])
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
            UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try save(
            arrays: weights, metadata: ["format": "mlx"],
            url: directory.appendingPathComponent("model.safetensors"))
        try loadWeights(modelDirectory: directory, model: target)
        #expect(target.parameters().flattened().count == count)
    }

    @Test
    func conversionRoundTripRetainsNormConvention() async throws {
        let fixture = try await makeFixture(vision: false, standalone: false, prefix: "mtp.")
        let output = FileManager.default.temporaryDirectory.appendingPathComponent(
            UUID().uuidString)
        defer {
            try? FileManager.default.removeItem(at: fixture.directory)
            try? FileManager.default.removeItem(at: output)
        }
        let context = try await fixture.factory.load(
            from: fixture.directory, using: UnusedTokenizerLoader())
        _ = try convert(
            modelDirectory: fixture.directory, model: context.model, to: output,
            bits: 4, groupSize: 32)
        let (_, metadata) = try loadArraysAndMetadata(
            url: output.appendingPathComponent("model.safetensors"))
        #expect(metadata[Qwen35CheckpointPolicy.normMetadataKey] == "scale")
        let reloaded = try await fixture.factory.load(from: output, using: UnusedTokenizerLoader())
        let norm = try #require(
            reloaded.model.parameters().flattened().first { $0.0 == "mtp.norm.weight" }?.1)
        #expect(norm.asArray(Float.self) == Array(repeating: 1, count: 64))
    }

    @Test
    func expertParametersAndSettingsAreTransformedTogether() throws {
        let policy = Qwen35CheckpointPolicy(layout: .embedded, preconvertedNorms: false)
        let prefix = "mtp.layers.0.mlp"
        var weights = [String: MLXArray]()
        for expert in 0 ..< 2 {
            for (parameter, shape) in [
                ("weight", [64, 8]), ("scales", [64, 2]), ("biases", [64, 2]),
            ] {
                weights["\(prefix).experts.\(expert).up_proj.\(parameter)"] = MLXArray.zeros(shape)
            }
        }
        let checkpoint = ModelCheckpoint(
            weights: weights,
            perLayerQuantization: .init(
                quantization: .init(groupSize: 32, bits: 4),
                perLayerQuantization: [
                    "\(prefix).experts.0.up_proj": .quantize(.init(groupSize: 32, bits: 8)),
                    "\(prefix).experts.1.up_proj": .quantize(.init(groupSize: 32, bits: 8)),
                ]))
        let prepared = try policy.prepare(checkpoint, mtpNumHiddenLayers: 1, numExperts: 2)
        #expect(prepared.weights["\(prefix).switch_mlp.up_proj.weight"]?.shape == [2, 64, 8])
        #expect(prepared.weights["\(prefix).switch_mlp.up_proj.scales"]?.shape == [2, 64, 2])
        #expect(prepared.weights["\(prefix).switch_mlp.up_proj.biases"]?.shape == [2, 64, 2])
        #expect(
            prepared.perLayerQuantization?.quantization(layer: "\(prefix).switch_mlp.up_proj")?.bits
                == 8)
        #expect(
            prepared.perLayerQuantization?.perLayerQuantization["\(prefix).experts.0.up_proj"]
                == nil)
    }

    @Test(arguments: [false, true], [4, 5, 8])
    func standaloneFactoryLoadsPublishedLayout(vision: Bool, bits: Int) async throws {
        let fixture = try await makeFixture(vision: vision, standalone: true, bits: bits)
        defer { try? FileManager.default.removeItem(at: fixture.directory) }
        let context = try await fixture.factory.load(
            from: fixture.directory, using: UnusedTokenizerLoader())
        let parameters = context.model.parameters().flattened()
        #expect(parameters.count == 31)
        let norm = try #require(parameters.first { $0.0 == "mtp.norm.weight" }?.1)
        #expect(norm.asArray(Float.self) == Array(repeating: 1, count: 64))
        let fc = try #require(
            (context.model as? MLXLLM.Qwen35MTPDraftModel)?.mtp.fc as? QuantizedLinear
                ?? (context.model as? MLXVLM.Qwen35VLMNextNDraftModel)?.mtp.fc as? QuantizedLinear)
        #expect(fc.bits == 8)
        let projections = context.model.modules().compactMap { $0 as? MLXNN.QuantizedLinear }
        #expect(projections.contains { $0.bits == bits })
    }

    @Test(arguments: [false, true], ["mtp.", "language_model.mtp."])
    func embeddedFactoryLoadsComponentWithoutTargetWeights(vision: Bool, prefix: String)
        async throws
    {
        let fixture = try await makeFixture(vision: vision, standalone: false, prefix: prefix)
        defer { try? FileManager.default.removeItem(at: fixture.directory) }
        let context = try await fixture.factory.load(
            from: fixture.directory, using: UnusedTokenizerLoader())
        let weights = Dictionary(uniqueKeysWithValues: context.model.parameters().flattened())
        #expect(weights.count == 15)
        #expect(weights["model.embed_tokens.weight"] == nil)
        #expect(weights["mtp.norm.weight"]?.asArray(Float.self) == Array(repeating: 1, count: 64))
    }

    @Test
    func targetMetadataDoesNotClassifyTheRawMTPShard() throws {
        let policy = Qwen35CheckpointPolicy(layout: .embedded, preconvertedNorms: false)
        let checkpoint = ModelCheckpoint(
            weights: ["mtp.norm.weight": MLXArray.zeros([64])],
            metadata: ["format": "mlx"], weightMetadata: ["mtp.norm.weight": [:]])
        let prepared = try policy.prepare(checkpoint, mtpNumHiddenLayers: 1, numExperts: 0)
        #expect(
            prepared.weights["mtp.norm.weight"]?.asArray(Float.self)
                == Array(repeating: 1, count: 64))
    }

    @Test
    func explicitNormConventionOverridesLegacyLayout() throws {
        let policy = Qwen35CheckpointPolicy(layout: .embedded, preconvertedNorms: false)
        let raw = ModelCheckpoint(
            weights: ["language_model.mtp.norm.weight": MLXArray.zeros([4])],
            metadata: [Qwen35CheckpointPolicy.normMetadataKey: "offset"])
        let prepared = try policy.prepare(raw, mtpNumHiddenLayers: 1, numExperts: 0)
        #expect(prepared.weights["mtp.norm.weight"]?.asArray(Float.self) == [1, 1, 1, 1])
        let scale = ModelCheckpoint(
            weights: prepared.weights,
            metadata: [Qwen35CheckpointPolicy.normMetadataKey: "scale"])
        let repeated = try policy.prepare(scale, mtpNumHiddenLayers: 1, numExperts: 0)
        #expect(repeated.weights["mtp.norm.weight"]?.asArray(Float.self) == [1, 1, 1, 1])
    }

    @Test
    func missingAndCompetingComponentsAreRejected() {
        let policy = Qwen35CheckpointPolicy(layout: .embedded, preconvertedNorms: false)
        for weights in [
            ["model.layers.0.weight": MLXArray(1)],
            ["mtp.norm.weight": MLXArray(1), "language_model.mtp.fc.weight": MLXArray(1)],
        ] {
            #expect(throws: CheckpointComponent.SelectionError.self) {
                try policy.prepare(.init(weights: weights), mtpNumHiddenLayers: 1, numExperts: 0)
            }
        }
    }

    @Test
    func conflictingAndUnknownNormMetadataIsRejected() {
        let policy = Qwen35CheckpointPolicy(layout: .embedded, preconvertedNorms: false)
        let weights = [
            "mtp.norm.weight": MLXArray.zeros([4]),
            "mtp.layers.0.input_layernorm.weight": MLXArray.zeros([4]),
        ]
        for metadata in [
            ["mtp.norm.weight": [Qwen35CheckpointPolicy.normMetadataKey: "unknown"]],
            [
                "mtp.norm.weight": [Qwen35CheckpointPolicy.normMetadataKey: "scale"],
                "mtp.layers.0.input_layernorm.weight": [
                    Qwen35CheckpointPolicy.normMetadataKey: "offset"
                ],
            ],
        ] {
            #expect(throws: Qwen35CheckpointPolicy.LoadingError.self) {
                try policy.prepare(
                    .init(weights: weights, weightMetadata: metadata), mtpNumHiddenLayers: 1,
                    numExperts: 0)
            }
        }
    }

    @Test
    func targetLayoutsShareComponentSelectionAndQuantizationMapping() throws {
        let checkpoint = ModelCheckpoint(
            weights: [
                "model.language_model.layers.0.self_attn.q_proj.weight": MLXArray(1),
                "mtp.fc.weight": MLXArray(2),
                "lm_head.weight": MLXArray(3),
                "notmtp.weight": MLXArray(4),
            ],
            perLayerQuantization: .init(perLayerQuantization: [
                "model.language_model.layers.0.self_attn.q_proj": .quantize(
                    .init(groupSize: 32, bits: 8))
            ]))
        for layout in [Qwen35CheckpointPolicy.TargetLayout.text, .wrappedText, .vision] {
            let prepared = try Qwen35CheckpointPolicy.prepareTarget(
                checkpoint, layout: layout, tiedWordEmbeddings: true)
            let prefix = layout == .text ? "model." : "language_model.model."
            let projection = prefix + "layers.0.self_attn.q_proj"
            #expect(prepared.weights[projection + ".weight"] != nil)
            #expect(prepared.perLayerQuantization?.quantization(layer: projection)?.bits == 8)
            #expect(prepared.weights.count == 2)
            #expect(prepared.weights["notmtp.weight"] != nil)
        }
    }

    @Test
    func unknownAndMissingWeightsStillFailStrictFactoryLoading() async throws {
        for missing in [false, true] {
            let fixture = try await makeFixture(vision: false, standalone: true)
            defer { try? FileManager.default.removeItem(at: fixture.directory) }
            let url = fixture.directory.appendingPathComponent("model.safetensors")
            var weights = try loadArrays(url: url)
            eval(Array(weights.values))
            if missing {
                weights.removeValue(forKey: "norm.weight")
            } else {
                weights["unexpected.weight"] = MLXArray(1)
            }
            try save(arrays: weights, metadata: ["format": "mlx"], url: url)
            await #expect(throws: (any Error).self) {
                try await fixture.factory.load(
                    from: fixture.directory, using: UnusedTokenizerLoader())
            }
        }
    }

    @Test
    func expertPrecisionCannotChangeDuringStacking() throws {
        let policy = Qwen35CheckpointPolicy(layout: .embedded, preconvertedNorms: false)
        let prefix = "mtp.layers.0.mlp.experts"
        let weights = [
            "\(prefix).0.up_proj.weight": MLXArray.zeros([64, 64]),
            "\(prefix).1.up_proj.weight": MLXArray.zeros([64, 64]),
        ]
        let checkpoint = ModelCheckpoint(
            weights: weights,
            perLayerQuantization: .init(
                quantization: .init(groupSize: 32, bits: 4),
                perLayerQuantization: [
                    "\(prefix).0.up_proj": .quantize(.init(groupSize: 32, bits: 8))
                ]))
        #expect(throws: Qwen35CheckpointPolicy.LoadingError.self) {
            try policy.prepare(checkpoint, mtpNumHiddenLayers: 1, numExperts: 2)
        }
        #expect(throws: Qwen35CheckpointPolicy.LoadingError.self) {
            try policy.prepare(
                .init(weights: ["\(prefix).0.up_proj.weight": MLXArray.zeros([64, 64])]),
                mtpNumHiddenLayers: 1, numExperts: 2)
        }
    }

    private struct Fixture {
        let directory: URL
        let factory: MTPDrafterModelFactory
    }

    private func makeFixture(
        vision: Bool, standalone: Bool, prefix: String = "", bits: Int? = nil
    ) async throws -> Fixture {
        let config = try fixtureConfiguration(
            vision: vision, standalone: standalone, bits: bits, prefix: prefix)
        let registry = ModelTypeRegistry<any MTPDrafterModel>()
        let modelType = standalone ? "qwen3_5_mtp" : "qwen3_5"
        await registry.registerModelType(modelType) { data in
            try makeDrafter(configuration: data, vision: vision)
        }
        let model = try makeDrafter(configuration: config, vision: vision)
        if let bits {
            quantize(model: model) { path, _ in (32, path == "mtp.fc" ? 8 : bits, .affine) }
        }
        var weights = [String: MLXArray]()
        for (name, array) in model.parameters().flattened() {
            let serialized = prefix + name.dropFirst("mtp.".count)
            let isNorm = name.contains("norm")
            weights[serialized] = isNorm ? MLXArray.ones(array.shape) : array
            if prefix == "mtp.", isNorm { weights[serialized] = MLXArray.zeros(array.shape) }
        }
        if !standalone { weights["model.embed_tokens.weight"] = MLXArray.zeros([16, 64]) }
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
            UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        do {
            try config.write(to: directory.appendingPathComponent("config.json"))
            try save(
                arrays: weights, metadata: standalone ? ["format": "mlx"] : [:],
                url: directory.appendingPathComponent("model.safetensors"))
        } catch {
            try? FileManager.default.removeItem(at: directory)
            throw error
        }
        return Fixture(
            directory: directory,
            factory: MTPDrafterModelFactory(
                typeRegistry: registry, modelRegistry: MTPDrafterRegistry.shared))
    }

    private func fixtureConfiguration(
        vision: Bool, standalone: Bool, bits: Int?, prefix: String = "mtp."
    ) throws -> Data {
        let json =
            vision
            ? qwen35VLMConfigJSON(mtpLayers: 1)
            : qwen35WrappedTextConfigJSON(modelType: "qwen3_5")
        let sized = json.replacingOccurrences(
            of: "\"hidden_size\": 16", with: "\"hidden_size\": 64"
        )
        .replacingOccurrences(of: "\"intermediate_size\": 32", with: "\"intermediate_size\": 128")
        .replacingOccurrences(of: "\"head_dim\": 8", with: "\"head_dim\": 32")
        var root = try #require(
            JSONSerialization.jsonObject(with: Data(sized.utf8)) as? [String: Any])
        if standalone { root["model_type"] = "qwen3_5_mtp" }
        if let bits {
            root["quantization"] = [
                "group_size": 32, "bits": bits,
                (standalone ? "fc" : prefix + "fc"): ["group_size": 32, "bits": 8],
            ]
        }
        return try JSONSerialization.data(withJSONObject: root)
    }
}

private func makeDrafter(configuration: Data, vision: Bool) throws -> any MTPDrafterModel {
    if vision {
        return MLXVLM.Qwen35VLMNextNDraftModel(
            try JSONDecoder().decode(MLXVLM.Qwen35Configuration.self, from: configuration))
    }
    return MLXLLM.Qwen35MTPDraftModel(
        try JSONDecoder().decode(MLXLLM.Qwen35Configuration.self, from: configuration))
}

private struct UnusedTokenizerLoader: TokenizerLoader {
    func load(from url: URL) async throws -> any Tokenizer {
        throw ModelFactoryError.invalidConfiguration(
            "MTP checkpoints must borrow the target tokenizer.")
    }
}
