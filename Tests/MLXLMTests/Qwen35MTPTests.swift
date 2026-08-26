import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import MLXLLM
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
func testQwen35MTPDraftersValidateTargetFamily() throws {
    let textConfig = try JSONDecoder().decode(
        MLXLLM.Qwen35TextConfiguration.self,
        from: Data(qwen35TextConfigJSON(mtpLayers: 1).utf8))
    let textTarget = MLXLLM.Qwen35TextModel(textConfig)
    let textDrafter = MLXLLM.Qwen35MTPDraftModel(textConfig)

    let vlmConfig = try JSONDecoder().decode(
        MLXVLM.Qwen35Configuration.self,
        from: Data(qwen35VLMConfigJSON(mtpLayers: 1).utf8))
    let vlmTarget = MLXVLM.Qwen35(vlmConfig)
    let vlmDrafter = MLXVLM.Qwen35VLMNextNDraftModel(vlmConfig)

    try textDrafter.validateCompatibility(with: textTarget)
    try vlmDrafter.validateCompatibility(with: vlmTarget)

    #expect(throws: MTPDrafterCompatibilityError.self) {
        try textDrafter.validateCompatibility(with: vlmTarget)
    }
    #expect(throws: MTPDrafterCompatibilityError.self) {
        try vlmDrafter.validateCompatibility(with: textTarget)
    }
}

@Test
func testQwen35MTPDraftSanitizeKeepsAndShiftsMTPNorms() throws {
    let cfg = try JSONDecoder().decode(
        MLXLLM.Qwen35TextConfiguration.self,
        from: Data(qwen35TextConfigJSON(mtpLayers: 1).utf8))
    let drafter = MLXLLM.Qwen35MTPDraftModel(cfg)

    let sanitized = drafter.sanitize(weights: [
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

@Test
func testQwen35StandaloneMTPDoesNotDoubleShiftConvertedNorms() throws {
    let cfg = try JSONDecoder().decode(
        MLXLLM.Qwen35Configuration.self,
        from: Data(qwen35StandaloneMTPConfigJSON().utf8))
    let drafter = MLXLLM.Qwen35MTPDraftModel(cfg, preconvertedNorms: true)

    let weight = MLXArray.zeros([16])
    let sanitized = drafter.sanitize(weights: ["mtp.norm.weight": weight])
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

    let sanitized = drafter.sanitize(weights: weights)

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
    let sanitized = drafter.sanitize(weights: [
        "mtp.embed_tokens.weight": MLXArray.zeros([16, 16]),
        "model.embed_tokens.weight": MLXArray.ones([16, 16]),
    ])
    #expect(sanitized["mtp.embed_tokens.weight"] != nil)
    #expect(sanitized["model.embed_tokens.weight"] == nil)
}

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
    func testQwen35TextDrafterFinalizesAndResumesCanonicalBoundary() throws {
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

        var state = drafter.makeState(parameters: nil)
        var bonus = MLXArray([Int32(4)])
        drafter.prepareDrafterState(
            target: target, promptTokens: prompt, targetHidden: promptHidden,
            firstBonus: bonus, positionDeltas: nil, state: &state, sampler: sampler)
        let proposal = drafter.draftBlock(
            target: target, lastToken: bonus,
            lastHidden: promptHidden[0..., (-1)..., 0...], sharedKV: [:],
            positionDeltas: nil, queryOffset: 3, blockSize: 2,
            state: &state, sampler: sampler)
        let verifyHidden = MLXArray.zeros([1, 2, cfg.hiddenSize])
        bonus = MLXArray([Int32(5)])
        drafter.commitDrafterState(
            target: target, targetHidden: verifyHidden, draftTokens: proposal,
            acceptedCount: 1, finalToken: bonus, positionDeltas: nil,
            state: &state, sampler: sampler)
        eval(state.seedToken!, state.seedHidden!)
        #expect(state.nextPosition == 5)
        #expect(state.cache.allSatisfy { $0.offset == 5 })

        let boundaryHidden = verifyHidden[0..., 1 ..< 2, 0...]
        #expect(
            drafter.finalizeDrafterState(
                target: target, targetBoundaryHidden: boundaryHidden,
                targetProcessedTokenCount: 5, discardedTargetTokens: 0,
                positionDeltas: nil, state: &state))
        #expect(state.nextPosition == 4)
        #expect(state.cache.allSatisfy { $0.offset == 4 })
        #expect(state.seedToken == nil)
        #expect(state.seedHidden == nil)

        let suffix = MLXArray([Int32(6), 7]).reshaped([1, 2])
        let suffixHidden = MLXArray.zeros([1, 2, cfg.hiddenSize])
        #expect(
            drafter.resumeDrafterState(
                target: target, suffixTokens: suffix,
                suffixTargetHidden: suffixHidden,
                targetBoundaryHidden: boundaryHidden,
                firstBonus: MLXArray([Int32(8)]), positionDeltas: nil,
                state: &state, sampler: sampler))
        eval(state.seedToken!, state.seedHidden!)
        #expect(state.nextPosition == 7)
        #expect(state.cache.allSatisfy { $0.offset == 7 })
    }

    @Test
    func testQwen35TextDrafterFinalizationAlsoDiscardsTargetLookahead() throws {
        let cfg = try JSONDecoder().decode(
            MLXLLM.Qwen35TextConfiguration.self,
            from: Data(qwen35TextConfigJSON(mtpLayers: 1).utf8))
        let target = MLXLLM.Qwen35TextModel(cfg)
        let drafter = MLXLLM.Qwen35MTPDraftModel(cfg)
        let sampler = GenerateParameters(temperature: 0).sampler()
        let prompt = MLXArray([Int32(1), 2, 3]).reshaped([1, 3])
        let hidden = MLXArray.zeros([1, 3, cfg.hiddenSize])
        var state = drafter.makeState(parameters: nil)
        drafter.prepareDrafterState(
            target: target, promptTokens: prompt, targetHidden: hidden,
            firstBonus: MLXArray([Int32(4)]), positionDeltas: nil,
            state: &state, sampler: sampler)
        let proposal = drafter.draftBlock(
            target: target, lastToken: MLXArray([Int32(4)]),
            lastHidden: hidden[0..., (-1)..., 0...], sharedKV: [:],
            positionDeltas: nil, queryOffset: 3, blockSize: 2,
            state: &state, sampler: sampler)
        let verifyHidden = MLXArray.zeros([1, 2, cfg.hiddenSize])
        drafter.commitDrafterState(
            target: target, targetHidden: verifyHidden, draftTokens: proposal,
            acceptedCount: 1, finalToken: MLXArray([Int32(5)]),
            positionDeltas: nil, state: &state, sampler: sampler)

        #expect(state.nextPosition == 5)
        #expect(
            drafter.finalizeDrafterState(
                target: target,
                targetBoundaryHidden: verifyHidden[0..., ..<1, 0...],
                targetProcessedTokenCount: 4, discardedTargetTokens: 1,
                positionDeltas: nil, state: &state))
        #expect(state.nextPosition == 3)
        #expect(state.cache.allSatisfy { $0.offset == 3 })
    }

    @Test
    func testQwen35VLMDrafterResumeKeepsPositionDeltaAlignment() throws {
        let cfg = try JSONDecoder().decode(
            MLXVLM.Qwen35Configuration.self,
            from: Data(qwen35VLMConfigJSON(mtpLayers: 1).utf8))
        let target = MLXVLM.Qwen35(cfg)
        let drafter = MLXVLM.Qwen35VLMNextNDraftModel(cfg)
        let sampler = GenerateParameters(temperature: 0).sampler()
        let prompt = MLXArray([Int32(1), 2, 3]).reshaped([1, 3])
        let hidden = MLXArray.zeros([1, 3, cfg.textConfiguration.hiddenSize])
        let positionDeltas = MLXArray([Int32(2)])
        var state = drafter.makeState(parameters: nil)

        drafter.prepareDrafterState(
            target: target, promptTokens: prompt, targetHidden: hidden,
            firstBonus: MLXArray([Int32(4)]), positionDeltas: positionDeltas,
            state: &state, sampler: sampler)
        #expect(state.nextPosition == 3)
        #expect(
            drafter.finalizeDrafterState(
                target: target,
                targetBoundaryHidden: hidden[0..., (-1)..., 0...],
                targetProcessedTokenCount: 3, discardedTargetTokens: 0,
                positionDeltas: positionDeltas, state: &state))
        #expect(state.nextPosition == 2)

        #expect(
            drafter.resumeDrafterState(
                target: target,
                suffixTokens: MLXArray([Int32(5), 6]).reshaped([1, 2]),
                suffixTargetHidden: MLXArray.zeros(
                    [1, 2, cfg.textConfiguration.hiddenSize]),
                targetBoundaryHidden: hidden[0..., (-1)..., 0...],
                firstBonus: MLXArray([Int32(7)]),
                positionDeltas: positionDeltas, state: &state, sampler: sampler))
        eval(state.seedToken!, state.seedHidden!)
        #expect(state.nextPosition == 5)
        #expect(state.cache.allSatisfy { $0.offset == 5 })
    }

    @Test
    func testQwen35TextChatSessionResumesActualDrafterState() async throws {
        let cfg = try JSONDecoder().decode(
            MLXLLM.Qwen35TextConfiguration.self,
            from: Data(qwen35TextConfigJSON(mtpLayers: 1).utf8))
        let target = MLXLLM.Qwen35TextModel(cfg)
        let tokenizer = ReversibleQwenMTPTokenizer()
        let processor = ReversibleQwenMTPInputProcessor(tokenizer: tokenizer)
        let context = ModelContext(
            configuration: processor.configuration,
            model: target,
            processor: processor,
            tokenizer: tokenizer)
        let drafterContainer = MTPDrafterContainer(
            context: MTPDrafterContext(
                configuration: processor.configuration,
                model: MLXLLM.Qwen35MTPDraftModel(cfg)))
        let speculative = ChatSession(
            context,
            speculativeDecoding: try SpeculativeDecodingConfig(
                mtpDrafter: drafterContainer, blockSize: 2),
            generateParameters: .init(maxTokens: 4, temperature: 0))
        let targetOnly = ChatSession(
            context,
            generateParameters: .init(maxTokens: 4, temperature: 0))

        let firstPrompt = tokenizer.text(for: 1)
        let secondPrompt = tokenizer.text(for: 2)
        let speculativeFirst = try await collectQwenChatDetails(
            speculative.streamDetails(to: firstPrompt))
        let targetFirst = try await collectQwenChatDetails(
            targetOnly.streamDetails(to: firstPrompt))
        let speculativeSecond = try await collectQwenChatDetails(
            speculative.streamDetails(to: secondPrompt))
        let targetSecond = try await collectQwenChatDetails(
            targetOnly.streamDetails(to: secondPrompt))

        #expect(speculativeFirst.text == targetFirst.text)
        #expect(speculativeSecond.text == targetSecond.text)
        #expect(speculativeSecond.info.promptTokenCount == 2)
        #expect(speculativeSecond.info.cachedPromptTokenCount > 0)
        #expect((speculativeSecond.info.proposedDraftTokens ?? 0) > 0)
        #expect(speculativeSecond.info.passthroughReason == nil)
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
            #expect(allClose(actual, reference, rtol: 1e-5, atol: 1e-5).item(Bool.self))
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
            #expect(allClose(actual, reference, rtol: 1e-5, atol: 1e-5).item(Bool.self))
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

private struct ReversibleQwenMTPTokenizer: Tokenizer {
    private let scalarBase = 0xE000

    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }

    func text(for token: Int) -> String {
        String(UnicodeScalar(scalarBase + token)!)
    }

    func encode(text: String, addSpecialTokens _: Bool) -> [Int] {
        text.unicodeScalars.map { scalar in
            let value = Int(scalar.value) - scalarBase
            return (0 ..< 16).contains(value) ? value : 1
        }
    }

    func decode(tokenIds: [Int], skipSpecialTokens _: Bool) -> String {
        tokenIds.map(text(for:)).joined()
    }

    func convertTokenToId(_: String) -> Int? { nil }
    func convertIdToToken(_ id: Int) -> String? { text(for: id) }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools _: [[String: any Sendable]]?,
        additionalContext _: [String: any Sendable]?
    ) throws -> [Int] {
        messages.flatMap { message in
            encode(
                text: message["content"] as? String ?? "",
                addSpecialTokens: false)
        }
    }
}

private struct ReversibleQwenMTPInputProcessor: UserInputProcessor {
    let tokenizer: ReversibleQwenMTPTokenizer
    let configuration = ModelConfiguration(id: "tiny-qwen-mtp-chat-session")

    func prepare(input: UserInput) async throws -> LMInput {
        let messages = DefaultMessageGenerator().generate(from: input)
        return LMInput(
            tokens: MLXArray(
                try tokenizer.applyChatTemplate(
                    messages: messages,
                    tools: input.tools,
                    additionalContext: input.additionalContext)))
    }
}

private func collectQwenChatDetails(
    _ stream: AsyncThrowingStream<Generation, Error>
) async throws -> (text: String, info: GenerateCompletionInfo) {
    var text = ""
    var info: GenerateCompletionInfo?
    for try await event in stream {
        text += event.chunk ?? ""
        if let completion = event.info { info = completion }
    }
    return (text, try #require(info))
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
