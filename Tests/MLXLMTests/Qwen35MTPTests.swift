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

@Test
func testQwen35MTPPredictorAdvancesOneLayerCachePerSpecStep() throws {
    let cfg = try JSONDecoder().decode(
        MLXLLM.Qwen35TextConfiguration.self,
        from: Data(qwen35TextConfigJSON(mtpLayers: 2).utf8))
    let predictor = MLXLLM.Qwen35MTPPredictor(cfg)
    let cache = predictor.newCache()
    let embeds = MLXArray.zeros([1, 1, 16])
    let hidden = MLXArray.zeros([1, 1, 16])

    let first = predictor(
        inputsEmbeds: embeds, hiddenStates: hidden, cache: cache[0], stepIndex: 0,
        positionOffset: 128)
    eval(first)
    #expect(cache[0].offset == 1)
    #expect(cache[1].offset == 0)

    let second = predictor(
        inputsEmbeds: embeds, hiddenStates: first, cache: cache[1], stepIndex: 1,
        positionOffset: 129)
    eval(second)
    #expect(cache[0].offset == 1)
    #expect(cache[1].offset == 1)
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
    let cache = model.newCache(parameters: nil as GenerateParameters?)
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
func testQwen35MTPRegistrationsCreateTextAndVLMDrafters() async throws {
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
}

@Test
func testQwen35MTPRegistrationsAreOrderIndependentForSharedModelTypes() async throws {
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
