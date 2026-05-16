// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
@_spi(Testing) import MLXVLM
import Testing

/// Locate `tools/fixtures/` relative to this test file by walking up.
private func toolsFixturesDir(file: String = #filePath) -> URL? {
    var dir = URL(fileURLWithPath: file).deletingLastPathComponent()
    let fs = FileManager.default
    for _ in 0 ..< 10 {
        let candidate = dir.appendingPathComponent("tools/fixtures")
        if fs.fileExists(atPath: candidate.path) {
            return candidate
        }
        let parent = dir.deletingLastPathComponent()
        if parent.path == dir.path { break }
        dir = parent
    }
    return nil
}

/// Find the cached snapshot directory for an HF model ID, if present.
private func hfSnapshotDir(modelId: String) -> URL? {
    let home = FileManager.default.homeDirectoryForCurrentUser
    let hub = home.appendingPathComponent(".cache/huggingface/hub")
    let folderName = "models--" + modelId.replacingOccurrences(of: "/", with: "--")
    let snapshots = hub.appendingPathComponent(folderName).appendingPathComponent("snapshots")
    guard
        let entries = try? FileManager.default.contentsOfDirectory(
            at: snapshots, includingPropertiesForKeys: nil)
    else { return nil }
    return entries.first
}

// MARK: - Config decode (no MLX kernels required)

@Test
func testGemma4AssistantConfigurationDecodesSyntheticJSON() throws {
    let json = """
        {
          "model_type": "gemma4_assistant",
          "backbone_hidden_size": 5376,
          "tie_word_embeddings": true,
          "use_ordered_embeddings": false,
          "num_centroids": 2048,
          "centroid_intermediate_top_k": 32,
          "text_config": {
            "model_type": "gemma4_text",
            "hidden_size": 1024,
            "num_hidden_layers": 4,
            "num_attention_heads": 32,
            "num_key_value_heads": 16,
            "head_dim": 256,
            "global_head_dim": 512,
            "vocab_size": 262144,
            "num_kv_shared_layers": 0,
            "hidden_size_per_layer_input": 0,
            "sliding_window": 512,
            "sliding_window_pattern": 5,
            "max_position_embeddings": 262144,
            "rms_norm_eps": 1e-6,
            "rope_traditional": false,
            "use_double_wide_mlp": false,
            "enable_moe_block": false,
            "attention_k_eq_v": true,
            "intermediate_size": 8192,
            "layer_types": ["sliding_attention", "sliding_attention", "sliding_attention", "full_attention"],
            "rope_parameters": {},
            "tie_word_embeddings": true
          }
        }
        """
    let data = Data(json.utf8)
    let cfg = try JSONDecoder().decode(Gemma4AssistantConfiguration.self, from: data)
    #expect(cfg.modelType == "gemma4_assistant")
    #expect(cfg.backboneHiddenSize == 5376)
    #expect(cfg.tieWordEmbeddings == true)
    #expect(cfg.useOrderedEmbeddings == false)
    #expect(cfg.numCentroids == 2048)
    #expect(cfg.centroidIntermediateTopK == 32)
    #expect(cfg.textConfiguration.hiddenSize == 1024)
    #expect(cfg.textConfiguration.hiddenLayers == 4)
    #expect(cfg.textConfiguration.vocabularySize == 262_144)
    #expect(cfg.textConfiguration.layerTypes.count == 4)
}

@Test
func testGemma4AssistantConfigurationDecodesRealCheckpoint() throws {
    let drafterDir = hfSnapshotDir(modelId: "mlx-community/gemma-4-31B-it-assistant-bf16")
    guard let drafterDir else {
        Issue.record("31B drafter checkpoint not present in HF cache; skipping")
        return
    }
    let configURL = drafterDir.appendingPathComponent("config.json")
    let data = try Data(contentsOf: configURL)
    let cfg = try JSONDecoder().decode(Gemma4AssistantConfiguration.self, from: data)
    #expect(cfg.modelType == "gemma4_assistant")
    #expect(cfg.backboneHiddenSize == 5376)
    #expect(cfg.tieWordEmbeddings == true)
    #expect(cfg.useOrderedEmbeddings == false)
    #expect(cfg.textConfiguration.hiddenLayers == 4)
    #expect(cfg.textConfiguration.hiddenSize == 1024)
    #expect(cfg.textConfiguration.vocabularySize == 262_144)
    #expect(
        cfg.textConfiguration.layerTypes
            == ["sliding_attention", "sliding_attention", "sliding_attention", "full_attention"])
}

// MARK: - Sanitize behavior (no MLX kernels — uses MLXArray scalar metadata only)

@Test
func testSanitizeDropsLmHeadWhenTied() {
    let cfg = syntheticConfig(tieWordEmbeddings: true)
    let model = Gemma4AssistantDraftModel(cfg)
    let weights: [String: MLXArray] = [
        "model.embed_tokens.weight": MLXArray.zeros([10, 4]),
        "lm_head.weight": MLXArray.zeros([10, 4]),
        "pre_projection.weight": MLXArray.zeros([4, 8]),
    ]
    let sanitized = model.sanitize(weights: weights)
    #expect(sanitized["lm_head.weight"] == nil)
    #expect(sanitized["model.embed_tokens.weight"] != nil)
    #expect(sanitized["pre_projection.weight"] != nil)
}

@Test
func testSanitizeKeepsLmHeadWhenNotTied() {
    let cfg = syntheticConfig(tieWordEmbeddings: false)
    let model = Gemma4AssistantDraftModel(cfg)
    let weights: [String: MLXArray] = [
        "model.embed_tokens.weight": MLXArray.zeros([10, 4]),
        "lm_head.weight": MLXArray.zeros([10, 4]),
    ]
    let sanitized = model.sanitize(weights: weights)
    #expect(sanitized["lm_head.weight"] != nil)
}

// MARK: - Rung 1: weight load against real checkpoint
//
// These tests require: HuggingFace cache contains the drafter checkpoint AND
// the `swift test` runtime can load the metallib. The latter is currently
// blocked in the SPM CLI; tests pass in Xcode.

@Test
func testRung1WeightsLoadFrom31BCheckpoint() throws {
    guard let drafterDir = hfSnapshotDir(modelId: "mlx-community/gemma-4-31B-it-assistant-bf16")
    else {
        Issue.record("31B drafter checkpoint not in HF cache; skipping Rung 1")
        return
    }
    let configURL = drafterDir.appendingPathComponent("config.json")
    let cfg = try JSONDecoder().decode(
        Gemma4AssistantConfiguration.self, from: Data(contentsOf: configURL))
    let model = Gemma4AssistantDraftModel(cfg)
    // loadWeights enumerates *.safetensors in the directory, sanitizes via the
    // model's sanitize hook, and applies via update(parameters:verify:).
    // If any keys are missing or unexpected, update(verify: [.all]) throws.
    try loadWeights(modelDirectory: drafterDir, model: model)
}

// MARK: - Rung 2/3: forward parity against fixtures

@Test
func testRung2And3ForwardMatchesFixture31BCase01() throws {
    guard let fixturesDir = toolsFixturesDir() else {
        Issue.record("fixtures dir not found; skipping Rung 2/3")
        return
    }
    guard let drafterDir = hfSnapshotDir(modelId: "mlx-community/gemma-4-31B-it-assistant-bf16")
    else {
        Issue.record("31B drafter checkpoint not in HF cache; skipping Rung 2/3")
        return
    }

    let configURL = drafterDir.appendingPathComponent("config.json")
    let cfg = try JSONDecoder().decode(
        Gemma4AssistantConfiguration.self, from: Data(contentsOf: configURL))
    let model = Gemma4AssistantDraftModel(cfg)
    try loadWeights(modelDirectory: drafterDir, model: model)

    let fixtureURL =
        fixturesDir
        .appendingPathComponent("drafter_forward/case_01_q1_kv32_baseline.safetensors")
    let arrays = try MLX.loadArrays(url: fixtureURL)
    guard
        let inputsEmbeds = arrays["inputs/inputs_embeds"],
        let positionIds = arrays["inputs/position_ids"],
        let fullKeys = arrays["inputs/shared_kv/full_attention/keys"],
        let fullValues = arrays["inputs/shared_kv/full_attention/values"],
        let slidingKeys = arrays["inputs/shared_kv/sliding_attention/keys"],
        let slidingValues = arrays["inputs/shared_kv/sliding_attention/values"],
        let expectedLastHidden = arrays["outputs/last_hidden"],
        let expectedLogits = arrays["outputs/logits"]
    else {
        Issue.record("fixture missing expected tensor keys")
        return
    }

    let sharedKV: [String: (MLXArray, MLXArray)] = [
        "full_attention": (fullKeys, fullValues),
        "sliding_attention": (slidingKeys, slidingValues),
    ]
    let (lastHidden, logits) = model.forwardHidden(
        inputsEmbeds: inputsEmbeds,
        sharedKV: sharedKV,
        positionIds: positionIds
    )

    // Shape parity.
    #expect(lastHidden.shape == expectedLastHidden.shape)
    #expect(logits.shape == expectedLogits.shape)

    // Finiteness check.
    #expect(!isNaN(lastHidden).any().item(Bool.self), "Swift last_hidden has NaN")
    #expect(!isInf(lastHidden).any().item(Bool.self), "Swift last_hidden has Inf")
    #expect(!isNaN(logits).any().item(Bool.self), "Swift logits has NaN")
    #expect(!isInf(logits).any().item(Bool.self), "Swift logits has Inf")

    // Value parity (Rung 2/3) — bf16 tolerance per PLAN §11 (atol=1e-3, rtol=1e-3).
    // Cast both to float32 to avoid bf16 quirks in allClose.
    let lhSwift = lastHidden.asType(.float32)
    let lhExpected = expectedLastHidden.asType(.float32)
    #expect(
        allClose(lhSwift, lhExpected, rtol: 1e-3, atol: 1e-3).item(Bool.self),
        "last_hidden values diverge from fixture")

    let logitsSwift = logits.asType(.float32)
    let logitsExpected = expectedLogits.asType(.float32)
    #expect(
        allClose(logitsSwift, logitsExpected, rtol: 1e-3, atol: 1e-3).item(Bool.self),
        "logits values diverge from fixture")
}

// MARK: - Synthetic shape test (no checkpoint needed)

@Test
func testGemma4AssistantDraftModelInstantiatesAndShape() {
    let cfg = syntheticConfig(tieWordEmbeddings: true)
    let model = Gemma4AssistantDraftModel(cfg)
    #expect(model.config.modelType == "gemma4_assistant")
    #expect(model.config.backboneHiddenSize == 4)
    #expect(model.config.tieWordEmbeddings == true)
    // The inner Embedding and Linears are constructed at init.
    // We don't run inference here (would need actual weights + metal kernels).
}

// MARK: - Helpers

private func syntheticConfig(tieWordEmbeddings: Bool) -> Gemma4AssistantConfiguration {
    // Build a tiny synthetic config sufficient for sanitize / instantiation tests.
    // Uses JSON round-trip to avoid manually constructing Gemma4TextConfiguration
    // (which has many defaulted fields and a custom decoder).
    let textJSON =
        """
        {
          "model_type": "gemma4_text",
          "hidden_size": 4,
          "num_hidden_layers": 1,
          "num_attention_heads": 2,
          "num_key_value_heads": 1,
          "head_dim": 2,
          "global_head_dim": 2,
          "vocab_size": 10,
          "num_kv_shared_layers": 0,
          "hidden_size_per_layer_input": 0,
          "sliding_window": 4,
          "sliding_window_pattern": 1,
          "max_position_embeddings": 16,
          "rms_norm_eps": 1e-6,
          "rope_traditional": false,
          "use_double_wide_mlp": false,
          "enable_moe_block": false,
          "attention_k_eq_v": true,
          "intermediate_size": 8,
          "layer_types": ["full_attention"],
          "rope_parameters": {},
          "tie_word_embeddings": \(tieWordEmbeddings)
        }
        """
    let json =
        """
        {
          "model_type": "gemma4_assistant",
          "backbone_hidden_size": 4,
          "tie_word_embeddings": \(tieWordEmbeddings),
          "use_ordered_embeddings": false,
          "num_centroids": 2,
          "centroid_intermediate_top_k": 1,
          "text_config": \(textJSON)
        }
        """
    return try! JSONDecoder().decode(
        Gemma4AssistantConfiguration.self, from: Data(json.utf8))
}
