// Copyright © 2026 Apple Inc.

import Foundation
import MLXLMCommon
import MLXVLM
import Testing

// MARK: - Type-registry registration

@Test
func testGemma4AssistantRegistrationRegistersType() async throws {
    await Gemma4AssistantRegistration.register()

    let json = """
        {
          "model_type": "gemma4_assistant",
          "backbone_hidden_size": 4,
          "tie_word_embeddings": true,
          "use_ordered_embeddings": false,
          "num_centroids": 2,
          "centroid_intermediate_top_k": 1,
          "text_config": {
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
            "tie_word_embeddings": true
          }
        }
        """
    let data = Data(json.utf8)

    let model = try await MTPDrafterTypeRegistry.shared.createModel(
        configuration: data, modelType: "gemma4_assistant")
    #expect(model is Gemma4AssistantDraftModel)
}

@Test
func testMTPDrafterTypeRegistryUnknownModelTypeThrows() async {
    // Don't pre-register; an unknown model type must throw
    // `unsupportedModelType`. Use a distinctive name so it doesn't collide
    // with any registration another test made (registration is shared).
    do {
        let _: any MTPDrafterModel =
            try await MTPDrafterTypeRegistry.shared.createModel(
                configuration: Data(),
                modelType: "definitely_not_a_drafter_type")
        Issue.record("expected unsupportedModelType to throw")
    } catch let error as ModelFactoryError {
        if case .unsupportedModelType(let t) = error {
            #expect(t == "definitely_not_a_drafter_type")
        } else {
            Issue.record("unexpected ModelFactoryError: \(error)")
        }
    } catch {
        Issue.record("unexpected error: \(error)")
    }
}

// MARK: - Model registry contents

@Test
func testMTPDrafterRegistryContainsBothReferenceCheckpoints() {
    let registry = MTPDrafterRegistry.shared
    #expect(registry.contains(id: "mlx-community/gemma-4-26B-A4B-it-assistant-bf16"))
    #expect(registry.contains(id: "mlx-community/gemma-4-31B-it-assistant-bf16"))
}

@Test
func testMTPDrafterRegistrySharedStaticAccessors() {
    let r26 = MTPDrafterRegistry.gemma4_26B_assistant_bf16
    let r31 = MTPDrafterRegistry.gemma4_31B_assistant_bf16
    // Don't depend on a specific case enumeration shape; just verify that
    // the registered name property contains the expected substring.
    #expect(r26.name.contains("26B-A4B-it-assistant"))
    #expect(r31.name.contains("31B-it-assistant"))
}

// MARK: - Factory load (gated on checkpoint presence)

@Test
func testMTPDrafterFactoryLoadFromDirectoryWhenCheckpointPresent() async throws {
    // Look for the 31B-assistant-bf16 checkpoint in the HF cache; skip
    // gracefully when absent.
    let home = FileManager.default.homeDirectoryForCurrentUser
    let hub = home.appendingPathComponent(".cache/huggingface/hub")
    let folder = hub.appendingPathComponent(
        "models--mlx-community--gemma-4-31B-it-assistant-bf16/snapshots"
    )
    guard
        let entries = try? FileManager.default.contentsOfDirectory(
            at: folder, includingPropertiesForKeys: nil),
        let snapshot = entries.first
    else {
        Issue.record("31B-assistant-bf16 checkpoint not in HF cache; skipping factory load test")
        return
    }

    await Gemma4AssistantRegistration.register()
    let factory = MTPDrafterModelFactory.shared

    let container = try await factory.loadContainer(
        from: snapshot, using: NoOpTokenizerLoader()
    )
    let isDrafter = await container.perform { ctx in
        ctx.model is Gemma4AssistantDraftModel
    }
    #expect(isDrafter)
}

// MARK: - Helpers

/// A `TokenizerLoader` placeholder. ``MTPDrafterModelFactory`` ignores the
/// loader (drafters borrow their target's tokenizer), but the protocol
/// requires a non-optional argument.
private final class NoOpTokenizerLoader: TokenizerLoader {
    func load(from url: URL) async throws -> any Tokenizer {
        fatalError("MTPDrafterModelFactory must not call tokenizer load")
    }
}
