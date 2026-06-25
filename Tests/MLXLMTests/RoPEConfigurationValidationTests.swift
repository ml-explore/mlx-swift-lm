// Copyright © 2026 Apple Inc.

import Foundation
import MLXLLM
import MLXLMCommon
import MLXVLM
import Testing

@Test
func qwen3RegistryRejectsUnsupportedRoPEType() async {
    let json = """
        {
          "hidden_size": 4,
          "num_hidden_layers": 1,
          "intermediate_size": 8,
          "num_attention_heads": 2,
          "rms_norm_eps": 1e-6,
          "vocab_size": 16,
          "num_key_value_heads": 1,
          "head_dim": 2,
          "rope_scaling": {
            "rope_type": "not_supported",
            "factor": 2.0
          }
        }
        """

    await expectInvalidConfiguration(
        containing: "unsupported type 'not_supported'"
    ) {
        _ = try await LLMTypeRegistry.shared.createModel(
            configuration: Data(json.utf8), modelType: "qwen3")
    }
}

@Test
func qwen2VLRegistryRejectsInvalidMRoPESection() async {
    let json = """
        {
          "model_type": "qwen2_vl",
          "hidden_size": 4,
          "num_hidden_layers": 1,
          "intermediate_size": 8,
          "num_attention_heads": 2,
          "rms_norm_eps": 1e-6,
          "vocab_size": 16,
          "num_key_value_heads": 1,
          "image_token_id": 151655,
          "video_token_id": 151656,
          "rope_scaling": {
            "type": "mrope",
            "mrope_section": [16, 16]
          },
          "vision_config": {
            "depth": 1,
            "embed_dim": 4,
            "hidden_size": 4,
            "num_heads": 2,
            "patch_size": 14,
            "mlp_ratio": 2.0,
            "spatial_patch_size": 14,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2
          }
        }
        """

    await expectInvalidConfiguration(
        containing: "mrope_section must contain three positive integers"
    ) {
        _ = try await VLMTypeRegistry.shared.createModel(
            configuration: Data(json.utf8), modelType: "qwen2_vl")
    }
}

private func expectInvalidConfiguration(
    containing expectedMessage: String,
    _ operation: () async throws -> Void
) async {
    do {
        try await operation()
        Issue.record("expected invalidConfiguration to throw")
    } catch let error as ModelFactoryError {
        guard case .invalidConfiguration(let message) = error else {
            Issue.record("unexpected ModelFactoryError: \(error)")
            return
        }
        #expect(message.contains(expectedMessage))
    } catch {
        Issue.record("unexpected error: \(error)")
    }
}
