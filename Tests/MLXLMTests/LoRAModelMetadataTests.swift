// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXLLM

final class LoRAModelMetadataTests: XCTestCase {
    func testFactoryUsesRegisteredModelLoRAMetadata() async throws {
        let typeRegistry = ModelTypeRegistry<LanguageModel>()
        await typeRegistry.registerModelType("metadata_test") { _ in
            MetadataTestModel()
        }
        let factory = LLMModelFactory(
            typeRegistry: typeRegistry,
            modelRegistry: AbstractModelRegistry()
        )

        let result = try await factory.loraMetadata(
            configurationData: Data(#"{"model_type":"metadata_test"}"#.utf8)
        )
        let metadata = try XCTUnwrap(result)

        XCTAssertEqual(metadata.layerCount, 1)
        XCTAssertEqual(metadata.defaultKeys, ["runtime.a_proj", "runtime.z_proj"])
    }

    func testBuiltInModelMetadataUsesRuntimeModulePaths() async throws {
        let configuration = Data(
            """
            {
              "model_type": "qwen3",
              "hidden_size": 16,
              "num_hidden_layers": 2,
              "intermediate_size": 32,
              "num_attention_heads": 2,
              "num_key_value_heads": 1,
              "head_dim": 8,
              "rms_norm_eps": 1e-5,
              "vocab_size": 64
            }
            """.utf8
        )

        let result = try await LLMModelFactory.shared.loraMetadata(
            configurationData: configuration
        )
        let metadata = try XCTUnwrap(result)

        XCTAssertEqual(metadata.layerCount, 2)
        XCTAssertEqual(
            metadata.defaultKeys,
            [
                "mlp.down_proj",
                "mlp.gate_proj",
                "mlp.up_proj",
                "self_attn.k_proj",
                "self_attn.o_proj",
                "self_attn.q_proj",
                "self_attn.v_proj",
            ]
        )
    }
}

private final class MetadataTestModel: Module, LanguageModel, LoRAModel,
    KVCacheDimensionProvider
{
    private let layer = Module()

    let kvHeads: [Int] = []

    var loraLayers: [Module] { [layer] }
    var loraDefaultKeys: [String] { ["runtime.z_proj", "runtime.a_proj"] }

    func prepare(
        _ input: LMInput, cache: [KVCache], state: LMOutput.State?, prefill: PrefillParameters
    ) throws -> PrepareResult {
        .tokens(input.text)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        inputs
    }
}
