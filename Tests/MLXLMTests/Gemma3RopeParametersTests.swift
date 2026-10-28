// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable @_spi(GemmaEncoder) import MLXLLM
@testable import MLXVLM

// Tests that Gemma 3 configs read the global-layer RoPE scaling from `rope_parameters`, which
// newer `transformers` exports (e.g. TranslateGemma) write in place of `rope_scaling`.

private let perLayerType = """
    "rope_scaling": null,
    "rope_parameters": {
      "full_attention": {"rope_type": "linear", "factor": 8.0},
      "sliding_attention": {"rope_type": "default"}
    }
    """

private let flat = """
    "rope_scaling": null,
    "rope_parameters": {"rope_type": "linear", "factor": 8.0}
    """

private func textConfigJSON(_ rope: String) -> String {
    """
    {
      "model_type": "gemma3_text", "hidden_size": 16, "num_hidden_layers": 6,
      "intermediate_size": 32, "num_attention_heads": 2, "num_key_value_heads": 1,
      "head_dim": 8, "vocab_size": 32, "sliding_window": 16, \(rope)
    }
    """
}

private func llmConfig(_ json: String) throws -> MLXLLM.Gemma3TextConfiguration {
    try JSONDecoder().decode(MLXLLM.Gemma3TextConfiguration.self, from: Data(json.utf8))
}

private func vlmConfig(_ json: String) throws -> MLXVLM.Gemma3TextConfiguration {
    try JSONDecoder().decode(MLXVLM.Gemma3TextConfiguration.self, from: Data(json.utf8))
}

private func isLinear8(_ scaling: [String: StringOrNumber]?) -> Bool {
    scaling?["rope_type"] == .string("linear") && scaling?["factor"]?.asFloat() == 8
}

struct Gemma3RopeParametersTests {

    @Test("LLM config reads rope_parameters keyed by layer type")
    func llmPerLayerType() throws {
        #expect(isLinear8(try llmConfig(textConfigJSON(perLayerType)).ropeScaling))
    }

    @Test("LLM config reads flat rope_parameters")
    func llmFlat() throws {
        #expect(isLinear8(try llmConfig(textConfigJSON(flat)).ropeScaling))
    }

    @Test("LLM config reads rope_parameters under text_config")
    func llmNestedTextConfig() throws {
        let json = #"{"model_type": "gemma3", "text_config": \#(textConfigJSON(perLayerType))}"#
        #expect(isLinear8(try llmConfig(json).ropeScaling))
    }

    @Test("rope_scaling still wins when both keys are present")
    func llmLegacyRopeScaling() throws {
        let both = """
            "rope_scaling": {"rope_type": "linear", "factor": 8.0},
            "rope_parameters": {"rope_type": "linear", "factor": 2.0}
            """
        #expect(isLinear8(try llmConfig(textConfigJSON(both)).ropeScaling))
        #expect(try llmConfig(textConfigJSON(#""rope_scaling": null"#)).ropeScaling == nil)
    }

    @Test("Global layers apply the rope_parameters factor; sliding layers do not")
    func llmGlobalLayerRope() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = Gemma3TextModel(try llmConfig(textConfigJSON(perLayerType)))
            let x = MLXRandom.normal([1, 2, 3, 8])
            let offset = 100

            let global = model.model.layers[5].selfAttention.rope
            let expected = RoPE(dimensions: 8, base: 1_000_000, scale: 1 / 8)
            #expect(
                allClose(global(x, offset: offset), expected(x, offset: offset)).item(Bool.self))

            let sliding = model.model.layers[0].selfAttention.rope
            let unscaled = RoPE(dimensions: 8, base: 10_000)
            #expect(
                allClose(sliding(x, offset: offset), unscaled(x, offset: offset)).item(Bool.self))
        }
    }

    @Test("VLM text config reads rope_parameters in both forms")
    func vlm() throws {
        #expect(isLinear8(try vlmConfig(textConfigJSON(perLayerType)).ropeScaling))
        #expect(isLinear8(try vlmConfig(textConfigJSON(flat)).ropeScaling))
        #expect(try vlmConfig(textConfigJSON(#""rope_scaling": null"#)).ropeScaling == nil)
    }
}
