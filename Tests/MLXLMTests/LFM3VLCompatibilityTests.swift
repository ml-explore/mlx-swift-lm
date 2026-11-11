import Foundation
import MLXLMCommon
import XCTest

@testable import MLXVLM

final class LFM3VLCompatibilityTests: XCTestCase {
    private func configuration(attentionHeads: Int = 2) throws -> LFM3VLConfiguration {
        let json = """
            {"text_config": {"model_type":"lfm3", "vocab_size":32,
              "hidden_size":16, "num_hidden_layers":2, "num_attention_heads":\(attentionHeads),
              "num_key_value_heads":1, "head_dim":8, "intermediate_size":24,
              "layer_types":["conv","full_attention"]},
             "vision_config": {"hidden_size":16, "intermediate_size":24,
              "num_attention_heads":2, "num_key_value_heads":1, "patch_size":2,
              "temporal_patch_size":2, "spatial_merge_size":2, "in_channels":3,
              "conv_kernel_size":3, "norm_eps":0.00001, "num_hidden_layers":2,
              "layer_types":["conv","full_attention"],
              "rope_parameters":{"rope_type":"axial","rope_theta":1000000}},
             "image_token_id":10, "video_token_id":11, "downsample_factor":2,
             "projector_hidden_size":24, "projector_bias":true,
             "projector_layer_norm_eps":0.00001, "projector_hidden_act":"gelu"}
            """
        return try JSONDecoder().decode(LFM3VLConfiguration.self, from: Data(json.utf8))
    }

    private struct ProbeTokenizer: Tokenizer {
        var brokenText: String?
        var wrongImageID = false
        var splitVideoToken = false
        let bosToken: String? = nil
        let eosToken: String? = nil
        let unknownToken: String? = nil

        func convertTokenToId(_ token: String) -> Int? {
            let vocabulary = [
                "<|startoftext|>": 124894, ".reduce": 96744, "line": 1421, ".čĊ": 64122,
                "This": 1942, "<|image|>": wrongImageID ? 12 : 10, "<|video|>": 11,
            ]
            return vocabulary[token]
        }
        func encode(text: String, addSpecialTokens: Bool) -> [Int] {
            if text == brokenText || (text == "<|video|>" && splitVideoToken) { return [1, 2] }
            if text == "line.\r\nThis" { return [1421, 64122, 1942] }
            return convertTokenToId(text).map { [$0] } ?? []
        }
        func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { "" }
        func convertIdToToken(_ id: Int) -> String? { nil }
        func applyChatTemplate(
            messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
            additionalContext: [String: any Sendable]?
        ) throws -> [Int] { [] }
    }

    func testSharedTokenizerValidationThroughBaseLanguageModel() throws {
        let model: any BaseLanguageModel = LFM3VL(try configuration())
        XCTAssertNoThrow(try model.validate(tokenizer: ProbeTokenizer()))
        for text in [".reduce", "line.\r\nThis"] {
            XCTAssertThrowsError(try model.validate(tokenizer: ProbeTokenizer(brokenText: text))) {
                XCTAssertTrue($0.localizedDescription.contains("#397 and #398"))
            }
        }
    }

    func testMediaTokenContract() throws {
        let model: any BaseLanguageModel = LFM3VL(try configuration())
        XCTAssertThrowsError(try model.validate(tokenizer: ProbeTokenizer(wrongImageID: true)))
        XCTAssertThrowsError(try model.validate(tokenizer: ProbeTokenizer(splitVideoToken: true)))
    }

    func testNestedTextConfigurationUsesSharedValidation() {
        XCTAssertThrowsError(try configuration(attentionHeads: 0))
        XCTAssertThrowsError(try configuration(attentionHeads: -1))
        XCTAssertNoThrow(try configuration(attentionHeads: 3))
    }
}
