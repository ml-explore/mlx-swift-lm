// Copyright © 2025 Apple Inc.

import Foundation
import MLXLMCommon
import XCTest

public class BaseConfigurationTests: XCTestCase {

    func testQuantization() throws {
        let json =
            """
            {
                "model_type": "Test",
                "quantization": {
                    "group_size": 128,
                    "bits": 4
                }
            }
            """

        let config = try JSONDecoder().decode(
            BaseConfiguration.self, from: json.data(using: .utf8)!)

        XCTAssertEqual(
            config.perLayerQuantization?.quantization(layer: "x"), .init(groupSize: 128, bits: 4))
    }

    func testHeterogenousQuantization() throws {
        // from https://huggingface.co/mlx-community/Qwen3-1.7B-4bit-AWQ/blob/main/config.json#L20
        let json =
            """
            {
                "model_type": "Test",
                "quantization": {
                    "group_size": 64,
                    "bits": 4,
                    "model.embed_tokens": {
                        "group_size": 32,
                        "bits": 4
                    },
                    "model.layers.0.self_attn.q_norm": false,
                    "true_layer": true
                }
            }
            """

        let config = try JSONDecoder().decode(
            BaseConfiguration.self, from: json.data(using: .utf8)!)

        // a random layer -- no specific configuration gets default
        XCTAssertEqual(
            config.perLayerQuantization?.quantization(layer: "x"),
            .init(groupSize: 64, bits: 4))

        // layer with an override
        XCTAssertEqual(
            config.perLayerQuantization?.quantization(layer: "model.embed_tokens"),
            .init(groupSize: 32, bits: 4))

        // layer with an override -- not quant
        XCTAssertNil(
            config.perLayerQuantization?.quantization(layer: "model.layers.0.self_attn.q_norm"))

        // layer with an override -- true, use the default
        XCTAssertEqual(
            config.perLayerQuantization?.quantization(layer: "true_layer"),
            .init(groupSize: 64, bits: 4))
    }

    func testNestedTextConfigurationEOSTokenIds() throws {
        let json =
            """
            {
                "model_type": "qwen3_5",
                "text_config": {
                    "eos_token_id": 248044
                }
            }
            """

        let config = try JSONDecoder().decode(
            BaseConfiguration.self, from: json.data(using: .utf8)!)

        XCTAssertNil(config.eosTokenIds)
        XCTAssertEqual(config.effectiveEOSTokenIds, [248044])
    }

    func testRootEOSTokenIdsTakePrecedenceOverNestedValues() throws {
        let json =
            """
            {
                "model_type": "qwen3_5",
                "eos_token_id": 248046,
                "text_config": {
                    "eos_token_id": [248044, 248046]
                }
            }
            """

        let config = try JSONDecoder().decode(
            BaseConfiguration.self, from: json.data(using: .utf8)!)

        XCTAssertEqual(config.eosTokenIds?.values, [248046])
        XCTAssertEqual(config.effectiveEOSTokenIds, [248046])
    }

    /// Upstream `quantization_config` formats map as mlx-lm's `load_model` maps them.
    func testUpstreamQuantizationConfig() throws {
        func quantization(_ config: String, nested: Bool = false) throws
            -> BaseConfiguration.Quantization?
        {
            let entry = #""quantization_config": \#(config)"#
            let json =
                nested
                ? #"{"model_type": "Test", "text_config": {\#(entry)}}"#
                : #"{"model_type": "Test", \#(entry)}"#
            return try JSONDecoder().decode(BaseConfiguration.self, from: Data(json.utf8))
                .perLayerQuantization?.quantization
        }

        let mxfp8 = BaseConfiguration.Quantization(groupSize: 32, bits: 8, mode: .mxfp8)
        let ue8m0 = #"{"quant_method": "fp8", "scale_fmt": "ue8m0", "weight_block_size": [32, 32]}"#
        XCTAssertEqual(try quantization(ue8m0), mxfp8)
        XCTAssertEqual(try quantization(ue8m0, nested: true), mxfp8)
        XCTAssertEqual(try quantization(#"{"quant_method": "mxfp8"}"#), mxfp8)
        XCTAssertEqual(
            try quantization(#"{"quant_method": "mxfp4"}"#),
            .init(groupSize: 32, bits: 4, mode: .mxfp4))
        XCTAssertEqual(
            try quantization(
                #"{"quant_method": "compressed-tensors", "format": "pack-quantized"}"#),
            .init(groupSize: 32, bits: 4, mode: .affine))
        XCTAssertEqual(
            try quantization(
                #"{"quant_method": "compressed-tensors", "format": "nvfp4-pack-quantized"}"#),
            .init(groupSize: 16, bits: 4, mode: .nvfp4))

        // Block fp8 dequantizes in sanitize, and unknown or malformed formats are not mapped.
        XCTAssertNil(
            try quantization(#"{"quant_method": "fp8", "weight_block_size": [128, 128]}"#))
        XCTAssertNil(
            try quantization(
                #"{"quant_method": "compressed-tensors", "format": "float-quantized"}"#))
        XCTAssertNil(try quantization(#"{"quant_method": "paroquant", "bits": 4}"#))
        XCTAssertNil(try quantization(#"{"quant_method": 7}"#))
    }

    /// An MLX `quantization` entry wins over the upstream `quantization_config`.
    func testQuantizationWinsOverUpstreamConfig() throws {
        let json = """
            {
                "model_type": "Test",
                "quantization": {"group_size": 64, "bits": 4},
                "quantization_config": {"quant_method": "mxfp8"}
            }
            """
        let config = try JSONDecoder().decode(BaseConfiguration.self, from: Data(json.utf8))
        XCTAssertEqual(
            config.perLayerQuantization?.quantization(layer: "x"), .init(groupSize: 64, bits: 4))
    }

}
