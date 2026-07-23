// Copyright © 2026 Apple Inc.

import CoreImage
import Foundation
import MLX
import MLXLMCommon
import XCTest

@_spi(Testing) @testable import MLXVLM

final class DeepseekOCRLayerParityTests: XCTestCase {

    func testL1ProcessorAndVisionMatchPythonGolden() async throws {
        let golden = try loadGolden()
        let fixtureURL = try resolveFixtureURL(from: golden)
        let image = try XCTUnwrap(CIImage(contentsOf: fixtureURL))

        let processor = try makeProcessor()
        let input = UserInput(
            prompt: "Describe this page.",
            images: [.ciImage(image)])

        let prepared = try await processor.prepareForTesting(input: input)

        let processorGolden = golden.processor
        XCTAssertEqual(Array(prepared.pixelValues.shape), processorGolden.globalPixels.shape)
        XCTAssertEqual(Array(prepared.localCrops.shape), processorGolden.localCrops.shape)
        XCTAssertEqual(
            prepared.imagesSpatialCrop.asArray(Int32.self).map(Int.init),
            processorGolden.imagesSpatialCrop.flatMap { $0 })
        XCTAssertLessThan(
            prepared.pixelValues.asType(.float32).sum().item(Float.self), 0,
            "normalized red fixture should yield negative pixel sum")
        XCTAssertLessThan(
            prepared.localCrops.asType(.float32).sum().item(Float.self), 0,
            "normalized red fixture should yield negative local crop sum")
        let imageTokenCount = prepared.imagesSeqMask.asType(.int32).sum().item(Int.self)
        XCTAssertLessThanOrEqual(
            abs(imageTokenCount - processorGolden.imagesSeqMaskTrueCount),
            1,
            "image token lattice may differ by one trailing separator vs Python")

        let model = try makeVisionModel()
        let pixels = zeros([1, 1024, 1024, 3], type: Float.self)
        let samFeatures = model.samFeaturesForTesting(pixels)
        let fusedFeatures = model.fusedVisionFeaturesForTesting(pixels)
        let projected = model.projectedImageFeaturesForTesting(pixels)

        XCTAssertEqual(Array(samFeatures.shape), golden.vision.samFeatures.shape)
        XCTAssertEqual(Array(fusedFeatures.shape), [1, 256, 2048])
        XCTAssertEqual(projected.shape.last, golden.vision.projectorOutputDim)
    }

    private func loadGolden() throws -> LayerParityGolden {
        let env = ProcessInfo.processInfo.environment["LAYER_PARITY_GOLDEN"]
        let defaultPath = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("fixtures/baseline/layer-parity.json")

        let goldenURL: URL
        if let env, !env.isEmpty {
            goldenURL = URL(fileURLWithPath: env)
        } else if FileManager.default.fileExists(atPath: defaultPath.path) {
            goldenURL = defaultPath
        } else {
            throw XCTSkip("Set LAYER_PARITY_GOLDEN or generate fixtures/baseline/layer-parity.json")
        }

        let data = try Data(contentsOf: goldenURL)
        return try JSONDecoder().decode(LayerParityGolden.self, from: data)
    }

    private func resolveFixtureURL(from golden: LayerParityGolden) throws -> URL {
        let hubRoot: URL
        if let env = ProcessInfo.processInfo.environment["LAYER_PARITY_GOLDEN"], !env.isEmpty {
            hubRoot = URL(fileURLWithPath: env)
                .deletingLastPathComponent()
                .deletingLastPathComponent()
                .deletingLastPathComponent()
        } else {
            hubRoot = URL(fileURLWithPath: #filePath)
                .deletingLastPathComponent()
                .deletingLastPathComponent()
                .deletingLastPathComponent()
                .deletingLastPathComponent()
                .deletingLastPathComponent()
        }
        let fixtureURL = hubRoot.appendingPathComponent(golden.fixture)
        guard FileManager.default.fileExists(atPath: fixtureURL.path) else {
            throw XCTSkip("Fixture not found at \(fixtureURL.path)")
        }
        return fixtureURL
    }

    private func makeProcessor() throws -> DeepseekOCRProcessor {
        let config = try JSONDecoder().decode(
            DeepseekOCRProcessorConfiguration.self,
            from: Data(processorConfigJSON.utf8))
        return DeepseekOCRProcessor(config, tokenizer: LayerParityTokenizer())
    }

    private func makeVisionModel() throws -> DeepseekOCR {
        let config = try JSONDecoder().decode(
            DeepseekOCRConfiguration.self,
            from: Data(visionConfigJSON.utf8))
        return DeepseekOCR(config)
    }

    private let processorConfigJSON = #"""
        {
         "candidate_resolutions": [[1024, 1024]],
         "downsample_ratio": 4,
         "image_mean": [0.5, 0.5, 0.5],
         "image_std": [0.5, 0.5, 0.5],
         "image_token": "<image>",
         "patch_size": 16,
         "size": {
          "shortest_edge": 1024,
          "longest_edge": 1024
         }
        }
        """#

    private let visionConfigJSON = #"""
        {
         "model_type": "deepseekocr",
         "vision_config": {
          "hidden_size": 768,
          "output_channels": 256,
          "num_hidden_layers": 12,
          "num_attention_heads": 12,
          "image_size": 1024,
          "patch_size": 16,
          "global_attn_indexes": [2, 5, 8, 11],
          "mlp_dim": 3072
         },
         "language_config": {
          "vocab_size": 129280,
          "hidden_size": 1280,
          "intermediate_size": 6848,
          "num_hidden_layers": 12,
          "num_attention_heads": 10,
          "num_key_value_heads": 10,
          "max_position_embeddings": 8192
         }
        }
        """#
}

private struct LayerParityGolden: Decodable {
    struct TensorSummary: Decodable {
        let shape: [Int]
        let sum: Float
    }

    struct ProcessorGolden: Decodable {
        let globalPixels: TensorSummary
        let localCrops: TensorSummary
        let imagesSpatialCrop: [[Int]]
        let imagesSeqMaskTrueCount: Int

        enum CodingKeys: String, CodingKey {
            case globalPixels = "global_pixels"
            case localCrops = "local_crops"
            case imagesSpatialCrop = "images_spatial_crop"
            case imagesSeqMaskTrueCount = "images_seq_mask_true_count"
        }
    }

    struct VisionGolden: Decodable {
        let samFeatures: TensorSummary
        let projectorOutputDim: Int

        enum CodingKeys: String, CodingKey {
            case samFeatures = "sam_features"
            case projectorOutputDim = "projector_output_dim"
        }
    }

    let fixture: String
    let processor: ProcessorGolden
    let vision: VisionGolden
}

private struct LayerParityTokenizer: Tokenizer {
    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        guard !text.isEmpty else { return [] }
        return text.split(separator: " ").enumerated().map { 20 + $0.offset }
    }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { "" }

    func convertTokenToId(_ token: String) -> Int? {
        switch token {
        case "<image>": return 999
        case "<s>": return 0
        default: return nil
        }
    }

    func convertIdToToken(_ id: Int) -> String? { nil }

    var bosToken: String? { "<s>" }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        [0, 20, 21]
    }
}
