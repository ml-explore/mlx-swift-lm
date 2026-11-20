// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXVLM
import Testing

/// `Idefics3.prepare` must reject image inputs the vision tower and the
/// image-token merge cannot handle by throwing, not by trapping (#762).
@Suite("Idefics3 prepare input validation", .serialized)
struct Idefics3PrepareValidationTests {

    /// A tiny random Idefics3 whose vision tower expects 16x16 images
    /// (16 patches of 4x4, so 4 image features per image after the 2x2 shuffle).
    private func model() async throws -> any VLMModel {
        let text: [String: Any] = [
            "model_type": "llama", "hidden_size": 32, "num_hidden_layers": 2,
            "intermediate_size": 64, "num_attention_heads": 2, "num_key_value_heads": 1,
            "head_dim": 16, "vocab_size": 100, "rms_norm_eps": 1e-5,
            "rope_theta": 10000, "max_position_embeddings": 512,
        ]
        let configuration: [String: Any] = [
            "model_type": "idefics3", "image_token_index": 10, "image_token_id": 10,
            "vocab_size": 100, "text_config": text, "scale_factor": 2,
            "vision_config": [
                "model_type": "siglip_vision_model", "hidden_size": 32,
                "num_hidden_layers": 1, "num_attention_heads": 2, "intermediate_size": 64,
                "patch_size": 4, "image_size": 16,
            ],
        ]
        let model = try await VLMTypeRegistry.shared.createModel(
            configuration: JSONSerialization.data(withJSONObject: configuration),
            modelType: "idefics3")
        model.train(false)
        eval(model)
        return try #require(model as? any VLMModel)
    }

    private func prepare(
        _ model: any VLMModel, imageTokens: Int, images: Int = 1, height: Int, width: Int,
        channels: Int = 3
    ) throws -> PrepareResult {
        let ids: [Int32] = [1] + Array(repeating: 10, count: imageTokens) + [2]
        let mask = MLXArray.ones([1, ids.count], type: Int32.self)
        let image = LMInput.ProcessedImage(
            pixels: MLXArray.ones([images, height, width, channels]),
            frames: Array(repeating: .init(1, height, width), count: images))
        let result = try model.prepare(
            LMInput(text: .init(tokens: MLXArray(ids), mask: mask), image: image),
            cache: model.newCache(parameters: nil), state: nil, prefill: .init())
        if case .logits(let output) = result { eval(output.logits) }
        return result
    }

    @Test(
        "Correctly sized images prepare and give finite logits",
        arguments: [(images: 1, imageTokens: 4), (images: 2, imageTokens: 8)])
    func validImages(images: Int, imageTokens: Int) async throws {
        let model = try await model()
        let result = try prepare(
            model, imageTokens: imageTokens, images: images, height: 16, width: 16)
        guard case .logits(let output) = result else {
            Issue.record("expected logits, got \(result)")
            return
        }
        #expect(output.logits.shape == [1, 1, 100])
        #expect(!(isNaN(output.logits) .|| isInf(output.logits)).any().item(Bool.self))
    }

    @Test("An image with the wrong patch count throws")
    func wrongImageSize() async throws {
        let model = try await model()
        // 12x12 gives 9 patches against 16 position embeddings.
        #expect(throws: VLMError.self) {
            try prepare(model, imageTokens: 4, height: 12, width: 12)
        }
    }

    @Test("A non-square patch grid throws even with the right patch count")
    func nonSquareImage() async throws {
        let model = try await model()
        // 8x32 gives 2x8 = 16 patches, but not the 4x4 grid the position embeddings cover.
        #expect(throws: VLMError.self) {
            try withError {
                try prepare(model, imageTokens: 4, height: 8, width: 32)
            }
        }
    }

    @Test("An image with the wrong channel count throws")
    func wrongChannelCount() async throws {
        let model = try await model()
        #expect(throws: VLMError.self) {
            try withError {
                try prepare(model, imageTokens: 4, height: 16, width: 16, channels: 4)
            }
        }
    }

    @Test("An image with the wrong patch count throws inside withError")
    func wrongImageSizeWithError() async throws {
        let model = try await model()
        #expect(throws: VLMError.self) {
            try withError {
                try prepare(model, imageTokens: 4, height: 12, width: 12)
            }
        }
    }

    @Test("An image-token count that does not match the image features throws")
    func wrongImageTokenCount() async throws {
        let model = try await model()
        for count in [1, 3, 5, 8] {
            #expect(throws: VLMError.self) {
                try withError {
                    try prepare(model, imageTokens: count, height: 16, width: 16)
                }
            }
        }
    }
}
