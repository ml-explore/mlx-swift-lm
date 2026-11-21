// Copyright © 2026 Apple Inc.
//
// Regression tests for Qwen3-VL-MoE attention masks after cache rotation.
// Same bug class as Qwen3VLContinuationTests: once a capacity-limited cache
// starts evicting, the mask must span the keys the cache still presents, not
// the absolute offset. The model previously built its mask from the offset and
// sliced it to an offset-derived length, so a warm multi-token continuation
// described more keys than the cache retained and SDPA rejected the shapes.
// M-RoPE positions stay absolute; only the mask width is under test.
// The image prefill tests below cover the image merge and the deepstack rows.

import Foundation
import MLX
import MLXLMCommon
import MLXVLM
import XCTest

final class Qwen3VLMoEContinuationTests: XCTestCase {

    // MARK: - Tiny model

    private func makeTinyModel() throws -> Qwen3VLMoE {
        let json = """
            {
                "model_type": "qwen3_vl_moe",
                "image_token_id": 500,
                "video_token_id": 501,
                "vision_start_token_id": 502,
                "vision_end_token_id": 503,
                "vocab_size": 512,
                "text_config": {
                    "model_type": "qwen3_vl_moe",
                    "hidden_size": 64,
                    "num_hidden_layers": 4,
                    "intermediate_size": 128,
                    "num_attention_heads": 4,
                    "num_key_value_heads": 2,
                    "head_dim": 16,
                    "vocab_size": 512,
                    "max_position_embeddings": 4096,
                    "rms_norm_eps": 1e-6,
                    "rope_theta": 100000.0,
                    "num_experts": 2,
                    "num_experts_per_tok": 1,
                    "decoder_sparse_step": 1,
                    "mlp_only_layers": [],
                    "moe_intermediate_size": 32,
                    "rope_scaling": {
                        "rope_type": "default",
                        "mrope_section": [4, 2, 2]
                    }
                },
                "vision_config": {
                    "model_type": "qwen3_vl_moe",
                    "depth": 2,
                    "hidden_size": 32,
                    "intermediate_size": 64,
                    "out_hidden_size": 64,
                    "num_heads": 2,
                    "patch_size": 16,
                    "spatial_merge_size": 2,
                    "temporal_patch_size": 2,
                    "num_position_embeddings": 64,
                    "deepstack_visual_indexes": [0, 1]
                }
            }
            """
        let config = try JSONDecoder().decode(
            Qwen3VLMoEConfiguration.self, from: Data(json.utf8))
        // Pin the initializer weights. Task-local rather than MLXRandom.seed:
        // parallel tests must not share (or perturb) the global random stream.
        return withRandomState(MLXRandom.RandomState(seed: 1)) { Qwen3VLMoE(config) }
    }

    /// The shared continuation fixtures, configured for Qwen3-VL-MoE's token ids.
    private let continuation = ContinuationAssertions(
        imageTokenId: 500, visionStartTokenId: 502)

    private func makeImage() -> LMInput.ProcessedImage { continuation.image() }

    private func imageRun() -> MLXArray {
        MLXArray([Int32](repeating: 500, count: 4)).expandedDimensions(axis: 0)
    }

    private func visionStart() -> MLXArray {
        MLXArray([Int32(502)]).expandedDimensions(axis: 0)
    }

    private func textTokens(_ count: Int, seed: Int32 = 0) -> MLXArray {
        continuation.textTokens(count, seed: seed)
    }
    private func lastLogits(_ result: PrepareResult) throws -> (MLXArray, LMOutput.State?) {
        try continuation.lastLogits(result)
    }
    private func maxAbsDiff(_ a: MLXArray, _ b: MLXArray) -> Float {
        continuation.maxAbsDiff(a, b)
    }

    // MARK: - Warm continuation

    /// Prefill, decode, then continue warm on the rotated cache: the warm
    /// multi-token forward must match token-by-token decode on a cache copy.
    /// Before the mask fix this trapped: the mask described more keys than the
    /// cache retained.
    func testWarmContinuationAfterCacheRotationMatchesDecode() throws {
        for (capacity, suffixLength) in [(16, 6), (16, 1), (16, 23), (1, 6)] {
            let model = try makeTinyModel()
            let parameters = GenerateParameters(
                kvCache: KVCacheConfiguration(
                    capacity: try .init(maxTokens: capacity, preservedPrefixTokens: 0)))
            let cache = try model.newCache(parameters: parameters)
            let (_, initialState) = try lastLogits(
                model.prepare(
                    LMInput(text: .init(tokens: textTokens(capacity))), cache: cache, state: nil,
                    prefill: .init(chunking: .unchunked)))
            var state = initialState
            let decoded = textTokens(5, seed: 3)
            for index in 0 ..< decoded.dim(1) {
                let output = model(
                    .init(tokens: decoded[0..., index ..< index + 1]), cache: cache, state: state)
                eval(output.logits)
                state = output.state
            }
            XCTAssertTrue(cache.allSatisfy { $0.offset == capacity + 5 })

            let referenceCache = cache.map { $0.copy() }
            let suffix = textTokens(suffixLength, seed: 9)
            var referenceState = state
            var expected = [MLXArray]()
            for index in 0 ..< suffix.dim(1) {
                let output = model(
                    .init(tokens: suffix[0..., index ..< index + 1]), cache: referenceCache,
                    state: referenceState)
                eval(output.logits)
                expected.append(output.logits)
                referenceState = output.state
            }

            let result = try model.prepare(
                LMInput(text: .init(tokens: suffix)), cache: cache, state: state,
                prefill: .init(chunking: .unchunked))
            guard case .logits(let output) = result else {
                return XCTFail("expected logits from warm continuation")
            }
            XCTAssertLessThanOrEqual(
                maxAbsDiff(output.logits, concatenated(expected, axis: 1)),
                MatmulPrecision.splitTolerance)
            XCTAssertTrue(cache.allSatisfy { $0.offset == capacity + 5 + suffixLength })
        }
    }

    /// The image path supplies M-RoPE position ids, the branch that also
    /// derived the mask width from the offset. A warm text continuation after
    /// an image turn must survive eviction and stay finite.
    func testImageContinuationAfterCacheRotationWithPreservedCachePrefix() throws {
        try withRandomState(MLXRandom.RandomState(seed: 47)) {
            let model = try makeTinyModel()
            let parameters = GenerateParameters(
                kvCache: KVCacheConfiguration(
                    capacity: try .init(maxTokens: 16, preservedPrefixTokens: 4)))
            let cache = try model.newCache(parameters: parameters)
            let prompt = concatenated(
                [textTokens(6), visionStart(), imageRun(), textTokens(12, seed: 2)], axis: 1)
            let (initialLogits, state) = try lastLogits(
                model.prepare(
                    LMInput(text: .init(tokens: prompt), image: makeImage()), cache: cache,
                    state: nil, prefill: .init()))
            eval(initialLogits)
            XCTAssertEqual(cache.first?.offset, 23)

            let (logits, _) = try lastLogits(
                model.prepare(
                    LMInput(text: .init(tokens: textTokens(9, seed: 5))), cache: cache,
                    state: state, prefill: .init()))
            XCTAssertTrue(logits.asArray(Float.self).allSatisfy(\.isFinite))
            XCTAssertEqual(cache.first?.offset, 32)
        }
    }

    // MARK: - Image prefill

    /// Two images of different sizes, with deepstack on. The image rows and deepstack
    /// rows must land only on the image tokens: the text before the first image must
    /// match a text-only prefill of that prefix.
    func testTwoImagePrefillLeavesPrecedingTextUnchanged() throws {
        try withRandomState(MLXRandom.RandomState(seed: 53)) {
            let model = try makeTinyModel()
            let images = LMInput.ProcessedImage(
                pixels: MLXRandom.normal([16 + 32, 3 * 2 * 16 * 16]),
                frames: [THW(1, 4, 4), THW(1, 4, 8)])
            let prefix = textTokens(7)
            let secondRun = MLXArray([Int32](repeating: 500, count: 8)).expandedDimensions(
                axis: 0)
            let prompt = concatenated(
                [
                    prefix, visionStart(), imageRun(), textTokens(5, seed: 3), visionStart(),
                    secondRun, textTokens(6, seed: 9),
                ], axis: 1)

            let result = try model.prepare(
                LMInput(text: .init(tokens: prompt), image: images),
                cache: try model.newCache(parameters: nil), state: nil,
                prefill: .init(chunking: .unchunked))
            guard case .logits(let output) = result else {
                return XCTFail("expected logits from image prefill")
            }
            XCTAssertEqual(output.logits.shape, [1, prompt.dim(1), 512])
            XCTAssertTrue(output.logits.asArray(Float.self).allSatisfy(\.isFinite))

            let textResult = try model.prepare(
                LMInput(text: .init(tokens: prefix)),
                cache: try model.newCache(parameters: nil), state: nil,
                prefill: .init(chunking: .unchunked))
            guard case .logits(let textOutput) = textResult else {
                return XCTFail("expected logits from text prefill")
            }
            XCTAssertLessThanOrEqual(
                maxAbsDiff(output.logits[0..., 0 ..< prefix.dim(1), 0...], textOutput.logits),
                MatmulPrecision.splitTolerance)
        }
    }

    /// With deepstack on, an image in turn 1 continued by text in turn 2 must match
    /// one full prefill.
    func testWarmImageContinuationMatchesFullPrefill() throws {
        try continuation.assertWarmImageContinuation(try makeTinyModel())
    }

    /// More image features than placeholder tokens must throw, not trap in the scatter.
    func testImageFeatureTokenMismatchThrows() throws {
        let model = try makeTinyModel()
        let tooFewPlaceholders = MLXArray([Int32](repeating: 500, count: 3))
            .expandedDimensions(axis: 0)
        let prompt = concatenated([textTokens(4), visionStart(), tooFewPlaceholders], axis: 1)

        XCTAssertThrowsError(
            try model.prepare(
                LMInput(text: .init(tokens: prompt), image: makeImage()),
                cache: try model.newCache(parameters: nil), state: nil,
                prefill: .init(chunking: .unchunked)))
    }
}
