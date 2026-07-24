// Copyright © 2026 Apple Inc.
//
// Equivalence tests for Qwen2.5-VL and Qwen2-VL windowed prefill and warm
// (cached-prefix) continuation, on tiny random-weight models so they run in CI
// without downloads. The invariant under test mirrors Qwen35ContinuationTests:
// however a prompt reaches the KV cache — one shot, windowed chunks, or split
// across a warm continuation — the next-token logits must match, because M-RoPE
// positions must be anchored at the cache offset (plus the carried rope delta),
// never restarted at zero.

import Foundation
import MLX
import MLXLMCommon
import MLXVLM
import XCTest

final class Qwen25VLContinuationTests: XCTestCase {

    // MARK: - Tiny models

    private func makeTinyQwen25VL() throws -> Qwen25VL {
        let json = """
            {
                "model_type": "qwen2_5_vl",
                "hidden_size": 64,
                "num_hidden_layers": 2,
                "intermediate_size": 128,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "rms_norm_eps": 1e-6,
                "vocab_size": 512,
                "max_position_embeddings": 4096,
                "rope_theta": 1000000.0,
                "rope_traditional": false,
                "tie_word_embeddings": false,
                "sliding_window": 32768,
                "use_sliding_window": false,
                "max_window_layers": 2,
                "image_token_id": 500,
                "video_token_id": 501,
                "vision_start_token_id": 502,
                "vision_end_token_id": 503,
                "vision_token_id": 504,
                "rope_scaling": {
                    "type": "mrope",
                    "mrope_section": [2, 3, 3]
                },
                "vision_config": {
                    "depth": 2,
                    "hidden_size": 32,
                    "intermediate_size": 64,
                    "out_hidden_size": 64,
                    "num_heads": 2,
                    "patch_size": 16,
                    "spatial_patch_size": 16,
                    "spatial_merge_size": 2,
                    "temporal_patch_size": 2,
                    "window_size": 64,
                    "fullatt_block_indexes": [1],
                    "tokens_per_second": 2,
                    "in_chans": 3
                }
            }
            """
        let config = try JSONDecoder().decode(
            Qwen25VLConfiguration.self, from: Data(json.utf8))
        return Qwen25VL(config)
    }

    private func makeTinyQwen2VL() throws -> Qwen2VL {
        let json = """
            {
                "model_type": "qwen2_vl",
                "hidden_size": 64,
                "num_hidden_layers": 2,
                "intermediate_size": 128,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "rms_norm_eps": 1e-6,
                "vocab_size": 512,
                "max_position_embeddings": 4096,
                "rope_theta": 1000000.0,
                "rope_traditional": false,
                "tie_word_embeddings": false,
                "image_token_id": 500,
                "video_token_id": 501,
                "rope_scaling": {
                    "type": "mrope",
                    "mrope_section": [2, 3, 3]
                },
                "vision_config": {
                    "depth": 2,
                    "embed_dim": 32,
                    "hidden_size": 64,
                    "num_heads": 2,
                    "patch_size": 16,
                    "mlp_ratio": 2.0,
                    "spatial_patch_size": 16,
                    "spatial_merge_size": 2,
                    "temporal_patch_size": 2,
                    "in_channels": 3
                }
            }
            """
        let config = try JSONDecoder().decode(
            Qwen2VLConfiguration.self, from: Data(json.utf8))
        return Qwen2VL(config)
    }

    // MARK: - Shared config constants

    private let imageTokenId: Int32 = 500
    private let visionStartTokenId: Int32 = 502

    // MARK: - Helpers

    /// Deterministic pseudo-random plain-text tokens, away from the special
    /// ids (500...504).
    private func textTokens(_ count: Int, seed: Int32 = 0) -> MLXArray {
        var values: [Int32] = []
        for i in 0 ..< count {
            let value: Int = (i * 13 + 7 + Int(seed)) % 480
            values.append(Int32(value))
        }
        return MLXArray(values).expandedDimensions(axis: 0)
    }

    private func lastLogits(_ result: PrepareResult) throws -> (MLXArray, LMOutput.State?) {
        guard case .logits(let out) = result else {
            throw XCTSkip("expected .logits from prepare")
        }
        return (out.logits[0..., -1, 0...], out.state)
    }

    private func maxAbsDiff(_ a: MLXArray, _ b: MLXArray) -> Float {
        abs(a - b).max().item(Float.self)
    }

    private func imageInput() -> LMInput.ProcessedImage {
        // One image: grid THW (1, 4, 4), merge 2 → 4 merged tokens in text.
        let pixels = MLXRandom.normal([16, 3 * 2 * 16 * 16])
        return LMInput.ProcessedImage(pixels: pixels, frames: [THW(1, 4, 4)])
    }

    private func imageRun() -> MLXArray {
        MLXArray([Int32](repeating: imageTokenId, count: 4)).expandedDimensions(axis: 0)
    }

    private func visionStart() -> MLXArray {
        MLXArray([visionStartTokenId]).expandedDimensions(axis: 0)
    }

    // MARK: - Generic assertions

    /// A warm continuation (prefix in the cache, remainder prefilled on top —
    /// the ChatSession cross-turn flow) must produce the same next-token logits
    /// as one cold prefill of the concatenation. The decode path (token by
    /// token, state threaded) is the offset-correct control bounding the noise.
    private func assertWarmTextContinuation<M: LanguageModel>(_ model: M) throws {
        MLXRandom.seed(7)
        let t1 = textTokens(40)
        let t2 = textTokens(8, seed: 3)

        let cacheF = model.newCache(parameters: nil)
        let (logitsF, _) = try lastLogits(
            model.prepare(
                LMInput(text: .init(tokens: concatenated([t1, t2], axis: 1))),
                cache: cacheF, state: nil, windowSize: nil))

        let cacheD = model.newCache(parameters: nil)
        let (_, s0) = try lastLogits(
            model.prepare(
                LMInput(text: .init(tokens: t1)), cache: cacheD, state: nil, windowSize: nil))
        var state = s0
        var logitsD = MLXArray(0)
        for j in 0 ..< t2.dim(1) {
            let out = model(
                LMInput.Text(tokens: t2[0..., j ..< (j + 1)]), cache: cacheD, state: state)
            state = out.state
            logitsD = out.logits[0..., -1, 0...]
        }
        let noiseFloor = maxAbsDiff(logitsD, logitsF)

        let cacheW = model.newCache(parameters: nil)
        _ = try lastLogits(
            model.prepare(
                LMInput(text: .init(tokens: t1)), cache: cacheW, state: nil, windowSize: nil))
        let (logitsW, _) = try lastLogits(
            model.prepare(
                LMInput(text: .init(tokens: t2)), cache: cacheW, state: nil, windowSize: nil))

        let drift = maxAbsDiff(logitsW, logitsF)
        XCTAssertLessThanOrEqual(
            drift, max(noiseFloor * 10, 1e-3),
            "warm continuation diverged from full prefill (noise floor \(noiseFloor))")
    }

    /// With an image in turn 1, the rope delta the image accumulated must be
    /// carried into turn 2's prefill: two-turn with threaded state ≡ one-shot
    /// full prefill.
    private func assertWarmImageContinuation<M: LanguageModel>(_ model: M) throws {
        MLXRandom.seed(5)
        let image = imageInput()
        let t1 = concatenated(
            [textTokens(10), visionStart(), imageRun(), textTokens(8, seed: 5)], axis: 1)
        let t2 = textTokens(8, seed: 9)
        let full = concatenated([t1, t2], axis: 1)

        let cacheF = model.newCache(parameters: nil)
        let (logitsF, _) = try lastLogits(
            model.prepare(
                LMInput(text: .init(tokens: full), image: image), cache: cacheF, state: nil,
                windowSize: nil))

        let cacheW = model.newCache(parameters: nil)
        let (_, s1) = try lastLogits(
            model.prepare(
                LMInput(text: .init(tokens: t1), image: image), cache: cacheW, state: nil,
                windowSize: nil))
        let (logitsW, _) = try lastLogits(
            model.prepare(
                LMInput(text: .init(tokens: t2)), cache: cacheW, state: s1, windowSize: nil))

        XCTAssertLessThanOrEqual(
            maxAbsDiff(logitsW, logitsF), 1e-3,
            "state-threaded warm continuation diverged from full prefill")
    }

    /// Three-turn round trip: a warm continuation whose remainder itself
    /// contains a new image must compute that image's positions from the anchor
    /// AND hand back a resume state that positions the following turn correctly.
    private func assertImageMidContinuationResumeState<M: LanguageModel>(_ model: M) throws {
        MLXRandom.seed(3)
        let image = imageInput()
        let t1 = textTokens(12)
        let t2 = concatenated(
            [textTokens(4, seed: 2), visionStart(), imageRun(), textTokens(6, seed: 4)], axis: 1)
        let t3 = textTokens(8, seed: 6)
        let full = concatenated([t1, t2, t3], axis: 1)

        let cacheF = model.newCache(parameters: nil)
        let (logitsF, _) = try lastLogits(
            model.prepare(
                LMInput(text: .init(tokens: full), image: image), cache: cacheF, state: nil,
                windowSize: nil))

        let cacheW = model.newCache(parameters: nil)
        let (_, s1) = try lastLogits(
            model.prepare(
                LMInput(text: .init(tokens: t1)), cache: cacheW, state: nil, windowSize: nil))
        let (_, s2) = try lastLogits(
            model.prepare(
                LMInput(text: .init(tokens: t2), image: image), cache: cacheW, state: s1,
                windowSize: nil))
        let (logitsW, _) = try lastLogits(
            model.prepare(
                LMInput(text: .init(tokens: t3)), cache: cacheW, state: s2, windowSize: nil))

        XCTAssertLessThanOrEqual(
            maxAbsDiff(logitsW, logitsF), 1e-3,
            "post-image resume state positioned the following turn wrong")
    }

    /// Windowed (chunked) prefill must produce the same first-token logits as
    /// the single-shot forward on plain text.
    private func assertWindowedTextPrefill<M: LanguageModel>(_ model: M) throws {
        MLXRandom.seed(11)
        let prompt = textTokens(40)

        let cacheS = model.newCache(parameters: nil)
        let (logitsS, _) = try lastLogits(
            model.prepare(
                LMInput(text: .init(tokens: prompt)), cache: cacheS, state: nil, windowSize: nil))

        let cacheC = model.newCache(parameters: nil)
        let (logitsC, _) = try lastLogits(
            model.prepare(
                LMInput(text: .init(tokens: prompt)), cache: cacheC, state: nil, windowSize: 8))

        XCTAssertLessThanOrEqual(
            maxAbsDiff(logitsC, logitsS), 1e-3,
            "windowed prefill diverged from single-shot")
    }

    /// Windowed prefill on an image-bearing prompt whose image straddles a
    /// window boundary must match the single-shot forward.
    private func assertWindowedImagePrefill<M: LanguageModel>(_ model: M) throws {
        MLXRandom.seed(13)
        let image = imageInput()
        // Image tokens sit at positions 11...14 — straddling the 8-token window
        // boundary, the hard case for chunked slicing.
        let prompt = concatenated(
            [textTokens(10), visionStart(), imageRun(), textTokens(12, seed: 7)], axis: 1)

        let cacheS = model.newCache(parameters: nil)
        let (logitsS, _) = try lastLogits(
            model.prepare(
                LMInput(text: .init(tokens: prompt), image: image), cache: cacheS, state: nil,
                windowSize: nil))

        let cacheC = model.newCache(parameters: nil)
        let (logitsC, _) = try lastLogits(
            model.prepare(
                LMInput(text: .init(tokens: prompt), image: image), cache: cacheC, state: nil,
                windowSize: 8))

        XCTAssertLessThanOrEqual(
            maxAbsDiff(logitsC, logitsS), 1e-3,
            "windowed image prefill diverged from single-shot")
    }

    // MARK: - Qwen2.5-VL

    func testQwen25VLWarmTextContinuationMatchesFullPrefill() throws {
        try assertWarmTextContinuation(makeTinyQwen25VL())
    }

    func testQwen25VLWarmImageContinuationMatchesFullPrefill() throws {
        try assertWarmImageContinuation(makeTinyQwen25VL())
    }

    func testQwen25VLImageMidContinuationResumeState() throws {
        try assertImageMidContinuationResumeState(makeTinyQwen25VL())
    }

    func testQwen25VLWindowedPrefillMatchesSingleShot() throws {
        try assertWindowedTextPrefill(makeTinyQwen25VL())
    }

    func testQwen25VLWindowedImagePrefillMatchesSingleShot() throws {
        try assertWindowedImagePrefill(makeTinyQwen25VL())
    }

    // MARK: - Qwen2-VL

    func testQwen2VLWarmTextContinuationMatchesFullPrefill() throws {
        try assertWarmTextContinuation(makeTinyQwen2VL())
    }

    func testQwen2VLWarmImageContinuationMatchesFullPrefill() throws {
        try assertWarmImageContinuation(makeTinyQwen2VL())
    }

    func testQwen2VLImageMidContinuationResumeState() throws {
        try assertImageMidContinuationResumeState(makeTinyQwen2VL())
    }

    func testQwen2VLWindowedPrefillMatchesSingleShot() throws {
        try assertWindowedTextPrefill(makeTinyQwen2VL())
    }

    func testQwen2VLWindowedImagePrefillMatchesSingleShot() throws {
        try assertWindowedImagePrefill(makeTinyQwen2VL())
    }
}
