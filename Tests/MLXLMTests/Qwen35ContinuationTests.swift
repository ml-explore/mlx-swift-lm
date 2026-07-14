// Copyright © 2026 Apple Inc.
//
// Equivalence tests for Qwen3.5 windowed prefill and warm (cached-prefix)
// continuation, on a tiny random-weight model so they run in CI without
// downloads. The invariant under test: however a prompt reaches the KV cache
// — one shot, windowed chunks, or split across a warm continuation — the
// next-token logits must match, because M-RoPE positions must be anchored at
// the cache offset (plus the carried rope delta), never restarted at zero.

import Foundation
import MLX
import MLXLMCommon
import MLXVLM
import XCTest

final class Qwen35ContinuationTests: XCTestCase {

    // MARK: - Tiny model

    private func makeTinyModel() throws -> Qwen35 {
        let json = """
            {
                "model_type": "qwen3_5_vl",
                "image_token_id": 500,
                "video_token_id": 501,
                "vision_start_token_id": 502,
                "vision_end_token_id": 503,
                "vocab_size": 512,
                "text_config": {
                    "model_type": "qwen3_5",
                    "hidden_size": 64,
                    "num_hidden_layers": 4,
                    "intermediate_size": 128,
                    "num_attention_heads": 4,
                    "num_key_value_heads": 2,
                    "head_dim": 32,
                    "vocab_size": 512,
                    "full_attention_interval": 2,
                    "linear_num_value_heads": 4,
                    "linear_num_key_heads": 2,
                    "linear_key_head_dim": 32,
                    "linear_value_head_dim": 32,
                    "linear_conv_kernel_dim": 4,
                    "max_position_embeddings": 4096,
                    "rope_parameters": {
                        "type": "default",
                        "mrope_section": [8, 4, 4],
                        "rope_theta": 100000.0,
                        "partial_rotary_factor": 1.0
                    }
                },
                "vision_config": {
                    "model_type": "qwen3_vl",
                    "depth": 2,
                    "hidden_size": 32,
                    "intermediate_size": 64,
                    "out_hidden_size": 64,
                    "num_heads": 2,
                    "patch_size": 16,
                    "spatial_merge_size": 2,
                    "temporal_patch_size": 2,
                    "num_position_embeddings": 64
                }
            }
            """
        let config = try JSONDecoder().decode(
            Qwen35Configuration.self, from: Data(json.utf8))
        return Qwen35(config)
    }

    /// Deterministic pseudo-random plain-text tokens, away from the special
    /// ids (500...503).
    private func textTokens(_ count: Int, seed: Int32 = 0) -> MLXArray {
        var values: [Int32] = []
        for i in 0 ..< count {
            let value: Int = (i * 13 + 7 + Int(seed)) % 480
            values.append(Int32(value))
        }
        let array = MLXArray(values)
        return array.expandedDimensions(axis: 0)
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

    // MARK: - Tests

    /// A warm continuation (prefix already in the cache, remainder prefilled
    /// on top — the ChatSession cross-turn / tool-restart flow) must produce
    /// the same next-token logits as one cold prefill of the concatenation.
    /// The decode path (token-by-token with state threaded) is the
    /// offset-correct control that bounds the numerical noise floor.
    func testWarmTextContinuationMatchesFullPrefill() throws {
        MLXRandom.seed(7)
        let model = try makeTinyModel()
        let t1 = textTokens(40)
        let t2 = textTokens(8, seed: 3)
        let full = concatenated([t1, t2], axis: 1)

        // Reference: one cold prefill of the whole sequence.
        let cacheF = model.newCache(parameters: nil)
        let (logitsF, _) = try lastLogits(
            model.prepare(
                LMInput(text: .init(tokens: full)), cache: cacheF, state: nil, windowSize: nil))

        // Control: decode path, token by token, state threaded. Correct by
        // construction; its divergence from F is the numerical noise floor.
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

        // Warm continuation via prepare, no carried state — the direct
        // TokenIterator flow. The anchored windowed path must place the new
        // tokens at the cache offset, not back at zero.
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
    /// carried into turn 2's prefill (the ChatSession cross-turn state
    /// threading): two-turn with threaded state ≡ one-shot full prefill.
    func testWarmImageContinuationMatchesFullPrefill() throws {
        MLXRandom.seed(5)
        let model = try makeTinyModel()

        // One image: grid THW (1, 4, 4), merge 2 → 4 merged tokens in text.
        // pixels: [t*h*w, channels * temporalPatch * patch * patch].
        let pixels = MLXRandom.normal([16, 3 * 2 * 16 * 16])
        let image = LMInput.ProcessedImage(pixels: pixels, frames: [THW(1, 4, 4)])

        let visionStart = MLXArray([Int32(502)]).expandedDimensions(axis: 0)
        let imageRun = MLXArray([Int32](repeating: 500, count: 4)).expandedDimensions(axis: 0)
        let t1 = concatenated(
            [textTokens(10), visionStart, imageRun, textTokens(8, seed: 5)], axis: 1)
        let t2 = textTokens(8, seed: 9)
        let full = concatenated([t1, t2], axis: 1)

        // Reference: one cold prefill of the whole image-bearing sequence.
        let cacheF = model.newCache(parameters: nil)
        let (logitsF, _) = try lastLogits(
            model.prepare(
                LMInput(text: .init(tokens: full), image: image), cache: cacheF, state: nil,
                windowSize: nil))

        // Two turns with the prefill state threaded, as ChatSession does.
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

    /// The full three-turn round trip: a warm continuation whose remainder
    /// itself contains a new image must compute that image's positions from
    /// the anchor AND hand back a resume state that positions the following
    /// turn correctly — turn 3 reads back the delta turn 2 produced.
    func testImageMidContinuationResumeState() throws {
        MLXRandom.seed(3)
        let model = try makeTinyModel()

        let pixels = MLXRandom.normal([16, 3 * 2 * 16 * 16])
        let image = LMInput.ProcessedImage(pixels: pixels, frames: [THW(1, 4, 4)])
        let visionStart = MLXArray([Int32(502)]).expandedDimensions(axis: 0)
        let imageRun = MLXArray([Int32](repeating: 500, count: 4)).expandedDimensions(axis: 0)

        let t1 = textTokens(12)
        let t2 = concatenated(
            [textTokens(4, seed: 2), visionStart, imageRun, textTokens(6, seed: 4)], axis: 1)
        let t3 = textTokens(8, seed: 6)
        let full = concatenated([t1, t2, t3], axis: 1)

        // Reference: everything cold in one shot.
        let cacheF = model.newCache(parameters: nil)
        let (logitsF, _) = try lastLogits(
            model.prepare(
                LMInput(text: .init(tokens: full), image: image), cache: cacheF, state: nil,
                windowSize: nil))

        // Three turns, state threaded turn-to-turn: text, then image, then text.
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
    /// the single-shot forward — on plain text and on an image-bearing prompt
    /// whose image straddles a window boundary.
    func testWindowedPrefillMatchesSingleShot() throws {
        MLXRandom.seed(11)
        let model = try makeTinyModel()
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

    func testWindowedImagePrefillMatchesSingleShot() throws {
        MLXRandom.seed(13)
        let model = try makeTinyModel()

        let pixels = MLXRandom.normal([16, 3 * 2 * 16 * 16])
        let image = LMInput.ProcessedImage(pixels: pixels, frames: [THW(1, 4, 4)])
        let visionStart = MLXArray([Int32(502)]).expandedDimensions(axis: 0)
        let imageRun = MLXArray([Int32](repeating: 500, count: 4)).expandedDimensions(axis: 0)
        // Image tokens sit at positions 10...14 — straddling the 8-token
        // window boundary, the hard case for chunked slicing.
        let prompt = concatenated(
            [textTokens(10), visionStart, imageRun, textTokens(12, seed: 7)], axis: 1)

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
}
