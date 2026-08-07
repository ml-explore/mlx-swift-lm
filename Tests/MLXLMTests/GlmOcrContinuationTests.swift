// Copyright © 2026 Apple Inc.
//
// Equivalence tests for GLM-OCR windowed prefill and warm (cached-prefix)
// continuation, on a tiny random-weight model so they run in CI without
// downloads. The invariant under test mirrors Qwen35ContinuationTests: however
// a prompt reaches the KV cache — one shot, windowed chunks, or split across a
// warm continuation — the next-token logits must match, because M-RoPE
// positions must be anchored at the cache offset (plus the carried rope delta),
// never restarted at zero.

import Foundation
import MLX
import MLXLMCommon
import MLXVLM
import XCTest

final class GlmOcrContinuationTests: XCTestCase {

    // MARK: - Tiny model

    private func makeTinyModel() throws -> GlmOcr {
        let json = """
            {
                "model_type": "glm_ocr",
                "vocab_size": 512,
                "image_token_id": 500,
                "video_token_id": 501,
                "image_start_token_id": 502,
                "hidden_size": 64,
                "text_config": {
                    "hidden_size": 64,
                    "num_hidden_layers": 2,
                    "intermediate_size": 128,
                    "num_attention_heads": 4,
                    "num_key_value_heads": 2,
                    "head_dim": 16,
                    "vocab_size": 512,
                    "rms_norm_eps": 1e-5,
                    "tie_word_embeddings": false,
                    "rope_parameters": {
                        "mrope_section": [4, 2, 2],
                        "rope_theta": 10000.0,
                        "partial_rotary_factor": 1.0
                    }
                },
                "vision_config": {
                    "depth": 2,
                    "hidden_size": 32,
                    "intermediate_size": 64,
                    "num_heads": 2,
                    "patch_size": 16,
                    "out_hidden_size": 64,
                    "spatial_merge_size": 2,
                    "temporal_patch_size": 2,
                    "in_channels": 3,
                    "rms_norm_eps": 1e-5
                }
            }
            """
        let config = try JSONDecoder().decode(
            GlmOcrConfiguration.self, from: Data(json.utf8))
        return GlmOcr(config)
    }

    /// One image with grid THW (1, 4, 4) and merge size 2 — four merged tokens
    /// in the text stream.
    private func makeImage() -> LMInput.ProcessedImage {
        LMInput.ProcessedImage(
            pixels: MLXRandom.normal([16, 3 * 2 * 16 * 16]), frames: [THW(1, 4, 4)])
    }

    private func imageRun() -> MLXArray {
        MLXArray([Int32](repeating: 500, count: 4)).expandedDimensions(axis: 0)
    }

    /// Deterministic pseudo-random plain-text tokens, away from the special
    /// ids (500...502).
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

    private let ropeDeltasKey = LMOutput.Key<MLXArray>("glmocr.ropeDeltas")

    // MARK: - Warm continuation

    func testWarmTextContinuationMatchesFullPrefill() throws {
        MLXRandom.seed(7)
        let model = try makeTinyModel()
        let t1 = textTokens(40)
        let t2 = textTokens(8, seed: 3)

        let cacheF = model.newCache(parameters: nil)
        let (logitsF, _) = try lastLogits(
            model.prepare(
                LMInput(text: .init(tokens: concatenated([t1, t2], axis: 1))),
                cache: cacheF, state: nil, windowSize: nil))

        let cacheW = model.newCache(parameters: nil)
        let (_, s1) = try lastLogits(
            model.prepare(
                LMInput(text: .init(tokens: t1)), cache: cacheW, state: nil, windowSize: nil))
        let (logitsW, _) = try lastLogits(
            model.prepare(
                LMInput(text: .init(tokens: t2)), cache: cacheW, state: s1, windowSize: nil))

        XCTAssertLessThanOrEqual(
            maxAbsDiff(logitsW, logitsF), 1e-3,
            "warm continuation diverged from full prefill")
    }

    /// With an image in turn 1, the rope delta the image accumulated must be
    /// carried into turn 2's prefill.
    func testWarmImageContinuationMatchesFullPrefill() throws {
        MLXRandom.seed(5)
        let model = try makeTinyModel()
        let image = makeImage()

        let t1 = concatenated([textTokens(10), imageRun(), textTokens(8, seed: 5)], axis: 1)
        let t2 = textTokens(8, seed: 9)

        let cacheF = model.newCache(parameters: nil)
        let (logitsF, _) = try lastLogits(
            model.prepare(
                LMInput(text: .init(tokens: concatenated([t1, t2], axis: 1)), image: image),
                cache: cacheF, state: nil, windowSize: nil))

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

    /// An image in the middle turn: the continuation must place the new image
    /// at the anchor and hand back a resume state positioning the next turn.
    func testImageMidContinuationResumeState() throws {
        MLXRandom.seed(13)
        let model = try makeTinyModel()
        let image = makeImage()

        let t1 = textTokens(12)
        let t2 = concatenated(
            [textTokens(4, seed: 2), imageRun(), textTokens(6, seed: 4)], axis: 1)
        let t3 = textTokens(8, seed: 6)

        let cacheF = model.newCache(parameters: nil)
        let (logitsF, _) = try lastLogits(
            model.prepare(
                LMInput(text: .init(tokens: concatenated([t1, t2, t3], axis: 1)), image: image),
                cache: cacheF, state: nil, windowSize: nil))

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

    // MARK: - Windowed prefill

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

    /// An image run straddling a window boundary: embeddings and positions must
    /// be sliced in lockstep per chunk.
    func testWindowedImagePrefillMatchesSingleShot() throws {
        MLXRandom.seed(17)
        let model = try makeTinyModel()
        let image = makeImage()

        // The run of 4 image tokens starts at index 6 with a window of 8, so it
        // spans the boundary between the first and second chunk.
        let prompt = concatenated(
            [textTokens(6), imageRun(), textTokens(30, seed: 3)], axis: 1)

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

    // MARK: - Fail-closed continuation state

    /// A warm cache continued without its anchor must throw; a cold cache
    /// through the same windowed path must not.
    /// A warm batched continuation carries one scalar anchor that cannot describe rows resuming
    /// at different positions. It must be refused rather than sharing the anchor across rows
    /// (silently wrong positions) or reaching getRopeIndex's batchSize precondition (a trap).
    /// This is independent of the missing-state guard: here the anchor is present and valid.
    func testWarmBatchedContinuationThrows() throws {
        MLXRandom.seed(23)
        let model = try makeTinyModel()

        let cache = model.newCache(parameters: nil)
        let (_, warmState) = try lastLogits(
            try model.prepare(
                LMInput(text: .init(tokens: textTokens(6))), cache: cache, state: nil,
                windowSize: nil))
        XCTAssertNotNil(warmState?[LMOutput.Key<MLXArray>("glmocr.ropeDeltas")])

        let batched = concatenated(
            [textTokens(4, seed: 1), textTokens(4, seed: 2)], axis: 0)
        XCTAssertEqual(batched.dim(0), 2)

        XCTAssertThrowsError(
            try model.prepare(
                LMInput(text: .init(tokens: batched)), cache: cache, state: warmState,
                windowSize: nil)
        ) { error in
            guard
                case ContinuationStateError.unsupportedBatchContinuation(let name)? =
                    error as? ContinuationStateError
            else {
                return XCTFail(
                    "expected ContinuationStateError.unsupportedBatchContinuation, got \(error)")
            }
            XCTAssertEqual(name, "GlmOcr")
        }
    }

    func testWarmContinuationWithoutStateThrows() throws {
        MLXRandom.seed(19)
        let model = try makeTinyModel()

        let cache = model.newCache(parameters: nil)
        XCTAssertNoThrow(
            try model.prepare(
                LMInput(text: .init(tokens: textTokens(40))), cache: cache, state: nil,
                windowSize: 8),
            "a long cold prefill carries no anchor and must not throw")

        XCTAssertThrowsError(
            try model.prepare(
                LMInput(text: .init(tokens: textTokens(6, seed: 2))), cache: cache, state: nil,
                windowSize: nil)
        ) { error in
            guard
                case ContinuationStateError.missingState(_, let key)? =
                    error as? ContinuationStateError
            else {
                return XCTFail("expected ContinuationStateError.missingState, got \(error)")
            }
            XCTAssertEqual(key, "glmocr.ropeDeltas")
        }
    }

    /// A cold prefill must always hand back an anchor — zero for a text-only
    /// prompt — so an ordinary text conversation never trips the guard.
    func testColdTextOnlyPrefillCarriesAnchor() throws {
        MLXRandom.seed(23)
        let model = try makeTinyModel()

        let cache = model.newCache(parameters: nil)
        let (_, state) = try lastLogits(
            model.prepare(
                LMInput(text: .init(tokens: textTokens(10))), cache: cache, state: nil,
                windowSize: nil))

        guard let anchor = state?[ropeDeltasKey] else {
            return XCTFail("cold text-only prefill returned no rope delta")
        }
        XCTAssertEqual(anchor.asType(.int32).item(Int.self), 0)
    }

    /// Decode after a warm image-bearing continuation: the resume state carries
    /// only the anchor, so the decode path has to reach its rope-delta branch
    /// without the precomputed position ids a cold prefill would have left.
    func testDecodeAfterWarmContinuationUsesAnchor() throws {
        MLXRandom.seed(31)
        let model = try makeTinyModel()
        let image = makeImage()

        let t1 = concatenated([textTokens(6), imageRun(), textTokens(4, seed: 2)], axis: 1)
        let t2 = textTokens(6, seed: 4)
        let next = textTokens(1, seed: 8)

        // Reference: one cold prefill of everything, then one decode step.
        let cacheF = model.newCache(parameters: nil)
        let (_, stateF) = try lastLogits(
            model.prepare(
                LMInput(text: .init(tokens: concatenated([t1, t2], axis: 1)), image: image),
                cache: cacheF, state: nil, windowSize: nil))
        let decodeF = model(LMInput.Text(tokens: next), cache: cacheF, state: stateF)

        // Warm: prefill t1, continue with t2, then decode the same token.
        let cacheW = model.newCache(parameters: nil)
        let (_, s1) = try lastLogits(
            model.prepare(
                LMInput(text: .init(tokens: t1), image: image), cache: cacheW, state: nil,
                windowSize: nil))
        let (_, s2) = try lastLogits(
            model.prepare(
                LMInput(text: .init(tokens: t2)), cache: cacheW, state: s1, windowSize: nil))
        XCTAssertNotNil(s2?[ropeDeltasKey], "continuation must hand back an anchor")
        let decodeW = model(LMInput.Text(tokens: next), cache: cacheW, state: s2)

        XCTAssertLessThanOrEqual(
            maxAbsDiff(decodeW.logits[0..., -1, 0...], decodeF.logits[0..., -1, 0...]), 1e-3,
            "decode after a warm continuation ignored the carried anchor")
    }

    // MARK: - Prompt cache round trip

    func testImageStateSurvivesPromptCacheRoundTrip() throws {
        MLXRandom.seed(29)
        let model = try makeTinyModel()
        let image = makeImage()

        let t1 = concatenated([textTokens(6), imageRun(), textTokens(6, seed: 2)], axis: 1)
        let t2 = textTokens(8, seed: 4)

        let warmCache = model.newCache(parameters: nil)
        let (_, savedState) = try lastLogits(
            model.prepare(
                LMInput(text: .init(tokens: t1), image: image), cache: warmCache, state: nil,
                windowSize: nil))
        XCTAssertNotNil(savedState)

        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("safetensors")
        defer { try? FileManager.default.removeItem(at: url) }
        try savePromptCache(url: url, cache: warmCache, state: savedState)

        let snapshot = try loadPromptCacheSnapshot(url: url)
        let (warmLogits, _) = try lastLogits(
            model.prepare(
                LMInput(text: .init(tokens: t2)), cache: warmCache, state: savedState,
                windowSize: nil))
        let (restoredLogits, _) = try lastLogits(
            model.prepare(
                LMInput(text: .init(tokens: t2)), cache: snapshot.cache, state: snapshot.state,
                windowSize: nil))

        XCTAssertLessThanOrEqual(
            maxAbsDiff(restoredLogits, warmLogits), 1e-6,
            "disk-restored state diverged from the live warm continuation")

        let keptCache = try loadPromptCacheSnapshot(url: url).cache
        XCTAssertThrowsError(
            try model.prepare(
                LMInput(text: .init(tokens: t2)), cache: keptCache, state: nil, windowSize: nil))
    }
}
