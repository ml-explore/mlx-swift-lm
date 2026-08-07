// Copyright © 2026 Apple Inc.
//
// Equivalence tests for Qwen3-VL windowed prefill and warm (cached-prefix)
// continuation, on a tiny random-weight model so they run in CI without
// downloads. The invariant under test mirrors Qwen35ContinuationTests: however
// a prompt reaches the KV cache — one shot, windowed chunks, or split across a
// warm continuation — the next-token logits must match, because M-RoPE
// positions must be anchored at the cache offset (plus the carried rope delta),
// never restarted at zero.
//
// Qwen3-VL adds per-layer deepstack features on top of that. They are packed
// per visual token, so a windowed forward has to slice them by visual-token
// count while slicing embeddings, positions, and the visual mask by token
// index. The straddling-image tests below are what cover that lockstep.

import Foundation
import MLX
import MLXLMCommon
import MLXVLM
import XCTest

final class Qwen3VLContinuationTests: XCTestCase {

    // MARK: - Tiny model

    private func makeTinyModel() throws -> Qwen3VL {
        let json = """
            {
                "model_type": "qwen3_vl",
                "image_token_id": 500,
                "video_token_id": 501,
                "vision_start_token_id": 502,
                "vision_end_token_id": 503,
                "vocab_size": 512,
                "text_config": {
                    "model_type": "qwen3_vl",
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
                    "rope_scaling": {
                        "type": "default",
                        "mrope_section": [4, 2, 2]
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
                    "num_position_embeddings": 64,
                    "deepstack_visual_indexes": [0, 1]
                }
            }
            """
        let config = try JSONDecoder().decode(
            Qwen3VLConfiguration.self, from: Data(json.utf8))
        return Qwen3VL(config)
    }

    /// One image with grid THW (1, 4, 4) and merge size 2 — four merged tokens
    /// in the text stream.
    private func makeImage() -> LMInput.ProcessedImage {
        LMInput.ProcessedImage(
            pixels: MLXRandom.normal([16, 3 * 2 * 16 * 16]), frames: [THW(1, 4, 4)])
    }

    private let imageRunLength = 4

    private func imageRun() -> MLXArray {
        MLXArray([Int32](repeating: 500, count: imageRunLength)).expandedDimensions(axis: 0)
    }

    private func visionStart() -> MLXArray {
        MLXArray([Int32(502)]).expandedDimensions(axis: 0)
    }

    /// Deterministic pseudo-random plain-text tokens, away from the special
    /// ids (500...503).
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

    // MARK: - Warm continuation

    /// A warm continuation (prefix already in the cache, remainder prefilled on
    /// top — the ChatSession cross-turn flow) must produce the same next-token
    /// logits as one cold prefill of the concatenation.
    func testWarmTextContinuationMatchesFullPrefill() throws {
        MLXRandom.seed(7)
        let model = try makeTinyModel()
        let t1 = textTokens(40)
        let t2 = textTokens(8, seed: 3)

        let cacheF = model.newCache(parameters: nil)
        let (logitsF, _) = try lastLogits(
            model.prepare(
                LMInput(text: .init(tokens: concatenated([t1, t2], axis: 1))),
                cache: cacheF, state: nil, prefill: .init()))

        let cacheW = model.newCache(parameters: nil)
        let (_, s1) = try lastLogits(
            model.prepare(
                LMInput(text: .init(tokens: t1)), cache: cacheW, state: nil, prefill: .init()))
        let (logitsW, _) = try lastLogits(
            model.prepare(
                LMInput(text: .init(tokens: t2)), cache: cacheW, state: s1, prefill: .init()))

        XCTAssertLessThanOrEqual(
            maxAbsDiff(logitsW, logitsF), 1e-3,
            "warm continuation diverged from full prefill")
    }

    /// A text-only follow-up may be rank-1 even though the cache was seeded by
    /// a batched image prompt. Warm routing must normalize it before slicing.
    func testRank1WarmImageContinuationMatchesFullPrefill() throws {
        MLXRandom.seed(41)
        let model = try makeTinyModel()
        let image = makeImage()
        let t1 = concatenated(
            [textTokens(10), visionStart(), imageRun(), textTokens(8, seed: 5)], axis: 1)
        let t2 = textTokens(8, seed: 9)

        let fullCache = model.newCache(parameters: nil)
        let (fullLogits, _) = try lastLogits(
            model.prepare(
                LMInput(text: .init(tokens: concatenated([t1, t2], axis: 1)), image: image),
                cache: fullCache, state: nil, prefill: .init()))

        let warmCache = model.newCache(parameters: nil)
        let (_, state) = try lastLogits(
            model.prepare(
                LMInput(text: .init(tokens: t1), image: image), cache: warmCache, state: nil,
                prefill: .init()))
        let (warmLogits, _) = try lastLogits(
            model.prepare(
                LMInput(text: .init(tokens: t2[0])), cache: warmCache, state: state,
                prefill: .init()))

        XCTAssertLessThanOrEqual(
            maxAbsDiff(warmLogits, fullLogits), 1e-3,
            "rank-1 warm continuation diverged from full prefill")
    }

    /// With an image in turn 1, the rope delta the image accumulated must be
    /// carried into turn 2's prefill.
    func testWarmImageContinuationMatchesFullPrefill() throws {
        MLXRandom.seed(5)
        let model = try makeTinyModel()
        let image = makeImage()

        let t1 = concatenated(
            [textTokens(10), visionStart(), imageRun(), textTokens(8, seed: 5)], axis: 1)
        let t2 = textTokens(8, seed: 9)

        let cacheF = model.newCache(parameters: nil)
        let (logitsF, _) = try lastLogits(
            model.prepare(
                LMInput(text: .init(tokens: concatenated([t1, t2], axis: 1)), image: image),
                cache: cacheF, state: nil, prefill: .init()))

        let cacheW = model.newCache(parameters: nil)
        let (_, s1) = try lastLogits(
            model.prepare(
                LMInput(text: .init(tokens: t1), image: image), cache: cacheW, state: nil,
                prefill: .init()))
        let (logitsW, _) = try lastLogits(
            model.prepare(
                LMInput(text: .init(tokens: t2)), cache: cacheW, state: s1, prefill: .init()))

        XCTAssertLessThanOrEqual(
            maxAbsDiff(logitsW, logitsF), 1e-3,
            "state-threaded warm continuation diverged from full prefill")
    }

    /// An image in the middle turn: the continuation must both place the new
    /// image at the anchor and hand back a resume state that positions the
    /// following turn correctly.
    func testImageMidContinuationResumeState() throws {
        MLXRandom.seed(13)
        let model = try makeTinyModel()
        let image = makeImage()

        let t1 = textTokens(12)
        let t2 = concatenated(
            [textTokens(4, seed: 2), visionStart(), imageRun(), textTokens(6, seed: 4)], axis: 1)
        let t3 = textTokens(8, seed: 6)

        let cacheF = model.newCache(parameters: nil)
        let (logitsF, _) = try lastLogits(
            model.prepare(
                LMInput(text: .init(tokens: concatenated([t1, t2, t3], axis: 1)), image: image),
                cache: cacheF, state: nil, prefill: .init()))

        let cacheW = model.newCache(parameters: nil)
        let (_, s1) = try lastLogits(
            model.prepare(
                LMInput(text: .init(tokens: t1)), cache: cacheW, state: nil, prefill: .init()))
        let (_, s2) = try lastLogits(
            model.prepare(
                LMInput(text: .init(tokens: t2), image: image), cache: cacheW, state: s1,
                prefill: .init()))
        let (logitsW, _) = try lastLogits(
            model.prepare(
                LMInput(text: .init(tokens: t3)), cache: cacheW, state: s2, prefill: .init()))

        XCTAssertLessThanOrEqual(
            maxAbsDiff(logitsW, logitsF), 1e-3,
            "post-image resume state positioned the following turn wrong")
    }

    // MARK: - Windowed prefill

    /// Windowed (chunked) prefill must produce the same first-token logits as
    /// the single-shot forward on plain text.
    /// `LanguageModel.prepare` documents that an implementation returning `.logits` owns its whole
    /// progress sequence, including the terminal `(total, total)`. Both routes through `prepare`
    /// return `.logits`, so both owe the contract: the windowed continuation (which delegates the
    /// per-chunk reports to `forEachChunk`) and the single-shot path.
    func testPrefillProgressReachesTheTotal() throws {
        MLXRandom.seed(29)
        let model = try makeTinyModel()

        func events(for prompt: MLXArray, stepSize: Int?) throws -> [[Int]] {
            final class Log: @unchecked Sendable { var events: [[Int]] = [] }
            let log = Log()
            var prefill = PrefillParameters(stepSize: stepSize)
            prefill.progress = { log.events.append([$0, $1]) }
            _ = try model.prepare(
                LMInput(text: .init(tokens: prompt)),
                cache: model.newCache(parameters: nil), state: nil, prefill: prefill)
            return log.events
        }

        let prompt = textTokens(40)
        for (label, stepSize) in [("windowed", 8), ("single-shot", 1024)] {
            let events = try events(for: prompt, stepSize: stepSize)
            XCTAssertEqual(
                events.last, [40, 40], "\(label) prefill must end at (total, total)")
            XCTAssertEqual(
                events.map { $0[0] }, events.map { $0[0] }.sorted(),
                "\(label) progress must be monotone")
            XCTAssertTrue(
                events.allSatisfy { $0[1] == 40 },
                "\(label) progress must report a stable total")
        }

        XCTAssertGreaterThan(
            try events(for: prompt, stepSize: 8).count, 1,
            "a windowed prefill should report more than just the terminal event")
    }

    func testWindowedPrefillMatchesSingleShot() throws {
        MLXRandom.seed(11)
        let model = try makeTinyModel()
        let prompt = textTokens(40)

        let cacheS = model.newCache(parameters: nil)
        let (logitsS, _) = try lastLogits(
            model.prepare(
                LMInput(text: .init(tokens: prompt)), cache: cacheS, state: nil, prefill: .init()))

        let cacheC = model.newCache(parameters: nil)
        let (logitsC, _) = try lastLogits(
            model.prepare(
                LMInput(text: .init(tokens: prompt)), cache: cacheC, state: nil,
                prefill: .init(stepSize: 8)))

        XCTAssertLessThanOrEqual(
            maxAbsDiff(logitsC, logitsS), 1e-3,
            "windowed prefill diverged from single-shot")
    }

    /// The hard case for chunking: an image run straddling a window boundary,
    /// so the visual mask and every per-layer deepstack tensor must be sliced
    /// in lockstep with the embeddings — the deepstack rows by visual-token
    /// count rather than by token index.
    func testWindowedImagePrefillMatchesSingleShot() throws {
        MLXRandom.seed(17)
        let model = try makeTinyModel()
        let image = makeImage()

        // The run of 4 image tokens starts at index 6 and the window is 8, so
        // the run spans the boundary between chunk 0 and chunk 1.
        let prompt = concatenated(
            [textTokens(5), visionStart(), imageRun(), textTokens(30, seed: 3)], axis: 1)

        let cacheS = model.newCache(parameters: nil)
        let (logitsS, _) = try lastLogits(
            model.prepare(
                LMInput(text: .init(tokens: prompt), image: image), cache: cacheS, state: nil,
                prefill: .init()))

        let cacheC = model.newCache(parameters: nil)
        let (logitsC, _) = try lastLogits(
            model.prepare(
                LMInput(text: .init(tokens: prompt), image: image), cache: cacheC, state: nil,
                prefill: .init(stepSize: 8)))

        XCTAssertLessThanOrEqual(
            maxAbsDiff(logitsC, logitsS), 1e-3,
            "windowed image prefill diverged from single-shot")
    }

    /// Windowing must not change M-RoPE semantics just because an input carries
    /// a padding mask: Qwen3-VL's baseline forwards no attention mask.
    func testPaddedWindowedImagePrefillMatchesSingleShot() throws {
        MLXRandom.seed(43)
        let model = try makeTinyModel()
        let image = makeImage()
        let prompt = concatenated(
            [textTokens(5), visionStart(), imageRun(), textTokens(12, seed: 3)], axis: 1)
        let padding = MLXArray([Int32](repeating: 0, count: 4)).expandedDimensions(axis: 0)
        let tokens = concatenated([prompt, padding], axis: 1)
        let mask = MLXArray(
            [Int32](repeating: 1, count: prompt.dim(1)) + [Int32](repeating: 0, count: 4)
        ).expandedDimensions(axis: 0)
        let input = LMInput(text: .init(tokens: tokens, mask: mask), image: image)

        let singleCache = model.newCache(parameters: nil)
        let (singleLogits, _) = try lastLogits(
            model.prepare(input, cache: singleCache, state: nil, prefill: .init()))

        let windowedCache = model.newCache(parameters: nil)
        let (windowedLogits, _) = try lastLogits(
            model.prepare(input, cache: windowedCache, state: nil, prefill: .init(stepSize: 8)))

        XCTAssertLessThanOrEqual(
            maxAbsDiff(windowedLogits, singleLogits), 1e-3,
            "padding changed Qwen3-VL windowed M-RoPE semantics")
    }

    // MARK: - Fail-closed continuation state

    /// A warm cache continued without its anchor must throw rather than
    /// silently repositioning the remainder. A cold cache needs no anchor, so a
    /// long cold prefill through the same windowed path still works.
    func testWarmContinuationWithoutStateThrows() throws {
        MLXRandom.seed(19)
        let model = try makeTinyModel()

        let cache = model.newCache(parameters: nil)
        XCTAssertNoThrow(
            try model.prepare(
                LMInput(text: .init(tokens: textTokens(40))), cache: cache, state: nil,
                prefill: .init(stepSize: 8)),
            "a long cold prefill carries no anchor and must not throw")

        XCTAssertThrowsError(
            try model.prepare(
                LMInput(text: .init(tokens: textTokens(6, seed: 2))), cache: cache, state: nil,
                prefill: .init())
        ) { error in
            guard
                case ContinuationStateError.missingState(_, let key)? =
                    error as? ContinuationStateError
            else {
                return XCTFail("expected ContinuationStateError.missingState, got \(error)")
            }
            XCTAssertEqual(key, "qwen35vl.ropeDeltas")
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
                prefill: .init()))

        guard let anchor = state?[LMOutput.Key<MLXArray>("qwen35vl.ropeDeltas")] else {
            return XCTFail("cold text-only prefill returned no rope delta")
        }
        XCTAssertEqual(anchor.asType(.int32).item(Int.self), 0)
    }

    // MARK: - Prompt cache round trip

    /// The end-to-end shape this feature exists for: save a warm image-bearing
    /// cache with its anchor, restore both, and continue equivalently.
    func testImageStateSurvivesPromptCacheRoundTrip() throws {
        MLXRandom.seed(29)
        let model = try makeTinyModel()
        let image = makeImage()

        let t1 = concatenated(
            [textTokens(6), visionStart(), imageRun(), textTokens(6, seed: 2)], axis: 1)
        let t2 = textTokens(8, seed: 4)

        let warmCache = model.newCache(parameters: nil)
        let (_, savedState) = try lastLogits(
            model.prepare(
                LMInput(text: .init(tokens: t1), image: image), cache: warmCache, state: nil,
                prefill: .init()))
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
                prefill: .init()))
        let (restoredLogits, _) = try lastLogits(
            model.prepare(
                LMInput(text: .init(tokens: t2)), cache: snapshot.cache, state: snapshot.state,
                prefill: .init()))

        XCTAssertLessThanOrEqual(
            maxAbsDiff(restoredLogits, warmLogits), 1e-6,
            "disk-restored state diverged from the live warm continuation")

        // Dropping the state on restore is the bug this feature prevents; it
        // must fail loudly rather than decode at the wrong positions.
        let keptCache = try loadPromptCacheSnapshot(url: url).cache
        XCTAssertThrowsError(
            try model.prepare(
                LMInput(text: .init(tokens: t2)), cache: keptCache, state: nil, prefill: .init()))
    }
}
