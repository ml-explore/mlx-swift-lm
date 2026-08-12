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
    /// The shared continuation equivalences, configured for Qwen3-VL's token ids.
    private let continuation = ContinuationAssertions(
        imageTokenId: 500, visionStartTokenId: 502)

    private func makeImage() -> LMInput.ProcessedImage { continuation.image() }

    private let imageRunLength = 4

    private func imageRun() -> MLXArray {
        MLXArray([Int32](repeating: 500, count: imageRunLength)).expandedDimensions(axis: 0)
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

    /// A warm continuation (prefix already in the cache, remainder prefilled on
    /// top — the ChatSession cross-turn flow) must produce the same next-token
    /// logits as one cold prefill of the concatenation.
    func testWarmTextContinuationMatchesFullPrefill() throws {
        try continuation.assertWarmTextContinuation(try makeTinyModel())
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
        try continuation.assertWarmImageContinuation(try makeTinyModel())
    }

    /// An image in the middle turn: the continuation must both place the new
    /// image at the anchor and hand back a resume state that positions the
    /// following turn correctly.
    func testImageMidContinuationResumeState() throws {
        try continuation.assertImageMidContinuationResumeState(try makeTinyModel())
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
        try continuation.assertWindowedTextPrefill(try makeTinyModel())
    }

    /// The hard case for chunking: an image run straddling a window boundary,
    /// so the visual mask and every per-layer deepstack tensor must be sliced
    /// in lockstep with the embeddings — the deepstack rows by visual-token
    /// count rather than by token index.
    func testWindowedImagePrefillMatchesSingleShot() throws {
        try continuation.assertWindowedImagePrefill(try makeTinyModel())
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
