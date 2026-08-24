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

    /// The shared continuation equivalences, configured for Qwen3.5-VL's token ids.
    private let continuation = ContinuationAssertions(
        imageTokenId: 500, visionStartTokenId: 502)

    private func textTokens(_ count: Int, seed: Int32 = 0) -> MLXArray {
        continuation.textTokens(count, seed: seed)
    }
    private func lastLogits(_ result: PrepareResult) throws -> (MLXArray, LMOutput.State?) {
        try continuation.lastLogits(result)
    }
    private func maxAbsDiff(_ a: MLXArray, _ b: MLXArray) -> Float {
        continuation.maxAbsDiff(a, b)
    }

    // MARK: - Tests

    func testDeclaresDualDialectToolFormat() throws {
        let model = try makeTinyModel()
        XCTAssertEqual(model.toolCallFormat, .qwen35)
        XCTAssertEqual(model.reasoningConfig, QwenReasoningProtocol.tagged)
    }

    /// A warm continuation (prefix already in the cache, remainder prefilled
    /// on top — the ChatSession cross-turn / tool-restart flow) must produce
    /// the same next-token logits as one cold prefill of the concatenation.
    /// The decode path (token-by-token with state threaded) is the
    /// offset-correct control that bounds the numerical noise floor.
    func testWarmTextContinuationMatchesFullPrefill() throws {
        try continuation.assertWarmTextContinuation(try makeTinyModel())
    }

    /// Public callers may supply a rank-1 text remainder. It must be normalized
    /// before warm routing rather than interpreted as a batch whose size is the
    /// sequence length.
    func testRank1WarmContinuationMatchesFullPrefill() throws {
        MLXRandom.seed(41)
        let model = try makeTinyModel()
        let t1 = textTokens(40)
        let t2 = textTokens(8, seed: 3)

        let fullCache = try model.newCache(parameters: nil)
        let (fullLogits, _) = try lastLogits(
            model.prepare(
                LMInput(text: .init(tokens: concatenated([t1, t2], axis: 1))),
                cache: fullCache, state: nil, prefill: .init()))

        let warmCache = try model.newCache(parameters: nil)
        let (_, state) = try lastLogits(
            model.prepare(
                LMInput(text: .init(tokens: t1)), cache: warmCache, state: nil, prefill: .init()))
        let (warmLogits, _) = try lastLogits(
            model.prepare(
                LMInput(text: .init(tokens: t2[0])), cache: warmCache, state: state,
                prefill: .init()))

        XCTAssertLessThanOrEqual(
            maxAbsDiff(warmLogits, fullLogits), 1e-3,
            "rank-1 warm continuation diverged from full prefill")
    }

    /// A warm cache continued without its anchor must throw. The model cannot
    /// tell whether the cached prefix held images, so continuing would risk
    /// silently repositioning the remainder; a text-only prefix is refused too
    /// rather than guessed at. A cold cache needs no anchor and still works.
    func testWarmContinuationWithoutStateThrows() throws {
        MLXRandom.seed(19)
        let model = try makeTinyModel()

        let cache = try model.newCache(parameters: nil)
        XCTAssertNoThrow(
            try model.prepare(
                LMInput(text: .init(tokens: textTokens(40))), cache: cache, state: nil,
                prefill: .init(stepSize: 8)),
            "a long cold prefill carries no anchor and must not throw")

        XCTAssertThrowsError(
            try model.prepare(
                LMInput(text: .init(tokens: textTokens(6, seed: 2)[0])), cache: cache, state: nil,
                prefill: .init())
        ) { error in
            guard
                case ContinuationStateError.missingState(_, let key)? =
                    error as? ContinuationStateError
            else {
                return XCTFail("expected ContinuationStateError.missingState, got \(error)")
            }
            XCTAssertEqual(key, "qwen35.ropeDeltas")
            XCTAssertTrue(
                (error as? ContinuationStateError)?.errorDescription?
                    .contains("loadPromptCacheSnapshot") == true,
                "error should name the snapshot loader")
        }
    }

    /// With an image in turn 1, the rope delta the image accumulated must be
    /// carried into turn 2's prefill (the ChatSession cross-turn state
    /// threading): two-turn with threaded state ≡ one-shot full prefill.
    func testWarmImageContinuationMatchesFullPrefill() throws {
        try continuation.assertWarmImageContinuation(try makeTinyModel())
    }

    func testImageStateSurvivesPromptCacheRoundTrip() throws {
        MLXRandom.seed(17)
        let model = try makeTinyModel()

        let pixels = MLXRandom.normal([16, 3 * 2 * 16 * 16])
        let frame = THW(1, 4, 4)
        let image = LMInput.ProcessedImage(pixels: pixels, frames: [frame])
        let bothImages = LMInput.ProcessedImage(
            pixels: concatenated([pixels, pixels]), frames: [frame, frame])
        let visionStart = MLXArray([Int32(502)]).expandedDimensions(axis: 0)
        let imageRun = MLXArray([Int32](repeating: 500, count: 4)).expandedDimensions(axis: 0)
        let turn1 = textTokens(12)
        let turn2 = concatenated(
            [textTokens(4, seed: 2), visionStart, imageRun, textTokens(6, seed: 4)], axis: 1)
        let turn3 = textTokens(8, seed: 6)
        let turn4 = concatenated(
            [textTokens(3, seed: 8), visionStart, imageRun, textTokens(5, seed: 10)], axis: 1)
        let full = concatenated([turn1, turn2, turn3, turn4], axis: 1)

        let coldCache = try model.newCache(parameters: nil)
        let (coldLogits, _) = try lastLogits(
            model.prepare(
                LMInput(text: .init(tokens: full), image: bothImages), cache: coldCache,
                state: nil, prefill: .init()))

        let warmCache = try model.newCache(parameters: nil)
        let (_, turn1State) = try lastLogits(
            model.prepare(
                LMInput(text: .init(tokens: turn1)), cache: warmCache, state: nil,
                prefill: .init()))
        let (_, savedState) = try lastLogits(
            model.prepare(
                LMInput(text: .init(tokens: turn2), image: image), cache: warmCache,
                state: turn1State, prefill: .init()))
        XCTAssertNotNil(savedState)

        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("safetensors")
        defer { try? FileManager.default.removeItem(at: url) }
        try savePromptCache(url: url, cache: warmCache, state: savedState)

        let snapshot = try loadPromptCacheSnapshot(url: url)
        let missingStateCache = try loadPromptCacheSnapshot(url: url).cache
        let (warmTextLogits, warmTextState) = try lastLogits(
            model.prepare(
                LMInput(text: .init(tokens: turn3)), cache: warmCache, state: savedState,
                prefill: .init()))
        let (restoredTextLogits, restoredTextState) = try lastLogits(
            model.prepare(
                LMInput(text: .init(tokens: turn3)), cache: snapshot.cache,
                state: snapshot.state, prefill: .init()))

        XCTAssertLessThanOrEqual(
            maxAbsDiff(restoredTextLogits, warmTextLogits), 1e-6,
            "disk-restored state diverged on the text continuation")

        // The negative control: restoring the KV arrays while dropping the state
        // is what this whole feature exists to prevent. It used to decode at the
        // wrong positions; it must now fail loudly instead.
        XCTAssertThrowsError(
            try model.prepare(
                LMInput(text: .init(tokens: turn3)), cache: missingStateCache, state: nil,
                prefill: .init())
        ) { error in
            guard
                case ContinuationStateError.missingState(_, let key)? =
                    error as? ContinuationStateError
            else {
                return XCTFail("expected ContinuationStateError.missingState, got \(error)")
            }
            XCTAssertEqual(key, "qwen35.ropeDeltas")
        }

        let (warmImageLogits, _) = try lastLogits(
            model.prepare(
                LMInput(text: .init(tokens: turn4), image: image), cache: warmCache,
                state: warmTextState, prefill: .init()))
        let (restoredImageLogits, _) = try lastLogits(
            model.prepare(
                LMInput(text: .init(tokens: turn4), image: image), cache: snapshot.cache,
                state: restoredTextState, prefill: .init()))

        XCTAssertLessThanOrEqual(
            maxAbsDiff(restoredImageLogits, warmImageLogits), 1e-6,
            "disk-restored state diverged when a later turn added another image")
        XCTAssertLessThanOrEqual(
            maxAbsDiff(restoredImageLogits, coldLogits), 1e-3,
            "restored two-image continuation diverged from full prefill")
    }

    /// The full three-turn round trip: a warm continuation whose remainder
    /// itself contains a new image must compute that image's positions from
    /// the anchor AND hand back a resume state that positions the following
    /// turn correctly — turn 3 reads back the delta turn 2 produced.
    func testImageMidContinuationResumeState() throws {
        try continuation.assertImageMidContinuationResumeState(try makeTinyModel())
    }

    /// Windowed (chunked) prefill must produce the same first-token logits as
    /// the single-shot forward — on plain text and on an image-bearing prompt
    /// whose image straddles a window boundary.
    func testWindowedPrefillMatchesSingleShot() throws {
        try continuation.assertWindowedTextPrefill(try makeTinyModel())
    }

    func testWindowedImagePrefillMatchesSingleShot() throws {
        try continuation.assertWindowedImagePrefill(try makeTinyModel())
    }
}
