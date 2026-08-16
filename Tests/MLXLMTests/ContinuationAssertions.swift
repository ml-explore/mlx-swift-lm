// Copyright © 2025 Apple Inc.

import MLX
import MLXLMCommon
import MLXVLM
import XCTest

/// The equivalences every M-RoPE continuation model must satisfy, independent of which model
/// it is.
///
/// These models place tokens from a continuation anchor — the cache offset plus the rope delta
/// the cached images accumulated — so the properties worth pinning are all *equivalences*:
/// prefilling a prompt in pieces, or in windows, must land where prefilling it whole does. Each
/// assertion runs the same prompt two ways and compares final-position logits.
///
/// Configure it with the model's special token ids; everything else (prompt layout, seeds,
/// tolerances) is fixed here so all models are held to the same bar. Models with extra structure
/// — deepstack features, batch guards, prompt-cache round trips — keep those tests in their own
/// suite.
struct ContinuationAssertions {
    /// The id repeated to form an image's placeholder run in the text.
    let imageTokenId: Int32

    /// The id that precedes an image run, for models whose templates emit one. GLM-OCR has none.
    let visionStartTokenId: Int32?

    init(imageTokenId: Int32, visionStartTokenId: Int32? = nil) {
        self.imageTokenId = imageTokenId
        self.visionStartTokenId = visionStartTokenId
    }

    // MARK: - Fixtures

    /// Deterministic pseudo-random plain-text tokens, away from the special ids (500...504).
    func textTokens(_ count: Int, seed: Int32 = 0) -> MLXArray {
        let values = (0 ..< count).map { Int32(($0 * 13 + 7 + Int(seed)) % 480) }
        return MLXArray(values).expandedDimensions(axis: 0)
    }

    /// One image: grid THW (1, 4, 4), merge 2 → 4 merged tokens in the text.
    func image() -> LMInput.ProcessedImage {
        LMInput.ProcessedImage(
            pixels: MLXRandom.normal([16, 3 * 2 * 16 * 16]), frames: [THW(1, 4, 4)])
    }

    /// The token run standing in for one image, including the vision-start marker if the model
    /// uses one. Four merged image tokens, matching ``image()``.
    func imageRun() -> MLXArray {
        var ids = [Int32](repeating: imageTokenId, count: 4)
        if let visionStartTokenId {
            ids.insert(visionStartTokenId, at: 0)
        }
        return MLXArray(ids).expandedDimensions(axis: 0)
    }

    func lastLogits(_ result: PrepareResult) throws -> (MLXArray, LMOutput.State?) {
        guard case .logits(let out) = result else {
            throw XCTSkip("expected .logits from prepare")
        }
        return (out.logits[0..., -1, 0...], out.state)
    }

    func maxAbsDiff(_ a: MLXArray, _ b: MLXArray) -> Float {
        abs(a - b).max().item(Float.self)
    }

    /// Prefill `tokens` on a fresh cache and return the final-position logits and resume state.
    private func prefill<M: LanguageModel>(
        _ model: M, _ tokens: MLXArray, image: LMInput.ProcessedImage? = nil,
        cache: [any KVCache], state: LMOutput.State? = nil, stepSize: Int? = nil
    ) throws -> (MLXArray, LMOutput.State?) {
        let prefill = stepSize.map { PrefillParameters(stepSize: $0) } ?? PrefillParameters()
        return try lastLogits(
            model.prepare(
                LMInput(text: .init(tokens: tokens), image: image),
                cache: cache, state: state, prefill: prefill))
    }

    // MARK: - Assertions

    /// A warm continuation (prefix in the cache, remainder prefilled on top — the ChatSession
    /// cross-turn flow) must produce the same next-token logits as one cold prefill of the
    /// concatenation.
    ///
    /// The decode path (token by token, state threaded) is the offset-correct control: it is
    /// known to position correctly, so the difference between it and the cold prefill is the
    /// numerical noise inherent to splitting the forward, and bounds what the warm path may drift.
    func assertWarmTextContinuation<M: LanguageModel>(
        _ model: M, file: StaticString = #filePath, line: UInt = #line
    ) throws {
        // Task-local rather than MLXRandom.seed: parallel tests must not share
        // (or perturb) the global random stream.
        try withRandomState(MLXRandom.RandomState(seed: 7)) {
            let t1 = textTokens(40)
            let t2 = textTokens(8, seed: 3)

            let cacheF = try model.newCache(parameters: nil)
            let (logitsF, _) = try prefill(model, concatenated([t1, t2], axis: 1), cache: cacheF)

            let cacheD = try model.newCache(parameters: nil)
            let (_, s0) = try prefill(model, t1, cache: cacheD)
            var state = s0
            var logitsD = MLXArray(0)
            for j in 0 ..< t2.dim(1) {
                let out = model(
                    LMInput.Text(tokens: t2[0..., j ..< (j + 1)]), cache: cacheD, state: state)
                state = out.state
                logitsD = out.logits[0..., -1, 0...]
            }
            let noiseFloor = maxAbsDiff(logitsD, logitsF)

            let cacheW = try model.newCache(parameters: nil)
            let (_, s1) = try prefill(model, t1, cache: cacheW)
            let (logitsW, _) = try prefill(model, t2, cache: cacheW, state: s1)

            XCTAssertLessThanOrEqual(
                maxAbsDiff(logitsW, logitsF), max(noiseFloor * 10, 1e-3),
                "warm continuation diverged from full prefill (noise floor \(noiseFloor))",
                file: file, line: line)
        }
    }

    /// With an image in turn 1, the rope delta that image accumulated must be carried into
    /// turn 2's prefill: two turns with threaded state ≡ one full prefill.
    func assertWarmImageContinuation<M: LanguageModel>(
        _ model: M, file: StaticString = #filePath, line: UInt = #line
    ) throws {
        try withRandomState(MLXRandom.RandomState(seed: 5)) {
            let image = image()
            let t1 = concatenated([textTokens(10), imageRun(), textTokens(8, seed: 5)], axis: 1)
            let t2 = textTokens(8, seed: 9)

            let cacheF = try model.newCache(parameters: nil)
            let (logitsF, _) = try prefill(
                model, concatenated([t1, t2], axis: 1), image: image, cache: cacheF)

            let cacheW = try model.newCache(parameters: nil)
            let (_, s1) = try prefill(model, t1, image: image, cache: cacheW)
            let (logitsW, _) = try prefill(model, t2, cache: cacheW, state: s1)

            XCTAssertLessThanOrEqual(
                maxAbsDiff(logitsW, logitsF), 1e-3,
                "state-threaded warm continuation diverged from full prefill",
                file: file, line: line)
        }
    }

    /// An append-only media turn: turn 2 adds its own image to an unchanged turn 1.
    /// `split` carves the suffix out of the full prepared input, and prefilling that
    /// suffix on turn 1's cache must land where a cold prefill of the whole prompt
    /// does -- but only if the vision encoder computes each image independently.
    ///
    /// `expectsIsolation` says which model this is. `Qwen25VL` masks each frame to
    /// itself, so the split is exact; `Qwen2VL` attends across the concatenated
    /// buffer, so dropping the cached image changes the kept image's features and
    /// the logits must differ. Asserting the divergence is what makes the
    /// conformance a measured claim rather than a stated one.
    func assertAppendOnlyMediaSplit<M: LanguageModel>(
        _ model: M, split: (LMInput, Int) -> LMInput?, expectsIsolation: Bool,
        file: StaticString = #filePath, line: UInt = #line
    ) throws {
        try withRandomState(MLXRandom.RandomState(seed: 17)) {
            let imageA = image()
            let imageB = image()
            let t1 = concatenated([textTokens(10), imageRun(), textTokens(8, seed: 5)], axis: 1)
            let t2 = concatenated(
                [textTokens(6, seed: 2), imageRun(), textTokens(4, seed: 9)], axis: 1)

            // The prepared input a VL processor hands over for the whole transcript:
            // both images' patch rows concatenated, one grid each.
            let bothImages = LMInput.ProcessedImage(
                pixels: concatenated([imageA.pixels, imageB.pixels], axis: 0),
                frames: (imageA.frames ?? []) + (imageB.frames ?? []))
            let fullInput = LMInput(
                text: .init(tokens: concatenated([t1, t2], axis: 1)), image: bothImages)

            let cacheF = try model.newCache(parameters: nil)
            let (logitsF, _) = try lastLogits(
                model.prepare(
                    fullInput, cache: cacheF, state: nil, prefill: PrefillParameters()))

            let cacheW = try model.newCache(parameters: nil)
            let (_, s1) = try prefill(model, t1, image: imageA, cache: cacheW)

            let suffix = try XCTUnwrap(
                split(fullInput, t1.dim(1)), "the split was declined", file: file, line: line)
            XCTAssertEqual(suffix.text.tokens.dim(-1), t2.dim(1), file: file, line: line)
            XCTAssertEqual(
                suffix.image?.frames?.count, 1, "the suffix must carry only the new image",
                file: file, line: line)

            let (logitsW, _) = try lastLogits(
                model.prepare(
                    suffix, cache: cacheW, state: s1, prefill: PrefillParameters()))

            let diff = maxAbsDiff(logitsW, logitsF)
            if expectsIsolation {
                XCTAssertLessThanOrEqual(
                    diff, 1e-3,
                    "split-suffix prefill diverged from full prefill", file: file, line: line)
            } else {
                XCTAssertGreaterThan(
                    diff, 1e-3,
                    "cross-image vision attention should have changed the logits",
                    file: file, line: line)
            }
        }
    }

    /// Three turns, with the image in the middle one. The continuation must place that image at
    /// the anchor *and* hand back a resume state that positions the following turn — the two
    /// halves of the anchor invariant, which a single-turn test cannot separate.
    func assertImageMidContinuationResumeState<M: LanguageModel>(
        _ model: M, file: StaticString = #filePath, line: UInt = #line
    ) throws {
        try withRandomState(MLXRandom.RandomState(seed: 3)) {
            let image = image()
            let t1 = textTokens(12)
            let t2 = concatenated(
                [textTokens(4, seed: 2), imageRun(), textTokens(6, seed: 4)], axis: 1)
            let t3 = textTokens(8, seed: 6)

            let cacheF = try model.newCache(parameters: nil)
            let (logitsF, _) = try prefill(
                model, concatenated([t1, t2, t3], axis: 1), image: image, cache: cacheF)

            let cacheW = try model.newCache(parameters: nil)
            let (_, s1) = try prefill(model, t1, cache: cacheW)
            let (_, s2) = try prefill(model, t2, image: image, cache: cacheW, state: s1)
            let (logitsW, _) = try prefill(model, t3, cache: cacheW, state: s2)

            XCTAssertLessThanOrEqual(
                maxAbsDiff(logitsW, logitsF), 1e-3,
                "post-image resume state positioned the following turn wrong",
                file: file, line: line)
        }
    }

    /// The ChatSession append-only media flow end to end: turn 1 (an image) sits in the
    /// cache, turn 2 (another image) arrives as a full prepared input that the model must
    /// split at the turn boundary. The suffix — carrying only the second image — prefilled
    /// on the carried state must land where one cold prefill of the concatenation does.
    ///
    /// The two images' pixel rows live in one buffer, and the prefix run slices the same
    /// tensor, so every arm sees byte-identical features and any divergence is positional.
    func assertSplitMediaSuffixContinuation<M: LanguageModel & PreparedInputSplitting>(
        _ model: M, file: StaticString = #filePath, line: UInt = #line
    ) throws {
        MLXRandom.seed(17)
        let t1 = concatenated([textTokens(10), imageRun(), textTokens(6, seed: 5)], axis: 1)
        let t2 = concatenated(
            [textTokens(4, seed: 8), imageRun(), textTokens(5, seed: 9)], axis: 1)
        let full = concatenated([t1, t2], axis: 1)

        let pixels = MLXRandom.normal([32, 3 * 2 * 16 * 16])
        let bothImages = LMInput.ProcessedImage(
            pixels: pixels, frames: [THW(1, 4, 4), THW(1, 4, 4)])
        let firstImage = LMInput.ProcessedImage(
            pixels: pixels[0 ..< 16, 0...], frames: [THW(1, 4, 4)])

        let cacheF = try model.newCache(parameters: nil)
        let (logitsF, _) = try prefill(model, full, image: bothImages, cache: cacheF)

        let cacheW = try model.newCache(parameters: nil)
        let (_, s1) = try prefill(model, t1, image: firstImage, cache: cacheW)

        let prepared = LMInput(text: .init(tokens: full), image: bothImages)
        let suffix = try XCTUnwrap(
            model.splitPreparedInput(prepared, droppingFirst: t1.dim(1)),
            "split refused a boundary between two turns", file: file, line: line)
        XCTAssertEqual(
            suffix.text.tokens.asArray(Int32.self), t2.asArray(Int32.self),
            "suffix tokens must be exactly the second turn", file: file, line: line)
        XCTAssertEqual(
            suffix.image?.frames?.count, 1,
            "suffix must retain exactly one image", file: file, line: line)

        let (logitsW, _) = try lastLogits(
            model.prepare(suffix, cache: cacheW, state: s1, prefill: PrefillParameters()))

        XCTAssertLessThanOrEqual(
            maxAbsDiff(logitsW, logitsF), 1e-3,
            "split media suffix continuation diverged from full prefill",
            file: file, line: line)
    }

    /// Windowed (chunked) prefill must agree with the single-shot forward on plain text.
    func assertWindowedTextPrefill<M: LanguageModel>(
        _ model: M, file: StaticString = #filePath, line: UInt = #line
    ) throws {
        try withRandomState(MLXRandom.RandomState(seed: 11)) {
            let prompt = textTokens(40)

            let cacheS = try model.newCache(parameters: nil)
            let (logitsS, _) = try prefill(model, prompt, cache: cacheS)

            let cacheC = try model.newCache(parameters: nil)
            let (logitsC, _) = try prefill(model, prompt, cache: cacheC, stepSize: 8)

            XCTAssertLessThanOrEqual(
                maxAbsDiff(logitsC, logitsS), 1e-3,
                "windowed prefill diverged from single-shot", file: file, line: line)
        }
    }

    /// Windowed prefill on an image-bearing prompt must agree with the single-shot forward — the
    /// hard case for chunked slicing, since embeddings, positions, and any per-layer visual
    /// features have to be sliced in lockstep.
    ///
    /// Swept across several step sizes rather than fixed at one. Balanced chunking derives its
    /// actual boundaries from the total length, so no single step size reliably lands a boundary
    /// *inside* the image run — and that split is the case worth covering. The small sizes here
    /// guarantee some run does.
    func assertWindowedImagePrefill<M: LanguageModel>(
        _ model: M, file: StaticString = #filePath, line: UInt = #line
    ) throws {
        try withRandomState(MLXRandom.RandomState(seed: 13)) {
            let image = image()
            let prompt = concatenated(
                [textTokens(10), imageRun(), textTokens(12, seed: 7)], axis: 1)

            let cacheS = try model.newCache(parameters: nil)
            let (logitsS, _) = try prefill(model, prompt, image: image, cache: cacheS)

            for stepSize in [3, 5, 8] {
                let cacheC = try model.newCache(parameters: nil)
                let (logitsC, _) = try prefill(
                    model, prompt, image: image, cache: cacheC, stepSize: stepSize)

                XCTAssertLessThanOrEqual(
                    maxAbsDiff(logitsC, logitsS), 1e-3,
                    "windowed image prefill diverged from single-shot at stepSize \(stepSize)",
                    file: file, line: line)
            }
        }
    }
}
