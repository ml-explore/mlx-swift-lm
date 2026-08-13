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
        MLXRandom.seed(7)
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

    /// With an image in turn 1, the rope delta that image accumulated must be carried into
    /// turn 2's prefill: two turns with threaded state ≡ one full prefill.
    func assertWarmImageContinuation<M: LanguageModel>(
        _ model: M, file: StaticString = #filePath, line: UInt = #line
    ) throws {
        MLXRandom.seed(5)
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

    /// Three turns, with the image in the middle one. The continuation must place that image at
    /// the anchor *and* hand back a resume state that positions the following turn — the two
    /// halves of the anchor invariant, which a single-turn test cannot separate.
    func assertImageMidContinuationResumeState<M: LanguageModel>(
        _ model: M, file: StaticString = #filePath, line: UInt = #line
    ) throws {
        MLXRandom.seed(3)
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

    /// Windowed (chunked) prefill must agree with the single-shot forward on plain text.
    func assertWindowedTextPrefill<M: LanguageModel>(
        _ model: M, file: StaticString = #filePath, line: UInt = #line
    ) throws {
        MLXRandom.seed(11)
        let prompt = textTokens(40)

        let cacheS = try model.newCache(parameters: nil)
        let (logitsS, _) = try prefill(model, prompt, cache: cacheS)

        let cacheC = try model.newCache(parameters: nil)
        let (logitsC, _) = try prefill(model, prompt, cache: cacheC, stepSize: 8)

        XCTAssertLessThanOrEqual(
            maxAbsDiff(logitsC, logitsS), 1e-3,
            "windowed prefill diverged from single-shot", file: file, line: line)
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
        MLXRandom.seed(13)
        let image = image()
        let prompt = concatenated([textTokens(10), imageRun(), textTokens(12, seed: 7)], axis: 1)

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
