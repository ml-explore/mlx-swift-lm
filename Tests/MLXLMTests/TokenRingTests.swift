// Copyright © 2026 Apple Inc.
//
// Verifies the #220 fix: `TokenRing.loadPrompt` must measure prompt length
// after flattening. Prompts coming from `*Processor.prepare` are typically
// shaped `(1, L)`; pre-fix the ring used `dim(0) == 1` as the length and
// `append` then crashed on a broadcast mismatch.

import MLX
import MLXLMCommon
import XCTest

final class TokenRingTests: XCTestCase {

    /// 2D `(1, L)` prompt longer than the context window must (a) load
    /// without crashing and (b) leave the next `didSample()` working.
    func test2DPromptLongerThanContextDoesNotCrash() {
        var ctx = RepetitionContext(repetitionPenalty: 1.1, repetitionContextSize: 64)
        // (1, 1009) — the exact shape range #220 reports.
        let promptTokens = (0 ..< 1009).map { Int32($0 % 1000) }
        let prompt = MLXArray(promptTokens, [1, 1009])

        // Pre-fix: this call set the ring's internal `count = 1` while
        // resizing buffer to length 1009 — internally inconsistent state
        // that would manifest later.
        ctx.prompt(prompt)

        // Pre-fix: the next `didSample` triggers `MLX.where(positions, token,
        // buffer)` where positions is shaped (64,) and buffer is shaped
        // (1009,) → broadcast crash.
        let nextToken = MLXArray(Int32(42))
        ctx.didSample(token: nextToken)

        // Process must produce a logits array of the same shape as input.
        let logits = MLXArray.zeros([1, 32_000], type: Float.self)
        let processed = ctx.process(logits: logits)
        XCTAssertEqual(processed.shape, logits.shape)
    }

    /// 1D prompt path (the only shape that worked pre-fix) still works —
    /// regression guard.
    func test1DPromptStillWorks() {
        var ctx = RepetitionContext(repetitionPenalty: 1.1, repetitionContextSize: 64)
        let promptTokens = (0 ..< 200).map { Int32($0) }
        let prompt = MLXArray(promptTokens)

        ctx.prompt(prompt)
        ctx.didSample(token: MLXArray(Int32(7)))

        let logits = MLXArray.zeros([1, 32_000], type: Float.self)
        let processed = ctx.process(logits: logits)
        XCTAssertEqual(processed.shape, logits.shape)
    }

    /// 2D prompt SHORTER than the context window must also load cleanly.
    func test2DPromptShorterThanContextDoesNotCrash() {
        var ctx = RepetitionContext(repetitionPenalty: 1.1, repetitionContextSize: 64)
        let promptTokens = (0 ..< 32).map { Int32($0) }
        let prompt = MLXArray(promptTokens, [1, 32])

        ctx.prompt(prompt)
        ctx.didSample(token: MLXArray(Int32(7)))

        let logits = MLXArray.zeros([1, 32_000], type: Float.self)
        let processed = ctx.process(logits: logits)
        XCTAssertEqual(processed.shape, logits.shape)
    }
}
