// Copyright © 2025 Apple Inc.

import MLX
import Testing

@testable import MLXGuidedGeneration

/// Forced-completion sampling tests for ``GuidedGenerationLoop/applyMaskAndSample``.
@Suite
struct ForcedCompletionSamplingTests {

    @Test("Closing bias overrides model logit, selecting quote over continuation token")
    func closingBiasSelectsQuoteOverContinuation() {
        // 'A' (65) has higher raw logit than '"' (34),
        // but closing bias on '"' should flip the result.
        var floats = [Float](repeating: 0.0, count: 256)
        floats[65] = 20.0  // 'A' continuation token, high logit
        floats[34] = 1.0  // '"' closing token, low logit
        let logits = MLXArray(floats)

        // Mask allowing both tokens
        var maskWords = [UInt32](repeating: 0, count: 256 / 32)
        maskWords[65 / 32] |= (1 << (65 % 32))
        maskWords[34 / 32] |= (1 << (34 % 32))

        // Closing bias: +100 on '"'
        var biasFloats = [Float](repeating: 0.0, count: 256)
        biasFloats[34] = 100.0
        let closingBias = MLXArray(biasFloats)

        let result = maskWords.withUnsafeBufferPointer { ptr in
            GuidedGenerationLoop.applyMaskAndSample(
                logits: logits[.newAxis, .newAxis, 0...],
                sampleMask: ptr.baseAddress,
                vocabSize: 256,
                closingBias: closingBias
            )
        }
        #expect(result == 34)  // " wins due to bias despite lower model logit
    }

    @Test("Closing bias has no effect when biased tokens are masked out by grammar")
    func closingBiasIgnoredWhenTokensMaskedOut() {
        // Only continuation tokens ('A'=65 and 'B'=66) are allowed.
        // '"' (34) has closing bias but is masked out -- must not be selected.
        var floats = [Float](repeating: 0.0, count: 256)
        floats[65] = 20.0  // 'A'
        floats[66] = 10.0  // 'B'
        floats[34] = 1.0  // '"' -- will be masked out
        let logits = MLXArray(floats)

        // Mask allowing only 'A' and 'B'
        var maskWords = [UInt32](repeating: 0, count: 256 / 32)
        maskWords[65 / 32] |= (1 << (65 % 32))
        maskWords[66 / 32] |= (1 << (66 % 32))

        // Closing bias: +100 on '"'
        var biasFloats = [Float](repeating: 0.0, count: 256)
        biasFloats[34] = 100.0
        let closingBias = MLXArray(biasFloats)

        let result = maskWords.withUnsafeBufferPointer { ptr in
            GuidedGenerationLoop.applyMaskAndSample(
                logits: logits[.newAxis, .newAxis, 0...],
                sampleMask: ptr.baseAddress,
                vocabSize: 256,
                closingBias: closingBias
            )
        }
        #expect(result == 65)  // 'A' wins -- highest logit among allowed tokens
    }

    @Test("Whitespace bias suppresses whitespace token, argmax selects non-whitespace")
    func whitespaceBiasSuppressesWhitespaceToken() {
        // Space (32) has the highest raw logit, but whitespace bias should
        // push it below 'A' (65) so the non-whitespace token wins.
        var floats = [Float](repeating: 0.0, count: 256)
        floats[32] = 30.0  // space -- highest raw logit
        floats[65] = 10.0  // 'A' -- lower raw logit
        let logits = MLXArray(floats)

        // Mask allowing both tokens
        var maskWords = [UInt32](repeating: 0, count: 256 / 32)
        maskWords[32 / 32] |= (1 << (32 % 32))  // space
        maskWords[65 / 32] |= (1 << (65 % 32))  // 'A'

        // Whitespace bias: -200 on space token
        var biasFloats = [Float](repeating: 0.0, count: 256)
        biasFloats[32] = -200.0
        let whitespaceBias = MLXArray(biasFloats)

        let result = maskWords.withUnsafeBufferPointer { ptr in
            GuidedGenerationLoop.applyMaskAndSample(
                logits: logits[.newAxis, .newAxis, 0...],
                sampleMask: ptr.baseAddress,
                vocabSize: 256,
                closingBias: whitespaceBias
            )
        }
        #expect(result == 65)  // 'A' wins -- whitespace bias suppressed space
    }

    @Test(
        "When all grammar-allowed tokens are whitespace, bias reduces but does not block selection"
    )
    func whitespaceBiasDoesNotBlockWhenAllAllowedAreWhitespace() {
        // Grammar allows only space (32) and tab (9). Both are whitespace.
        // Whitespace bias makes them negative, but they should still be
        // selectable (least-negative beats -inf on disallowed tokens).
        var floats = [Float](repeating: 0.0, count: 256)
        floats[32] = 10.0  // space -- higher raw logit
        floats[9] = 5.0  // tab -- lower raw logit
        let logits = MLXArray(floats)

        // Mask allowing only space and tab
        var maskWords = [UInt32](repeating: 0, count: 256 / 32)
        maskWords[32 / 32] |= (1 << (32 % 32))  // space
        maskWords[9 / 32] |= (1 << (9 % 32))  // tab

        // Whitespace bias: -200 on both whitespace tokens
        var biasFloats = [Float](repeating: 0.0, count: 256)
        biasFloats[32] = -200.0
        biasFloats[9] = -200.0
        let whitespaceBias = MLXArray(biasFloats)

        let result = maskWords.withUnsafeBufferPointer { ptr in
            GuidedGenerationLoop.applyMaskAndSample(
                logits: logits[.newAxis, .newAxis, 0...],
                sampleMask: ptr.baseAddress,
                vocabSize: 256,
                closingBias: whitespaceBias
            )
        }
        // Space has logit 10 + bias -200 = -190; tab has 5 + -200 = -195.
        // All other tokens are -inf (masked out). Space wins as least-negative.
        #expect(result == 32)
    }

    @Test("nil closingBias selects highest allowed logit")
    func nilClosingBiasMatchesOriginalBehavior() {
        // Without closing bias, argmax of allowed tokens wins.
        var floats = [Float](repeating: 0.0, count: 256)
        floats[65] = 20.0  // 'A' -- highest
        floats[34] = 15.0  // '"' -- second highest
        let logits = MLXArray(floats)

        // Mask allowing both
        var maskWords = [UInt32](repeating: 0, count: 256 / 32)
        maskWords[65 / 32] |= (1 << (65 % 32))
        maskWords[34 / 32] |= (1 << (34 % 32))

        let result = maskWords.withUnsafeBufferPointer { ptr in
            GuidedGenerationLoop.applyMaskAndSample(
                logits: logits[.newAxis, .newAxis, 0...],
                sampleMask: ptr.baseAddress,
                vocabSize: 256,
                closingBias: nil
            )
        }
        #expect(result == 65)  // 'A' wins -- no bias applied
    }
}
