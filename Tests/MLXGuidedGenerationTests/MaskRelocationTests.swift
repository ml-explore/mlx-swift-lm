// Copyright © 2025 Apple Inc.

import Foundation
import MLX
import Testing

@testable import MLXGuidedGeneration

/// Unit tests for the relocated grammar-mask build (Approach B): the mask
/// array is now constructed in the eval loop's overlap window and passed into
/// `applyMaskAndSample`, instead of being built inside it.
@Suite
struct MaskRelocationTests {

    @Test
    func buildMaskArrayMapsAllowedToZeroAndRestToNegInf() {
        // vocab 8, allow only token id 5.
        var words = [Int32](repeating: 0, count: 1)  // ceil(8 / 32) == 1 word
        words[0] |= Int32(1 << 5)
        let mask = MaskResult(mask: words, isTerminated: false, needsApply: true)

        let arr = GuidedGenerationLoop.buildMaskArray(for: mask, vocabSize: 8, logitDim: 8)
        let values = arr!.asArray(Float.self)

        #expect(values.count == 8)
        #expect(values[5] == 0.0)
        for i in [0, 1, 2, 3, 4, 6, 7] {
            #expect(values[i] == -Float.infinity)
        }
    }

    @Test
    func buildMaskArrayReturnsNilWhenNeedsApplyFalse() {
        let mask = MaskResult(
            mask: [Int32](repeating: 0, count: 1), isTerminated: false, needsApply: false)
        #expect(GuidedGenerationLoop.buildMaskArray(for: mask, vocabSize: 8, logitDim: 8) == nil)
    }
}
