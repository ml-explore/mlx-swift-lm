// Copyright © 2026 Apple Inc.

import MLX
import Testing

@testable import MLXLMCommon

/// `carriesPaddingMask` decides whether ChatSession may drop a mask when it
/// prefills only a suffix.
@Suite
struct LMInputTextMaskTests {

    private func text(mask: MLXArray?) -> LMInput.Text {
        LMInput.Text(tokens: MLXArray([1, 2, 3, 4]).expandedDimensions(axis: 0), mask: mask)
    }

    @Test func `no mask is not padding`() {
        #expect(!text(mask: nil).carriesPaddingMask)
    }

    @Test func `an all-ones integer or boolean mask is not padding`() {
        let ones = MLXArray([1, 1, 1, 1]).expandedDimensions(axis: 0)
        #expect(!text(mask: ones.asType(.int8)).carriesPaddingMask)
        #expect(!text(mask: ones.asType(.int32)).carriesPaddingMask)
        #expect(!text(mask: ones.asType(.bool)).carriesPaddingMask)
    }

    @Test func `a mask with a zero is padding`() {
        let mask = MLXArray([0, 1, 1, 1]).expandedDimensions(axis: 0)
        #expect(text(mask: mask.asType(.int8)).carriesPaddingMask)
        #expect(text(mask: mask.asType(.bool)).carriesPaddingMask)
    }

    @Test func `a zero in any row of a batch mask is padding`() {
        let ones = MLXArray.ones([2, 4], dtype: .int8)
        #expect(!text(mask: ones).carriesPaddingMask)

        let mask = MLXArray([1, 1, 1, 1, 1, 1, 0, 1] as [Int8]).reshaped(2, 4)
        #expect(text(mask: mask).carriesPaddingMask)
    }

    @Test func `a floating-point mask is always padding`() {
        let mask = MLXArray([1, 1, 1, 1]).expandedDimensions(axis: 0).asType(.float32)
        #expect(text(mask: mask).carriesPaddingMask)
    }
}
