// Copyright © 2024 Apple Inc.

import XCTest

@testable import MLXVLM

/// The Qwen processors resize to the image budget the model card recommends
/// (`1280 * 28 * 28`), override-able per request via `UserInput.Processing`.
/// Both feed `QwenVL.targetSize`, so this exercises the budget: the recommended
/// cap must produce a smaller, factor-aligned grid than the config's large
/// `max_pixels`, and never exceed the budget.
final class QwenImageBudgetTests: XCTestCase {

    func testMaxPixelsBudgetShrinksGrid() throws {
        let factor = 28  // patchSize (14) * mergeSize (2)
        let (w, h) = (2000, 1500)

        // Model default (Qwen2.5-VL ships max_pixels = 12,845,056).
        let large = try QwenVL.targetSize(
            height: h, width: w, factor: factor,
            minPixels: 256 * 28 * 28, maxPixels: 12_845_056)
        XCTAssertEqual(large.0, 1512)
        XCTAssertEqual(large.1, 1988)

        // Caller-supplied recommended budget (the override path).
        let capped = try QwenVL.targetSize(
            height: h, width: w, factor: factor,
            minPixels: 256 * 28 * 28, maxPixels: 1280 * 28 * 28)
        XCTAssertEqual(capped.0, 840)
        XCTAssertEqual(capped.1, 1148)

        // The override shrinks the grid, stays within budget, and both
        // dimensions remain multiples of the patch factor.
        XCTAssertLessThan(capped.0 * capped.1, large.0 * large.1)
        XCTAssertLessThanOrEqual(capped.0 * capped.1, 1280 * 28 * 28)
        XCTAssertEqual(capped.0 % factor, 0)
        XCTAssertEqual(capped.1 % factor, 0)
    }
}
