// Copyright © 2026 Apple Inc.
//
// Verifies `Qwen3VLProcessorConfiguration` decodes the new-style
// `size.{longest_edge, shortest_edge}` pixel-area budget that recent Qwen3-VL
// configs (e.g. Qwen3.6-27B PARO) ship, in addition to the legacy
// `min_pixels`/`max_pixels`. Pre-fix the new keys were ignored and the budget
// silently fell back to the hardcoded defaults — wrong for every model on the
// new config format.

import Foundation
import MLXVLM
import XCTest

final class Qwen3VLProcessorConfigTests: XCTestCase {

    private func decode(_ json: String) throws -> Qwen3VLProcessorConfiguration {
        try JSONDecoder().decode(
            Qwen3VLProcessorConfiguration.self, from: Data(json.utf8))
    }

    /// Common required fields; the budget block is appended per-case.
    private func config(budget: String) -> String {
        """
        {
            "image_mean": [0.5, 0.5, 0.5],
            "image_std": [0.5, 0.5, 0.5],
            "merge_size": 2,
            "patch_size": 16,
            "temporal_patch_size": 2,
            "image_processor_type": "Qwen2VLImageProcessorFast"\(budget.isEmpty ? "" : ",\n    " + budget)
        }
        """
    }

    /// The real Qwen3.6/3.5 PARO shape: only new-style edges, no legacy keys.
    func testNewStyleEdgesDecodeToBudget() throws {
        let cfg = try decode(
            config(
                budget: #""size": { "longest_edge": 16777216, "shortest_edge": 65536 }"#))
        XCTAssertEqual(cfg.minPixels, 65536, "shortest_edge → minPixels")
        XCTAssertEqual(cfg.maxPixels, 16_777_216, "longest_edge → maxPixels")
    }

    /// Legacy top-level keys still win.
    func testLegacyTopLevelPixelsHonored() throws {
        let cfg = try decode(
            config(
                budget: #""min_pixels": 1024, "max_pixels": 200000"#))
        XCTAssertEqual(cfg.minPixels, 1024)
        XCTAssertEqual(cfg.maxPixels, 200_000)
    }

    /// Legacy keys nested inside `size` are honored.
    func testLegacySizePixelsHonored() throws {
        let cfg = try decode(
            config(
                budget: #""size": { "min_pixels": 2048, "max_pixels": 300000 }"#))
        XCTAssertEqual(cfg.minPixels, 2048)
        XCTAssertEqual(cfg.maxPixels, 300_000)
    }

    /// Top-level legacy keys take precedence over a `size` block.
    func testTopLevelPixelsBeatSizeEdges() throws {
        let cfg = try decode(
            config(
                budget:
                    #""min_pixels": 100, "max_pixels": 999, "size": { "longest_edge": 5, "shortest_edge": 5 }"#
            ))
        XCTAssertEqual(cfg.minPixels, 100)
        XCTAssertEqual(cfg.maxPixels, 999)
    }

    /// No budget at all falls back to the historical defaults.
    func testAbsentBudgetFallsBackToDefaults() throws {
        let cfg = try decode(config(budget: ""))
        XCTAssertEqual(cfg.minPixels, 4 * 28 * 28)
        XCTAssertEqual(cfg.maxPixels, 16384 * 28 * 28)
    }
}
